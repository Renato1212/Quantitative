"""DuckDB-over-Parquet store.

Feature code never touches this class. It goes through ``PointInTimeView``, which is
the only public read path. The store exists to hold the connection and the dataset
paths, and to be able to hand out a physically truncated copy of itself — which is how
the leakage detector proves a feature is not reaching around the view.

Determinism (C7): the connection is pinned to one thread and every query carries an
explicit ``ORDER BY``. SQL makes no row-order guarantee without one, and a pipeline
whose Parquet output depends on scan order is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from src.config import REPO_ROOT, Config

# Dataset name -> the timestamp column that defines when its rows became knowable.
KNOWABLE_AT: dict[str, str] = {
    "ticks": "ts",
    "bars_time": "close_ts",
    "bars_volume": "close_ts",
    "bars_dollar": "close_ts",
}


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path

    @property
    def knowable_at(self) -> str:
        return KNOWABLE_AT[self.name]


class Store:
    """A read-only handle over the Parquet datasets under ``data/``."""

    def __init__(self, datasets: dict[str, Path], cfg: Config):
        missing = [n for n, p in datasets.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"datasets not built: {', '.join(sorted(missing))}")
        self._datasets = {n: Dataset(n, Path(p)) for n, p in datasets.items()}
        self._cfg = cfg
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"SET threads TO {cfg.get('determinism.duckdb_threads', 1)}")

    @staticmethod
    def data_root(cfg: Config) -> Path:
        """The configured data root, resolved against the repository. One definition."""
        root = Path(cfg.get("data.root"))
        return root if root.is_absolute() else REPO_ROOT / root

    @staticmethod
    def layout(root: Path, name: str) -> Path:
        """Where a dataset lives under a data root. One definition, used everywhere."""
        folder = "raw" if name == "ticks" else "bars"
        return Path(root) / folder / f"{name}.parquet"

    @classmethod
    def from_config(cls, cfg: Config, root: Path | None = None) -> Store:
        base = Path(root) if root else cls.data_root(cfg)
        return cls({name: cls.layout(base, name) for name in KNOWABLE_AT}, cfg)

    @property
    def names(self) -> list[str]:
        return sorted(self._datasets)

    def _read(self, name: str, where: str, params: list, order_by: str, limit: int | None) -> pl.DataFrame:
        dataset = self._datasets[name]
        sql = (
            f"SELECT * FROM read_parquet(?) WHERE {where} "
            f"ORDER BY {order_by}" + (f" LIMIT {int(limit)}" if limit else "")
        )
        frame = self._con.execute(sql, [str(dataset.path), *params]).pl()
        # DuckDB labels its timestamps "Etc/UTC"; the calendar uses zoneinfo's "UTC".
        # They are the same instant and polars refuses to compare them, so normalise here
        # rather than at every call site.
        return frame.with_columns(
            [
                pl.col(name).dt.convert_time_zone("UTC")
                for name, dtype in frame.schema.items()
                if isinstance(dtype, pl.Datetime) and dtype.time_zone
            ]
        )

    def read_upto(
        self, name: str, cutoff: datetime, *, extra_sql: str = "", params: list | None = None,
        tail: int | None = None, since: datetime | None = None,
    ) -> pl.DataFrame:
        """Rows knowable at or before ``cutoff``. The only truncating read.

        ``tail`` returns the most recent ``n`` rows, still in ascending order.
        ``since`` bounds the window by time instead. Prefer ``since`` for anything
        session-relative: a row cap looks like a lookback but silently returns fewer
        sessions than asked for when bar density changes, and the caller averages over
        whatever it happened to get.
        """
        dataset = self._datasets[name]
        column = dataset.knowable_at
        where = f'"{column}" <= ?'
        args = [cutoff]
        if since is not None:
            where += f' AND "{column}" >= ?'
            args.append(since)
        if extra_sql:
            where += f" AND ({extra_sql})"
        args += params or []
        if tail is None:
            return self._read(name, where, args, f'"{column}" ASC', None)
        rows = self._read(name, where, args, f'"{column}" DESC', tail)
        return rows.reverse()

    def truncated_copy(self, cutoff: datetime, destination: Path) -> Store:
        """Write a copy of every dataset containing only rows knowable by ``cutoff``.

        The leakage detector runs a feature against this and against the full store. A
        feature that only reads through ``PointInTimeView`` cannot tell the difference.
        One that reaches around it produces a different answer, or fails outright.
        """
        destination = Path(destination)
        paths: dict[str, Path] = {}
        for name, dataset in self._datasets.items():
            out = self.layout(destination, name)
            out.parent.mkdir(parents=True, exist_ok=True)
            # COPY ... TO takes a literal, not a parameter, so the path is inlined.
            target = str(out).replace("'", "''")
            self._con.execute(
                f'COPY (SELECT * FROM read_parquet(?) WHERE "{dataset.knowable_at}" <= ? '
                f'ORDER BY "{dataset.knowable_at}") TO \'{target}\' (FORMAT PARQUET)',
                [str(dataset.path), cutoff],
            )
            paths[name] = out
        return Store(paths, self._cfg)

    def perturbed_copy(self, cutoff: datetime, destination: Path, seed: int) -> Store:
        """Copy every dataset with all *post-cutoff* numeric values scrambled.

        Complements truncation. A feature that reads the future but tolerates missing
        rows can survive truncation unchanged; it will not survive the future holding
        different numbers. Rows before the cutoff are untouched, so a correct feature
        returns exactly what it returned before. Only float columns are scrambled, so
        this probe is blind to a feature that reads future *timestamps* — that is what
        truncation is for.
        """
        destination = Path(destination)
        rng_offset = float(seed % 997) + 137.0
        paths: dict[str, Path] = {}
        for name, dataset in self._datasets.items():
            frame = pl.read_parquet(dataset.path)
            column = dataset.knowable_at
            floats = [c for c, t in frame.schema.items() if t in (pl.Float64, pl.Float32)]
            future = pl.col(column) > cutoff
            frame = frame.with_columns(
                [
                    pl.when(future).then(pl.col(c) * 1.37 + rng_offset).otherwise(pl.col(c)).alias(c)
                    for c in floats
                ]
            )
            out = self.layout(destination, name)
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(out)
            paths[name] = out
        return Store(paths, self._cfg)

    def close(self) -> None:
        self._con.close()
