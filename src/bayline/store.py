"""The warehouse: DuckDB over Parquet.

Millions of telemetry rows, aggregated in the database rather than in Python. The columnar
layout is not decoration — the health layer's window functions over a per-vehicle,
per-component ordering are the expensive part of the pipeline, and doing them in pandas
would take minutes and several gigabytes instead of seconds.

Every read goes through this class so the SQL lives in one place and the schemas are
enforced on write rather than discovered on read.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from bayline.config import Config

TABLES = ("vehicles", "telemetry", "events", "health", "risk", "plan")


class Warehouse:
    """A DuckDB connection plus the Parquet layout it reads."""

    def __init__(self, cfg: Config, root: Path | None = None):
        self.cfg = cfg
        self.root = Path(root) if root else cfg.warehouse
        self.root.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"SET threads TO {cfg.get('determinism.duckdb_threads', 4)}")

    # ------------------------------------------------------------------ paths

    def path(self, table: str) -> Path:
        if table not in TABLES:
            raise KeyError(f"unknown table {table!r}")
        return self.root / f"{table}.parquet"

    def exists(self, *tables: str) -> bool:
        return all(self.path(t).exists() for t in (tables or TABLES))

    def missing(self, *tables: str) -> list[str]:
        return [t for t in tables if not self.path(t).exists()]

    # ------------------------------------------------------------------ writes

    def write_arrow(self, table: str, data: pa.Table) -> Path:
        """Write an Arrow table straight to Parquet without a Python round-trip."""
        target = self.path(table)
        self.con.register("_incoming", data)
        try:
            self.con.execute(
                f"COPY (SELECT * FROM _incoming) TO '{_quote(target)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            self.con.unregister("_incoming")
        return target

    def write_columns(self, table: str, columns: dict[str, np.ndarray]) -> Path:
        return self.write_arrow(table, pa.table(columns))

    def write_rows(self, table: str, rows: list[dict]) -> Path:
        if not rows:
            raise ValueError(f"refusing to write an empty {table} table")
        keys = list(rows[0])
        return self.write_arrow(table, pa.table({k: [r.get(k) for r in rows] for k in keys}))

    # ------------------------------------------------------------------ reads

    def sql(self, query: str, params: list | None = None):
        """Run a query with ``{table}`` placeholders resolved to Parquet scans."""
        resolved = query.format(**{t: f"read_parquet('{_quote(self.path(t))}')" for t in TABLES})
        return self.con.execute(resolved, params or [])

    def arrow(self, query: str, params: list | None = None) -> pa.Table:
        return self.sql(query, params).arrow()

    def rows(self, query: str, params: list | None = None) -> list[dict]:
        result = self.sql(query, params)
        names = [d[0] for d in result.description]
        return [dict(zip(names, row)) for row in result.fetchall()]

    def scalar(self, query: str, params: list | None = None):
        row = self.sql(query, params).fetchone()
        return row[0] if row else None

    def count(self, table: str) -> int:
        return int(self.scalar(f"SELECT count(*) FROM {{{table}}}"))

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.parquet"))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def day_to_date(cfg: Config, day_index: int) -> date:
    """Day 0 is the first day of the history window; the last day is the config end date."""
    return cfg.end_date - timedelta(days=int(cfg.get("history.days")) - 1 - int(day_index))


def date_to_day(cfg: Config, when: date) -> int:
    return int(cfg.get("history.days")) - 1 - (cfg.end_date - when).days


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")
