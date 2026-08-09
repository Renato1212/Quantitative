"""Phase 1 pipeline: build the data root, then check the gate.

    python -m src.pipeline build     # ticks -> bars, sessions, roll schedule
    python -m src.pipeline gate      # leakage canaries + byte-identical rerun
    python -m src.pipeline status    # what exists, what it was built from

The gate is the one in ``CLAUDE.md`` §6 Phase 1, tightened: a canary per leak mechanism
rather than a single one-bar peek, and the determinism check compares a manifest of
artefact checksums across two independent builds.

The experiment log is deliberately excluded from the determinism manifest. It records
wall-clock time and a fresh run id by design — it is a ledger, not an artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src import config as config_module
from src.bars.build import BUILDERS
from src.config import REPO_ROOT, Config
from src.data.point_in_time import PointInTimeView
from src.data.store import Store
from src.experiment.log import ExperimentLog
from src.ingest import synthetic
from src.ingest.calendar import SessionCalendar
from src.ingest.contracts import back_adjust, volume_roll_schedule
from src.ingest.schemas import BarSchema, SessionSchema, TickSchema, validate
from src.validation import canaries
from src.validation.leakage import audit_all

# Artefacts whose bytes must match across runs. The SQLite log is not one of them.
MANIFEST_GLOBS = ("raw/*.parquet", "bars/*.parquet", "events/*.parquet")


# --------------------------------------------------------------------------- build


def build(
    cfg: Config,
    root: Path,
    *,
    start: date | None = None,
    end: date | None = None,
    ticks_per_session: int = 4000,
) -> dict[str, int]:
    """Materialise the data root. Returns row counts per dataset."""
    if not cfg.is_synthetic:
        raise NotImplementedError(
            "Rithmic ingest is not written. data.version does not start with 'synthetic', "
            "so this run expects a real snapshot that does not exist. See "
            "research/reviews/2026-08-09-spec-review.md §3 for what must be confirmed first."
        )

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for folder in ("raw", "bars", "events"):
        (root / folder).mkdir(parents=True)

    calendar = SessionCalendar(cfg)
    counts: dict[str, int] = {}

    ticks = synthetic.generate_ticks(cfg, start, end, ticks_per_session=ticks_per_session)
    validate(TickSchema, ticks)
    _write(ticks, Store.layout(root, "ticks"), cfg)
    counts["ticks"] = ticks.height

    sessions = calendar.frame().filter(
        pl.col("session_date").is_in(ticks["session_date"].unique())
    )
    validate(SessionSchema, sessions)
    _write(sessions, root / "events" / "sessions.parquet", cfg)
    counts["sessions"] = sessions.height

    schedule = volume_roll_schedule(synthetic.session_volumes(ticks), cfg)
    _write(schedule, root / "events" / "roll_schedule.parquet", cfg)
    counts["roll_sessions"] = schedule.height
    counts["roll_weeks"] = int(schedule["is_roll_week"].sum())

    # Only the scheduled front month forms the continuous series.
    front = ticks.join(schedule.select("session_date", "front"), on="session_date", how="inner")
    front = front.filter(pl.col("contract") == pl.col("front")).drop("front")

    for kind, builder in BUILDERS.items():
        bars = builder(front, cfg, calendar)
        validate(BarSchema, bars)
        bars = back_adjust(bars, schedule)
        _write(bars, Store.layout(root, f"bars_{kind}"), cfg)
        counts[f"bars_{kind}"] = bars.height

    return counts


def _write(frame: pl.DataFrame, path: Path, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(
        path,
        compression=cfg.get("determinism.parquet_compression"),
        compression_level=cfg.get("determinism.parquet_compression_level"),
        statistics=True,
    )


# --------------------------------------------------------------------------- manifest


def manifest(root: Path) -> dict[str, str]:
    """SHA-256 of every artefact, keyed by path relative to the root. Sorted."""
    out: dict[str, str] = {}
    for pattern in MANIFEST_GLOBS:
        for path in sorted(Path(root).glob(pattern)):
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- gate


@dataclass(frozen=True)
class GateResult:
    canaries_caught: dict[str, bool]
    clean_flagged: dict[str, bool]
    deterministic: bool
    manifest_diff: tuple[str, ...]
    anchors: int

    @property
    def passed(self) -> bool:
        return (
            bool(self.canaries_caught)
            and all(self.canaries_caught.values())
            and not any(self.clean_flagged.values())
            and self.deterministic
        )

    def report(self) -> str:
        lines = ["Phase 1 gate", "=" * 60, "", f"Leakage canaries ({self.anchors} anchors each)"]
        for name, caught in sorted(self.canaries_caught.items()):
            lines.append(f"  {'caught  ' if caught else 'MISSED  '} {name}")
        lines.append("")
        lines.append("Clean features (must not be flagged)")
        for name, flagged in sorted(self.clean_flagged.items()):
            lines.append(f"  {'FLAGGED ' if flagged else 'passed  '} {name}")
        lines.append("")
        lines.append(f"Determinism: {'byte-identical' if self.deterministic else 'DIFFERS'}")
        for entry in self.manifest_diff:
            lines.append(f"  {entry}")
        lines.append("")
        lines.append(f"GATE: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def anchors(cfg: Config, root: Path, n: int = 6) -> list[datetime]:
    """A spread of RTH anchor timestamps, drawn deterministically from built bars.

    Anchors land *inside* a bar rather than exactly on a close. Real triggers fire at
    arbitrary moments, and an anchor that always coincides with a bar boundary would
    never exercise the case leak L1 is about — a bar that has opened but not finished.
    """
    bars = pl.read_parquet(Store.layout(root, "bars_time")).filter(pl.col("in_rth")).sort("close_ts")
    if bars.is_empty():
        raise ValueError("no RTH bars in the data root")

    lag = cfg.get("execution_realism.decision_lag_seconds")
    # Keep only bars whose successor abuts them, so there is a bar in progress at the
    # cutoff. Skip the first stretch so trailing windows have history to read.
    spans = bars.with_columns(
        pl.col("open_ts").shift(-1).alias("_next_open"),
        pl.col("close_ts").shift(-1).alias("_next_close"),
    ).filter(pl.col("_next_open") == pl.col("close_ts"))
    usable = spans.slice(int(spans.height * 0.35))
    if usable.is_empty():
        raise ValueError("no contiguous RTH bars to anchor inside")

    step = max(1, usable.height // n)
    picked = usable[::step].head(n)
    return [
        opened + (closed - opened) / 2 + timedelta(seconds=lag)
        for opened, closed in zip(picked["_next_open"], picked["_next_close"])
    ]


def run_gate(cfg: Config, root: Path, *, rebuild_args: dict) -> GateResult:
    t0s = anchors(cfg, root)
    results = audit_all(canaries.ALL, t0s, cfg, root)

    caught = {n: results[n].leaks for n in canaries.LEAKY}
    flagged = {n: results[n].leaks for n in canaries.CLEAN}

    before = manifest(root)
    with tempfile.TemporaryDirectory() as tmp:
        second = Path(tmp) / "data"
        build(cfg, second, **rebuild_args)
        after = manifest(second)

    diff = tuple(
        f"{key}: {before.get(key, '<missing>')[:12]} vs {after.get(key, '<missing>')[:12]}"
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    )
    return GateResult(caught, flagged, not diff, diff, len(t0s))


# --------------------------------------------------------------------------- cli


def _range(cfg: Config, sessions: int) -> dict:
    """A contiguous slice of the study period, taken from the start."""
    days = [s.session_date for s in SessionCalendar(cfg).sessions]
    return {"start": days[0], "end": days[min(sessions, len(days)) - 1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.pipeline", description=__doc__)
    parser.add_argument("command", choices=["build", "gate", "status"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=90, help="sessions to build")
    parser.add_argument("--ticks-per-session", type=int, default=4000)
    args = parser.parse_args(argv)

    cfg = config_module.load(args.config)
    root = Store.data_root(cfg)
    build_args = _range(cfg, args.sessions) | {"ticks_per_session": args.ticks_per_session}

    if cfg.is_synthetic:
        print("!! SYNTHETIC DATA — nothing produced by this run is a research result.\n")

    if args.command == "build":
        counts = build(cfg, root, **build_args)
        with ExperimentLog() as log:
            run_id = log.record(
                cfg, phase="phase1_build", result=counts, notes=f"sessions={args.sessions}"
            )
        for name, count in counts.items():
            print(f"  {name:16s} {count:>10,}")
        print(f"\nconfig {cfg.hash[:12]}  run {run_id[:12]}  -> {root}")
        return 0

    if args.command == "gate":
        result = run_gate(cfg, root, rebuild_args=build_args)
        print(result.report())
        with ExperimentLog() as log:
            log.record(
                cfg,
                phase="phase1_gate",
                result={
                    "passed": result.passed,
                    "canaries_caught": result.canaries_caught,
                    "clean_flagged": result.clean_flagged,
                    "deterministic": result.deterministic,
                },
            )
        return 0 if result.passed else 1

    print(json.dumps({"config_hash": cfg.hash, "data_root": str(root),
                      "data_version": cfg.get("data.version"),
                      "manifest": manifest(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
