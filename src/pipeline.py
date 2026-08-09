"""Phase 1 pipeline: build the data root, then check the gate.

    python -m src.pipeline build     # ticks -> bars, sessions, roll schedule
    python -m src.pipeline gate      # leakage canaries + byte-identical rerun
    python -m src.pipeline phase2    # trigger families, labels, per-year counts
    python -m src.pipeline phase3    # base rates, tails, stability, cost floor
    python -m src.pipeline phase4    # pre-registered hypotheses, BH corrected
    python -m src.pipeline phase5    # clustering + GBM diagnosis
    python -m src.pipeline phase6    # meta-labeling readiness
    python -m src.pipeline all       # build, gate, then every phase in order
    python -m src.pipeline status    # what exists, what it was built from

Phases 3 onward run on the **training block only**. The holdout is opened once, at the
end, by a human who has decided to open it — never as a side effect of running the
pipeline.

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
from src.events.families import build_events, events_per_year
from src.features.context import FEATURES, build_feature_matrix
from src.labels.atr import atr_table
from src.labels.triple_barrier import label_events, stop_distance_multiples
from src.models import discovery, meta_labeling
from src.reporting import reports
from src.stats import base_rates, hypotheses
from src.validation import canaries
from src.validation.leakage import audit_all

# Artefacts `build` produces, whose bytes must match across runs. The SQLite log is not
# one of them; nor are the phase-2 outputs, which are checked separately by recomputing
# them rather than by rebuilding the whole tape.
MANIFEST_GLOBS = (
    "raw/*.parquet",
    "bars/*.parquet",
    "events/sessions.parquet",
    "events/roll_schedule.parquet",
)


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
    derived_deterministic: bool = True

    @property
    def passed(self) -> bool:
        return (
            bool(self.canaries_caught)
            and all(self.canaries_caught.values())
            and not any(self.clean_flagged.values())
            and self.deterministic
            and self.derived_deterministic
        )

    def report(self) -> str:
        lines = ["Phase 1 gate", "=" * 60, "", f"Leakage canaries ({self.anchors} anchors each)"]
        for name, caught in sorted(self.canaries_caught.items()):
            lines.append(f"  {'caught  ' if caught else 'MISSED  '} {name}")
        lines.append("")
        lines.append("Clean and production features (must not be flagged)")
        for name, flagged in sorted(self.clean_flagged.items()):
            lines.append(f"  {'FLAGGED ' if flagged else 'passed  '} {name}")
        lines.append("")
        lines.append(f"Determinism (build): {'byte-identical' if self.deterministic else 'DIFFERS'}")
        for entry in self.manifest_diff:
            lines.append(f"  {entry}")
        lines.append(
            f"Determinism (events, ATR, labels): "
            f"{'identical' if self.derived_deterministic else 'DIFFERS'}"
        )
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
        opened + (closed - opened) / 2
        for opened, closed in zip(picked["_next_open"], picked["_next_close"])
    ]


def run_gate(cfg: Config, root: Path, *, rebuild_args: dict) -> GateResult:
    t0s = anchors(cfg, root)
    # The production feature set is audited by the same probes as the canaries. A gate
    # that only checks synthetic examples proves the detector works, not that the
    # features are clean.
    results = audit_all(canaries.ALL | FEATURES, t0s, cfg, root)

    caught = {n: results[n].leaks for n in canaries.LEAKY}
    flagged = {n: results[n].leaks for n in canaries.CLEAN} | {
        n: results[n].leaks for n in FEATURES
    }

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
    return GateResult(caught, flagged, not diff, diff, len(t0s), _derived_is_deterministic(cfg, root))


def _derived_is_deterministic(cfg: Config, root: Path) -> bool:
    """Recompute events, ATR and labels twice and compare.

    The phase-2 outputs are pure functions of the bars, so this checks them without
    regenerating the tape. The feature matrix is excluded here because it is the slow
    part; ``tests/test_phases.py`` covers it on a small panel.
    """
    calendar = SessionCalendar(cfg)
    data = _load(cfg, root)

    def once() -> str:
        events = build_events(data["bars"], cfg, calendar, data["schedule"])
        atr = atr_table(data["bars"], cfg)
        labels = label_events(events, data["ticks"], atr, cfg)
        digest = hashlib.sha256()
        for frame in (events, atr, labels):
            digest.update(frame.hash_rows().to_numpy().tobytes())
        return digest.hexdigest()

    return once() == once()


# --------------------------------------------------------------------------- phases


def _load(cfg: Config, root: Path) -> dict:
    return {
        "bars": pl.read_parquet(Store.layout(root, "bars_time")),
        "ticks": pl.read_parquet(Store.layout(root, "ticks")),
        "schedule": pl.read_parquet(root / "events" / "roll_schedule.parquet"),
    }


def run_phase2(cfg: Config, root: Path) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Trigger families and labels. Writes events, labels and the feature matrix."""
    calendar = SessionCalendar(cfg)
    data = _load(cfg, root)

    events = build_events(data["bars"], cfg, calendar, data["schedule"])
    _write(events, root / "events" / "events.parquet", cfg)

    atr = atr_table(data["bars"], cfg)
    _write(atr, root / "events" / "atr.parquet", cfg)

    labels = stop_distance_multiples(label_events(events, data["ticks"], atr, cfg), cfg)
    _write(labels, root / "labels" / "labels.parquet", cfg)

    store = Store.from_config(cfg, root=root)
    try:
        features = build_feature_matrix(events, store, cfg, calendar)
    finally:
        store.close()
    _write(features, root / "features" / "context.parquet", cfg)

    return events_per_year(events), labels, events.height - labels.height


def training_panel(cfg: Config, root: Path) -> pl.DataFrame:
    """Labels joined to features, restricted to the training block (C3).

    The holdout is not merely unused here — it is not read. A function that returns it by
    default is a function that will eventually be called by accident.
    """
    labels = pl.read_parquet(root / "labels" / "labels.parquet")
    features = pl.read_parquet(root / "features" / "context.parquet")
    panel = labels.join(features, on=["trigger_family", "t0"], how="inner")

    train_start, train_end = SessionCalendar(cfg).chronological_splits()["train"]
    return panel.filter(
        (pl.col("session_date") >= train_start) & (pl.col("session_date") <= train_end)
    )


def run_phase3(cfg: Config, panel: pl.DataFrame) -> dict:
    strata = base_rates.with_strata(panel, cfg)
    return {
        "quantiles": base_rates.quantile_table(panel),
        "tails": base_rates.tail_table(panel, cfg),
        "by_year": base_rates.stratified_table(strata, cfg, "year"),
        "by_regime": base_rates.stratified_table(strata, cfg, "vol_regime"),
        "by_time": base_rates.stratified_table(strata, cfg, "time_of_day"),
        "costs": base_rates.cost_floor(cfg),
    }


def run_phase4(cfg: Config, panel: pl.DataFrame):
    families = hypotheses.load_families()
    if not families:
        raise FileNotFoundError("no pre-registered hypotheses in research/hypotheses/")
    family = families[0]
    return family, hypotheses.test_family(family, panel, cfg)


def run_phase5(cfg: Config, panel: pl.DataFrame):
    features = sorted(FEATURES)
    clusters = discovery.cluster_states(panel, features, cfg)
    diagnosis = discovery.diagnose(panel, features, cfg)
    return clusters, diagnosis, discovery.effect_size_sanity(diagnosis.r2)


# --------------------------------------------------------------------------- cli


def _range(cfg: Config, sessions: int) -> dict:
    """A contiguous slice of the study period, taken from the start."""
    days = [s.session_date for s in SessionCalendar(cfg).sessions]
    return {"start": days[0], "end": days[min(sessions, len(days)) - 1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.pipeline", description=__doc__)
    parser.add_argument(
        "command",
        choices=["build", "gate", "phase2", "phase3", "phase4", "phase5", "phase6", "all", "status"],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=90, help="sessions to build")
    parser.add_argument("--ticks-per-session", type=int, default=4000)
    args = parser.parse_args(argv)

    cfg = config_module.load(args.config)
    root = Store.data_root(cfg)
    build_args = _range(cfg, args.sessions) | {"ticks_per_session": args.ticks_per_session}

    if cfg.is_synthetic:
        print("!! SYNTHETIC DATA — nothing produced by this run is a research result.\n")

    if args.command in ("build", "all"):
        counts = build(cfg, root, **build_args)
        with ExperimentLog() as log:
            run_id = log.record(
                cfg, phase="phase1_build", result=counts, notes=f"sessions={args.sessions}"
            )
        for name, count in counts.items():
            print(f"  {name:16s} {count:>10,}")
        print(f"\nconfig {cfg.hash[:12]}  run {run_id[:12]}  -> {root}")
        if args.command == "build":
            return 0

    if args.command in ("gate", "all"):
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
        if not result.passed:
            return 1
        if args.command == "gate":
            return 0

    if args.command in ("phase2", "phase3", "phase4", "phase5", "phase6", "all"):
        return _run_phases(cfg, root, args)

    print(json.dumps({"config_hash": cfg.hash, "data_root": str(root),
                      "data_version": cfg.get("data.version"),
                      "manifest": manifest(root)}, indent=2))
    return 0


def _run_phases(cfg: Config, root: Path, args) -> int:
    wanted = ["phase2", "phase3", "phase4", "phase5", "phase6"] if args.command == "all" else [args.command]
    written: list[Path] = []

    with ExperimentLog() as log:
        if "phase2" in wanted:
            per_year, labels, dropped = run_phase2(cfg, root)
            written.append(reports.phase2(cfg, per_year, labels, dropped))
            log.record(cfg, phase="phase2", result={
                "events": int(per_year["n"].sum()), "labelled": labels.height, "dropped": dropped
            })
            print(f"  phase2: {labels.height} labelled events across "
                  f"{per_year['trigger_family'].n_unique()} families")

        panel = training_panel(cfg, root) if wanted != ["phase6"] else pl.DataFrame()
        if not panel.is_empty():
            print(f"  training panel: {panel.height} events, "
                  f"{panel['session_date'].n_unique()} sessions (holdout not read)")

        if "phase3" in wanted:
            tables = run_phase3(cfg, panel)
            written.append(reports.phase3(cfg, **tables))
            log.record(cfg, phase="phase3", result={"n": panel.height})
            print("  phase3: base rates written")

        if "phase4" in wanted:
            family, results = run_phase4(cfg, panel)
            trials = log.trial_count()
            written.append(reports.phase4(cfg, results, family, trials))
            for row in results.iter_rows(named=True):
                log.record(cfg, phase="phase4", hypothesis=row["id"], result={
                    "effect": row["effect"], "p_value": row["p_value"],
                    "survives_bh": row["survives_bh"], "supported": row["supported"],
                })
            supported = int(results["supported"].sum())
            print(f"  phase4: {supported}/{family.size} hypotheses supported "
                  f"({int(results['survives_bh'].sum())} survived BH)")

        if "phase5" in wanted:
            clusters, diagnosis, sanity = run_phase5(cfg, panel)
            written.append(reports.phase5(cfg, clusters, diagnosis, sanity))
            log.record(cfg, phase="phase5", result={
                "stability_ari": clusters.stability_ari, "spearman_ic": diagnosis.spearman_ic,
                "r2": diagnosis.r2,
            })
            print(f"  phase5: clusters {'stable' if clusters.stable else 'UNSTABLE'} "
                  f"(ARI {clusters.stability_ari:.3f}), IC {diagnosis.spearman_ic:+.4f}")

        if "phase6" in wanted:
            note = meta_labeling.readiness(meta_labeling.load_trade_log())
            written.append(reports.phase6(cfg, note))
            log.record(cfg, phase="phase6", result={"readiness": note})
            print(f"  phase6: {note}")

        print(f"\ntrial count: {log.trial_count()} "
              f"({log.trial_count(exclude_synthetic=True)} against real data)")

    for path in written:
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
