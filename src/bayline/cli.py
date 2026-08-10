"""Bayline pipeline.

    python -m bayline simulate   # fleet + telemetry + event history -> warehouse
    python -m bayline health     # daily component health signals (SQL, in-database)
    python -m bayline risk       # calibrated failure probabilities + model scorecards
    python -m bayline plan       # price every job, solve the MILP, compare to baselines
    python -m bayline app        # build the static web application
    python -m bayline all        # everything, in order
    python -m bayline status     # what exists, and what produced it

Each stage writes Parquet into the warehouse and the next reads it, so a stage can be rerun
alone. `plan` and `app` also write JSON the web application consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from bayline import config as config_module
from bayline.config import REPO_ROOT, Config
from bayline.economics.costs import build_job_costs
from bayline.health import signals
from bayline.risk import survival
from bayline.schedule import optimiser
from bayline.simulate import fleet as fleet_module
from bayline.simulate import telemetry as telemetry_module
from bayline.store import Warehouse, day_to_date


def _log(message: str) -> None:
    print(f"  {message}", flush=True)


def stage_simulate(cfg: Config, warehouse: Warehouse) -> dict:
    started = time.perf_counter()
    fleet = fleet_module.build_fleet(cfg)
    warehouse.write_rows("vehicles", fleet_module.fleet_rows(fleet, cfg))
    _log(f"vehicles      {len(fleet):>10,}")

    columns, failures = telemetry_module.simulate(cfg, fleet)
    warehouse.write_columns("telemetry", columns)
    rows = len(columns["day_index"])
    _log(f"telemetry     {rows:>10,} rows  ({warehouse.size_bytes() / 1e6:.1f} MB on disk)")

    services = telemetry_module.planned_services(cfg, fleet, failures)
    events = [
        {
            "vehicle_id": f["vehicle_id"],
            "component_id": f["component_id"],
            "day_index": f["day_index"],
            "kind": "failure",
            "immobilising": f["immobilising"],
        }
        for f in failures
    ] + [{**s, "immobilising": False} for s in services]
    events.sort(key=lambda e: (e["day_index"], e["vehicle_id"], e["component_id"]))
    warehouse.write_rows("events", events)
    _log(f"events        {len(events):>10,}  ({len(failures):,} failures, {len(services):,} planned)")

    return {
        "vehicles": len(fleet),
        "telemetry_rows": rows,
        "failures": len(failures),
        "planned_services": len(services),
        "seconds": round(time.perf_counter() - started, 1),
    }


def stage_health(cfg: Config, warehouse: Warehouse) -> dict:
    started = time.perf_counter()
    table = signals.build(cfg, warehouse)
    warehouse.write_arrow("health", table)
    _log(f"health        {table.num_rows:>10,} rows over {len(cfg.components)} components")
    return {
        "rows": table.num_rows,
        "per_component": signals.summarise(warehouse),
        "seconds": round(time.perf_counter() - started, 1),
    }


def stage_risk(cfg: Config, warehouse: Warehouse) -> dict:
    started = time.perf_counter()
    table, scorecards = survival.score_current(cfg, warehouse)
    warehouse.write_arrow("risk", table)

    tolerance = float(cfg.get("risk.max_calibration_error"))
    failed = [s["component_id"] for s in scorecards if not s["passes_gate"]]
    for card in scorecards:
        mark = "ok  " if card["passes_gate"] else "FAIL"
        _log(
            f"{mark} {card['component_id']:<10} AUC {card['auc']:.3f}  "
            f"cal.err {card['calibration_error']:.3f}  "
            f"skill {card['brier_skill']:+.3f}  base {card['base_rate'] * 100:.2f}%"
        )
    if failed:
        _log(f"!! {len(failed)} model(s) miss the {tolerance:.0%} calibration gate: {', '.join(failed)}")
        _log("   their euro figures are flagged as unreliable in the app rather than hidden")

    return {
        "rows": table.num_rows,
        "scorecards": scorecards,
        "gate_failures": failed,
        "seconds": round(time.perf_counter() - started, 1),
    }


def stage_plan(cfg: Config, warehouse: Warehouse) -> dict:
    started = time.perf_counter()
    vehicles = {r["vehicle_id"]: r for r in warehouse.rows("SELECT * FROM {vehicles}")}
    risk_rows = warehouse.rows("SELECT * FROM {risk}")

    jobs = build_job_costs(cfg, risk_rows, vehicles)
    # Only jobs worth considering reach the solver. A component with a negligible expected
    # saving is noise in the model, not a maintenance decision, and including 2,100 of them
    # makes the MILP slower and the screen unreadable.
    threshold = 25.0
    candidates = [
        j for j in jobs if j["expected_saving_eur"] >= threshold or j["safety_critical"]
    ]
    _log(f"candidates    {len(candidates):>10,} of {len(jobs):,} priced jobs above €{threshold:.0f}")

    plan = optimiser.solve(cfg, candidates)
    comparison = optimiser.compare(cfg, candidates, plan)
    _log(
        f"MILP          {plan.status} in {plan.solve_seconds:.2f}s  "
        f"{plan.scheduled} scheduled, {plan.deferred} deferred"
    )
    for row in comparison[1:]:
        _log(
            f"  vs {row['policy']:<32} €{row['total_expected_cost_eur']:>12,.0f}  "
            f"saving €{row['saving_vs_this_eur']:>10,.0f}"
        )

    values = optimiser.bay_value(cfg, candidates, plan)
    for row in values:
        _log(
            f"  +1 bay at {row['depot_id']:<9} worth €{row['value_eur']:>9,.0f} "
            f"over {cfg.get('risk.planning_horizon_days')} days "
            f"({row['extra_jobs_scheduled']:+d} jobs)"
        )

    assignments = plan.by_job()
    plan_rows = [
        {
            "job_id": j["job_id"],
            "vehicle_id": j["vehicle_id"],
            "component_id": j["component_id"],
            "depot_id": j["depot_id"],
            "start_day": assignments[j["job_id"]].start_day,
            "bay_days": j["bay_days"],
            "expected_cost_eur": assignments[j["job_id"]].cost_eur,
            "cost_if_never_eur": j["cost_if_never_eur"],
            "expected_saving_eur": j["expected_saving_eur"],
            "failure_probability": j["failure_probability"],
        }
        for j in candidates
    ]
    warehouse.write_rows("plan", plan_rows)

    payload = {
        "jobs": candidates,
        "assignments": [
            {"job_id": a.job_id, "start_day": a.start_day, "cost_eur": a.cost_eur}
            for a in plan.assignments
        ],
        "comparison": comparison,
        "bay_utilisation": plan.bay_utilisation,
        "bay_value": values,
        "totals": {
            "total_expected_cost_eur": plan.total_cost_eur,
            "scheduled": plan.scheduled,
            "deferred": plan.deferred,
            "status": plan.status,
            # Wall-clock time is logged, not stored. It belongs to the run, not to the
            # plan, and embedding it makes the shipped artefact differ between two runs of
            # the same config — which is exactly the property the config hash promises.
            "candidates": len(candidates),
            "priced_jobs": len(jobs),
        },
    }
    (warehouse.root / "plan.json").write_text(json.dumps(payload), encoding="utf-8")
    return {**payload["totals"], "seconds": round(time.perf_counter() - started, 1)}


def stage_app(cfg: Config, warehouse: Warehouse, risk_summary: dict | None) -> dict:
    from bayline.webapp import build as webapp_build

    started = time.perf_counter()
    written = webapp_build.build(cfg, warehouse, risk_summary)
    for path in written:
        _log(f"wrote         {path.relative_to(REPO_ROOT)}")
    return {"files": len(written), "seconds": round(time.perf_counter() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayline", description=__doc__)
    parser.add_argument(
        "command",
        choices=["simulate", "health", "risk", "plan", "app", "all", "status"],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--vehicles", type=int, default=None, help="override fleet size")
    parser.add_argument("--days", type=int, default=None, help="override history length")
    args = parser.parse_args(argv)

    cfg = config_module.load(args.config)
    if args.vehicles:
        cfg = cfg.with_overrides(**{"fleet.vehicles": args.vehicles})
    if args.days:
        cfg = cfg.with_overrides(**{"history.days": args.days})

    warehouse = Warehouse(cfg)
    print(
        f"\nBayline · config {cfg.hash[:12]} · {cfg.get('fleet.vehicles')} vehicles · "
        f"{cfg.get('history.days')} days · warehouse {warehouse.root.name}/\n"
    )

    try:
        if args.command == "status":
            return _status(cfg, warehouse)

        risk_summary = None
        order = ["simulate", "health", "risk", "plan", "app"]
        wanted = order if args.command == "all" else [args.command]

        for stage in wanted:
            print(f"[{stage}]")
            if stage == "simulate":
                stage_simulate(cfg, warehouse)
            elif stage == "health":
                _require(warehouse, "vehicles", "telemetry", "events")
                stage_health(cfg, warehouse)
            elif stage == "risk":
                _require(warehouse, "health", "events", "vehicles")
                risk_summary = stage_risk(cfg, warehouse)
            elif stage == "plan":
                _require(warehouse, "risk", "vehicles")
                stage_plan(cfg, warehouse)
            elif stage == "app":
                _require(warehouse, "plan", "risk", "vehicles")
                if risk_summary is None:
                    risk_summary = _cached_risk_summary(warehouse)
                stage_app(cfg, warehouse, risk_summary)
            print()

        if args.command in ("all", "risk", "app") and risk_summary:
            (warehouse.root / "risk_summary.json").write_text(
                json.dumps(risk_summary), encoding="utf-8"
            )
        return 0
    finally:
        warehouse.close()


def _require(warehouse: Warehouse, *tables: str) -> None:
    missing = warehouse.missing(*tables)
    if missing:
        raise SystemExit(
            f"missing warehouse tables: {', '.join(missing)}. "
            "Run `python -m bayline all` to build the chain from scratch."
        )


def _cached_risk_summary(warehouse: Warehouse) -> dict | None:
    path = warehouse.root / "risk_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _status(cfg: Config, warehouse: Warehouse) -> int:
    rows = []
    for table in ("vehicles", "telemetry", "events", "health", "risk", "plan"):
        if warehouse.path(table).exists():
            rows.append((table, warehouse.count(table), warehouse.path(table).stat().st_size))
        else:
            rows.append((table, None, 0))
    print(f"  {'table':<12} {'rows':>12} {'size':>10}")
    for name, count, size in rows:
        shown = f"{count:,}" if count is not None else "—"
        print(f"  {name:<12} {shown:>12} {size / 1e6:>9.1f} MB")
    print(f"\n  first day {day_to_date(cfg, 0)}   last day {day_to_date(cfg, int(cfg.get('history.days')) - 1)}")
    print(f"  total warehouse {warehouse.size_bytes() / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
