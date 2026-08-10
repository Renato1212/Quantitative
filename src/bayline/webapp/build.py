"""Build the static application.

The whole product ships as three files — HTML, CSS, one JavaScript module — with the plan
embedded as JSON. No framework, no bundler, no CDN. A depot office on a bad connection
loads it once and it works; a customer's security team can read every line that runs.

The data payload is column-oriented (a `columns` spec plus arrays of row values) rather
than an array of objects. At ~1,700 jobs the object form repeats every key 1,700 times and
triples the download for nothing.
"""

from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

from bayline.config import Config
from bayline.store import Warehouse, day_to_date

ASSETS = Path(__file__).resolve().parent / "assets"

# Order matters: the client indexes by position.
JOB_COLUMNS = [
    "job_id", "vehicle_id", "registration", "vehicle_class", "depot_id", "contract_id",
    "component_id", "component_name", "start_day", "bay_days", "failure_probability",
    "horizon_probability",
    "planned_cost_eur", "unplanned_cost_eur", "premium_eur", "expected_saving_eur",
    "safety_critical", "immobilising", "risk_source", "days_since_service",
    "sensor_level", "sensor_slope_7d", "repair_eur", "recovery_eur", "driver_eur",
    "penalty_eur", "revenue_loss_eur", "penalty_per_day_eur", "revenue_per_day_eur",
    "off_road_days", "planned_repair_eur", "planned_driver_eur", "planned_revenue_eur",
]


def _payload(cfg: Config, warehouse: Warehouse, risk_summary: dict | None) -> dict:
    plan = json.loads((warehouse.root / "plan.json").read_text(encoding="utf-8"))
    assignments = {a["job_id"]: a for a in plan["assignments"]}
    vehicles = {r["vehicle_id"]: r for r in warehouse.rows("SELECT * FROM {vehicles}")}
    risk_by_key = {
        (r["vehicle_id"], r["component_id"]): r
        for r in warehouse.rows("SELECT * FROM {risk}")
    }

    horizon = int(cfg.get("risk.planning_horizon_days"))
    start_date = day_to_date(cfg, int(cfg.get("history.days")) - 1) + timedelta(days=1)

    rows = []
    for job in plan["jobs"]:
        assignment = assignments[job["job_id"]]
        vehicle = vehicles[job["vehicle_id"]]
        unplanned, planned = job["unplanned_breakdown"], job["planned_breakdown"]
        risk = risk_by_key.get((job["vehicle_id"], job["component_id"]), {})
        off_road = (
            unplanned["revenue_loss_eur"] / vehicle["revenue_per_day_eur"]
            if vehicle["revenue_per_day_eur"]
            else 0.0
        )
        rows.append(
            [
                job["job_id"], job["vehicle_id"], job["registration"], job["vehicle_class"],
                job["depot_id"], job["contract_id"], job["component_id"],
                job["component_name"], assignment["start_day"], job["bay_days"],
                job["failure_probability"], job["horizon_probability"],
                job["planned_cost_eur"],
                job["unplanned_cost_eur"], job["premium_eur"], job["expected_saving_eur"],
                bool(job["safety_critical"]), bool(job["immobilising"]),
                risk.get("risk_source", "model"), job["days_since_service"],
                job["sensor_level"], job["sensor_slope_7d"],
                unplanned["repair_eur"], unplanned["recovery_eur"], unplanned["driver_eur"],
                unplanned["penalty_eur"], unplanned["revenue_loss_eur"],
                vehicle["penalty_eur_per_missed_day"], vehicle["revenue_per_day_eur"],
                round(off_road, 3),
                planned["repair_eur"], planned["driver_eur"], planned["revenue_loss_eur"],
            ]
        )

    depot_names = {d["id"]: d["name"] for d in cfg.depots}
    contract_names = {c["id"]: c["name"] for c in cfg.contracts}

    fleet_stats = warehouse.rows(
        """
        SELECT depot_id, count(*) AS vehicles,
               round(avg(odometer_km) / 1000, 1) AS mean_odometer_km_000
        FROM {vehicles} GROUP BY depot_id ORDER BY depot_id
        """
    )

    return {
        "meta": {
            "config_hash": cfg.hash[:12],
            "generated_for": "simulated fleet",
            "vehicles": int(cfg.get("fleet.vehicles")),
            "telemetry_rows": warehouse.count("telemetry"),
            "history_days": int(cfg.get("history.days")),
            "horizon_days": horizon,
            "start_date": start_date.isoformat(),
            "label_horizon_days": int(cfg.get("risk.label_horizon_days")),
            "warehouse_mb": round(warehouse.size_bytes() / 1e6, 1),
            "depots": depot_names,
            "contracts": contract_names,
            "components": {c.id: c.name for c in cfg.components},
            "planned_eur": {c.id: c.planned_eur for c in cfg.components},
            "recovery_tow_eur": float(cfg.get("economics.recovery_tow_eur")),
            "fleet_stats": fleet_stats,
        },
        "columns": JOB_COLUMNS,
        "jobs": rows,
        "totals": plan["totals"],
        "comparison": plan["comparison"],
        "bay_utilisation": plan["bay_utilisation"],
        "scorecards": (risk_summary or {}).get("scorecards", []),
        "bay_value": plan.get("bay_value", []),
    }


def _html(payload: dict) -> str:
    meta = payload["meta"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bayline — maintenance plan</title>
<meta name="description" content="Expected-cost-of-delay engine for commercial fleet maintenance.">
<meta name="robots" content="noindex">
<link rel="stylesheet" href="app.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='13'>🔧</text></svg>">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>

<header class="topbar">
  <div class="wordmark">Bayline <span>the cost of waiting</span></div>
  <div class="topbar-meta">
    <span class="badge badge-sim" title="No real vehicle data is used anywhere in this build">
      simulated fleet
    </span>
    <span class="badge tnum">{meta['vehicles']} vehicles</span>
    <span class="badge tnum">{meta['telemetry_rows']:,} telemetry rows</span>
    <span class="badge tnum">config {meta['config_hash']}</span>
    <button id="theme" class="badge" type="button" aria-label="Toggle colour theme">◐</button>
  </div>
</header>

<main id="main">
  <p class="notice">
    <span aria-hidden="true">⚠</span>
    <span><strong>This fleet is generated, not observed.</strong> Bayline is the decision
    engine — the hazard model, the cost arithmetic and the bay scheduler. It is running here
    against {meta['telemetry_rows']:,} rows of simulated telemetry so it can be seen working
    end to end. Point it at a real telematics feed and only the ingest module changes.</span>
  </p>

  <div class="headline">
    <h1 id="headline">—</h1>
    <p class="sub" id="headline-sub">Loading the plan…</p>
  </div>

  <section class="kpis" id="kpis" aria-label="Headline measures"></section>

  <div class="grid grid-2">
    <div>
      <section class="panel">
        <div class="panel-head">
          <h2>The plan</h2>
          <p id="plan-count">—</p>
          <div class="right">
            <button type="button" class="ghost" id="only-scheduled" aria-pressed="false">
              Scheduled only
            </button>
          </div>
        </div>
        <div class="controls">
          <div class="control">
            <label for="f-depot">Depot</label>
            <select id="f-depot"><option value="">All</option></select>
          </div>
          <div class="control">
            <label for="f-component">Component</label>
            <select id="f-component"><option value="">All</option></select>
          </div>
          <div class="control">
            <label for="f-contract">Contract</label>
            <select id="f-contract"><option value="">All</option></select>
          </div>
          <div class="control">
            <label for="f-search">Search</label>
            <input type="search" id="f-search" placeholder="Vehicle or registration">
          </div>
        </div>
        <div class="panel-body flush">
          <div class="table-wrap">
            <table id="plan-table">
              <thead><tr id="plan-head"></tr></thead>
              <tbody id="plan-body"></tbody>
            </table>
          </div>
          <p class="empty" id="plan-empty" hidden>No jobs match these filters.</p>
        </div>
      </section>
    </div>

    <div>
      <section class="panel">
        <div class="panel-head">
          <h2>What if the numbers are wrong?</h2>
          <p>Your prices, not ours</p>
        </div>
        <div class="panel-body">
          <p class="caption flush-top">
            Every euro on this page rests on four assumptions. Move them and the ranking
            re-computes in the browser — the same arithmetic the scheduler used, so you can
            see whether the plan is robust to your own numbers or balanced on one of ours.
          </p>
          <div class="slider-row">
            <label for="s-penalty" class="slider-label">Contract penalties</label>
            <input type="range" id="s-penalty" min="0" max="200" value="100" step="5">
            <output id="o-penalty" class="tnum">100%</output>
          </div>
          <div class="slider-row">
            <label for="s-revenue" class="slider-label">Revenue per day</label>
            <input type="range" id="s-revenue" min="0" max="200" value="100" step="5">
            <output id="o-revenue" class="tnum">100%</output>
          </div>
          <div class="slider-row">
            <label for="s-recovery" class="slider-label">Recovery and tow</label>
            <input type="range" id="s-recovery" min="0" max="300" value="100" step="10">
            <output id="o-recovery" class="tnum">100%</output>
          </div>
          <div class="slider-row">
            <label for="s-risk" class="slider-label">Failure risk</label>
            <input type="range" id="s-risk" min="25" max="250" value="100" step="5">
            <output id="o-risk" class="tnum">100%</output>
          </div>
          <p class="caption" id="whatif-readout"></p>
          <button type="button" id="reset-whatif">Reset to configured prices</button>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Where the bays run out</h2>
          <p>Booked bay-days by depot</p>
        </div>
        <div class="panel-body" id="heatmap"></div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Against your current policy</h2></div>
        <div class="panel-body" id="comparison"></div>
      </section>
    </div>
  </div>

  <section class="panel">
    <div class="panel-head">
      <h2>Can the risk numbers be trusted?</h2>
      <p>One scorecard per component, scored on a held-out later period</p>
    </div>
    <div class="panel-body" id="scorecards"></div>
  </section>
</main>

<div class="scrim" id="scrim" hidden></div>
<aside class="drawer" id="drawer" data-open="false" aria-hidden="true" aria-labelledby="drawer-title">
  <div class="drawer-head">
    <div>
      <h2 id="drawer-title">—</h2>
      <p id="drawer-sub">—</p>
    </div>
    <button class="drawer-close" id="drawer-close" type="button" aria-label="Close">✕</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</aside>

<footer>
  <p><strong>How the number is built.</strong> Hourly telemetry becomes daily component
  health signals in the database ({meta['telemetry_rows']:,} rows over
  {meta['history_days']} days). A gradient-boosted hazard model per component predicts
  failure inside {meta['label_horizon_days']} days, trained on the earlier period and scored
  on the later one; its probabilities are calibrated and a component whose model cannot beat
  its own base rate falls back to that base rate and says so. Those probabilities multiply
  an itemised failure cost — repair, recovery, driver hours, contract penalty, lost revenue
  — to give the expected cost of leaving each component in service. A mixed-integer program
  then assigns jobs to bay-days to minimise total expected cost, and HiGHS proves the
  solution optimal rather than merely reporting the best it found.</p>
  <p><strong>What is real and what is not.</strong> The engine is real and runs on your data
  unchanged. The fleet is simulated: <code>src/bayline/simulate/</code> generates it from a
  seed, and it is the only module a customer replaces. Prices come from
  <code>config/bayline.yaml</code> and are hashed into every plan, so any schedule can be
  traced to the assumptions that produced it.</p>
</footer>

<script id="plan-data" type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>
<script src="app.js" type="module"></script>
</body>
</html>
"""


def build(cfg: Config, warehouse: Warehouse, risk_summary: dict | None) -> list[Path]:
    out = cfg.app_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = _payload(cfg, warehouse, risk_summary)

    written = []
    (out / "index.html").write_text(_html(payload), encoding="utf-8")
    written.append(out / "index.html")
    for asset in ("app.css", "app.js"):
        shutil.copyfile(ASSETS / asset, out / asset)
        written.append(out / asset)
    return written
