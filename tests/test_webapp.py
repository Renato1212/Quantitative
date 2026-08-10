"""The client's arithmetic must reproduce the engine's.

The what-if sliders re-price every job in the browser. That is the right design — a
round-trip per keystroke would make the page useless — but it means the same pricing logic
exists twice, in two languages, and the two can drift.

An earlier version drifted twice. First the client reverse-engineered the planned cost out
of a total (``planned_cost_eur - planned_eur - 14``) and was wrong whenever bay hours were
not what it assumed. Then it used the 21-day probability where the engine used the 28-day
one, and the penalty slider at 180% moved total exposure by −0.2% when it should have moved
it by +30%. Both were invisible on screen; both are caught here.

The test executes the *actual shipped* ``reprice`` from ``app.js`` under Node, so it cannot
pass by testing a Python re-implementation of what the JavaScript was supposed to do.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap

import pytest

from bayline.config import REPO_ROOT
from bayline.economics import costs
from bayline.webapp.build import JOB_COLUMNS

APP_JS = REPO_ROOT / "src" / "bayline" / "webapp" / "assets" / "app.js"
NODE = shutil.which("node")


def test_the_payload_declares_every_field_the_client_prices_with():
    """The client indexes jobs by column position, so a missing field is a silent NaN."""
    required = {
        "repair_eur", "recovery_eur", "driver_eur", "penalty_eur", "revenue_loss_eur",
        "planned_repair_eur", "planned_driver_eur", "planned_revenue_eur",
        "failure_probability", "horizon_probability",
    }
    assert required <= set(JOB_COLUMNS), sorted(required - set(JOB_COLUMNS))


def test_job_columns_are_unique():
    assert len(JOB_COLUMNS) == len(set(JOB_COLUMNS))


INDEX_HTML = REPO_ROOT / "app" / "index.html"


@pytest.mark.skipif(not INDEX_HTML.exists(), reason="app not built")
def test_the_shipped_payload_carries_no_wall_clock_value():
    """Two runs of the same config must produce the same bytes.

    The MILP's solve time was embedded in the page, which meant a rebuild changed the
    artefact without changing a single input — and the config hash printed beside it
    quietly stopped meaning what it claims to mean. Timing belongs in the run log.
    """
    source = INDEX_HTML.read_text(encoding="utf-8")
    start = source.index('<script id="plan-data" type="application/json">')
    body = source[source.index(">", start) + 1 : source.index("</script>", start)]
    payload = json.loads(body)

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not any(
                    token in key for token in ("seconds", "elapsed", "timestamp", "_at")
                ), f"{path}.{key} looks like a wall-clock measurement"
                walk(value, f"{path}.{key}")

    walk(payload["meta"], "meta")
    walk(payload["totals"], "totals")


def _engine_job(cfg) -> dict:
    """One priced job straight out of the engine, with its itemised parts."""
    vehicle = {
        "vehicle_id": "BL-0001",
        "depot_id": "DEP-BCN",
        "vehicle_class": "artic",
        "contract_id": "pharma",
        "registration": "1234 ABC",
        "revenue_per_day_eur": 940.0,
        "penalty_eur_per_missed_day": 2400.0,
    }
    risk = {
        "vehicle_id": "BL-0001",
        "component_id": "clutch",
        "failure_probability": 0.1734,
        "horizon_days": int(cfg.get("risk.label_horizon_days")),
        "staleness_days": 0,
        "days_since_service": 190,
        "sensor_level": 0.62,
        "sensor_slope_7d": 0.014,
        "model_passes_gate": True,
    }
    job = costs.build_job_costs(cfg, [risk], {"BL-0001": vehicle})[0]
    unplanned, planned = job["unplanned_breakdown"], job["planned_breakdown"]
    return {
        **job,
        "repair_eur": unplanned["repair_eur"],
        "recovery_eur": unplanned["recovery_eur"],
        "driver_eur": unplanned["driver_eur"],
        "penalty_eur": unplanned["penalty_eur"],
        "revenue_loss_eur": unplanned["revenue_loss_eur"],
        "planned_repair_eur": planned["repair_eur"],
        "planned_driver_eur": planned["driver_eur"],
        "planned_revenue_eur": planned["revenue_loss_eur"],
    }


def _run_reprice(job: dict, whatif: dict) -> dict:
    """Execute the shipped ``reprice`` against a job, in Node, with no other page state."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function reprice(job)")
    end = source.index("\n}", start) + 2
    harness = textwrap.dedent(
        """
        const state = { whatif: %s };
        %s
        console.log(JSON.stringify(reprice(%s)));
        """
    ) % (json.dumps(whatif), source[start:end], json.dumps(job))
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", harness],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


NEUTRAL = {"penalty": 1.0, "revenue": 1.0, "recovery": 1.0, "risk": 1.0}


@pytest.mark.skipif(NODE is None, reason="node is required to run the shipped client code")
def test_the_client_reproduces_the_engine_exactly_at_neutral_settings(cfg):
    """With every slider at 100% the browser must agree with the engine to the cent.

    Any disagreement means the page is showing a number the plan was not built from, which
    is worse than showing no number at all.
    """
    job = _engine_job(cfg)
    client = _run_reprice(job, NEUTRAL)

    assert client["unplanned"] == pytest.approx(job["unplanned_cost_eur"], abs=0.01)
    assert client["planned"] == pytest.approx(job["planned_cost_eur"], abs=0.01)
    assert client["premium"] == pytest.approx(job["premium_eur"], abs=0.01)
    assert client["saving"] == pytest.approx(job["expected_saving_eur"], abs=0.01)


@pytest.mark.skipif(NODE is None, reason="node is required to run the shipped client code")
def test_the_client_prices_money_on_the_planning_horizon(cfg):
    """The bug that made the penalty slider look inert: 21-day probability, 28-day money."""
    job = _engine_job(cfg)
    client = _run_reprice(job, NEUTRAL)
    assert client["probability"] == pytest.approx(job["horizon_probability"], abs=1e-9)
    assert client["shown_probability"] == pytest.approx(job["failure_probability"], abs=1e-9)
    assert client["probability"] > client["shown_probability"]


@pytest.mark.skipif(NODE is None, reason="node is required to run the shipped client code")
def test_raising_the_penalty_assumption_raises_exposure(cfg):
    """A job on a penalty-bearing contract must respond to the penalty slider, and by an
    amount that reconciles with the engine's own arithmetic rather than approximately."""
    job = _engine_job(cfg)
    assert job["unplanned_breakdown"]["penalty_eur"] > 0
    base = _run_reprice(job, NEUTRAL)
    raised = _run_reprice(job, {**NEUTRAL, "penalty": 1.8})

    expected_premium = base["premium"] + 0.8 * job["unplanned_breakdown"]["penalty_eur"]
    assert raised["premium"] == pytest.approx(expected_premium, abs=0.01)
    assert raised["saving"] > base["saving"] * 1.1


@pytest.mark.skipif(NODE is None, reason="node is required to run the shipped client code")
def test_probability_is_clamped_below_one(cfg):
    """The risk slider goes to 300%. A probability above 1 would produce negative savings
    further down and a plan that recommends breaking things on purpose."""
    client = _run_reprice(_engine_job(cfg), {**NEUTRAL, "risk": 50.0})
    assert client["probability"] <= 0.999
    assert client["shown_probability"] <= 0.999
