"""The scheduler, and the one invariant that caught a real bug.

An exact MILP minimising an objective cannot be beaten by a greedy heuristic on that same
objective under the same constraints. When a baseline came out €24,799 cheaper than the
optimum, the arithmetic was not wrong — the baselines were allowed to defer safety-critical
brake work that the MILP was forced to schedule. They were buying their advantage with brake
jobs, and the headline saving was understated as a result.

That class of error is invisible in a spot check and obvious to an invariant, which is why
``test_no_baseline_can_beat_the_optimum`` is the most valuable test in this file.
"""

from __future__ import annotations

import pytest

from bayline.schedule import optimiser

from conftest import make_job


@pytest.fixture
def small_cfg(cfg):
    """Two depots, tight capacity, a short horizon — solvable in well under a second."""
    return cfg.with_overrides(
        **{
            "risk.planning_horizon_days": 10,
            "fleet.depots": [
                {"id": "DEP-BCN", "name": "Barcelona", "bays": 2, "share": 0.5},
                {"id": "DEP-MAD", "name": "Madrid", "bays": 1, "share": 0.5},
            ],
        }
    )


def contended_jobs(n: int = 24) -> list[dict]:
    """More work than bay-days, so capacity actually binds and the ordering matters."""
    jobs = []
    for i in range(n):
        jobs.append(
            make_job(
                horizon=10,
                job_id=f"BL-{i:04d}:c",
                vehicle_id=f"BL-{i:04d}",
                depot_id="DEP-BCN" if i % 3 else "DEP-MAD",
                bay_days=1 + (i % 3),
                saving=200.0 + 130.0 * (i % 7),
                planned=400.0,
                failure_probability=0.02 + 0.01 * (i % 9),
                horizon_probability=0.03 + 0.01 * (i % 9),
                days_since_service=60 + 11 * (i % 13),
                safety_critical=(i % 8 == 0),
            )
        )
    return jobs


def test_no_baseline_can_beat_the_optimum(small_cfg):
    jobs = contended_jobs()
    optimal = optimiser.solve(small_cfg, jobs)
    for baseline in (
        optimiser.baseline_risk_ranked(small_cfg, jobs),
        optimiser.baseline_mileage_interval(small_cfg, jobs),
    ):
        assert optimal.total_cost_eur <= baseline.total_cost_eur + 1e-6, (
            f"{baseline.status} beat the exact MILP by "
            f"€{baseline.total_cost_eur - optimal.total_cost_eur:,.2f} — the two are not "
            "being scored under the same constraints"
        )


def test_every_policy_schedules_all_safety_critical_work(small_cfg):
    """The comparison is only fair if the baselines play by the MILP's rules."""
    jobs = contended_jobs()
    critical = {j["job_id"] for j in jobs if j["safety_critical"]}
    for plan in (
        optimiser.solve(small_cfg, jobs),
        optimiser.baseline_risk_ranked(small_cfg, jobs),
        optimiser.baseline_mileage_interval(small_cfg, jobs),
    ):
        deferred = {a.job_id for a in plan.assignments if a.start_day is None}
        assert not (deferred & critical), f"{plan.status} deferred safety-critical work"


def test_capacity_is_never_exceeded(small_cfg):
    """Jobs occupy consecutive bay-days. A knapsack formulation would pass this by accident
    on one-day jobs and fail it the moment a three-day clutch straddles a Friday."""
    jobs = contended_jobs()
    plan = optimiser.solve(small_cfg, jobs)
    bays = {d["id"]: d["bays"] for d in small_cfg.depots}
    horizon = int(small_cfg.get("risk.planning_horizon_days"))
    by_job = {j["job_id"]: j for j in jobs}

    occupancy: dict[tuple[str, int], int] = {}
    for a in plan.assignments:
        if a.start_day is None:
            continue
        job = by_job[a.job_id]
        assert a.start_day + job["bay_days"] <= horizon, "a job runs past the horizon"
        for t in range(a.start_day, a.start_day + job["bay_days"]):
            key = (job["depot_id"], t)
            occupancy[key] = occupancy.get(key, 0) + 1

    for (depot, day), used in occupancy.items():
        assert used <= bays[depot], f"{depot} day {day}: {used} jobs in {bays[depot]} bays"


def test_reported_total_matches_the_assignments(small_cfg):
    """The headline euro figure is the sum of the parts, not a separate calculation."""
    jobs = contended_jobs()
    plan = optimiser.solve(small_cfg, jobs)
    by_job = {j["job_id"]: j for j in jobs}
    total = 0.0
    for a in plan.assignments:
        job = by_job[a.job_id]
        expected = (
            job["cost_if_never_eur"]
            if a.start_day is None
            else job["cost_by_start_day"][a.start_day]
        )
        assert a.cost_eur == pytest.approx(expected, abs=0.01)
        total += expected
    assert plan.total_cost_eur == pytest.approx(total, abs=0.05)


def test_uncontended_work_is_scheduled_on_day_zero(small_cfg):
    """With capacity to spare, waiting is strictly worse and the optimum should not wait."""
    jobs = [make_job(horizon=10, job_id="a:c", depot_id="DEP-BCN", bay_days=1)]
    plan = optimiser.solve(small_cfg, jobs)
    assert plan.assignments[0].start_day == 0


def test_a_bay_is_never_worth_less_than_nothing(small_cfg):
    """Extra capacity relaxes the feasible set, so the optimum cannot get worse.

    A negative shadow price here would mean the solver returned a suboptimal answer for one
    of the two problems, which is worth knowing before it is printed next to 'what one more
    bay is worth'.
    """
    jobs = contended_jobs()
    optimal = optimiser.solve(small_cfg, jobs)
    for row in optimiser.bay_value(small_cfg, jobs, optimal):
        assert row["value_eur"] >= -0.01, row
        assert row["extra_jobs_scheduled"] >= 0


def test_empty_input_is_a_plan_not_a_crash(small_cfg):
    plan = optimiser.solve(small_cfg, [])
    assert plan.status == "empty" and plan.total_cost_eur == 0.0
