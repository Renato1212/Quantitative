"""The pricing layer, which is where a sign error becomes a spending decision."""

from __future__ import annotations

import math

import pytest

from bayline.economics import costs
from bayline.risk.survival import cumulative_failure_probability, daily_hazard


def vehicle(**overrides) -> dict:
    base = {
        "vehicle_id": "BL-0001",
        "depot_id": "DEP-BCN",
        "vehicle_class": "artic",
        "contract_id": "general",
        "registration": "0000 AAA",
        "revenue_per_day_eur": 940.0,
        "penalty_eur_per_missed_day": 0.0,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ hazard arithmetic


def test_waiting_zero_days_carries_no_risk():
    assert cumulative_failure_probability(0.4, 21, 0) == 0.0


def test_cumulative_probability_is_monotone_in_wait():
    values = [cumulative_failure_probability(0.2, 21, d) for d in range(0, 29)]
    assert values == sorted(values)


def test_horizon_probability_round_trips():
    """Spreading a horizon probability over days and re-accumulating must return it."""
    assert cumulative_failure_probability(0.35, 21, 21) == pytest.approx(0.35, abs=1e-9)


def test_daily_hazard_handles_certainty_without_blowing_up():
    assert 0.0 < daily_hazard(1.0, 21) < 1.0
    assert daily_hazard(0.0, 21) == 0.0


# ------------------------------------------------------------------ cost structure


def test_unplanned_always_costs_more_than_planned(cfg):
    for component in cfg.components:
        for contract in cfg.contracts:
            v = vehicle(penalty_eur_per_missed_day=contract["penalty_eur_per_missed_day"])
            unplanned = costs.failure_cost(cfg, component, v)
            planned = costs.planned_cost(cfg, component, v)
            assert unplanned.total_eur > planned.total_eur, component.id


def test_immobilising_failures_carry_a_recovery_and_a_longer_outage(cfg):
    immobilising = next(c for c in cfg.components if c.immobilising)
    derate = next(c for c in cfg.components if not c.immobilising)
    assert costs.failure_cost(cfg, immobilising, vehicle()).recovery_eur > 0
    assert costs.failure_cost(cfg, derate, vehicle()).recovery_eur == 0
    assert costs.days_off_road(immobilising, unplanned=True) > costs.days_off_road(
        immobilising, unplanned=False
    )


def test_a_cheap_part_on_an_expensive_contract_can_outrank_an_expensive_part(cfg):
    """The asymmetry the whole product rests on.

    A risk-ranked list cannot see this, because both jobs have the same probability and the
    turbo is six times the repair bill. If this assertion ever fails, the economics layer has
    stopped saying anything a telematics alert list does not already say.
    """
    battery = cfg.component("battery")
    turbo = cfg.component("turbo")
    pharma = vehicle(penalty_eur_per_missed_day=4100.0, revenue_per_day_eur=610.0)
    spot = vehicle(penalty_eur_per_missed_day=0.0, revenue_per_day_eur=940.0)

    assert battery.repair_eur < turbo.repair_eur
    battery_premium = (
        costs.failure_cost(cfg, battery, pharma).total_eur
        - costs.planned_cost(cfg, battery, pharma).total_eur
    )
    turbo_premium = (
        costs.failure_cost(cfg, turbo, spot).total_eur
        - costs.planned_cost(cfg, turbo, spot).total_eur
    )
    assert battery_premium > turbo_premium


def test_planned_work_carries_no_penalty(cfg):
    """A booked slot is planned around. If it were not, there would be nothing to sell."""
    v = vehicle(penalty_eur_per_missed_day=4100.0)
    assert costs.planned_cost(cfg, cfg.components[0], v).penalty_eur == 0.0


# ------------------------------------------------------------------ job construction


def risk_row(component_id: str, probability: float) -> dict:
    return {
        "vehicle_id": "BL-0001",
        "component_id": component_id,
        "failure_probability": probability,
        "horizon_days": 21,
        "staleness_days": 0,
        "days_since_service": 140,
        "sensor_level": 0.4,
        "sensor_slope_7d": 0.01,
        "model_passes_gate": True,
    }


def test_cost_of_starting_later_never_decreases(cfg):
    jobs = costs.build_job_costs(
        cfg, [risk_row("clutch", 0.18)], {"BL-0001": vehicle(penalty_eur_per_missed_day=850.0)}
    )
    series = jobs[0]["cost_by_start_day"]
    assert series == sorted(series)
    assert series[0] == pytest.approx(jobs[0]["planned_cost_eur"])


def test_deferral_is_priced_at_the_horizon_not_at_certainty(cfg):
    """Charging deferral at P=1 would bully the scheduler into servicing everything.

    ``cost_if_never`` is what the plan pays for a job the bays could not take. It has to be
    the honest expected cost of leaving it, or capacity looks infinitely valuable and the bay
    shadow price — the number a finance director acts on — is fiction.
    """
    job = costs.build_job_costs(cfg, [risk_row("dpf", 0.10)], {"BL-0001": vehicle()})[0]
    ceiling = job["planned_cost_eur"] + job["premium_eur"]
    assert job["cost_by_start_day"][-1] < job["cost_if_never_eur"] < ceiling


def test_the_two_probabilities_are_distinct_and_ordered(cfg):
    """21-day for the human, 28-day for the money. Swapping them understates every euro."""
    job = costs.build_job_costs(cfg, [risk_row("cooling", 0.12)], {"BL-0001": vehicle()})[0]
    assert job["horizon_days"] == 21
    assert job["horizon_probability"] > job["failure_probability"]
    assert math.isclose(
        job["expected_saving_eur"],
        job["horizon_probability"] * job["premium_eur"],
        rel_tol=1e-4,
    )
