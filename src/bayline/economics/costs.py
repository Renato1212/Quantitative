"""What a failure costs, and what waiting costs.

This is the module the product is actually about. Predicting failures is a solved problem
sold by a dozen vendors. Pricing the *decision* is not, because it requires putting the
repair bill, the recovery truck, the driver's idle hours, the contract penalty and the lost
revenue day into the same arithmetic — and those numbers live in four different systems and
one person's head.

The central quantity is the **cost of waiting**: for a component that is not yet broken,
the difference between servicing it on a chosen day and letting it run. It is not a risk
score. It is euros, and it is the number a fleet director can take to a finance meeting.

Two asymmetries that make the answer non-obvious, and that a risk-ranked list gets wrong:

- A cheap component on an expensive contract outranks an expensive component on general
  haulage. A €620 battery on an automotive line-side run carries more expected cost than a
  €3,850 turbo on spot freight, because the penalty dwarfs the repair.
- An immobilising failure costs a recovery and a full day; a derate does not. Two components
  with identical failure probability can differ threefold in expected cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from bayline.config import Component, Config
from bayline.risk.survival import cumulative_failure_probability


@dataclass(frozen=True)
class FailureCost:
    """The breakdown of one unplanned failure, itemised so it can be argued with."""

    repair_eur: float
    recovery_eur: float
    driver_eur: float
    penalty_eur: float
    revenue_loss_eur: float

    @property
    def total_eur(self) -> float:
        return (
            self.repair_eur
            + self.recovery_eur
            + self.driver_eur
            + self.penalty_eur
            + self.revenue_loss_eur
        )

    def as_dict(self) -> dict:
        return {
            "repair_eur": round(self.repair_eur, 2),
            "recovery_eur": round(self.recovery_eur, 2),
            "driver_eur": round(self.driver_eur, 2),
            "penalty_eur": round(self.penalty_eur, 2),
            "revenue_loss_eur": round(self.revenue_loss_eur, 2),
            "total_eur": round(self.total_eur, 2),
        }


def days_off_road(component: Component, *, unplanned: bool) -> float:
    """Calendar days the vehicle is unavailable.

    An unplanned failure costs more than the repair time: the vehicle has to be recovered,
    it joins the queue rather than a booked slot, and parts were not pre-staged. One extra
    day for a roadside strand, half a day otherwise — before the repair itself starts.
    """
    repair_days = component.bay_hours / 10.0
    if not unplanned:
        return repair_days
    return repair_days + (1.0 if component.immobilising else 0.5)


def failure_cost(cfg: Config, component: Component, vehicle: dict) -> FailureCost:
    """The all-in cost of this component failing on this vehicle, in service."""
    economics = cfg.get("economics")
    idle_rate = float(economics["driver_idle_eur_per_hour"])
    driver_hours = float(
        economics["driver_hours_lost_roadside"]
        if component.immobilising
        else economics["driver_hours_lost_depot"]
    )
    off_road = days_off_road(component, unplanned=True)
    return FailureCost(
        repair_eur=component.repair_eur,
        recovery_eur=float(economics["recovery_tow_eur"]) if component.immobilising else 0.0,
        driver_eur=idle_rate * driver_hours,
        penalty_eur=float(vehicle["penalty_eur_per_missed_day"]) * off_road,
        revenue_loss_eur=float(vehicle["revenue_per_day_eur"]) * off_road,
    )


def planned_cost(cfg: Config, component: Component, vehicle: dict) -> FailureCost:
    """The cost of doing the same work as booked maintenance.

    The same shape as a failure so the two are directly comparable. No recovery, minimal
    driver idle, and no contract penalty — a booked slot is planned around, which is the
    entire value of planning it.
    """
    economics = cfg.get("economics")
    off_road = days_off_road(component, unplanned=False)
    return FailureCost(
        repair_eur=component.planned_eur,
        recovery_eur=0.0,
        driver_eur=float(economics["driver_idle_eur_per_hour"]) * 0.5,
        penalty_eur=0.0,
        revenue_loss_eur=float(vehicle["revenue_per_day_eur"]) * off_road,
    )


def expected_cost_of_waiting(
    cfg: Config,
    component: Component,
    vehicle: dict,
    *,
    failure_probability: float,
    horizon_days: int,
    wait_days: int,
) -> dict:
    """Euros expected to be lost by leaving this component in service for ``wait_days``.

    The comparison is against servicing it today, so the planned repair bill cancels out of
    both branches and what remains is the *risk premium* plus the difference in downtime.
    That framing matters: the question is never "should this be repaired" — it will be
    repaired either way — but "who chooses the day, us or the component".
    """
    p_fail = cumulative_failure_probability(failure_probability, horizon_days, wait_days)
    unplanned = failure_cost(cfg, component, vehicle)
    planned = planned_cost(cfg, component, vehicle)
    # If it fails while waiting we pay the unplanned bill instead of the planned one.
    premium = unplanned.total_eur - planned.total_eur
    return {
        "wait_days": wait_days,
        "failure_probability": round(p_fail, 6),
        "expected_cost_eur": round(p_fail * premium, 2),
        "premium_if_it_fails_eur": round(premium, 2),
        "unplanned": unplanned.as_dict(),
        "planned": planned.as_dict(),
    }


def build_job_costs(cfg: Config, risk_rows: list[dict], vehicles: dict[str, dict]) -> list[dict]:
    """One candidate job per (vehicle, component), priced across the planning horizon.

    ``cost_by_start_day[d]`` is the total expected cost of starting this job on day ``d``:
    the planned work plus the risk carried while waiting for it. ``cost_if_never`` is the
    cost of not doing it inside the horizon at all, which is what the scheduler pays when
    the bays are full — and the number that turns "we are short of capacity" into a price.
    """
    horizon = int(cfg.get("risk.planning_horizon_days"))
    jobs: list[dict] = []

    for row in risk_rows:
        component = cfg.component(row["component_id"])
        vehicle = vehicles[row["vehicle_id"]]
        probability = float(row["failure_probability"])
        label_horizon = int(row["horizon_days"])

        planned = planned_cost(cfg, component, vehicle)
        unplanned = failure_cost(cfg, component, vehicle)
        premium = unplanned.total_eur - planned.total_eur

        p_horizon = cumulative_failure_probability(probability, label_horizon, horizon)
        cost_by_day = []
        for start in range(horizon):
            p_fail = cumulative_failure_probability(probability, label_horizon, start)
            cost_by_day.append(round(planned.total_eur + p_fail * premium, 2))

        # Deferral beyond the horizon: risk accrues for the whole window and the work still
        # has to happen afterwards. Charged at the horizon's probability, not at 100%, so
        # the scheduler is not bullied into servicing everything by an inflated penalty.
        jobs.append(
            {
                "job_id": f"{row['vehicle_id']}:{component.id}",
                "vehicle_id": row["vehicle_id"],
                "component_id": component.id,
                "component_name": component.name,
                "depot_id": vehicle["depot_id"],
                "vehicle_class": vehicle["vehicle_class"],
                "contract_id": vehicle["contract_id"],
                "registration": vehicle["registration"],
                "failure_probability": round(probability, 6),
                # Two probabilities, and they are not interchangeable. The first is over the
                # model's label horizon and is what a person should be shown; the second is
                # over the planning horizon and is what the money is computed from. Using
                # one where the other belongs understates every euro on the page.
                "horizon_probability": round(p_horizon, 6),
                "horizon_days": label_horizon,
                "bay_hours": component.bay_hours,
                "bay_days": max(1, round(component.bay_hours / 10.0 + 0.4)),
                "immobilising": component.immobilising,
                "safety_critical": component.safety_critical,
                "deferrable": component.deferrable,
                "planned_cost_eur": round(planned.total_eur, 2),
                "unplanned_cost_eur": round(unplanned.total_eur, 2),
                "premium_eur": round(premium, 2),
                "cost_by_start_day": cost_by_day,
                "cost_if_never_eur": round(planned.total_eur + p_horizon * premium, 2),
                "expected_saving_eur": round(p_horizon * premium, 2),
                "staleness_days": int(row["staleness_days"]),
                "days_since_service": int(row["days_since_service"]),
                "sensor_level": round(float(row["sensor_level"]), 3),
                "sensor_slope_7d": round(float(row["sensor_slope_7d"]), 4),
                "model_passes_gate": bool(row["model_passes_gate"]),
                "unplanned_breakdown": unplanned.as_dict(),
                "planned_breakdown": planned.as_dict(),
            }
        )
    return jobs
