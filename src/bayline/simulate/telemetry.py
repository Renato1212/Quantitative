"""Hourly telemetry and the failure events it precedes.

Roughly 2–4 million rows for a 300-vehicle fleet over 18 months, which is the scale a real
mid-size fleet generates and enough that the aggregation has to be done in the database
rather than in Python.

The generative story, which the risk model has to recover without being told any of it:

1. Each component accumulates damage at a rate set by the vehicle's duty cycle, modulated
   by weather and load on the day.
2. Damage becomes a **hazard** through a Weibull with the component's shape parameter, so
   the failure rate accelerates as wear-out approaches rather than being a step at a
   threshold.
3. Sensors observe damage through a saturating transform, a per-vehicle bias and heavy
   noise. The mapping is monotone but not linear, and a single reading is nearly useless —
   the signal lives in the *trend*, which is why the feature layer differences it.
4. On failure the component is replaced and its damage resets, so a vehicle's history holds
   several cycles and the model cannot simply learn "old truck bad".

Two deliberate traps. Sensor bias means an absolute threshold on any single sensor performs
badly, and the weather term means a naive model will over-attribute winter battery failures
to wear. Both are things real fleets get wrong.
"""

from __future__ import annotations

import numpy as np

from bayline.config import Config
from bayline.simulate.fleet import Vehicle, driver_rate_per_day

# Sensor scales, chosen so each reading looks like the instrument it imitates.
SENSOR_SCALE = {
    "boost_deviation_pct": (0.5, 14.0),
    "rail_pressure_cv": (0.8, 7.5),
    "soot_load_index": (5.0, 98.0),
    "pad_wear_index": (2.0, 96.0),
    "clutch_slip_index": (0.4, 22.0),
    "coolant_excursion_index": (1.0, 34.0),
    "crank_voltage_sag": (0.15, 2.4),
}


def _seasonal_temperature(day_of_year: np.ndarray) -> np.ndarray:
    """Iberian ambient temperature. Cold snaps are what kill batteries."""
    return 17.5 - 9.5 * np.cos(2.0 * np.pi * (day_of_year - 15) / 365.25)


def simulate(cfg: Config, fleet: list[Vehicle]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Generate hourly telemetry and the failure log.

    Returns column arrays for the telemetry fact table and a list of failure events. Column
    arrays rather than row dicts because at a few million rows the difference between the
    two is a minute of wall clock and several gigabytes of peak memory.
    """
    rng = np.random.default_rng(cfg.seed + 977)
    days = int(cfg.get("history.days"))
    irreducible = float(cfg.get("risk.irreducible_failure_share", 0.0))
    hours_per_day = int(cfg.get("history.operating_hours_per_day"))
    components = cfg.components
    end = cfg.end_date
    start_ordinal = end.toordinal() - days + 1

    per_vehicle_rows = days * hours_per_day
    total = per_vehicle_rows * len(fleet)

    columns = {
        "vehicle_id": np.empty(total, dtype=object),
        "day_index": np.empty(total, dtype=np.int32),
        "hour": np.empty(total, dtype=np.int8),
        "km": np.empty(total, dtype=np.float32),
        "engine_hours_delta": np.empty(total, dtype=np.float32),
        "load_factor": np.empty(total, dtype=np.float32),
        "ambient_c": np.empty(total, dtype=np.float32),
        "urban_share": np.empty(total, dtype=np.float32),
        **{sensor: np.empty(total, dtype=np.float32) for sensor in SENSOR_SCALE},
    }
    failures: list[dict] = []

    day_index = np.arange(days, dtype=np.int32)
    day_of_year = np.array(
        [(start_ordinal + int(d)) % 365 for d in day_index], dtype=np.float64
    )
    ambient_daily = _seasonal_temperature(day_of_year)
    # Weekends: most vehicles work a five-day week, so damage does not accrue uniformly.
    weekday = np.array([(start_ordinal + int(d)) % 7 for d in day_index])
    working = weekday < 5

    cursor = 0
    for vehicle in fleet:
        rows = per_vehicle_rows
        span = slice(cursor, cursor + rows)
        cursor += rows

        # Daily multipliers: traffic, load, weather. Autocorrelated, because a bad week is
        # a bad week — independent daily noise would wash out of every trend feature.
        shocks = rng.normal(0.0, 1.0, size=days)
        smooth = np.convolve(shocks, np.ones(5) / 5.0, mode="same")
        intensity = np.clip(1.0 + 0.17 * smooth, 0.45, 1.75) * working
        load = np.clip(0.55 + 0.30 * rng.normal(0.0, 1.0, size=days) + 0.15 * smooth, 0.1, 1.0)

        # Damage accrues per day, then the hazard is evaluated once per day.
        damage = dict(vehicle.damage)
        daily_rate = {c.id: driver_rate_per_day(c.driver, vehicle) for c in components}
        sensor_daily = {c.sensor: np.zeros(days, dtype=np.float32) for c in components}

        for d in range(days):
            factor = float(intensity[d])
            cold = max(0.0, 6.0 - float(ambient_daily[d])) / 6.0
            for component in components:
                effective_life = component.characteristic_life * vehicle.life_multiplier[
                    component.id
                ]
                # Weather modulates two drivers and nothing else: cold batteries, hot
                # cooling systems. Applying a global weather term would be a fiction.
                stress = factor
                if component.id == "battery":
                    stress *= 1.0 + 1.6 * cold
                elif component.id == "cooling":
                    stress *= 1.0 + 0.9 * max(0.0, float(ambient_daily[d]) - 24.0) / 12.0
                elif component.id == "brakes":
                    stress *= 0.6 + 0.8 * float(load[d])

                damage[component.id] += daily_rate[component.id] * stress

                # Weibull discrete-time hazard over one day at the current damage level.
                ratio = damage[component.id] / effective_life
                shape = component.weibull_shape
                increment = daily_rate[component.id] * stress / effective_life
                hazard = shape * (ratio ** (shape - 1.0)) * increment if ratio > 0 else 0.0
                hazard = float(np.clip(hazard, 0.0, 0.9))

                sensor_daily[component.sensor][d] = _sensor_reading(
                    rng, component.sensor, ratio, vehicle.sensor_bias[component.id]
                )

                # A slice of failures that wear does not cause. Spread uniformly over the
                # component's life so no feature can see them coming — the ceiling on how
                # good any model gets, here and on a real feed.
                if irreducible > 0:
                    baseline = irreducible * increment / max(1.0 - irreducible, 1e-6)
                    hazard = float(np.clip(hazard + baseline, 0.0, 0.9))

                if working[d] and rng.random() < hazard:
                    failures.append(
                        {
                            "vehicle_id": vehicle.vehicle_id,
                            "component_id": component.id,
                            "day_index": int(d),
                            "damage_ratio_at_failure": round(float(ratio), 4),
                            "immobilising": bool(component.immobilising),
                        }
                    )
                    damage[component.id] = 0.0  # replaced

        # Expand daily quantities to hourly rows.
        hours = np.tile(np.arange(hours_per_day, dtype=np.int8), days)
        day_col = np.repeat(day_index, hours_per_day)
        intensity_h = np.repeat(intensity, hours_per_day).astype(np.float32)
        mean_speed = 62.0 - 34.0 * vehicle.urban_fraction

        columns["vehicle_id"][span] = vehicle.vehicle_id
        columns["day_index"][span] = day_col
        columns["hour"][span] = hours
        columns["km"][span] = (
            (vehicle.km_per_day / hours_per_day) * intensity_h
            * rng.normal(1.0, 0.10, size=rows).astype(np.float32)
        ).clip(0.0)
        columns["engine_hours_delta"][span] = (columns["km"][span] / mean_speed).astype(
            np.float32
        )
        columns["load_factor"][span] = np.repeat(load, hours_per_day).astype(np.float32)
        columns["ambient_c"][span] = (
            np.repeat(ambient_daily, hours_per_day)
            + rng.normal(0.0, 2.2, size=rows)
            # Diurnal swing, so an hourly table is actually hourly.
            + 4.5 * np.sin(2.0 * np.pi * (hours.astype(np.float64) - 4) / 24.0)
        ).astype(np.float32)
        columns["urban_share"][span] = np.float32(vehicle.urban_fraction)

        for sensor, daily in sensor_daily.items():
            lo, hi = SENSOR_SCALE[sensor]
            hourly = np.repeat(daily, hours_per_day)
            # Hour-to-hour instrument noise on top of the daily level.
            noise = rng.normal(0.0, 0.035 * (hi - lo), size=rows)
            columns[sensor][span] = np.clip(hourly + noise, lo, hi).astype(np.float32)

    return columns, failures


def _sensor_reading(
    rng: np.random.Generator, sensor: str, damage_ratio: float, bias: float
) -> float:
    """Observe damage through a saturating, biased, noisy instrument.

    Monotone in damage, so the information is there. Saturating, so the top of the range
    cannot separate a component at 80% of life from one at 110% — which is exactly why a
    fixed sensor threshold is a poor maintenance trigger and why the model is given the
    trend rather than the level.
    """
    lo, hi = SENSOR_SCALE[sensor]
    saturated = damage_ratio / (0.55 + damage_ratio)
    level = lo + (hi - lo) * saturated * (1.0 + bias)
    return float(np.clip(level + rng.normal(0.0, 0.06 * (hi - lo)), lo, hi))


def planned_services(cfg: Config, fleet: list[Vehicle], failures: list[dict]) -> list[dict]:
    """Historical planned services, so the history is not all breakdowns.

    A fleet doing nothing but reactive repair is not a realistic baseline, and a model
    trained only on failures would never see a censored component. Roughly six in ten
    components get replaced on a mileage schedule before they fail, which is the incumbent
    policy Bayline is measured against.
    """
    rng = np.random.default_rng(cfg.seed + 4231)
    days = int(cfg.get("history.days"))
    failed = {(f["vehicle_id"], f["component_id"], f["day_index"]) for f in failures}
    out: list[dict] = []
    for vehicle in fleet:
        for component in cfg.components:
            interval_days = component.characteristic_life / max(
                driver_rate_per_day(component.driver, vehicle), 1e-6
            )
            # The incumbent policy: service at a fixed fraction of nominal life, ignoring
            # the per-vehicle spread entirely. Sometimes far too early, sometimes too late.
            policy_days = interval_days * 0.78
            day = float(rng.uniform(0, min(policy_days, days)))
            while day < days:
                key = (vehicle.vehicle_id, component.id, int(day))
                if key not in failed:
                    out.append(
                        {
                            "vehicle_id": vehicle.vehicle_id,
                            "component_id": component.id,
                            "day_index": int(day),
                            "kind": "planned",
                        }
                    )
                day += policy_days * float(rng.normal(1.0, 0.09))
    return out
