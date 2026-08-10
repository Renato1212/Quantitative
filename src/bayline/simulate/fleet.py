"""The fleet: vehicles, depots, contracts, and each vehicle's hidden wear state.

**This is a simulated fleet.** Bayline is an engine for pricing maintenance decisions, and
an engine needs a fleet to run against. Everything here is generated from a seed, and the
UI says so on every screen. What is *not* simulated is the decision logic: the hazard
model, the cost arithmetic and the scheduler are the product, and they work identically on
a real telematics feed — `src/bayline/simulate/` is the only module a customer replaces.

The generator is built to be *hard*, not flattering. Each vehicle carries a latent damage
threshold drawn per component, so two trucks with identical mileage fail at different
times; sensors observe damage through noise and a class-specific bias; and duty cycle
decides which components wear at all. A model that scores well here has found a real
signal, because the noise floor was put there on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayline.config import Config


@dataclass
class Vehicle:
    """One vehicle, plus the wear state a real fleet cannot observe directly."""

    vehicle_id: str
    registration: str
    vehicle_class: str
    depot_id: str
    contract_id: str
    model_year: int
    odometer_km: float
    engine_hours: float
    urban_fraction: float
    km_per_day: float
    # Per-component multiplier on characteristic life. Below 1.0 is a bad example of the
    # build; above is a good one. This is the term that makes odometer-based servicing
    # wrong, and the reason a model that reads sensors can beat a mileage schedule.
    life_multiplier: dict[str, float] = field(default_factory=dict)
    # Accumulated damage in driver units, carried forward from before the study window.
    damage: dict[str, float] = field(default_factory=dict)
    # Sensor bias per component: a permanently optimistic or pessimistic sensor.
    sensor_bias: dict[str, float] = field(default_factory=dict)


def _weighted_choice(rng: np.random.Generator, entries: list[dict], n: int) -> np.ndarray:
    weights = np.array([entry["share"] for entry in entries], dtype=float)
    return rng.choice(len(entries), size=n, p=weights / weights.sum())


def build_fleet(cfg: Config) -> list[Vehicle]:
    """Generate the fleet deterministically from the configured seed."""
    rng = np.random.default_rng(cfg.seed)
    count = int(cfg.get("fleet.vehicles"))
    classes = cfg.vehicle_classes
    depots = cfg.depots
    contracts = cfg.contracts
    components = cfg.components

    class_idx = _weighted_choice(rng, classes, count)
    contract_idx = _weighted_choice(rng, contracts, count)
    # Depots are sized by their bay count: a six-bay depot carries more vehicles.
    bay_weights = np.array([d["bays"] for d in depots], dtype=float)
    depot_idx = rng.choice(len(depots), size=count, p=bay_weights / bay_weights.sum())

    fleet: list[Vehicle] = []
    for i in range(count):
        klass = classes[class_idx[i]]
        # Age drives starting wear. A five-year-old tractor is most of the way through
        # its second turbo, and the plan has to know that on day one.
        age_years = float(np.clip(rng.gamma(shape=2.1, scale=1.9), 0.2, 11.0))
        model_year = cfg.end_date.year - int(age_years)
        km_per_day = float(klass["km_per_day"] * rng.normal(1.0, 0.13))
        km_per_day = max(km_per_day, 40.0)
        odometer = km_per_day * 250.0 * age_years * rng.normal(1.0, 0.08)
        # Average road speed by duty cycle: urban work burns hours, not kilometres.
        mean_speed = 62.0 - 34.0 * float(klass["urban_fraction"])
        engine_hours = odometer / mean_speed

        vehicle = Vehicle(
            vehicle_id=f"BL-{i + 1:04d}",
            registration=_registration(rng),
            vehicle_class=klass["id"],
            depot_id=depots[depot_idx[i]]["id"],
            contract_id=contracts[contract_idx[i]]["id"],
            model_year=model_year,
            odometer_km=float(odometer),
            engine_hours=float(engine_hours),
            urban_fraction=float(
                np.clip(klass["urban_fraction"] * rng.normal(1.0, 0.18), 0.03, 0.97)
            ),
            km_per_day=km_per_day,
        )

        for component in components:
            # Lognormal build quality: a long right tail of unusually good examples and a
            # floor of bad ones. This is the spread a mileage-based schedule cannot see.
            vehicle.life_multiplier[component.id] = float(
                np.clip(rng.lognormal(mean=0.0, sigma=0.26), 0.45, 2.1)
            )
            # Damage carried in from before the window, as a fraction of the vehicle's own
            # effective life. Cycled components restart, so this is taken modulo one life.
            effective_life = component.characteristic_life * vehicle.life_multiplier[component.id]
            elapsed = _driver_units(component.driver, vehicle, days=age_years * 250.0)
            vehicle.damage[component.id] = float(elapsed % effective_life)
            vehicle.sensor_bias[component.id] = float(rng.normal(0.0, 0.075))

        fleet.append(vehicle)
    return fleet


def _registration(rng: np.random.Generator) -> str:
    letters = "BCDFGHJKLMNPRSTVWXYZ"
    return (
        f"{rng.integers(1000, 9999)} "
        + "".join(letters[i] for i in rng.integers(0, len(letters), size=3))
    )


def _driver_units(driver: str, vehicle: Vehicle, *, days: float) -> float:
    """How much of a component's damage driver accumulates over a number of days.

    Each component wears on its own clock. Expressing them in shared units is the mistake
    that makes a fleet service everything on odometer: a van that never leaves the city
    destroys clutches at ten times the rate its mileage suggests and turbos at a tenth.
    """
    mean_speed = 62.0 - 34.0 * vehicle.urban_fraction
    hours_per_day = vehicle.km_per_day / mean_speed
    urban_hours = hours_per_day * vehicle.urban_fraction

    if driver == "engine_hours":
        return hours_per_day * days
    if driver == "urban_hours":
        return urban_hours * days
    if driver == "brake_energy":
        # Braking scales with stop density and with mass, which the class encodes.
        mass_factor = {"artic": 1.0, "rigid": 0.72, "van": 0.38}
        return (
            vehicle.km_per_day
            * days
            * (0.35 + 1.9 * vehicle.urban_fraction)
            * mass_factor.get(vehicle.vehicle_class, 0.7)
            / 100.0
        )
    if driver == "clutch_engagements":
        # Roughly one engagement per 250 m of urban driving, far fewer on trunk roads.
        return vehicle.km_per_day * days * (4.0 * vehicle.urban_fraction + 0.25)
    if driver == "thermal_stress":
        # Heat load rises with load factor and with hills; urban idling is worse still.
        return hours_per_day * days * (0.6 + 0.9 * vehicle.urban_fraction)
    if driver == "cold_starts":
        # Starts per day, weighted for multi-drop work.
        return days * (1.2 + 5.5 * vehicle.urban_fraction)
    raise ValueError(f"unknown damage driver {driver!r}")


def driver_rate_per_day(driver: str, vehicle: Vehicle) -> float:
    """Damage-driver units accumulated per operating day. Used by the risk features."""
    return _driver_units(driver, vehicle, days=1.0)


def fleet_rows(fleet: list[Vehicle], cfg: Config) -> list[dict]:
    """The vehicle dimension table. Latent wear state is deliberately excluded.

    ``life_multiplier`` and ``damage`` are ground truth the simulator knows and a real
    fleet does not. Writing them into the warehouse would let a model read the answer, so
    they stay in memory and never reach the store.
    """
    revenue = cfg.get("economics.revenue_per_day_eur")
    penalties = {entry["id"]: entry["penalty_eur_per_missed_day"] for entry in cfg.contracts}
    return [
        {
            "vehicle_id": v.vehicle_id,
            "registration": v.registration,
            "vehicle_class": v.vehicle_class,
            "depot_id": v.depot_id,
            "contract_id": v.contract_id,
            "model_year": v.model_year,
            "odometer_km": round(v.odometer_km, 1),
            "engine_hours": round(v.engine_hours, 1),
            "urban_fraction": round(v.urban_fraction, 4),
            "km_per_day": round(v.km_per_day, 2),
            "revenue_per_day_eur": float(revenue[v.vehicle_class]),
            "penalty_eur_per_missed_day": float(penalties[v.contract_id]),
        }
        for v in fleet
    ]
