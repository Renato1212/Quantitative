"""Fixtures.

The expensive fixture builds a real, small warehouse — simulate and health only — because
the properties worth testing (no lookahead, cycle-aware exposure) are properties of SQL over
actual data and cannot be checked against a mock. It is session-scoped and takes a few
seconds; everything else in the suite runs on synthetic dictionaries in milliseconds.
"""

from __future__ import annotations

import pytest

from bayline import config as config_module
from bayline.health import signals
from bayline.simulate import fleet as fleet_module
from bayline.simulate import telemetry as telemetry_module
from bayline.store import Warehouse

TINY_VEHICLES = 24
TINY_DAYS = 260


@pytest.fixture(scope="session")
def cfg():
    return config_module.load()


@pytest.fixture(scope="session")
def tiny_cfg(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("warehouse")
    return cfg.with_overrides(
        **{
            "fleet.vehicles": TINY_VEHICLES,
            "history.days": TINY_DAYS,
            "storage.warehouse": str(root),
        }
    )


@pytest.fixture(scope="session")
def tiny_warehouse(tiny_cfg):
    """simulate + health over a small fleet, built once for the whole session."""
    warehouse = Warehouse(tiny_cfg)
    fleet = fleet_module.build_fleet(tiny_cfg)
    warehouse.write_rows("vehicles", fleet_module.fleet_rows(fleet, tiny_cfg))

    columns, failures = telemetry_module.simulate(tiny_cfg, fleet)
    warehouse.write_columns("telemetry", columns)

    services = telemetry_module.planned_services(tiny_cfg, fleet, failures)
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

    warehouse.write_arrow("health", signals.build(tiny_cfg, warehouse))
    yield warehouse
    warehouse.close()


def make_job(**overrides) -> dict:
    """A minimal priced job, shaped exactly as ``build_job_costs`` emits one.

    Defaults are deliberately boring: one bay-day, one depot, linear cost in the start day.
    Tests override only the field under examination, so a failure names its own cause.
    """
    horizon = overrides.pop("horizon", 28)
    saving = overrides.pop("saving", 1000.0)
    planned = overrides.pop("planned", 500.0)
    job = {
        "job_id": "BL-0001:turbo",
        "vehicle_id": "BL-0001",
        "component_id": "turbo",
        "depot_id": "DEP-BCN",
        "bay_days": 1,
        "safety_critical": False,
        "failure_probability": 0.05,
        "horizon_probability": 0.07,
        "days_since_service": 100,
        # Cost rises linearly with the start day and tops out at cost_if_never.
        "cost_by_start_day": [planned + saving * d / horizon for d in range(horizon)],
        "cost_if_never_eur": planned + saving,
    }
    job.update(overrides)
    return job
