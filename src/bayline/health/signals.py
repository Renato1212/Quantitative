"""Daily component health signals, computed in SQL over the hourly telemetry.

One row per (vehicle, component, day). Each carries what a decision on that day could
actually have known — nothing later, and nothing from before the component was last
replaced.

Three properties this module exists to guarantee:

**No lookahead.** Every window is trailing and closed at the day in question. A rolling
mean that includes the current *and following* days is the single easiest way to build a
maintenance model that looks excellent and is worthless, because failures are preceded by
readings the model would not have had.

**Cycle-aware exposure.** Damage resets when a component is replaced, so exposure is
measured from the last replacement, not from the vehicle's build date. Without this the
model learns vehicle age, which is exactly the mileage-schedule logic Bayline is meant to
beat.

**Trend over level.** The sensors saturate and carry a per-vehicle bias, so the absolute
reading is a weak signal and the *slope* is a strong one. The features are differences and
ratios of trailing windows for that reason.
"""

from __future__ import annotations

import pyarrow as pa

from bayline.config import Config
from bayline.store import Warehouse

SHORT_WINDOW = 7
LONG_WINDOW = 28

FEATURE_COLUMNS = (
    "sensor_level",
    "sensor_slope_7d",
    "sensor_ratio_7_28",
    "sensor_volatility_7d",
    "exposure_since_service",
    "exposure_rate_7d",
    "duty_urban_share",
    "load_factor_7d",
    "ambient_min_7d",
    "cold_days_28d",
    "km_since_service",
    "days_since_service",
)


def build(cfg: Config, warehouse: Warehouse) -> pa.Table:
    """Compute the health table for every component and concatenate.

    One query per component rather than one query over a pivoted table: each component
    reads a different sensor column and accumulates a different damage driver, and a single
    query trying to do all seven ends up as an unreadable CASE expression that is slower
    anyway.
    """
    parts = [_component_health(cfg, warehouse, component_id=c.id) for c in cfg.components]
    return pa.concat_tables(parts)


def _driver_expression(driver: str) -> str:
    """SQL for one day's worth of a damage driver, from the hourly rows.

    Mirrors ``simulate.fleet._driver_units``. The duplication is deliberate: the simulator
    generates from latent state, this reads observable telemetry, and tying them to one
    implementation would let the simulator's ground truth leak into the features.
    """
    if driver == "engine_hours":
        return "sum(engine_hours_delta)"
    if driver == "urban_hours":
        return "sum(engine_hours_delta * urban_share)"
    if driver == "brake_energy":
        return (
            "sum(km * (0.35 + 1.9 * urban_share) * (0.6 + 0.8 * load_factor) "
            "* CASE vehicle_class WHEN 'artic' THEN 1.0 WHEN 'rigid' THEN 0.72 "
            "ELSE 0.38 END) / 100.0"
        )
    if driver == "clutch_engagements":
        return "sum(km * (4.0 * urban_share + 0.25))"
    if driver == "thermal_stress":
        return "sum(engine_hours_delta * (0.6 + 0.9 * urban_share))"
    if driver == "cold_starts":
        return "count(*) / 14.0 * (1.2 + 5.5 * max(urban_share))"
    raise ValueError(f"unknown damage driver {driver!r}")


def _component_health(cfg: Config, warehouse: Warehouse, *, component_id: str) -> pa.Table:
    component = cfg.component(component_id)
    driver_sql = _driver_expression(component.driver)
    sensor = component.sensor

    query = f"""
    WITH daily AS (
        SELECT
            vehicle_id,
            day_index,
            avg({sensor})                       AS sensor_level,
            stddev_samp({sensor})               AS sensor_intraday_sd,
            sum(km)                             AS km,
            {driver_sql}                        AS driver_units,
            avg(load_factor)                    AS load_factor,
            min(ambient_c)                      AS ambient_min,
            max(urban_share)                    AS urban_share
        FROM {{telemetry}} t
        -- Vehicle class is needed by the brake-energy driver: braking scales with mass, and
        -- a feature that ignores it under-reads every artic in the fleet.
        JOIN {{vehicles}} v USING (vehicle_id)
        GROUP BY vehicle_id, day_index
    ),
    -- Every service or failure for this component resets its damage. `cycle` numbers the
    -- periods between resets so exposure can be measured within one.
    resets AS (
        SELECT vehicle_id, day_index
        FROM {{events}}
        WHERE component_id = ?
    ),
    marked AS (
        SELECT
            d.*,
            -- Trailing count of resets strictly before today: a reset dated today has not
            -- yet reduced the wear that today's decision faces.
            (SELECT count(*) FROM resets r
             WHERE r.vehicle_id = d.vehicle_id AND r.day_index < d.day_index) AS cycle
        FROM daily d
    ),
    windowed AS (
        SELECT
            vehicle_id,
            day_index,
            cycle,
            sensor_level,
            sensor_intraday_sd,
            km,
            driver_units,
            load_factor,
            ambient_min,
            urban_share,
            -- All windows are trailing and inclusive of today only.
            avg(sensor_level) OVER w7                                   AS sensor_mean_7,
            avg(sensor_level) OVER w28                                  AS sensor_mean_28,
            stddev_samp(sensor_level) OVER w7                           AS sensor_sd_7,
            first_value(sensor_level) OVER w7                           AS sensor_first_7,
            sum(driver_units) OVER cyc                                  AS exposure_since_service,
            sum(driver_units) OVER w7                                   AS exposure_7,
            sum(km) OVER cyc                                            AS km_since_service,
            avg(load_factor) OVER w7                                    AS load_factor_7d,
            min(ambient_min) OVER w7                                    AS ambient_min_7d,
            sum(CASE WHEN ambient_min < 4 THEN 1 ELSE 0 END) OVER w28   AS cold_days_28d,
            count(*) OVER cyc                                           AS days_since_service
        FROM marked
        WINDOW
            w7  AS (PARTITION BY vehicle_id, cycle ORDER BY day_index
                    ROWS BETWEEN {SHORT_WINDOW - 1} PRECEDING AND CURRENT ROW),
            w28 AS (PARTITION BY vehicle_id, cycle ORDER BY day_index
                    ROWS BETWEEN {LONG_WINDOW - 1} PRECEDING AND CURRENT ROW),
            cyc AS (PARTITION BY vehicle_id, cycle ORDER BY day_index
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
    SELECT
        vehicle_id,
        ? AS component_id,
        day_index,
        cycle,
        sensor_level,
        -- Slope across the short window, per day. The strongest single feature, because
        -- it is insensitive to the per-vehicle sensor bias that defeats a level threshold.
        (sensor_level - sensor_first_7) / {SHORT_WINDOW - 1}.0        AS sensor_slope_7d,
        CASE WHEN sensor_mean_28 > 0
             THEN sensor_mean_7 / sensor_mean_28 ELSE 1.0 END         AS sensor_ratio_7_28,
        coalesce(sensor_sd_7, sensor_intraday_sd, 0.0)                AS sensor_volatility_7d,
        exposure_since_service,
        exposure_7 / {SHORT_WINDOW}.0                                 AS exposure_rate_7d,
        urban_share                                                   AS duty_urban_share,
        load_factor_7d,
        ambient_min_7d,
        cold_days_28d,
        km_since_service,
        days_since_service
    FROM windowed
    -- The first four weeks of a cycle have no trend to read; a slope from three points is
    -- noise wearing a feature's clothes.
    WHERE days_since_service >= {LONG_WINDOW}
    ORDER BY vehicle_id, day_index
    """
    return warehouse.arrow(query, [component_id, component_id])


def summarise(warehouse: Warehouse) -> dict:
    """Row counts and coverage, for the pipeline log."""
    return warehouse.rows(
        """
        SELECT component_id,
               count(*)                       AS rows,
               count(DISTINCT vehicle_id)     AS vehicles,
               round(avg(days_since_service), 1) AS mean_days_in_cycle
        FROM {health}
        GROUP BY component_id
        ORDER BY component_id
        """
    )
