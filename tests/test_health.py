"""No lookahead, tested by construction rather than by inspection.

Reading the SQL and confirming every window says ``ROWS BETWEEN n PRECEDING AND CURRENT ROW``
is not a test — it is the same person who wrote the bug checking for it. The test here
rebuilds the health table from telemetry physically truncated at day D and asserts the row
for day D is byte-identical to the row the full-history build produced. If any window, join,
subquery or aggregate reaches forward, the two disagree.

This is the check that would have caught the leak, and it is the reason the suite pays for a
real warehouse.
"""

from __future__ import annotations

import math

import pytest
import pyarrow as pa

from bayline.health import signals
from bayline.store import Warehouse

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cutoff_day(tiny_warehouse) -> int:
    """A working day around two thirds through the history.

    Chosen rather than hardcoded: the fleet does not move at weekends, and truncating on a
    Saturday compares two quiet days and proves less than it appears to.
    """
    return int(
        tiny_warehouse.scalar(
            """
            SELECT max(day_index) FROM (
                SELECT day_index, sum(km) AS km FROM {telemetry}
                WHERE day_index <= 200 GROUP BY day_index
            ) WHERE km > 0
            """
        )
    )


@pytest.fixture(scope="module")
def truncated(tiny_cfg, tiny_warehouse, cutoff_day, tmp_path_factory):
    """A second warehouse holding only telemetry and events at or before the cutoff."""
    root = tmp_path_factory.mktemp("truncated")
    other = Warehouse(tiny_cfg, root=root)
    other.write_arrow(
        "vehicles", tiny_warehouse.arrow("SELECT * FROM {vehicles}")
    )
    other.write_arrow(
        "telemetry",
        tiny_warehouse.arrow("SELECT * FROM {telemetry} WHERE day_index <= ?", [cutoff_day]),
    )
    other.write_arrow(
        "events",
        tiny_warehouse.arrow("SELECT * FROM {events} WHERE day_index <= ?", [cutoff_day]),
    )
    yield other
    other.close()


def _rows_on_cutoff(warehouse: Warehouse, cutoff_day: int) -> dict[tuple[str, str], dict]:
    rows = warehouse.rows(
        "SELECT * FROM {health} WHERE day_index = ? ORDER BY vehicle_id, component_id",
        [cutoff_day],
    )
    return {(r["vehicle_id"], r["component_id"]): r for r in rows}


def test_features_do_not_change_when_the_future_is_removed(
    tiny_cfg, tiny_warehouse, truncated, cutoff_day
):
    truncated.write_arrow("health", signals.build(tiny_cfg, truncated))

    full = _rows_on_cutoff(tiny_warehouse, cutoff_day)
    partial = _rows_on_cutoff(truncated, cutoff_day)

    assert partial, "the truncated build produced no rows on the cutoff day"
    assert set(partial) <= set(full)

    for key, expected in full.items():
        if key not in partial:
            continue
        actual = partial[key]
        for column, value in expected.items():
            other = actual[column]
            if isinstance(value, float) and isinstance(other, float):
                assert value == pytest.approx(other, rel=1e-9, abs=1e-12), (
                    f"{key} column {column!r} changed when the future was removed: "
                    f"{value} with full history, {other} without — this feature looks ahead"
                )
            else:
                assert value == other, f"{key} column {column!r} looks ahead"


def test_a_deliberately_leaky_feature_is_caught(tiny_warehouse, truncated, cutoff_day):
    """The canary. If the comparison above cannot detect a one-day peek, it proves nothing.

    A leading window is the most common way this happens in practice — a ``rolling()`` with
    no shift, or a library indicator that centres its window — and it is invisible in the
    output until the model is unaccountably good.
    """
    # The window is computed in a subquery and filtered outside it. Filtering first would
    # leave one row per partition and the peek would have nothing to reach forward to —
    # which is how the first version of this canary passed while proving nothing.
    leaky = """
    SELECT vehicle_id, day_index, peeking_km FROM (
        SELECT vehicle_id, day_index,
               avg(km) OVER (PARTITION BY vehicle_id ORDER BY day_index
                             ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS peeking_km
        FROM (SELECT vehicle_id, day_index, sum(km) AS km FROM {telemetry}
              GROUP BY vehicle_id, day_index)
    )
    WHERE day_index = ?
    ORDER BY vehicle_id
    """
    full = tiny_warehouse.rows(leaky, [cutoff_day])
    partial = truncated.rows(leaky, [cutoff_day])
    differences = sum(
        1
        for a, b in zip(full, partial)
        if not math.isclose(a["peeking_km"], b["peeking_km"], rel_tol=1e-9)
    )
    assert differences > 0, (
        "the truncation harness failed to detect a one-day-ahead window, so its verdict on "
        "the production features is worthless"
    )


def test_exposure_resets_when_a_component_is_replaced(tiny_cfg, tiny_warehouse):
    """Without a cycle reset the model learns vehicle age — the mileage-schedule logic
    Bayline exists to beat, laundered through a gradient booster."""
    row = tiny_warehouse.rows(
        """
        SELECT vehicle_id, component_id, cycle,
               min(days_since_service) AS first_day,
               max(exposure_since_service) AS peak
        FROM {health}
        WHERE cycle > 0
        GROUP BY vehicle_id, component_id, cycle
        LIMIT 5
        """
    )
    assert row, "no component was replaced in the tiny fleet — widen the fixture"
    for entry in row:
        assert entry["first_day"] == signals.LONG_WINDOW
        assert entry["peak"] > 0


def test_early_cycle_rows_are_withheld(tiny_warehouse):
    """A slope from three points is noise wearing a feature's clothes."""
    minimum = tiny_warehouse.scalar("SELECT min(days_since_service) FROM {health}")
    assert minimum >= signals.LONG_WINDOW


def test_every_declared_feature_column_exists_and_is_finite(tiny_warehouse):
    table = tiny_warehouse.arrow("SELECT * FROM {health} USING SAMPLE 5000 ROWS")
    for column in signals.FEATURE_COLUMNS:
        assert column in table.column_names, f"{column} is declared but not produced"
        values = table[column].to_pylist()
        assert all(v is not None and math.isfinite(float(v)) for v in values), column


def test_health_is_one_row_per_vehicle_component_day(tiny_warehouse):
    duplicates = tiny_warehouse.scalar(
        """
        SELECT count(*) FROM (
            SELECT vehicle_id, component_id, day_index
            FROM {health} GROUP BY 1, 2, 3 HAVING count(*) > 1
        )
        """
    )
    assert duplicates == 0


def test_the_simulator_keeps_its_latent_state_out_of_the_warehouse(tiny_warehouse):
    """``life_multiplier`` and ``damage`` are ground truth. If either reached a feature the
    model would be reading the answer key."""
    columns = set(tiny_warehouse.arrow("SELECT * FROM {vehicles} LIMIT 1").column_names)
    columns |= set(tiny_warehouse.arrow("SELECT * FROM {telemetry} LIMIT 1").column_names)
    forbidden = {"life_multiplier", "damage", "sensor_bias", "true_hazard"}
    assert not (columns & forbidden), sorted(columns & forbidden)


def test_the_health_table_is_deterministic(tiny_cfg, tiny_warehouse):
    """Same warehouse, same SQL, byte-identical output. Nothing here may depend on scan
    order or on a hash-join's whim."""
    first = signals.build(tiny_cfg, tiny_warehouse)
    second = signals.build(tiny_cfg, tiny_warehouse)
    assert first.equals(second)
    assert isinstance(first, pa.Table)
