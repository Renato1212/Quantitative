"""Triple-barrier outcomes and excursion metrics, against hand-constructed paths.

Excursion arithmetic is the kind of code that looks right and is off by a sign. These
tests build tick paths whose answers are known by inspection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.labels.atr import atr_table, session_true_range
from src.labels.triple_barrier import barrier_unit_scale, label_events

START = datetime(2024, 3, 20, 15, 0, tzinfo=timezone.utc)
SESSION = START.date()


def _ticks(prices: list[float], step_seconds: int = 60) -> pl.DataFrame:
    stamps = [START + timedelta(seconds=step_seconds * i) for i in range(len(prices))]
    return pl.DataFrame(
        {
            "ts": pl.Series(stamps, dtype=pl.Datetime("us", "UTC")),
            "price": pl.Series(prices, dtype=pl.Float64),
            "size": pl.Series([1] * len(prices), dtype=pl.Int64),
            "aggressor": pl.Series([1] * len(prices), dtype=pl.Int8),
            "contract": ["ESH24"] * len(prices),
            "session_date": pl.Series([SESSION] * len(prices), dtype=pl.Date),
        }
    )


def _event(side: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trigger_family": ["test"],
            "t0": pl.Series([START - timedelta(seconds=1)], dtype=pl.Datetime("us", "UTC")),
            "side": pl.Series([side], dtype=pl.Int8),
            "session_date": pl.Series([SESSION], dtype=pl.Date),
        }
    )


def _atr(value: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_date": pl.Series([SESSION], dtype=pl.Date),
            "atr": [value],
            "atr_long": [value],
        }
    )


@pytest.fixture
def unscaled(cfg):
    """Barrier unit equal to the ATR, so the arithmetic in these tests is readable."""
    return cfg.with_overrides(**{"labels.horizon_scaled_barriers": False})


def test_target_touched_first(unscaled):
    # Entry 100, ATR 10, target +2 ATR = 120, stop -1 ATR = 90. Path reaches 120.
    out = label_events(_event(), _ticks([100, 105, 121, 95]), _atr(10.0), unscaled)
    assert out["label"][0] == 1
    assert out["entry"][0] == pytest.approx(100.0)


def test_stop_touched_first(unscaled):
    out = label_events(_event(), _ticks([100, 95, 89, 130]), _atr(10.0), unscaled)
    assert out["label"][0] == -1


def test_time_barrier_when_neither_is_touched(unscaled):
    out = label_events(_event(), _ticks([100, 101, 99, 102]), _atr(10.0), unscaled)
    assert out["label"][0] == 0


def test_whichever_comes_first_wins(unscaled):
    """The stop is hit at index 1 and the target at index 2. The stop must win."""
    out = label_events(_event(), _ticks([100, 89, 121]), _atr(10.0), unscaled)
    assert out["label"][0] == -1


def test_window_end_records_the_resolution_time(unscaled):
    out = label_events(_event(), _ticks([100, 105, 121, 95]), _atr(10.0), unscaled)
    resolved = out["window_end"][0]
    assert resolved == START + timedelta(seconds=120)  # the bar where 121 printed
    assert resolved > out["t0"][0]


def test_short_side_mirrors_the_long_side(unscaled):
    up = label_events(_event(1), _ticks([100, 130]), _atr(10.0), unscaled)
    down = label_events(_event(-1), _ticks([100, 70]), _atr(10.0), unscaled)
    assert up["label"][0] == down["label"][0] == 1
    assert up["mfe"][0] == pytest.approx(down["mfe"][0])


def test_excursions_are_measured_in_atr_units(unscaled):
    out = label_events(_event(), _ticks([100, 115, 92]), _atr(10.0), unscaled)
    assert out["mfe"][0] == pytest.approx(1.5)  # +15 points on a 10-point ATR
    assert out["mae"][0] == pytest.approx(0.8)  # -8 points


def test_both_directions_are_recorded_regardless_of_side(unscaled):
    out = label_events(_event(-1), _ticks([100, 115, 92]), _atr(10.0), unscaled)
    assert out["mfe_up"][0] == pytest.approx(1.5)
    assert out["mfe_down"][0] == pytest.approx(0.8)


def test_mae_before_mfe_ignores_heat_taken_afterwards(unscaled):
    """The 30-point drawdown comes after the peak and must not enter the ratio."""
    out = label_events(_event(), _ticks([100, 95, 130, 70]), _atr(10.0), unscaled)
    assert out["mae_before_mfe"][0] == pytest.approx(0.5)  # only the -5 before the peak
    assert out["mae"][0] == pytest.approx(3.0)  # the full -30 is still recorded


def test_time_to_mfe_and_velocity(unscaled):
    out = label_events(_event(), _ticks([100, 110, 130, 105]), _atr(10.0), unscaled)
    assert out["time_to_mfe_min"][0] == pytest.approx(2.0)
    assert out["velocity"][0] == pytest.approx(out["mfe"][0] / 2.0)


def test_asymmetry_uses_the_configured_floor(unscaled):
    """MAE_before is zero here; without a floor the ratio is infinite."""
    out = label_events(_event(), _ticks([100, 110, 130]), _atr(10.0), unscaled)
    floor = unscaled.get("labels.mae_floor_ticks") * unscaled.get("contract.tick_size")
    assert out["mae_before_mfe"][0] == pytest.approx(0.0)
    assert out["asymmetry"][0] == pytest.approx(30.0 / floor)
    assert np.isfinite(out["asymmetry"][0])


def test_events_without_an_atr_are_dropped(unscaled):
    empty = pl.DataFrame(
        {"session_date": pl.Series([SESSION], dtype=pl.Date), "atr": [None], "atr_long": [None]},
        schema_overrides={"atr": pl.Float64, "atr_long": pl.Float64},
    )
    assert label_events(_event(), _ticks([100, 130]), empty, unscaled).is_empty()


def test_decision_lag_moves_the_entry(cfg):
    """Entry is the first print at or after the lag, not the price at the anchor."""
    lagged = cfg.with_overrides(
        **{"execution_realism.decision_lag_seconds": 120, "labels.horizon_scaled_barriers": False}
    )
    out = label_events(_event(), _ticks([100, 105, 111, 95]), _atr(10.0), lagged)
    assert out["entry"][0] == pytest.approx(111.0)


def test_horizon_scaling_shrinks_the_barrier(cfg):
    scale = barrier_unit_scale(cfg)
    assert 0 < scale < 1
    assert scale == pytest.approx(
        (cfg.get("labels.time_barrier_minutes") / cfg.get("labels.rth_minutes")) ** 0.5
    )


def test_atr_uses_only_prior_sessions(cfg, data_root):
    """The ATR is part of the label. A same-session estimate corrupts the outcome (L3)."""
    from src.data.store import Store

    bars = pl.read_parquet(Store.layout(data_root, "bars_time"))
    table = atr_table(bars, cfg)
    ranges = session_true_range(bars).sort("session_date")
    window = cfg.get("labels.atr_sessions")

    joined = table.sort("session_date").with_columns(
        ranges["true_range"].shift(1).rolling_mean(window, min_periods=window).alias("expected")
    ).drop_nulls("atr")
    assert joined.height > 0
    assert np.allclose(joined["atr"].to_numpy(), joined["expected"].to_numpy())


def test_atr_warmup_is_null_not_zero(cfg, data_root):
    from src.data.store import Store

    table = atr_table(pl.read_parquet(Store.layout(data_root, "bars_time")), cfg).sort("session_date")
    warmup = table.head(cfg.get("labels.atr_sessions"))
    assert warmup["atr"].is_null().all()
