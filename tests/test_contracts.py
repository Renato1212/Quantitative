"""Contract identity, the volume roll, and back-adjustment."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.ingest.contracts import Contract, back_adjust, quarterly_contracts, volume_roll_schedule


def test_contract_codes_and_expiries():
    march = Contract(2024, 3, "ES")
    assert march.code == "ESH24"
    assert march.expiry == date(2024, 3, 15)  # third Friday
    assert Contract(2024, 12, "ES").expiry == date(2024, 12, 20)


def test_quarterly_contracts_are_ordered_and_quarterly(cfg):
    contracts = quarterly_contracts(cfg, date(2021, 1, 1), date(2023, 1, 1))
    assert contracts == sorted(contracts)
    assert {c.month for c in contracts} == {3, 6, 9, 12}


def _volumes(rows: list[tuple[date, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_date": [r[0] for r in rows],
            "contract": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
        },
        schema_overrides={"session_date": pl.Date, "volume": pl.Int64},
    )


@pytest.fixture
def crossover(cfg):
    """Five sessions; the deferred contract overtakes on the third."""
    days = [date(2024, 3, 11), date(2024, 3, 12), date(2024, 3, 13), date(2024, 3, 14), date(2024, 3, 15)]
    rows = []
    for i, day in enumerate(days):
        front, deferred = (900, 100) if i < 2 else (100, 900)
        rows += [(day, "ESH24", front), (day, "ESM24", deferred)]
    return days, volume_roll_schedule(_volumes(rows), cfg)


def test_roll_takes_effect_after_the_crossover_is_observable(cfg, crossover):
    """Leak L7: the crossover is an end-of-session fact, so the new front starts later."""
    days, schedule = crossover
    lag = cfg.get("contract.roll_publication_lag_sessions")
    fronts = dict(zip(schedule["session_date"], schedule["front"]))

    lookback = cfg.get("contract.roll_volume_lookback_sessions")
    # Trailing volume crosses over on the session at index `lookback`; with the
    # publication lag the front cannot change before the session after that.
    first_new = next(d for d in days if fronts[d] == "ESM24")
    assert days.index(first_new) >= lookback + lag


def test_front_month_never_moves_backwards(cfg):
    """A one-off volume spike in the old contract must not un-roll the series."""
    days = [date(2024, 3, d) for d in (11, 12, 13, 14, 15, 18, 19)]
    rows = []
    for i, day in enumerate(days):
        if i < 2:
            front, deferred = 900, 100
        elif i == 5:
            front, deferred = 5000, 100  # spike back into the expiring contract
        else:
            front, deferred = 100, 900
        rows += [(day, "ESH24", front), (day, "ESM24", deferred)]

    schedule = volume_roll_schedule(_volumes(rows), cfg)
    order = {"ESH24": 0, "ESM24": 1}
    ranks = [order[c] for c in schedule["front"]]
    assert ranks == sorted(ranks)


def test_roll_week_flag_covers_the_whole_week(cfg, crossover):
    _, schedule = crossover
    assert schedule["is_roll_week"].any()
    rolled = schedule.filter(pl.col("front") != pl.col("front").shift(1))
    week = rolled["session_date"][0].isocalendar()
    same_week = schedule.filter(
        pl.col("session_date").map_elements(
            lambda d: d.isocalendar()[:2] == week[:2], return_dtype=pl.Boolean
        )
    )
    assert same_week["is_roll_week"].all()


def test_back_adjustment_leaves_raw_prices_untouched(data_root, cfg):
    """Levels-based features read raw; a silent overwrite would corrupt every one."""
    from src.data.store import Store

    bars = pl.read_parquet(Store.layout(data_root, "bars_time"))
    assert (bars["close"] > 0).all()
    assert "close_adj" in bars.columns
    # The most recent segment carries no adjustment, by construction.
    last = bars.sort("close_ts").tail(1)
    assert last["close"][0] == pytest.approx(last["close_adj"][0])


def test_back_adjustment_removes_the_roll_gap():
    """Adjusted closes must be continuous across a roll where raw ones jump."""
    days = [date(2024, 3, 14), date(2024, 3, 15)]
    bars = pl.DataFrame(
        {
            "close_ts": pl.Series(
                ["2024-03-14T20:00:00Z", "2024-03-15T20:00:00Z"]
            ).str.to_datetime(time_zone="UTC"),
            "session_date": pl.Series(days, dtype=pl.Date),
            "open": [100.0, 130.0],
            "high": [101.0, 131.0],
            "low": [99.0, 129.0],
            "close": [100.0, 130.0],
        }
    )
    schedule = pl.DataFrame(
        {"session_date": pl.Series(days, dtype=pl.Date), "front": ["ESH24", "ESM24"]}
    )
    adjusted = back_adjust(bars, schedule).sort("close_ts")
    raw_jump = adjusted["close"][1] - adjusted["close"][0]
    adj_jump = adjusted["close_adj"][1] - adjusted["close_adj"][0]
    assert raw_jump == pytest.approx(30.0)
    assert adj_jump == pytest.approx(0.0)
