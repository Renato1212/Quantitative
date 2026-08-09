"""Bar construction, schema contracts, and the threshold-recalibration leak surface."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from src.bars.build import BUILDERS, _trailing_thresholds, time_bars, volume_bars
from src.data.store import Store
from src.ingest.schemas import BarSchema, validate

KINDS = ["time", "volume", "dollar"]


@pytest.fixture(scope="module", params=KINDS)
def bars(request, data_root) -> pl.DataFrame:
    return pl.read_parquet(Store.layout(data_root, f"bars_{request.param}"))


def test_bars_satisfy_the_schema(bars):
    validate(BarSchema, bars.drop([c for c in bars.columns if c.endswith("_adj")]))


def test_every_bar_has_a_duration(bars):
    """Leak L1: a bar timestamped once cannot express 'not finished yet'."""
    assert (bars["close_ts"] > bars["open_ts"]).all()


def test_close_timestamps_are_unique_and_increasing(bars):
    assert bars["close_ts"].is_sorted()
    assert bars["close_ts"].n_unique() == bars.height


def test_high_low_bracket_the_body(bars):
    assert (bars["high"] >= bars[["open", "close"]].max_horizontal()).all()
    assert (bars["low"] <= bars[["open", "close"]].min_horizontal()).all()
    assert (bars["vwap"] <= bars["high"]).all()
    assert (bars["vwap"] >= bars["low"]).all()


def test_volume_and_tick_counts_are_positive(bars):
    assert (bars["volume"] > 0).all()
    assert (bars["n_ticks"] > 0).all()


def test_bars_carry_both_raw_and_adjusted_prices(bars):
    """Levels features use raw, return features use adjusted. Both must exist."""
    for column in ("open", "high", "low", "close"):
        assert column in bars.columns
        assert f"{column}_adj" in bars.columns


def test_time_bars_tile_the_clock(cfg, calendar, data_root):
    """Consecutive time bars in a session must abut: no gap, no overlap."""
    bars = pl.read_parquet(Store.layout(data_root, "bars_time"))
    every = cfg.get("bars.time_seconds")
    spans = bars.select(
        (pl.col("close_ts") - pl.col("open_ts")).dt.total_seconds().alias("span")
    )
    assert spans["span"].unique().to_list() == [every]


def test_threshold_bars_are_contiguous_within_a_contract(data_root):
    """Volume bars tile time too: each opens where the previous closed.

    The roll is the one legitimate break — a new contract starts its own chain.
    """
    bars = pl.read_parquet(Store.layout(data_root, "bars_volume")).sort("close_ts")
    breaks = bars.with_columns(
        (pl.col("open_ts") != pl.col("close_ts").shift(1)).alias("gap"),
        (pl.col("contract") != pl.col("contract").shift(1)).alias("rolled"),
    ).slice(1)
    assert not breaks.filter(~pl.col("rolled"))["gap"].any()


def test_ticks_sharing_a_timestamp_are_never_split(cfg, calendar):
    """A consumer at close_ts could not have seen half of a simultaneous group."""
    from datetime import date as _date

    stamps = pl.datetime_range(
        datetime(2021, 1, 4, 15, 0, tzinfo=timezone.utc),
        datetime(2021, 1, 4, 15, 0, 30, tzinfo=timezone.utc),
        interval="1s", eager=True, time_zone="UTC",
    )
    # Every timestamp carries three prints of 7 lots against a threshold of 10, so a
    # naive cumulative-sum split would cut straight through a simultaneous group.
    n = len(stamps) * 3
    ticks = pl.DataFrame(
        {
            "ts": pl.Series(sorted(list(stamps) * 3)),
            "price": pl.Series([4000.0] * n),
            "size": pl.Series([7] * n, dtype=pl.Int64),
            "aggressor": pl.Series([1] * n, dtype=pl.Int8),
            "contract": ["ESH21"] * n,
            "session_date": pl.Series([_date(2021, 1, 4)] * n, dtype=pl.Date),
        }
    )

    built = volume_bars(ticks, cfg.with_overrides(**{
        "bars.volume_contracts": 10, "bars.threshold_recalibration.enabled": False
    }), calendar)

    # Each bar's volume must be a whole number of 21-lot groups, and nothing is lost.
    assert built["volume"].sum() == ticks["size"].sum()
    assert all(v % 21 == 0 for v in built["volume"]), built["volume"].to_list()
    assert built["close_ts"].n_unique() == built.height


def test_trailing_threshold_uses_only_prior_sessions(cfg):
    """Leak L3's cousin: sizing a session's bars off its own volume is same-bar lookahead."""
    sessions = pl.date_range(
        pl.date(2021, 1, 4), pl.date(2021, 3, 1), interval="1d", eager=True
    )
    volumes = [100.0] * (len(sessions) - 1) + [10_000_000.0]  # a huge final session
    ticks = pl.DataFrame(
        {
            "session_date": sessions,
            "size": pl.Series(volumes).cast(pl.Int64),
        }
    )
    thresholds = _trailing_thresholds(
        ticks, cfg, pl.col("size").sum().cast(pl.Float64), fallback=1234.0
    ).sort("session_date")

    # The spike session must not have sized its own bars off its own volume.
    lookback = cfg.get("bars.threshold_recalibration.trailing_sessions")
    quiet = thresholds["_threshold"][lookback + 1]
    spike_day = thresholds["_threshold"][-1]
    assert spike_day == pytest.approx(quiet)


def test_disabled_recalibration_falls_back_to_the_constant(cfg):
    ticks = pl.DataFrame(
        {
            "session_date": pl.date_range(
                pl.date(2021, 1, 4), pl.date(2021, 2, 4), interval="1d", eager=True
            ),
            "size": pl.Series([5] * 32, dtype=pl.Int64),
        }
    )
    off = cfg.with_overrides(**{"bars.threshold_recalibration.enabled": False})
    thresholds = _trailing_thresholds(ticks, off, pl.col("size").sum().cast(pl.Float64), 99.0)
    assert thresholds["_threshold"].unique().to_list() == [99.0]


@pytest.mark.parametrize("kind", KINDS)
def test_builders_preserve_total_volume(cfg, calendar, data_root, kind):
    """Bars must account for every contract traded in the front month — none dropped."""
    bars = pl.read_parquet(Store.layout(data_root, f"bars_{kind}"))
    reference = pl.read_parquet(Store.layout(data_root, "bars_time"))
    assert bars["volume"].sum() == reference["volume"].sum()


def test_all_three_builders_are_registered():
    assert sorted(BUILDERS) == sorted(KINDS)
    assert BUILDERS["time"] is time_bars
