"""``PointInTimeView`` — the primitive every no-lookahead claim rests on (C1).

If these tests pass and feature code only ever reads through the view, lookahead is
structurally impossible rather than merely absent by convention.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.data.point_in_time import PointInTimeView
from src.data.store import Store


@pytest.fixture(scope="module")
def store(cfg, data_root):
    handle = Store.from_config(cfg, root=data_root)
    yield handle
    handle.close()


@pytest.fixture(scope="module")
def anchor(cfg, data_root) -> datetime:
    from src.pipeline import anchors

    return anchors(cfg, data_root, n=1)[0]


def test_naive_timestamps_are_rejected(store, cfg):
    """A naive t0 hides exactly the DST bugs the calendar works to avoid."""
    with pytest.raises(ValueError, match="timezone-aware"):
        PointInTimeView(store, datetime(2024, 3, 20, 15, 0), cfg)


def test_cutoff_applies_the_decision_lag(store, cfg, anchor):
    """Leak L2: a feature computed at t0 is available to a human a moment later."""
    view = PointInTimeView(store, anchor, cfg)
    lag = cfg.get("execution_realism.decision_lag_seconds")
    assert view.cutoff == anchor - timedelta(seconds=lag)
    assert view.cutoff < view.t0


@pytest.mark.parametrize("kind", ["time", "volume", "dollar"])
def test_no_bar_closes_after_the_cutoff(store, cfg, anchor, kind):
    view = PointInTimeView(store, anchor, cfg)
    bars = view.bars(kind)
    assert not bars.is_empty()
    assert bars["close_ts"].max() <= view.cutoff


def test_bars_are_filtered_on_close_not_open(store, cfg, anchor, data_root):
    """Leak L1. The bar in progress opened before the cutoff and must still be hidden."""
    view = PointInTimeView(store, anchor, cfg)
    visible = view.bars("time")
    everything = pl.read_parquet(Store.layout(data_root, "bars_time"))
    in_progress = everything.filter(
        (pl.col("open_ts") <= view.cutoff) & (pl.col("close_ts") > view.cutoff)
    )
    assert in_progress.height >= 1, "fixture has no bar straddling the cutoff to test with"
    assert in_progress["close_ts"][0] not in set(visible["close_ts"].to_list())


def test_ticks_are_truncated_too(store, cfg, anchor):
    view = PointInTimeView(store, anchor, cfg)
    assert view.ticks(tail=500)["ts"].max() <= view.cutoff


def test_tail_returns_the_most_recent_rows_in_ascending_order(store, cfg, anchor):
    view = PointInTimeView(store, anchor, cfg)
    everything = view.bars("time")
    tail = view.bars("time", tail=10)
    assert tail.height == 10
    assert tail["close_ts"].is_sorted()
    assert tail["close_ts"].to_list() == everything["close_ts"].to_list()[-10:]


def test_session_bars_stop_at_the_cutoff(store, cfg, anchor):
    """Leak L4: the developing session aggregate, not the one known at the close."""
    view = PointInTimeView(store, anchor, cfg)
    session = view.current_session()
    assert session is not None
    bars = view.session_bars("time")
    assert not bars.is_empty()
    assert bars["session_date"].unique().to_list() == [session.session_date]
    assert bars["close_ts"].max() <= view.cutoff
    assert bars["in_rth"].all()


def test_prior_sessions_exclude_the_one_in_progress(store, cfg, anchor):
    view = PointInTimeView(store, anchor, cfg)
    current = view.current_session()
    prior = view.prior_sessions(5)
    assert current is not None
    assert all(s.session_date < current.session_date for s in prior)
    assert all(s.rth_close <= view.cutoff for s in prior)
    assert [s.session_date for s in prior] == sorted(s.session_date for s in prior)


def test_earlier_cutoff_never_sees_more_than_a_later_one(store, cfg, anchor):
    early = PointInTimeView(store, anchor - timedelta(hours=2), cfg)
    late = PointInTimeView(store, anchor, cfg)
    assert early.bars("time").height <= late.bars("time").height


def test_a_cutoff_before_all_data_returns_nothing(store, cfg):
    view = PointInTimeView(store, datetime(2000, 1, 1, tzinfo=timezone.utc), cfg)
    assert view.bars("time").is_empty()
    assert view.current_session() is None
    assert view.prior_sessions(3) == []
