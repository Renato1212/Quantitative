"""Canary features: deliberate leaks, and the clean features they are paired with.

One canary per mechanism in the spec review's ranked list. A detector that catches only
the obvious one-bar peek gives false confidence in exactly the cases that matter, so the
Phase 1 gate requires every leak below to be caught *and* every clean feature to pass
untouched. A detector that flags everything is as useless as one that flags nothing.

The leaky features read the Parquet files directly, which is how this bug is actually
written: someone needs a frame the view will not hand over, reaches for the path, and
the resulting number looks entirely plausible.
"""

from __future__ import annotations

import math

import polars as pl

from src.data.point_in_time import PointInTimeView
from src.data.store import Store

BARS = "bars_time"
LOOKBACK = 20


def _raw_bars(view: PointInTimeView) -> pl.DataFrame:
    """Read the bar file directly, bypassing the view. Only canaries may do this."""
    return pl.read_parquet(Store.layout(Store.data_root(view.config), BARS)).sort("close_ts")


# ------------------------------------------------------------------ clean features


def clean_trailing_return(view: PointInTimeView) -> float:
    """Return over the last ``LOOKBACK`` completed bars. Reads only through the view."""
    bars = view.bars("time", tail=LOOKBACK + 1)
    if bars.height < LOOKBACK + 1:
        return math.nan
    return float(bars["close"][-1] / bars["close"][0] - 1.0)


def clean_session_vwap_so_far(view: PointInTimeView) -> float:
    """Developing session VWAP — the anytime form of a session aggregate."""
    bars = view.session_bars("time")
    if bars.is_empty():
        return math.nan
    weighted = (bars["vwap"] * bars["volume"]).sum()
    return float(weighted / bars["volume"].sum())


def clean_trailing_range(view: PointInTimeView) -> float:
    """High minus low over the trailing window, in points."""
    bars = view.bars("time", tail=LOOKBACK)
    if bars.height < LOOKBACK:
        return math.nan
    return float(bars["high"].max() - bars["low"].min())


# ------------------------------------------------------------------ deliberate leaks


def leak_peek_next_bar(view: PointInTimeView) -> float:
    """L0 — the obvious one. Reads the bar that closes after the cutoff.

    This is the canary ``CLAUDE.md`` §6 asks for. It is the easiest to catch and the
    least likely to be written by accident.
    """
    bars = _raw_bars(view)
    ahead = bars.filter(pl.col("close_ts") > view.cutoff)
    if ahead.is_empty():
        return math.nan
    return float(ahead["close"][0])


def leak_open_ts_filter(view: PointInTimeView) -> float:
    """L1 — filters on ``open_ts`` instead of ``close_ts``.

    Includes the bar that is still forming, whose high, low and close have not happened
    yet. A one-character difference from correct code, and the most likely real leak in
    the project.
    """
    bars = _raw_bars(view).filter(pl.col("open_ts") <= view.cutoff)
    if bars.height < LOOKBACK + 1:
        return math.nan
    return float(bars["close"][-1] / bars["close"][-(LOOKBACK + 1)] - 1.0)


def leak_global_zscore(view: PointInTimeView) -> float:
    """L3 — normalises against the whole study period's mean and standard deviation.

    Insidious because it corrupts the *labels* too, wherever barriers are ATR-scaled off
    a globally computed volatility. The feature looks stationary and well-behaved, which
    is exactly why nobody checks it.
    """
    bars = _raw_bars(view)
    latest = view.bars("time", tail=1)
    if latest.is_empty():
        return math.nan
    mean, std = bars["close"].mean(), bars["close"].std()
    if not std:
        return math.nan
    return float((latest["close"][0] - mean) / std)


def leak_full_session_vwap(view: PointInTimeView) -> float:
    """L4 — session VWAP over the entire session, including bars after the cutoff.

    The number is a perfectly reasonable VWAP. It is simply the one that will be known
    at the close, not the one known now.
    """
    session = view.current_session()
    if session is None:
        return math.nan
    bars = _raw_bars(view).filter(pl.col("session_date") == session.session_date)
    if bars.is_empty():
        return math.nan
    return float((bars["vwap"] * bars["volume"]).sum() / bars["volume"].sum())


def leak_centered_moving_average(view: PointInTimeView) -> float:
    """L5 — a centred window, which most charting libraries offer and some default to."""
    bars = _raw_bars(view)
    idx = bars.with_row_index().filter(pl.col("close_ts") <= view.cutoff)
    if idx.is_empty():
        return math.nan
    here = int(idx["index"][-1])
    half = LOOKBACK // 2
    window = bars["close"][max(0, here - half) : here + half + 1]
    return float(window.mean()) if len(window) else math.nan


CLEAN = {
    "clean_trailing_return": clean_trailing_return,
    "clean_session_vwap_so_far": clean_session_vwap_so_far,
    "clean_trailing_range": clean_trailing_range,
}

LEAKY = {
    "leak_peek_next_bar": leak_peek_next_bar,
    "leak_open_ts_filter": leak_open_ts_filter,
    "leak_global_zscore": leak_global_zscore,
    "leak_full_session_vwap": leak_full_session_vwap,
    "leak_centered_moving_average": leak_centered_moving_average,
}

ALL = CLEAN | LEAKY
