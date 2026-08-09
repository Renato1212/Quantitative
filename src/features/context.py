"""Context features, all computed through ``PointInTimeView``.

Every function here takes a view and nothing else. It cannot reach the data root, cannot
see past its cutoff, and is audited by the same probes as the leakage canaries — the
Phase 1 gate runs this whole set through ``audit_all`` and requires every one to come
back clean.

Each docstring states the information window the feature uses, as the §5 features
contract requires. Where a feature is normalised, the normaliser is trailing and
point-in-time; a full-period scaler here would corrupt everything downstream (leak L3).

Features return NaN rather than a guess when their window is not yet full. The NaN policy
is explicit and uniform: insufficient history is missing, not zero.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.data.point_in_time import PointInTimeView

SHORT_WINDOW = 30
LONG_WINDOW = 120


def _atr_proxy(view: PointInTimeView, sessions: int = 14) -> float:
    """Trailing session range average, from completed sessions only.

    A feature-side echo of ``labels/atr.py``. It is recomputed here rather than joined
    so that features stay a pure function of the view — a joined column would be a second
    read path, and the audit could not see it.
    """
    prior = view.prior_sessions(sessions)
    if len(prior) < sessions:
        return math.nan
    # Bounded by date, not by row count: a row cap silently returns fewer sessions than
    # asked for whenever bar density changes, and the average is then over the wrong window.
    bars = view.bars("time", since=prior[0].eth_open)
    if bars.is_empty():
        return math.nan
    wanted = {s.session_date for s in prior}
    ranges = (
        bars.filter(pl.col("in_rth") & pl.col("session_date").is_in(list(wanted)))
        .group_by("session_date")
        .agg((pl.col("high").max() - pl.col("low").min()).alias("range"))
    )
    return float(ranges["range"].mean()) if ranges.height else math.nan


def minutes_into_session(view: PointInTimeView) -> float:
    """Minutes from the RTH open to the cutoff. Window: the calendar, known in advance."""
    session = view.current_session()
    if session is None:
        return math.nan
    return (view.cutoff - session.rth_open).total_seconds() / 60.0


def overnight_range_atr(view: PointInTimeView) -> float:
    """Electronic-session range before the RTH open, in ATR units.

    Window: bars from the overnight open up to the RTH open of the current session.
    Returns NaN before the RTH open, when the window is not yet closed.
    """
    session = view.current_session()
    atr = _atr_proxy(view)
    if session is None or not atr or math.isnan(atr):
        return math.nan
    bars = view.bars("time", since=session.eth_open).filter(
        pl.col("close_ts") <= session.rth_open
    )
    if bars.is_empty():
        return math.nan
    return float(bars["high"].max() - bars["low"].min()) / atr


def gap_atr(view: PointInTimeView) -> float:
    """RTH open against the prior session's last RTH price, in ATR units.

    Window: the prior completed session and the first bar of the current one. Uses raw
    prices — a gap is a level comparison, so it must not use the back-adjusted series.
    """
    session = view.current_session()
    atr = _atr_proxy(view)
    if session is None or not atr or math.isnan(atr):
        return math.nan
    prior = view.prior_sessions(1)
    if not prior:
        return math.nan
    bars = view.bars("time", since=prior[-1].eth_open)
    previous = bars.filter(pl.col("session_date") == prior[-1].session_date).filter(pl.col("in_rth"))
    opening = bars.filter((pl.col("session_date") == session.session_date) & pl.col("in_rth"))
    if previous.is_empty() or opening.is_empty():
        return math.nan
    return float(opening["open"][0] - previous["close"][-1]) / atr


def initial_balance_range_atr(view: PointInTimeView) -> float:
    """Initial balance range so far, in ATR units.

    Window: RTH bars from the open to the cutoff, capped at the configured balance
    period. Before the balance period closes this is the *developing* range, which is
    what a decision at that moment could actually see.
    """
    session = view.current_session()
    atr = _atr_proxy(view)
    if session is None or not atr or math.isnan(atr):
        return math.nan
    minutes = view.config.get("events.initial_balance_minutes")
    bars = view.session_bars("time")
    if bars.is_empty():
        return math.nan
    cutoff_minutes = min(minutes, (view.cutoff - session.rth_open).total_seconds() / 60.0)
    window = bars.filter(
        (pl.col("close_ts") - pl.lit(session.rth_open)).dt.total_seconds() <= cutoff_minutes * 60
    )
    if window.is_empty():
        return math.nan
    return float(window["high"].max() - window["low"].min()) / atr


def distance_from_vwap_atr(view: PointInTimeView) -> float:
    """Last price against the developing session VWAP, in ATR units, signed.

    Window: RTH bars of the current session up to the cutoff. The developing VWAP, never
    the close-of-session one (leak L4).
    """
    atr = _atr_proxy(view)
    bars = view.session_bars("time")
    if bars.is_empty() or not atr or math.isnan(atr):
        return math.nan
    vwap = float((bars["vwap"] * bars["volume"]).sum() / bars["volume"].sum())
    return (float(bars["close"][-1]) - vwap) / atr


def session_volume_ratio(view: PointInTimeView) -> float:
    """Session volume so far against the prior session's full RTH volume.

    Window: current session to the cutoff, plus one completed prior session. A crude
    participation measure; it deliberately does not normalise by time of day, which
    makes it correlated with ``minutes_into_session`` and is stated here so nobody
    reads the pair as two independent findings.
    """
    prior = view.prior_sessions(1)
    bars = view.session_bars("time")
    if not prior or bars.is_empty():
        return math.nan
    previous = view.bars("time", since=prior[-1].eth_open).filter(
        pl.col("in_rth") & (pl.col("session_date") == prior[-1].session_date)
    )
    if previous.is_empty() or previous["volume"].sum() == 0:
        return math.nan
    return float(bars["volume"].sum() / previous["volume"].sum())


def cumulative_delta_ratio(view: PointInTimeView) -> float:
    """Signed aggressor volume as a share of total, over the session so far.

    Window: ticks from the current session's RTH open to the cutoff. Requires an
    exchange-provided aggressor flag; if the real feed turns out to infer it, this
    feature carries that inference's error and must be re-examined (decision D10).
    """
    session = view.current_session()
    if session is None:
        return math.nan
    ticks = view.ticks(since=session.rth_open)
    if ticks.is_empty():
        return math.nan
    total = float(ticks["size"].sum())
    if total == 0:
        return math.nan
    signed = float((ticks["size"] * ticks["aggressor"]).sum())
    return signed / total


def realised_vol_ratio(view: PointInTimeView) -> float:
    """Short-window realised volatility over long-window, both trailing.

    Window: the last ``LONG_WINDOW`` completed bars. A ratio above one says the market
    has sped up relative to its own recent norm — the closest thing here to a
    positioning-stress proxy, and the feature most of the pre-registered hypotheses lean
    on.
    """
    bars = view.bars("time", tail=LONG_WINDOW + 1)
    if bars.height < LONG_WINDOW + 1:
        return math.nan
    closes = bars["close"].to_numpy()
    returns = np.diff(np.log(closes))
    short, long = returns[-SHORT_WINDOW:], returns
    if long.std() == 0:
        return math.nan
    return float(short.std() / long.std())


def trailing_return_atr(view: PointInTimeView) -> float:
    """Signed move over the last ``SHORT_WINDOW`` completed bars, in ATR units.

    Window: the last ``SHORT_WINDOW`` + 1 completed bars. Uses raw closes; over a roll
    boundary this is discontinuous, which is why levels-based families are suppressed
    there (decision D11).
    """
    atr = _atr_proxy(view)
    bars = view.bars("time", tail=SHORT_WINDOW + 1)
    if bars.height < SHORT_WINDOW + 1 or not atr or math.isnan(atr):
        return math.nan
    return float(bars["close"][-1] - bars["close"][0]) / atr


FEATURES = {
    "minutes_into_session": minutes_into_session,
    "overnight_range_atr": overnight_range_atr,
    "gap_atr": gap_atr,
    "initial_balance_range_atr": initial_balance_range_atr,
    "distance_from_vwap_atr": distance_from_vwap_atr,
    "session_volume_ratio": session_volume_ratio,
    "cumulative_delta_ratio": cumulative_delta_ratio,
    "realised_vol_ratio": realised_vol_ratio,
    "trailing_return_atr": trailing_return_atr,
}


def build_feature_matrix(events: pl.DataFrame, store, cfg, calendar) -> pl.DataFrame:
    """One row per event, keyed ``(trigger_family, t0)`` as the §4 layout requires."""
    from src.data.point_in_time import PointInTimeView as View

    rows = []
    for family, t0 in zip(events["trigger_family"], events["t0"]):
        view = View(store, t0, cfg, calendar)
        rows.append(
            {"trigger_family": family, "t0": t0}
            | {name: float(fn(view)) for name, fn in FEATURES.items()}
        )
    schema = {"trigger_family": pl.Utf8, "t0": pl.Datetime("us", "UTC")} | {
        name: pl.Float64 for name in FEATURES
    }
    return pl.DataFrame(rows, schema=schema)
