"""Trigger families.

Each family fires on a mechanically computable condition and is **outcome-agnostic**: it
fires on the sessions that go nowhere exactly as readily as on the sessions that run. The
sessions that go nowhere are the control group, and a trigger that quietly selects for
movement destroys the comparison the whole study rests on.

All four are evaluated over completed RTH bars with explicit shifts, and every anchor is a
bar close. Computing them bar-wise rather than through a ``PointInTimeView`` per candidate
timestamp is a speed decision, not a safety one:
``tests/test_events.py::test_every_event_is_reproducible_through_the_view`` re-derives a
sample of anchors through the view and requires them to agree, which is the §5 events
contract ("trigger conditions verified reproducible from bars").

Levels-based families are suppressed for the first sessions of a new front month, per
decision D11 — a prior-session high is a behavioural level in the contract people were
actually trading, and carrying it arithmetically across a roll asserts a continuity the
order book does not have.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.config import Config
from src.ingest.calendar import SessionCalendar

EVENT_COLUMNS = ["trigger_family", "t0", "side", "session_date"]


def _rth(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.filter(pl.col("in_rth")).sort("close_ts")


def _with_session_clock(bars: pl.DataFrame, calendar: SessionCalendar) -> pl.DataFrame:
    windows = calendar.frame().select("session_date", "rth_open", "rth_close")
    return bars.join(windows, on="session_date", how="inner").with_columns(
        (pl.col("close_ts") - pl.col("rth_open")).dt.total_minutes().alias("minutes_into_session")
    )


def _first_per_session(frame: pl.DataFrame, condition: pl.Expr, family: str, side: int) -> pl.DataFrame:
    """The first bar in each session satisfying the condition. At most one per session."""
    hits = frame.filter(condition).sort("close_ts")
    if hits.is_empty():
        return pl.DataFrame(schema={"trigger_family": pl.Utf8, "t0": pl.Datetime("us", "UTC"),
                                    "side": pl.Int8, "session_date": pl.Date})
    return (
        hits.group_by("session_date", maintain_order=True)
        .first()
        .select(
            pl.lit(family).alias("trigger_family"),
            pl.col("close_ts").alias("t0"),
            pl.lit(side, dtype=pl.Int8).alias("side"),
            pl.col("session_date"),
        )
        .sort("t0")
    )


def _suppress_after_roll(events: pl.DataFrame, schedule: pl.DataFrame, sessions: int) -> pl.DataFrame:
    """Drop events in the first ``sessions`` trade dates of a new front month (D11)."""
    marked = schedule.sort("session_date").with_columns(
        (pl.col("front") != pl.col("front").shift(1)).fill_null(False).alias("_new_front")
    )
    blocked: set = set()
    dates = marked["session_date"].to_list()
    for i, is_new in enumerate(marked["_new_front"]):
        if is_new:
            blocked.update(dates[i : i + sessions])
    return events.filter(~pl.col("session_date").is_in(list(blocked)))


# --------------------------------------------------------------------- families


def prior_day_levels(bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """First interaction with the prior session's RTH high or low.

    Raw prices, not back-adjusted: a level is a price people traded at.
    """
    rth = _rth(bars)
    prior = (
        rth.group_by("session_date", maintain_order=True)
        .agg(pl.col("high").max().alias("_h"), pl.col("low").min().alias("_l"))
        .sort("session_date")
        .with_columns(
            pl.col("_h").shift(1).alias("prior_high"), pl.col("_l").shift(1).alias("prior_low")
        )
        .select("session_date", "prior_high", "prior_low")
    )
    frame = rth.join(prior, on="session_date", how="inner").filter(pl.col("prior_high").is_not_null())
    return pl.concat(
        [
            _first_per_session(frame, pl.col("high") >= pl.col("prior_high"), "prior_day_high", 1),
            _first_per_session(frame, pl.col("low") <= pl.col("prior_low"), "prior_day_low", -1),
        ]
    )


def initial_balance_extension(bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """First trade beyond the initial balance once the balance period has closed."""
    minutes = cfg.get("events.initial_balance_minutes")
    frame = _with_session_clock(_rth(bars), calendar)
    balance = (
        frame.filter(pl.col("minutes_into_session") <= minutes)
        .group_by("session_date")
        .agg(pl.col("high").max().alias("ib_high"), pl.col("low").min().alias("ib_low"))
    )
    after = frame.join(balance, on="session_date", how="inner").filter(
        pl.col("minutes_into_session") > minutes
    )
    return pl.concat(
        [
            _first_per_session(after, pl.col("high") > pl.col("ib_high"), "ib_extension_up", 1),
            _first_per_session(after, pl.col("low") < pl.col("ib_low"), "ib_extension_down", -1),
        ]
    )


def opening_range_break(bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """First break of the opening range. Fires on most sessions, which is the point."""
    minutes = cfg.get("events.opening_range_minutes")
    frame = _with_session_clock(_rth(bars), calendar)
    opening = (
        frame.filter(pl.col("minutes_into_session") <= minutes)
        .group_by("session_date")
        .agg(pl.col("high").max().alias("or_high"), pl.col("low").min().alias("or_low"))
    )
    after = frame.join(opening, on="session_date", how="inner").filter(
        pl.col("minutes_into_session") > minutes
    )
    return pl.concat(
        [
            _first_per_session(after, pl.col("high") > pl.col("or_high"), "opening_range_up", 1),
            _first_per_session(after, pl.col("low") < pl.col("or_low"), "opening_range_down", -1),
        ]
    )


def vwap_reclaim(bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """Reclaim of the developing session VWAP after a sustained stretch on one side.

    The VWAP is cumulative within the session over completed bars — the anytime form, not
    the whole-session number known at the close (leak L4).
    """
    persistence = cfg.get("events.vwap_persistence_bars")
    frame = (
        _rth(bars)
        .with_columns(
            (
                (pl.col("vwap") * pl.col("volume")).cum_sum().over("session_date")
                / pl.col("volume").cum_sum().over("session_date")
            ).alias("session_vwap")
        )
        .with_columns((pl.col("close") > pl.col("session_vwap")).alias("_above"))
        .with_columns(
            pl.col("_above").cast(pl.Int32).rolling_sum(persistence).over("session_date").alias("_above_run")
        )
    )
    reclaim_up = pl.col("_above") & (pl.col("_above_run").shift(1) == 0)
    reclaim_down = ~pl.col("_above") & (pl.col("_above_run").shift(1) == persistence)
    return pl.concat(
        [
            _first_per_session(frame, reclaim_up, "vwap_reclaim_up", 1),
            _first_per_session(frame, reclaim_down, "vwap_reclaim_down", -1),
        ]
    )


FAMILIES = {
    "prior_day_levels": prior_day_levels,
    "initial_balance_extension": initial_balance_extension,
    "opening_range_break": opening_range_break,
    "vwap_reclaim": vwap_reclaim,
}

# Families whose condition is a price level carried over from a prior session.
LEVELS_BASED = {"prior_day_levels"}


def build_events(
    bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar, schedule: pl.DataFrame
) -> pl.DataFrame:
    """Every family, deduplicated on ``(trigger_family, t0)`` per the §5 events contract."""
    frames = []
    suppress = cfg.get("events.suppress_sessions_after_roll")
    for name, builder in FAMILIES.items():
        events = builder(bars, cfg, calendar)
        if name in LEVELS_BASED:
            events = _suppress_after_roll(events, schedule, suppress)
        frames.append(events)

    combined = pl.concat(frames).sort("t0", "trigger_family")
    return combined.unique(subset=["trigger_family", "t0"], keep="first", maintain_order=True)


def events_per_year(events: pl.DataFrame) -> pl.DataFrame:
    """The Phase 2 gate table: n per family per year."""
    return (
        events.with_columns(pl.col("session_date").dt.year().alias("year"))
        .group_by("trigger_family", "year")
        .len()
        .rename({"len": "n"})
        .sort("trigger_family", "year")
    )
