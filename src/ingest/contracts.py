"""Quarterly contract identity and the front-month roll.

Two decisions live here and both are leakage surfaces.

*When the roll is knowable.* A volume-based rule says "roll when the deferred contract
out-trades the front". That comparison uses a whole session's volume, so it is settled
only after the close. A feature that says "we are in the new front month" during the
crossover session is using information that did not exist yet. The schedule therefore
applies ``contract.roll_publication_lag_sessions`` before the new front takes effect.
This is leak L7 in the spec review.

*Levels across the roll.* Back-adjustment makes returns continuous and levels false;
raw makes levels true and returns discontinuous. Both series are produced. The
adjustment offsets are also the map for carrying a prior-session level across a roll
boundary — a prior-day high set in the old contract is not comparable to price in the
new one without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from src.config import Config

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


@dataclass(frozen=True, order=True)
class Contract:
    """A single delivery month, e.g. ESH24."""

    year: int
    month: int
    root: str = "ES"

    @property
    def code(self) -> str:
        letter = next(k for k, v in MONTH_CODES.items() if v == self.month)
        return f"{self.root}{letter}{self.year % 100:02d}"

    @property
    def expiry(self) -> date:
        """Third Friday of the delivery month — CME equity index convention."""
        first = date(self.year, self.month, 1)
        return first + timedelta(days=(4 - first.weekday()) % 7 + 14)

    def __str__(self) -> str:
        return self.code


def quarterly_contracts(cfg: Config, start: date, end: date) -> list[Contract]:
    """Every quarterly contract whose expiry could serve the window, in order."""
    months = sorted(MONTH_CODES[m] for m in cfg.get("contract.months"))
    root = cfg.get("scope.primary_instrument")
    out = [
        Contract(year, month, root)
        for year in range(start.year, end.year + 2)
        for month in months
    ]
    # Keep one contract either side so the window always has a front and a deferred.
    return [c for c in out if start - timedelta(days=200) <= c.expiry <= end + timedelta(days=200)]


def volume_roll_schedule(volumes: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Assign a front month to each session from per-contract session volume.

    ``volumes`` has columns ``session_date``, ``contract``, ``volume``. The rule: the
    front month is the unexpired contract with the highest trailing volume; once the
    front month advances it never goes back, and the change takes effect
    ``roll_publication_lag_sessions`` sessions after the crossover is observable.

    Returns ``session_date``, ``front``, ``is_roll_week``.
    """
    lookback = cfg.get("contract.roll_volume_lookback_sessions")
    lag = cfg.get("contract.roll_publication_lag_sessions")

    trailing = (
        volumes.sort(["contract", "session_date"])
        .with_columns(
            pl.col("volume")
            .rolling_sum(window_size=lookback, min_periods=1)
            .over("contract")
            .alias("trailing_volume")
        )
        .sort(["session_date", "trailing_volume", "contract"], descending=[False, True, False])
        .group_by("session_date", maintain_order=True)
        .first()
        .select("session_date", pl.col("contract").alias("observed_front"))
        .sort("session_date")
    )

    # Monotonicity: the front month only ever moves forward.
    ranks = {c: i for i, c in enumerate(sorted(volumes["contract"].unique().to_list()))}
    best, forward = -1, []
    for name in trailing["observed_front"]:
        best = max(best, ranks[name])
        forward.append(next(c for c, i in ranks.items() if i == best))

    schedule = trailing.with_columns(pl.Series("monotonic_front", forward))

    # The crossover is an end-of-session fact, so the new front starts `lag` sessions later.
    schedule = schedule.with_columns(
        pl.col("monotonic_front").shift(lag).backward_fill().alias("front")
    ).select("session_date", "front")

    changed = schedule.select(
        pl.col("session_date"),
        (pl.col("front") != pl.col("front").shift(1)).fill_null(False).alias("rolled"),
    )
    roll_dates = changed.filter("rolled")["session_date"].to_list()
    week_keys = {(d.isocalendar().year, d.isocalendar().week) for d in roll_dates}

    return schedule.with_columns(
        pl.col("session_date")
        .map_elements(
            lambda d: (d.isocalendar().year, d.isocalendar().week) in week_keys,
            return_dtype=pl.Boolean,
        )
        .alias("is_roll_week")
    )


def back_adjust(bars: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Add back-adjusted prices alongside raw ones.

    Offsets accumulate backwards from the most recent contract so that today's adjusted
    price equals today's raw price — the usual convention, and the one that keeps recent
    levels interpretable. Every price column gains an ``_adj`` twin; the raw columns are
    left untouched because levels-based features must use them.
    """
    fronts = schedule.select("session_date", "front")
    joined = bars.join(fronts, on="session_date", how="left")

    # Gap at each roll: the front contract's first close against the previous front's last.
    per_session = (
        joined.sort("close_ts")
        .group_by("session_date", maintain_order=True)
        .agg(pl.col("close").last().alias("last_close"), pl.col("front").last())
        .sort("session_date")
    )
    gaps = per_session.with_columns(
        pl.when(pl.col("front") != pl.col("front").shift(1))
        .then(pl.col("last_close") - pl.col("last_close").shift(1))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("gap")
    )
    # A session's offset is the sum of every roll gap that happens *after* it, so the
    # final segment is unadjusted and earlier segments are lifted onto it.
    total = gaps["gap"].sum()
    gaps = gaps.with_columns((total - pl.col("gap").cum_sum()).alias("offset"))

    out = joined.join(gaps.select("session_date", "offset"), on="session_date", how="left")
    price_cols = [c for c in ("open", "high", "low", "close", "vwap") if c in out.columns]
    return out.with_columns(
        [(pl.col(c) + pl.col("offset")).alias(f"{c}_adj") for c in price_cols]
    ).drop("offset")
