"""Trailing ATR, computed from completed sessions only.

Barriers are ATR-scaled, so the ATR is not a feature — it is part of the *label*. A
volatility estimate fitted on the whole period would therefore corrupt every outcome in
the study, not merely a column of the feature matrix. That is leak L3 in its most
expensive form, and it is why this lives in ``labels/`` rather than ``features/``.

Two lookbacks are produced, both pre-registered in config: the primary and the robustness
scaling from decision D7. A finding that appears under only one of them is a
scaling-conditional finding and is reported as one.
"""

from __future__ import annotations

import polars as pl

from src.config import Config


def session_true_range(bars: pl.DataFrame) -> pl.DataFrame:
    """Per-session true range from RTH bars: the usual max of the three spans."""
    daily = (
        bars.filter(pl.col("in_rth"))
        .sort("close_ts")
        .group_by("session_date", maintain_order=True)
        .agg(
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
        )
        .sort("session_date")
    )
    return daily.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("true_range")
    )


def atr_table(bars: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """ATR per session, usable *during* that session.

    ``shift(1)`` before the rolling mean is the whole point: the value attached to a
    session is built from sessions strictly before it, so a barrier set at 09:35 does not
    know what the range will turn out to be by 15:00.

    Sessions before the lookback is full carry a null ATR, and events anchored in them are
    dropped rather than labelled against a half-formed estimate.
    """
    daily = session_true_range(bars)
    columns = []
    for name, window in (
        ("atr", cfg.get("labels.atr_sessions")),
        ("atr_long", cfg.get("labels.atr_sessions_robustness")),
    ):
        columns.append(
            pl.col("true_range")
            .shift(1)
            .rolling_mean(window_size=window, min_periods=window)
            .alias(name)
        )
    return daily.with_columns(columns).select("session_date", "atr", "atr_long", "true_range")
