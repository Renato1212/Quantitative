"""Bar construction: time, volume and dollar.

Every bar carries ``open_ts`` and ``close_ts``, and bars tile the time axis so that
``close_ts`` of one is ``open_ts`` of the next. Point-in-time truncation filters on
``close_ts``, which makes a bar invisible until it has finished forming. Timestamping a
bar once — at its open — is the single most likely lookahead bug in the project (leak
L1), so the representation refuses to allow it.

Ticks sharing a timestamp are never split across bars. A threshold reached in the
middle of a burst of same-timestamp prints extends to take the whole burst, because a
consumer at ``close_ts`` could not have seen half of a simultaneous group.

Volume and dollar thresholds are recalibrated from a trailing window of prior sessions
only. Sizing them off the whole study period is the same class of error as a global
z-score — it is leak L3 wearing a different hat, and it silently changes what a "bar"
means in 2021 based on what happened in 2025.
"""

from __future__ import annotations

import polars as pl

from src.config import Config
from src.ingest.calendar import SessionCalendar

_AGGREGATIONS = [
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
    pl.col("size").sum().alias("volume"),
    (pl.col("price") * pl.col("size")).sum().alias("_pv"),
    pl.len().cast(pl.Int64).alias("n_ticks"),
    pl.col("session_date").first(),
    pl.col("contract").first(),
    pl.col("ts").first().alias("_first_ts"),
    pl.col("ts").last().alias("_last_ts"),
]


def _finalise(bars: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """Derived columns, RTH flag, and the column order the schema expects."""
    point_value = cfg.get("contract.point_value")
    windows = calendar.frame().select("session_date", "rth_open", "rth_close")
    return (
        bars.with_columns(
            (pl.col("_pv") / pl.col("volume")).alias("vwap"),
            (pl.col("_pv") * point_value).alias("dollar_volume"),
        )
        .join(windows, on="session_date", how="left")
        .with_columns(
            (
                (pl.col("close_ts") > pl.col("rth_open"))
                & (pl.col("close_ts") <= pl.col("rth_close"))
            ).alias("in_rth")
        )
        .select(
            "open_ts",
            "close_ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_volume",
            "vwap",
            "n_ticks",
            "session_date",
            "contract",
            "in_rth",
        )
        .sort("close_ts")
    )


def time_bars(ticks: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    """Fixed-duration bars. The window end is the close timestamp, always."""
    every = f"{cfg.get('bars.time_seconds')}s"
    bars = (
        ticks.sort("ts")
        .group_by_dynamic("ts", every=every, closed="left", label="left", group_by="contract")
        .agg([a for a in _AGGREGATIONS if a.meta.output_name() != "contract"])
        .rename({"ts": "open_ts"})
        .with_columns(
            (pl.col("open_ts") + pl.duration(seconds=cfg.get("bars.time_seconds"))).alias("close_ts")
        )
    )
    return _finalise(bars, cfg, calendar)


def _threshold_bars(
    ticks: pl.DataFrame,
    weights: pl.Expr,
    thresholds: pl.DataFrame,
    cfg: Config,
    calendar: SessionCalendar,
) -> pl.DataFrame:
    """Bars that close once an accumulating quantity crosses a per-session threshold."""
    working = (
        ticks.sort("ts")
        .join(thresholds, on="session_date", how="left")
        .with_columns(weights.alias("_w"))
        .with_columns(
            (pl.col("_w").cum_sum().over("session_date", "contract") / pl.col("_threshold"))
            .floor()
            .cast(pl.Int64)
            .alias("_raw_bar")
        )
        # Never split a group of ticks that share a timestamp across two bars.
        .with_columns(pl.col("_raw_bar").max().over("session_date", "contract", "ts").alias("_bar"))
    )

    bars = (
        working.group_by("session_date", "contract", "_bar", maintain_order=True)
        .agg([a for a in _AGGREGATIONS if a.meta.output_name() not in ("session_date", "contract")])
        .sort("_last_ts")
        .with_columns(pl.col("_last_ts").alias("close_ts"))
        .with_columns(
            pl.col("close_ts")
            .shift(1)
            .over("contract")
            .fill_null(pl.col("_first_ts") - pl.duration(microseconds=1))
            .alias("open_ts")
        )
    )
    return _finalise(bars, cfg, calendar)


def _trailing_thresholds(
    ticks: pl.DataFrame, cfg: Config, quantity: pl.Expr, fallback: float
) -> pl.DataFrame:
    """Per-session threshold from prior sessions only.

    ``shift(1)`` before the rolling mean is what makes it prior-only: without it the
    session sizes its own bars using its own volume, which is a same-bar lookahead.
    Sessions before the window is full fall back to the configured constant.
    """
    if not cfg.get("bars.threshold_recalibration.enabled"):
        return (
            ticks.select("session_date")
            .unique()
            .with_columns(pl.lit(fallback, dtype=pl.Float64).alias("_threshold"))
        )

    lookback = cfg.get("bars.threshold_recalibration.trailing_sessions")
    target = cfg.get("bars.threshold_recalibration.target_bars_per_session")
    per_session = (
        ticks.group_by("session_date").agg(quantity.alias("_total")).sort("session_date")
    )
    return per_session.with_columns(
        (
            pl.col("_total")
            .shift(1)
            .rolling_mean(window_size=lookback, min_periods=lookback)
            / target
        )
        .fill_null(fallback)
        .alias("_threshold")
    ).select("session_date", "_threshold")


def volume_bars(ticks: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    thresholds = _trailing_thresholds(
        ticks, cfg, pl.col("size").sum().cast(pl.Float64), float(cfg.get("bars.volume_contracts"))
    )
    return _threshold_bars(ticks, pl.col("size").cast(pl.Float64), thresholds, cfg, calendar)


def dollar_bars(ticks: pl.DataFrame, cfg: Config, calendar: SessionCalendar) -> pl.DataFrame:
    point_value = cfg.get("contract.point_value")
    notional = pl.col("price") * pl.col("size") * point_value
    thresholds = _trailing_thresholds(
        ticks, cfg, notional.sum(), float(cfg.get("bars.dollar_notional"))
    )
    return _threshold_bars(ticks, notional, thresholds, cfg, calendar)


BUILDERS = {"time": time_bars, "volume": volume_bars, "dollar": dollar_bars}
