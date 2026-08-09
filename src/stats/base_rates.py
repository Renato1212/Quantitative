"""Unconditional forward-excursion distributions (Phase 3).

Distributions, not means. The research question is about the right tail, and a mean
excursion tells you nothing about it — two families with identical means can have tail
probabilities differing by a factor of five.

Everything here is unconditional. C2 requires that no conditional statistic is ever
reported without these beside it, and ``stats/hypotheses.py`` enforces that by construction:
it cannot emit a conditional table without the base rate it was computed against.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import Config
from src.stats.bootstrap import Interval, block_bootstrap, exceedance

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TAIL_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 6.0)
METRIC = "mfe_stops"  # MFE in multiples of the stop distance, per §6 Phase 3


def quantile_table(labels: pl.DataFrame, metric: str = METRIC) -> pl.DataFrame:
    """Full quantiles per trigger family. The shape, before any summary of it."""
    rows = []
    for family, group in _by_family(labels):
        values = _finite(group[metric])
        if values.size == 0:
            continue
        rows.append(
            {"trigger_family": family, "n": int(values.size)}
            | {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}
        )
    return pl.DataFrame(rows).sort("trigger_family") if rows else pl.DataFrame()


def tail_table(labels: pl.DataFrame, cfg: Config, metric: str = METRIC) -> pl.DataFrame:
    """``P(MFE > k x stop)`` per family, with block-bootstrapped intervals.

    The interval width is the point of this table. At 4x and 6x the counts fall into
    single digits and the interval spans a factor of several — which is the finding, and
    it is reported rather than hidden behind a point estimate.
    """
    rows = []
    for family, group in _by_family(labels):
        values = _finite(group[metric])
        groups = _blocks(group, metric)
        if values.size == 0:
            continue
        for threshold in TAIL_THRESHOLDS:
            interval = _ci(values, groups, exceedance(threshold), cfg)
            rows.append(
                {
                    "trigger_family": family,
                    "threshold": threshold,
                    "p": interval.point,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "n": interval.n,
                    "n_blocks": interval.n_blocks,
                    "n_exceeding": int((values > threshold).sum()),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def stratified_table(
    labels: pl.DataFrame, cfg: Config, stratum: str, metric: str = METRIC, threshold: float = 2.0
) -> pl.DataFrame:
    """One tail probability per (family, stratum) cell — the stability check.

    A base rate that holds in 2021 and collapses in 2024 is a regime-conditional base
    rate, and §7 requires it to be labelled one rather than averaged into a single number.
    """
    rows = []
    for (family, level), group in labels.group_by(["trigger_family", stratum], maintain_order=True):
        values = _finite(group[metric])
        if values.size == 0:
            continue
        interval = _ci(values, _blocks(group, metric), exceedance(threshold), cfg)
        rows.append(
            {
                "trigger_family": family,
                stratum: level,
                "threshold": threshold,
                "p": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "n": interval.n,
                "n_blocks": interval.n_blocks,
            }
        )
    return pl.DataFrame(rows).sort("trigger_family", stratum) if rows else pl.DataFrame()


def with_strata(labels: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Add the stratification columns Phase 3 reports across.

    The volatility regime is a **within-training** tercile of the trailing ATR. It is a
    trailing quantity, so no future information enters an individual row; the tercile
    boundaries themselves are fitted on whatever frame is passed, which must therefore be
    the training block and never the holdout.
    """
    hours = pl.col("t0").dt.hour()
    return labels.with_columns(
        pl.col("session_date").dt.year().alias("year"),
        pl.when(hours < 15).then(pl.lit("open")).when(hours < 18).then(pl.lit("midday"))
        .otherwise(pl.lit("close")).alias("time_of_day"),
        pl.col("atr").qcut(3, labels=["low_vol", "mid_vol", "high_vol"]).alias("vol_regime"),
    )


def cost_floor(cfg: Config) -> dict[str, float]:
    """What the tail must clear before any of this is tradeable (spec review W7).

    Commission plus a spread assumption plus volatility-scaled slippage, expressed as a
    fraction of the stop distance. If the unconditional tail probability implies an
    expectancy below this, the phenomenon is not tradeable and the project changes shape
    at the Phase 3 gate rather than at Phase 6.
    """
    tick = cfg.get("contract.tick_size")
    point_value = cfg.get("contract.point_value")
    commission = cfg.get("execution_realism.commission_per_side_usd") * 2
    spread_points = cfg.get("execution_realism.spread_ticks") * tick
    slippage_points = cfg.get("execution_realism.slippage_ticks_per_atr") * tick
    cost_points = (commission / point_value) + spread_points + 2 * slippage_points
    return {
        "round_turn_cost_points": cost_points,
        "round_turn_cost_usd": cost_points * point_value,
        "note": "add to the stop distance in points to get the break-even target",
    }


# --------------------------------------------------------------------------- helpers


def _by_family(labels: pl.DataFrame):
    if labels.is_empty():
        return []
    return [(f[0], g) for f, g in labels.group_by(["trigger_family"], maintain_order=True)]


def _finite(series: pl.Series) -> np.ndarray:
    values = series.to_numpy().astype(float)
    return values[np.isfinite(values)]


def _blocks(group: pl.DataFrame, metric: str) -> list:
    mask = np.isfinite(group[metric].to_numpy().astype(float))
    return [d for d, ok in zip(group["session_date"].to_list(), mask) if ok]


def _ci(values, groups, statistic, cfg: Config) -> Interval:
    return block_bootstrap(
        values,
        groups,
        statistic,
        resamples=cfg.get("statistics.bootstrap_resamples"),
        confidence=cfg.get("statistics.confidence_level"),
        seed=cfg.get("determinism.seed"),
    )
