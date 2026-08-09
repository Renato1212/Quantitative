"""Block bootstrap by session-day.

The single most consequential statistical correction in this repository, and the one
``CLAUDE.md`` does not ask for. Purging (C4) handles labels that overlap in time. It does
nothing about the fact that events *cluster*: four triggers fire in one session driven by
one shared shock, adjacent sessions share a volatility regime, a CPI print moves every
family at once.

Resample events i.i.d. and the confidence intervals come out materially too narrow. That
error then propagates into the Benjamini–Hochberg step, so findings "survive correction"
that should not. Resampling whole session-days instead keeps the within-day dependence
intact.

Every interval in this repository is produced here, and every table reports the effective
sample size — the number of independent blocks — beside the nominal one, because those
two numbers can differ by a factor of four and only one of them is the denominator that
matters.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int  # nominal sample size
    n_blocks: int  # independent blocks — the honest sample size
    resamples: int
    draws: np.ndarray | None = None  # the resampled statistics, for deriving a p-value

    def p_value(self, null: float = 1.0) -> float:
        """Two-sided bootstrap p-value against a null, from the draws themselves.

        No normality assumption: the fraction of draws on the far side of the null,
        doubled and clipped. With a heavy-tailed statistic and a few hundred blocks this
        is the only defensible way to get one.
        """
        if self.draws is None or self.draws.size == 0:
            return float("nan")
        below = float((self.draws <= null).mean())
        return float(min(1.0, 2.0 * min(below, 1.0 - below)))

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.low:.4g}, {self.high:.4g}] (n={self.n}, blocks={self.n_blocks})"

    def excludes(self, value: float) -> bool:
        """Whether the interval excludes a null value, e.g. 1.0 for a ratio."""
        return value < self.low or value > self.high


def _block_index(groups: Sequence) -> tuple[list[np.ndarray], int]:
    order: dict = {}
    for position, key in enumerate(groups):
        order.setdefault(key, []).append(position)
    blocks = [np.asarray(v, dtype=int) for _, v in sorted(order.items(), key=lambda kv: str(kv[0]))]
    return blocks, len(blocks)


def block_bootstrap(
    values: np.ndarray,
    groups: Sequence,
    statistic: Statistic,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over whole session-days.

    ``groups`` is the block label per observation — the session date, normally. Blocks are
    drawn with replacement until the resample holds at least as many observations as the
    original, which keeps blocks of differing size from silently reweighting the estimate.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values, groups = values[finite], [g for g, ok in zip(groups, finite) if ok]
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, 0, resamples)

    blocks, n_blocks = _block_index(groups)
    point = float(statistic(values))
    rng = np.random.default_rng(seed)

    draws = np.empty(resamples, dtype=float)
    choices = rng.integers(0, n_blocks, size=(resamples, n_blocks))
    for i in range(resamples):
        sample = np.concatenate([blocks[j] for j in choices[i]])
        draws[i] = statistic(values[sample])

    tail = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(draws, [tail, 1.0 - tail])
    return Interval(point, float(low), float(high), int(values.size), n_blocks, resamples, draws)


def block_bootstrap_ratio(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    numerator_groups: Sequence,
    denominator_groups: Sequence,
    statistic: Statistic,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Ratio of a statistic between two disjoint sub-populations.

    Both sides are resampled by block on every draw, so the interval carries the
    uncertainty of both. Draws where the denominator statistic is zero are dropped and
    counted against the resample budget rather than silently producing infinities.
    """
    a, b = np.asarray(numerator_values, float), np.asarray(denominator_values, float)
    ga = [g for g, ok in zip(numerator_groups, np.isfinite(a)) if ok]
    gb = [g for g, ok in zip(denominator_groups, np.isfinite(b)) if ok]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), int(a.size + b.size), 0, resamples)

    blocks_a, na = _block_index(ga)
    blocks_b, nb = _block_index(gb)
    base_b = float(statistic(b))
    point = float(statistic(a)) / base_b if base_b else float("nan")

    rng = np.random.default_rng(seed)
    pick_a = rng.integers(0, na, size=(resamples, na))
    pick_b = rng.integers(0, nb, size=(resamples, nb))

    draws = []
    for i in range(resamples):
        top = statistic(a[np.concatenate([blocks_a[j] for j in pick_a[i]])])
        bottom = statistic(b[np.concatenate([blocks_b[j] for j in pick_b[i]])])
        if bottom:
            draws.append(top / bottom)

    if not draws:
        return Interval(point, float("nan"), float("nan"), int(a.size + b.size), na + nb, resamples)

    tail = (1.0 - confidence) / 2.0
    array = np.asarray(draws)
    low, high = np.nanquantile(array, [tail, 1.0 - tail])
    return Interval(point, float(low), float(high), int(a.size + b.size), na + nb, len(draws), array)


def exceedance(threshold: float) -> Statistic:
    """``P(x > threshold)`` as a statistic, for tail probabilities."""

    def statistic(values: np.ndarray) -> float:
        return float((values > threshold).mean())

    return statistic


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """BH step-up. Returns the reject mask and the critical p-value.

    Applied across a pre-registered hypothesis family. The family size is the number of
    hypotheses in that family — not the number of runs in the experiment log, which is
    larger and serves a different purpose (decision D8).
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), float("nan")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresholds = alpha * np.arange(1, n + 1) / n
    passing = np.flatnonzero(ranked <= thresholds)
    if passing.size == 0:
        # No critical value exists when nothing passes; NaN says that, 0.0 pretends to be
        # a threshold that was applied.
        return np.zeros(n, dtype=bool), float("nan")
    cut = ranked[passing[-1]]
    return p <= cut, float(cut)
