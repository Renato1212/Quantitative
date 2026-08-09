"""Structure discovery (Phase 5): state clustering and gradient boosting for diagnosis.

Both techniques are deliberately narrow, and neither is used for prediction.

*Clustering* is fitted on context features with **no reference to outcomes**, and the
forward excursion distribution is attached afterwards. That ordering is the whole point:
it produces a vocabulary of market states that cannot have been contaminated by what
happened next. Clusters that do not survive a time split are noise and are reported as
noise — a cluster vocabulary that reorganises itself every year is not a vocabulary.

*Gradient boosting* is read for which context variables move the excursion distribution
and in what shape, not for its predictions. Shallow trees, strong regularisation, purged
cross-validation.

The metric is not R². On a heavy-tailed target, R² is dominated by a handful of
observations and swings across folds for reasons unconnected to whether the model learned
anything (spec review A11). Spearman rank IC and pinball loss at pre-declared quantiles
are the headline; R² is reported as a secondary line because ``CLAUDE.md`` §6 expects to
see it near zero, and it should be visible when it is.

The scaler and the cluster centres are fitted on training data only and applied unchanged
elsewhere. Fitting a scaler on the full panel is leak L3 with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.validation.purged import PurgedKFold

QUANTILES = (0.5, 0.9)


def complete_rows(panel: pl.DataFrame, features: list[str], metric: str) -> pl.DataFrame:
    """Rows where every feature and the target are finite.

    ``drop_nulls`` is not enough: features return NaN for an unfilled window, and in
    polars NaN is a value, not a null. Silently feeding those to a scaler produces a
    model fitted on a subset nobody chose.
    """
    condition = pl.col(metric).is_finite()
    for name in features:
        condition = condition & pl.col(name).is_not_null() & pl.col(name).is_finite()
    return panel.filter(condition)


@dataclass
class ClusterResult:
    k: int
    labels: np.ndarray
    profile: pl.DataFrame
    stability_ari: float
    feature_names: list[str] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        """Adjusted Rand above 0.5 between halves. Below that the vocabulary is noise."""
        return bool(np.isfinite(self.stability_ari) and self.stability_ari >= 0.5)


def cluster_states(
    panel: pl.DataFrame, features: list[str], cfg: Config, *, k: int = 4, metric: str = "mfe_stops"
) -> ClusterResult:
    """Cluster on context, then attach outcomes. Never the other way round."""
    usable = complete_rows(panel, features, metric)
    matrix = usable.select(features).to_numpy().astype(float)
    if matrix.shape[0] < k * 10:
        raise ValueError(f"{matrix.shape[0]} rows is too few for {k} clusters")

    seed = cfg.get("determinism.seed")
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled)
    assignments = model.labels_

    attached = usable.with_columns(pl.Series("cluster", assignments))
    profile = (
        attached.group_by("cluster")
        .agg(
            pl.len().alias("n"),
            pl.col("session_date").n_unique().alias("n_blocks"),
            pl.col(metric).median().alias("median_mfe_stops"),
            pl.col(metric).quantile(0.9).alias("q90_mfe_stops"),
            (pl.col(metric) > 2.0).mean().alias("p_gt_2x"),
            (pl.col(metric) > 3.0).mean().alias("p_gt_3x"),
            *[pl.col(f).mean().alias(f"mean_{f}") for f in features],
        )
        .sort("cluster")
    )

    return ClusterResult(
        k=k,
        labels=assignments,
        profile=profile,
        stability_ari=_stability(scaled, usable, k, seed),
        feature_names=features,
    )


def _stability(scaled: np.ndarray, panel: pl.DataFrame, k: int, seed: int) -> float:
    """Fit on the first half, fit on the second, compare on the overlap of assignments.

    Both models score the *whole* panel; the Rand index then asks whether the two halves
    of history agree about what the states are. Instability means the vocabulary is a
    property of the period, not of the market.
    """
    order = np.argsort(panel["t0"].to_numpy())
    half = order.size // 2
    if half < k * 5:
        return float("nan")
    first = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled[order[:half]])
    second = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled[order[half:]])
    return float(adjusted_rand_score(first.predict(scaled), second.predict(scaled)))


@dataclass
class GBMDiagnosis:
    folds: int
    spearman_ic: float
    pinball: dict[float, float]
    r2: float
    importance: pl.DataFrame
    purged: int
    embargoed: int

    def summary(self) -> str:
        pin = ", ".join(f"q{int(q * 100)}={v:.4f}" for q, v in sorted(self.pinball.items()))
        return (
            f"purged CV over {self.folds} folds ({self.purged} purged, {self.embargoed} embargoed)\n"
            f"  Spearman IC {self.spearman_ic:+.4f} | pinball {pin} | R2 {self.r2:+.4f}"
        )


def diagnose(
    panel: pl.DataFrame, features: list[str], cfg: Config, *, metric: str = "mfe_stops", folds: int = 5
) -> GBMDiagnosis:
    """Shallow, regularised GBM under purged CV. Read for shape, not for predictions."""
    from datetime import timedelta

    usable = complete_rows(panel, features, metric).sort("t0")
    X = usable.select(features).to_numpy().astype(float)
    y = usable[metric].to_numpy().astype(float)

    splitter = PurgedKFold(
        usable["t0"].to_numpy(),
        usable["window_end"].to_numpy(),
        n_splits=folds,
        embargo=timedelta(minutes=cfg.get("labels.time_barrier_minutes")),
    )

    predictions = np.full(y.shape, np.nan)
    purged = embargoed = 0
    for fold in splitter.split():
        if fold.train.size < 30:
            continue
        model = _model(cfg)
        model.fit(X[fold.train], y[fold.train])
        predictions[fold.test] = model.predict(X[fold.test])
        purged += fold.purged
        embargoed += fold.embargoed

    scored = np.isfinite(predictions)
    if scored.sum() < 30:
        raise ValueError("purging left too few scored samples to diagnose")

    ic = float(spearmanr(predictions[scored], y[scored]).statistic)
    pinball = {q: _pinball(y[scored], predictions[scored], q) for q in QUANTILES}
    r2 = float(r2_score(y[scored], predictions[scored]))

    full = _model(cfg).fit(X, y)
    perm = permutation_importance(
        full, X, y, n_repeats=10, random_state=cfg.get("determinism.seed"), scoring="r2"
    )
    importance = pl.DataFrame(
        {"feature": features, "importance": perm.importances_mean, "sd": perm.importances_std}
    ).sort("importance", descending=True)

    return GBMDiagnosis(folds, ic, pinball, r2, importance, purged, embargoed)


def _model(cfg: Config) -> HistGradientBoostingRegressor:
    """Deliberately weak. A model that can fit this data well has found a leak."""
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=150,
        learning_rate=0.03,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=cfg.get("determinism.seed"),
    )


def _pinball(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    delta = actual - predicted
    return float(np.mean(np.maximum(q * delta, (q - 1) * delta)))


def effect_size_sanity(r2: float) -> str:
    """§7's prior: genuine out-of-sample R² of 1-3% in intraday futures is a real finding.

    Anything substantially above that is a bug report, not a result, and the write-up says
    so at the point the number appears rather than in a footnote.
    """
    if r2 > 0.10:
        return (
            f"R2 of {r2:.3f} is far above the 1-3% prior for intraday futures. Treat this as a "
            "bug report: search for the leak before interpreting anything."
        )
    if r2 > 0.03:
        return f"R2 of {r2:.3f} is above the stated prior. Worth a leak hunt before believing it."
    if r2 > 0.0:
        return f"R2 of {r2:.3f} is within the 1-3% band where a real effect would live."
    return f"R2 of {r2:.3f} is at or below zero, which is the expected outcome and not a failure."
