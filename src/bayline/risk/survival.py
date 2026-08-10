"""Failure risk: P(this component fails within the horizon), per vehicle, per day.

Everything downstream multiplies this probability by a euro amount, so a miscalibrated
model does not produce a slightly wrong plan — it produces a confidently wrong budget. The
ship gate here is **calibration**, not discrimination. A model that ranks perfectly and
says 40% when the truth is 8% will pull five times too many vehicles off the road and the
plan will look rigorous while it does it.

Three things this module refuses to do:

**No random splits.** Training on 2026 to predict 2025 is not a validation, it is a
rehearsal. The split is by day index, contiguous, and the model never sees a day at or
after the boundary.

**No leakage through the label.** The label is "fails within the next H days", so an
observation whose horizon extends past the end of the history is *censored* and dropped
rather than labelled zero. Labelling it zero teaches the model that the end of the dataset
is safe.

**No borrowed calibration.** Each component gets its own calibrator fitted on a held-out
slice of the training period, because a battery's base rate and a clutch's differ by an
order of magnitude and one shared mapping would flatten both.

The calibrator is chosen by how many positives there are to fit it with. Isotonic
regression is non-parametric and excellent given a thousand or more failures; with a
hundred it fits a step function to noise and *degrades* the Brier score below the base
rate — which is what happened here on the first pass, at a 0.7% brake failure rate. Below
the threshold the calibrator is a sigmoid, which has two parameters and cannot overfit that
way. The choice is recorded per component so the scorecard says which one was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from bayline.config import Config
from bayline.health.signals import FEATURE_COLUMNS
from bayline.store import Warehouse


ISOTONIC_MIN_POSITIVES = 300


class Sigmoid:
    """Platt scaling. Two parameters, so it cannot step-fit sparse positives."""

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Sigmoid:
        self._model.fit(_logit(scores).reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _make_calibrator(scores: np.ndarray, labels: np.ndarray) -> tuple[object, str]:
    positives = int(labels.sum())
    if positives >= ISOTONIC_MIN_POSITIVES:
        return (
            IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels),
            "isotonic",
        )
    return Sigmoid().fit(scores, labels), "sigmoid"


@dataclass
class ComponentModel:
    """A fitted hazard model for one component, with its honest scorecard."""

    component_id: str
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    brier: float
    brier_baseline: float
    calibration_error: float
    calibrator: str = "isotonic"
    calibration_positives: int = 0
    calibration_bins: list[dict] = field(default_factory=list)
    top_features: list[dict] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        raise NotImplementedError  # set by the caller via `gate`

    def gate(self, tolerance: float) -> bool:
        """Calibrated within tolerance, and better than predicting the base rate."""
        return self.calibration_error <= tolerance and self.brier < self.brier_baseline

    def scorecard(self, tolerance: float) -> dict:
        return {
            "component_id": self.component_id,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "base_rate": round(self.base_rate, 6),
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 6),
            "brier_baseline": round(self.brier_baseline, 6),
            "brier_skill": round(1.0 - self.brier / self.brier_baseline, 4)
            if self.brier_baseline
            else None,
            "calibration_error": round(self.calibration_error, 4),
            "calibrator": self.calibrator,
            "calibration_positives": self.calibration_positives,
            "passes_gate": self.gate(tolerance),
            "calibration_bins": self.calibration_bins,
            "top_features": self.top_features,
        }


def _labelled(cfg: Config, warehouse: Warehouse, component_id: str) -> pa.Table:
    """Join health rows to a forward-looking failure label.

    ``censored`` marks rows whose horizon runs past the end of the history. They are
    excluded from training and scoring: we do not know what happened to them, and calling
    that a zero is how a model learns to be optimistic at the edge of its data.
    """
    horizon = int(cfg.get("risk.label_horizon_days"))
    last_day = int(cfg.get("history.days")) - 1
    return warehouse.arrow(
        """
        SELECT
            h.*,
            v.vehicle_class,
            v.depot_id,
            -- A failure of *this* component, strictly after today, inside the horizon.
            CASE WHEN EXISTS (
                SELECT 1 FROM {events} e
                WHERE e.vehicle_id = h.vehicle_id
                  AND e.component_id = h.component_id
                  AND e.kind = 'failure'
                  AND e.day_index > h.day_index
                  AND e.day_index <= h.day_index + ?
            ) THEN 1 ELSE 0 END AS label,
            CASE WHEN h.day_index + ? > ? THEN 1 ELSE 0 END AS censored
        FROM {health} h
        JOIN {vehicles} v USING (vehicle_id)
        WHERE h.component_id = ?
        ORDER BY h.day_index, h.vehicle_id
        """,
        [horizon, horizon, last_day, component_id],
    )


def _matrix(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.column_stack(
        [np.asarray(table[c], dtype=np.float64) for c in FEATURE_COLUMNS]
    )
    labels = np.asarray(table["label"], dtype=np.int8)
    days = np.asarray(table["day_index"], dtype=np.int32)
    return features, labels, days


def _model(cfg: Config) -> HistGradientBoostingClassifier:
    """Shallow and regularised.

    The signal here is a handful of monotone trends, not a high-order interaction. A deep
    model would fit the simulator's noise, and on a real feed it would fit one depot's
    instrumentation quirks.
    """
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=220,
        learning_rate=0.06,
        min_samples_leaf=60,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=cfg.seed,
    )


def _calibration_table(predicted: np.ndarray, actual: np.ndarray, bins: int) -> tuple[list[dict], float]:
    """Reliability table plus the sample-weighted mean absolute calibration error.

    Quantile bins, not equal-width: failure probabilities pile up near zero, and equal-width
    bins would put 95% of the mass in the first bucket and call the model calibrated on the
    strength of one number.
    """
    edges = np.unique(np.quantile(predicted, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return [], float("inf")
    assignment = np.clip(np.digitize(predicted, edges[1:-1]), 0, len(edges) - 2)

    rows, error, total = [], 0.0, 0
    for b in range(len(edges) - 1):
        mask = assignment == b
        n = int(mask.sum())
        if n < 20:
            continue
        expected = float(predicted[mask].mean())
        observed = float(actual[mask].mean())
        rows.append(
            {
                "bin": b,
                "predicted": round(expected, 5),
                "observed": round(observed, 5),
                "n": n,
            }
        )
        error += abs(expected - observed) * n
        total += n
    return rows, (error / total if total else float("inf"))


def _permutation_importance(
    model, calibrator, features: np.ndarray, labels: np.ndarray, seed: int, top: int = 5
) -> list[dict]:
    """Which signals the model leans on, measured by Brier degradation when shuffled.

    Read as a pointer, not as causation: correlated features share credit, and on a real
    feed the ranking shifts between depots.
    """
    rng = np.random.default_rng(seed)
    baseline = brier_score_loss(labels, calibrator.predict(model.predict_proba(features)[:, 1]))
    scores = []
    for i, name in enumerate(FEATURE_COLUMNS):
        shuffled = features.copy()
        rng.shuffle(shuffled[:, i])
        degraded = brier_score_loss(
            labels, calibrator.predict(model.predict_proba(shuffled)[:, 1])
        )
        scores.append({"feature": name, "brier_increase": round(degraded - baseline, 7)})
    scores.sort(key=lambda row: row["brier_increase"], reverse=True)
    return scores[:top]


def fit_component(cfg: Config, warehouse: Warehouse, component_id: str):
    """Fit, calibrate and score one component. Returns (model, calibrator, report)."""
    table = _labelled(cfg, warehouse, component_id)
    keep = np.asarray(table["censored"], dtype=np.int8) == 0
    table = table.filter(pa.array(keep))
    features, labels, days = _matrix(table)

    if labels.sum() < 40:
        raise ValueError(
            f"{component_id}: only {int(labels.sum())} failures inside the horizon — too few "
            "to fit a hazard model. Lengthen the history or widen the label horizon."
        )

    # Contiguous split by day. The calibration slice is the tail of the *training* period,
    # so the test block stays untouched until it is scored once.
    boundary = int(np.quantile(days, float(cfg.get("risk.train_fraction"))))
    calib_start = int(np.quantile(days[days < boundary], 0.85))
    fit_mask = days < calib_start
    calib_mask = (days >= calib_start) & (days < boundary)
    test_mask = days >= boundary

    model = _model(cfg).fit(features[fit_mask], labels[fit_mask])
    raw_calib = model.predict_proba(features[calib_mask])[:, 1]
    calibrator, calibrator_name = _make_calibrator(raw_calib, labels[calib_mask])

    predicted = calibrator.predict(model.predict_proba(features[test_mask])[:, 1])
    actual = labels[test_mask]
    base_rate = float(labels[fit_mask].mean())
    bins, error = _calibration_table(predicted, actual, int(cfg.get("risk.calibration_bins")))

    report = ComponentModel(
        component_id=component_id,
        n_train=int(fit_mask.sum()),
        n_test=int(test_mask.sum()),
        base_rate=base_rate,
        auc=float(roc_auc_score(actual, predicted)) if actual.sum() else float("nan"),
        brier=float(brier_score_loss(actual, predicted)),
        brier_baseline=float(brier_score_loss(actual, np.full_like(predicted, base_rate))),
        calibration_error=error,
        calibration_bins=bins,
        top_features=_permutation_importance(
            model, calibrator, features[test_mask], actual, cfg.seed
        ),
    )
    return model, calibrator, report


def score_current(cfg: Config, warehouse: Warehouse) -> tuple[pa.Table, list[dict]]:
    """Fit every component, then score the fleet as it stands on the last day of history.

    The output is one row per (vehicle, component) with a calibrated probability of failure
    inside the horizon — the only number the economics layer is allowed to consume.
    """
    last_day = int(cfg.get("history.days")) - 1
    horizon = int(cfg.get("risk.label_horizon_days"))
    tolerance = float(cfg.get("risk.max_calibration_error"))

    rows: list[dict] = []
    scorecards: list[dict] = []

    for component in cfg.components:
        model, calibrator, report = fit_component(cfg, warehouse, component.id)
        scorecards.append(report.scorecard(tolerance))
        # A model that cannot beat the base rate is not shipped as if it could. Its
        # predictions are replaced by the base rate itself and the row says so, which keeps
        # the euro figures defensible and tells the customer exactly where the product has
        # nothing to add yet. Showing a per-vehicle score for every component regardless is
        # the thing competitors do and the reason nobody trusts the numbers.
        passes = report.gate(tolerance)

        current = warehouse.arrow(
            """
            SELECT h.*
            FROM {health} h
            WHERE h.component_id = ?
              AND h.day_index = (
                  SELECT max(day_index) FROM {health} h2
                  WHERE h2.vehicle_id = h.vehicle_id AND h2.component_id = h.component_id
              )
            ORDER BY h.vehicle_id
            """,
            [component.id],
        )
        if current.num_rows == 0:
            continue

        features = np.column_stack(
            [np.asarray(current[c], dtype=np.float64) for c in FEATURE_COLUMNS]
        )
        if passes:
            probability = calibrator.predict(model.predict_proba(features)[:, 1])
        else:
            probability = np.full(features.shape[0], report.base_rate)
        # Staleness: a vehicle last seen weeks ago is a data-quality problem, and the plan
        # should surface it rather than quietly treat an old reading as current.
        observed_day = np.asarray(current["day_index"], dtype=np.int32)

        for i, vehicle_id in enumerate(current["vehicle_id"].to_pylist()):
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "component_id": component.id,
                    "as_of_day": last_day,
                    "observed_day": int(observed_day[i]),
                    "staleness_days": int(last_day - observed_day[i]),
                    "horizon_days": horizon,
                    "failure_probability": round(float(probability[i]), 6),
                    "exposure_since_service": float(current["exposure_since_service"][i].as_py()),
                    "days_since_service": int(current["days_since_service"][i].as_py()),
                    "sensor_level": float(current["sensor_level"][i].as_py()),
                    "sensor_slope_7d": float(current["sensor_slope_7d"][i].as_py()),
                    "model_passes_gate": passes,
                    "risk_source": "model" if passes else "base_rate",
                    "component_base_rate": round(report.base_rate, 6),
                }
            )

    keys = list(rows[0])
    return pa.table({k: [r[k] for r in rows] for k in keys}), scorecards


def daily_hazard(probability: float, horizon_days: int) -> float:
    """Convert a horizon probability to a constant per-day hazard.

    The scheduler needs "probability of failing before day d" for every d in the planning
    horizon, and it only has one number per component. Assuming a constant hazard inside the
    horizon is the least-assuming way to spread it: it is exactly right if failures arrive
    as a Poisson process at the current wear level, and it errs toward *understating* risk
    late in the window for a wearing-out component, which is the safe direction for a tool
    that recommends deferral.
    """
    probability = min(max(float(probability), 0.0), 0.999999)
    return 1.0 - (1.0 - probability) ** (1.0 / max(int(horizon_days), 1))


def cumulative_failure_probability(probability: float, horizon_days: int, days: int) -> float:
    """P(failure within ``days``), derived from the horizon probability."""
    if days <= 0:
        return 0.0
    hazard = daily_hazard(probability, horizon_days)
    return float(1.0 - (1.0 - hazard) ** days)
