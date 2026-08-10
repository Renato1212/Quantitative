"""The risk layer's ship gate, tested without fitting a model.

The gate is calibration, not discrimination, and the interesting cases are the ones where a
model looks good and should still be refused: a perfect ranker that is wrong about the
level, and a calibrator that fits noise because it was given a hundred positives.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayline.risk import survival
from bayline.risk.survival import ComponentModel


def model(**overrides) -> ComponentModel:
    base = dict(
        component_id="turbo",
        n_train=10_000,
        n_test=3_000,
        base_rate=0.02,
        auc=0.75,
        brier=0.018,
        brier_baseline=0.0196,
        calibration_error=0.01,
    )
    base.update(overrides)
    return ComponentModel(**base)


def test_a_perfect_ranker_with_the_wrong_level_is_refused():
    """AUC 0.95 and a 5x overstatement pulls five times too many trucks off the road.

    This is the failure the gate exists for. Discrimination is what a vendor demos;
    calibration is what makes the euro figure mean anything.
    """
    assert not model(auc=0.95, calibration_error=0.30).gate(0.06)


def test_a_model_that_cannot_beat_the_base_rate_is_refused():
    assert not model(brier=0.0196, brier_baseline=0.0196).gate(0.06)
    assert not model(brier=0.021, brier_baseline=0.0196).gate(0.06)


def test_a_calibrated_and_skilful_model_passes():
    assert model().gate(0.06)


def test_the_scorecard_reports_the_failure_rather_than_hiding_it():
    card = model(calibration_error=0.30).scorecard(0.06)
    assert card["passes_gate"] is False
    assert card["calibration_error"] == 0.30
    assert card["brier_skill"] is not None


# ------------------------------------------------------------------ calibrator choice


def test_scarce_positives_get_a_sigmoid_not_isotonic():
    """Isotonic on ~100 positives fits a step function to noise and degrades the Brier score
    below the base rate — which is exactly what happened on the first pass at a 0.7% brake
    failure rate. Two parameters cannot overfit that way."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 5_000)
    labels = (rng.uniform(0, 1, 5_000) < 0.02).astype(int)
    assert labels.sum() < survival.ISOTONIC_MIN_POSITIVES
    _, name = survival._make_calibrator(scores, labels)
    assert name == "sigmoid"


def test_abundant_positives_get_isotonic():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 20_000)
    labels = (rng.uniform(0, 1, 20_000) < scores).astype(int)
    assert labels.sum() >= survival.ISOTONIC_MIN_POSITIVES
    _, name = survival._make_calibrator(scores, labels)
    assert name == "isotonic"


def test_the_sigmoid_calibrator_recovers_a_known_distortion():
    """Scores inflated 4x should come back down to roughly the true rate."""
    rng = np.random.default_rng(7)
    truth = rng.uniform(0.005, 0.05, 30_000)
    labels = (rng.uniform(0, 1, 30_000) < truth).astype(int)
    inflated = np.clip(truth * 4.0, 1e-6, 0.999)
    calibrator = survival.Sigmoid().fit(inflated, labels)
    assert calibrator.predict(inflated).mean() == pytest.approx(labels.mean(), abs=0.01)


# ------------------------------------------------------------------ reliability table


def test_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    rng = np.random.default_rng(3)
    predicted = rng.uniform(0.01, 0.4, 40_000)
    actual = (rng.uniform(0, 1, 40_000) < predicted).astype(int)
    _, error = survival._calibration_table(predicted, actual, 10)
    assert error < 0.01


def test_calibration_error_catches_a_uniform_overstatement():
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.01, 0.2, 40_000)
    actual = (rng.uniform(0, 1, 40_000) < truth).astype(int)
    _, error = survival._calibration_table(np.clip(truth * 3, 0, 1), actual, 10)
    assert error > 0.06, "a 3x overstatement must not clear the 6% gate"


def test_quantile_bins_do_not_collapse_on_a_skewed_distribution():
    """Equal-width bins would put 95% of the mass in one bucket and call it calibrated."""
    rng = np.random.default_rng(11)
    predicted = rng.beta(0.4, 40, 30_000)
    actual = (rng.uniform(0, 1, 30_000) < predicted).astype(int)
    bins, _ = survival._calibration_table(predicted, actual, 10)
    assert len(bins) >= 8
    assert max(b["n"] for b in bins) < 0.3 * len(predicted)
