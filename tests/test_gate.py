"""The Phase 1 gate: leakage canaries and byte-identical reruns.

``CLAUDE.md`` §6 asks for one canary that peeks a bar ahead. That proves only that the
detector catches the leak we already thought of, so the gate here requires one canary per
mechanism in the spec review's ranked list — and requires the clean features to come
through unflagged, because a detector that fires on everything is no detector.
"""

from __future__ import annotations

import pytest

from src import config as config_module
from src.experiment.log import ExperimentLog
from src.pipeline import anchors, build, manifest, run_gate
from src.validation import canaries
from src.validation.leakage import audit_all

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def audit(cfg, data_root):
    return audit_all(canaries.ALL, anchors(cfg, data_root, n=3), cfg, data_root)


@pytest.mark.parametrize("name", sorted(canaries.LEAKY))
def test_every_deliberate_leak_is_caught(audit, name):
    assert audit[name].leaks, audit[name].summary()


@pytest.mark.parametrize("name", sorted(canaries.CLEAN))
def test_no_clean_feature_is_flagged(audit, name):
    assert not audit[name].leaks, audit[name].summary()


def test_both_probes_contribute(audit):
    """Truncation and perturbation must each catch something the gate depends on."""
    probes = {f.probe for name in canaries.LEAKY for f in audit[name].findings}
    assert probes == {"truncation", "perturbation"}


def _probe(cfg, data_root, feature, probe):
    return audit_all({"f": feature}, anchors(cfg, data_root, n=2), cfg, data_root, probes=(probe,))["f"]


def test_truncation_catches_what_perturbation_cannot(cfg, data_root):
    """Perturbation scrambles float columns. A future *timestamp* slips straight past it."""
    import polars as pl

    from src.data.store import Store

    def reads_a_future_timestamp(view):
        bars = pl.read_parquet(Store.layout(Store.data_root(view.config), "bars_time"))
        ahead = bars.filter(pl.col("close_ts") > view.cutoff)
        if ahead.is_empty():
            return 0.0
        return float((ahead["close_ts"][0] - view.cutoff).total_seconds())

    assert _probe(cfg, data_root, reads_a_future_timestamp, "truncation").leaks
    assert not _probe(cfg, data_root, reads_a_future_timestamp, "perturbation").leaks


def test_perturbation_catches_what_truncation_cannot(cfg, data_root):
    """A future value entering through a clamp that usually binds.

    Truncation removes the future rows, the fallback returns the same clamped number,
    and nothing looks wrong. Perturbation pushes the future value past the clamp.
    """
    import polars as pl

    from src.data.store import Store

    clamp = 5000.0  # above every real price here, below every perturbed one

    def clamped_future_reader(view):
        bars = pl.read_parquet(Store.layout(Store.data_root(view.config), "bars_time"))
        ahead = bars.filter(pl.col("close_ts") > view.cutoff)
        if ahead.is_empty():
            return clamp
        return max(clamp, float(ahead["close"][0]))

    assert not _probe(cfg, data_root, clamped_future_reader, "truncation").leaks
    assert _probe(cfg, data_root, clamped_future_reader, "perturbation").leaks


def test_anchors_land_inside_rth_and_are_ordered(cfg, data_root):
    t0s = anchors(cfg, data_root, n=5)
    assert t0s == sorted(t0s)
    assert len(set(t0s)) == len(t0s)


def test_full_gate_passes(cfg, data_root, build_args, tmp_path):
    result = run_gate(cfg, data_root, rebuild_args=build_args)
    assert result.deterministic, result.report()
    assert all(result.canaries_caught.values()), result.report()
    assert not any(result.clean_flagged.values()), result.report()
    assert result.passed, result.report()


def test_rebuild_is_byte_identical(cfg, data_root, build_args, tmp_path):
    """C7. Same config plus same data yields the same bytes, not merely the same numbers."""
    again = tmp_path / "again"
    build(cfg, again, **build_args)
    assert manifest(again) == manifest(data_root)


def test_manifest_excludes_the_experiment_log(cfg, data_root):
    """The log records wall-clock time by design; it is a ledger, not an artefact."""
    assert not any("sqlite" in key for key in manifest(data_root))


def test_config_hash_is_stable_and_order_independent(tmp_path):
    cfg = config_module.load()
    assert cfg.hash == config_module.load().hash
    assert cfg.hash == cfg.with_overrides(**{"data.version": cfg.get("data.version")}).hash


def test_config_override_changes_the_hash(cfg):
    assert cfg.with_overrides(**{"determinism.seed": 1}).hash != cfg.hash


def test_config_override_preserves_dates(cfg):
    """A JSON round-trip would turn study_start into a string and break the calendar."""
    from datetime import date

    assert isinstance(cfg.with_overrides(**{"data.root": "x"}).get("scope.study_start"), date)


def test_experiment_log_counts_trials(cfg, tmp_path):
    with ExperimentLog(tmp_path / "runs.sqlite") as log:
        assert log.trial_count() == 0
        log.record(cfg, phase="phase1_gate", result={"passed": True})
        log.record(cfg, phase="phase4", result={"effect": 0.1}, hypothesis="h1")
        log.record(cfg, phase="phase4", result={"effect": 0.2}, hypothesis="h2")
        assert log.trial_count() == 3
        assert log.trial_count(phase="phase4") == 2
        assert log.family_size("phase4") == 2
        assert log.trial_count(exclude_synthetic=True) == 0  # this config is synthetic
        assert len(log.runs()) == 3


def test_build_refuses_to_pretend_synthetic_data_is_real(cfg, tmp_path):
    """There is no Rithmic ingest yet. Asking for real data must fail loudly."""
    real = cfg.with_overrides(**{"data.version": "rithmic-2026-08-01"})
    assert not real.is_synthetic
    with pytest.raises(NotImplementedError, match="Rithmic ingest is not written"):
        build(real, tmp_path / "real", ticks_per_session=10)
