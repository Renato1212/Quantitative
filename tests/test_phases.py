"""Phases 2 to 6: triggers, statistics, hypothesis testing, discovery, meta-labeling.

The load-bearing checks here are the ones that protect against self-deception rather than
against crashes: that triggers are reproducible through the view, that block bootstrap
intervals are wider than i.i.d. ones, that every registered hypothesis appears in the
results table, and that the reports cannot lose their synthetic banner.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.data.point_in_time import PointInTimeView
from src.data.store import Store
from src.events.families import FAMILIES, build_events, events_per_year
from src.features.context import FEATURES, build_feature_matrix
from src.labels.atr import atr_table
from src.labels.triple_barrier import label_events, stop_distance_multiples
from src.models import discovery, meta_labeling
from src.reporting import reports
from src.stats import base_rates, hypotheses
from src.stats.bootstrap import benjamini_hochberg, block_bootstrap, exceedance

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def artefacts(cfg, data_root, calendar):
    bars = pl.read_parquet(Store.layout(data_root, "bars_time"))
    ticks = pl.read_parquet(Store.layout(data_root, "ticks"))
    schedule = pl.read_parquet(data_root / "events" / "roll_schedule.parquet")
    events = build_events(bars, cfg, calendar, schedule)
    atr = atr_table(bars, cfg)
    labels = stop_distance_multiples(label_events(events, ticks, atr, cfg), cfg)
    return {"bars": bars, "ticks": ticks, "events": events, "atr": atr, "labels": labels}


@pytest.fixture(scope="module")
def panel(cfg, data_root, calendar, artefacts):
    store = Store.from_config(cfg, root=data_root)
    try:
        features = build_feature_matrix(artefacts["events"], store, cfg, calendar)
    finally:
        store.close()
    return artefacts["labels"].join(features, on=["trigger_family", "t0"], how="inner")


# --------------------------------------------------------------------------- phase 2


def test_every_family_fires(artefacts):
    fired = set(artefacts["events"]["trigger_family"].unique())
    assert len(fired) >= len(FAMILIES), f"only {fired} fired"


def test_events_are_unique_per_family_and_anchor(artefacts):
    """The §5 events contract: no duplicate (trigger_family, t0)."""
    events = artefacts["events"]
    assert events.unique(subset=["trigger_family", "t0"]).height == events.height


def test_anchors_fall_inside_a_session(artefacts, calendar):
    """The §5 events contract: t0 must land in a valid RTH window."""
    windows = calendar.frame().select("session_date", "rth_open", "rth_close")
    joined = artefacts["events"].join(windows, on="session_date", how="left")
    assert joined["rth_open"].is_not_null().all()
    assert ((joined["t0"] > joined["rth_open"]) & (joined["t0"] <= joined["rth_close"])).all()


def test_every_event_is_reproducible_through_the_view(cfg, data_root, calendar, artefacts):
    """Triggers are computed bar-wise for speed; this is what makes that safe.

    A sample of anchors is re-derived through ``PointInTimeView``: the triggering bar must
    be visible at its own anchor, and the condition must hold on data the view will return.
    """
    store = Store.from_config(cfg, root=data_root)
    try:
        sample = artefacts["events"].filter(pl.col("trigger_family") == "prior_day_high").head(5)
        assert sample.height > 0
        for t0, session_date in zip(sample["t0"], sample["session_date"]):
            view = PointInTimeView(store, t0, cfg, calendar)
            visible = view.bars("time", since=view.current_session().eth_open)
            triggering = visible.filter(pl.col("close_ts") == t0)
            assert triggering.height == 1, "the anchor's own bar must be visible at the anchor"

            prior = view.prior_sessions(1)[-1]
            previous = view.bars("time", since=prior.eth_open).filter(
                pl.col("in_rth") & (pl.col("session_date") == prior.session_date)
            )
            assert float(triggering["high"][0]) >= float(previous["high"].max())
    finally:
        store.close()


def test_no_event_precedes_its_own_session(artefacts):
    assert (artefacts["events"]["t0"].dt.date() == artefacts["events"]["session_date"]).all()


def test_events_per_year_covers_every_family(artefacts):
    table = events_per_year(artefacts["events"])
    assert set(table["trigger_family"]) == set(artefacts["events"]["trigger_family"])
    assert (table["n"] > 0).all()


def test_labels_carry_a_window_end_for_purging(artefacts):
    labels = artefacts["labels"]
    assert "window_end" in labels.columns
    assert (labels["window_end"] >= labels["t0"]).all()


# --------------------------------------------------------------------------- features


def test_feature_matrix_is_keyed_by_family_and_anchor(panel, artefacts):
    assert panel.height <= artefacts["events"].height
    assert set(FEATURES).issubset(panel.columns)


def test_feature_matrix_is_deterministic(cfg, data_root, calendar, artefacts):
    store = Store.from_config(cfg, root=data_root)
    try:
        sample = artefacts["events"].head(20)
        first = build_feature_matrix(sample, store, cfg, calendar)
        second = build_feature_matrix(sample, store, cfg, calendar)
    finally:
        store.close()
    assert first.equals(second)


def test_features_are_not_all_nan(panel):
    """A feature that never computes is a feature nobody will notice is broken."""
    for name in FEATURES:
        finite = np.isfinite(panel[name].to_numpy().astype(float)).sum()
        assert finite > 0, f"{name} produced no finite values"


# --------------------------------------------------------------------------- statistics


def test_block_bootstrap_is_wider_than_iid_when_events_cluster(cfg):
    """The reason blocking exists (spec review W2).

    Values are perfectly correlated within a day and independent across days. An i.i.d.
    bootstrap sees 200 observations; a blocked one sees 20 days, and says so.
    """
    rng = np.random.default_rng(0)
    per_day = rng.normal(size=20)
    values = np.repeat(per_day, 10)
    days = np.repeat(np.arange(20), 10)

    blocked = block_bootstrap(values, days, np.mean, resamples=2000, seed=1)
    iid = block_bootstrap(values, np.arange(values.size), np.mean, resamples=2000, seed=1)

    assert blocked.n_blocks == 20
    assert iid.n_blocks == 200
    assert (blocked.high - blocked.low) > 2 * (iid.high - iid.low)


def test_interval_reports_both_sample_sizes(cfg):
    values = np.arange(50, dtype=float)
    interval = block_bootstrap(values, np.repeat(np.arange(10), 5), np.mean, resamples=500)
    assert interval.n == 50
    assert interval.n_blocks == 10


def test_exceedance_statistic():
    assert exceedance(2.0)(np.array([1.0, 3.0, 5.0])) == pytest.approx(2 / 3)


def test_benjamini_hochberg_is_less_strict_than_bonferroni():
    p = [0.001, 0.008, 0.02, 0.04, 0.3]
    reject, critical = benjamini_hochberg(p, alpha=0.05)
    assert reject.sum() >= sum(x <= 0.05 / len(p) for x in p)
    assert not reject[-1]
    assert 0 < critical <= 0.05


def test_benjamini_hochberg_rejects_nothing_when_all_null():
    reject, _ = benjamini_hochberg([0.4, 0.6, 0.9], alpha=0.05)
    assert not reject.any()


def test_tail_table_reports_intervals_and_block_counts(cfg, panel):
    table = base_rates.tail_table(panel, cfg)
    assert not table.is_empty()
    assert {"p", "ci_low", "ci_high", "n", "n_blocks", "n_exceeding"}.issubset(table.columns)
    assert (table["ci_low"] <= table["p"] + 1e-9).all()
    assert (table["ci_high"] >= table["p"] - 1e-9).all()
    assert (table["n_blocks"] <= table["n"]).all()


def test_tail_probabilities_are_monotone_in_the_threshold(cfg, panel):
    table = base_rates.tail_table(panel, cfg).sort("trigger_family", "threshold")
    for _, group in table.group_by("trigger_family"):
        probabilities = group.sort("threshold")["p"].to_list()
        assert probabilities == sorted(probabilities, reverse=True)


def test_cost_floor_is_positive(cfg):
    costs = base_rates.cost_floor(cfg)
    assert costs["round_turn_cost_points"] > 0
    assert costs["round_turn_cost_usd"] > 0


# --------------------------------------------------------------------------- phase 4


def test_every_registered_hypothesis_appears_in_the_results(cfg, panel):
    """C6 and the Phase 4 gate: no path through the code drops a hypothesis."""
    family = hypotheses.load_families()[0]
    results = hypotheses.test_family(family, panel, cfg)
    assert results.height == family.size
    assert set(results["id"]) == {h.id for h in family.hypotheses}


def test_conditional_results_carry_their_base_rate(cfg, panel):
    """C2: a conditional statistic is never reported alone."""
    family = hypotheses.load_families()[0]
    results = hypotheses.test_family(family, panel, cfg)
    assert {"base_rate", "base_rate_ci_low", "base_rate_ci_high"}.issubset(results.columns)


def test_support_requires_effect_size_not_only_significance(cfg, panel):
    family = hypotheses.load_families()[0]
    results = hypotheses.test_family(family, panel, cfg)
    for row in results.iter_rows(named=True):
        if row["supported"]:
            assert row["survives_bh"]
            assert row["effect"] >= family.minimum_effect


def test_family_size_is_not_the_trial_count(cfg, panel, tmp_path):
    """Decision D8: two denominators, reported separately."""
    from src.experiment.log import ExperimentLog

    family = hypotheses.load_families()[0]
    with ExperimentLog(tmp_path / "runs.sqlite") as log:
        for h in family.hypotheses:
            log.record(cfg, phase="phase4", hypothesis=h.id, result={})
        log.record(cfg, phase="phase2", result={})
        assert log.family_size("phase4") == family.size
        assert log.trial_count() > family.size


def test_hypothesis_registry_lists_everything(cfg):
    entries = hypotheses.registry()
    assert entries
    assert all({"id", "feature", "direction", "falsified_if"} <= set(e) for e in entries)


# --------------------------------------------------------------------------- phase 5


def test_clustering_never_sees_the_outcome(cfg, panel):
    """Swapping the outcome column must not change a single cluster assignment."""
    features = sorted(FEATURES)
    scrambled = panel.with_columns(
        pl.Series("mfe_stops", np.random.default_rng(7).permutation(panel["mfe_stops"].to_numpy()))
    )
    a = discovery.cluster_states(panel, features, cfg)
    b = discovery.cluster_states(scrambled, features, cfg)
    assert np.array_equal(a.labels, b.labels)


def test_cluster_profile_attaches_the_excursion_distribution(cfg, panel):
    result = discovery.cluster_states(panel, sorted(FEATURES), cfg)
    assert {"n", "median_mfe_stops", "p_gt_2x", "p_gt_3x"}.issubset(result.profile.columns)
    assert result.profile["n"].sum() > 0


def test_gbm_uses_purged_folds(cfg, panel):
    diagnosis = discovery.diagnose(panel, sorted(FEATURES), cfg)
    assert diagnosis.purged > 0, "purging removed nothing — the embargo is misconfigured"
    assert set(diagnosis.pinball) == set(discovery.QUANTILES)
    assert diagnosis.importance.height == len(FEATURES)


def test_effect_size_sanity_flags_implausible_r2():
    assert "bug report" in discovery.effect_size_sanity(0.4)
    assert "leak hunt" in discovery.effect_size_sanity(0.06)
    assert "expected outcome" in discovery.effect_size_sanity(-0.01)


# --------------------------------------------------------------------------- phase 6


def test_meta_labeling_refuses_to_fit_without_trades(cfg):
    empty = meta_labeling.load_trade_log(None)
    assert empty.is_empty()
    assert "No trade log" in meta_labeling.readiness(empty)
    with pytest.raises(NotImplementedError, match="Fitting now"):
        meta_labeling.fit(empty, pl.DataFrame(), cfg)


def test_calibration_detects_a_miscalibrated_model(cfg):
    rng = np.random.default_rng(3)
    predicted = np.full(400, 0.9)
    realised = (rng.random(400) < 0.3).astype(float)  # promises 90%, delivers 30%
    days = np.repeat(np.arange(40), 10)
    result = meta_labeling.calibration_curve(predicted, realised, days, cfg, bins=5)
    assert not result.calibrated
    assert "NOT CALIBRATED" in result.summary()


def test_calibration_accepts_a_calibrated_model(cfg):
    rng = np.random.default_rng(11)
    predicted = rng.uniform(0.05, 0.95, 2000)
    realised = (rng.random(2000) < predicted).astype(float)
    days = np.repeat(np.arange(200), 10)
    assert meta_labeling.calibration_curve(predicted, realised, days, cfg, bins=5).calibrated


def test_size_multiplier_is_continuous_and_capped():
    base = 0.3
    assert meta_labeling.size_multiplier(base, base_rate=base) == pytest.approx(1.0)
    assert meta_labeling.size_multiplier(0.6, base_rate=base) > 1.0
    assert meta_labeling.size_multiplier(0.1, base_rate=base) < 1.0
    assert meta_labeling.size_multiplier(0.99, base_rate=base) <= 2.0


# --------------------------------------------------------------------------- reporting


def test_synthetic_runs_always_carry_the_banner(cfg):
    assert cfg.is_synthetic
    assert "SYNTHETIC DATA" in reports.banner(cfg)
    assert reports.banner(cfg.with_overrides(**{"data.version": "rithmic-2026-08"})) == ""


def test_reports_contain_no_wall_clock(cfg):
    """C7: a report that embeds the time it was generated can never be byte-identical."""
    from pathlib import Path

    import re

    for path in Path("research/reports").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\d{2}:\d{2}:\d{2}", text), f"{path} embeds a timestamp"


def test_markdown_table_handles_an_empty_frame():
    assert reports.markdown_table(pl.DataFrame()) == "_no rows_"
