"""Assemble the dashboard's data file from a pipeline run.

The site generator renders committed artefacts and computes nothing itself. That rule
is what keeps hosting free of the research pipeline, so the dashboard needs its numbers
handed to it: this module writes `research/dashboard.json`, and `reporting/dashboard.py`
reads it.

Deterministic by the same rules as the reports — no wall clock, no paths, fixed float
precision so a rebuild does not churn the diff.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import REPO_ROOT, Config

SUMMARY_PATH = REPO_ROOT / "research" / "dashboard.json"

BIN_WIDTH = 0.25
BIN_CAP = 4.0
TAIL_FOCUS = 2.0  # the threshold the headline reads from
STABILITY_GATE = 0.5


def _round(value: Any, places: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else round(float(value), places)
    return value


def _rows(frame: pl.DataFrame) -> list[dict]:
    return [{k: _round(v) for k, v in row.items()} for row in frame.iter_rows(named=True)]


def excursion_bins(values: np.ndarray) -> list[dict]:
    """Fixed-width bins with an overflow bucket.

    Fixed edges rather than data-driven ones: the reader is comparing this distribution
    to the stop distance, so the bins have to sit at round multiples of it, and they must
    not move when the sample changes.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    edges = np.arange(0.0, BIN_CAP + BIN_WIDTH, BIN_WIDTH)
    counts, _ = np.histogram(np.clip(finite, 0, BIN_CAP), bins=edges)
    bins = [
        {"lo": round(float(edges[i]), 2), "hi": round(float(edges[i + 1]), 2), "n": int(counts[i])}
        for i in range(len(counts))
    ]
    overflow = int((finite > BIN_CAP).sum())
    if overflow:
        bins.append({"lo": BIN_CAP, "hi": round(float(finite.max()), 2), "n": overflow})
    return bins


def build_summary(
    cfg: Config,
    *,
    gate,
    per_year: pl.DataFrame,
    labels: pl.DataFrame,
    tails: pl.DataFrame,
    by_year: pl.DataFrame,
    costs: dict,
    family,
    hypotheses: pl.DataFrame,
    clusters,
    diagnosis,
    sanity: str,
    readiness: str,
    trial_count: int,
    panel_rows: int,
    panel_sessions: int,
) -> dict:
    """Everything the dashboard needs, and nothing it does not."""
    metric = labels["mfe_stops"].to_numpy().astype(float)
    focus = tails.filter(pl.col("threshold") == TAIL_FOCUS)
    base_rate = float(focus["p"][0]) if not focus.is_empty() else float("nan")

    supported = hypotheses.filter(pl.col("supported")) if "supported" in hypotheses.columns else pl.DataFrame()
    survived = int(hypotheses["survives_bh"].sum()) if "survives_bh" in hypotheses.columns else 0

    # Annualise from sessions actually covered. Dividing by the number of calendar years
    # the sample touches turns a six-week slice into "a year" and understates every rate.
    sessions_per_year = 252
    scale = sessions_per_year / max(panel_sessions, 1)
    families = (
        per_year.group_by("trigger_family")
        .agg(pl.col("n").sum().alias("n"))
        .with_columns((pl.col("n") * scale).alias("per_year"))
        .sort("per_year", descending=True)
    )

    return {
        "schema": 1,
        "config_hash": cfg.hash[:12],
        "data_version": str(cfg.get("data.version")),
        "is_synthetic": cfg.is_synthetic,
        "instrument": cfg.get("scope.primary_instrument"),
        "period": f"{cfg.get('scope.study_start')} .. {cfg.get('scope.study_end')}",
        "resamples": cfg.get("statistics.bootstrap_resamples"),
        "minimum_effect": float(family.minimum_effect),
        "family_size": family.size,
        "trial_count": trial_count,
        "panel": {"events": panel_rows, "sessions": panel_sessions},
        "verdict": _verdict(cfg, supported.height, family, base_rate, clusters, diagnosis),
        "gate": {
            "passed": bool(gate.passed),
            "canaries_caught": sum(gate.canaries_caught.values()),
            "canaries_total": len(gate.canaries_caught),
            "features_clean": sum(1 for flagged in gate.clean_flagged.values() if not flagged),
            "features_total": len(gate.clean_flagged),
            "deterministic": bool(gate.deterministic and gate.derived_deterministic),
        },
        "phases": _phases(gate, per_year, labels, supported.height, family, clusters, readiness),
        "excursion": {
            "bins": excursion_bins(metric),
            "median": _round(float(np.nanmedian(metric))) if metric.size else None,
            "q90": _round(float(np.nanquantile(metric[np.isfinite(metric)], 0.9))) if metric.size else None,
            "emphasis_from": TAIL_FOCUS,
            "n": int(np.isfinite(metric).sum()),
        },
        "tails": _rows(tails),
        "by_year": _rows(by_year),
        "hypotheses": _rows(
            hypotheses.select(
                "id", "feature", "predicted_direction", "effect", "ci_low", "ci_high",
                "p_value", "n", "base_rate", "survives_bh", "supported", "note",
            )
        ),
        "hypotheses_survived_bh": survived,
        "families": _rows(families),
        "annualised_from_sessions": panel_sessions,
        "underpowered": families.filter(pl.col("per_year") < 150)["trigger_family"].to_list(),
        "clusters": _rows(clusters.profile.select("cluster", "n", "median_mfe_stops", "p_gt_2x", "p_gt_3x")),
        "cluster_stability": _round(clusters.stability_ari),
        "cluster_gate": STABILITY_GATE,
        "gbm": {
            "ic": _round(diagnosis.spearman_ic),
            "r2": _round(diagnosis.r2),
            "purged": diagnosis.purged,
            "embargoed": diagnosis.embargoed,
            "sanity": sanity,
            "importance": _rows(diagnosis.importance.head(5)),
        },
        "costs": {k: _round(v) if isinstance(v, float) else v for k, v in costs.items()},
        "base_rate_2x": _round(base_rate),
        "meta_labeling": readiness,
    }


def _verdict(cfg: Config, supported: int, family, base_rate: float, clusters, diagnosis) -> dict:
    """The one statement the dashboard leads with.

    Computed, not written. If a later run does find something, this changes on its own —
    a hand-written headline is a headline that goes stale silently.
    """
    if cfg.is_synthetic:
        return {
            "state": "not-a-result",
            "headline": "No tradeable edge — and none was possible",
            "detail": (
                "This run used a seeded random walk, not market data. Every null below is "
                "the correct answer for a tape with no structure in it, and is evidence "
                "the apparatus works rather than evidence about ES."
            ),
        }
    if supported == 0:
        return {
            "state": "null",
            "headline": "No tradeable edge found",
            "detail": (
                f"None of the {family.size} pre-registered hypotheses cleared both the "
                f"significance and the {family.minimum_effect}× effect bar. The forward "
                f"excursion distribution does not shift enough on any measured context to "
                f"pay for the risk taken."
            ),
        }
    return {
        "state": "finding",
        "headline": f"{supported} of {family.size} hypotheses supported",
        "detail": (
            "Treat this as provisional until it survives the holdout. An effect this size "
            "in intraday futures is at the edge of what is plausible; search for the leak "
            "before acting on it."
        ),
    }


def _phases(gate, per_year, labels, supported, family, clusters, readiness) -> list[dict]:
    """Pipeline state as a strip. Status carries an icon and a word, never colour alone."""
    return [
        {
            "n": 1, "name": "Foundation",
            "state": "pass" if gate.passed else "fail",
            "note": (
                f"{sum(gate.canaries_caught.values())}/{len(gate.canaries_caught)} leak canaries caught, "
                f"builds byte-identical"
            ),
        },
        {
            "n": 2, "name": "Events & labels", "state": "pass",
            "note": f"{labels.height:,} labelled events, {per_year['trigger_family'].n_unique()} families",
        },
        {"n": 3, "name": "Base rates", "state": "pass", "note": "distributions and intervals computed"},
        {
            "n": 4, "name": "Hypotheses",
            "state": "pass" if supported else "null",
            "note": f"{supported} of {family.size} supported",
        },
        {
            "n": 5, "name": "Structure",
            "state": "pass" if clusters.stable else "null",
            "note": f"clusters {'stable' if clusters.stable else 'unstable'} (ARI {clusters.stability_ari:.2f})",
        },
        {"n": 6, "name": "Meta-labeling", "state": "blocked", "note": readiness},
    ]


def write_summary(summary: dict, path: Path | None = None) -> Path:
    target = Path(path) if path else SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


