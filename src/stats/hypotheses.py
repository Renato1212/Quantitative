"""Pre-registered hypothesis testing (Phase 4).

Loads ``research/hypotheses/*.yaml`` — files written and committed before any conditional
analysis ran — and tests each one on the training block only.

Three properties this module enforces rather than merely encourages:

*Every hypothesis is reported.* :func:`test_family` returns a row per registered
hypothesis, pass or fail. There is no path through this code that drops one.

*The base rate travels with the conditional.* Each row carries the unconditional tail
probability for the same population, because a conditional number alone is unreadable
(C2).

*The effect must clear a pre-declared size, not merely a p-value.* A ratio whose interval
excludes 1.0 but sits below the registered minimum effect is recorded as significant and
too small, which is a different result from a finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.config import REPO_ROOT, Config
from src.stats.bootstrap import benjamini_hochberg, block_bootstrap, block_bootstrap_ratio, exceedance

HYPOTHESIS_DIR = REPO_ROOT / "research" / "hypotheses"


@dataclass(frozen=True)
class Hypothesis:
    id: str
    feature: str
    trigger_family: str
    direction: str
    threshold: float
    rationale: str
    falsified_if: str


@dataclass(frozen=True)
class Family:
    name: str
    registered: str
    alpha: float
    metric: str
    split: str
    minimum_effect: float
    hypotheses: tuple[Hypothesis, ...]
    source: Path

    @property
    def size(self) -> int:
        """The BH denominator. Distinct from the experiment log's trial count (D8)."""
        return len(self.hypotheses)


def load_families(directory: Path | None = None) -> list[Family]:
    out = []
    for path in sorted((directory or HYPOTHESIS_DIR).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append(
            Family(
                name=raw["family"],
                registered=str(raw["registered"]),
                alpha=float(raw["alpha"]),
                metric=raw["metric"],
                split=raw["split"],
                minimum_effect=float(raw["minimum_effect"]),
                hypotheses=tuple(Hypothesis(**h) for h in raw["hypotheses"]),
                source=path,
            )
        )
    return out


def test_family(family: Family, panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Test every hypothesis in a family on ``panel`` — the training block, joined.

    ``panel`` must carry the metric, the features, and ``session_date`` for blocking.
    """
    rows = [_test_one(h, family, panel, cfg) for h in family.hypotheses]
    table = pl.DataFrame(rows)
    if table.is_empty():
        return table

    reject, critical = benjamini_hochberg(table["p_value"].to_list(), family.alpha)
    return table.with_columns(
        pl.Series("survives_bh", reject),
        pl.lit(critical).alias("bh_critical_p"),
        pl.lit(family.size).alias("family_size"),
        pl.lit(family.name).alias("family"),
    ).with_columns(
        (
            pl.col("survives_bh")
            & (pl.col("effect") >= family.minimum_effect)
            & pl.col("direction_as_predicted")
        ).alias("supported")
    )


def _test_one(h: Hypothesis, family: Family, panel: pl.DataFrame, cfg: Config) -> dict:
    scope = panel if h.trigger_family == "all" else panel.filter(
        pl.col("trigger_family") == h.trigger_family
    )
    metric = family.metric
    usable = scope.filter(
        pl.col(metric).is_finite() & pl.col(h.feature).is_not_null() & pl.col(h.feature).is_finite()
    )

    base = _unconditional(usable, metric, h.threshold, cfg)
    row = {
        "id": h.id,
        "feature": h.feature,
        "trigger_family": h.trigger_family,
        "predicted_direction": h.direction,
        "threshold": h.threshold,
        "n": usable.height,
        "base_rate": base.point,
        "base_rate_ci_low": base.low,
        "base_rate_ci_high": base.high,
        "n_blocks": base.n_blocks,
    }

    if usable.height < 30:
        return row | {
            "effect": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "p_value": float("nan"), "direction_as_predicted": False,
            "n_high": 0, "n_low": 0, "note": "insufficient sample",
        }

    low_cut, high_cut = np.nanquantile(usable[h.feature].to_numpy().astype(float), [1 / 3, 2 / 3])
    high = usable.filter(pl.col(h.feature) >= high_cut)
    low = usable.filter(pl.col(h.feature) <= low_cut)

    # A "negative" hypothesis predicts the bottom tercile carries the fatter tail, so the
    # ratio is inverted before comparison. The pre-declared minimum effect is one-sided.
    if h.direction == "negative":
        high, low = low, high

    interval = block_bootstrap_ratio(
        high[metric].to_numpy().astype(float),
        low[metric].to_numpy().astype(float),
        high["session_date"].to_list(),
        low["session_date"].to_list(),
        exceedance(h.threshold),
        resamples=cfg.get("statistics.bootstrap_resamples"),
        confidence=cfg.get("statistics.confidence_level"),
        seed=cfg.get("determinism.seed"),
    )
    note = ""
    if not np.isfinite(interval.point):
        # The comparison tercile had no exceedances at all, so the ratio is undefined
        # rather than large. Reporting it as a big effect would be the easiest lie here.
        note = "undefined: comparison tercile had zero exceedances"
    return row | {
        "effect": interval.point,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "p_value": interval.p_value(1.0),
        "direction_as_predicted": bool(np.isfinite(interval.point) and interval.point > 1.0),
        "n_high": high.height,
        "n_low": low.height,
        "note": note,
    }


def _unconditional(frame: pl.DataFrame, metric: str, threshold: float, cfg: Config):
    return block_bootstrap(
        frame[metric].to_numpy().astype(float),
        frame["session_date"].to_list(),
        exceedance(threshold),
        resamples=cfg.get("statistics.bootstrap_resamples"),
        confidence=cfg.get("statistics.confidence_level"),
        seed=cfg.get("determinism.seed"),
    )


def registry() -> list[dict]:
    """Everything registered, for the report's appendix. Nothing is filtered out."""
    return [
        asdict(h) | {"family": family.name, "registered": family.registered}
        for family in load_families()
        for h in family.hypotheses
    ]
