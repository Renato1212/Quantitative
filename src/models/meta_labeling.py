"""Meta-labeling interface (Phase 6). Built, and deliberately unfed.

The principal's discretionary read is the primary model: it supplies direction and entry.
This secondary model estimates ``P(this trade reaches its target before its stop | context)``.
It does not predict the market. It calibrates the principal's own signal population, which
is a much smaller and more tractable question.

The output feeds **position sizing**, not a binary filter. A filter throws away the
principal's edge on the trades it vetoes; a size multiplier keeps them and weights them.
:func:`size_multiplier` is the only consumer-facing function and it deliberately returns a
continuous number.

**The gate is calibration, not discrimination.** A model with excellent AUC and
miscalibrated probabilities will size positions wrongly and confidently. Predicted
probabilities must match realised frequencies within bootstrap intervals before this
ships, regardless of how well it ranks.

Nothing here has been fitted. There is no trade log yet. :func:`fit` raises rather than
training on a placeholder, because a meta-model fitted to synthetic events would be a
calibrated estimate of a random number generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.config import REPO_ROOT, Config
from src.stats.bootstrap import block_bootstrap

TRADE_LOG = REPO_ROOT / "data" / "trades" / "principal_trades.parquet"
MINIMUM_TRADES = 300

TRADE_LOG_SCHEMA = {
    "trade_id": pl.Utf8,
    "t0": pl.Datetime("us", "UTC"),  # when the principal committed, not when he thought about it
    "side": pl.Int8,
    "entry": pl.Float64,
    "stop": pl.Float64,
    "target": pl.Float64,
    "exit_ts": pl.Datetime("us", "UTC"),
    "exit": pl.Float64,
    "outcome": pl.Int8,  # 1 target first, -1 stop first, 0 discretionary exit
    "session_date": pl.Date,
}


@dataclass(frozen=True)
class Calibration:
    bins: pl.DataFrame
    brier: float
    n: int
    n_blocks: int

    @property
    def calibrated(self) -> bool:
        """Every bin's *predicted* probability inside the realised frequency's interval.

        The comparison runs this way round on purpose. Asking whether the realised rate
        falls inside its own bootstrap interval is a tautology — it always does. The
        question is whether what the model promised is consistent with what happened.
        """
        if self.bins.is_empty():
            return False
        return bool(
            (
                (self.bins["predicted"] >= self.bins["ci_low"])
                & (self.bins["predicted"] <= self.bins["ci_high"])
            ).all()
        )

    def summary(self) -> str:
        verdict = "calibrated" if self.calibrated else "NOT CALIBRATED — do not ship"
        return f"Brier {self.brier:.4f} over n={self.n} ({self.n_blocks} blocks): {verdict}"


def load_trade_log(path: Path | None = None) -> pl.DataFrame:
    """The principal's logged trades. Returns an empty, correctly-typed frame if absent."""
    target = Path(path) if path else TRADE_LOG
    if not target.exists():
        return pl.DataFrame(schema=TRADE_LOG_SCHEMA)
    frame = pl.read_parquet(target)
    missing = set(TRADE_LOG_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"trade log is missing columns: {', '.join(sorted(missing))}")
    return frame


def readiness(trades: pl.DataFrame) -> str:
    if trades.is_empty():
        return f"No trade log. Phase 6 needs at least {MINIMUM_TRADES} logged trades."
    if trades.height < MINIMUM_TRADES:
        return f"{trades.height} trades logged; {MINIMUM_TRADES} needed before fitting is honest."
    return f"{trades.height} trades logged — ready to fit."


def fit(trades: pl.DataFrame, features: pl.DataFrame, cfg: Config):
    """Fit the secondary model. Refuses until enough real trades exist."""
    if trades.height < MINIMUM_TRADES:
        raise NotImplementedError(
            f"{readiness(trades)} Fitting now would produce a calibrated estimate of nothing. "
            "The interface exists so that the day the log is full, this is the only line to change."
        )
    raise NotImplementedError(
        "Model choice is deferred until the trade population exists and can be looked at. "
        "Picking an estimator against an imagined distribution is how you end up with one "
        "that fits the imagination."
    )


def calibration_curve(
    predicted: np.ndarray, realised: np.ndarray, session_dates, cfg: Config, *, bins: int = 5
) -> Calibration:
    """Realised frequency against predicted probability, with block-bootstrapped intervals.

    The intervals are what make this a test rather than a picture. Blocking is by
    session-day for the same reason it is everywhere else: trades cluster within a day.
    """
    predicted = np.asarray(predicted, float)
    realised = np.asarray(realised, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = assignment == b
        if mask.sum() == 0:
            continue
        interval = block_bootstrap(
            realised[mask],
            [d for d, keep in zip(session_dates, mask) if keep],
            lambda v: float(v.mean()),
            resamples=cfg.get("statistics.bootstrap_resamples"),
            confidence=cfg.get("statistics.confidence_level"),
            seed=cfg.get("determinism.seed"),
        )
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "predicted": float(predicted[mask].mean()),
                "realised": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "n": int(mask.sum()),
            }
        )

    brier = float(np.mean((predicted - realised) ** 2)) if predicted.size else float("nan")
    return Calibration(
        bins=pl.DataFrame(rows),
        brier=brier,
        n=int(predicted.size),
        n_blocks=len(set(session_dates)),
    )


def size_multiplier(probability: float, *, base_rate: float, cap: float = 2.0) -> float:
    """Position size as a multiple of baseline, from a calibrated probability.

    Linear in the odds ratio against the population base rate, capped. Sizing is the
    output because a binary filter discards the principal's edge on the trades it vetoes;
    a multiplier keeps them and weights them by what the context says.

    Meaningless unless the probability is calibrated — hence the gate.
    """
    if not (0.0 < base_rate < 1.0) or not np.isfinite(probability):
        return float("nan")
    odds = probability / (1.0 - min(probability, 0.999))
    base_odds = base_rate / (1.0 - base_rate)
    return float(np.clip(odds / base_odds, 0.0, cap))
