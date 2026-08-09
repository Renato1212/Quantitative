"""Triple-barrier outcomes and continuous excursion metrics.

Computed from ticks, not bars. Bar-granularity excursion understates the true maximum
favourable excursion — the high of a one-minute bar is not the path within it — and the
right tail is exactly where that understatement bites hardest.

What is stored, per event:

- ``label`` — +1 target first, -1 stop first, 0 the clock ran out. The categorical
  outcome, kept because meta-labeling in Phase 6 needs it.
- ``window_end`` — when the label actually resolved. The purging logic consumes this;
  without it, purged cross-validation cannot know what overlaps what.
- ``mfe`` / ``mae`` — maximum favourable and adverse excursion in ATR units, side-aligned.
- ``mfe_up`` / ``mfe_down`` — both directions regardless of side, because the taxonomy
  should be able to ask what happened, not only what happened in the direction we guessed.
- ``mae_before_mfe`` — how much heat was taken before the favourable extreme. A 4× MFE
  that first went 3× against is not a trade anyone holds.
- ``time_to_mfe_min`` and ``velocity`` — the "fast" half of "large and fast" (decision D7).
- ``asymmetry`` — ``mfe / max(mae_before_mfe, floor)``. The floor is pre-registered
  because at tick granularity the denominator is often zero, and an unfloored ratio
  produces a right tail that is pure artefact.

Nothing is thresholded into "big move" here. Continuous values are stored and the
thresholds are applied later, in the open, where they can be varied and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from src.config import Config

NS = 1_000_000_000


@dataclass(frozen=True)
class TickPath:
    """Ticks as flat arrays, so each event is a slice rather than a query."""

    ts: np.ndarray  # datetime64[ns], ascending
    price: np.ndarray

    @classmethod
    def from_frame(cls, ticks: pl.DataFrame) -> TickPath:
        ordered = ticks.sort("ts")
        return cls(
            ts=ordered["ts"].to_numpy().astype("datetime64[ns]"),
            price=ordered["price"].to_numpy().astype(float),
        )

    def slice(self, start: np.datetime64, end: np.datetime64) -> tuple[np.ndarray, np.ndarray]:
        lo = int(np.searchsorted(self.ts, start, side="left"))
        hi = int(np.searchsorted(self.ts, end, side="right"))
        return self.ts[lo:hi], self.price[lo:hi]


def label_events(events: pl.DataFrame, ticks: pl.DataFrame, atr: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Attach barrier outcomes and excursion metrics to every event.

    ``events`` needs ``trigger_family``, ``t0``, ``side`` and ``session_date``. Events
    whose session has no ATR yet (the warm-up window) are dropped, and the count of drops
    is recoverable by comparing heights.
    """
    horizon = timedelta(minutes=cfg.get("labels.time_barrier_minutes"))
    lag = timedelta(seconds=cfg.get("execution_realism.decision_lag_seconds"))
    target_mult = cfg.get("labels.target_multiple")
    stop_mult = cfg.get("labels.stop_multiple")
    floor_points = cfg.get("labels.mae_floor_ticks") * cfg.get("contract.tick_size")

    unit = barrier_unit_scale(cfg)
    joined = (
        events.join(atr.select("session_date", "atr", "atr_long"), on="session_date", how="left")
        .filter(pl.col("atr").is_not_null() & (pl.col("atr") > 0))
        .sort("t0")
    )
    if joined.is_empty():
        return _empty_labels()

    path = TickPath.from_frame(ticks)
    horizon_ns = np.timedelta64(int(horizon.total_seconds()), "s")
    lag_ns = np.timedelta64(int(lag.total_seconds()), "s")

    rows = [
        _label_one(
            t0=np.datetime64(t0.replace(tzinfo=None), "ns"),
            side=int(side),
            atr_value=float(atr_value) * unit,
            path=path,
            horizon=horizon_ns,
            lag=lag_ns,
            target_mult=target_mult,
            stop_mult=stop_mult,
            floor_points=floor_points,
        )
        for t0, side, atr_value in zip(joined["t0"], joined["side"], joined["atr"])
    ]

    labels = pl.DataFrame(rows, schema=_LABEL_SCHEMA)
    return pl.concat([joined, labels], how="horizontal").filter(pl.col("n_ticks_in_window") > 0)


_LABEL_SCHEMA = {
    "entry": pl.Float64,
    "label": pl.Int8,
    "window_end": pl.Datetime("us", "UTC"),
    "mfe": pl.Float64,
    "mae": pl.Float64,
    "mfe_up": pl.Float64,
    "mfe_down": pl.Float64,
    "mae_before_mfe": pl.Float64,
    "time_to_mfe_min": pl.Float64,
    "velocity": pl.Float64,
    "asymmetry": pl.Float64,
    "n_ticks_in_window": pl.Int64,
}


def barrier_unit_scale(cfg: Config) -> float:
    """How much of a session's ATR the label horizon is worth.

    Square-root-of-time. Crude — it assumes a driftless walk and ignores the intraday
    volatility smile — but it is the difference between barriers that can be touched and
    barriers that cannot. Set ``labels.horizon_scaled_barriers: false`` to use the raw
    session ATR, and expect almost every event to time out.

    The multiples themselves must be re-examined against real data at the Phase 2 gate;
    calibrating them on a synthetic tape would be fitting to a random number generator.
    """
    if not cfg.get("labels.horizon_scaled_barriers"):
        return 1.0
    horizon = cfg.get("labels.time_barrier_minutes")
    session = cfg.get("labels.rth_minutes")
    return float((horizon / session) ** 0.5)


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(schema=_LABEL_SCHEMA)


def _label_one(
    *,
    t0: np.datetime64,
    side: int,
    atr_value: float,
    path: TickPath,
    horizon: np.timedelta64,
    lag: np.timedelta64,
    target_mult: float,
    stop_mult: float,
    floor_points: float,
) -> dict:
    """One event's outcome. Entry is the first print at or after the decision lag."""
    entry_from = t0 + lag
    ts, price = path.slice(entry_from, entry_from + horizon)
    if ts.size == 0:
        return dict(
            entry=float("nan"), label=0, window_end=_as_utc(entry_from + horizon),
            mfe=float("nan"), mae=float("nan"), mfe_up=float("nan"), mfe_down=float("nan"),
            mae_before_mfe=float("nan"), time_to_mfe_min=float("nan"),
            velocity=float("nan"), asymmetry=float("nan"), n_ticks_in_window=0,
        )

    entry = float(price[0])
    excursion = (price - entry) * side  # positive = favourable, in points

    target = target_mult * atr_value
    stop = -stop_mult * atr_value

    hit_target = np.flatnonzero(excursion >= target)
    hit_stop = np.flatnonzero(excursion <= stop)
    first_target = int(hit_target[0]) if hit_target.size else None
    first_stop = int(hit_stop[0]) if hit_stop.size else None

    if first_target is not None and (first_stop is None or first_target < first_stop):
        label, resolved = 1, ts[first_target]
    elif first_stop is not None:
        label, resolved = -1, ts[first_stop]
    else:
        label, resolved = 0, ts[-1]

    # Excursions are measured over the full horizon, not truncated at the barrier: the
    # question is what the market did, not what a particular exit would have captured.
    peak = int(np.argmax(excursion))
    mfe_points = float(excursion[peak])
    heat_before = float(-np.min(excursion[: peak + 1])) if peak >= 0 else 0.0

    up = float(np.max(price - entry))
    down = float(np.max(entry - price))
    elapsed_min = float((ts[peak] - ts[0]) / np.timedelta64(1, "s")) / 60.0

    mfe = mfe_points / atr_value
    denominator = max(heat_before, floor_points)
    return dict(
        entry=entry,
        label=label,
        window_end=_as_utc(resolved),
        mfe=mfe,
        mae=float(-np.min(excursion)) / atr_value,
        mfe_up=up / atr_value,
        mfe_down=down / atr_value,
        mae_before_mfe=heat_before / atr_value,
        time_to_mfe_min=elapsed_min,
        velocity=mfe / elapsed_min if elapsed_min > 0 else float("nan"),
        asymmetry=mfe_points / denominator,
        n_ticks_in_window=int(ts.size),
    )


def _as_utc(stamp: np.datetime64):
    from datetime import timezone

    return stamp.astype("datetime64[us]").item().replace(tzinfo=timezone.utc)


def stop_distance_multiples(labels: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Express MFE in multiples of the stop distance, which is what §6 Phase 3 asks for.

    ATR units and stop-distance units differ by the configured stop multiple. Reporting
    ``P(MFE > 3x)`` without saying which unit is meant is a good way to compare two
    incompatible numbers.
    """
    stop = cfg.get("labels.stop_multiple") * barrier_unit_scale(cfg)
    return labels.with_columns((pl.col("mfe") / stop).alias("mfe_stops"))
