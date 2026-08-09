"""Seeded synthetic tape, for exercising and testing the pipeline.

**This is not market data and nothing produced from it is a research result.** It exists
because the Phase 1 gate — leakage canaries caught, byte-identical reruns — is a
property of the machinery, not of the market, and can be met before a Rithmic snapshot
exists. Any config whose ``data.version`` starts with ``synthetic`` marks every
downstream artefact accordingly, and the reporting layer refuses to present such a run
as a finding.

The generator is deliberately unrealistic in the ways that matter. It has no
positioning imbalance, no informational catalysts, and no auction structure — the very
things the research question is about. It has session boundaries, a volume roll, an
intraday volatility shape, and heavy-tailed jumps, because those are what the plumbing
has to survive. Fitting anything to this tape would be fitting to a random number
generator with a calendar attached.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from src.config import Config
from src.ingest.calendar import SessionCalendar
from src.ingest.contracts import Contract, quarterly_contracts

# Deferred contract's share of volume over the sessions leading into the front's expiry.
# Crossing 0.5 is what makes the volume roll rule fire.
_ROLL_RAMP = np.array([0.02, 0.03, 0.05, 0.09, 0.16, 0.28, 0.45, 0.62, 0.78, 0.90])


def _session_ticks(
    rng: np.random.Generator,
    n: int,
    start_price: float,
    tick_size: float,
    vol_per_tick: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prices, sizes and aggressor flags for one session's ticks."""
    # A U-shaped intraday volatility profile, plus rare jumps to give the tape a tail.
    clock = np.linspace(0.0, 1.0, n)
    shape = 0.6 + 1.9 * (clock - 0.5) ** 2 * 4
    steps = rng.standard_normal(n) * vol_per_tick * shape
    jumps = (rng.random(n) < 0.0004) * rng.standard_normal(n) * vol_per_tick * 45
    prices = start_price + np.cumsum(steps + jumps)
    prices = np.round(prices / tick_size) * tick_size

    sizes = 1 + rng.geometric(0.45, size=n)
    # Aggressor correlates with the price change it caused, as a real tape does.
    delta = np.diff(prices, prepend=prices[0])
    aggressor = np.where(delta > 0, 1, np.where(delta < 0, -1, rng.choice([-1, 1], size=n)))
    return prices, sizes, aggressor.astype(np.int8)


def generate_ticks(
    cfg: Config,
    start: date | None = None,
    end: date | None = None,
    ticks_per_session: int = 4000,
) -> pl.DataFrame:
    """Generate a deterministic synthetic tape over the calendar's trading sessions."""
    cal = SessionCalendar(cfg)
    sessions = [s for s in cal.sessions if (start is None or s.session_date >= start) and (end is None or s.session_date <= end)]
    if not sessions:
        raise ValueError("no sessions in the requested range")

    contracts = quarterly_contracts(cfg, sessions[0].session_date, sessions[-1].session_date)
    tick_size = cfg.get("contract.tick_size")
    rng = np.random.default_rng(cfg.get("determinism.seed"))

    price = 3800.0
    frames: list[pl.DataFrame] = []

    for session in sessions:
        day = session.session_date
        front, deferred = _front_and_deferred(contracts, day)
        share = _deferred_share(front, sessions, day)

        # Ticks span the electronic session so overnight context exists for features.
        span_ns = int((session.eth_close - session.eth_open).total_seconds() * 1e9)
        offsets = np.sort(rng.integers(0, span_ns, size=ticks_per_session))
        ts = np.datetime64(session.eth_open.replace(tzinfo=None), "ns") + offsets.astype("timedelta64[ns]")

        prices, sizes, aggressor = _session_ticks(
            rng, ticks_per_session, price, tick_size, vol_per_tick=0.32
        )
        price = float(prices[-1])

        on_deferred = rng.random(ticks_per_session) < share
        names = np.where(on_deferred, deferred.code, front.code)
        # The deferred contract trades at a small, drifting basis to the front.
        prices = prices + on_deferred * round(rng.normal(0.0, 6.0) / tick_size) * tick_size

        frames.append(
            pl.DataFrame(
                {
                    "ts": ts,
                    "price": prices,
                    "size": sizes.astype(np.int64),
                    "aggressor": aggressor,
                    "contract": names,
                    "session_date": pl.Series([day] * ticks_per_session, dtype=pl.Date),
                }
            )
        )

    return (
        pl.concat(frames)
        .with_columns(
            pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            pl.col("price").cast(pl.Float64),
            pl.col("session_date").cast(pl.Date),
        )
        .sort("ts")
    )


def _front_and_deferred(contracts: list[Contract], day: date) -> tuple[Contract, Contract]:
    live = [c for c in contracts if c.expiry > day]
    if len(live) < 2:
        raise ValueError(f"no deferred contract available on {day}")
    return live[0], live[1]


def _deferred_share(front: Contract, sessions, day: date) -> float:
    """Deferred volume share, ramping over the sessions before the front expires."""
    remaining = sum(1 for s in sessions if day < s.session_date <= front.expiry)
    if remaining >= len(_ROLL_RAMP):
        return float(_ROLL_RAMP[0])
    return float(_ROLL_RAMP[len(_ROLL_RAMP) - 1 - remaining])


def session_volumes(ticks: pl.DataFrame) -> pl.DataFrame:
    """Per-session, per-contract volume — the input the volume roll rule consumes."""
    return (
        ticks.group_by("session_date", "contract")
        .agg(pl.col("size").sum().alias("volume"))
        .sort("session_date", "contract")
    )
