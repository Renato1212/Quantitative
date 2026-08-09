"""``PointInTimeView`` — the core safety primitive.

Constructed with an anchor ``t0``. Every read through it is truncated at the source to
information knowable by the cutoff. Feature code receives one of these and nothing else;
a feature that needs a frame the view will not give it is a feature that wants the
future.

Three details carry most of the weight:

*The cutoff is the information boundary, and it is charged once.* Leak L2 is real —
a decision at ``t0`` cannot be executed at ``t0`` — but the lag belongs on the *fill*,
not on the data. ``t0`` is a bar close, and that bar is precisely what made the trigger
observable; hiding it from the feature that follows the trigger would be incoherent. So
the view sees everything up to ``t0``, and ``labels/triple_barrier.py`` enters at
``t0 + decision_lag``. ``information_lag_seconds`` exists separately and defaults to zero;
it is where feed latency goes if the real timestamps turn out to be receive times.

*Bars are filtered on ``close_ts``, never ``open_ts``.* A bar that opened before the
cutoff but closes after it contains the future. This is leak L1, and it is a
one-character bug.

*The session calendar is not truncated, deliberately.* Holidays and closing times are
published years ahead, so knowing next Thursday is a half day at 09:00 on Monday is not
lookahead. Anything derived from *prices* inside a session is a different matter and
goes through the truncating reads.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from src.config import Config
from src.data.store import Store
from src.ingest.calendar import Session, SessionCalendar


class PointInTimeView:
    """A read handle that physically cannot return information from after its cutoff."""

    __slots__ = ("_store", "_cfg", "_calendar", "_memo", "t0", "cutoff")

    def __init__(self, store: Store, t0: datetime, cfg: Config, calendar: SessionCalendar | None = None):
        if t0.tzinfo is None:
            raise ValueError("t0 must be timezone-aware; naive timestamps hide DST bugs")
        self._store = store
        self._cfg = cfg
        self._calendar = calendar or SessionCalendar(cfg)
        self.t0 = t0
        self.cutoff = t0 - timedelta(
            seconds=cfg.get("execution_realism.information_lag_seconds", 0)
        )
        self._memo: dict = {}

    def __repr__(self) -> str:
        return f"PointInTimeView(t0={self.t0.isoformat()}, cutoff={self.cutoff.isoformat()})"

    # ---------------------------------------------------------------- price reads

    def bars(
        self,
        kind: str = "time",
        *,
        tail: int | None = None,
        since: datetime | None = None,
        contract: str | None = None,
    ) -> pl.DataFrame:
        """Completed bars only, oldest first.

        ``tail`` takes the most recent ``n`` bars; ``since`` takes everything after a
        timestamp. Prefer ``since`` for session-relative windows — a row cap looks like a
        lookback but returns fewer sessions than asked for whenever bar density changes.
        """
        return self._read("bars", kind, tail, since, contract)

    def ticks(
        self,
        *,
        tail: int | None = None,
        since: datetime | None = None,
        contract: str | None = None,
    ) -> pl.DataFrame:
        """Prints up to the cutoff, oldest first. Same windowing rules as :meth:`bars`."""
        return self._read("ticks", None, tail, since, contract)

    def _read(self, dataset: str, kind, tail, since, contract) -> pl.DataFrame:
        """Memoised truncating read.

        A view is immutable and single-cutoff, so identical arguments give identical
        answers. Feature groups ask for the same trailing window repeatedly; without this
        a feature matrix costs one Parquet scan per feature per event.
        """
        key = (dataset, kind, tail, since, contract)
        if key not in self._memo:
            extra, params = ("", [])
            if contract is not None:
                extra, params = ("contract = ?", [contract])
            name = "ticks" if dataset == "ticks" else f"bars_{kind}"
            self._memo[key] = self._store.read_upto(
                name, self.cutoff, extra_sql=extra, params=params, tail=tail, since=since
            )
        return self._memo[key]

    def session_bars(self, kind: str = "time", *, rth_only: bool = True) -> pl.DataFrame:
        """Bars of the current session so far — the anytime version of a session aggregate.

        Session VWAP, the initial balance, the developing value area: all of them must be
        built from this, not from the whole session. Computing them on the full session is
        leak L4, and it is invisible in the output because the number still looks sane.
        """
        session = self.current_session()
        if session is None:
            return self.bars(kind).clear()
        frame = self.bars(kind, since=session.eth_open).filter(
            pl.col("session_date") == session.session_date
        )
        return frame.filter(pl.col("in_rth")) if rth_only else frame

    # ---------------------------------------------------------------- calendar

    def current_session(self) -> Session | None:
        """The session the cutoff falls inside, or ``None`` outside trading hours."""
        for session in reversed(self._calendar.sessions):
            if session.eth_open <= self.cutoff <= session.eth_close:
                return session
            if session.eth_close < self.cutoff:
                return None
        return None

    def prior_sessions(self, n: int) -> list[Session]:
        """The ``n`` completed sessions before the current one, oldest first.

        A session counts as complete only once its close is at or before the cutoff, so
        the session in progress is never included.
        """
        done = [s for s in self._calendar.sessions if s.rth_close <= self.cutoff]
        return done[-n:] if n else []

    @property
    def calendar(self) -> SessionCalendar:
        """Session boundaries. Published in advance, so not truncated — see module docstring."""
        return self._calendar

    @property
    def config(self) -> Config:
        return self._cfg
