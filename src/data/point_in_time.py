"""``PointInTimeView`` — the core safety primitive.

Constructed with an anchor ``t0``. Every read through it is truncated at the source to
information knowable by the cutoff. Feature code receives one of these and nothing else;
a feature that needs a frame the view will not give it is a feature that wants the
future.

Three details carry most of the weight:

*The cutoff is not ``t0``.* It is ``t0 - decision_lag``. A feature computed at the
instant of the anchor is available to a human a moment later, and a finding that is
knife-edge on instantaneous reaction is not a finding. The lag is configured once and
applied uniformly (leak L2).

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

    __slots__ = ("_store", "_cfg", "_calendar", "t0", "cutoff")

    def __init__(self, store: Store, t0: datetime, cfg: Config, calendar: SessionCalendar | None = None):
        if t0.tzinfo is None:
            raise ValueError("t0 must be timezone-aware; naive timestamps hide DST bugs")
        self._store = store
        self._cfg = cfg
        self._calendar = calendar or SessionCalendar(cfg)
        self.t0 = t0
        self.cutoff = t0 - timedelta(seconds=cfg.get("execution_realism.decision_lag_seconds"))

    def __repr__(self) -> str:
        return f"PointInTimeView(t0={self.t0.isoformat()}, cutoff={self.cutoff.isoformat()})"

    # ---------------------------------------------------------------- price reads

    def bars(self, kind: str = "time", *, tail: int | None = None, contract: str | None = None) -> pl.DataFrame:
        """Completed bars only, oldest first.

        ``tail`` takes the most recent ``n``, which is what a lookback window wants.
        """
        extra, params = ("", [])
        if contract is not None:
            extra, params = ("contract = ?", [contract])
        return self._store.read_upto(
            f"bars_{kind}", self.cutoff, extra_sql=extra, params=params, tail=tail
        )

    def ticks(self, *, tail: int | None = None, contract: str | None = None) -> pl.DataFrame:
        extra, params = ("", [])
        if contract is not None:
            extra, params = ("contract = ?", [contract])
        return self._store.read_upto(
            "ticks", self.cutoff, extra_sql=extra, params=params, tail=tail
        )

    def session_bars(self, kind: str = "time", *, rth_only: bool = True) -> pl.DataFrame:
        """Bars of the current session so far — the anytime version of a session aggregate.

        Session VWAP, the initial balance, the developing value area: all of them must be
        built from this, not from the whole session. Computing them on the full session is
        leak L4, and it is invisible in the output because the number still looks sane.
        """
        session = self.current_session()
        if session is None:
            return self.bars(kind).clear()
        frame = self.bars(kind).filter(pl.col("session_date") == session.session_date)
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
