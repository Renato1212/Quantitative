"""CME equity index session calendar.

Produces, for each trade date in the study period, the UTC boundaries of the regular
trading hours window and of the surrounding electronic session.

Two things this module exists to get right:

*Daylight saving.* RTH is defined in Chicago local time, so its UTC offset moves twice
a year, on dates that do not coincide with Europe's. Boundaries are therefore built as
local wall-clock times and converted through ``zoneinfo``. A hardcoded UTC offset is a
bug that appears for a few weeks each spring and autumn.

*Half days.* An early close truncates the session. A session-relative feature computed
against the wrong close silently rescales itself, and nothing downstream will complain.

The holiday table in ``config/calendar/cme_equity.yaml`` is unverified — see the header
of that file. Floating holidays are computed from their rules here rather than listed,
because a rule cannot be mistyped for one year and correct for the rest.
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import cached_property
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from src.config import REPO_ROOT, Config

UTC = ZoneInfo("UTC")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n = -1`` means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last = date(year, month, _calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """US observance shift: Saturday holidays move back, Sunday holidays move forward."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@dataclass(frozen=True)
class Session:
    """One trade date's boundaries. All timestamps are timezone-aware UTC."""

    session_date: date
    rth_open: datetime
    rth_close: datetime
    eth_open: datetime  # previous business day's evening open
    eth_close: datetime
    is_early_close: bool

    def contains_rth(self, ts: datetime) -> bool:
        return self.rth_open <= ts < self.rth_close


class SessionCalendar:
    """Trade dates and their session boundaries over the study period."""

    def __init__(self, cfg: Config, holidays_path: Path | None = None):
        self._cfg = cfg
        self._tz = ZoneInfo(cfg.get("session.timezone"))
        path = holidays_path or REPO_ROOT / cfg.get("session.calendar_file")
        self._table = yaml.safe_load(path.read_text(encoding="utf-8"))
        self._start: date = cfg.get("scope.study_start")
        self._end: date = cfg.get("scope.study_end")

    # ------------------------------------------------------------------ holidays

    @cached_property
    def closures(self) -> frozenset[date]:
        """Full-closure dates covering the study period, inclusive of a year either side."""
        out: set[date] = set()
        for day in self._table.get("good_friday", []):
            out.add(day)
        for year in range(self._start.year - 1, self._end.year + 2):
            out.add(_observed(date(year, 1, 1)))
            out.add(_nth_weekday(year, 1, 0, 3))  # MLK
            out.add(_nth_weekday(year, 2, 0, 3))  # Presidents' Day
            out.add(_nth_weekday(year, 5, 0, -1))  # Memorial Day
            out.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
            out.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
            for entry in self._table.get("fixed_closures", []):
                if year < entry.get("from_year", 0):
                    continue
                out.add(_observed(date(year, entry["month"], entry["day"])))
        return frozenset(out)

    @cached_property
    def early_closes(self) -> frozenset[date]:
        return frozenset(self._table.get("early_closes", []))

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.closures

    def previous_trading_day(self, day: date) -> date:
        probe = day - timedelta(days=1)
        while not self.is_trading_day(probe):
            probe -= timedelta(days=1)
        return probe

    # ------------------------------------------------------------------ sessions

    def _utc(self, day: date, hhmm: str) -> datetime:
        hour, minute = (int(x) for x in hhmm.split(":"))
        local = datetime.combine(day, time(hour, minute), tzinfo=self._tz)
        return local.astimezone(UTC)

    def session(self, day: date) -> Session:
        if not self.is_trading_day(day):
            raise ValueError(f"{day} is not a trading day")
        early = day in self.early_closes
        close = self._cfg.get("session.early_close" if early else "session.rth_close")
        return Session(
            session_date=day,
            rth_open=self._utc(day, self._cfg.get("session.rth_open")),
            rth_close=self._utc(day, close),
            eth_open=self._utc(
                self.previous_trading_day(day), self._cfg.get("session.eth_open")
            ),
            eth_close=self._utc(day, self._cfg.get("session.eth_close")),
            is_early_close=early,
        )

    @cached_property
    def sessions(self) -> list[Session]:
        out, day = [], self._start
        while day <= self._end:
            if self.is_trading_day(day):
                out.append(self.session(day))
            day += timedelta(days=1)
        return out

    def frame(self) -> pl.DataFrame:
        """The calendar as a table, for joining and for writing to disk."""
        return pl.DataFrame(
            {
                "session_date": [s.session_date for s in self.sessions],
                "rth_open": [s.rth_open for s in self.sessions],
                "rth_close": [s.rth_close for s in self.sessions],
                "eth_open": [s.eth_open for s in self.sessions],
                "eth_close": [s.eth_close for s in self.sessions],
                "is_early_close": [s.is_early_close for s in self.sessions],
            },
            schema_overrides={
                "session_date": pl.Date,
                "rth_open": pl.Datetime("us", "UTC"),
                "rth_close": pl.Datetime("us", "UTC"),
                "eth_open": pl.Datetime("us", "UTC"),
                "eth_close": pl.Datetime("us", "UTC"),
            },
        )

    # ------------------------------------------------------------------ splits

    def chronological_splits(self) -> dict[str, tuple[date, date]]:
        """Contiguous train / validation / holdout ranges, split by session count.

        Splitting on session count rather than calendar days keeps the blocks
        comparably powered — calendar quarters differ in trading-day count by enough
        to matter. C3: contiguous and chronological, never shuffled.
        """
        days = [s.session_date for s in self.sessions]
        n = len(days)
        n_train = int(n * self._cfg.get("scope.splits.train"))
        n_val = int(n * self._cfg.get("scope.splits.validation"))
        if n_train == 0 or n_val == 0 or n_train + n_val >= n:
            raise ValueError(f"split fractions leave an empty block over {n} sessions")
        return {
            "train": (days[0], days[n_train - 1]),
            "validation": (days[n_train], days[n_train + n_val - 1]),
            "holdout": (days[n_train + n_val], days[-1]),
        }
