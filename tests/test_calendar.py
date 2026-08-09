"""Session calendar: daylight saving, half days, holidays, and the chronological splits."""

from __future__ import annotations

from datetime import date, timedelta

import pytest


def test_rth_open_shifts_with_daylight_saving(calendar):
    """RTH is 08:30 Chicago. In UTC that is 14:30 in winter and 13:30 in summer."""
    winter = calendar.session(date(2024, 1, 10))
    summer = calendar.session(date(2024, 7, 10))
    assert winter.rth_open.hour == 14
    assert summer.rth_open.hour == 13
    assert winter.rth_open.minute == summer.rth_open.minute == 30


@pytest.mark.parametrize(
    "holiday",
    [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # MLK, third Monday
        date(2024, 2, 19),  # Presidents' Day, third Monday
        date(2024, 3, 29),  # Good Friday, from the table
        date(2024, 5, 27),  # Memorial Day, last Monday
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day, first Monday
        date(2024, 11, 28),  # Thanksgiving, fourth Thursday
        date(2024, 12, 25),  # Christmas
    ],
)
def test_known_holidays_are_not_trading_days(calendar, holiday):
    assert not calendar.is_trading_day(holiday)


def test_weekend_holidays_shift_observance(calendar):
    """2022-01-01 fell on a Saturday, so the observed closure is Friday the 31st."""
    assert date(2021, 12, 31) in calendar.closures
    assert date(2022, 1, 1) not in calendar.closures


def test_juneteenth_only_from_2022(calendar):
    assert date(2021, 6, 18) not in calendar.closures
    assert date(2021, 6, 21) not in calendar.closures
    assert date(2022, 6, 20) in calendar.closures


def test_half_day_closes_early(calendar, cfg):
    early = calendar.session(date(2024, 7, 3))
    normal = calendar.session(date(2024, 7, 2))
    assert early.is_early_close
    assert not normal.is_early_close
    assert early.rth_close - early.rth_open < normal.rth_close - normal.rth_open


def test_sessions_are_ordered_and_unique(calendar):
    days = [s.session_date for s in calendar.sessions]
    assert days == sorted(days)
    assert len(days) == len(set(days))
    assert all(d.weekday() < 5 for d in days)


def test_eth_precedes_rth_and_wraps_the_prior_day(calendar):
    session = calendar.session(date(2024, 3, 20))
    assert session.eth_open < session.rth_open < session.rth_close <= session.eth_close
    assert session.eth_open.date() < session.session_date


def test_previous_trading_day_skips_weekends_and_holidays(calendar):
    assert calendar.previous_trading_day(date(2024, 1, 2)) == date(2023, 12, 29)
    assert calendar.previous_trading_day(date(2024, 7, 5)) == date(2024, 7, 3)


def test_splits_are_contiguous_chronological_and_exhaustive(calendar):
    """C3. Adjacent blocks touch, never overlap, and cover the period end to end."""
    splits = calendar.chronological_splits()
    days = [s.session_date for s in calendar.sessions]
    (tr_a, tr_b), (va_a, va_b), (ho_a, ho_b) = (
        splits["train"], splits["validation"], splits["holdout"]
    )

    assert tr_a == days[0] and ho_b == days[-1]
    assert tr_b < va_a <= va_b < ho_a
    assert days[days.index(tr_b) + 1] == va_a
    assert days[days.index(va_b) + 1] == ho_a


def test_holdout_is_roughly_the_final_quarter(calendar, cfg):
    splits = calendar.chronological_splits()
    days = [s.session_date for s in calendar.sessions]
    holdout = [d for d in days if d >= splits["holdout"][0]]
    share = len(holdout) / len(days)
    assert abs(share - cfg.get("scope.splits.holdout")) < 0.01


def test_session_lookup_rejects_non_trading_days(calendar):
    with pytest.raises(ValueError, match="not a trading day"):
        calendar.session(date(2024, 12, 25))


def test_calendar_covers_every_year_of_the_study_period(calendar, cfg):
    """A missing year in the holiday table would silently produce phantom sessions."""
    years = {s.session_date.year for s in calendar.sessions}
    expected = set(range(cfg.get("scope.study_start").year, cfg.get("scope.study_end").year + 1))
    assert years == expected


def test_no_session_gap_exceeds_a_long_weekend(calendar):
    """Any gap over four days means the holiday table has invented a closure."""
    days = [s.session_date for s in calendar.sessions]
    gaps = [(b - a, a, b) for a, b in zip(days, days[1:]) if b - a > timedelta(days=4)]
    assert not gaps, f"suspicious gaps: {gaps}"
