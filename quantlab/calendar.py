"""Approximate NYSE trading calendar.

Used only for *diagnostics*: to tell a missing observation apart from a normal
market holiday. It is never used to create, fill or shift prices.

The rules cover the regular NYSE holiday schedule since 1998 plus notable
unscheduled closures. It is an approximation: very old history, half-days and
rare one-off closures may not be represented, so a reported gap is a prompt to
investigate, not proof of bad data.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import MO, TH


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        # NYSE does not close on the preceding Friday when Jan 1 is a Saturday.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=DateOffset(weekday=MO(3)),
            start_date=pd.Timestamp("1998-01-01"),
        ),
        Holiday("Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date=pd.Timestamp("2022-01-01"),
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# Unscheduled or discontinued full-day closures since 1970 (election days, national days of mourning,
# weather and blackouts, 9/11, Hurricane Sandy).
SPECIAL_CLOSURES = pd.DatetimeIndex(
    [
        "1972-11-07",  # presidential election day (closed on election days until 1980)
        "1972-12-28",  # national day of mourning, President Truman
        "1973-01-25",  # national day of mourning, President Johnson
        "1976-11-02",  # presidential election day
        "1977-07-14",  # New York City blackout
        "1980-11-04",  # presidential election day
        "1985-09-27",  # Hurricane Gloria
        "1994-04-27",
        "2001-09-11",
        "2001-09-12",
        "2001-09-13",
        "2001-09-14",
        "2004-06-11",
        "2007-01-02",
        "2012-10-29",
        "2012-10-30",
        "2018-12-05",
        "2025-01-09",
    ]
)


def expected_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Weekdays between ``start`` and ``end`` (inclusive) that are not NYSE holidays."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(start, end)
    holidays = NYSEHolidayCalendar().holidays(start, end)
    closures = SPECIAL_CLOSURES[(SPECIAL_CLOSURES >= start) & (SPECIAL_CLOSURES <= end)]
    return weekdays.difference(holidays).difference(closures)


def session_gaps(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Compare observed dates with the approximate NYSE calendar.

    Returns ``(missing, unexpected)``: expected sessions with no observation, and
    observations on dates the calendar does not expect to be trading days.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])
    observed = pd.DatetimeIndex(index).normalize()
    expected = expected_sessions(observed.min(), observed.max())
    return expected.difference(observed), observed.difference(expected)
