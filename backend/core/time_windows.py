"""Reusable local-time window matcher (months / weekdays / hours / Swedish holidays).

Ported from upstream (d500af01 ``backend/core/time_windows.py``) and extended with the
compact ``hours: "06-22"`` form. Used by the time-of-use grid transfer fee
(:mod:`backend.core.grid_tariff`); written to be reusable by anything that needs
"which local hours count" semantics.

Semantics (all evaluated on LOCAL wall-clock time — callers convert aware datetimes
to the site timezone first, which makes matching DST-safe by construction):

- ``months``: list of 1-12; omitted/empty matches every month.
- ``weekdays``: list of 0-6 (0=Mon .. 6=Sun, Python ``weekday()``); omitted/empty
  matches every day.
- ``hours``: half-open local-hour range ``[start, end)`` with ``0 <= start <= 23``,
  ``1 <= end <= 24`` and ``start != end``; ``start > end`` wraps past midnight.
  Accepted forms: ``"06-22"``, ``[6, 22]`` or ``{start: 6, end: 22}``. Omitted
  matches the whole day.
- ``exclude_holidays``: when true the window never matches on a Swedish public holiday
  or the de-facto non-working eves (see :func:`swedish_public_holidays`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, cast

SATURDAY = 5
SUNDAY = 6

_HOURS_STR_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


@dataclass(frozen=True)
class TimeWindow:
    """A parsed, validated local-time window."""

    months: frozenset[int] = frozenset()
    weekdays: frozenset[int] = frozenset()
    hour_start: int | None = None
    hour_end: int | None = None
    exclude_holidays: bool = False


def _parse_int_list(raw: Any, field: str, lo: int, hi: int) -> frozenset[int]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list")
    values: set[int] = set()
    for item in cast("list[Any]", raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field} values must be integers {lo}-{hi}")
        if not lo <= item <= hi:
            raise ValueError(f"{field} value {item} is outside {lo}-{hi}")
        values.add(item)
    return frozenset(values)


def _parse_hour(raw: Any, field: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"hours.{field} must be an integer {lo}-{hi}")
    if not lo <= raw <= hi:
        raise ValueError(f"hours.{field} {raw} is outside {lo}-{hi}")
    return raw


def _parse_hours(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        m = _HOURS_STR_RE.match(raw)
        if not m:
            raise ValueError(f"hours {raw!r} must look like 'HH-HH' (e.g. '06-22')")
        start_raw, end_raw = int(m.group(1)), int(m.group(2))
    elif isinstance(raw, list):
        pair = cast("list[Any]", raw)
        if len(pair) != 2:
            raise ValueError("hours list must be [start, end]")
        start_raw, end_raw = pair[0], pair[1]
    elif isinstance(raw, dict):
        rng = cast("dict[str, Any]", raw)
        start_raw, end_raw = rng.get("start"), rng.get("end")
    else:
        raise ValueError("hours must be 'HH-HH', [start, end] or {start, end}")
    start = _parse_hour(start_raw, "start", 0, 23)
    end = _parse_hour(end_raw, "end", 1, 24)
    if start == end:
        raise ValueError("hours start and end must differ")
    return start, end


def parse_window(raw: Any) -> TimeWindow:
    """Parse a window mapping. Raises ``ValueError`` on invalid input."""
    if not isinstance(raw, dict):
        raise ValueError("window must be a mapping")
    window = cast("dict[str, Any]", raw)
    months = _parse_int_list(window.get("months"), "months", 1, 12)
    weekdays = _parse_int_list(window.get("weekdays"), "weekdays", 0, 6)
    excl = window.get("exclude_holidays", False)
    if not isinstance(excl, bool):
        raise ValueError("exclude_holidays must be true/false")
    hours = _parse_hours(window.get("hours"))
    if hours is None:
        return TimeWindow(months=months, weekdays=weekdays, exclude_holidays=excl)
    return TimeWindow(
        months=months,
        weekdays=weekdays,
        hour_start=hours[0],
        hour_end=hours[1],
        exclude_holidays=excl,
    )


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _saturday_on_or_after(start: date) -> date:
    """First Saturday on or after ``start`` (used for 7-day floating holidays)."""
    return start + timedelta(days=(SATURDAY - start.weekday()) % 7)


@lru_cache(maxsize=32)
def swedish_public_holidays(year: int) -> frozenset[date]:
    """Swedish public holidays plus the non-working eves (Midsommar-, Jul-, Nyårsafton).

    Påsk- and Pingstafton always fall on a Saturday, so a weekday filter already
    excludes them.
    """
    easter = _easter_sunday(year)
    midsummer_day = _saturday_on_or_after(date(year, 6, 20))
    return frozenset(
        {
            date(year, 1, 1),  # Nyårsdagen
            date(year, 1, 6),  # Trettondedag jul
            easter - timedelta(days=2),  # Långfredagen
            easter,  # Påskdagen
            easter + timedelta(days=1),  # Annandag påsk
            date(year, 5, 1),  # Första maj
            easter + timedelta(days=39),  # Kristi himmelsfärdsdag
            date(year, 6, 6),  # Sveriges nationaldag
            easter + timedelta(days=49),  # Pingstdagen
            midsummer_day - timedelta(days=1),  # Midsommarafton
            midsummer_day,  # Midsommardagen
            _saturday_on_or_after(date(year, 10, 31)),  # Alla helgons dag
            date(year, 12, 24),  # Julafton
            date(year, 12, 25),  # Juldagen
            date(year, 12, 26),  # Annandag jul
            date(year, 12, 31),  # Nyårsafton
        }
    )


def is_swedish_holiday(day: date) -> bool:
    return day in swedish_public_holidays(day.year)


def _hour_matches(window: TimeWindow, hour: int) -> bool:
    if window.hour_start is None or window.hour_end is None:
        return True
    if window.hour_start < window.hour_end:
        return window.hour_start <= hour < window.hour_end
    return hour >= window.hour_start or hour < window.hour_end


def matches(window: TimeWindow, local_dt: datetime) -> bool:
    """True if the LOCAL wall-clock datetime falls inside the window."""
    if window.months and local_dt.month not in window.months:
        return False
    if window.weekdays and local_dt.weekday() not in window.weekdays:
        return False
    if window.exclude_holidays and is_swedish_holiday(local_dt.date()):
        return False
    return _hour_matches(window, local_dt.hour)
