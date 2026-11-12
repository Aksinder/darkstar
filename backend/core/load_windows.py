"""Time windows for cyclic loads in ``mode: window`` (room heaters on a smart plug).

A window-mode load has no daily energy need: it is ON while a window is active and
OFF otherwise. A window comes from Home Assistant entities — typically a calendar
("Kim is here this week"), but any on/off entity works (input_boolean, schedule,
binary_sensor, switch).

Shared by the planner (which books the load as fixed consumption inside the windows
of its horizon) and the executor (which gates the plug live), so both sides read a
calendar the same way:

* every datetime in here is timezone-aware. The container runs LOCAL time and this
  codebase has been bitten repeatedly by comparing that against UTC;
* an all-day event covers local midnight to local midnight (``end.date`` is
  exclusive, as in iCalendar), which is 23 or 25 hours on a DST change day;
* overlapping and touching events are merged into one window.

Nothing here raises on bad input: a calendar that cannot be read yields ``None``
("unknown"), never an invented window.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any, cast

logger = logging.getLogger("darkstar.core.load_windows")

Window = tuple[datetime, datetime]

MODE_DAILY = "daily"
MODE_WINDOW = "window"
MODE_INVALID = "invalid"

# How far ahead the planner asks a calendar for events. Longer than any horizon the
# planner solves; the cost of a wider range is a few more list items.
PLAN_LOOKAHEAD = timedelta(days=7)
# A calendar that cannot be read is replaced by its last successful answer for this
# long. Windows are absolute times, so an old answer stays right unless the calendar
# was edited meanwhile; six hours rides out an HA restart or an integration hiccup
# without planning a week from a stale list.
CALENDAR_LAST_GOOD_MAX_AGE_S = 6 * 3600.0

_ON = "on"
_OFF = "off"

# (entity_id, lowercased match) -> (time.time() of the fetch, windows)
_last_good: dict[tuple[str, str], tuple[float, list[Window]]] = {}
# Calendars whose last read failed (so the failure is logged once per outage).
_unreadable: set[str] = set()


def load_mode(raw: Mapping[str, Any]) -> str:
    """``daily`` (default: a daily kWh need), ``window``, or ``invalid``.

    An unrecognised value is NOT folded into the default: a typo of ``window`` would
    otherwise be planned as a daily need of 0 kWh, i.e. an explicit "off" in every
    slot — a heater silently held off by a spelling mistake.
    """
    mode = str(raw.get("mode") or MODE_DAILY).strip().lower()
    if mode in (MODE_DAILY, "cyclic", "default"):
        return MODE_DAILY
    if mode == MODE_WINDOW:
        return MODE_WINDOW
    return MODE_INVALID


def is_calendar(entity_id: str) -> bool:
    return entity_id.startswith("calendar.")


def on_off_state(raw: Any) -> bool | None:
    """True for ``on``, False for ``off``, None for anything else (unknown,
    unavailable, missing): an entity that cannot answer has not said "off"."""
    state = str(raw).strip().lower() if raw is not None else ""
    if state == _ON:
        return True
    if state == _OFF:
        return False
    return None


def _localize(naive: datetime, tz: tzinfo) -> datetime:
    localize = getattr(tz, "localize", None)  # pytz
    if callable(localize):
        return cast("datetime", localize(naive))
    return naive.replace(tzinfo=tz)


def _event_edge(raw: Any, tz: tzinfo) -> datetime | None:
    """One ``start``/``end`` object of a calendar event as an aware datetime."""
    if not isinstance(raw, Mapping):
        return None
    edge = cast("Mapping[str, Any]", raw)
    stamp = edge.get("dateTime")
    if stamp:
        try:
            parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        # HA sends an offset; a bare local time can only mean the site's own zone.
        return _localize(parsed, tz) if parsed.tzinfo is None else parsed
    day = edge.get("date")
    if day:
        try:
            d = date.fromisoformat(str(day))
        except ValueError:
            return None
        return _localize(datetime(d.year, d.month, d.day), tz)
    return None


def merge_windows(windows: Iterable[Window]) -> list[Window]:
    """Sorted, with overlapping and touching windows joined."""
    merged: list[Window] = []
    for start, end in sorted(windows, key=lambda w: w[0]):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def parse_calendar_events(
    payload: Any, tz: tzinfo, *, match: str | None = None
) -> list[Window]:
    """HA ``GET /api/calendars/<entity>`` payload -> merged windows.

    ``match``: keep only events whose summary contains this text (case-insensitive),
    so one shared calendar can gate several rooms. A single malformed event is
    skipped; a payload that is not a list at all raises ValueError — the caller must
    treat that as "calendar unreadable", not as "no events".
    """
    if not isinstance(payload, list):
        raise ValueError(f"calendar payload is {type(payload).__name__}, not a list")
    needle = (match or "").strip().lower()
    windows: list[Window] = []
    for item in cast("list[Any]", payload):
        if not isinstance(item, Mapping):
            continue
        event = cast("Mapping[str, Any]", item)
        if needle and needle not in str(event.get("summary") or "").lower():
            continue
        start = _event_edge(event.get("start"), tz)
        end = _event_edge(event.get("end"), tz)
        if start is None or end is None or end <= start:
            logger.debug("calendar event skipped (bad start/end): %r", event)
            continue
        windows.append((start, end))
    return merge_windows(windows)


def extend_for_preheat(windows: Iterable[Window], preheat_minutes: float) -> list[Window]:
    """Each window starting ``preheat_minutes`` earlier (the plug is on by then)."""
    lead = timedelta(minutes=max(0.0, float(preheat_minutes or 0.0)))
    return merge_windows((start - lead, end) for start, end in windows)


async def fetch_calendar_windows(
    entity_id: str,
    start: datetime,
    end: datetime,
    tz: tzinfo,
    *,
    match: str | None = None,
) -> list[Window] | None:
    """Events of one HA calendar overlapping [start, end] as merged windows.

    Events are returned with their REAL start and end, so one that began before
    ``start`` or runs past ``end`` is not truncated. ``None`` when the calendar cannot
    be read (HA down, entity missing, malformed answer) — distinct from ``[]``, which
    is a readable calendar with nothing in range.
    """
    from backend.core import secrets
    from backend.core.ha_client import ha_http_client, make_ha_headers

    ha_config = secrets.load_home_assistant_config()
    url = ha_config.get("url")
    token = ha_config.get("token")
    if not url or not token or not entity_id:
        return None
    try:
        async with ha_http_client() as client:
            response = await client.get(
                f"{url.rstrip('/')}/api/calendars/{entity_id}",
                headers=make_ha_headers(token),
                params={
                    "start": start.astimezone(UTC).isoformat(),
                    "end": end.astimezone(UTC).isoformat(),
                },
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
        found = parse_calendar_events(payload, tz, match=match)
    except Exception as exc:
        # Once per outage, not once per retry: the executor asks again every minute.
        if entity_id not in _unreadable:
            _unreadable.add(entity_id)
            logger.warning("Calendar %s could not be read: %s", entity_id, exc)
        return None
    if entity_id in _unreadable:
        _unreadable.discard(entity_id)
        logger.info("Calendar %s is readable again", entity_id)
    return found


def windows_to_iso(windows: Iterable[Window]) -> list[list[str]]:
    return [[s.isoformat(), e.isoformat()] for s, e in windows]


def windows_from_iso(raw: Any) -> list[Window] | None:
    """Inverse of :func:`windows_to_iso`; None for anything that is not a list of
    aware [start, end] pairs (an absent or damaged entry means "unknown")."""
    if not isinstance(raw, list):
        return None
    out: list[Window] = []
    for pair in cast("list[Any]", raw):
        try:
            start = datetime.fromisoformat(str(pair[0]))
            end = datetime.fromisoformat(str(pair[1]))
        except (ValueError, TypeError, IndexError, KeyError):
            return None
        if start.tzinfo is None or end.tzinfo is None:
            return None
        out.append((start, end))
    return merge_windows(out)


async def plan_windows_for_load(
    raw: Mapping[str, Any],
    now: datetime,
    tz: tzinfo,
    *,
    get_state: Callable[[str], Awaitable[Mapping[str, Any] | None]],
    fetch: Callable[..., Awaitable[list[Window] | None]] = fetch_calendar_windows,
) -> list[Window] | None:
    """The windows of one window-mode load over the planner's horizon.

    * calendar entity: its events from ``now`` to ``now + PLAN_LOOKAHEAD`` (filtered
      by ``window_event_match``), each starting ``preheat_minutes`` early. If the
      calendar cannot be read, its last successful answer is reused while that is
      younger than CALENDAR_LAST_GOOD_MAX_AGE_S.
    * any other entity has no future schedule: its CURRENT state is assumed to
      persist over the whole horizon (on = one window from the start of the current
      hour, so the in-progress slot is covered; off = none).

    Returns ``None`` — "unknown, book nothing" — when any entity can be neither read
    nor replaced by a recent answer. A window is never invented.
    """
    load_id = str(raw.get("id", ""))
    raw_entities: Any = raw.get("window_entities") or []
    if isinstance(raw_entities, str):
        raw_entities = [raw_entities]
    entities = [str(e).strip() for e in cast("list[Any]", raw_entities) if str(e).strip()]
    if not entities:
        logger.warning(
            "Window load %s has no window_entities — nothing to plan from", load_id
        )
        return None
    match = str(raw.get("window_event_match") or "").strip() or None
    try:
        preheat = float(raw.get("preheat_minutes") or 0.0)
    except (TypeError, ValueError):
        preheat = 0.0
    horizon_end = now + PLAN_LOOKAHEAD

    windows: list[Window] = []
    for entity in entities:
        if is_calendar(entity):
            key = (entity, (match or "").lower())
            found = await fetch(entity, now, horizon_end, tz, match=match)
            if found is not None:
                _last_good[key] = (time.time(), found)
            else:
                cached = _last_good.get(key)
                age_s = time.time() - cached[0] if cached else None
                if cached is None or age_s is None or age_s > CALENDAR_LAST_GOOD_MAX_AGE_S:
                    logger.warning(
                        "Window load %s: calendar %s is unreadable and there is no "
                        "answer younger than %.0f h to reuse — booking NO load for it "
                        "in this plan (the executor still gates the plug live)",
                        load_id,
                        entity,
                        CALENDAR_LAST_GOOD_MAX_AGE_S / 3600.0,
                    )
                    return None
                logger.warning(
                    "Window load %s: calendar %s is unreadable — reusing the answer "
                    "from %.0f min ago",
                    load_id,
                    entity,
                    age_s / 60.0,
                )
                found = cached[1]
            windows.extend(extend_for_preheat(found, preheat))
            continue
        try:
            state = await get_state(entity)
        except Exception as exc:
            logger.warning("Window load %s: %s read failed (%s)", load_id, entity, exc)
            state = None
        is_on = on_off_state(state.get("state") if state else None)
        if is_on is None:
            logger.warning(
                "Window load %s: gate %s is unreadable — booking NO load for it in "
                "this plan (the executor still gates the plug live)",
                load_id,
                entity,
            )
            return None
        if is_on:
            windows.append((now.replace(minute=0, second=0, microsecond=0), horizon_end))
    return [(s, e) for s, e in merge_windows(windows) if e > now]


def _reset_last_good_for_tests() -> None:
    _last_good.clear()
    _unreadable.clear()
