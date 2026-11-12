"""Live control of ``mode: window`` cyclic loads (room heaters on a smart plug).

The planner only BOOKS these loads (fixed consumption inside their windows). Whether
the plug is on is decided here, every tick, from a live read — because the plan can
be stale in exactly the ways that matter for a bedroom: kept after a solver timeout,
or computed before somebody edited the calendar.

Three rules shape everything below:

* An unreadable gate is not "off". A calendar integration that hiccups must neither
  switch off a warm room nor switch on an empty one, so an unknown gate HOLDS the
  relay as it is (no write at all) for GATE_HOLD_S and only then falls back to what
  the plan slot says.
* The room thermostat is optional and distrusts its sensor: with no sensor, an
  unknown reading or a stale one, control is plain window on/off.
* The relay is spared: gate- and temperature-driven switches are at least
  MIN_DWELL_S apart, so a noisy sensor cannot chatter it.

The decision helpers are pure; WindowLoadRuntime holds the small per-load memory
(event cache, hold clock, dwell clock, last published status) and does the reads.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytz

from backend.core.ha_timestamps import reading_timestamp
from backend.core.load_windows import (
    Window,
    fetch_calendar_windows,
    is_calendar,
    on_off_state,
)

logger = logging.getLogger(__name__)

# The event list of a calendar (needed for preheat, window_event_match and the
# "next window" attributes) is refreshed this often — not on every 60 s tick.
EVENT_REFRESH_S = 300.0
# After a failed refresh, try again no sooner than this.
EVENT_RETRY_S = 60.0
# A list that could not be refreshed is still followed until it is this old; it is a
# timetable, and a half-hour-old timetable is right unless someone just edited it.
EVENT_MAX_AGE_S = 1800.0
# How long an unknown gate holds the relay as it is before the plan slot takes over.
GATE_HOLD_S = 1800.0
# Minimum time between two gate/temperature-driven switches of one plug.
MIN_DWELL_S = 300.0
# The status sensor is re-published on change, and at least this often (an HA
# restart forgets states that were only ever POSTed).
PUBLISH_REFRESH_S = 600.0
EVENT_LOOKAHEAD = timedelta(days=15)

STATE_PAUSED = "paused"
STATE_OFF = "off"
STATE_PREHEAT = "preheat"
STATE_ON = "on"
STATE_TEMP_HOLD = "temp_hold"
STATE_SETBACK = "setback"
STATE_OVERRIDE_ON = "override_on"
STATE_OVERRIDE_OFF = "override_off"
STATE_FUSE_SHED = "fuse_shed"
STATE_GATE_UNKNOWN = "gate_unknown"


@dataclass(frozen=True)
class Gate:
    """One reading of a load's window. ``active`` None = could not be determined."""

    active: bool | None
    preheat: bool = False
    # The active window, or the next one when none is active (None = not known).
    window_start: datetime | None = None
    window_end: datetime | None = None
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    """``want_on`` None = leave the relay exactly as it is (no write)."""

    want_on: bool | None
    state: str
    reason: str
    room_temp: float | None = None
    target_temp: float | None = None


def gate_from_windows(windows: list[Window], now: datetime, preheat_minutes: float) -> Gate:
    """Where ``now`` stands against merged, timezone-aware windows."""
    lead = timedelta(minutes=max(0.0, preheat_minutes))
    upcoming: Window | None = None
    for start, end in windows:
        if start <= now < end:
            return Gate(True, False, start, end)
        if start > now and (upcoming is None or start < upcoming[0]):
            upcoming = (start, end)
    if upcoming is None:
        return Gate(False)
    return Gate(False, lead > timedelta(0) and now >= upcoming[0] - lead, *upcoming)


def thermostat_wants_heat(
    temp_c: float, target_c: float, hysteresis_c: float, heating: bool | None
) -> bool:
    """On at or below target - hysteresis, off at or above target + hysteresis, and
    unchanged in between. ``heating`` None (relay state unknown) picks the side of
    the target the room is on."""
    if temp_c <= target_c - hysteresis_c:
        return True
    if temp_c >= target_c + hysteresis_c:
        return False
    return temp_c < target_c if heating is None else heating


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())


def status_entity_id(load_id: str) -> str:
    return f"sensor.darkstar_{_slug(load_id)}_state"


class WindowLoadRuntime:
    """Per-load memory and reads for window-mode loads. One instance per engine."""

    def __init__(
        self, fetch: Callable[..., Awaitable[list[Window] | None]] = fetch_calendar_windows
    ) -> None:
        self._fetch = fetch
        # (calendar entity, lowercased match) -> (fetch time, windows)
        self._events: dict[tuple[str, str], tuple[float, list[Window]]] = {}
        self._event_try_ts: dict[tuple[str, str], float] = {}
        # load id -> when its gate first could not be determined (this episode)
        self._unknown_since: dict[str, float] = {}
        self._fallback_logged: set[str] = set()
        # load id -> what we last commanded, and when a gate-driven write last
        # actually switched the relay (the dwell clock).
        self._last_want: dict[str, bool] = {}
        self._last_switch_ts: dict[str, float] = {}
        self._last_gate: dict[str, Gate] = {}
        self._published: dict[str, tuple[float, str, dict[str, Any]]] = {}

    async def _event_windows(
        self, entity: str, match: str | None, now_ts: float, now: datetime, tz: Any
    ) -> list[Window] | None:
        """The calendar's cached event windows; None when there is no usable list."""
        key = (entity, (match or "").lower())
        cached = self._events.get(key)
        last_try = self._event_try_ts.get(key)
        due = cached is None or now_ts - cached[0] >= EVENT_REFRESH_S
        if due and (last_try is None or now_ts - last_try >= EVENT_RETRY_S):
            self._event_try_ts[key] = now_ts
            try:
                found = await self._fetch(entity, now, now + EVENT_LOOKAHEAD, tz, match=match)
            except Exception as exc:
                logger.debug("Calendar %s event read failed: %s", entity, exc)
                found = None
            if found is not None:
                cached = (now_ts, found)
                self._events[key] = cached
        if cached is None or now_ts - cached[0] > EVENT_MAX_AGE_S:
            return None
        return cached[1]

    async def read_gate(self, ha: Any, load: Any, now_ts: float, tz_name: str) -> Gate:
        """Is the load in a window right now? ANY window entity being on opens it.

        A calendar WITHOUT window_event_match is gated on its bare state (cheap, and
        live); its event list only adds preheat and the next-window times. WITH a
        match the bare state cannot tell the rooms apart, so the event list is the
        gate. Other entities are gated on their state and have no preheat.
        """
        tz = pytz.timezone(tz_name)
        now = datetime.fromtimestamp(now_ts, UTC).astimezone(tz)
        match = load.window_event_match
        readings: list[Gate] = []
        unknown: list[str] = []
        for entity in load.window_entities:
            listed: list[Window] | None = None
            if is_calendar(entity):
                listed = await self._event_windows(entity, match, now_ts, now, tz)
            if is_calendar(entity) and match:
                if listed is None:
                    unknown.append(f"{entity} event list unreadable")
                    continue
                readings.append(gate_from_windows(listed, now, load.preheat_minutes))
                continue
            is_on = on_off_state(await ha.get_state_value(entity))
            if is_on is None:
                unknown.append(f"{entity} unreadable")
                continue
            from_list = gate_from_windows(listed or [], now, load.preheat_minutes)
            if is_on:
                # The state is the authority; the list only knows when it ends.
                readings.append(from_list if from_list.active else Gate(True))
            else:
                readings.append(from_list if not from_list.active else Gate(False))

        active = [g for g in readings if g.active]
        if active:
            ends = [g.window_end for g in active if g.window_end is not None]
            starts = [g.window_start for g in active if g.window_start is not None]
            return Gate(
                True,
                False,
                min(starts) if starts else None,
                max(ends) if len(ends) == len(active) else None,
            )
        if unknown:
            return Gate(None, detail="; ".join(unknown))
        coming = [g for g in readings if g.window_start is not None]
        nxt = min(coming, key=lambda g: g.window_start or now) if coming else None
        return Gate(
            False,
            any(g.preheat for g in readings),
            nxt.window_start if nxt else None,
            nxt.window_end if nxt else None,
        )

    async def _fresh_temp(self, ha: Any, load: Any, now_ts: float) -> float | None:
        """The room temperature, or None when missing, unparsable or too old."""
        state = await ha.get_state(load.temp_sensor)
        if not isinstance(state, Mapping):
            return None
        try:
            temp = float(state.get("state"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(temp):
            return None
        stamp = reading_timestamp(state)
        if stamp is None:
            return None  # freshness cannot be shown, so it is not assumed
        if now_ts - stamp.timestamp() > load.temp_max_age_minutes * 60.0:
            return None
        return temp

    async def decide(
        self,
        ha: Any,
        load: Any,
        plans: Mapping[str, float],
        now_ts: float,
        tz_name: str,
    ) -> Decision:
        """What the gate and the optional thermostat want from the relay now."""
        gate = await self.read_gate(ha, load, now_ts, tz_name)
        self._last_gate[load.id] = gate

        if gate.active is None:
            since = self._unknown_since.setdefault(load.id, now_ts)
            if now_ts - since < GATE_HOLD_S:
                if since == now_ts:
                    logger.warning(
                        "Window load %s: gate cannot be determined (%s) — holding the "
                        "relay as it is for up to %.0f min",
                        load.id, gate.detail, GATE_HOLD_S / 60.0,
                    )
                return Decision(
                    None, STATE_GATE_UNKNOWN, f"{gate.detail} — holding the relay as it is"
                )
            first = load.id not in self._fallback_logged
            self._fallback_logged.add(load.id)
            if load.id not in plans:
                if first:
                    logger.warning(
                        "Window load %s: gate unknown for over %.0f min and the load is "
                        "not in the current plan — leaving it alone",
                        load.id, GATE_HOLD_S / 60.0,
                    )
                return Decision(
                    None,
                    STATE_GATE_UNKNOWN,
                    f"{gate.detail} — and not in the current plan, leaving it alone",
                )
            want = float(plans.get(load.id) or 0.0) > 0.0
            if first:
                logger.warning(
                    "Window load %s: gate unknown for over %.0f min — following the "
                    "plan slot (%s) until it can be read again",
                    load.id, GATE_HOLD_S / 60.0, "on" if want else "off",
                )
            want, reason = self._dwell(
                load.id, want, f"{gate.detail} — following the plan slot", now_ts
            )
            return Decision(want, STATE_ON if want else STATE_OFF, reason)

        if self._unknown_since.pop(load.id, None) is not None:
            logger.info("Window load %s: gate readable again", load.id)
        self._fallback_logged.discard(load.id)

        in_window = bool(gate.active or gate.preheat)
        want = in_window
        if gate.active:
            reason = "window active"
        elif gate.preheat:
            reason = f"preheating {load.preheat_minutes:.0f} min before the window"
        else:
            reason = "no window active"

        # Optional thermostat: the comfort target inside a window, the setback
        # (if any) outside. Without a fresh reading the window alone decides.
        target = load.target_temp_c if in_window else load.setback_temp_c
        temp: float | None = None
        if load.temp_sensor and target is not None:
            temp = await self._fresh_temp(ha, load, now_ts)
            if temp is None:
                reason += "; room temperature missing or stale — window only"
                target = None
            else:
                heating = self._last_want.get(load.id)
                if heating is None:
                    # Fresh process: take the relay's own word for the dead band.
                    heating = on_off_state(await ha.get_state_value(load.switch_entity))
                want = thermostat_wants_heat(temp, target, load.hysteresis_c, heating)
                reason = (
                    f"{reason}; room {temp:.1f} C, "
                    f"{'target' if in_window else 'setback'} {target:.1f} C"
                )
        else:
            target = None
        thermostat_on = want and temp is not None

        want, reason = self._dwell(load.id, want, reason, now_ts)
        if want:
            if gate.active:
                state = STATE_ON
            elif gate.preheat:
                state = STATE_PREHEAT
            else:
                state = STATE_SETBACK if thermostat_on else STATE_ON
        else:
            state = STATE_TEMP_HOLD if in_window else STATE_OFF
        return Decision(want, state, reason, temp, target)

    def _dwell(self, load_id: str, want: bool, reason: str, now_ts: float) -> tuple[bool, str]:
        """Hold the previous command while the relay switched under MIN_DWELL_S ago."""
        last = self._last_want.get(load_id)
        switched = self._last_switch_ts.get(load_id)
        if (
            last is not None
            and want != last
            and switched is not None
            and 0.0 <= now_ts - switched < MIN_DWELL_S
        ):
            return last, (
                f"{reason}; switched under {MIN_DWELL_S / 60.0:.0f} min ago — "
                f"staying {'on' if last else 'off'}"
            )
        return want, reason

    def forget(self, load_id: str) -> None:
        """Drop what is remembered about a load we have stopped controlling (control
        pause): a human owns the plug now, so the last command, the dwell clock, the
        hold clock and the last gate reading no longer describe it. Control resumes
        like a fresh process."""
        self._last_want.pop(load_id, None)
        self._last_switch_ts.pop(load_id, None)
        self._unknown_since.pop(load_id, None)
        self._fallback_logged.discard(load_id)
        self._last_gate.pop(load_id, None)

    def note_write(
        self, load_id: str, want_on: bool, result: Any, now_ts: float, *, gate_driven: bool
    ) -> None:
        """Remember a command that went out. Only a gate-driven write that really
        switched the relay starts the dwell clock — a human's override must not make
        the next automatic switch wait."""
        if not getattr(result, "success", False):
            return
        self._last_want[load_id] = want_on
        if gate_driven and not getattr(result, "skipped", False):
            self._last_switch_ts[load_id] = now_ts

    async def publish(
        self,
        ha: Any,
        load: Any,
        state: str,
        reason: str,
        now_ts: float,
        tz_name: str,
        *,
        shadow: bool,
        room_temp: float | None = None,
        target_temp: float | None = None,
    ) -> None:
        """Publish ``sensor.darkstar_<id>_state`` — on change, or to keep it alive."""
        if ha is None:
            return
        gate = self._last_gate.get(load.id)
        tz = pytz.timezone(tz_name)

        def _iso(moment: datetime | None) -> str | None:
            return moment.astimezone(tz).isoformat(timespec="minutes") if moment else None

        attrs: dict[str, Any] = {
            "friendly_name": f"{load.name or load.id} status",
            "icon": "mdi:radiator",
            "load_id": load.id,
            "window_active": gate.active if gate else None,
            "next_window_start": _iso(gate.window_start) if gate else None,
            "next_window_end": _iso(gate.window_end) if gate else None,
            "room_temp": round(room_temp, 1) if room_temp is not None else None,
            "target_temp": target_temp,
            "reason": reason,
            "shadow_mode": shadow,
        }
        prev = self._published.get(load.id)
        if (
            prev is not None
            and prev[1] == state
            and prev[2] == attrs
            and 0.0 <= now_ts - prev[0] < PUBLISH_REFRESH_S
        ):
            return
        try:
            if await ha.set_state(status_entity_id(load.id), state, attrs):
                self._published[load.id] = (now_ts, state, attrs)
        except Exception:
            logger.debug("Window load %s: status publish failed", load.id, exc_info=True)
