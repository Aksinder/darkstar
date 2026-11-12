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
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
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

from .window_price import (
    PHASE_EARLY,
    PHASE_PEAK_OFF,
    RuleState,
    end_peak_off,
    rule_state_from_attributes,
    step,
    thresholds,
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
# Our own status sensor is trusted as a record of the price-rule state after a
# restart only while it is this fresh (it is re-published every PUBLISH_REFRESH_S).
RULE_STATE_RESTORE_MAX_AGE_S = 2 * PUBLISH_REFRESH_S

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
STATE_EARLY_START = "early_start"
STATE_PEAK_OFF = "peak_off"
STATE_DWELL_HOLD = "dwell_hold"
STATE_PAUSE_UNKNOWN = "pause_unknown"

PAUSE_PAUSED = "paused"
PAUSE_HOLD = "hold"
PAUSE_CLEAR = "clear"

# A room temperature outside this range is a glitch, not a room.
TEMP_PLAUSIBLE_C = (-30.0, 50.0)
_UNREADABLE = ("", "none", "unknown", "unavailable")


@dataclass(frozen=True)
class Gate:
    """One reading of a load's window. ``active`` None = could not be determined."""

    active: bool | None
    preheat: bool = False
    # The active window, or the next one when none is active (None = not known).
    window_start: datetime | None = None
    window_end: datetime | None = None
    detail: str = ""
    # False when a calendar's event list could not be read: window_start then says
    # nothing about the future ("no known window" is not "no window").
    schedule_known: bool = True


@dataclass(frozen=True)
class Decision:
    """``want_on`` None = leave the relay exactly as it is (no write)."""

    want_on: bool | None
    state: str
    reason: str
    room_temp: float | None = None
    target_temp: float | None = None
    # Price rules (None when the rule is off or the price is unknown).
    price_now: float | None = None
    early_threshold: float | None = None
    peak_threshold: float | None = None
    # Something a human should hear about once: ("alert" | "recovered", text).
    notice: tuple[str, str] | None = None


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
    """An HA object-id fragment: [a-z0-9_] only, underscores collapsed. HA rejects
    anything else (non-ASCII letters, double underscores) on every publish."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "load"


def status_entity_id(load_id: str) -> str:
    # "_heating_state", not "_state": the deferrable appliances publish
    # sensor.darkstar_<id>_state and an id shared between the two must not collide.
    return f"sensor.darkstar_{_slug(load_id)}_heating_state"


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
        # Price rules: the early-start latch and the peak-off clocks, per load.
        # Restored after a restart from our own status sensor while that is fresh
        # (restore_rule_state), otherwise started safe (_seed_rule_state).
        self._rule_state: dict[str, RuleState] = {}
        # load id -> was the window open at the last reading (to confirm a close).
        self._raw_in_window: dict[str, bool] = {}
        # Thermostat: (last decision, for the comfort side?) and when its reading
        # first went stale.
        self._thermo_last: dict[str, tuple[bool, bool]] = {}
        self._thermo_stale_since: dict[str, float] = {}
        # load id -> when its pause helper first could not be read (this episode).
        self._pause_unknown_since: dict[str, float] = {}
        self._pause_unknown_logged: set[str] = set()
        # One-time notices for a human, picked up by the engine (pop_notice).
        self._notices: dict[str, tuple[str, str]] = {}
        # The last gate that COULD be determined, and when the plug would next have
        # come on by the last readable event list (keeps an early-start latch alive
        # through an unreadable list).
        self._last_known_gate: dict[str, Gate] = {}
        self._last_next_on: dict[str, datetime] = {}
        self._restore_tried: set[str] = set()
        self._price_memo: dict[str, tuple[tuple[int, float], float, list[float]]] = {}
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
        schedule_known = True
        for entity in load.window_entities:
            listed: list[Window] | None = None
            if is_calendar(entity):
                listed = await self._event_windows(entity, match, now_ts, now, tz)
                schedule_known = schedule_known and listed is not None
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
                schedule_known=schedule_known,
            )
        if unknown:
            return Gate(None, detail="; ".join(unknown), schedule_known=schedule_known)
        coming = [g for g in readings if g.window_start is not None]
        nxt = min(coming, key=lambda g: g.window_start or now) if coming else None
        return Gate(
            False,
            any(g.preheat for g in readings),
            nxt.window_start if nxt else None,
            nxt.window_end if nxt else None,
            schedule_known=schedule_known,
        )

    async def _read_temp(
        self, ha: Any, load: Any, now_ts: float
    ) -> tuple[float | None, bool]:
        """(room temperature in C, is it fresh). (None, False) when there is no
        usable reading at all: missing, unparsable, implausible, or undated.

        Freshness is the age of HA's last REPORT (last_reported, falling back to
        last_updated/last_changed on an old HA): a Zigbee sensor in a steady room
        reports the same value for hours, which moves last_reported but not
        last_updated. Fahrenheit is converted; a value outside TEMP_PLAUSIBLE_C is a
        glitch and must not start setback heating or stop a heater.
        """
        state = await ha.get_state(load.temp_sensor)
        if not isinstance(state, Mapping):
            return None, False
        try:
            temp = float(state.get("state"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None, False
        if not math.isfinite(temp):
            return None, False
        attrs = state.get("attributes")
        unit = str(attrs.get("unit_of_measurement") or "") if isinstance(attrs, Mapping) else ""
        if unit.strip().upper().lstrip("°").strip() in ("F", "℉"):
            temp = (temp - 32.0) * 5.0 / 9.0
        if not TEMP_PLAUSIBLE_C[0] <= temp <= TEMP_PLAUSIBLE_C[1]:
            return None, False
        stamp = reading_timestamp(state)
        if stamp is None:
            return None, False  # freshness cannot be shown, so it is not assumed
        return temp, now_ts - stamp.timestamp() <= load.temp_max_age_minutes * 60.0

    async def pause_state(self, ha: Any, load: Any, now_ts: float) -> tuple[str, str]:
        """Is this load control-paused? (PAUSE_PAUSED | PAUSE_HOLD | PAUSE_CLEAR, why).

        Everywhere else an unreadable pause helper reads as NOT paused. For a room
        heater that is the wrong way round: with the pause on and the plug set by
        hand, one tick where the helper reads "unavailable" would let the gate
        switch the plug. So here an unreadable helper HOLDS — no write this tick —
        for at most GATE_HOLD_S, after which it counts as not paused (logged once),
        so a helper that is gone for good cannot park the heater for ever.
        """
        unreadable: list[str] = []
        for entity in load.control_pause_entities:
            raw = await ha.get_state_value(entity)
            value = str(raw).strip().lower() if raw is not None else ""
            if value == "on":
                self._pause_readable_again(load)
                return PAUSE_PAUSED, entity
            if value in _UNREADABLE:
                unreadable.append(entity)
        if not unreadable:
            self._pause_readable_again(load)
            return PAUSE_CLEAR, ""
        since = self._pause_unknown_since.setdefault(load.id, now_ts)
        detail = f"pause helper {', '.join(unreadable)} unreadable"
        if now_ts - since < GATE_HOLD_S:
            if since == now_ts:
                logger.warning(
                    "Window load %s: %s — leaving the plug alone for up to %.0f min",
                    load.id, detail, GATE_HOLD_S / 60.0,
                )
            return PAUSE_HOLD, f"{detail} — leaving the plug alone"
        if load.id not in self._pause_unknown_logged:
            self._pause_unknown_logged.add(load.id)
            logger.warning(
                "Window load %s: %s for over %.0f min — treating it as NOT paused",
                load.id, detail, GATE_HOLD_S / 60.0,
            )
            self._notices[load.id] = (
                "alert",
                f"{load.name or load.id}: {detail} i över {GATE_HOLD_S / 60.0:.0f} min — "
                "elementet räknas som INTE pausat och styrs igen.",
            )
        return PAUSE_CLEAR, ""

    def _pause_readable_again(self, load: Any) -> None:
        self._pause_unknown_since.pop(load.id, None)
        if load.id in self._pause_unknown_logged:
            self._pause_unknown_logged.discard(load.id)
            self._notices[load.id] = (
                "recovered",
                f"{load.name or load.id}: paushjälparen går att läsa igen.",
            )

    async def restore_rule_state(self, ha: Any, load: Any, now_ts: float) -> None:
        """Once per process and load: pick the price-rule state up from our own
        status sensor, if that was published within RULE_STATE_RESTORE_MAX_AGE_S.

        Without this a restart during a peak-off leaves a plug that is off with
        nobody remembering why — and if the gate or the pause helper happens to be
        unreadable just then, the "hold as it is" branches would keep it off. With
        it, the peak-off continues on its REAL clock (and can be ended by a hold).
        An older or missing sensor is not trusted: _seed_rule_state then starts
        from the safe reading of the plug instead.
        """
        if load.id in self._restore_tried or not load.price_rules.enabled:
            return
        self._restore_tried.add(load.id)
        if load.id in self._rule_state:
            return
        try:
            status = await ha.get_state(status_entity_id(load.id))
        except Exception:
            return
        if not isinstance(status, Mapping):
            return
        stamp = reading_timestamp(status)
        if stamp is None or not 0 <= now_ts - stamp.timestamp() <= RULE_STATE_RESTORE_MAX_AGE_S:
            return
        attrs = status.get("attributes")
        restored = rule_state_from_attributes(attrs if isinstance(attrs, Mapping) else None)
        if restored is not None and restored != RuleState():
            self._rule_state[load.id] = restored
            logger.info("Window load %s: price-rule state restored: %s", load.id, restored)

    def pop_notice(self, load_id: str) -> tuple[str, str] | None:
        """A pending one-time notice for a human ("alert" | "recovered", text)."""
        return self._notices.pop(load_id, None)

    def end_peak_off_for_hold(self, load_id: str, now_ts: float) -> bool:
        """True when the load is in a price peak-off that must be ended (plug ON)
        before a "leave the plug as it is" branch takes over.

        THE INVARIANT of the price rule: a plug is never off on account of the price
        for longer than peak_off_max_minutes (plus one dwell). Every hold — unknown
        gate, unreadable pause helper, a pause switched on — stops running the
        rule's clock, so a plug that is off for the price when the hold begins would
        stay off for as long as the hold lasts. It is switched on first.
        """
        state = self._rule_state.get(load_id)
        if state is None or state.peak_off_since is None:
            return False
        self._rule_state[load_id] = end_peak_off(state, datetime.fromtimestamp(now_ts, UTC))
        return True

    async def decide(
        self,
        ha: Any,
        load: Any,
        plans: Mapping[str, float],
        now_ts: float,
        tz_name: str,
        *,
        price: float | None = None,
        price_window: Sequence[float] = (),
    ) -> Decision:
        """What the gate, the price rules and the optional thermostat want from the
        relay now. ``price`` / ``price_window`` are only read when the load has a
        price rule; an unknown price (None) switches the rules off, never the heater.
        """
        gate = await self.read_gate(ha, load, now_ts, tz_name)
        self._last_gate[load.id] = gate

        rules = load.price_rules
        if gate.active is None:
            since = self._unknown_since.setdefault(load.id, now_ts)
            if self.end_peak_off_for_hold(load.id, now_ts):
                # Off for the price when the gate went dark: on first, then hold.
                logger.warning(
                    "Window load %s: gate cannot be determined (%s) during a price "
                    "peak-off — switching ON before holding",
                    load.id, gate.detail,
                )
                return Decision(
                    True,
                    STATE_GATE_UNKNOWN,
                    f"{gate.detail} — price peak-off ended, then holding the relay",
                )
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
            # Said ONCE to a human as well: from here on the heater runs on a plan
            # slot or not at all, which nobody would notice from the log.
            notice = (
                (
                    "alert",
                    f"{load.name or load.id}: {gate.detail} i över "
                    f"{GATE_HOLD_S / 60.0:.0f} min — elementet styrs inte längre av "
                    "kalendern.",
                )
                if first
                else None
            )
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
                    notice=notice,
                )
            want = float(plans.get(load.id) or 0.0) > 0.0
            if first:
                logger.warning(
                    "Window load %s: gate unknown for over %.0f min — following the "
                    "plan slot (%s) until it can be read again",
                    load.id, GATE_HOLD_S / 60.0, "on" if want else "off",
                )
            reason = f"{gate.detail} — following the plan slot"
            if rules.peak_enabled and self._believed_heating(load.id, now_ts):
                # The plan's 0 may be a peak-off the PLANNER simulated, and every
                # replan can start a new one: the executor's clocks do not run while
                # the gate is dark. So while the last readable gate says the heater
                # belongs on, a plan "off" is treated as what it is most likely to
                # be — a price peak — and bounded by the same max-off / min-on
                # clocks as a live one.
                now = datetime.fromtimestamp(now_ts, UTC)
                state = self._rule_state[load.id]
                state, verdict = step(
                    rules, state, now=now, in_window=True, next_on=None,
                    price=1.0 if not want else 0.0,
                    early_threshold=None, peak_threshold=1.0,
                )
                self._rule_state[load.id] = state
                if verdict.heat and not want:
                    reason += "; plan says off, but the peak-off time bound keeps it on"
                want = verdict.heat
            want, reason = self._dwell(load.id, want, reason, now_ts)
            return Decision(want, STATE_ON if want else STATE_OFF, reason, notice=notice)

        notice = None
        if self._unknown_since.pop(load.id, None) is not None:
            logger.info("Window load %s: gate readable again", load.id)
            if load.id in self._fallback_logged:
                notice = (
                    "recovered",
                    f"{load.name or load.id}: kalendern/grinden går att läsa igen.",
                )
        self._fallback_logged.discard(load.id)

        # A window that CLOSES is confirmed on the next tick before the plug is
        # switched off: a calendar coming back from "unavailable" can read "off"
        # for one tick, and that tick would cost five minutes of heat (the dwell).
        # Opening is immediate, and a fresh process has nothing to confirm.
        self._last_known_gate[load.id] = gate
        raw_in_window = bool(gate.active or gate.preheat)
        closing = not raw_in_window and self._raw_in_window.get(load.id) is True
        self._raw_in_window[load.id] = raw_in_window
        in_window = raw_in_window or closing
        if gate.active:
            reason = "window active"
        elif gate.preheat:
            reason = f"preheating {load.preheat_minutes:.0f} min before the window"
        elif closing:
            reason = "window closed — confirming on the next tick"
        else:
            reason = "no window active"

        # The room temperature is read at most once, and only when something below
        # needs it.
        temp_memo: list[tuple[float | None, bool]] = []

        async def room_temp() -> tuple[float | None, bool]:
            if not temp_memo:
                temp_memo.append(await self._read_temp(ha, load, now_ts))
            return temp_memo[0]

        # Price rules: cheap early start before the window, off at price peaks.
        heat = in_window
        phase: str | None = None
        early_thr: float | None = None
        peak_thr: float | None = None
        if rules.enabled:
            now = datetime.fromtimestamp(now_ts, UTC)
            next_on = (
                gate.window_start - timedelta(minutes=load.preheat_minutes)
                if not in_window and gate.window_start is not None
                else None
            )
            if next_on is not None:
                self._last_next_on[load.id] = next_on
            elif not in_window and not gate.schedule_known:
                # The event list is unreadable, so "no next window" is not known —
                # only not readable. A latch must not be dropped over that: keep
                # the start time it was held for, until that time passes or a
                # fresh list says the window moved or vanished.
                remembered = self._last_next_on.get(load.id)
                if remembered is not None and remembered > now:
                    next_on = remembered
            else:
                self._last_next_on.pop(load.id, None)
            state = self._rule_state.get(load.id)
            if state is None:
                state = await self._seed_rule_state(ha, load, now, in_window, next_on)
            early_thr, peak_thr = thresholds(rules, price_window)
            keep_warm = False
            if (
                rules.peak_enabled
                and load.temp_sensor
                and load.target_temp_c is not None
                and (in_window or state.early_latched)
            ):
                room, room_fresh = await room_temp()
                keep_warm = room is not None and room_fresh and room < (
                    load.target_temp_c - load.hysteresis_c - rules.peak_off_temp_margin_c
                )
            state, verdict = step(
                rules,
                state,
                now=now,
                in_window=in_window,
                next_on=next_on,
                price=price,
                early_threshold=early_thr,
                peak_threshold=peak_thr,
                keep_warm=keep_warm,
            )
            self._rule_state[load.id] = state
            heat = verdict.heat
            phase = verdict.phase
            if phase in (PHASE_EARLY, PHASE_PEAK_OFF):
                reason = verdict.reason
            elif in_window:
                reason += verdict.reason.removeprefix("window active")
        want = heat

        # Optional thermostat: the comfort target while heat is wanted, the setback
        # (if any) otherwise. Without a fresh reading the window alone decides.
        target = load.target_temp_c if heat else load.setback_temp_c
        temp: float | None = None
        thermostat_on = False
        if load.temp_sensor and target is not None:
            temp, fresh = await room_temp()
            side = "target" if heat else "setback"
            if temp is not None and fresh:
                heating = self._last_want.get(load.id)
                if heating is None:
                    # Fresh process: take the relay's own word for the dead band.
                    heating = on_off_state(await ha.get_state_value(load.switch_entity))
                want = thermostat_wants_heat(temp, target, load.hysteresis_c, heating)
                thermostat_on = want
                self._thermo_last[load.id] = (want, heat)
                self._thermo_stale_since.pop(load.id, None)
                reason = f"{reason}; room {temp:.1f} C, {side} {target:.1f} C"
            else:
                # No fresh reading. A sensor in a steady room can simply be quiet,
                # so the last thermostat decision stands for one more max-age
                # period before the window alone decides.
                last = self._thermo_last.get(load.id)
                stale_since = self._thermo_stale_since.setdefault(load.id, now_ts)
                if (
                    last is not None
                    and last[1] == heat
                    and now_ts - stale_since <= load.temp_max_age_minutes * 60.0
                ):
                    want = last[0]
                    thermostat_on = want
                    reason += (
                        "; room temperature missing or stale — holding the last "
                        f"thermostat decision ({'on' if want else 'off'})"
                    )
                else:
                    self._thermo_last.pop(load.id, None)
                    reason += "; room temperature missing or stale — window only"
                    temp = None
                    target = None
        else:
            target = None

        wanted = want
        want, reason = self._dwell(load.id, want, reason, now_ts)
        if want:
            if phase == PHASE_PEAK_OFF:
                # Only the setback floor (or the dwell) keeps it on through a peak.
                state_label = STATE_SETBACK if thermostat_on else STATE_ON
            elif gate.active:
                state_label = STATE_ON
            elif gate.preheat:
                state_label = STATE_PREHEAT
            elif phase == PHASE_EARLY:
                state_label = STATE_EARLY_START
            else:
                state_label = STATE_SETBACK if thermostat_on else STATE_ON
        elif phase == PHASE_PEAK_OFF:
            state_label = STATE_PEAK_OFF
        else:
            state_label = STATE_TEMP_HOLD if heat else STATE_OFF
        if want != wanted:
            # The dwell overruled the decision: say so, instead of a state that
            # blames the thermostat or claims a window.
            state_label = STATE_DWELL_HOLD
        return Decision(
            want,
            state_label,
            reason,
            temp,
            target,
            price if rules.enabled else None,
            early_thr,
            peak_thr,
            notice,
        )

    async def _seed_rule_state(
        self, ha: Any, load: Any, now: datetime, in_window: bool, next_on: datetime | None
    ) -> RuleState:
        """The price-rule state of a load we have no memory of: control just handed
        back after a pause, or a fresh process whose status sensor was too old (or
        absent) to restore from (restore_rule_state). The start is made SAFE:

        * the min-on clock starts now, so no peak-off can begin until the heater has
          been on for the full peak_off_min_on_minutes — a restart can never grant a
          fresh off-stretch right after one ended (a restart IN a peak-off ends it);
        * a plug that is ON inside the early-start span is taken to be latched, so a
          restart does not switch off a heater that started on last night's cheap
          price just because the price has risen since.
        """
        rules = load.price_rules
        latched = False
        if (
            rules.early_enabled
            and not in_window
            and next_on is not None
            and next_on - timedelta(hours=rules.early_start_hours) <= now < next_on
        ):
            latched = on_off_state(await ha.get_state_value(load.switch_entity)) is True
        return RuleState(
            early_latched=latched, heat_since=now if (in_window or latched) else None
        )

    async def price_inputs(
        self,
        load: Any,
        now_ts: float,
        read_price: Callable[[], Awaitable[float | None]],
        read_window: Callable[[float], Awaitable[list[float]]],
    ) -> tuple[float | None, list[float]]:
        """(current import price, rolling price window) for a load's price rules,
        from the executor's own price source. Kept per load for the 15-minute slot —
        the feed's own resolution; an hour's memo acted on a price up to 59 minutes
        old — but an UNKNOWN price is never kept: it is asked for again on the next
        tick. Any failure reads as unknown."""
        hours = float(load.price_rules.price_window_hours)
        key = (int(now_ts // 900), hours)
        memo = self._price_memo.get(load.id)
        if memo is not None and memo[0] == key:
            return memo[1], memo[2]
        try:
            price = await read_price()
            window = list(await read_window(hours)) if price is not None else []
        except Exception as exc:
            logger.debug("Window load %s: price read failed: %s", load.id, exc)
            return None, []
        if price is None or not window:
            return None, []
        self._price_memo[load.id] = (key, price, window)
        return price, window

    def _believed_heating(self, load_id: str, now_ts: float) -> bool:
        """By what was last known, does this heater belong on now? The rule state
        says whether a heat period was running (it is reset whenever nothing asks
        for heat, and survives a restart through the status sensor); the last
        readable gate says when that window was due to end, if it knew."""
        state = self._rule_state.get(load_id)
        if state is None or not (
            state.early_latched
            or state.heat_since is not None
            or state.peak_off_since is not None
        ):
            return False
        last = self._last_known_gate.get(load_id)
        return (
            last is None
            or last.window_end is None
            or now_ts < last.window_end.timestamp()
        )

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
        self._rule_state.pop(load_id, None)
        self._raw_in_window.pop(load_id, None)
        self._last_known_gate.pop(load_id, None)
        self._last_next_on.pop(load_id, None)
        self._thermo_last.pop(load_id, None)
        self._thermo_stale_since.pop(load_id, None)

    def note_write(
        self, load_id: str, want_on: bool, result: Any, now_ts: float, *, gate_driven: bool
    ) -> None:
        """Remember a command. ``_last_want`` follows what the plug is known to be
        doing (already there, or switched and read back). The dwell clock starts only
        on a gate-driven write that VERIFIABLY switched the relay: a human's override
        must not make the next automatic switch wait, and neither must a write to an
        unreachable plug that switched nothing."""
        if not getattr(result, "success", False):
            return
        skipped = bool(getattr(result, "skipped", False))
        switched = not skipped and getattr(result, "verification_success", None) is True
        if skipped or switched:
            self._last_want[load_id] = want_on
        if gate_driven and switched:
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
        price_now: float | None = None,
        early_threshold: float | None = None,
        peak_threshold: float | None = None,
    ) -> None:
        """Publish ``sensor.darkstar_<id>_heating_state`` — on change, or to keep it
        alive."""
        if ha is None:
            return
        gate = self._last_gate.get(load.id)
        tz = pytz.timezone(tz_name)

        def _iso(moment: datetime | None) -> str | None:
            return moment.astimezone(tz).isoformat(timespec="minutes") if moment else None

        def _round(value: float | None) -> float | None:
            return round(value, 4) if value is not None else None

        rule = self._rule_state.get(load.id, RuleState())
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
            # Price rules: null when the rule is off (or the price unknown).
            "price_now": _round(price_now),
            "early_start_threshold": _round(early_threshold),
            "peak_off_threshold": _round(peak_threshold),
            # The rules' memory, so the planner can start its forward simulation
            # from where the plug really is.
            "early_start_latched": rule.early_latched,
            "peak_off_since": _iso(rule.peak_off_since),
            "heat_since": _iso(rule.heat_since),
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
