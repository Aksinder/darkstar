"""Executor-side rules for spa bathing times (the backstop under the planner).

The planner books the spa so that it is at ``ready_temp_c`` when a bath starts. The
plan can still be wrong — the thermal parameters are estimates, a lid can be left
open, a kept-previous plan may predate a calendar edit — so the executor applies three
rules of its own, every tick, from LIVE reads:

1. **In window** (``start <= now < end``): never write off. The planned off is replaced
   by ``ready_temp_c``. A bath is not interrupted by an expensive hour.
2. **Catch-up** (next window not yet started): when the time left until the window
   STARTS is shorter than the time the element needs to reach ``ready_temp_c`` (plus
   ``catch_up_margin_c`` worth of heating), heat now, whatever the price says.
3. **Eco floor**: a spa that has cooled to ``eco_floor_c - 1`` while the plan says off
   is heated at ``floor_write_temp_c`` until it is back at ``eco_floor_c + 1``. The
   write value sits above the bridge's 30 C ``fan_only`` threshold on purpose.

Every rule fails safe to "do nothing" (the legacy path): unknown windows disable rules
1-2, an unknown temperature disables rules 2-3, and a window is never invented. Only
tz-aware datetimes are accepted — the container runs local time and a naive/UTC mixup
here would silently shift every bath by an hour or two.

Precedence lives in the engine: control pause, fuse shed and the manual override all
run BEFORE these rules; surplus-boost, idle-hold, drift and the write run after.

This module must not import ``planner/`` (the executor never does); the shared
physics lives in ``backend.core.spa_thermal``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pytz

from backend.core.load_windows import (
    Window,
    fetch_calendar_windows,
    is_calendar,
    merge_windows,
    on_off_state,
)
from backend.core.spa_bathing_config import BathingConfig, load_learned_params
from backend.core.spa_thermal import heat_rate_c_per_h, hours_to_reach_c

from .window_load import EVENT_LOOKAHEAD, EVENT_MAX_AGE_S, EVENT_REFRESH_S, EVENT_RETRY_S

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)

RULE_IN_WINDOW = "in_window"
RULE_CATCH_UP = "catch_up"
RULE_ECO_FLOOR = "eco_floor"
RULE_NONE = "none"

# A spa water temperature outside this range is a sensor glitch, not water.
SPA_TEMP_PLAUSIBLE_C = (0.0, 50.0)
# Ambient (outdoor) temperature plausibility; outside => the configured default.
AMBIENT_PLAUSIBLE_C = (-40.0, 45.0)
# Learned thermal parameters are re-read from disk no more often than this.
LEARNED_PARAMS_REFRESH_S = 900.0

_UNREADABLE = ("", "none", "unknown", "unavailable")


@dataclass(frozen=True)
class BathingDecision:
    """What the bathing rules want this tick. ``none`` = leave the planned temp alone."""

    action: Literal["none", "heat", "no_off"]
    temp_c: int | None
    rule: str
    reason: str


NO_DECISION = BathingDecision("none", None, RULE_NONE, "")


@dataclass(frozen=True)
class BathingStep:
    """What the engine does with a LIVE (non-shadow) decision this tick."""

    temp_c: int | None  # replaces the planned temperature when set
    skip_idle_hold: bool
    skip_surplus_boost: bool


def _is_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() is not None


def _cap(temp_c: float, device_max_temp: int | None) -> int:
    value = round(temp_c)
    if device_max_temp is not None:
        value = min(value, int(device_max_temp))
    return int(value)


def decide_bathing_rules(
    *,
    cfg: BathingConfig,
    now: datetime,
    windows: Sequence[tuple[datetime, datetime]] | None,
    measured_temp_c: float | None,
    planned_temp: int,
    off_temp: int,
    normal_temp: int,
    device_max_temp: int | None,
    power_kw: float,
    ambient_c: float | None,
    floor_latched: bool,
    heating_now: bool | None,
) -> tuple[BathingDecision, bool]:
    """The bathing rules for one tick: ``(decision, new_floor_latched)``.

    Pure. First match wins: in-window, catch-up, eco floor. ``windows`` None means
    "unknown" (never "no bath"); ``measured_temp_c`` None means unknown. ``now`` and
    every window bound must be tz-aware, otherwise no window rule fires.
    ``normal_temp`` is accepted for symmetry with the plan mapping; the rules write
    ``ready_temp_c``, which is what the planner aims at.
    """
    del normal_temp  # the rules aim at ready_temp_c, not the device's normal setpoint
    plan_says_off = int(planned_temp) <= int(off_temp)
    temp = (
        measured_temp_c if measured_temp_c is not None and math.isfinite(measured_temp_c) else None
    )
    if temp is not None and not (SPA_TEMP_PLAUSIBLE_C[0] <= temp <= SPA_TEMP_PLAUSIBLE_C[1]):
        temp = None

    # Naive datetimes are refused outright (see the module header).
    usable_windows: list[tuple[datetime, datetime]] | None
    if windows is None or not _is_aware(now):
        usable_windows = None
    elif any(not (_is_aware(s) and _is_aware(e)) for s, e in windows):
        logger.error("spa-bathing: naive window datetime refused — window rules disabled this tick")
        usable_windows = None
    else:
        usable_windows = sorted((s, e) for s, e in windows if e > s)

    # --- Eco-floor latch bookkeeping (temperature only) -------------------------
    latched = floor_latched
    if temp is not None:
        if latched and temp >= cfg.eco_floor_c + 1.0:
            latched = False
        elif not latched and temp <= cfg.eco_floor_c - 1.0 and heating_now is not True:
            latched = True

    # --- 1. In window: never write off -------------------------------------------
    if usable_windows is not None:
        active = next((w for w in usable_windows if w[0] <= now < w[1]), None)
        if active is not None:
            if not plan_says_off:
                return NO_DECISION, latched
            return (
                BathingDecision(
                    "no_off",
                    _cap(cfg.ready_temp_c, device_max_temp),
                    RULE_IN_WINDOW,
                    f"bad pågår till {active[1].isoformat()} — ingen av-skrivning",
                ),
                latched,
            )

    # --- 2. Catch-up for the next window -------------------------------------------
    if usable_windows is not None and temp is not None and power_kw > 0:
        upcoming = [w for w in usable_windows if w[0] > now]
        if upcoming:
            start = upcoming[0][0]
            time_left_h = (start - now).total_seconds() / 3600.0
            amb = (
                ambient_c
                if ambient_c is not None and math.isfinite(ambient_c)
                else cfg.ambient_default_c
            )
            c = cfg.heat_capacity_kwh_per_c
            k = cfg.cooling_k_per_h
            reach_h = hours_to_reach_c(temp, cfg.ready_temp_c, amb, power_kw, c, k)
            rate = heat_rate_c_per_h(power_kw, c)
            margin_h = cfg.catch_up_margin_c / rate if rate > 0 else math.inf
            behind = reach_h is None or time_left_h < reach_h + margin_h
            if behind and temp < cfg.ready_temp_c:
                if not plan_says_off:
                    return NO_DECISION, latched
                need = (
                    "ej nåbar"
                    if reach_h is None
                    else f"behöver {reach_h + margin_h:.2f} h, {time_left_h:.2f} h kvar"
                )
                return (
                    BathingDecision(
                        "heat",
                        _cap(cfg.ready_temp_c, device_max_temp),
                        RULE_CATCH_UP,
                        f"spa: efter schemat, värmer utanför plan "
                        f"({temp:.1f} -> {cfg.ready_temp_c:.1f} C till "
                        f"{start.isoformat()}: {need})",
                    ),
                    latched,
                )

    # --- 3. Eco floor ---------------------------------------------------------------
    if temp is not None and latched and plan_says_off:
        return (
            BathingDecision(
                "heat",
                _cap(cfg.floor_write_temp_c, device_max_temp),
                RULE_ECO_FLOOR,
                f"under eko-golvet ({temp:.1f} C <= {cfg.eco_floor_c - 1.0:.1f} C), "
                f"värmer till {cfg.eco_floor_c + 1.0:.1f} C",
            ),
            latched,
        )
    return NO_DECISION, latched


def window_active(
    windows: Sequence[tuple[datetime, datetime]] | None, now: datetime
) -> bool | None:
    """True/False when known, None when the windows (or ``now``) are unusable."""
    if windows is None or not _is_aware(now):
        return None
    if any(not (_is_aware(s) and _is_aware(e)) for s, e in windows):
        return None
    return any(s <= now < e for s, e in windows)


def _parse_temp(raw: Any, lo: float, hi: float) -> float | None:
    if raw is None or str(raw).strip().lower() in _UNREADABLE:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not (lo <= value <= hi):
        return None
    return value


class SpaBathingRuntime:
    """Per-engine memory for the bathing rules: event cache, latch, logged state."""

    def __init__(
        self,
        fetch: Callable[..., Awaitable[list[Window] | None]] = fetch_calendar_windows,
        learned: Callable[[], dict[str, float] | None] = load_learned_params,
    ) -> None:
        self._fetch = fetch
        self._learned = learned
        # (entity, match) -> (fetched_at_ts, windows)
        self._events: dict[tuple[str, str], tuple[float, list[Window]]] = {}
        self._event_try_ts: dict[tuple[str, str], float] = {}
        # device id -> eco-floor latch
        self.floor_latched: dict[str, bool] = {}
        # device id -> last logged (shadow, action, temp, rule)
        self._logged: dict[str, tuple[bool, str, int | None, str]] = {}
        self._learned_cache: tuple[float, dict[str, float] | None] | None = None

    async def _calendar(
        self, entity: str, match: str | None, now_ts: float, now: datetime, tz: Any
    ) -> list[Window] | None:
        key = (entity, (match or "").lower())
        cached = self._events.get(key)
        last_try = self._event_try_ts.get(key)
        due = cached is None or now_ts - cached[0] >= EVENT_REFRESH_S
        if due and (last_try is None or now_ts - last_try >= EVENT_RETRY_S):
            self._event_try_ts[key] = now_ts
            try:
                found = await self._fetch(entity, now, now + EVENT_LOOKAHEAD, tz, match=match)
            except Exception as exc:
                logger.debug("spa-bathing: calendar %s read failed: %s", entity, exc)
                found = None
            if found is not None:
                cached = (now_ts, found)
                self._events[key] = cached
        if cached is None or now_ts - cached[0] > EVENT_MAX_AGE_S:
            return None
        return cached[1]

    async def windows(
        self, cfg: BathingConfig, now_ts: float, tz_name: str, ha: Any = None
    ) -> list[Window] | None:
        """Bathing windows that have not ended yet, tz-aware; None = unknown.

        A calendar contributes its (cached) events. Any other entity has no future:
        on = one window from now over the lookahead, off = none — and it needs ``ha``
        to be read. Any entity that cannot be read makes the whole answer unknown.
        """
        if not cfg.window_entities:
            return None
        tz = pytz.timezone(tz_name)
        now = datetime.fromtimestamp(now_ts, UTC).astimezone(tz)
        match = cfg.window_event_match.strip() or None
        found: list[Window] = []
        for entity in cfg.window_entities:
            if is_calendar(entity):
                listed = await self._calendar(entity, match, now_ts, now, tz)
                if listed is None:
                    return None
                found.extend(listed)
                continue
            if ha is None:
                return None
            try:
                is_on = on_off_state(await ha.get_state_value(entity))
            except Exception as exc:
                logger.debug("spa-bathing: %s read failed: %s", entity, exc)
                is_on = None
            if is_on is None:
                return None
            if is_on:
                found.append((now, now + EVENT_LOOKAHEAD))
        if any(not (_is_aware(s) and _is_aware(e)) for s, e in found):
            logger.error("spa-bathing: calendar returned naive datetimes — windows unknown")
            return None
        return [w for w in merge_windows(found) if w[1] > now]

    async def read_temp_c(self, ha: Any, entity: str) -> float | None:
        """The spa water temperature, 0..50 C, else None.

        No staleness check: the LayZSpa sensor only reports on change, so a steady
        tub legitimately has an old timestamp.
        """
        if not entity:
            return None
        try:
            raw = await ha.get_state_value(entity)
        except Exception as exc:
            logger.debug("spa-bathing: %s read failed: %s", entity, exc)
            return None
        return _parse_temp(raw, *SPA_TEMP_PLAUSIBLE_C)

    async def read_ambient_c(self, ha: Any, cfg: BathingConfig) -> float:
        """The ambient temperature, or ``ambient_default_c`` when unset/unreadable."""
        if cfg.ambient_sensor:
            try:
                raw = await ha.get_state_value(cfg.ambient_sensor)
            except Exception:
                raw = None
            value = _parse_temp(raw, *AMBIENT_PLAUSIBLE_C)
            if value is not None:
                return value
        return cfg.ambient_default_c

    def effective_config(self, cfg: BathingConfig, now_ts: float | None = None) -> BathingConfig:
        """``cfg`` with learned C and k swapped in when ``learn`` is on and valid."""
        if not cfg.learn:
            return cfg
        ts = time.time() if now_ts is None else now_ts
        cached = self._learned_cache
        if cached is None or ts - cached[0] >= LEARNED_PARAMS_REFRESH_S:
            try:
                params = self._learned()
            except Exception as exc:
                logger.debug("spa-bathing: learned params unreadable: %s", exc)
                params = None
            cached = (ts, params)
            self._learned_cache = cached
        params = cached[1]
        if not params:
            return cfg
        try:
            c = float(params["heat_capacity_kwh_per_c"])
            k = float(params["cooling_k_per_h"])
        except (KeyError, TypeError, ValueError):
            return cfg
        if not (math.isfinite(c) and math.isfinite(k) and c > 0 and k > 0):
            return cfg
        return replace(cfg, heat_capacity_kwh_per_c=c, cooling_k_per_h=k)

    def log_transition(self, device_id: str, decision: BathingDecision, *, shadow: bool) -> None:
        """Log a decision once per change (this runs every tick)."""
        key = (shadow, decision.action, decision.temp_c, decision.rule)
        if self._logged.get(device_id) == key:
            return
        self._logged[device_id] = key
        if shadow:
            if decision.action == "none":
                logger.info("[SHADOW spa-bathing] %s: would follow the plan", device_id)
            else:
                logger.info(
                    "[SHADOW spa-bathing] %s: would %s %s (%s: %s)",
                    device_id,
                    decision.action,
                    decision.temp_c,
                    decision.rule,
                    decision.reason,
                )
            return
        if decision.action == "none":
            logger.info("spa-bathing %s: rules idle, following the plan", device_id)
        elif decision.rule == RULE_CATCH_UP:
            logger.warning(
                "spa-bathing %s: %s -> %s C", device_id, decision.reason, decision.temp_c
            )
        else:
            logger.info(
                "spa-bathing %s: %s -> %s C (%s)",
                device_id,
                decision.rule,
                decision.temp_c,
                decision.reason,
            )
