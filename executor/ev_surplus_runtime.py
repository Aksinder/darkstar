"""
EV surplus controller — runtime wiring (config + live read + actuation).

Wraps the pure control law in ``ev_surplus.py``: reads the live sensors and each charger's
state from HA, calls ``compute_ev_surplus``, then actuates through the write-guard. Default
OFF. Tesla current is set via ``number.set_value``; the Easee via its flash-SAFE
``easee.set_charger_dynamic_limit`` service (NEVER the non-dynamic max/circuit limits).

Kept separate from the engine so the integration there is a two-line construct-and-call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .ev_surplus import (
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    WriteGuardConfig,
    battery_tier_active,
    compute_ev_surplus,
    should_write_current,
)

logger = logging.getLogger("darkstar.ev_surplus")

_TRUEISH = {"on", "home", "true", "charging", "connected", "plugged", "1"}


@dataclass
class EVSurplusChargerCfg:
    """One charger's control wiring for the surplus controller."""

    id: str
    switch_entity: str | None = None  # on/off
    current_entity: str | None = None  # number.* set_value path (e.g. Tesla)
    easee_device_id: str | None = None  # easee.set_charger_dynamic_limit device (e.g. Easee)
    power_entity: str | None = None
    plug_entity: str | None = None
    home_entity: str | None = None  # device_tracker; absent => assume home
    home_states: tuple[str, ...] = ("home",)
    override_entity: str | None = None  # input_select auto/force_on/force_off
    priority: int = 0
    min_current_a: float = 6.0
    max_current_a: float = 16.0
    phases: int = 3
    voltage_v: float = 230.0
    # --- Departure / target-SoC awareness (all optional) ---
    soc_entity: str | None = None  # current SoC (input_number.fmb_soc / sensor.*_battery_level)
    target_soc_entity: str | None = None  # user-settable target % (input_number); None => no cap
    # Config-constant comfort cap, used when no target_soc_entity is wired (owner decision:
    # the FMB's 150 km cap is a plain config value, not another helper). An entity, when
    # configured AND readable, wins; this is the fallback. None + no entity => no cap.
    target_soc: float | None = None
    # Comfort-demotion threshold: at/above this SoC the charger yields surplus to every
    # non-demoted charger but keeps charging on what remains, up to the cap (owner:
    # "FMB till 150 km, sedan Teslan, sedan FMB mer"). An ACTIVE manual priority
    # selection disables demotion — an explicit order is taken literally.
    comfort_soc: float | None = None
    departure_entity: str | None = None  # input_datetime (date+time) -> 'timestamp' attr (epoch)
    # The guarantee band's upper SoC (what the deadline floor charges toward). Plain config
    # value per owner decision — the current SoC comes from soc_entity, the comfort cap from
    # target_soc_entity; this is the third, distinct number. None => floor uses the cap (legacy).
    floor_soc: float | None = None
    # Recurring deadline (e.g. the commuter car's weekday 07:30): days as lowercase
    # three-letter names, local wall-clock time "HH:MM" in `timezone`. Effective deadline =
    # earliest(future departure_entity timestamp, next recurring occurrence) — the
    # input_datetime stays as a one-off override (an extra trip), the recurrence never rots.
    recurring_deadline_days: tuple[str, ...] = ()
    recurring_deadline_time: str | None = None
    capacity_kwh: float = 0.0  # usable battery capacity (sizes the deadline floor); 0 => no floor
    charge_efficiency: float = 0.9  # plug->battery efficiency for the deadline floor
    # When vacation is active (see EVSurplusRuntimeConfig.vacation_entity) the target switches to
    # this. e.g. FMB -> 15 (cap at 15%, solar-only since it has no departure deadline). None =>
    # vacation does not change this charger's target (e.g. the Tesla leaving FOR the trip).
    vacation_target_soc: float | None = None
    # Per-charger write pacing (None => the global guard). Lets an API-expensive
    # charger (Tesla: every write wakes the car) be paced harder than a cheap one
    # (Easee dynamic limit: RAM-safe, free).
    write_guard: WriteGuardConfig | None = None
    # Restart dwell after a stop (s): once stopped, the charger may not restart for
    # this long. Deadline floors are exempt. 0 = no dwell (legacy).
    min_off_s: float = 0.0
    # Per-charger shadow: log decisions + advance the write-guard state, but never
    # call HA services. Rollout gate for a re-introduced charger — the GLOBAL
    # executor shadow flag would shadow battery/water too.
    shadow: bool = False

    @property
    def controllable(self) -> bool:
        return bool(self.current_entity or self.easee_device_id)


@dataclass
class EVSurplusRuntimeConfig:
    """Full executor.ev_surplus config: sources + policy + chargers."""

    enabled: bool = False
    pv_power_entity: str | None = None
    grid_power_entity: str | None = None  # signed, + import
    battery_power_entity: str | None = None  # signed, + charge
    battery_soc_entity: str | None = None
    price_entity: str | None = None
    remaining_solar_entity: str | None = None  # optional; absent => battery-assist tier inert
    vacation_entity: str | None = None  # input_boolean.vacation_mode; flips per-charger vacation targets
    # Manual fleet priority (owner-adjustable): an input_select whose selected option maps,
    # via priority_orders, to an explicit charger ordering for the SURPLUS class only —
    # floors (deadline/plan) always outrank, so the selector distributes comfort, not need.
    # Missing/unknown option (or unreadable entity) => configured per-charger priorities.
    priority_entity: str | None = None  # e.g. input_select.darkstar_ev_priority
    priority_orders: dict[str, list[str]] = field(default_factory=dict)
    # Local timezone for recurring deadlines (wall-clock "07:30" must survive DST — the
    # naive-datetime trap class; never compare naive local against epoch).
    timezone: str = "Europe/Stockholm"
    policy: EVSurplusConfig = field(default_factory=EVSurplusConfig)
    guard: WriteGuardConfig = field(default_factory=WriteGuardConfig)
    chargers: list[EVSurplusChargerCfg] = field(default_factory=lambda: [])


def _f(v: Any) -> float | None:
    if v is None or v in ("unknown", "unavailable", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _hhmm(v: Any) -> str | None:
    """Coerce a config time to "HH:MM"; YAML 1.1 parses unquoted 7:30 as int 450."""
    if isinstance(v, int):
        return f"{v // 60:02d}:{v % 60:02d}" if 0 <= v <= 1439 else None
    return str(v or "") or None


def next_recurring_deadline_ts(
    days: tuple[str, ...], time_str: str | None, now_ts: float, tz_name: str
) -> float | None:
    """Epoch of the next occurrence of a recurring local wall-clock deadline, else None.

    ``days`` are lowercase three-letter names; ``time_str`` is local "HH:MM". All wall-clock
    math runs in the configured IANA timezone and only the final result is converted to epoch
    — the container runs LOCAL time and naive/epoch mixups are this repo's recurring bug
    class (see the tz-trap incidents), so no naive datetime ever leaves this function.
    Invalid day names are ignored; no valid day or time => None (feature off).
    """
    if not days or not time_str:
        return None
    # YAML 1.1 sexagesimal defense (mirrors the planner's calculate_ev_deadline): an
    # unquoted `recurring_deadline_time: 7:30` parses as int 450 (minutes since midnight).
    if isinstance(time_str, int):
        if 0 <= time_str <= 1439:
            time_str = f"{time_str // 60:02d}:{time_str % 60:02d}"
        else:
            logger.warning("EV surplus: invalid recurring_deadline_time integer %r", time_str)
            return None
    try:
        hour, minute = (int(p) for p in time_str.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, AttributeError):
        logger.warning("EV surplus: invalid recurring_deadline_time %r", time_str)
        return None
    wanted = {d for d in days if d in _DAY_NAMES}
    if not wanted:
        logger.warning("EV surplus: no valid recurring_deadline_days in %r", days)
        return None
    try:
        tz = ZoneInfo(tz_name)
    except (KeyError, ZoneInfoNotFoundError):
        logger.warning("EV surplus: unknown timezone %r, using Europe/Stockholm", tz_name)
        tz = ZoneInfo("Europe/Stockholm")
    now_local = datetime.fromtimestamp(now_ts, tz)
    for day_offset in range(8):  # today + 7 covers every weekday pattern
        cand = (now_local + timedelta(days=day_offset)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if _DAY_NAMES[cand.weekday()] in wanted and cand.timestamp() > now_ts:
            return cand.timestamp()
    return None  # unreachable with a valid day set; defensive


def parse_ev_surplus_config(
    executor_data: dict[str, Any], timezone: str | None = None
) -> EVSurplusRuntimeConfig | None:
    """Build the runtime config from ``executor.ev_surplus``; None if absent.

    ``timezone`` is the RESOLVED site timezone (root-level config key) — the caller must
    pass it; ``executor.timezone`` does not exist in the schema, so reading it here would
    silently pin recurring deadlines to the Stockholm fallback on every non-SE site.
    """
    raw_any = executor_data.get("ev_surplus")
    if not isinstance(raw_any, dict):
        return None
    raw = cast("dict[str, Any]", raw_any)
    pol = cast("dict[str, Any]", raw.get("policy", {}) or {})
    ba = cast("dict[str, Any]", pol.get("battery_assist", {}) or {})
    guard_raw = cast("dict[str, Any]", raw.get("write_guard", {}) or {})
    policy = EVSurplusConfig(
        enabled=bool(raw.get("enabled", False)),
        cheap_grid_price_sek=_f(pol.get("cheap_grid_price_sek")),
        cheap_grid_allowance_w=float(pol.get("cheap_grid_allowance_w", 3680.0)),
        battery_assist_enabled=bool(ba.get("enabled", False)),
        battery_assist_max_price_sek=float(ba.get("max_price_sek", 0.0)),
        battery_assist_min_remaining_solar_kwh=float(ba.get("min_remaining_solar_kwh", 8.0)),
        battery_assist_floor_soc=float(ba.get("floor_soc", 40.0)),
        battery_assist_allowance_w=float(ba.get("allowance_w", 6000.0)),
        battery_assist_soc_hysteresis=float(ba.get("soc_hysteresis", 3.0)),
        gain=float(pol.get("gain", 0.5)),
        deadband_w=float(pol.get("deadband_w", 250.0)),
        # NOTE: must match the pure-layer default (1.0) — a diverging parse default
        # here silently reintroduces the coarse 2 A grid on sites without the key.
        current_step_a=float(pol.get("current_step_a", 1.0)),
        start_hysteresis=float(pol.get("start_hysteresis", 0.15)),
        quantum_deadband_k=float(pol.get("quantum_deadband_k", 1.5)),
        schmitt_fraction=float(pol.get("schmitt_fraction", 0.7)),
    )

    def _parse_guard(raw_g: dict[str, Any], fallback: WriteGuardConfig) -> WriteGuardConfig:
        def _opt(key: str) -> float | None:
            v = _f(raw_g.get(key))
            return v if v is not None else getattr(fallback, key)

        return WriteGuardConfig(
            min_step_a=float(raw_g.get("min_step_a", fallback.min_step_a)),
            min_interval_s=float(raw_g.get("min_interval_s", fallback.min_interval_s)),
            min_interval_up_s=_opt("min_interval_up_s"),
            min_interval_down_s=_opt("min_interval_down_s"),
        )

    guard = _parse_guard(guard_raw, WriteGuardConfig())
    chargers: list[EVSurplusChargerCfg] = []
    for c in cast("list[dict[str, Any]]", raw.get("chargers", []) or []):
        if not c.get("id"):
            continue
        chargers.append(
            EVSurplusChargerCfg(
                id=str(c["id"]),
                switch_entity=c.get("switch_entity") or None,
                current_entity=c.get("current_entity") or None,
                easee_device_id=c.get("easee_device_id") or None,
                power_entity=c.get("power_entity") or None,
                plug_entity=c.get("plug_entity") or None,
                home_entity=c.get("home_entity") or None,
                home_states=tuple(c.get("home_states", ["home"])),
                override_entity=c.get("override_entity") or None,
                priority=int(c.get("priority", 0)),
                min_current_a=float(c.get("min_current_a", 6.0)),
                max_current_a=float(c.get("max_current_a", 16.0)),
                phases=int(c.get("phases", 3)),
                voltage_v=float(c.get("voltage_v", 230.0)),
                soc_entity=c.get("soc_entity") or None,
                target_soc_entity=c.get("target_soc_entity") or None,
                target_soc=_f(c.get("target_soc")),
                comfort_soc=_f(c.get("comfort_soc")),
                departure_entity=c.get("departure_entity") or None,
                floor_soc=_f(c.get("floor_soc")),
                recurring_deadline_days=tuple(
                    str(d).lower()[:3]
                    for d in cast("list[Any]", c.get("recurring_deadline_days", []) or [])
                ),
                recurring_deadline_time=_hhmm(c.get("recurring_deadline_time")),
                capacity_kwh=float(c.get("capacity_kwh", 0.0)),
                charge_efficiency=float(c.get("charge_efficiency", 0.9)),
                vacation_target_soc=_f(c.get("vacation_target_soc")),
                write_guard=(
                    _parse_guard(cast("dict[str, Any]", c["write_guard"]), guard)
                    if isinstance(c.get("write_guard"), dict)
                    else None
                ),
                min_off_s=float(c.get("min_off_s", 0.0)),
                shadow=bool(c.get("shadow", False)),
            )
        )
        # Easee 6 A hard floor (owner-confirmed: the FMB STOPS CHARGING below 6 A).
        # Structural backstop against a config typo — parse clamps + warns; the
        # runtime additionally hard-refuses any 1-5 A Easee write (belt & braces).
        last = chargers[-1]
        if last.easee_device_id and last.min_current_a < 6.0:
            logger.warning(
                "Charger %s: min_current_a %.1f below the Easee 6 A floor — clamping to 6",
                last.id,
                last.min_current_a,
            )
            last.min_current_a = 6.0
    return EVSurplusRuntimeConfig(
        enabled=bool(raw.get("enabled", False)),
        pv_power_entity=raw.get("pv_power_entity") or None,
        grid_power_entity=raw.get("grid_power_entity") or None,
        battery_power_entity=raw.get("battery_power_entity") or None,
        battery_soc_entity=raw.get("battery_soc_entity") or None,
        price_entity=raw.get("price_entity") or None,
        remaining_solar_entity=raw.get("remaining_solar_entity") or None,
        vacation_entity=raw.get("vacation_entity") or None,
        priority_entity=raw.get("priority_entity") or None,
        priority_orders={
            str(k).lower(): [str(cid) for cid in cast("list[Any]", v)]
            for k, v in cast(
                "dict[str, Any]", raw.get("priority_orders", {}) or {}
            ).items()
            if isinstance(v, list)
        },
        timezone=str(timezone or executor_data.get("timezone") or "Europe/Stockholm"),
        policy=policy,
        guard=guard,
        chargers=chargers,
    )


class EVSurplusController:
    """Stateful runtime: reads HA, computes, actuates through the write-guard."""

    def __init__(self, cfg: EVSurplusRuntimeConfig):
        self.cfg = cfg
        # Per-charger write-guard memory.
        self._last_a: dict[str, float] = {}
        self._last_ts: dict[str, float] = {}
        self._last_switch: dict[str, bool] = {}
        self._last_stop_ts: dict[str, float] = {}  # min-OFF dwell anchors
        self._battery_tier_prev: bool = False  # battery-assist SoC hysteresis memory

    async def _read_f(self, ha: Any, entity: str | None, default: float | None = None) -> float | None:
        if not entity:
            return default
        v = _f(await ha.get_state_value(entity))
        return v if v is not None else default

    async def _read_attr_f(
        self, ha: Any, entity: str | None, attr: str, default: float | None = None
    ) -> float | None:
        """Read a numeric attribute (e.g. an input_datetime's epoch 'timestamp')."""
        if not entity:
            return default
        state = cast("dict[str, Any] | None", await ha.get_state(entity))
        if not state:
            return default
        attrs = cast("dict[str, Any]", state.get("attributes") or {})
        v = _f(attrs.get(attr))
        return v if v is not None else default

    async def _read_on(
        self,
        ha: Any,
        entity: str | None,
        states: tuple[str, ...],
        default: bool,
        unreadable_default: bool | None = None,
    ) -> bool:
        """``default`` applies when NO entity is configured (intentional, e.g. a
        sensorless "assume plugged" setup). ``unreadable_default`` applies when a
        CONFIGURED entity cannot be read at all (integration deleted + HA restart
        returns None) — the phantom-car fix passes False there so a vanished
        charger never looks plugged-in/home. None => same as ``default``."""
        if not entity:
            return default
        v = await ha.get_state_value(entity)
        if v is None:
            resolved = default if unreadable_default is None else unreadable_default
            if resolved != default:
                logger.warning("EV surplus: %s configured but unreadable -> %s", entity, resolved)
            return resolved
        return str(v).lower() in {s.lower() for s in states}

    async def _read_override(self, ha: Any, entity: str | None) -> str:
        if not entity:
            return "auto"
        v = await ha.get_state_value(entity)
        val = str(v).lower() if v else "auto"
        return val if val in ("auto", "force_on", "force_off") else "auto"

    async def _read_priority_order(self, ha: Any) -> list[str] | None:
        """Resolve the manual priority selector to a charger ordering, else None.

        None (no entity, unreadable, or an option without a priority_orders mapping —
        including "auto") => configured per-charger priorities. Unknown options degrade
        safely the same way, so a renamed/removed helper can never strand the fleet.
        """
        if not self.cfg.priority_entity or not self.cfg.priority_orders:
            return None
        v = await ha.get_state_value(self.cfg.priority_entity)
        if not v:
            return None
        return self.cfg.priority_orders.get(str(v).lower())

    async def _read_charger(
        self, ha: Any, c: EVSurplusChargerCfg, now_ts: float, vacation: bool
    ) -> ChargerState:
        """Read one charger's live state (its entity reads run concurrently)."""
        # gather over mixed return types collapses to a union list; narrow each back explicitly.
        res = await asyncio.gather(
            self._read_f(ha, c.power_entity, 0.0),
            self._read_on(
                ha, c.plug_entity, ("on", "true", "plugged", "connected"), True,
                unreadable_default=False,
            ),
            self._read_on(ha, c.home_entity, c.home_states, True, unreadable_default=False),
            self._read_override(ha, c.override_entity),
            self._read_f(ha, c.soc_entity, None),
            self._read_f(ha, c.target_soc_entity, None),
            self._read_attr_f(ha, c.departure_entity, "timestamp", None),
        )
        power = cast("float | None", res[0])
        plugged = bool(res[1])
        at_home = bool(res[2])
        override = str(res[3])
        soc = cast("float | None", res[4])
        target = cast("float | None", res[5])
        if target is None:
            target = c.target_soc  # config-constant cap when no entity is wired/readable
        dep_ts = cast("float | None", res[6])
        # Effective deadline = earliest of the one-off departure entity (if in the future)
        # and the recurring weekday deadline. The entity is a per-trip override that can rot
        # (a past date is simply inert); the recurrence never does.
        candidates: list[float] = []
        if dep_ts is not None and dep_ts > now_ts:
            candidates.append(dep_ts)
        rec_ts = next_recurring_deadline_ts(
            c.recurring_deadline_days, c.recurring_deadline_time, now_ts, self.cfg.timezone
        )
        if rec_ts is not None:
            candidates.append(rec_ts)
        deadline_hours: float | None = None
        if candidates:
            deadline_hours = (min(candidates) - now_ts) / 3600.0
        # Vacation = solar-only for EVERY charger: clear any deadline unconditionally so no
        # car is grid-forced while the household is away (a weekday recurrence would otherwise
        # keep force-charging the commuter car all vacation). The target override additionally
        # applies only where a vacation_target_soc is configured (e.g. FMB -> 15%);
        # ``soc`` / ``target`` of None disable the cap and deadline floor entirely.
        if vacation:
            deadline_hours = None
            if c.vacation_target_soc is not None:
                target = c.vacation_target_soc
        # Commanded state: _last_a is authoritative for controllable chargers (every
        # actuated stop zeroes it — see _actuate), the switch memory for binary ones.
        # None (never actuated this process) => the pure layer infers from power.
        commanded_on: bool | None = None
        if c.id in self._last_a:
            commanded_on = self._last_a[c.id] > 0.0
        elif c.id in self._last_switch:
            commanded_on = self._last_switch[c.id]
        start_inhibited = (
            now_ts - self._last_stop_ts.get(c.id, float("-inf"))
        ) < c.min_off_s
        return ChargerState(
            id=c.id, plugged=plugged, at_home=at_home, enabled=True,
            current_power_w=power or 0.0, max_current_a=c.max_current_a,
            min_current_a=c.min_current_a, phases=c.phases, voltage_v=c.voltage_v,
            controllable=c.controllable, priority=c.priority, override=override,
            soc_percent=soc, target_soc_percent=target, floor_soc_percent=c.floor_soc,
            comfort_soc_percent=c.comfort_soc,
            capacity_kwh=c.capacity_kwh,
            deadline_hours=deadline_hours, charge_efficiency=c.charge_efficiency,
            commanded_on=commanded_on, start_inhibited=start_inhibited,
        )

    async def run(self, ha: Any, now_ts: float, shadow: bool = False) -> dict[str, Any]:
        """One control cycle. Returns a summary (for logging / UI)."""
        cfg = self.cfg
        if not cfg.enabled or not cfg.chargers:
            return {"enabled": False}

        # Read all sources AND all chargers concurrently. Serial awaits here meant ~14 HA
        # REST round-trips per tick (~5 s), which loaded the HA API and starved the in-process
        # web server (slow dashboard). asyncio.gather collapses that to ~one round-trip latency.
        src = await asyncio.gather(
            self._read_f(ha, cfg.pv_power_entity, 0.0),
            self._read_f(ha, cfg.grid_power_entity, None),
            self._read_f(ha, cfg.battery_power_entity, None),
            self._read_f(ha, cfg.battery_soc_entity, 0.0),
            self._read_f(ha, cfg.price_entity, 999.0),
            self._read_f(ha, cfg.remaining_solar_entity, 0.0),
            self._read_on(ha, cfg.vacation_entity, ("on", "true"), False),
            self._read_priority_order(ha),
        )
        # The grid + battery meters ARE the control signal — computing with a fake 0
        # for either would command the cars blind (e.g. full-throttle into an outage).
        # Hold last commands and skip the tick instead.
        if src[1] is None or src[2] is None:
            logger.warning(
                "EV surplus: core sensors unreadable (grid=%s battery=%s) — skipping tick",
                src[1],
                src[2],
            )
            return {"enabled": True, "skipped": "core sensors unreadable"}
        pv_w = src[0] or 0.0
        grid_w = src[1]
        battery_w = src[2]
        # Unknown home-battery SoC must DISABLE battery assist (0), never enable it (100).
        soc = src[3] if src[3] is not None else 0.0
        price = src[4] if src[4] is not None else 999.0
        remaining_solar = src[5] or 0.0
        vacation = bool(src[6])
        priority_order = cast("list[str] | None", src[7])
        states: list[ChargerState] = list(
            await asyncio.gather(
                *(self._read_charger(ha, c, now_ts, vacation) for c in cfg.chargers)
            )
        )
        # Manual fleet priority: remap ChargerState.priority from the selected ordering.
        # Floors are untouched by design — in the pure sort, floor class + deadline urgency
        # rank BEFORE priority, so the selector only redistributes the surplus class.
        # Unlisted chargers keep their configured priority pushed behind every listed one.
        # An explicit order also DISABLES comfort-demotion: the owner's literal choice
        # beats the soft "yield above comfort" rule (auto keeps the smart ordering).
        if priority_order:
            rank = {cid: i for i, cid in enumerate(priority_order)}
            n = len(priority_order)
            for s in states:
                # max(0, ...) keeps the "unlisted goes behind every listed" invariant even
                # for a (legal) negative configured priority.
                s.priority = rank.get(s.id, n + max(0, s.priority))
                s.comfort_soc_percent = None

        inputs = EVSurplusInputs(
            pv_w=pv_w, grid_w=grid_w, battery_w=battery_w, battery_soc_percent=soc,
            import_price_sek=price, remaining_solar_kwh=remaining_solar,
            battery_tier_active_prev=self._battery_tier_prev, chargers=states,
        )
        commands = compute_ev_surplus(inputs, cfg.policy)
        # Track the tier through the SAME helper the pure layer uses (hysteresis memory).
        self._battery_tier_prev = battery_tier_active(inputs, cfg.policy)
        cfg_by_id = {c.id: c for c in cfg.chargers}

        applied: list[dict[str, Any]] = []
        for cmd in commands:
            ccfg = cfg_by_id.get(cmd.id)
            if ccfg is None:
                continue
            try:
                await self._actuate(ha, ccfg, cmd, now_ts, shadow)
            except Exception:
                # One charger's dead entity must not starve the others' actuation.
                logger.exception("EV surplus: actuation failed for %s — continuing", cmd.id)
                continue
            applied.append({"id": cmd.id, "on": cmd.switch_on, "a": cmd.set_current_a, "why": cmd.reason})

        logger.info("EV surplus: grid=%.0fW batt=%.0fW soc=%.0f%% price=%.2f vac=%s -> %s",
                    grid_w, battery_w, soc, price, vacation,
                    [(a["id"], a["on"], a["a"]) for a in applied])
        return {"enabled": True, "applied": applied}

    async def _actuate(self, ha: Any, ccfg: EVSurplusChargerCfg, cmd: Any, now_ts: float, shadow: bool) -> None:
        # Per-charger shadow (rollout gate): suppress service calls but keep the full
        # decision path + guard state, so shadow logs show realistic write rates.
        shadow = shadow or ccfg.shadow
        # Per-charger write pacing wins over the global guard.
        guard = ccfg.write_guard or self.cfg.guard

        # Switch: only toggle on change.
        if ccfg.switch_entity is not None and self._last_switch.get(ccfg.id) != cmd.switch_on:
            if not shadow:
                svc = "turn_on" if cmd.switch_on else "turn_off"
                await ha.call_service("switch", svc, ccfg.switch_entity)
            if not cmd.switch_on:
                # A stop must ZERO the commanded-amps memory: commanded_on derives
                # from _last_a, and without this the start kick / min-OFF dwell /
                # start hysteresis all read the charger as still ON after its first
                # switch-path stop — permanently, since nothing else resets it.
                if self._last_switch.get(ccfg.id) is True:
                    self._last_stop_ts[ccfg.id] = now_ts
                self._last_a[ccfg.id] = 0.0
            self._last_switch[ccfg.id] = cmd.switch_on

        # Pause a switchless Easee: with no on/off switch, dynamic limit 0 IS the stop. Without
        # this an "off" command (SoC cap / force_off) on a switchless Easee would leave it
        # charging at its last dynamic limit. The write-guard always allows a stop immediately.
        if not cmd.switch_on and ccfg.switch_entity is None and ccfg.easee_device_id:
            prev_a = self._last_a.get(ccfg.id)
            if should_write_current(
                prev_a, self._last_ts.get(ccfg.id), 0.0, now_ts, guard
            ):
                if not shadow:
                    await ha.call_service(
                        "easee", "set_charger_dynamic_limit", None,
                        {"device_id": ccfg.easee_device_id, "current": 0, "time_to_live": 0},
                    )
                if prev_a is not None and prev_a > 0.0:
                    self._last_stop_ts[ccfg.id] = now_ts
                self._last_a[ccfg.id] = 0.0
                self._last_ts[ccfg.id] = now_ts
            return

        # Current: only when on, controllable, and the write-guard allows it.
        if not (cmd.switch_on and ccfg.controllable and cmd.set_current_a is not None):
            return
        new_a = float(cmd.set_current_a)
        last_a = self._last_a.get(ccfg.id)
        # Schmitt quantizer: a +/-1-step move is only real once the RAW (unsnapped)
        # target has cleared schmitt_fraction of a step away from the written value.
        # Kills midpoint dither and the config-vs-real-voltage churn that a 1 A grid
        # would otherwise unmask. Stops and starts are exempt (handled above/guard).
        raw = getattr(cmd, "raw_amps", None)
        if (
            last_a is not None
            and last_a > 0.0
            and new_a > 0.0
            and new_a != last_a
            and raw is not None
            and abs(float(raw) - last_a)
            < self.cfg.policy.schmitt_fraction * max(0.0, self.cfg.policy.current_step_a)
        ):
            return
        if not should_write_current(last_a, self._last_ts.get(ccfg.id), new_a, now_ts, guard):
            return
        # Easee hard floor (owner-confirmed: the FMB stops charging below 6 A) —
        # final backstop after the pure-layer clamp and the parse-time config clamp.
        if ccfg.easee_device_id and 0 < round(new_a) < 6:
            logger.error(
                "EV surplus: refusing %.1f A write to Easee %s (below the 6 A floor)",
                new_a,
                ccfg.id,
            )
            return
        if not shadow:
            if ccfg.current_entity:
                await ha.call_service("number", "set_value", ccfg.current_entity, {"value": new_a})
            elif ccfg.easee_device_id:
                # FLASH-SAFE dynamic limit only (volatile/RAM). Never the max/circuit limits.
                await ha.call_service(
                    "easee", "set_charger_dynamic_limit", None,
                    {"device_id": ccfg.easee_device_id, "current": round(new_a), "time_to_live": 0},
                )
        self._last_a[ccfg.id] = new_a
        self._last_ts[ccfg.id] = now_ts
