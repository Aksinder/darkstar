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
    ChargerCommand,
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    WriteGuardConfig,
    battery_fill_slack_kwh,
    battery_reserve_active,
    battery_tier_active,
    compute_ev_surplus,
    fuse_battery_charge_cap_w,
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
    # Optional button that wakes a sleeping car (e.g. button.white_betty_wake).
    # A sleeping Tesla answers switch.turn_on with HTTP 500, so a plugged-in car
    # sitting on surplus can never start. On an actuation failure the runtime
    # presses this once (cooldown-limited) so the backoff retry lands on a woken
    # car. Pressing costs one API call and no vehicle wake-lock beyond the usual.
    wake_entity: str | None = None
    # Believe what we COMMANDED while the power sensor has not yet caught up with it.
    # A Tesla reports charger power on a multi-minute poll, so right after a start the
    # servo reads 0 W from a car that is already pulling 5 kW — and since the control
    # law is "target = measured draw + headroom", a car that has just eaten the export
    # looks exactly like a cloud. The servo then switches off the load it just created.
    # Only ever substitutes while the reading demonstrably predates our own write.
    trust_commanded_draw: bool = True
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
    # Which grid phases this car's DRAW lands on (lowercase names matching the fuse
    # guard's phase_entities keys). EMPTY = unknown => conservative (all phases).
    # Maps the CAR, not the box: the 3-phase-wired Easee carries the 1-phase FMB.
    phase_map: tuple[str, ...] = ()
    # Restart dwell after a stop (s): once stopped, the charger may not restart for
    # this long. Deadline floors are exempt. 0 = no dwell (legacy).
    min_off_s: float = 0.0
    # S3 bridge: honour kepler's per-slot charging plan as a grid-backed FLOOR for
    # this charger (price-optimal night placement instead of the evenly-smeared
    # deadline backstop). Default OFF for staged rollout.
    plan_floor: bool = False
    # Plan floors are honoured only BELOW this SoC — single-sourced at parse time
    # from the PLANNER's first penalty band (its guarantee band), NOT floor_soc:
    # the FMB's guarantee band tops at 86 while its emergency floor_soc is 40, and
    # gating on 40 would kill night charging for the whole 40-86 band (S3 note).
    # Above the gate, planned slots are surplus HINTS, never grid-backed (F11:
    # a cloud gap in a topup slot must not buy 0.35-valued energy at import price).
    plan_gate_soc: float | None = None
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
    # --- 25 A/phase main-fuse guard ---
    # phase_entities maps phase name -> grid meter current sensor (|A|, the actual
    # fuse current). Stale/unreadable readings (age > stale_after_s, or unavailable)
    # trigger the fail-safe: all cars OFF — a blind guard must not hold 16 A commands
    # while VVB/battery/house stack underneath (min-clamp math: 36.8 A = guaranteed
    # fuse blow). The meter and these sensors share one Modbus integration and die
    # together, so the fail-safe also arms from the core-sensor skip path.
    fuse_guard_enabled: bool = False
    fuse_limit_a: float = 25.0
    fuse_margin_a: float = 2.0
    fuse_phase_entities: dict[str, str] = field(default_factory=dict)
    fuse_stale_after_s: float = 180.0
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


def _valid_phase_map(
    raw_map: list, fg: dict, charger_id: str
) -> tuple[str, ...]:
    """Lowercase the map and drop names not in fuse_guard.phase_entities.

    A typo ('l1' vs 'a') would otherwise make every reading lookup miss and
    silently turn the mapped charger into a no-increase hold — dropping to ()
    instead degrades to the CONSERVATIVE all-phases budget, loudly.
    """
    names = tuple(str(p).lower() for p in raw_map)
    known = {str(k).lower() for k in (fg.get("phase_entities") or {})}
    if not names or not known:
        return names
    kept = tuple(n for n in names if n in known)
    if kept != names:
        logger.warning(
            "Charger %s: phase_map %r has names outside phase_entities %r — "
            "using conservative all-phases budgeting",
            charger_id, names, sorted(known),
        )
        return ()
    return kept


def _plan_gate_soc(planner_entry: dict | None, charger_id: str) -> float | None:
    """The plan-floor soc-gate = the PLANNER's first penalty band's max_soc.

    Single source (F6): the gate IS the guarantee band, so an owner edit to the
    band moves the gate with it. None (no planner entry / no bands) => a
    plan_floor charger gets NO grid-backed plan floors (gate everything).
    """
    if not planner_entry:
        return None
    bands = planner_entry.get("penalty_levels") or []
    if not bands:
        return None
    v = _f(bands[0].get("max_soc"))
    if v is None:
        logger.warning("Charger %s: unparsable first penalty band — plan floors gated off", charger_id)
    return v


def parse_ev_surplus_config(
    executor_data: dict[str, Any],
    timezone: str | None = None,
    planner_ev_chargers: list[dict[str, Any]] | None = None,
) -> EVSurplusRuntimeConfig | None:
    """Build the runtime config from ``executor.ev_surplus``; None if absent.

    ``timezone`` is the RESOLVED site timezone (root-level config key) — the caller must
    pass it; ``executor.timezone`` does not exist in the schema, so reading it here would
    silently pin recurring deadlines to the Stockholm fallback on every non-SE site.
    ``planner_ev_chargers`` is the ROOT ev_chargers list — the plan-floor soc-gate is
    single-sourced from each charger's FIRST penalty band so servo gate and planner
    guarantee band can never drift apart (F6).
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
        battery_yield_soc=float(pol.get("battery_yield_soc", 0.0)),
        battery_capacity_kwh=float(pol.get("battery_capacity_kwh", 0.0)),
        battery_charge_efficiency=float(pol.get("battery_charge_efficiency", 0.95)),
        battery_fill_margin_kwh=float(pol.get("battery_fill_margin_kwh", 3.0)),
        battery_fill_margin_hysteresis_kwh=float(
            pol.get("battery_fill_margin_hysteresis_kwh", 2.0)
        ),
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
    fg = cast("dict[str, Any]", raw.get("fuse_guard", {}) or {})
    fuse_enabled = bool(fg.get("enabled", False))
    if fuse_enabled and (
        not bool(raw.get("enabled", False))
        or not raw.get("chargers")
        or not fg.get("phase_entities")
    ):
        # A guard that can never run its read path would permanently zero the
        # engine's battery cap (never-read => fail-safe 0) or fail-safe every
        # tick on an empty entity map — reject the combination loudly instead.
        logger.warning(
            "fuse_guard disabled: requires ev_surplus.enabled, chargers and "
            "phase_entities"
        )
        fuse_enabled = False
    # The pure clamp activates via policy.fuse_budget_a (limit minus margin).
    if fuse_enabled:
        policy.fuse_budget_a = float(fg.get("limit_a", 25.0)) - float(fg.get("margin_a", 2.0))
    planner_by_id: dict[str, dict[str, Any]] = {
        str(e.get("id", "")): e for e in (planner_ev_chargers or [])
    }
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
                wake_entity=c.get("wake_entity") or None,
                trust_commanded_draw=bool(c.get("trust_commanded_draw", True)),
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
                phase_map=_valid_phase_map(
                    cast("list[Any]", c.get("phase_map", []) or []),
                    fg if fuse_enabled else {},
                    str(c.get("id", "?")),
                ),
                min_off_s=float(c.get("min_off_s", 0.0)),
                plan_floor=bool(c.get("plan_floor", False)),
                plan_gate_soc=_plan_gate_soc(
                    planner_by_id.get(str(c.get("id", ""))), str(c.get("id", "?"))
                ),
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
    for _cc in chargers:
        if _cc.plan_floor:
            _pe = planner_by_id.get(_cc.id) or {}
            if str(_pe.get("switch_entity") or "").strip():
                # Two live actuation paths would fight in the same tick — hard-
                # disable the bridge for this charger rather than just warning.
                logger.warning(
                    "Charger %s: plan_floor DISABLED — the planner entry has a "
                    "switch_entity and the legacy _control_ev_charger path runs "
                    "BEFORE the servo in the same tick (F4). Clear the planner "
                    "switch_entity to enable the bridge.",
                    _cc.id,
                )
                _cc.plan_floor = False
            elif _cc.plan_gate_soc is None:
                logger.warning(
                    "Charger %s: plan_floor is set but no planner guarantee band "
                    "was found — every plan slot will be gated off (silent no-op "
                    "otherwise). Add penalty_levels to the planner entry.",
                    _cc.id,
                )
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
        fuse_guard_enabled=fuse_enabled,
        fuse_limit_a=float(fg.get("limit_a", 25.0)),
        fuse_margin_a=float(fg.get("margin_a", 2.0)),
        fuse_phase_entities={
            str(k).lower(): str(v)
            for k, v in cast("dict[str, Any]", fg.get("phase_entities", {}) or {}).items()
            if v
        },
        fuse_stale_after_s=float(fg.get("stale_after_s", 180.0)),
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
        self._battery_reserve_prev: bool = False  # battery fill-deadline hysteresis memory
        # Fuse-guard state, also consumed by the engine's battery-charge cap.
        self.last_phase_currents_a: dict[str, float] = {}
        self.last_phase_ok_ts: float | None = None
        self.last_battery_w: float = 0.0
        self.last_grid_w: float = 0.0
        # Ampere increases the EV clamp granted THIS tick, per phase — the engine's
        # battery cap subtracts these so the two levers never double-spend the same
        # meter snapshot (review-caught).
        self.last_ev_alloc_a: dict[str, float] = {}
        self._core_skip_since: float | None = None
        # S3 continuity hold: (held_w, until_ts) per charger. Kepler's 15-min replans
        # can flip equivalent-cost slot checkerboards; without a hold each flip would
        # stop/start the car (Tesla wakes, Easee relay churn). A started plan floor is
        # held for >= PLAN_HOLD_S even if the next replan drops the slot — bounded
        # cost (<= 30 min at the held level), and the soc-gate still ends it early.
        self._plan_hold: dict[str, tuple[float, float]] = {}
        # Why each plan_floor charger did/didn't act this tick ('active' /
        # 'gated' / 'vacation' / 'idle') — read by the engine's EV-charge-failure
        # notifier so an intentionally soc-gated plan is never counted as failure.
        self.last_plan_note: dict[str, str] = {}
        # Actuation-failure backoff: (last_fail_ts, consecutive_failures) per charger.
        # A failed HA call (Tesla asleep => switch 500) must NOT retry every tick —
        # the failure doesn't update _last_switch, so without backoff the servo
        # re-sends up to 60 calls/h against an API with a documented rate-limit ban
        # history. Exponential: 120 s doubling to a 900 s ceiling; any SUCCESSFUL
        # actuation clears it.
        self._act_fail: dict[str, tuple[float, int]] = {}
        self._last_wake_ts: dict[str, float] = {}
        # When we last WROTE anything for a charger — a power reading older than
        # this cannot reflect the command (see trust_commanded_draw).
        self._last_cmd_ts: dict[str, float] = {}

    def fuse_battery_cap_w(self, now_ts: float) -> float | None:
        """Battery charge-setpoint cap for the engine, or None when the guard is off.

        Fresh phase readings => iterative headroom cap (see fuse_battery_charge_cap_w).
        Stale/never-read => 0.0: no grid charging on blind sensors — the battery is the
        guard's only non-EV shed lever, so it must fail SAFE (PV falls through to
        export, which costs nothing; recovery is automatic when readings return).
        """
        if not self.cfg.fuse_guard_enabled:
            return None
        budget = self.cfg.fuse_limit_a - self.cfg.fuse_margin_a
        if (
            self.last_phase_ok_ts is None
            or (now_ts - self.last_phase_ok_ts) > self.cfg.fuse_stale_after_s
        ):
            return 0.0
        return fuse_battery_charge_cap_w(
            self.last_phase_currents_a,
            self.last_battery_w,
            budget,
            grid_w=self.last_grid_w,
            ev_alloc_a=self.last_ev_alloc_a,
        )

    @property
    def fuse_budget_a(self) -> float | None:
        """Limit minus margin, or None when the guard is off. Shared with the water
        layer, which uses the same budget as its own shed trigger."""
        if not self.cfg.fuse_guard_enabled:
            return None
        return self.cfg.fuse_limit_a - self.cfg.fuse_margin_a

    async def read_phase_currents(
        self, ha: Any, now_ts: float
    ) -> dict[str, float] | None:
        """Public alias so the water layer can share ONE freshness policy with the
        EV clamp — two readers would eventually disagree about what 'stale' means."""
        return await self._read_phase_currents(ha, now_ts)

    async def _read_phase_currents(
        self, ha: Any, now_ts: float
    ) -> dict[str, float] | None:
        """Fresh per-phase |A| readings, or None if ANY phase is stale/unreadable.

        Freshness uses last_updated age (the sensors jitter at 10 s cadence, so a
        frozen last_updated IS staleness); unavailable/unknown/parse failure counts
        immediately. All-or-nothing: a partially blind guard budgets wrong phases.
        """
        out: dict[str, float] = {}
        for phase, entity in self.cfg.fuse_phase_entities.items():
            try:
                st = cast("dict[str, Any] | None", await ha.get_state(entity))
            except Exception:
                # A non-retryable read error (404 renamed entity — this site's
                # documented history — 401 rotated token, decode failure) is
                # BLINDNESS, not a reason to crash past the fail-safe.
                logger.exception("EV surplus: phase read failed for %s", entity)
                return None
            if not st:
                return None
            val = _f(st.get("state"))
            if val is None:
                return None
            lu = st.get("last_updated")
            if lu:
                try:
                    age = now_ts - datetime.fromisoformat(str(lu)).timestamp()
                except (ValueError, TypeError):
                    return None
                if age > self.cfg.fuse_stale_after_s:
                    return None
            out[phase] = abs(val)
        return out or None

    async def _failsafe_stop_all(
        self, ha: Any, now_ts: float, why: str, shadow: bool
    ) -> None:
        """Guard-blind => cars OFF. Stops need no sensors and bypass all pacing.

        ``shadow`` is the GLOBAL observe-only flag threaded from run() — an
        observe-only executor must not actuate live devices even to fail safe
        (review-caught: the hardcoded False here sent a real Easee stop in
        shadow mode). Per-charger shadow composes inside _actuate as before.
        """
        logger.warning("EV surplus FUSE FAIL-SAFE: %s — stopping all chargers", why)
        for ccfg in self.cfg.chargers:
            cmd = ChargerCommand(
                ccfg.id, switch_on=False,
                set_current_a=0.0 if ccfg.controllable else None,
                target_power_w=0.0, reason=f"off: fuse fail-safe ({why})",
                fuse_limited=True,
            )
            try:
                await self._actuate(ha, ccfg, cmd, now_ts, shadow)
            except Exception:
                logger.exception(
                    "EV surplus: fail-safe stop failed for %s — continuing", ccfg.id
                )

    async def _read_f(self, ha: Any, entity: str | None, default: float | None = None) -> float | None:
        if not entity:
            return default
        v = _f(await ha.get_state_value(entity))
        return v if v is not None else default

    async def _read_power_with_ts(
        self, ha: Any, entity: str | None
    ) -> tuple[float | None, float | None]:
        """Read a power sensor plus the epoch of its last update (None if unknown)."""
        if not entity:
            return None, None
        state = cast("dict[str, Any] | None", await ha.get_state(entity))
        if not state:
            return None, None
        value = _f(state.get("state"))
        raw_ts = state.get("last_updated") or state.get("last_changed")
        ts: float | None = None
        if isinstance(raw_ts, str):
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = None
        return value, ts

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
            self._read_power_with_ts(ha, c.power_entity),
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
        power, power_ts = cast("tuple[float | None, float | None]", res[0])
        if power is None:
            power = 0.0
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
        # Close the loop on our own actuation. The control law is
        # "target = measured draw + headroom", so a car that has just eaten the export
        # while its power sensor still reads 0 is indistinguishable from a cloud — and
        # the servo switches off the very load it created (observed live 2026-08-15:
        # commanded 8 A at 12:07:04, grid went -5.4 kW -> -0.15 kW, charger power stayed
        # 0.0, switched off at 12:08:05, then locked out by min_off_s for 15 min).
        # Substitute the commanded draw ONLY while the reading provably predates our
        # write; the moment the sensor catches up, measurement wins again — including
        # when it reports 0 because the car declined.
        if c.trust_commanded_draw and c.id in self._last_a:
            cmd_ts = self._last_cmd_ts.get(c.id)
            if cmd_ts is not None and power_ts is not None and power_ts < cmd_ts:
                power = self._last_a[c.id] * c.phases * c.voltage_v

        start_inhibited = (
            now_ts - self._last_stop_ts.get(c.id, float("-inf"))
        ) < c.min_off_s
        return ChargerState(
            id=c.id, plugged=plugged, at_home=at_home, enabled=True,
            current_power_w=power or 0.0, max_current_a=c.max_current_a,
            min_current_a=c.min_current_a, phases=c.phases, voltage_v=c.voltage_v,
            controllable=c.controllable, priority=c.priority, override=override,
            soc_percent=soc, target_soc_percent=target, floor_soc_percent=c.floor_soc,
            comfort_soc_percent=c.comfort_soc, phase_map=c.phase_map,
            capacity_kwh=c.capacity_kwh,
            deadline_hours=deadline_hours, charge_efficiency=c.charge_efficiency,
            commanded_on=commanded_on, start_inhibited=start_inhibited,
        )

    async def run(
        self,
        ha: Any,
        now_ts: float,
        shadow: bool = False,
        plan_kw: dict[str, float] | None = None,
        plan_battery_charge_kw: float = 0.0,
    ) -> dict[str, Any]:
        """One control cycle. Returns a summary (for logging / UI).

        ``plan_kw`` is the executor's current-slot per-charger plan
        (SlotPlan.ev_charger_plans, kW) — the S3 bridge. Only chargers with
        plan_floor: true consume it, after the soc-gate and continuity hold.
        """
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
        # Hold last commands and skip the tick instead. With the fuse guard armed the
        # hold is only allowed briefly: the grid meter and the phase sensors share one
        # Modbus integration and die TOGETHER, so a prolonged skip means the guard is
        # blind while the cars hold their last (up to 16 A) commands — after
        # stale_after_s of consecutive skips, stop the cars (fail safe).
        if src[1] is None or src[2] is None:
            logger.warning(
                "EV surplus: core sensors unreadable (grid=%s battery=%s) — skipping tick",
                src[1],
                src[2],
            )
            if self.cfg.fuse_guard_enabled:
                if self._core_skip_since is None:
                    self._core_skip_since = now_ts
                elif (now_ts - self._core_skip_since) > self.cfg.fuse_stale_after_s:
                    await self._failsafe_stop_all(ha, now_ts, "core sensors dark", shadow)
                    return {"enabled": True, "fuse_failsafe": "core sensors dark"}
            return {"enabled": True, "skipped": "core sensors unreadable"}
        self._core_skip_since = None
        pv_w = src[0] or 0.0
        grid_w = src[1]
        battery_w = src[2]
        # Unknown home-battery SoC must DISABLE battery assist (0), never enable it (100).
        soc = src[3] if src[3] is not None else 0.0
        price = src[4] if src[4] is not None else 999.0
        remaining_solar = src[5] or 0.0
        vacation = bool(src[6])
        priority_order = cast("list[str] | None", src[7])
        # Fuse guard: fresh per-phase currents, or the fail-safe. The readings also
        # feed the engine's battery-charge cap via last_phase_currents_a.
        phase_currents: dict[str, float] = {}
        if self.cfg.fuse_guard_enabled:
            read = await self._read_phase_currents(ha, now_ts)
            if read is None:
                if (
                    self.last_phase_ok_ts is None
                    or (now_ts - self.last_phase_ok_ts) > self.cfg.fuse_stale_after_s
                ):
                    await self._failsafe_stop_all(ha, now_ts, "phase sensors stale", shadow)
                    return {"enabled": True, "fuse_failsafe": "phase sensors stale"}
                # Briefly stale: proceed with NO increases (pure clamp holds/reduces).
                phase_currents = {}
            else:
                phase_currents = read
                self.last_phase_currents_a = read
                self.last_phase_ok_ts = now_ts
        self.last_battery_w = battery_w
        self.last_grid_w = grid_w
        states: list[ChargerState] = list(
            await asyncio.gather(
                *(self._read_charger(ha, c, now_ts, vacation) for c in cfg.chargers)
            )
        )
        # S3 plan floors: gate + continuity hold, then attach to the states.
        # Vacation gates the WHOLE attach (mirroring the unconditional deadline
        # clear): plan slots computed pre-vacation must not grid-force an away
        # household even for the ~15 min until the vacation-stripped replan.
        cfg_by_id_pre = {c.id: c for c in cfg.chargers}
        self.last_plan_note = {}
        for st in states:
            ccfg = cfg_by_id_pre.get(st.id)
            if ccfg is None or not ccfg.plan_floor:
                continue
            if vacation:
                self._plan_hold.pop(st.id, None)
                self.last_plan_note[st.id] = "vacation"
                continue
            raw_w = max(0.0, float((plan_kw or {}).get(st.id, 0.0)) * 1000.0)
            min_on_w = max(ccfg.min_current_a, cfg.policy.min_charge_current_a) \
                * ccfg.voltage_v * ccfg.phases
            gated = (
                ccfg.plan_gate_soc is None
                or st.soc_percent is None
                or st.soc_percent >= ccfg.plan_gate_soc
            )
            plan_w = 0.0 if (gated or raw_w < min_on_w) else raw_w
            if plan_w > 0.0:
                # Hold horizon = ONE replan period (~16 min): bridges kepler's
                # 15-min checkerboard flips, but bounds the tail after a GENUINE
                # block end to <= one slot (a rolling 30-min hold fired at every
                # nightly block end, buying energy kepler priced as not-worth-it
                # — review-caught, with the battery-discharge lockout on top).
                self._plan_hold[st.id] = (plan_w, now_ts + 960.0)
                self.last_plan_note[st.id] = "active"
            else:
                held = self._plan_hold.get(st.id)
                if held is not None and now_ts < held[1] and not gated:
                    plan_w = held[0]  # replan flip — hold the started floor
                    logger.info(
                        "EV surplus: plan-floor hold for %s (%.0f W, %.0f s left)",
                        st.id, plan_w, held[1] - now_ts,
                    )
                    self.last_plan_note[st.id] = "active"
                else:
                    if held is not None and now_ts >= held[1]:
                        del self._plan_hold[st.id]
                    self.last_plan_note[st.id] = "gated" if gated else "idle"
            st.plan_floor_w = plan_w

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
            battery_tier_active_prev=self._battery_tier_prev,
            battery_reserve_active_prev=self._battery_reserve_prev,
            plan_battery_charge_w=max(0.0, plan_battery_charge_kw) * 1000.0,
            phase_currents_a=phase_currents, chargers=states,
        )
        commands = compute_ev_surplus(inputs, cfg.policy)
        if cfg.fuse_guard_enabled:
            _alloc: dict[str, float] = {}
            _by_id = {s2.id: s2 for s2 in states}
            for _cmd in commands:
                _st = _by_id.get(_cmd.id)
                if _st is None or not _cmd.switch_on:
                    continue
                _amps = (
                    float(_cmd.set_current_a)
                    if _cmd.set_current_a is not None
                    else _cmd.target_power_w / (_st.voltage_v * _st.phases)
                )
                _granted = max(
                    0.0, _amps - _st.current_power_w / (_st.voltage_v * _st.phases)
                )
                if _granted <= 0.0:
                    continue
                for _p in _st.phase_map or tuple(phase_currents.keys()):
                    _alloc[_p] = _alloc.get(_p, 0.0) + _granted
            self.last_ev_alloc_a = _alloc
        # Track the tier through the SAME helper the pure layer uses (hysteresis memory).
        self._battery_tier_prev = battery_tier_active(inputs, cfg.policy)
        reserve = battery_reserve_active(inputs, cfg.policy)
        if reserve != self._battery_reserve_prev:
            slack = battery_fill_slack_kwh(inputs, cfg.policy)
            logger.info(
                "EV surplus: battery fill-reserve %s (slack %s kWh, soc %.0f%%, "
                "remaining solar %.1f kWh)",
                "ENGAGED — battery claims its inflow" if reserve else "released — cars first",
                "n/a" if slack is None else f"{slack:.1f}",
                inputs.battery_soc_percent, inputs.remaining_solar_kwh,
            )
        self._battery_reserve_prev = reserve
        cfg_by_id = {c.id: c for c in cfg.chargers}

        applied: list[dict[str, Any]] = []
        for cmd in commands:
            ccfg = cfg_by_id.get(cmd.id)
            if ccfg is None:
                continue
            fail = self._act_fail.get(cmd.id)
            if fail is not None:
                backoff_s = min(900.0, 120.0 * (2.0 ** (fail[1] - 1)))
                if (now_ts - fail[0]) < backoff_s:
                    continue  # in failure backoff — spare the (Tesla) API
            try:
                await self._actuate(ha, ccfg, cmd, now_ts, shadow)
            except Exception:
                # One charger's dead entity must not starve the others' actuation,
                # and a sleeping Tesla must not be hammered every tick.
                n = (fail[1] + 1) if fail is not None else 1
                self._act_fail[cmd.id] = (now_ts, n)
                # A sleeping car is the dominant failure cause (switch.turn_on ->
                # HTTP 500). Press its wake button once per WAKE_COOLDOWN_S so the
                # backoff retry finds it awake; only for commands that wanted it ON
                # (a failed stop needs no wake — the car isn't drawing anyway).
                if ccfg.wake_entity and cmd.switch_on and not (shadow or ccfg.shadow):
                    last_wake = self._last_wake_ts.get(cmd.id, float("-inf"))
                    if (now_ts - last_wake) >= 300.0:
                        self._last_wake_ts[cmd.id] = now_ts
                        try:
                            await ha.call_service(
                                "button", "press", ccfg.wake_entity
                            )
                            logger.info(
                                "EV surplus: pressed wake button %s for %s",
                                ccfg.wake_entity, cmd.id,
                            )
                        except Exception:
                            logger.warning(
                                "EV surplus: wake press failed for %s", cmd.id
                            )
                logger.exception(
                    "EV surplus: actuation failed for %s (fail #%d, backoff %.0fs) — continuing",
                    cmd.id, n, min(900.0, 120.0 * (2.0 ** (n - 1))),
                )
                continue
            self._act_fail.pop(cmd.id, None)
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
            # Any write invalidates a power reading older than it (trust_commanded_draw).
            self._last_cmd_ts[ccfg.id] = now_ts

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
                self._last_cmd_ts[ccfg.id] = now_ts
            return

        # Pause a switchless current-entity charger (e.g. a Tesla wired without its
        # switch): write 0 A — otherwise an "off" command silently no-ops and the
        # car keeps charging at its last current (fail-safe included).
        if (
            not cmd.switch_on
            and ccfg.switch_entity is None
            and ccfg.easee_device_id is None
            and ccfg.current_entity
        ):
            prev_a = self._last_a.get(ccfg.id)
            if should_write_current(
                prev_a, self._last_ts.get(ccfg.id), 0.0, now_ts, guard
            ):
                if not shadow:
                    await ha.call_service(
                        "number", "set_value", ccfg.current_entity, {"value": 0}
                    )
                if prev_a is not None and prev_a > 0.0:
                    self._last_stop_ts[ccfg.id] = now_ts
                self._last_a[ccfg.id] = 0.0
                self._last_ts[ccfg.id] = now_ts
                self._last_cmd_ts[ccfg.id] = now_ts
            return

        # Current: only when on, controllable, and the write-guard allows it.
        if not (cmd.switch_on and ccfg.controllable and cmd.set_current_a is not None):
            return
        new_a = float(cmd.set_current_a)
        last_a = self._last_a.get(ccfg.id)
        # A fuse-guard reduction is overload relief — it must never be Schmitt-
        # suppressed or paced (same safety class as a stop).
        fuse_relief = bool(getattr(cmd, "fuse_limited", False)) and (
            last_a is None or new_a < last_a
        )
        # Schmitt quantizer: a +/-1-step move is only real once the RAW (unsnapped)
        # target has cleared schmitt_fraction of a step away from the written value.
        # Kills midpoint dither and the config-vs-real-voltage churn that a 1 A grid
        # would otherwise unmask. Stops and starts are exempt (handled above/guard).
        raw = getattr(cmd, "raw_amps", None)
        if (
            not fuse_relief
            and last_a is not None
            and last_a > 0.0
            and new_a > 0.0
            and new_a != last_a
            and raw is not None
            and abs(float(raw) - last_a)
            < self.cfg.policy.schmitt_fraction * max(0.0, self.cfg.policy.current_step_a)
        ):
            return
        if not should_write_current(
            last_a, self._last_ts.get(ccfg.id), new_a, now_ts, guard, fuse_relief=fuse_relief
        ):
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
        self._last_cmd_ts[ccfg.id] = now_ts
        self._last_ts[ccfg.id] = now_ts
