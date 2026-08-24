"""
Executor Configuration

Loads and validates the executor configuration from config.yaml.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def _int_or_none(value: Any) -> int | None:
    """Convert an optional numeric config value to int, or None when absent/blank.

    Used for per-device overrides where "not set" must fall back to a global —
    distinct from 0, which is a legitimate temperature.
    """
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    """Convert config value to str or None. Empty strings become None.

    Used to normalize entity IDs from YAML - empty values should be None, not empty strings.
    This ensures `if not entity:` guards work correctly in executor actions.

    Args:
        value: Any value from config (str, None, or other)

    Returns:
        str if value is non-empty string, None otherwise
    """
    if value is None or value == "" or str(value).strip() == "":
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    """Convert a config value to float, or None if absent/blank/unparseable."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_list(value: Any) -> list[str]:
    """Normalize a config value into a list of non-empty entity-id strings.

    Accepts a YAML list/tuple, a single scalar (wrapped into a one-element list),
    or None/blank (=> empty list). Blank/None elements are dropped. Used for
    ``control_pause_entities`` so both ``foo`` and ``[foo, bar]`` parse cleanly.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        items = cast("list[Any]", value)
        return [s for s in (_str_or_none(i) for i in items) if s is not None]
    s = _str_or_none(value)
    return [s] if s is not None else []


def _parse_departure_time(value: Any) -> str | None:
    """Parse departure time from config value.

    Handles both string "HH:MM" format and integer minutes-since-midnight (0-1439).
    Defensive conversion for YAML 1.1 sexagesimal misparse (e.g., 16:00 -> 960).

    Args:
        value: Any value from config (str, int, None, or other)

    Returns:
        str in "HH:MM" format if valid, None otherwise
    """
    if value is None or value == "":
        return None

    if isinstance(value, int):
        if 0 <= value <= 1439:
            return f"{value // 60:02d}:{value % 60:02d}"
        return None

    return str(value) or None


@dataclass
class InverterConfig:
    """Inverter control entity configuration."""

    # Standardized names (Rev IP4)
    work_mode: str | None = None
    soc_target: str | None = None
    grid_charging_enable: str | None = None
    grid_charge_power: str | None = None
    minimum_reserve: str | None = None
    grid_max_export_power: str | None = None
    max_charge_current: str | None = None
    max_discharge_current: str | None = None
    grid_max_export_power_switch: str | None = None
    max_charge_power: str | None = None
    max_discharge_power: str | None = None

    # Control unit (A or W)
    control_unit: str = "A"

    # Dynamic entities for complex profiles (Rev IP2)
    custom_entities: dict[str, str | None] = field(default_factory=dict[str, str | None])


@dataclass
class ExportCurtailmentConfig:
    """Price-conditioned grid-export curtailment (C3).

    When the current slot's effective export price (spot + premium + grid_benefit - fee, i.e.
    what you are actually paid) is below ``threshold_sek_per_kwh`` the executor forces the
    inverter export-power limit to 0 W so surplus PV is clipped instead of exported at a loss.
    Above the threshold the limit is restored to ``restore_limit_w`` (or, when that is 0, the
    feed-in limit auto-captured the moment before the first curtailment). Off by default.
    """

    enabled: bool = False
    threshold_sek_per_kwh: float = 0.0
    restore_limit_w: float = 0.0
    # "number" (legacy): clamp the export-limit NUMBER to 0 W / restore it to
    # restore_limit_w — the limit MODE switch stays on throughout.
    # "switch": curtail = write clamp_limit_w to the number (a known-device-legal
    # low value) + mode switch ON; restore = mode switch OFF (truly unlimited,
    # no number write). Built 2026-08-04 for the "no limit except minus prices"
    # policy on devices whose limit register rejects out-of-range values
    # (Sungrow SH10RT reg 13073: 10000 rejected with pymodbus isError, 8500
    # accepted — the exact ceiling is device-firmware-specific, so restoring by
    # writing a high number is fragile; switch-off is not).
    method: str = "number"
    # Curtailment level for method="switch" (W). Keep it a value the device has
    # demonstrably accepted (Burgbyn10: 400 W sat in the register for 11 days).
    clamp_limit_w: float = 400.0


@dataclass
class WaterHeaterGlobalConfig:
    """Global water heater temperature configuration (house-level preferences)."""

    temp_normal: int = 60
    temp_off: int = 40
    temp_boost: int = 70
    temp_max: int = 85
    # Manual-ON respect (switch/input_boolean targets only): when a HUMAN turns the
    # relay on while the plan wants it off, honor it as an implicit boost for this
    # many minutes instead of reverting on the next tick. 0 = legacy (plan always
    # wins). Anything the executor commanded itself is still enforced normally.
    manual_on_respect_minutes: float = 90.0
    # Anti-short-cycle dwell (switch/input_boolean targets only): minimum time the
    # relay must stay in a state before a plan-driven flip to the other state is
    # allowed. min_on holds a just-turned-ON relay ON (so a burst delivers
    # meaningful kWh and the Shelly relay is not hammered); min_off holds a
    # just-turned-OFF relay OFF (kills re-ignition thrash). Bypassed by boost
    # (force ON) and safety/override (force OFF). Worst-case toggle rate is bounded
    # to 1 cycle / (min_on + min_off). 0 disables the respective gate.
    min_on_minutes: float = 30.0
    min_off_minutes: float = 15.0


# Backward compatibility alias
WaterHeaterConfig = WaterHeaterGlobalConfig


class ExcessPVSinkType(Enum):
    """Type of excess PV sink."""

    WATER_HEATER_BOOST = "water_heater_boost"
    CUSTOM_ENTITY = "custom_entity"
    DISABLED = "disabled"


@dataclass
class ExcessPVCustomEntityConfig:
    """One excess-PV sink: an HA entity that soaks surplus PV when told to.

    Historically the single ``executor.excess_pv.custom_entity`` block; now also
    one rung of the ordered ``executor.excess_pv.sinks`` ladder (see
    ``ExcessPVSinkSpec`` alias). Supports two actuation styles, chosen
    automatically by the entity's domain:

    * Plain entities (switch/input_boolean/number/...): the executor writes
      ``on_value``/``off_value`` directly (legacy behaviour).
    * ``climate.*`` entities (e.g. the villavagn AC used as a surplus cooling sink):
      the executor calls ``climate.set_hvac_mode`` (``climate_mode`` when ON, ``off``
      when OFF) and ``climate.set_temperature`` (``target_temp``). If the unit's current
      temperature is already at or below ``comfort_min_temp`` the ON action is skipped so
      surplus never over-cools the space.

    ``price_ceiling_sek_per_kwh`` (planner-side gate) restricts activation to slots whose
    export price is at or below the ceiling — the "low or minus price" trigger.
    """

    id: str = "custom_entity"
    entity: str | None = None
    on_value: str = "1"
    off_value: str = "0"
    power_kw: float = 1.0
    # Independent opt-in: when True the custom-entity sink runs REGARDLESS of the
    # primary `sink` selector, so it can coexist with water_heater_boost (e.g. the
    # villavagn AC cooling sink alongside the main-VVB excess-PV boost). The legacy
    # path (sink == "custom_entity") still works and implies enabled.
    enabled: bool = False
    # Climate-sink fields (used only when ``entity`` is a climate.* entity).
    climate_mode: str = "cool"  # hvac_mode to set when ON
    target_temp: float | None = None  # setpoint to apply when ON (None => leave as-is)
    comfort_min_temp: float | None = None  # skip ON if current_temperature <= this
    # Planner-side export-price ceiling (SEK/kWh); None => no price gate.
    price_ceiling_sek_per_kwh: float | None = None
    # Control-pause: HA input_boolean entity ids that put this sink HANDS-OFF. The
    # sink is PAUSED (executor skips all actuation, leaving whatever a human set)
    # if ANY listed entity reads state "on". Used to hand the villavagn AC to a
    # renter. Empty => always managed. Fail-safe: an unreadable entity is treated
    # as NOT paused (normal control). See dispatcher.is_control_paused.
    control_pause_entities: list[str] = field(default_factory=lambda: [])


# A ladder rung IS the old custom-entity config plus an id — same actuation code.
ExcessPVSinkSpec = ExcessPVCustomEntityConfig


@dataclass
class ExcessPVConfig:
    """Excess PV dispatch configuration."""

    sink: ExcessPVSinkType = ExcessPVSinkType.DISABLED
    boost_reward_sek_per_kwh: float = 0.5
    soc_threshold_percent: float = 95.0
    custom_entity: ExcessPVCustomEntityConfig = field(default_factory=ExcessPVCustomEntityConfig)
    # Ordered excess-PV sink ladder (list order = priority order, after the
    # water-heater boost rung). Populated by the loader from
    # ``executor.excess_pv.sinks`` or synthesized from the legacy custom_entity
    # block via ``normalize_excess_pv_sinks`` (dual-read, no config migration).
    sinks: list[ExcessPVSinkSpec] = field(default_factory=lambda: [])


def normalize_excess_pv_sinks(excess_pv_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the ordered excess-PV sink ladder from an ``executor.excess_pv`` dict.

    Shared by the executor config loader AND the planner adapter so both sides see
    the exact same ladder. Dual-read semantics:

    * ``sinks`` present with at least one valid entry => it wins (list order =
      priority order). Disabled entries are KEPT (observe-first rollout: the
      planner skips them, the executor skips them, but the ladder shape is stable).
    * otherwise, when the legacy ``custom_entity`` block has an entity and is
      active (``enabled: true`` or ``sink: custom_entity``), synthesize a
      single-rung ladder from it LOSSLESSLY — a live config keeps behaving
      byte-identically without migration.

    Returns plain normalized dicts (floats coerced, blanks -> None) so each side
    can build its own dataclass without importing the other's types.
    """

    def _entry(sink_id: str, raw: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
        power = _float_or_none(raw.get("power_kw"))
        return {
            "id": sink_id,
            "entity": _str_or_none(raw.get("entity")),
            "enabled": enabled,
            "power_kw": power if power is not None else 1.0,
            "price_ceiling_sek_per_kwh": _float_or_none(raw.get("price_ceiling_sek_per_kwh")),
            "on_value": str(raw.get("on_value", "1")),
            "off_value": str(raw.get("off_value", "0")),
            "climate_mode": str(raw.get("climate_mode", "cool")),
            "target_temp": _float_or_none(raw.get("target_temp")),
            "comfort_min_temp": _float_or_none(raw.get("comfort_min_temp")),
            "control_pause_entities": _str_list(raw.get("control_pause_entities")),
        }

    custom_raw = excess_pv_data.get("custom_entity")
    custom: dict[str, Any] = custom_raw if isinstance(custom_raw, dict) else {}
    legacy_active = bool(custom.get("enabled", False)) or (
        str(excess_pv_data.get("sink", "disabled")).lower() == "custom_entity"
    )
    legacy_entity = _str_or_none(custom.get("entity"))

    raw_sinks = excess_pv_data.get("sinks")
    entries: list[dict[str, Any]] = []
    if isinstance(raw_sinks, list):
        for idx, raw in enumerate(cast("list[Any]", raw_sinks)):
            if not isinstance(raw, dict):
                logger.warning("executor.excess_pv.sinks[%d] is not a mapping - skipping", idx)
                continue
            raw_dict = cast("dict[str, Any]", raw)
            entity = _str_or_none(raw_dict.get("entity"))
            if entity is None:
                logger.warning("executor.excess_pv.sinks[%d] has no entity - skipping", idx)
                continue
            sink_id = _str_or_none(raw_dict.get("id")) or entity
            entries.append(_entry(sink_id, raw_dict, enabled=bool(raw_dict.get("enabled", False))))
    if entries:
        # Fail-loud: a non-empty sinks list REPLACES the legacy block. If the
        # legacy sink is live and its entity is not a rung, it silently stops
        # being planned/actuated — potentially left physically ON. Warn on
        # every load (this normalizer is shared by planner AND executor).
        # Matching by entity means listing the legacy entity as a disabled
        # rung is treated as an explicit choice and does not warn.
        if (
            legacy_active
            and legacy_entity is not None
            and legacy_entity not in {e["entity"] for e in entries}
        ):
            logger.warning(
                "executor.excess_pv.sinks is set and REPLACES the enabled legacy "
                "custom_entity sink: %s will NO LONGER be planned or actuated "
                "(it may be left ON) - add it as a rung in sinks[] to keep managing it",
                legacy_entity,
            )
        return entries

    if legacy_active and legacy_entity is not None:
        return [_entry("custom_entity", custom, enabled=True)]
    return []


@dataclass
class WaterHeaterDeviceConfig:
    """Per-device water heater control configuration."""

    id: str = ""
    name: str = ""
    target_entity: str | None = None
    power_kw: float = 3.0
    # Control-pause: HA input_boolean entity ids that put this heater HANDS-OFF.
    # The device is PAUSED (executor skips all actuation — plan, boost, and forced
    # OFF — leaving whatever a human set) if ANY listed entity reads state "on".
    # Used to hand the villavagn VVB to a renter. Empty (e.g. main_tank) => always
    # managed. Fail-safe: an unreadable entity is treated as NOT paused (normal
    # control). See dispatcher.is_control_paused.
    control_pause_entities: list[str] = field(default_factory=lambda: [])
    # Per-device temperature overrides (None => the global executor.water_heater
    # values). Needed for thermostatic loads whose range differs from the tanks':
    # the spa runs 20-40 C, so the global temp_off=40 would read as "heat to 40"
    # on its bridge and temp_normal=60 exceeds its max. The values are written to
    # the device's target_entity exactly like the globals — for a numeric target
    # (input_number bridge) they ARE the setpoint; for a switch target only the
    # off/on distinction matters.
    temp_off: int | None = None
    temp_normal: int | None = None
    temp_boost: int | None = None
    temp_max: int | None = None
    # Self-thermostatted heater (the spa): when the plan wants OFF, skip the write
    # while the appliance is idle or heating on export, so its own thermostat keeps
    # the standing warmth. See executor/water_hold.py for the full rule. Dumb tanks
    # leave this False — for them an off-write IS the control.
    idle_hold: bool = False
    # Above this import price the off-write always happens, idle or not. None =
    # no ceiling. Set it at/near the heater's WTP so the hold never outbids the plan.
    idle_hold_max_price_sek_per_kwh: float | None = None
    # ...or express that ceiling as a PERCENTILE of the coming price window instead, so
    # it tracks the market rather than a level tuned in one season. Wins over the
    # absolute value above when set. See price_percentile() in executor/water_hold.py.
    idle_hold_max_price_percentile: float | None = None
    # How long a window that percentile spans. Future-first, backfilled from today's
    # passed hours when the forward series is shorter (Nordpool publishes only today
    # and tomorrow, so ~48 h is the hard ceiling). A comfort load that can wait out a
    # whole expensive stretch wants MORE than 24 h — otherwise it just picks the
    # cheapest hours of an expensive day.
    idle_hold_price_window_hours: float = 24.0
    # At or below this measured load the heater counts as not heating.
    idle_power_w: float = 100.0
    # Measured power source for the idle test; falls back to the heater's own
    # power_sensor/sensor when unset.
    power_entity: str | None = None
    # Push to temp_boost while there is MEASURED surplus and the price is under the
    # idle-hold ceiling. Unlike idle_hold this commands heat, so it honours the daily
    # energy bound below (the planner's absorb cap, which it bypasses by acting in
    # real time rather than from the forecast).
    # The REAL appliance behind a bridge (e.g. climate.layzspa_temperature_control).
    # Writes are change-gated against target_entity — Darkstar's own helper — so an
    # externally moved appliance is invisible. Given this, the executor compares the
    # appliance's mode and measured draw against its own intent each tick and
    # re-asserts on a mismatch, the way the switch path already self-heals.
    state_entity: str | None = None
    # hvac_mode to force on a climate ``state_entity`` when the appliance sits in a
    # non-heating mode (LayZSpa: fan_only) while the plan intends heat. Nudging the
    # helper only asks the HA bridge to relay the target again — if the bridge missed
    # it, or the tub's own panel moved the mode, that never lands and the correction
    # loop just logs. Set (e.g. "heat") to let the executor drive the climate entity
    # directly. Unset = old behaviour, helper nudge only. Heat direction ONLY: the OFF
    # direction stays with the helper, since "off" on the appliance is not the same as
    # the bridge's fan_only and would stop the circulation pump.
    climate_heat_mode: str | None = None
    surplus_boost: bool = False
    # Let the S4 fuse guard force this heater OFF when a phase it sits on exceeds the
    # house budget. A tank waits an hour happily; a car may be leaving in the morning,
    # so without this the guard sheds the expensive option to save the cheap one.
    fuse_shed: bool = False
    # Manual override, mirroring the EV chargers' input_select: auto / force_on /
    # force_off. Acts on the EXECUTOR, so it takes effect on the next tick rather than
    # waiting for a replan — which is what a human wants from "heat it now".
    # It is the companion to may_skip_day: a load allowed to sit out an expensive
    # PERIOD may stay cold for days, so there has to be a way to say "anyway".
    override_entity: str | None = None
    # Auto-expiry in minutes, 0 = never. A forgotten force_on cannot run away (the
    # appliance's own thermostat caps the temperature) but it CAN quietly buy at peak
    # for days, so an expiry is worth setting on anything expensive.
    override_timeout_minutes: float = 0.0
    # Consecutive ticks the appliance may disagree with our intent before we tell a
    # human. Needs state_entity to mean anything. 0 disables the alert (drift is
    # still corrected — only the escalation goes quiet).
    drift_alert_after: int = 3
    # Which house phases this heater draws on (lowercase, matching the guard's
    # phase_entities keys). EMPTY = unknown => counts against EVERY phase, the same
    # conservative convention the EV phase_map uses.
    phase_map: tuple[str, ...] = ()
    absorb_cap_kwh_per_day: float | None = None


@dataclass
class CyclicLoadConfig:
    """A recurring on/off load: pool pump, filter, circulation.

    Same planning problem as a water heater — a daily energy need, splittable across
    slots, with spacing and gap constraints — so it rides the SAME solver primitive
    (see cyclic_loads_as_heater_specs in the planner adapter). What differs is only
    the actuation: a switch, not a temperature. Keeping the config surface honest
    matters, though: a pool pump should not have to be spelled as a water heater.
    """

    id: str
    name: str = ""
    switch_entity: str = ""
    power_kw: float = 0.0
    # Measured draw, for the same own-draw and verification logic the heaters use.
    power_entity: str | None = None
    # Hands-off while any of these is on (rent-out, guest mode).
    control_pause_entities: list[str] = field(default_factory=lambda: [])
    # input_select auto / force_on / force_off, mirroring the heaters and the EVs.
    override_entity: str | None = None
    override_timeout_minutes: float = 0.0
    # Let the S4 fuse guard shed it. Same conservative phase convention as elsewhere:
    # an empty phase_map counts against every phase.
    fuse_shed: bool = False
    phase_map: tuple[str, ...] = ()
    # Opportunistic run gates ON TOP of the plan (see executor/cyclic_run.py). They
    # can only ADD runtime, never cancel a planned block. max_extra_hours_per_day is
    # what makes them opportunistic: unset means no room, never unlimited.
    surplus_run: bool = False
    max_price_percentile: float | None = None
    presence_entities: list[str] = field(default_factory=lambda: [])
    presence_max_price_percentile: float | None = None
    price_window_hours: float = 24.0
    max_extra_hours_per_day: float | None = None
    enabled: bool = True


DEFAULT_PENALTY_LEVELS = {
    "emergency": 10.0,
    "high": 2.0,
    "normal": 0.5,
    "opportunistic": 0.1,
}


@dataclass
class EVChargerConfig:
    """EV charger control configuration (legacy single-charger)."""

    switch_entity: str | None = None
    max_power_kw: float = 7.4
    battery_capacity_kwh: float | None = None
    replan_on_plugin: bool = True
    replan_on_unplug: bool = False


@dataclass
class EVChargerDeviceConfig:
    """Per-device EV charger configuration."""

    id: str = ""
    switch_entity: str | None = None
    max_power_kw: float = 7.4
    battery_capacity_kwh: float | None = None
    replan_on_plugin: bool = True
    replan_on_unplug: bool = False
    departure_time: str | None = None


# --------------------------------------------------------------------------
# Real-time per-phase load balancing (main-fuse protection).
#
# Ported from upstream ergetie/darkstar (commits b96f58c2, 5dc3c2ca, 16d0c99f),
# which our fork never carried. Taken as a targeted port rather than a
# cherry-pick: those commits conflict across 12 files each against our diverged
# engine/config, while executor/load_balancer.py itself depends on nothing but
# LoadBalancingConfig -- so the logic transplants cleanly and its 40 upstream
# tests come along as a correctness oracle.
# --------------------------------------------------------------------------


class BalancedLoadType(Enum):
    """Type of device a load-balancing entry refers to."""

    EV_CHARGER = "ev_charger"
    WATER_HEATER = "water_heater"
    CUSTOM_ENTITY = "custom_entity"


@dataclass
class BalancedLoadConfig:
    """A single shed-able on/off load managed by the real-time load balancer.

    EV chargers configured with type="current" get dedicated ampere throttling
    (see EVChargerDeviceConfig) and do not need an entry here; this is for
    on/off shedding (water heaters, custom entities, and binary-type chargers).
    Give-way ordering lives in LoadBalancingConfig.give_way_order, not here.
    """

    device_type: BalancedLoadType = BalancedLoadType.WATER_HEATER
    device_id: str = ""
    phases: list[int] = field(default_factory=lambda: [])
    # Custom entity actuation (only used when device_type == CUSTOM_ENTITY)
    entity: str | None = None
    on_value: str = "1"
    off_value: str = "0"


@dataclass
class GiveWayOrderEntry:
    """One entry in the unified give-way order (top gives way first).

    kind="charger" references a type="current" ev_chargers[].id (throttle to
    floor, then pause); kind="shed" references a loads[].device_id (switch off).
    """

    kind: str = "shed"  # "charger" | "shed"
    id: str = ""


@dataclass
class LoadBalancingConfig:
    """Real-time per-phase load balancing (fuse protection) configuration."""

    enabled: bool = False
    # Sourced from system.grid.main_fuse_a in YAML; folded in here for convenience
    # since it is always consumed alongside the rest of this config as a unit.
    main_fuse_a: int | None = None
    resume_delay_s: int = 120
    resume_margin_percent: float = 90.0
    increase_step_a: int = 1
    sensor_stale_after_s: int = 30
    # Fallback voltage (V) for converting a power-mode phase to current when
    # that phase has no configured grid_voltage_l* entity. Unrelated to
    # ControllerConfig.nominal_voltage_v (DC battery voltage).
    nominal_voltage_v: float = 220.0
    loads: list[BalancedLoadConfig] = field(default_factory=lambda: [])
    # Unified give-way order across chargers and shed loads; the top entry
    # gives way first. Self-healed on load (see heal_give_way_order).
    give_way_order: list[GiveWayOrderEntry] = field(default_factory=lambda: [])
    # Notify (HA notify / Discord fallback) on shed, pause, and stale-fallback
    # transitions. Routine throttle/ramp adjustments never notify.
    notify_interventions: bool = False
    # Trigger one replan (via the plug/unplug replan path) after a charger has
    # been held below its planner target (or paused) this long, continuously.
    replan_after_throttled_s: int = 600


@dataclass
class NotificationConfig:
    """Notification settings per action type."""

    service: str | None = None
    on_charge_start: bool = True
    on_charge_stop: bool = False
    on_export_start: bool = True
    on_export_stop: bool = True
    on_water_heat_start: bool = True
    on_water_heat_stop: bool = False
    on_soc_target_change: bool = False
    on_override_activated: bool = True
    on_error: bool = True
    # A write that never reached the appliance. Off by default only in the
    # sense that no heater alerts until it has a state_entity to check.
    on_write_unverified: bool = True


@dataclass
class ControllerConfig:
    """Controller parameters for current/power calculations."""

    battery_capacity_kwh: float = 27.0
    nominal_voltage_v: float = 48.0
    min_voltage_v: float = 46.0
    min_charge_a: float = 10.0
    max_charge_a: float = 185.0
    max_discharge_a: float = 185.0
    round_step_a: float = 5.0
    write_threshold_a: float = 5.0
    # Watt-based limits
    max_charge_w: float = 5000.0
    max_discharge_w: float = 5000.0
    min_charge_w: float = 500.0
    round_step_w: float = 100.0
    write_threshold_w: float = 100.0
    charge_efficiency: float = 0.92
    # Runtime battery-export SoC floor (arbitrage gate R1): the planner's
    # export_floor_soc_percent exists only inside the MILP — a stale plan or SoC
    # drift between replans could otherwise force-discharge below the floor. The
    # controller downgrades an export intent to self_consumption at/below this.
    # Mirrors config export.export_floor_soc_percent.
    export_floor_soc_percent: float = 20.0


@dataclass
class ExecutorConfig:
    """Main executor configuration."""

    enabled: bool = False
    shadow_mode: bool = False  # Log only, don't execute
    interval_seconds: int = 300  # 5 minutes

    automation_toggle_entity: str | None = None
    manual_override_entity: str | None = None

    inverter: InverterConfig = field(default_factory=InverterConfig)
    water_heater: WaterHeaterGlobalConfig = field(default_factory=WaterHeaterGlobalConfig)
    water_heater_devices: list[WaterHeaterDeviceConfig] = field(default_factory=lambda: [])
    cyclic_loads: list[CyclicLoadConfig] = field(default_factory=lambda: [])
    load_balancing: LoadBalancingConfig = field(default_factory=LoadBalancingConfig)
    ev_charger: EVChargerConfig = field(default_factory=EVChargerConfig)  # legacy compat
    ev_chargers: list[EVChargerDeviceConfig] = field(default_factory=lambda: [])
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    excess_pv: ExcessPVConfig = field(default_factory=ExcessPVConfig)
    export_curtailment: ExportCurtailmentConfig = field(default_factory=ExportCurtailmentConfig)

    history_retention_days: int = 30
    schedule_path: str = "data/schedule.json"
    timezone: str = "Europe/Stockholm"
    pause_reminder_minutes: int = 30  # Send notification after N minutes paused

    # System profile toggles (Rev O1)
    has_solar: bool = True
    has_battery: bool = True
    has_water_heater: bool = True
    inverter_profile: str = "generic"


def load_yaml(path: str) -> dict[str, Any]:
    """Load YAML file with strict typing."""
    try:
        with Path(path).open(encoding="utf-8") as f:
            yaml_loader = YAML(typ="safe")
            raw_data = yaml_loader.load(f)  # pyright: ignore[reportUnknownMemberType]
            return cast("dict[str, Any]", raw_data) if isinstance(raw_data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error("Failed to load YAML %s: %s", path, e)
        return {}


def _parse_load_balancing_config(
    data: dict[str, Any], system_data: dict[str, Any]
) -> LoadBalancingConfig:
    """Parse system.grid.main_fuse_a and the top-level load_balancing: section."""
    grid_data: dict[str, Any] = (
        system_data.get("grid", {}) if isinstance(system_data.get("grid"), dict) else {}
    )
    main_fuse_a_raw = grid_data.get("main_fuse_a")
    main_fuse_a: int | None
    try:
        main_fuse_a = int(main_fuse_a_raw) if main_fuse_a_raw is not None else None
    except (TypeError, ValueError):
        logger.warning("Invalid system.grid.main_fuse_a value: %r", main_fuse_a_raw)
        main_fuse_a = None

    lb_data: dict[str, Any] = (
        data.get("load_balancing", {}) if isinstance(data.get("load_balancing"), dict) else {}
    )

    loads_raw = lb_data.get("loads", [])
    loads: list[BalancedLoadConfig] = []
    if isinstance(loads_raw, list):
        for item in cast("list[Any]", loads_raw):
            if not isinstance(item, dict):
                continue
            load_item = cast("dict[str, Any]", item)
            type_raw = str(load_item.get("device_type", "water_heater")).lower()
            try:
                device_type = BalancedLoadType(type_raw)
            except ValueError:
                logger.warning(
                    "load_balancing.loads: unknown device_type %r, skipping entry", type_raw
                )
                continue
            phases_raw = load_item.get("phases", [])
            phases = (
                [int(p) for p in cast("list[Any]", phases_raw)]
                if isinstance(phases_raw, list)
                else []
            )
            loads.append(
                BalancedLoadConfig(
                    device_type=device_type,
                    device_id=str(load_item.get("device_id", "")),
                    phases=phases,
                    entity=_str_or_none(load_item.get("entity")),
                    on_value=str(load_item.get("on_value", "1")),
                    off_value=str(load_item.get("off_value", "0")),
                )
            )

    give_way_raw = lb_data.get("give_way_order", [])
    give_way_order: list[GiveWayOrderEntry] = []
    if isinstance(give_way_raw, list):
        for item in cast("list[Any]", give_way_raw):
            if not isinstance(item, dict):
                continue
            entry_item = cast("dict[str, Any]", item)
            kind = str(entry_item.get("kind", "")).lower()
            entry_id = str(entry_item.get("id", ""))
            if kind not in ("charger", "shed") or not entry_id:
                logger.warning(
                    "load_balancing.give_way_order: invalid entry %r, skipping", entry_item
                )
                continue
            give_way_order.append(GiveWayOrderEntry(kind=kind, id=entry_id))

    return LoadBalancingConfig(
        enabled=bool(lb_data.get("enabled", False)),
        main_fuse_a=main_fuse_a,
        resume_delay_s=int(lb_data.get("resume_delay_s", LoadBalancingConfig.resume_delay_s)),
        resume_margin_percent=float(
            lb_data.get("resume_margin_percent", LoadBalancingConfig.resume_margin_percent)
        ),
        increase_step_a=int(lb_data.get("increase_step_a", LoadBalancingConfig.increase_step_a)),
        sensor_stale_after_s=int(
            lb_data.get("sensor_stale_after_s", LoadBalancingConfig.sensor_stale_after_s)
        ),
        nominal_voltage_v=float(
            lb_data.get("nominal_voltage_v", LoadBalancingConfig.nominal_voltage_v)
        ),
        loads=loads,
        give_way_order=give_way_order,
        notify_interventions=bool(lb_data.get("notify_interventions", False)),
        replan_after_throttled_s=int(
            lb_data.get("replan_after_throttled_s", LoadBalancingConfig.replan_after_throttled_s)
        ),
    )


def heal_give_way_order(lb: LoadBalancingConfig, current_type_charger_ids: list[str]) -> None:
    """Self-heal load_balancing.give_way_order on config load (in place).

    - Drops entries referencing devices that no longer exist, or chargers no
      longer type="current" (logged warning).
    - Appends current-type chargers missing from the list after the last
      charger entry (at the top when there is none).
    - Appends loads[] entries missing from the list at the end.
    """
    shed_ids = [ld.device_id for ld in lb.loads if ld.device_id]

    healed: list[GiveWayOrderEntry] = []
    for entry in lb.give_way_order:
        if (entry.kind == "charger" and entry.id in current_type_charger_ids) or (
            entry.kind == "shed" and entry.id in shed_ids
        ):
            healed.append(entry)
        else:
            logger.warning(
                "load_balancing.give_way_order: dropping %s entry '%s' — no matching "
                "%s (device removed or charger no longer type: current)",
                entry.kind,
                entry.id,
                "type: current EV charger" if entry.kind == "charger" else "loads[] entry",
            )

    listed_chargers = {e.id for e in healed if e.kind == "charger"}
    missing_chargers = [c for c in current_type_charger_ids if c not in listed_chargers]
    if missing_chargers:
        last_charger_idx = max((i for i, e in enumerate(healed) if e.kind == "charger"), default=-1)
        for offset, charger_id in enumerate(missing_chargers):
            healed.insert(
                last_charger_idx + 1 + offset, GiveWayOrderEntry(kind="charger", id=charger_id)
            )
            logger.info("load_balancing.give_way_order: appended missing charger '%s'", charger_id)

    listed_sheds = {e.id for e in healed if e.kind == "shed"}
    for shed_id in shed_ids:
        if shed_id not in listed_sheds:
            healed.append(GiveWayOrderEntry(kind="shed", id=shed_id))
            logger.info("load_balancing.give_way_order: appended missing shed load '%s'", shed_id)

    lb.give_way_order = healed


def load_executor_config(config_path: str = "config.yaml") -> ExecutorConfig:
    """
    Load executor configuration from config.yaml.

    Falls back to defaults if executor section is missing.
    """
    try:
        with Path(config_path).open(encoding="utf-8") as f:
            yaml_loader = YAML(typ="safe")
            raw_data = yaml_loader.load(f)  # pyright: ignore[reportUnknownMemberType]
            data: dict[str, Any] = (
                cast("dict[str, Any]", raw_data) if isinstance(raw_data, dict) else {}
            )
    except FileNotFoundError:
        logger.warning("Config file not found at %s, using defaults", config_path)
        return ExecutorConfig()
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return ExecutorConfig()

    # Get timezone from root config
    timezone = str(data.get("timezone", "Europe/Stockholm"))

    # System toggles (Rev O1)
    system_data: dict[str, Any] = (
        data.get("system", {}) if isinstance(data.get("system"), dict) else {}
    )
    has_solar = bool(system_data.get("has_solar", True))
    has_battery = bool(system_data.get("has_battery", True))
    has_water_heater = bool(system_data.get("has_water_heater", True))
    inverter_profile = str(system_data.get("inverter_profile", "generic"))

    # Load balancing is a TOP-LEVEL key, independent of the executor: section,
    # so it is parsed before the early return below — fuse protection must not
    # depend on whether the executor happens to be configured.
    load_balancing = _parse_load_balancing_config(data, system_data)

    # Self-heal give_way_order against the enabled type="current" chargers.
    ev_chargers_raw = data.get("ev_chargers", [])
    current_type_charger_ids: list[str] = []
    if isinstance(ev_chargers_raw, list):
        for idx, item in enumerate(cast("list[Any]", ev_chargers_raw)):
            if not isinstance(item, dict):
                continue
            charger_item = cast("dict[str, Any]", item)
            if not charger_item.get("enabled", True):
                continue
            if str(charger_item.get("type", "binary")).lower() != "current":
                continue
            current_type_charger_ids.append(str(charger_item.get("id", f"ev_charger_{idx}")))
    heal_give_way_order(load_balancing, current_type_charger_ids)

    executor_data: dict[str, Any] = (
        data.get("executor", {}) if isinstance(data.get("executor"), dict) else {}
    )
    if not executor_data:
        logger.info("No executor section in config, using defaults")
        return ExecutorConfig(timezone=timezone, load_balancing=load_balancing)

    # Parse nested configs
    inverter_data: dict[str, Any] = (
        executor_data.get("inverter", {}) if isinstance(executor_data.get("inverter"), dict) else {}
    )

    # Helper for fallback loading
    def get_ent(key: str, old_key: str) -> str | None:
        return _str_or_none(inverter_data.get(key) or inverter_data.get(old_key))

    inverter = InverterConfig(
        work_mode=get_ent("work_mode", "work_mode_entity"),
        soc_target=_str_or_none(
            inverter_data.get("soc_target")
            or inverter_data.get("soc_target_entity")
            or executor_data.get("soc_target_entity")  # Fallback to legacy root location
        ),
        grid_charging_enable=get_ent("grid_charging_enable", "grid_charging_entity"),
        grid_charge_power=get_ent("grid_charge_power", "grid_charge_power_entity"),
        minimum_reserve=get_ent("minimum_reserve", "minimum_reserve_entity"),
        grid_max_export_power=get_ent("grid_max_export_power", "grid_max_export_power_entity"),
        grid_max_export_power_switch=get_ent(
            "grid_max_export_power_switch", "grid_max_export_power_switch_entity"
        ),
        max_charge_current=get_ent("max_charge_current", "max_charging_current_entity"),
        max_discharge_current=get_ent("max_discharge_current", "max_discharging_current_entity"),
        max_charge_power=get_ent("max_charge_power", "max_charging_power_entity"),
        max_discharge_power=get_ent("max_discharge_power", "max_discharging_power_entity"),
        control_unit=str(inverter_data.get("control_unit", "A")),
        # Capture all other keys as custom entities (Rev IP2)
        # REV F71: Add "custom_entities" to exclusion set to prevent stringification of nested dict
        custom_entities={
            k: _str_or_none(v)
            for k, v in inverter_data.items()
            if k
            not in {
                "work_mode",
                "work_mode_entity",
                "soc_target",
                "soc_target_entity",
                "grid_charging_enable",
                "grid_charging_entity",
                "grid_charge_power",
                "grid_charge_power_entity",
                "minimum_reserve",
                "minimum_reserve_entity",
                "grid_max_export_power",
                "grid_max_export_power_entity",
                "grid_max_export_power_switch",
                "grid_max_export_power_switch_entity",
                "max_charge_current",
                "max_charging_current_entity",
                "max_discharge_current",
                "max_discharging_current_entity",
                "max_charge_power",
                "max_charging_power_entity",
                "max_discharge_power",
                "max_discharging_power_entity",
                "control_unit",
                "custom_entities",  # REV F71: Don't stringify nested custom_entities dict
            }
        },
    )

    # REV F71: Explicitly merge nested custom_entities from YAML
    # This handles the case where users define custom_entities as a nested dict
    nested_custom: dict[str, Any] = (
        inverter_data.get("custom_entities", {})
        if isinstance(inverter_data.get("custom_entities"), dict)
        else {}
    )
    for k, v in nested_custom.items():
        if k not in inverter.custom_entities or inverter.custom_entities.get(k) is None:
            inverter.custom_entities[k] = _str_or_none(v)

    water_data: dict[str, Any] = (
        executor_data.get("water_heater", {})
        if isinstance(executor_data.get("water_heater"), dict)
        else {}
    )

    # Global water heater temperature config (house-level preferences, from executor.water_heater)
    water_heater = WaterHeaterGlobalConfig(
        temp_normal=int(water_data.get("temp_normal", WaterHeaterGlobalConfig.temp_normal)),
        temp_off=int(water_data.get("temp_off", WaterHeaterGlobalConfig.temp_off)),
        temp_boost=int(water_data.get("temp_boost", WaterHeaterGlobalConfig.temp_boost)),
        temp_max=int(water_data.get("temp_max", WaterHeaterGlobalConfig.temp_max)),
        manual_on_respect_minutes=float(
            water_data.get(
                "manual_on_respect_minutes",
                WaterHeaterGlobalConfig.manual_on_respect_minutes,
            )
        ),
        min_on_minutes=float(
            water_data.get("min_on_minutes", WaterHeaterGlobalConfig.min_on_minutes)
        ),
        min_off_minutes=float(
            water_data.get("min_off_minutes", WaterHeaterGlobalConfig.min_off_minutes)
        ),
    )

    # Per-device water heater configs (from water_heaters[] array)
    water_heaters_array = data.get("water_heaters", [])
    water_heater_devices_list: list[WaterHeaterDeviceConfig] = []
    for idx, heater in enumerate(cast("list[dict[str, Any]]", water_heaters_array)):
        if not heater.get("enabled", True):
            continue
        target_ent = _str_or_none(heater.get("target_entity"))
        if not target_ent:
            continue  # Only include heaters with a target_entity
        heater_id = str(heater.get("id", f"water_heater_{idx}"))
        water_heater_devices_list.append(
            WaterHeaterDeviceConfig(
                id=heater_id,
                name=str(heater.get("name", heater_id)),
                target_entity=target_ent,
                power_kw=float(heater.get("power_kw", WaterHeaterDeviceConfig.power_kw)),
                control_pause_entities=_str_list(heater.get("control_pause_entities")),
                temp_off=_int_or_none(heater.get("temp_off")),
                temp_normal=_int_or_none(heater.get("temp_normal")),
                temp_boost=_int_or_none(heater.get("temp_boost")),
                temp_max=_int_or_none(heater.get("temp_max")),
                idle_hold=bool(heater.get("idle_hold", False)),
                idle_hold_max_price_sek_per_kwh=_float_or_none(
                    heater.get("idle_hold_max_price_sek_per_kwh")
                ),
                idle_hold_max_price_percentile=_float_or_none(
                    heater.get("idle_hold_max_price_percentile")
                ),
                idle_hold_price_window_hours=float(
                    heater.get("idle_hold_price_window_hours")
                    or WaterHeaterDeviceConfig.idle_hold_price_window_hours
                ),
                idle_power_w=float(
                    heater.get("idle_power_w", WaterHeaterDeviceConfig.idle_power_w)
                ),
                power_entity=_str_or_none(
                    heater.get("power_sensor") or heater.get("sensor")
                ),
                state_entity=_str_or_none(heater.get("state_entity")),
                climate_heat_mode=_str_or_none(heater.get("climate_heat_mode")),
                surplus_boost=bool(heater.get("surplus_boost", False)),
                fuse_shed=bool(heater.get("fuse_shed", False)),
                override_entity=_str_or_none(heater.get("override_entity")),
                override_timeout_minutes=float(
                    heater.get("override_timeout_minutes") or 0.0
                ),
                drift_alert_after=int(
                    heater.get("drift_alert_after")
                    if heater.get("drift_alert_after") is not None
                    else WaterHeaterDeviceConfig.drift_alert_after
                ),
                phase_map=tuple(
                    str(x).strip().lower()
                    for x in (heater.get("phase_map") or [])
                    if str(x).strip()
                ),
                absorb_cap_kwh_per_day=_float_or_none(
                    heater.get("absorb_cap_kwh_per_day")
                ),
            )
        )

    # Cyclic loads (pool pump, filter, ...) — planned like a water heater, switched
    # like a switch. See CyclicLoadConfig.
    cyclic_loads: list[CyclicLoadConfig] = []
    for raw_cl in cast("list[dict[str, Any]]", data.get("cyclic_loads", []) or []):
        if not isinstance(raw_cl, dict):
            continue
        cl_id = str(raw_cl.get("id") or "").strip()
        sw = _str_or_none(raw_cl.get("switch_entity"))
        if not cl_id or not sw:
            logger.warning(
                "cyclic_load %r skipped: needs both id and switch_entity", cl_id or raw_cl
            )
            continue
        cyclic_loads.append(
            CyclicLoadConfig(
                id=cl_id,
                name=str(raw_cl.get("name", cl_id)),
                switch_entity=sw,
                power_kw=float(raw_cl.get("power_kw", 0.0) or 0.0),
                power_entity=_str_or_none(
                    raw_cl.get("power_sensor") or raw_cl.get("sensor")
                ),
                control_pause_entities=_str_list(raw_cl.get("control_pause_entities")),
                override_entity=_str_or_none(raw_cl.get("override_entity")),
                override_timeout_minutes=float(
                    raw_cl.get("override_timeout_minutes") or 0.0
                ),
                fuse_shed=bool(raw_cl.get("fuse_shed", False)),
                phase_map=tuple(
                    str(x).strip().lower()
                    for x in (raw_cl.get("phase_map") or [])
                    if str(x).strip()
                ),
                surplus_run=bool(raw_cl.get("surplus_run", False)),
                max_price_percentile=_float_or_none(
                    raw_cl.get("max_price_percentile")
                ),
                presence_entities=_str_list(raw_cl.get("presence_entities")),
                presence_max_price_percentile=_float_or_none(
                    raw_cl.get("presence_max_price_percentile")
                ),
                price_window_hours=float(raw_cl.get("price_window_hours", 24.0) or 24.0),
                max_extra_hours_per_day=_float_or_none(
                    raw_cl.get("max_extra_hours_per_day")
                ),
                enabled=bool(raw_cl.get("enabled", True)),
            )
        )

    # EV Charger config (REV K25 Phase 5)
    ev_data: dict[str, Any] = (
        executor_data.get("ev_charger", {})
        if isinstance(executor_data.get("ev_charger"), dict)
        else {}
    )
    ev_charger = EVChargerConfig(
        switch_entity=_str_or_none(ev_data.get("switch_entity")),
        max_power_kw=float(ev_data.get("max_power_kw", EVChargerConfig.max_power_kw)),
        battery_capacity_kwh=ev_data.get("battery_capacity_kwh"),
        replan_on_plugin=bool(ev_data.get("replan_on_plugin", EVChargerConfig.replan_on_plugin)),
        replan_on_unplug=bool(ev_data.get("replan_on_unplug", EVChargerConfig.replan_on_unplug)),
    )

    # Per-device EV charger config (multi-device support)
    ev_chargers_array = data.get("ev_chargers", [])
    ev_chargers_list: list[EVChargerDeviceConfig] = []
    for idx, charger in enumerate(cast("list[dict[str, Any]]", ev_chargers_array)):
        if not charger.get("enabled", True):
            continue
        charger_id = str(charger.get("id", f"ev_charger_{idx}"))
        ev_chargers_list.append(
            EVChargerDeviceConfig(
                id=charger_id,
                switch_entity=_str_or_none(charger.get("switch_entity")),
                max_power_kw=float(
                    charger.get("max_power_kw") or EVChargerDeviceConfig.max_power_kw
                ),
                battery_capacity_kwh=charger.get("battery_capacity_kwh"),
                replan_on_plugin=bool(
                    charger.get("replan_on_plugin", EVChargerDeviceConfig.replan_on_plugin)
                ),
                replan_on_unplug=bool(
                    charger.get("replan_on_unplug", EVChargerDeviceConfig.replan_on_unplug)
                ),
                departure_time=_parse_departure_time(charger.get("departure_time")),
            )
        )

    notif_data: dict[str, Any] = (
        executor_data.get("notifications", {})
        if isinstance(executor_data.get("notifications"), dict)
        else {}
    )
    notifications = NotificationConfig(
        service=_str_or_none(notif_data.get("service", NotificationConfig.service)),
        on_charge_start=bool(notif_data.get("on_charge_start", NotificationConfig.on_charge_start)),
        on_charge_stop=bool(notif_data.get("on_charge_stop", NotificationConfig.on_charge_stop)),
        on_export_start=bool(notif_data.get("on_export_start", NotificationConfig.on_export_start)),
        on_export_stop=bool(notif_data.get("on_export_stop", NotificationConfig.on_export_stop)),
        on_water_heat_start=bool(
            notif_data.get("on_water_heat_start", NotificationConfig.on_water_heat_start)
        ),
        on_water_heat_stop=bool(
            notif_data.get("on_water_heat_stop", NotificationConfig.on_water_heat_stop)
        ),
        on_soc_target_change=bool(
            notif_data.get("on_soc_target_change", NotificationConfig.on_soc_target_change)
        ),
        on_override_activated=bool(
            notif_data.get("on_override_activated", NotificationConfig.on_override_activated)
        ),
        on_error=bool(notif_data.get("on_error", NotificationConfig.on_error)),
    )

    # Root battery config (New SSOT for REV F17)
    battery_data: dict[str, Any] = (
        data.get("battery", {}) if isinstance(data.get("battery"), dict) else {}
    )

    ctrl_data: dict[str, Any] = (
        executor_data.get("controller", {})
        if isinstance(executor_data.get("controller"), dict)
        else {}
    )

    # Function to get with fallback (Rev F17 Migration)
    def get_fb(
        key: str,
        legacy_key: str,
        default: Any,
        source: dict[str, Any] = battery_data,
        legacy_source: dict[str, Any] = ctrl_data,
    ) -> Any:
        # 1. Try new source
        val: Any = source.get(key)
        if val is not None:
            return val
        # 2. Try legacy source
        val = legacy_source.get(legacy_key)
        if val is not None:
            # logger.warning(f"Using legacy config key '{legacy_key}'. Please move to battery section.") # Logged by migration module
            return val
        return default

    controller = ControllerConfig(
        battery_capacity_kwh=float(
            str(
                get_fb(
                    "capacity_kwh", "battery_capacity_kwh", ControllerConfig.battery_capacity_kwh
                )
            )
        ),
        nominal_voltage_v=float(
            str(get_fb("nominal_voltage_v", "system_voltage_v", ControllerConfig.nominal_voltage_v))
        ),
        min_voltage_v=float(
            str(get_fb("min_voltage_v", "worst_case_voltage_v", ControllerConfig.min_voltage_v))
        ),
        min_charge_a=float(str(ctrl_data.get("min_charge_a", ControllerConfig.min_charge_a))),
        max_charge_a=float(
            str(get_fb("max_charge_a", "max_charge_a", ControllerConfig.max_charge_a))
        ),
        max_discharge_a=float(
            str(get_fb("max_discharge_a", "max_discharge_a", ControllerConfig.max_discharge_a))
        ),
        round_step_a=float(str(ctrl_data.get("round_step_a", ControllerConfig.round_step_a))),
        write_threshold_a=float(
            str(ctrl_data.get("write_threshold_a", ControllerConfig.write_threshold_a))
        ),
        max_charge_w=float(
            str(get_fb("max_charge_w", "max_charge_w", ControllerConfig.max_charge_w))
        ),
        max_discharge_w=float(
            str(get_fb("max_discharge_w", "max_discharge_w", ControllerConfig.max_discharge_w))
        ),
        min_charge_w=float(str(ctrl_data.get("min_charge_w", ControllerConfig.min_charge_w))),
        round_step_w=float(str(ctrl_data.get("round_step_w", ControllerConfig.round_step_w))),
        write_threshold_w=float(
            str(ctrl_data.get("write_threshold_w", ControllerConfig.write_threshold_w))
        ),
        charge_efficiency=float(
            str(ctrl_data.get("charge_efficiency", ControllerConfig.charge_efficiency))
        ),
        # From the ROOT export block — the same floor the planner enforces in-model.
        export_floor_soc_percent=float(
            str(
                (data.get("export", {}) if isinstance(data.get("export"), dict) else {}).get(
                    "export_floor_soc_percent", ControllerConfig.export_floor_soc_percent
                )
            )
        ),
    )

    excess_pv_data: dict[str, Any] = (
        executor_data.get("excess_pv", {})
        if isinstance(executor_data.get("excess_pv"), dict)
        else {}
    )
    sink_raw = str(excess_pv_data.get("sink", "disabled")).lower()
    try:
        sink_type = ExcessPVSinkType(sink_raw)
    except ValueError:
        sink_type = ExcessPVSinkType.DISABLED

    custom_entity_data: dict[str, Any] = (
        excess_pv_data.get("custom_entity", {})
        if isinstance(excess_pv_data.get("custom_entity"), dict)
        else {}
    )
    custom_entity = ExcessPVCustomEntityConfig(
        entity=_str_or_none(custom_entity_data.get("entity")),
        on_value=str(custom_entity_data.get("on_value", "1")),
        off_value=str(custom_entity_data.get("off_value", "0")),
        power_kw=float(custom_entity_data.get("power_kw", 1.0)),
        enabled=bool(custom_entity_data.get("enabled", False)),
        climate_mode=str(custom_entity_data.get("climate_mode", "cool")),
        target_temp=_float_or_none(custom_entity_data.get("target_temp")),
        comfort_min_temp=_float_or_none(custom_entity_data.get("comfort_min_temp")),
        price_ceiling_sek_per_kwh=_float_or_none(
            custom_entity_data.get("price_ceiling_sek_per_kwh")
        ),
    )
    excess_pv_sinks = [
        ExcessPVSinkSpec(
            id=str(d["id"]),
            entity=cast("str | None", d["entity"]),
            on_value=str(d["on_value"]),
            off_value=str(d["off_value"]),
            power_kw=float(d["power_kw"]),
            enabled=bool(d["enabled"]),
            climate_mode=str(d["climate_mode"]),
            target_temp=cast("float | None", d["target_temp"]),
            comfort_min_temp=cast("float | None", d["comfort_min_temp"]),
            price_ceiling_sek_per_kwh=cast("float | None", d["price_ceiling_sek_per_kwh"]),
            control_pause_entities=cast("list[str]", d["control_pause_entities"]),
        )
        for d in normalize_excess_pv_sinks(excess_pv_data)
    ]
    excess_pv = ExcessPVConfig(
        sink=sink_type,
        boost_reward_sek_per_kwh=float(excess_pv_data.get("boost_reward_sek_per_kwh", 0.5)),
        soc_threshold_percent=float(excess_pv_data.get("soc_threshold_percent", 95.0)),
        custom_entity=custom_entity,
        sinks=excess_pv_sinks,
    )

    ec_data: dict[str, Any] = (
        executor_data.get("export_curtailment", {})
        if isinstance(executor_data.get("export_curtailment"), dict)
        else {}
    )
    export_curtailment = ExportCurtailmentConfig(
        enabled=bool(ec_data.get("enabled", False)),
        threshold_sek_per_kwh=float(ec_data.get("threshold_sek_per_kwh", 0.0)),
        restore_limit_w=float(ec_data.get("restore_limit_w", 0.0)),
        method=str(ec_data.get("method", "number")),
        clamp_limit_w=float(ec_data.get("clamp_limit_w", 400.0)),
    )

    return ExecutorConfig(
        enabled=bool(executor_data.get("enabled", False)),
        shadow_mode=bool(executor_data.get("shadow_mode", False)),
        interval_seconds=int(executor_data.get("interval_seconds", 300)),
        automation_toggle_entity=_str_or_none(executor_data.get("automation_toggle_entity")),
        manual_override_entity=_str_or_none(executor_data.get("manual_override_entity")),
        inverter=inverter,
        water_heater=water_heater,
        water_heater_devices=water_heater_devices_list,
        cyclic_loads=cyclic_loads,
        load_balancing=load_balancing,
        ev_charger=ev_charger,
        ev_chargers=ev_chargers_list,
        notifications=notifications,
        controller=controller,
        excess_pv=excess_pv,
        export_curtailment=export_curtailment,
        history_retention_days=int(executor_data.get("history_retention_days", 30)),
        schedule_path=str(executor_data.get("schedule_path", "data/schedule.json")),
        timezone=timezone,
        pause_reminder_minutes=int(executor_data.get("pause_reminder_minutes", 30)),
        has_solar=has_solar,
        has_battery=has_battery,
        has_water_heater=has_water_heater,
        inverter_profile=inverter_profile,
    )
