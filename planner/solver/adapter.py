"""
Kepler Adapter Module

Convert between Planner DataFrame format and Kepler solver types.
Migrated from backend/kepler/adapter.py during Rev K13 modularization.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .types import (
    DeferrableLoadInput,
    EVChargerInput,
    IncentiveBucket,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    KeplerResult,
    WaterHeaterInput,
)

logger = logging.getLogger(__name__)


def _get_config_version(config: dict[str, Any]) -> int:
    """Detect config format version for ARC15 migration compatibility."""
    return int(config.get("config_version", 1))


def build_water_heater_inputs(
    water_heaters_config: list[dict[str, Any]],
    global_wh_cfg: dict[str, Any] | None = None,
    water_heater_states: list[dict[str, Any]] | None = None,
) -> list[WaterHeaterInput]:
    """
    Build per-device WaterHeaterInput list from water_heaters[] config + state.

    Args:
        water_heaters_config: List of water heater config dicts from config.yaml
        global_wh_cfg: Global water_heating section (for enable_top_ups, spacing override)
        water_heater_states: Optional per-device state dicts.
            Each dict should have: id, heated_today_kwh (float), force_on_slots (list[int] | None)
            If None or empty, defaults are used.

    Returns:
        List of WaterHeaterInput objects for enabled heaters with power_kw > 0.
    """
    enabled_wh: list[dict[str, Any]] = [
        wh
        for wh in water_heaters_config
        if wh.get("enabled", True) and float(wh.get("power_kw", 0.0)) > 0
    ]

    if not enabled_wh:
        return []

    global_cfg = global_wh_cfg or {}
    enable_top_ups: bool = bool(global_cfg.get("enable_top_ups", True))

    # Build state lookup by heater ID
    state_by_id: dict[str, dict[str, Any]] = {}
    if water_heater_states:
        for s in water_heater_states:
            heater_id = s.get("id", "")
            if heater_id:
                state_by_id[heater_id] = s

    result: list[WaterHeaterInput] = []
    for wh in enabled_wh:
        heater_id: str = str(wh.get("id", ""))
        if not heater_id:
            continue

        # The config array uses 'water_min_spacing_hours' (with prefix) - handle both
        spacing_hours_raw: Any = wh.get("water_min_spacing_hours") or wh.get("min_spacing_hours")
        spacing_hours: float = float(spacing_hours_raw or 5.0)

        # When top-ups disabled globally, disable per-device spacing too
        if not enable_top_ups:
            spacing_hours = 0.0

        state = state_by_id.get(heater_id, {})
        heated_today = float(state.get("heated_today_kwh", 0.0))
        force_on_slots: list[int] | None = state.get("force_on_slots")

        result.append(
            WaterHeaterInput(
                id=heater_id,
                power_kw=float(wh.get("power_kw", 3.0)),
                min_kwh_per_day=float(wh.get("min_kwh_per_day", 0.0)),
                max_hours_between_heating=float(wh.get("max_hours_between_heating", 8.0)),
                min_spacing_hours=spacing_hours,
                force_on_slots=force_on_slots if force_on_slots else None,
                heated_today_kwh=heated_today,
            )
        )

    return result


def ev_state_for_solver(
    ha_state: dict[str, Any], charger_id: str, deadline: Any
) -> dict[str, Any]:
    """Build the per-charger state dict the solver consumes from a live HA state.

    Carries through ``at_home`` (the home-zone gate set by ``get_initial_state``) so a
    car that is away is excluded by ``build_ev_charger_inputs``. Dropping it here is
    exactly the bug that let an away EV get scheduled — keep this the single place that
    constructs the solver state so the gate cannot be silently bypassed.
    """
    return {
        "id": charger_id,
        "soc_percent": ha_state.get("soc_percent", 0.0),
        "plugged_in": ha_state.get("plugged_in", False),
        "at_home": ha_state.get("at_home", True),
        "deadline": deadline,
    }


def build_ev_charger_inputs(
    ev_chargers_config: list[dict[str, Any]],
    ev_charger_states: list[dict[str, Any]] | None = None,
) -> list[EVChargerInput]:
    """
    Build per-device EVChargerInput list from ev_chargers[] config + HA state.

    Args:
        ev_chargers_config: List of ev_charger config dicts from config.yaml
        ev_charger_states: Optional list of per-device HA state dicts.
            Each dict should have: id, soc_percent, plugged_in, deadline (optional)
            If None or empty, HA state defaults are used (0% SoC, not plugged in).

    Returns:
        List of EVChargerInput objects for enabled chargers.
        Chargers not plugged in are still included (solver skips them internally).
    """
    enabled_ev: list[dict[str, Any]] = [ev for ev in ev_chargers_config if ev.get("enabled", True)]

    # Filter out chargers registered as disabled (missing/zero max_power_kw)
    enabled_ev = [ev for ev in enabled_ev if not ev.get("disabled_reason")]

    # Build state lookup by charger ID
    state_by_id: dict[str, dict[str, Any]] = {}
    if ev_charger_states:
        for s in ev_charger_states:
            charger_id = s.get("id", "")
            if charger_id:
                state_by_id[charger_id] = s

    result: list[EVChargerInput] = []
    for ev in enabled_ev:
        charger_id: str = ev.get("id", "")
        if not charger_id:
            continue

        state = state_by_id.get(charger_id, {})

        soc_percent = float(state.get("soc_percent", 0.0))
        plugged_in = bool(state.get("plugged_in", False))
        # Home-zone gate (EV presence): when the car is away, exclude it from the plan
        # entirely so charging the car reports via its API at another location never
        # becomes a phantom load the MILP schedules around. Default True keeps existing
        # setups (no home_entity configured) unchanged.
        if not bool(state.get("at_home", True)):
            plugged_in = False
        deadline = state.get("deadline")  # datetime | None

        # Build incentive buckets from penalty_levels config
        penalty_levels: list[dict[str, Any]] = ev.get("penalty_levels", [])
        buckets: list[IncentiveBucket] = [
            IncentiveBucket(
                threshold_soc=float(p.get("max_soc", 100.0)),
                value_sek=float(p.get("penalty_sek", 0.0)),
            )
            for p in penalty_levels
            if "max_soc" in p or "penalty_sek" in p
        ]

        result.append(
            EVChargerInput(
                id=charger_id,
                max_power_kw=float(ev.get("max_power_kw") or 0.0),
                battery_capacity_kwh=float(ev.get("battery_capacity_kwh", 0.0)),
                current_soc_percent=soc_percent,
                plugged_in=plugged_in,
                deadline=deadline,
                incentive_buckets=buckets,
            )
        )

    return result


def planner_to_kepler_input(
    df: pd.DataFrame,
    initial_soc_kwh: float,
    max_dc_input_kw: float | None = None,
) -> KeplerInput:
    """
    Convert Planner DataFrame to KeplerInput.
    Expects DataFrame index to be timestamps (start_time).

    Args:
        df: DataFrame with forecast data
        initial_soc_kwh: Initial battery state of charge in kWh
        max_dc_input_kw: Maximum DC power from PV strings (clips PV forecast)
    """
    slots: list[KeplerInputSlot] = []

    # Ensure required columns exist
    required_cols = ["load_forecast_kwh", "pv_forecast_kwh", "import_price_sek_kwh"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    if "export_price_sek_kwh" not in df.columns:
        df["export_price_sek_kwh"] = df["import_price_sek_kwh"]

    for idx, row in df.iterrows():
        start_time: datetime = idx  # type: ignore[assignment]
        if "end_time" in df.columns:
            end_time: datetime = row["end_time"]  # type: ignore[assignment]
        else:
            end_time = start_time + pd.Timedelta(minutes=15)

        # Prefer adjusted forecasts if available (already represents Base Load)
        load = float(row.get("adjusted_load_kwh", row.get("load_forecast_kwh", 0.0)))
        pv = float(row.get("adjusted_pv_kwh", row.get("pv_forecast_kwh", 0.0)))

        # PV DC clipping: limit PV to max DC input capacity
        if max_dc_input_kw is not None:
            slot_hours = (end_time - start_time).total_seconds() / 3600.0
            max_pv_kwh = max_dc_input_kw * slot_hours
            if pv > max_pv_kwh:
                original_pv = pv
                pv = max_pv_kwh
                logger.debug(
                    f"PV clipped from {original_pv:.3f} to {pv:.3f} kWh at {start_time} "
                    f"(max_dc={max_dc_input_kw}kW)"
                )

        slots.append(
            KeplerInputSlot(
                start_time=start_time,
                end_time=end_time,
                load_kwh=load,
                pv_kwh=pv,
                import_price_sek_kwh=float(row["import_price_sek_kwh"]),
                export_price_sek_kwh=float(row["export_price_sek_kwh"]),
            )
        )

    return KeplerInput(slots=slots, initial_soc_kwh=initial_soc_kwh)  # type: ignore[arg-type]


def _comfort_level_to_penalty(
    comfort_level: int, daily_kwh: float = 0.0, heater_power_kw: float = 0.0
) -> dict[str, float]:
    """Map comfort level (1-5) to penalty configuration and dynamic window size.

    Level 1: Economy - comfort is nice-to-have, large windows for bulk heating
    Level 5: Maximum - almost hard constraint, small windows for frequent heating

    Args:
        comfort_level: 1-5 comfort level
        daily_kwh: Daily water heating requirement
        heater_power_kw: Water heater power rating
    """
    # Comfort multipliers for window sizing
    COMFORT_MULTIPLIERS = {
        1: 1.5,  # Economy: Larger windows = bulk heating in cheapest period
        2: 1.0,  # Balanced: Moderate windows
        3: 0.8,  # Neutral: Slight spacing preference
        4: 0.5,  # Priority: More frequent heating
        5: 0.25,  # Maximum: Very frequent = stable temperature
    }

    # Calculate dynamic window size
    if daily_kwh > 0 and heater_power_kw > 0:
        min_heating_hours = daily_kwh / heater_power_kw
        multiplier = COMFORT_MULTIPLIERS.get(comfort_level, 0.8)
        max_block_hours = min_heating_hours * multiplier
    else:
        # Fallback to current behavior if parameters missing
        max_block_hours = 2.0

    # Penalty parameters explained:
    # - water_reliability_penalty_sek: Applied ONCE PER DAY when daily minimum (min_kwh_per_day) is not met.
    #   Higher values = stricter enforcement of daily heating requirement.
    #   Example: 300 SEK = 300 SEK total penalty if day fails to meet minimum.
    # - water_block_start_penalty_sek: Applied ONCE PER HEATING BLOCK when a new heating session starts.
    #   Higher values = preference for fewer, longer blocks vs many short blocks.
    #   Example: 1.5 SEK = 1.5 SEK penalty for each heating block created.
    # - water_block_penalty_sek: Applied PER SLOT when heating block exceeds max_block_hours.
    #   Higher values = stricter enforcement of window size limits.
    #   Example: 10 SEK = 10 SEK penalty for each 15-minute slot that overshoots the window.
    # - max_block_hours: Maximum duration for a single heating block (calculated dynamically).
    #   Smaller values = more frequent heating = more stable temperature.

    COMFORT_MAP = {
        # Level: {reliability, block_start, block, max_block_hours}
        1: {
            "water_reliability_penalty_sek": 2.0,
            "water_block_start_penalty_sek": 1.5,
            "water_block_penalty_sek": 0.5,
            "max_block_hours": max_block_hours,
        },  # Economy
        2: {
            "water_reliability_penalty_sek": 7.0,
            "water_block_start_penalty_sek": 2.25,
            "water_block_penalty_sek": 1.0,
            "max_block_hours": max_block_hours,
        },  # Balanced
        3: {
            "water_reliability_penalty_sek": 15.0,
            "water_block_start_penalty_sek": 3.0,
            "water_block_penalty_sek": 2.0,
            "max_block_hours": max_block_hours,
        },  # Neutral
        4: {
            "water_reliability_penalty_sek": 30.0,
            "water_block_start_penalty_sek": 4.5,
            "water_block_penalty_sek": 5.0,
            "max_block_hours": max_block_hours,
        },  # Priority
        5: {
            "water_reliability_penalty_sek": 300.0,
            "water_block_start_penalty_sek": 1.0,
            "water_block_penalty_sek": 10.0,
            "max_block_hours": max_block_hours,
        },  # Maximum
    }
    # Default to Level 3 (Neutral) if invalid
    params = COMFORT_MAP.get(comfort_level, COMFORT_MAP[3]).copy()
    # Explicitly disable legacy gap penalty
    params["water_comfort_penalty_sek"] = 0.0
    return params


def _apply_bulk_mode_override(params: dict[str, float]) -> dict[str, float]:
    """Apply bulk heating mode override - forces single block behavior.

    Overrides ONLY block-related parameters while preserving reliability penalties.
    This allows users to request bulk heating (single block) while maintaining
    their chosen reliability level (e.g., Level 5 + bulk mode = strict reliability
    but consolidated heating).

    Args:
        params: Penalty parameters from comfort level

    Returns:
        Modified parameters with bulk mode overrides
    """
    params["max_block_hours"] = 24.0  # Allow entire day as one block
    params["water_block_penalty_sek"] = 0.0  # No penalty for long blocks
    # Keep water_reliability_penalty_sek and water_block_start_penalty_sek unchanged
    return params


def _slot_minutes(slots: list[Any] | None, default: float = 15.0) -> float:
    """Infer slot length in minutes from the first slot, falling back to default."""
    if slots and len(slots) >= 1:
        s = slots[0]
        try:
            return (s.end_time - s.start_time).total_seconds() / 60.0
        except (AttributeError, TypeError):
            pass
    return default


def _resolve_deadline_slot(
    load_cfg: dict[str, Any],
    slots: list[Any] | None,
    slot_minutes: float,
) -> tuple[int | None, bool]:
    """Resolve (deadline_slot, hard) from a load's deadline config.

    - ``hard_deadline``: finish by the next occurrence of an "HH:MM" clock time.
    - ``cheapest_within_hours``: soft finish-by a number of hours from now.
    Returns (None, False) when no deadline applies (run any time in horizon).
    """
    mode = str(load_cfg.get("deadline_mode", "cheapest_within_hours"))
    if mode == "hard_deadline" and slots:
        hhmm = load_cfg.get("hard_deadline")
        target = _next_clock_time(slots, hhmm)
        if target is None:
            return None, True
        # Last slot that ends at/before the deadline.
        idx = None
        for i, s in enumerate(slots):
            if getattr(s, "end_time", None) is not None and s.end_time <= target:
                idx = i
        return (idx if idx is not None else None), True
    # Soft "cheapest within X hours".
    window_h = float(load_cfg.get("window_hours", 12.0))
    n = max(1, round(window_h * 60.0 / slot_minutes))
    horizon = len(slots) if slots else n
    return min(horizon - 1, n - 1), False


def _next_clock_time(slots: list[Any], hhmm: Any) -> datetime | None:
    """Next datetime matching "HH:MM" at/after the first slot start."""
    if not slots or not hhmm:
        return None
    try:
        hour, minute = (int(x) for x in str(hhmm).split(":"))
    except (ValueError, AttributeError):
        return None
    base = getattr(slots[0], "start_time", None)
    if base is None:
        return None
    target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target < base:
        target = target + timedelta(days=1)
    return target


def build_deferrable_load_inputs(
    loads_config: list[dict[str, Any]],
    deferrable_load_states: list[dict[str, Any]] | None,
    slots: list[Any] | None,
    learned_phases: dict[str, str] | None = None,
) -> list[DeferrableLoadInput]:
    """Build DeferrableLoadInput list for loads with a *pending* run.

    A run is scheduled only when its state dict has ``pending=True`` (the HA
    bridge sets this when a cycle is queued/started). Duration and energy come
    from learned values in the state dict when present, otherwise the config
    seed. Loads without a pending run are omitted (no scheduling).

    Args:
        loads_config: ``deferrable_loads`` list from config.yaml.
        deferrable_load_states: per-load HA state dicts
            (``id``, ``pending``, optional ``duration_min``/``energy_kwh``/
            ``earliest_offset_min``).
        slots: KeplerInputSlot list (for slot length and deadline resolution).
    """
    slot_min = _slot_minutes(slots)
    state_by_id: dict[str, dict[str, Any]] = {
        str(s.get("id", "")): s for s in (deferrable_load_states or []) if s.get("id")
    }
    learned = learned_phases or {}
    out: list[DeferrableLoadInput] = []
    for cfg in loads_config:
        if not cfg.get("enabled", True):
            continue
        lid = str(cfg.get("id", ""))
        if not lid:
            continue
        state = state_by_id.get(lid)
        if not state or not state.get("pending", False):
            continue  # only schedule a run that is actually queued

        duration_min = float(state.get("duration_min", cfg.get("duration_min", 120.0)))
        energy_kwh = float(state.get("energy_kwh", cfg.get("energy_kwh", 1.0)))
        duration_slots = max(1, math.ceil(duration_min / slot_min))
        earliest_offset_min = float(state.get("earliest_offset_min", 0.0))
        earliest_slot = max(0, int(earliest_offset_min / slot_min))
        deadline_slot, hard = _resolve_deadline_slot(cfg, slots, slot_min)

        out.append(
            DeferrableLoadInput(
                id=lid,
                energy_kwh=energy_kwh,
                duration_slots=duration_slots,
                earliest_start_slot=earliest_slot,
                deadline_slot=deadline_slot,
                deadline_hard=hard,
                # Explicit config phase wins; otherwise fall back to the learned
                # device->phase mapping (Phase 3 control) so the deferrable phase
                # penalty balances real single-phase appliances without manual wiring.
                phase=cfg.get("phase") or learned.get(lid),
            )
        )
    return out


def config_to_kepler_config(
    planner_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    slots: list[Any] | None = None,
    water_heater_states: list[dict[str, Any]] | None = None,
    ev_charger_states: list[dict[str, Any]] | None = None,
    deferrable_load_states: list[dict[str, Any]] | None = None,
) -> KeplerConfig:
    """
    Convert the main config dictionary to KeplerConfig.

    Args:
        planner_config: Main configuration dictionary
        overrides: Optional runtime overrides
        slots: Optional list of KeplerInputSlot (Legacy argument, currently unused)
    """
    system = planner_config.get("system", {})
    battery = system.get("battery", planner_config.get("battery", {}))

    kepler_overrides: dict[str, Any] = {}
    if overrides and "kepler" in overrides:
        kepler_overrides = overrides["kepler"]

    def get_val(key: str, default: float) -> float:
        # Check runtime overrides first, then config file, then default
        # Check kepler_overrides (legacy support)
        if key in kepler_overrides:
            return float(kepler_overrides[key])

        val = planner_config.get(key)
        # Check overrides (standard)
        if overrides and key in overrides:
            val = overrides[key]
        return float(val) if val is not None else default

    # ARC15: Detect config format and load water heating settings
    config_version = _get_config_version(planner_config)
    water_heaters_cfg = planner_config.get("water_heaters", [])
    ev_chargers = planner_config.get("ev_chargers", [])
    global_wh = planner_config.get("water_heating", {})

    # Build per-device WaterHeaterInput list
    if config_version >= 2 and water_heaters_cfg:
        water_inputs = build_water_heater_inputs(water_heaters_cfg, global_wh, water_heater_states)
    else:
        # Legacy format: no per-device support
        water_inputs = []

    # For global comfort settings, use the water_heating section
    wh_cfg = global_wh

    # ARC15: Load EV charging settings (per-device)
    if config_version >= 2 and ev_chargers:
        ev_inputs = build_ev_charger_inputs(ev_chargers, ev_charger_states)
    else:
        ev_inputs = []

    capacity = float(battery.get("capacity_kwh", 13.5))
    charge_eff = float(battery.get("charge_efficiency", 0.95))
    discharge_eff = float(battery.get("discharge_efficiency", 0.95))

    # Dynamic Power Limits (Rev F17)
    # Hardware limits (Amps or Watts) drive the Optimizer limits (kW)
    inverter_cfg = planner_config.get("executor", {}).get("inverter", {})
    control_unit = inverter_cfg.get("control_unit", "A")

    if control_unit == "W":
        max_charge_kw = float(battery.get("max_charge_w", 0.0)) / 1000.0
        max_discharge_kw = float(battery.get("max_discharge_w", 0.0)) / 1000.0
    else:
        # Amps mode - use nominal voltage for planning
        voltage = float(battery.get("nominal_voltage_v", battery.get("system_voltage_v", 48.0)))
        max_charge_kw = (float(battery.get("max_charge_a", 0.0)) * voltage) / 1000.0
        max_discharge_kw = (float(battery.get("max_discharge_a", 0.0)) * voltage) / 1000.0

    kepler_cfg = KeplerConfig(
        capacity_kwh=capacity,
        min_soc_percent=float(battery.get("min_soc_percent", 10.0)),
        max_soc_percent=float(battery.get("max_soc_percent", 100.0)),
        max_charge_power_kw=max_charge_kw,
        max_discharge_power_kw=max_discharge_kw,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        wear_cost_sek_per_kwh=float(
            planner_config.get("battery_economics", {}).get(
                "battery_cycle_cost_kwh", get_val("wear_cost_sek_per_kwh", 0.0)
            )
        ),
        max_export_power_kw=(
            float(system.get("grid", {}).get("max_power_kw"))
            if system.get("grid", {}).get("max_power_kw")
            else None
        ),
        max_import_power_kw=(
            float(system.get("grid", {}).get("max_power_kw"))
            if system.get("grid", {}).get("max_power_kw")
            else None
        ),
        max_inverter_ac_kw=(
            float(system.get("inverter", {}).get("max_ac_power_kw"))
            if system.get("inverter", {}).get("max_ac_power_kw")
            else None
        ),
        ramping_cost_sek_per_kw=float(
            planner_config.get("kepler", {}).get(
                "ramping_cost_sek_per_kw", get_val("ramping_cost_sek_per_kw", 0.05)
            )
        ),
        curtailment_penalty_sek=float(
            planner_config.get("kepler", {}).get("curtailment_penalty_sek", 0.001)
        ),
        export_threshold_sek_per_kwh=get_val("export_threshold_sek_per_kwh", 0.0),
        # Per-device water heater inputs
        water_heaters=water_inputs,
        # Global water heating settings
        # PRODUCTION FIX B1: Disable gap constraint when enable_top_ups=false
        water_heating_max_gap_hours=float(
            wh_cfg.get("max_hours_between_heating", 8.0)
            if wh_cfg.get("enable_top_ups", True)
            else 0.0
        ),
        # Rev K23: Multi-Parameter Comfort Control (ALWAYS applied)
        # Rev K24: Apply bulk mode override if enable_top_ups=false
        # Use total power/daily_kwh across all heaters for max_block_hours calculation
        **(  # type: ignore[arg-type]
            _apply_bulk_mode_override(
                _comfort_level_to_penalty(
                    int(wh_cfg.get("comfort_level", 3)),
                    daily_kwh=sum(h.min_kwh_per_day for h in water_inputs),
                    heater_power_kw=sum(h.power_kw for h in water_inputs),
                )
            )
            if not wh_cfg.get("enable_top_ups", True)
            else _comfort_level_to_penalty(
                int(wh_cfg.get("comfort_level", 3)),
                daily_kwh=sum(h.min_kwh_per_day for h in water_inputs),
                heater_power_kw=sum(h.power_kw for h in water_inputs),
            )
        ),
        defer_up_to_hours=float(wh_cfg.get("defer_up_to_hours", 0.0)),
        # Rev E4: Export Toggle
        enable_export=bool(planner_config.get("export", {}).get("enable_export", True)),
        # Export SoC Floor: minimum SoC required to allow grid export
        export_floor_soc_percent=(
            float(planner_config.get("export", {}).get("export_floor_soc_percent"))
            if planner_config.get("export", {}).get("export_floor_soc_percent") is not None
            else None
        ),
        # Per-device EV charger inputs (multi-device support)
        ev_chargers=ev_inputs,
        # Excess PV dispatch
        excess_pv_slots=[],  # Populated by pipeline after forecast analysis
        excess_pv_sink=str(
            planner_config.get("executor", {}).get("excess_pv", {}).get("sink", "disabled")
        ),
        excess_pv_reward_sek_per_kwh=float(
            planner_config.get("executor", {})
            .get("excess_pv", {})
            .get("boost_reward_sek_per_kwh", 0.5)
        ),
        excess_pv_soc_threshold_percent=float(
            planner_config.get("executor", {})
            .get("excess_pv", {})
            .get("soc_threshold_percent", 95.0)
        ),
        excess_pv_custom_entity_power_kw=float(
            planner_config.get("executor", {})
            .get("excess_pv", {})
            .get("custom_entity", {})
            .get("power_kw", 1.0)
        ),
    )

    # Deferrable household loads (dishwasher, washing machine, ...).
    # Only pending runs are scheduled; the master toggle lives in the pipeline
    # (default off), so absent state/config means nothing changes.
    loads_cfg = planner_config.get("deferrable_loads", []) or []
    if loads_cfg:
        # Phase 3 control: fall back to learned device->phase mappings for any load
        # without an explicit config phase, so the deferrable phase penalty balances
        # real single-phase appliances. Opt-in via phase_observer; empty otherwise.
        phase_obs_cfg: dict[str, Any] = planner_config.get("phase_observer", {}) or {}
        learned_phases: dict[str, str] = {}
        if phase_obs_cfg.get("enabled", False):
            from backend.learning.phase_learning import load_device_phases

            learned_phases = load_device_phases(planner_config)
        kepler_cfg.deferrable_loads = build_deferrable_load_inputs(
            loads_cfg, deferrable_load_states, slots, learned_phases
        )
    defl_settings = planner_config.get("deferrable_load_settings", {}) or {}
    kepler_cfg.deferrable_soft_deadline_penalty_sek = float(
        defl_settings.get("soft_deadline_penalty_sek", 30.0)
    )
    kepler_cfg.deferrable_hard_deadline_penalty_sek = float(
        defl_settings.get("hard_deadline_penalty_sek", 1000.0)
    )
    kepler_cfg.deferrable_phase_penalty_sek = float(defl_settings.get("phase_penalty_sek", 0.0))

    return kepler_cfg


def kepler_result_to_dataframe(
    result: KeplerResult, capacity_kwh: float = 0.0, initial_soc_kwh: float = 0.0
) -> pd.DataFrame:
    """
    Convert KeplerResult to a DataFrame suitable for logging/comparison.
    Matches the column structure expected by the UI.
    """
    records: list[dict[str, Any]] = []
    prev_soc_kwh = initial_soc_kwh

    for s in result.slots:
        duration_h = (s.end_time - s.start_time).total_seconds() / 3600.0
        if duration_h <= 0:
            duration_h = 0.25

        charge_kw = s.charge_kwh / duration_h
        discharge_kw = s.discharge_kwh / duration_h

        action = "Hold"
        if charge_kw > 0.01:
            action = "Charge"
        elif discharge_kw > 0.01:
            action = "Export" if s.grid_export_kwh > 0.01 else "Discharge"

        entry_soc_kwh = prev_soc_kwh
        entry_soc_percent = (entry_soc_kwh / capacity_kwh * 100.0) if capacity_kwh > 0 else 0.0
        prev_soc_kwh = s.soc_kwh

        records.append(  # type: ignore[arg-type]
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "kepler_charge_kwh": s.charge_kwh,
                "kepler_discharge_kwh": s.discharge_kwh,
                "kepler_import_kwh": s.grid_import_kwh,
                "kepler_export_kwh": s.grid_export_kwh,
                "kepler_soc_kwh": s.soc_kwh,
                "kepler_cost_sek": s.cost_sek,
                "planned_cost_sek": (s.grid_import_kwh * s.import_price_sek_kwh)
                - (s.grid_export_kwh * s.export_price_sek_kwh),
                "battery_charge_kw": charge_kw,
                "battery_discharge_kw": discharge_kw,
                "discharge_kw": discharge_kw,  # Alias for simulation.py compatibility
                "charge_kw": min(s.charge_kwh, s.grid_import_kwh) / duration_h,
                "projected_soc_kwh": s.soc_kwh,
                "projected_soc_percent": (
                    (s.soc_kwh / capacity_kwh * 100.0) if capacity_kwh > 0 else 0.0
                ),
                "_entry_soc_percent": entry_soc_percent,
                "action": action,
                "grid_import_kw": s.grid_import_kwh / duration_h,
                "grid_export_kw": s.grid_export_kwh / duration_h,
                "import_kwh": s.grid_import_kwh,
                "export_kwh": s.grid_export_kwh,
                "water_heating_kw": s.water_heat_kw,  # Aggregate (backward compat)
                "water_heaters": s.water_heater_results,  # Per-device: heater_id -> kW
                "water_heating_boost": s.water_heating_boost,  # Per-device: heater_id -> bool
                "custom_entity_active": s.custom_entity_active,  # Whether custom entity should be on
                "water_from_grid_kwh": 0.0,
                "water_from_pv_kwh": 0.0,
                "water_from_battery_kwh": 0.0,
                "ev_charging_kw": s.ev_charge_kw,  # Aggregate EV charging power (backward compat)
                "ev_chargers": s.ev_charger_results,  # Per-device: charger_id -> kW
                "deferrable_load_kw": s.deferrable_load_kw,  # Aggregate deferrable-load power
                "deferrable_loads": s.deferrable_load_results,  # Per-device: load_id -> kW
                "projected_battery_cost": 0.0,
            }
        )

    df = pd.DataFrame(records)  # type: ignore[arg-type]
    if not df.empty:
        df.set_index("start_time", inplace=True)
    return df
