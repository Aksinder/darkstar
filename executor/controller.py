"""
Controller Logic

Action decision-making based on slot plan and current state.
Ported from the n8n Helios Executor "Controller" code node.

Determines:
- Which inverter work mode to set
- Whether to enable grid charging
- What charge/discharge currents to command
- SoC target to set
- Water heater temperature target
"""

import logging
import math
from dataclasses import dataclass

# from typing import Any, Dict, Optional, Tuple
from .config import (
    ControllerConfig,
    InverterConfig,
    WaterHeaterDeviceConfig,
    WaterHeaterGlobalConfig,
)
from .override import OverrideResult, SlotPlan, SystemState
from .profiles import InverterProfile

logger = logging.getLogger(__name__)

# The planner labels a slot "Hold" when both battery charge and discharge are
# <= 0.01 kW (planner/solver/adapter.py kepler_result_to_dataframe). The
# controller uses the same threshold so a rounding-dust discharge does not turn
# a Hold slot back into self_consumption.
HOLD_DISCHARGE_EPS_KW = 0.01


def effective_export_floor_pct(config_floor: float, state: object) -> float:
    """The export floor in force right now.

    A vacation floor may only ever LOWER it. The guard itself must survive any
    away-mode relaxation: it exists because SoC can drift below the plan's
    assumption and Sungrow forced-discharge would otherwise run to BMS cutoff at
    full power. The vacation value is RUNTIME state (the engine re-reads
    input_sensors.vacation_mode every tick), so homecoming restores the
    configured floor on the very next tick — no stale-config class.
    """
    vac = getattr(state, "vacation_export_floor_percent", None)
    if vac is None:
        return config_floor
    return min(float(config_floor), float(vac))


class MinSocFloorLatch:
    """Min-SoC floor guard with hysteresis (upstream de8b93ee, implemented natively).

    ENGAGES when the rounded live SoC is at or below ``min_soc_percent`` and
    RELEASES only once it reaches ``min_soc_percent + release_margin``. The
    Controller is rebuilt every tick, so the engine owns one instance for its
    lifetime and copies the result into ``SystemState.min_soc_latched``.

    - Unknown SoC neither engages nor releases: the previous state stands (a held
      floor stays held — the safe direction; an unlatched one stays unlatched).
    - Restart: the latch starts released, so the first known reading applies the
      plain ``SoC <= min_soc`` rule — exactly the pre-latch behaviour. Inside the
      band (min < SoC < min + margin) a restart therefore releases early once;
      the cost is at most one flip, never a discharge below min_soc.
    """

    def __init__(self) -> None:
        self.engaged = False

    @staticmethod
    def _number(value: object) -> float | None:
        """A finite real number, else None. The engine fills SystemState from HA
        reads (and tests from doubles), so the annotation is not a guarantee."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return None if math.isnan(value) or math.isinf(value) else float(value)

    def update(self, state: SystemState, release_margin: float) -> bool:
        soc_raw = self._number(state.current_soc_percent)
        floor = self._number(state.min_soc_percent)
        if state.soc_known is not True or soc_raw is None or floor is None:
            # Unknown / malformed reading: neither engage nor release.
            return self.engaged
        soc = round(soc_raw)
        if self.engaged:
            # "> floor" too, so margin 0 reproduces the stateless rule exactly
            # (held AT the floor, released one step above it).
            if soc >= floor + max(0.0, release_margin) and soc > floor:
                self.engaged = False
                logger.info(
                    "Min-SoC latch released: SoC %d%% >= min %.0f%% + %.0f",
                    soc, floor, release_margin,
                )
        elif soc <= floor:
            self.engaged = True
            logger.info(
                "Min-SoC latch engaged: SoC %d%% <= min %.0f%% (releases at %.0f%%)",
                soc, floor, floor + max(0.0, release_margin),
            )
        return self.engaged


@dataclass
class ControllerDecision:
    """The controller's decision on what actions to take.

    Uses 4 mode intents: charge, export, idle, self_consumption.
    The executor looks up the mode definition from the profile to get
    the ordered list of actions to execute.
    """

    # Mode intent - the primary field (required)
    # Values: "charge", "export", "idle", "self_consumption"
    mode_intent: str

    # Charge/discharge values (templates for executor to resolve)
    charge_value: float = 0.0
    discharge_value: float = 0.0

    # SoC target
    soc_target: int = 10

    # Water heater
    water_temp: int = 40  # Aggregate/fallback scalar (backward compat)
    water_temps: dict[str, int] = None  # type: ignore[assignment]  # Per-device: heater_id → temp

    def __post_init__(self) -> None:
        if self.water_temps is None:  # type: ignore[comparison-overlap]
            self.water_temps = {}

    export_power_w: float = 0.0  # Planned grid export power in Watts
    export_with_load_w: float = 0.0  # Export power + house load (for Fronius export mode)
    # Battery discharge power to COMMAND in export mode (W): the plan's discharge_kw,
    # rounded and clamped — not max_discharge. A profile that renders this into its
    # forced-discharge register sells what the plan priced, not everything the pack
    # can push. Override force_export keeps max (a manual export is "all of it").
    export_discharge_w: float = 0.0
    export_price_sek_kwh: float | None = None  # Effective export price; None = unknown (C3 stands down)

    # User's configured max limits (for templates like {{max_charge}})
    max_charge: float = 0.0
    max_discharge: float = 0.0

    # Flags
    write_charge_current: bool = False  # Only write if significant change
    write_discharge_current: bool = False
    control_unit: str = "A"  # "A" or "W"
    source: str = "plan"  # "plan" or "override"
    reason: str = ""
    # True only when THIS idle came from the Hold-slot (or PV-driven charge) branch and nothing else
    # (min-SoC floor, EV isolation) also wants it idle. The engine may then run the
    # tick as self_consumption while the fuse guard sees a phase at/over budget
    # (apply_hold_fuse_relief) — the battery relieving the fuse beats holding flat.
    hold_fuse_relief_ok: bool = False


def apply_hold_fuse_relief(decision: ControllerDecision) -> None:
    """Turn a relief-eligible Hold idle into self_consumption, in place.

    Mirrors what decide() renders for a self_consumption slot with no planned
    charge: charge limit = the configured max (PV may still charge the pack), the
    discharge limit is already the pack maximum. The fuse battery cap is applied
    after this by the engine, so a grid-charging overload is still capped.
    """
    if not decision.hold_fuse_relief_ok or decision.mode_intent != "idle":
        return
    decision.mode_intent = "self_consumption"
    if decision.charge_value <= 0:
        decision.charge_value = decision.max_charge
        decision.write_charge_current = True
    decision.reason = f"{decision.reason} | Hold-slot fuse relief: battery covers load".strip(
        " |"
    )


class Controller:
    """
    Determines actions based on slot plan and current state.

    Ported from n8n Helios Executor "Controller" JavaScript node.
    """

    def __init__(
        self,
        config: ControllerConfig,
        inverter_config: InverterConfig,
        water_heater_config: WaterHeaterGlobalConfig | None = None,
        water_heater_devices: list[WaterHeaterDeviceConfig] | None = None,
        profile: InverterProfile | None = None,
    ):
        self.config = config
        self.inverter_config = inverter_config
        self.water_heater_config = water_heater_config or WaterHeaterGlobalConfig()
        self.water_heater_devices = water_heater_devices or []
        self.profile = profile

    def decide(
        self,
        slot: SlotPlan,
        state: SystemState,
        override: OverrideResult | None = None,
    ) -> ControllerDecision:
        """
        Determine what actions to take based on slot plan and override.

        Args:
            slot: The current slot's planned values
            state: Current system state
            override: Override result if any override is active

        Returns:
            ControllerDecision with all action parameters
        """
        # If override is active, apply override actions
        if override and override.override_needed:
            return self._apply_override(slot, state, override)

        # Normal operation - follow the plan
        return self._follow_plan(slot, state)

    def _apply_override(
        self,
        slot: SlotPlan,
        state: SystemState,
        override: OverrideResult,
    ) -> ControllerDecision:
        """Apply override actions instead of plan using 4 mode intents."""
        actions = override.actions or {}

        # Determine mode intent from override type
        mode_intent = "idle"  # default

        if override.override_type.value == "force_charge":
            mode_intent = "charge"
        elif override.override_type.value == "force_export":
            # R1 SoC floor applies to MANUAL exports too (live-caught: a 15-min
            # force_export quick action ran 29 -> 17.8 %, straight through the 20 %
            # floor — only the timer stopped it; the plan path had the guard, this
            # path did not). A forced discharge would otherwise run to the quick
            # action's soc_target (default 10) at full power.
            _floor = effective_export_floor_pct(
                self.config.export_floor_soc_percent, state
            )
            if not state.soc_known:
                logger.warning(
                    "Force-export blocked: battery SoC unknown — downgrading "
                    "override to self_consumption"
                )
                mode_intent = "self_consumption"
            elif round(state.current_soc_percent) <= _floor:
                logger.warning(
                    "Force-export blocked: SoC %.1f%% at/below export floor %.0f%% "
                    "— downgrading override to self_consumption",
                    state.current_soc_percent,
                    _floor,
                )
                mode_intent = "self_consumption"
            else:
                mode_intent = "export"
        elif override.override_type.value == "plan_stale":
            # Plan too old to trust (opt-in cutoff): plain self-consumption — the
            # battery covers the house from PV and its own charge, nothing is
            # bought from or forced to the grid. Two holds, same as the plan path:
            # at the min-SoC floor (latched), and while a car draws power (the
            # battery must not feed the car; the slot carries ev_isolation).
            if self._at_min_soc_floor(state):
                logger.info(
                    "Plan-stale safe mode at the floor: SoC %.1f%% (min %.0f%%) - idle",
                    state.current_soc_percent, state.min_soc_percent,
                )
                mode_intent = "idle"
            elif slot.ev_isolation:
                mode_intent = "idle"
            else:
                mode_intent = "self_consumption"

        # For overrides, we typically don't actively charge/discharge
        # unless specifically requested
        charge_value = 0.0
        discharge_value = 0.0
        write_charge = False
        write_discharge = False

        # Handle quick action charging
        if override.override_type.value == "force_charge":
            # Force charge - use max charging value
            if self.inverter_config.control_unit == "W":
                charge_value = self.config.max_charge_w
            else:
                charge_value = self.config.max_charge_a
            write_charge = True

        # Handle quick action exporting (only when the floor guard above kept the
        # export intent — a floor-downgraded override must not force discharge).
        if override.override_type.value == "force_export" and mode_intent == "export":
            # Force export - allow max discharge
            if self.inverter_config.control_unit == "W":
                discharge_value = self.config.max_discharge_w
            else:
                discharge_value = self.config.max_discharge_a
            write_discharge = True

        # Get SoC target and water temp from override
        soc_target = int(actions.get("soc_target", 10))
        water_temp = int(actions.get("water_temp", 40))

        # User's configured max limits (for templates)
        unit = (
            self.profile.behavior.control_unit
            if self.profile
            else self.inverter_config.control_unit
        )
        max_charge = self.config.max_charge_w if unit == "W" else self.config.max_charge_a
        max_discharge = self.config.max_discharge_w if unit == "W" else self.config.max_discharge_a

        if override.override_type.value == "plan_stale" and mode_intent == "self_consumption":
            # As in _follow_plan: a self_consumption charge limit of 0 would stop
            # PV from charging the pack on profiles that write charge_value.
            charge_value = max_charge
            write_charge = True

        return ControllerDecision(
            mode_intent=mode_intent,
            charge_value=charge_value,
            discharge_value=discharge_value,
            export_power_w=0.0,
            export_with_load_w=0.0,
            export_discharge_w=(max_discharge if mode_intent == "export" else 0.0),
            soc_target=soc_target,
            water_temp=water_temp,
            max_charge=max_charge,
            max_discharge=max_discharge,
            write_charge_current=write_charge,
            write_discharge_current=write_discharge,
            control_unit=unit,
            source="override",
            reason=override.reason,
            # The slot's price is known even under an override — dropping it would
            # make C3 stand down (None = unknown) for every override tick.
            export_price_sek_kwh=slot.export_price_sek_kwh,
        )

    @staticmethod
    def _at_min_soc_floor(state: SystemState) -> bool:
        """Is the battery held at the min-SoC floor this tick?

        The engine's latch (state.min_soc_latched) decides when supplied; it
        carries the hysteresis and keeps its state through an unreadable SoC.
        Without it (direct callers) the old stateless rule applies: rounded SoC
        at or below min_soc, and an unknown SoC is never "at the floor".
        """
        latched = state.min_soc_latched
        if latched is not None:
            return bool(latched)
        return state.soc_known and (round(state.current_soc_percent) <= state.min_soc_percent)

    def _follow_plan(self, slot: SlotPlan, state: SystemState) -> ControllerDecision:
        """Follow the slot plan for normal operation using 4 mode intents."""
        # Determine mode intent based on slot plan
        # Order matters: export > charge > idle > self_consumption
        #
        # PV surplus vs battery export distinction:
        # - Battery export: discharge_kw > 0 (battery actively discharging to grid)
        # - PV surplus: discharge_kw == 0, charge_kw > 0 (excess PV exports while charging)
        # Battery->grid is the SMALLER of the two flows: export_kw counts PV surplus
        # too, and discharge_kw counts what the battery gives the house. Only their
        # overlap is battery energy sold. Export mode used to fire on `both > 0` and
        # then command max_discharge regardless — a 0.3 kW planned discharge in a
        # sunny slot became a 9.5 kW dump to the SoC floor (2026-09-12, seven times).
        battery_to_grid_kw = min(slot.export_kw, slot.discharge_kw)
        # Floor for the branch below that chooses self_consumption on the strength
        # of a PV FORECAST (the PV-driven charge; the PV-surplus hold that used to
        # share it now always idles — see the hold branch).
        # Both assume the roof covers the house; when it does not, self_consumption
        # serves the house from the battery and nothing in either branch stops it —
        # the PV-charge branch never looks at SoC, and the hold branch is only
        # reached once SoC is already at or below the slot target. Live 2026-10-05
        # 10:41-12:00: Hold slots at target 10 %, forecast PV 2.5-3.1 kW over a
        # 1.2-1.6 kW load, and the pack went 7.7 -> 2.2 -> 0.0 % while every replan
        # answered with a 1-3 minute grid-charge burst of up to 7 kW that the next
        # slot spent again (2026-10-03 and -04 the same, in PV-charge slots too).
        # At or below min_soc these branches hold instead: idle blocks discharge
        # only, so real PV surplus still charges the pack, and one sensor step above
        # the floor the ordinary rules apply again. An unknown SoC changes nothing.
        # With the engine's latch (MinSocFloorLatch) "one sensor step above" becomes
        # min_soc + min_soc_release_margin_percent: without hysteresis the mode
        # flapped idle <-> self_consumption on every 1 % step around the floor.
        at_floor = self._at_min_soc_floor(state)
        hold_branch = False
        pv_charge_branch = False
        if (
            battery_to_grid_kw > HOLD_DISCHARGE_EPS_KW
            and battery_to_grid_kw >= self.config.export_min_battery_kw
        ):
            # Battery discharge to grid - use export mode.
            # R1 runtime SoC-floor guard (arbitrage gate): the export floor exists
            # only inside the MILP — between replans the plan is stale and the real
            # SoC can drift below it (or a bad plan could slip through). Sungrow
            # Forced discharge would otherwise run to BMS cutoff at full power.
            # At/below the floor, downgrade to self_consumption (battery serves the
            # house only, never the grid).
            _floor = effective_export_floor_pct(
                self.config.export_floor_soc_percent, state
            )
            _price_floor = self.config.min_export_price_sek_kwh
            if not state.soc_known:
                # The floor cannot be checked against a number we do not have.
                logger.warning(
                    "Export intent blocked: battery SoC unknown — downgrading to "
                    "self_consumption"
                )
                mode_intent = "self_consumption"
            elif round(state.current_soc_percent) <= _floor:
                logger.warning(
                    "Export intent blocked: SoC %.1f%% at/below export floor %.0f%% "
                    "— downgrading to self_consumption",
                    state.current_soc_percent,
                    _floor,
                )
                mode_intent = "self_consumption"
            elif (
                _price_floor is not None
                and slot.export_price_sek_kwh is not None
                and slot.export_price_sek_kwh < _price_floor
            ):
                # Runtime price sanity floor. The MILP priced the sale; this is the
                # backstop for the plan it should never have produced. Unpriced
                # slots pass — a missing price is not evidence of a bad one.
                logger.warning(
                    "Export intent blocked: export price %.2f SEK/kWh below floor "
                    "%.2f — downgrading to self_consumption (planned battery->grid "
                    "%.2f kW)",
                    slot.export_price_sek_kwh, _price_floor, battery_to_grid_kw,
                )
                mode_intent = "self_consumption"
            else:
                mode_intent = "export"
        elif slot.export_kw > 0 and slot.discharge_kw > HOLD_DISCHARGE_EPS_KW:
            # Both flows planned but the overlap is below export_min_battery_kw: the
            # battery is serving the house while PV surplus exports on its own.
            # self_consumption does exactly that. NOT idle — idle blocks discharge and
            # would starve the load the plan meant the battery to cover.
            # Discharge must clear HOLD_DISCHARGE_EPS_KW like the Hold test: the
            # schedule formatter rounds to 2 decimals, so 0.005-0.01 kW of solver
            # dust reads 0.01 and, in a sunny export slot, used to land here and
            # reopen the Hold-slot ratchet.
            logger.info(
                "Export intent skipped: planned battery->grid %.2f kW below "
                "export_min_battery_kw %.2f — self_consumption",
                battery_to_grid_kw, self.config.export_min_battery_kw,
            )
            mode_intent = "self_consumption"
        elif (
            slot.charge_kw > 0
            and slot.export_kw == 0
            and slot.pv_kw < slot.charge_kw + slot.load_kw
        ):
            # Genuine grid charge: forecast PV cannot cover the planned charge plus
            # the house load, so the shortfall is imported from the grid. Force
            # grid-charge mode (Forced mode → Forced charge from AC).
            mode_intent = "charge"
        elif slot.charge_kw > 0:
            # PV-driven charge (discharge_kw == 0). Two cases land here:
            #   (a) PV surplus with export headroom (export_kw > 0), or
            #   (b) PV covers charge + load but export_kw == 0 — e.g. a near-full
            #       battery topping off the last few %, or export curtailed/limited.
            # In BOTH cases no grid import is needed to charge, so do NOT force
            # grid-charge: that sets max_charge to the (tiny) charge value and, via
            # Forced charge, blocks the battery from discharging — leaving a full
            # battery idle while single-phase/transient loads the balanced PV feed
            # can't cover are pulled from the grid. Use self_consumption so the
            # battery covers those deficits and refills from the surplus PV.
            #
            # Idle, not self_consumption (unless pv_charge_allow_discharge): the
            # plan has discharge_kw == 0 here, so any discharge is energy it never
            # priced. self_consumption commands max_discharge, and when real PV
            # comes in under the forecast (over-forecast, overcast) the pack feeds
            # the whole house instead of charging — the shortfall the plan meant
            # the grid to cover. Live 2026-10-25..11-02: ~8.1 kWh drained that way
            # in PV-charge slots (2026-11-01 overcast the worst). Same reasoning as
            # the Hold-slot fix (2a90a35c): idle on the Sungrow profile caps
            # discharge only (max_charge stays at the configured maximum), so real
            # PV surplus still charges the pack and PV surplus beyond it still
            # exports; PV < load -> battery rests, grid covers the rest. The
            # transient/phase argument above costs nothing on a phase-netting
            # (Swedish) meter. A phase at/over the fuse budget may still release
            # it for a tick (hold_fuse_relief_ok, engine._apply_hold_fuse_relief).
            # At the floor it idles regardless (see at_floor above).
            if at_floor:
                logger.info(
                    "PV-driven charge at the floor: SoC %.1f%% <= min_soc %.0f%% - idle "
                    "(charges from real PV surplus, does not discharge)",
                    state.current_soc_percent, state.min_soc_percent,
                )
                mode_intent = "idle"
            elif self.config.pv_charge_allow_discharge:
                mode_intent = "self_consumption"
            else:
                mode_intent = "idle"
                pv_charge_branch = True
        elif slot.discharge_kw <= HOLD_DISCHARGE_EPS_KW:
            # Hold slot: the plan models no battery discharge (the planner's own
            # "Hold" test is charge and discharge both <= 0.01 kW), so block
            # discharge whatever the SoC. Idle still lets real PV surplus charge
            # the pack (Sungrow: self-consumption EMS with a 10 W discharge cap).
            #
            # This used to self-consume in two cases: SoC above the target, and
            # at/below it whenever FORECAST PV > load (d9dc45a9, for transients
            # the PV feed cannot reach). Both let the pack discharge in a slot the
            # plan holds flat, and the planner sets a Hold slot's target to the
            # SoC it entered with — so the next replan starts lower, sets a lower
            # target, and the slot drains again: a ratchet to the min-SoC floor.
            # Live 2026-10-25..11-01: 43.6 kWh drained in Hold slots executed as
            # self_consumption (34.7 via the PV>load exception under over-forecast
            # PV, 8.9 above target); 2026-11-01 08:15-13:00 the targets stepped
            # 70 -> 10 slot by slot, then the planner grid-charged again at 14:00.
            # Same fix as upstream 1dfd8c85. EV-charging hold slots land here too
            # (REV F76 Phase 3: no battery -> EV).
            mode_intent = "idle"
            hold_branch = True
        elif round(state.current_soc_percent) <= slot.soc_target:
            # Discharge slot at its floor - hold the battery.
            # Round current SoC to integer for consistent comparison with plan target.
            mode_intent = "idle"
        else:
            # Above SoC target - use self_consumption
            mode_intent = "self_consumption"

        if at_floor and mode_intent in ("self_consumption", "export"):
            # Shared min-SoC floor guard (upstream de8b93ee D1). The branches above
            # only consult at_floor where SoC <= the slot target; a slot whose
            # target sits below the current SoC landed in "above target ->
            # self_consumption" — so the latch's band (min < SoC < min + margin)
            # was never held there and the mode flapped one sensor step above the
            # floor. Idle blocks discharge only: PV surplus still charges the pack,
            # and a planned grid charge (mode "charge") is untouched.
            logger.info(
                "Min-SoC floor hold: SoC %.1f%% (min %.0f%%, latched) - %s -> idle",
                state.current_soc_percent, state.min_soc_percent, mode_intent,
            )
            mode_intent = "idle"

        if slot.ev_isolation and mode_intent in ("self_consumption", "export"):
            # A car is drawing power right now — planned or not. discharge_kw=0
            # alone changes nothing here: _calculate_discharge_limit always commands
            # the pack maximum, so in self_consumption the inverter feeds the car
            # from the battery at up to max_discharge_w. 2026-09-25 05:02-05:15 the
            # Tesla started outside the plan and took ~2 kWh that way until the SoC
            # floor forced idle; the night before, the same. Only the mode blocks
            # the flow. A grid charge is left alone: Forced charge cannot discharge.
            logger.info(
                "EV source isolation: %s -> idle (car drawing power, no EV slot in plan)",
                mode_intent,
            )
            mode_intent = "idle"

        # Calculate charge/discharge values
        charge_value, write_charge = self._calculate_charge_limit(slot, state)
        discharge_value, write_discharge = self._calculate_discharge_limit(slot, state)

        # Planned grid export power (kW to W)
        export_power_w = slot.export_kw * 1000.0

        # Calculate export_with_load_w for Fronius export mode
        # This adds house load to export power so battery delivers both
        if slot.export_kw > 0:
            step = self.profile.behavior.round_step_w if self.profile else 100.0
            raw_export_with_load = export_power_w + (slot.load_kw * 1000.0)
            export_with_load_w = round(raw_export_with_load / step) * step
            # The battery cannot discharge past the ceiling we ourselves command
            # (_calculate_discharge_limit writes exactly this number), so asking
            # for export+load above it is incoherent: we would be telling the
            # inverter "at most X" and "give me more than X" in the same tick.
            # It also fails LOUDLY in the wrong way — a value outside an HA
            # number helper's range makes HA reject the whole write, so the
            # export command silently does nothing rather than doing less.
            # Clamped AFTER rounding on purpose: rounding a clamped value to the
            # nearest step could land back above the limit.
            export_with_load_w = min(export_with_load_w, self.config.max_discharge_w)
        else:
            export_with_load_w = 0.0

        # Battery discharge to COMMAND in export mode: the plan's rate, rounded to
        # the profile step and clamped to the pack's max. Zero outside export mode so
        # a stale value can never leak into a self_consumption write.
        export_discharge_w = 0.0
        if mode_intent == "export":
            step = self.profile.behavior.round_step_w if self.profile else self.config.round_step_w
            raw_discharge_w = slot.discharge_kw * 1000.0
            export_discharge_w = min(
                self.config.max_discharge_w, max(step, round(raw_discharge_w / step) * step)
            )

        # SoC target from plan
        soc_target = slot.soc_target

        # Water heater from plan (scalar + per-device)
        water_temp = self._determine_water_temp(slot)
        water_temps = self._determine_water_temps(slot)

        # User's configured max limits (for templates)
        unit = (
            self.profile.behavior.control_unit
            if self.profile
            else self.inverter_config.control_unit
        )
        max_charge = self.config.max_charge_w if unit == "W" else self.config.max_charge_a
        max_discharge = self.config.max_discharge_w if unit == "W" else self.config.max_discharge_a

        # Override for self_consumption default fallback
        # When no charge is planned (charge_value <= 0) and we fall back to self_consumption,
        # we should use max charge instead of 0 to allow PV to charge the battery.
        if mode_intent == "self_consumption" and charge_value <= 0:
            charge_value = max_charge
            write_charge = True

        reason = self._generate_reason(slot, mode_intent)

        return ControllerDecision(
            mode_intent=mode_intent,
            charge_value=charge_value,
            discharge_value=discharge_value,
            export_power_w=export_power_w,
            export_with_load_w=export_with_load_w,
            export_discharge_w=export_discharge_w,
            export_price_sek_kwh=slot.export_price_sek_kwh,
            soc_target=soc_target,
            water_temp=water_temp,
            water_temps=water_temps,
            max_charge=max_charge,
            max_discharge=max_discharge,
            write_charge_current=write_charge,
            write_discharge_current=write_discharge,
            control_unit=unit,
            source="plan",
            reason=reason,
            hold_fuse_relief_ok=(
                (hold_branch or pv_charge_branch)
                and mode_intent == "idle"
                and not at_floor
                and not slot.ev_isolation
                and slot.ev_charging_kw <= 0.1
            ),
        )

    def _calculate_charge_limit(self, slot: SlotPlan, state: SystemState) -> tuple[float, bool]:
        """
        Calculate the charge limit to command (Amps or Watts).
        """
        unit = self.inverter_config.control_unit
        if self.profile and self.profile.behavior.control_unit:
            unit = self.profile.behavior.control_unit

        if slot.charge_kw <= 0:
            logger.debug("No charge planned, returning 0")
            return 0.0, False

        if unit == "W":
            # Watts Logic
            raw_val = slot.charge_kw * 1000.0

            # Round to step
            is_grid_charge = slot.charge_kw > 0
            step = self.profile.behavior.round_step_w if self.profile else self.config.round_step_w

            # Use grid-specific rounding if available (Rev IP1 Phase 4)
            if is_grid_charge and self.profile and self.profile.behavior.grid_charge_round_step_w:
                step = self.profile.behavior.grid_charge_round_step_w
                logger.debug("Using grid-specific rounding step: %.1fW", step)

            rounded = round(raw_val / step) * step

            # Clamp
            min_w = self.profile.behavior.min_charge_w if self.profile else self.config.min_charge_w
            max_w = self.config.max_charge_w
            clamped = max(min_w, min(max_w, rounded))

            # Decide if we should write
            should_write = clamped >= min_w

            return clamped, should_write

        else:
            # Amps Logic (Default)
            # kW to Amps: I = P * 1000 / V
            raw_current = (slot.charge_kw * 1000) / self.config.min_voltage_v

            # Round to step
            round_step_a = (
                self.profile.behavior.round_step_a if self.profile else self.config.round_step_a
            )
            rounded = round(raw_current / round_step_a) * round_step_a

            # Clamp to limits
            min_a = self.profile.behavior.min_charge_a if self.profile else self.config.min_charge_a
            max_a = self.config.max_charge_a
            clamped = max(min_a, min(max_a, rounded))

            # Decide if we should write
            should_write = clamped >= min_a

            return clamped, should_write

    def _calculate_discharge_limit(self, slot: SlotPlan, state: SystemState) -> tuple[float, bool]:
        """
        Calculate the discharge limit to command.
        ALWAYS return MAX to allow load coverage.
        """
        if self.inverter_config.control_unit == "W":
            return self.config.max_discharge_w, True
        else:
            return self.config.max_discharge_a, True

    def _determine_water_temp(self, slot: SlotPlan) -> int:
        """Determine aggregate water heater target temperature from slot plan (backward compat)."""
        if slot.water_kw > 0:
            return self.water_heater_config.temp_normal
        return self.water_heater_config.temp_off

    def _device_temp(self, device: WaterHeaterDeviceConfig, kind: str) -> int:
        """One temperature for one device: per-device override, else the global.

        Thermostatic loads don't share a temperature range — the spa runs 20-40 C,
        so the tanks' global temp_off=40 would read as "heat to 40" on the spa's
        bridge and temp_normal=60 exceeds its max. A device may override any of
        temp_off / temp_normal / temp_boost / temp_max; unset falls back.
        """
        override = getattr(device, f"temp_{kind}", None)
        if override is not None:
            return int(override)
        return int(getattr(self.water_heater_config, f"temp_{kind}"))

    def _determine_water_temps(self, slot: SlotPlan) -> dict[str, int]:
        """Determine per-device water heater temperatures from slot plan (task 6.4)."""
        if not self.water_heater_devices:
            return {}
        temps: dict[str, int] = {}
        for device in self.water_heater_devices:
            planned_kw = slot.water_heater_plans.get(device.id, 0.0)
            is_boost = slot.water_heating_boost.get(device.id, False)
            if is_boost and planned_kw > 0:
                # Excess PV boost uses temp_max (85°C, the PV dump target).
                # temp_boost (70°C) is reserved for the manual dashboard boost button.
                temps[device.id] = self._device_temp(device, "max")
            elif planned_kw > 0:
                temps[device.id] = self._device_temp(device, "normal")
            else:
                temps[device.id] = self._device_temp(device, "off")
        return temps

    def _generate_reason(self, slot: SlotPlan, mode_intent: str) -> str:
        """Generate a human-readable reason for the decision."""
        parts: list[str] = []

        if slot.charge_kw > 0:
            parts.append(f"Charge {slot.charge_kw:.1f}kW")
        if slot.export_kw > 0:
            parts.append(f"Export {slot.export_kw:.1f}kW")
        if slot.water_kw > 0:
            parts.append(f"Water {slot.water_kw:.1f}kW")
        if slot.ev_isolation:
            parts.append("EV isolation")

        if not parts:
            parts.append("Hold/Idle")

        # Use mode intent directly for reason string
        mode_str = mode_intent.replace("_", " ").title()

        return f"Plan: {', '.join(parts)} | {mode_str} | SoC→{slot.soc_target}%"


def make_decision(
    slot: SlotPlan,
    state: SystemState,
    override: OverrideResult | None = None,
    config: ControllerConfig | None = None,
    inverter_config: InverterConfig | None = None,
    water_heater_config: WaterHeaterGlobalConfig | None = None,
    water_heater_devices: list[WaterHeaterDeviceConfig] | None = None,
    profile: InverterProfile | None = None,
) -> ControllerDecision:
    """
    Convenience function to make a controller decision.

    Returns:
        ControllerDecision with all action parameters
    """
    controller = Controller(
        config or ControllerConfig(),
        inverter_config or InverterConfig(),
        water_heater_config,
        water_heater_devices,
        profile,
    )
    return controller.decide(slot, state, override)
