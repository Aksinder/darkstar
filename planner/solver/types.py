"""
Kepler Solver Types

Type definitions for the Kepler MILP solver input/output.
Migrated from backend/kepler/types.py for the new planner package.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IncentiveBucket:
    """EV incentive bucket based on SoC threshold."""

    threshold_soc: float
    value_sek: float


@dataclass
class WaterHeaterInput:
    """Per-device water heater input for the Kepler MILP solver."""

    id: str
    power_kw: float
    min_kwh_per_day: float
    max_hours_between_heating: float
    min_spacing_hours: float
    force_on_slots: list[int] | None = None
    heated_today_kwh: float = 0.0


@dataclass
class EVChargerInput:
    """Per-device EV charger input for the Kepler MILP solver."""

    id: str
    max_power_kw: float
    battery_capacity_kwh: float
    current_soc_percent: float
    plugged_in: bool
    deadline: datetime | None
    incentive_buckets: list[IncentiveBucket] = field(default_factory=lambda: [])


@dataclass
class DeferrableLoadInput:
    """Per-device deferrable household load (dishwasher, washing machine, ...).

    Represents a single *pending* run: a non-interruptible, contiguous block of
    ``duration_slots`` slots that must run once within the planning horizon,
    consuming ``energy_kwh`` total, ideally finishing by ``deadline_slot``.
    Only loads with a pending run should appear in the solver input.
    """

    id: str
    energy_kwh: float
    duration_slots: int
    earliest_start_slot: int = 0
    # Finish-by slot index (inclusive). None = end of horizon.
    deadline_slot: int | None = None
    # True => deadline strongly enforced (high tardiness penalty); False => soft
    # "cheapest within window" (moderate penalty, economics may run late).
    deadline_hard: bool = True
    # Optional grid phase ("A"/"B"/"C") for phase-balancing (see phase penalty).
    phase: str | None = None


@dataclass
class LoadPriority:
    """Resolved per-load willingness-to-pay (WTP) parameters.

    Already merged from tier defaults + intra-tier rank + per-load overrides by the
    adapter, so the solver consumes pure numbers. The WTP is a reservation price
    (SEK/kWh): the load is worth running in a slot when its WTP for that slot meets
    or exceeds the slot's marginal energy price. ``base_wtp`` is the floor value at
    zero urgency; ``urgency_wtp`` is the additional WTP fully ramped in by the
    deadline/comfort gap (linear ramp), so a load tolerates progressively more
    expensive energy as its window closes. ``rank_epsilon`` is a tiny signed
    tiebreak (lower rank => slightly higher WTP) that breaks intra-tier ties
    deterministically without perturbing real economics.
    """

    tier_rank: int = 0
    base_wtp_sek_per_kwh: float = 0.0
    urgency_wtp_sek_per_kwh: float = 0.0
    rank_epsilon_sek_per_kwh: float = 0.0


@dataclass
class KeplerConfig:
    """Configuration for the Kepler MILP solver."""

    capacity_kwh: float
    min_soc_percent: float
    max_soc_percent: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    wear_cost_sek_per_kwh: float
    # Optional export limits (if any)
    max_export_power_kw: float | None = None
    max_import_power_kw: float | None = None
    max_inverter_ac_kw: float | None = (
        None  # Inverter AC output limit (PV + battery discharge combined)
    )
    target_soc_kwh: float | None = None  # Minimum SoC at end of horizon
    target_soc_penalty_sek: float = 0.0  # Set by pipeline (Safety Floor penalty)
    curtailment_penalty_sek: float = 0.0  # Penalty for wasting available solar power
    ramping_cost_sek_per_kw: float = 0.0  # Penalty for power changes
    export_threshold_sek_per_kwh: float = 0.0  # Min spread to export
    grid_import_limit_kw: float | None = None  # Soft constraint
    # Per-device water heater inputs (replaces scalar water fields)
    water_heaters: list[WaterHeaterInput] = field(default_factory=lambda: [])

    # Global water heating settings (apply to all heaters)
    water_heating_max_gap_hours: float = 0.0  # Threshold for gap penalty (0 = disabled)
    water_comfort_penalty_sek: float = 0.50  # Penalty per hour beyond gap threshold (deprecated)
    water_block_penalty_sek: float = 0.0  # Penalty per slot for overshooting block window
    water_reliability_penalty_sek: float = 0.0  # Penalty per day for missing daily minimum
    max_block_hours: float = 2.0  # Rev K24: Dynamic window size per comfort level (global)
    water_spacing_penalty_sek: float = (
        0.20  # DEPRECATED (PERF1): No longer used. Spacing is now a hard constraint per device.
    )
    water_block_start_penalty_sek: float = 0.0  # Penalty per block start (global)
    defer_up_to_hours: float = 0.0  # Allow heating until N hours into next day (global)

    # Rev E4: Export Toggle
    enable_export: bool = True  # If False, enforce 0 export

    # Export SoC Floor: minimum SoC required to allow grid export
    export_floor_soc_percent: float | None = None

    # EV Charging as deferrable load (per-device, multi-charger support)
    ev_chargers: list[EVChargerInput] = field(
        default_factory=lambda: []
    )  # Per-device EV charger inputs

    # Excess PV dispatch
    excess_pv_slots: list[bool] = field(
        default_factory=lambda: []
    )  # Per-slot flags: True if excess PV available
    excess_pv_sink: str = "disabled"  # water_heater_boost | custom_entity | disabled
    excess_pv_reward_sek_per_kwh: float = 0.5  # Reward for using excess PV at sink vs exporting
    excess_pv_soc_threshold_percent: float = 95.0  # Battery SoC % required before sink activates
    excess_pv_custom_entity_power_kw: float = 1.0  # Estimated power of custom entity (kW)
    # Independent opt-in for the custom-entity sink. When True it activates regardless
    # of excess_pv_sink, so it can coexist with the water_heater_boost sink (one `sink`
    # string can't select both). sink == "custom_entity" still implies this.
    excess_pv_custom_entity_enabled: bool = False
    # Optional export-price ceiling (SEK/kWh) for the custom-entity sink. When set, the
    # sink may only activate in slots where export_price <= ceiling — i.e. soak surplus
    # locally only when grid export pays little or nothing (incl. negative prices), and
    # sell it otherwise. None => no price gate (legacy behaviour). This is what makes the
    # villavagn-AC cooling sink fire on "low or minus price" rather than on any surplus.
    excess_pv_price_ceiling_sek_per_kwh: float | None = None

    # Deferrable household loads (dishwasher, washing machine, ...)
    deferrable_loads: list[DeferrableLoadInput] = field(default_factory=lambda: [])
    # Tardiness penalty (SEK per slot finished after the deadline).
    deferrable_soft_deadline_penalty_sek: float = 30.0  # for soft "cheapest within X h"
    deferrable_hard_deadline_penalty_sek: float = 1000.0  # for hard "done by HH:MM"
    # Soft penalty (SEK per slot) for two deferrable loads running on the same
    # phase at the same time (0 = phase-balancing disabled).
    deferrable_phase_penalty_sek: float = 0.0

    # ---- Load priority / willingness-to-pay (WTP) layer (flag-gated, default OFF) ----
    # A unified tier+rank+time->WTP reservation-price model. When enabled, a load
    # that has a LoadPriority entry is run by the planner only while its WTP for the
    # slot meets the marginal energy price (cheap/surplus) — low-priority loads
    # (e.g. spa) defer or skip under scarcity, high-priority loads keep running, and
    # a linear urgency ramp pulls a load in before its deadline. When disabled (or
    # for any load without an entry) behaviour is byte-identical to before.
    load_priority_enabled: bool = False
    # Map of load id -> resolved LoadPriority. Empty => no WTP applied to any load.
    load_priorities: dict[str, LoadPriority] = field(default_factory=lambda: {})

    # ---- Phase-aware imbalance cost (flag-gated, default OFF) ----
    # The single-net-node LP nets a heavy phase's import against the light phases'
    # export to ~zero, hiding the real cost (buy high on the heavy phase, sell low on
    # the others). When enabled, the solver prices that hidden EXTRA into the objective
    # and may discharge the battery to raise balanced supply and cover the heavy phase —
    # but only WHEN ECONOMIC (the import avoided must beat the export spilled on the
    # light phases plus the battery value spent). phase_load_fractions is the static
    # per-phase split; phase_load_profile is the per-hour {hour: {A,B,C}} forecast used
    # when present so the cost reflects which phase is heavy at each slot's hour.
    phase_aware_enabled: bool = False
    phase_aware_weight: float = 1.0  # scales the imbalance EXTRA (1.0 = full economic cost)
    phase_load_fractions: dict[str, float] | None = None
    phase_load_profile: dict[int, dict[str, float]] | None = None

    # Improvement B (Predbat-inspired): continuous stored-energy value (SEK/kWh).
    # Rewards energy left in the battery at the END of the horizon at its expected
    # forward worth, applied to soc[T] ONLY. This is a terminal credit (symmetric:
    # no discharge cost without an offsetting charge credit), so unlike the removed
    # K20 term it cannot distort mid-horizon cycling. It softens the hard terminal
    # SoC floor: the floor stays as a low hard-safety reserve while this term makes
    # the economic "how much extra to keep" decision continuously. 0 = disabled.
    battery_value_sek_per_kwh: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Rev F39: Validate battery configuration
        if self.capacity_kwh > 0:
            if self.max_charge_power_kw <= 0:
                raise ValueError(
                    f"Battery capacity is {self.capacity_kwh} kWh but max_charge_power_kw is {self.max_charge_power_kw}. Battery cannot charge!"
                )
            if self.max_discharge_power_kw <= 0:
                raise ValueError(
                    f"Battery capacity is {self.capacity_kwh} kWh but max_discharge_power_kw is {self.max_discharge_power_kw}. Battery cannot discharge!"
                )

        # Log actual values for debugging
        import logging

        logger = logging.getLogger("darkstar.kepler.config")
        logger.info(
            f"Kepler Config: capacity={self.capacity_kwh}kWh, charge={self.max_charge_power_kw}kW, discharge={self.max_discharge_power_kw}kW"
        )


@dataclass
class KeplerInputSlot:
    """Input data for a single time slot."""

    start_time: datetime
    end_time: datetime
    load_kwh: float
    pv_kwh: float
    import_price_sek_kwh: float
    export_price_sek_kwh: float


@dataclass
class KeplerInput:
    """Complete input for a solver run."""

    slots: list[KeplerInputSlot]
    initial_soc_kwh: float


@dataclass
class KeplerResultSlot:
    """Solver output for a single time slot."""

    start_time: datetime
    end_time: datetime
    charge_kwh: float
    discharge_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    soc_kwh: float
    cost_sek: float
    import_price_sek_kwh: float = 0.0
    export_price_sek_kwh: float = 0.0
    water_heat_kw: float = 0.0  # Aggregate water heating power (backward compat)
    water_heater_results: dict[str, float] = field(
        default_factory=lambda: {}
    )  # Per-device: heater_id -> kW
    ev_charge_kw: float = 0.0  # Aggregate EV charging power in this slot (backward compat)
    ev_charger_results: dict[str, float] = field(
        default_factory=lambda: {}
    )  # Per-device: charger_id -> kW
    water_heating_boost: dict[str, bool] = field(
        default_factory=lambda: {}
    )  # Per-device: heater_id -> boost active
    custom_entity_active: bool = False  # Whether custom entity sink should be on
    deferrable_load_kw: float = 0.0  # Aggregate deferrable-load power this slot
    deferrable_load_results: dict[str, float] = field(
        default_factory=lambda: {}
    )  # Per-device: load_id -> kW
    is_optimal: bool = True


@dataclass
class KeplerResult:
    """Complete solver output."""

    slots: list[KeplerResultSlot]
    total_cost_sek: float
    is_optimal: bool
    status_msg: str
    solve_time_ms: float = 0.0
