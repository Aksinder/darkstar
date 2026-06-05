"""
Simulation Module

Functions for simulating battery state and costs based on a schedule.

``simulate_schedule`` projects an idealised SoC. ``simulate_realistic`` (Predbat-
inspired) replays the schedule against the *real* executor/inverter behaviour the
single-net-node MILP hides — chiefly that a 3-phase inverter spreads its support
evenly, so an unbalanced (single-phase) load draws grid on the heavy phase even
when the net balances and the battery is full. It reports the cost the LP missed
(``realism_gap_sek``) and flags exposed slots, for observability only.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def simulate_schedule(
    df: pd.DataFrame, config: dict[str, Any], initial_state: dict[str, Any]
) -> pd.DataFrame:
    """
    Simulate a schedule with given battery actions and return the projected results.

    Args:
        df: DataFrame with charge_kw, discharge_kw, etc. set
        config: Configuration dictionary
        initial_state: Initial battery state

    Returns:
        DataFrame with simulated projections (projected_soc_percent, etc.)
    """
    df = df.copy()

    battery_config = config.get("battery", {})
    capacity_kwh = float(battery_config.get("capacity_kwh", 0.0))
    float(battery_config.get("min_soc_percent", 10.0))
    float(battery_config.get("max_soc_percent", 100.0))

    # Efficiency
    roundtrip = float(battery_config.get("roundtrip_efficiency_percent", 95.0))
    eff_one_way = (roundtrip / 100.0) ** 0.5

    # Initial SoC
    current_soc_kwh = float(initial_state.get("battery_soc_kwh", 0.0))
    if "battery_soc_percent" in initial_state and "battery_soc_kwh" not in initial_state:
        current_soc_kwh = (float(initial_state["battery_soc_percent"]) / 100.0) * capacity_kwh

    # Iterate and update state
    projected_soc_kwh: list[float] = []
    projected_soc_pct: list[float] = []

    for idx, row in df.iterrows():
        # Determine slot duration
        duration_h = (row["end_time"] - idx).total_seconds() / 3600.0 if "end_time" in row else 0.25
        # Default 15 min

        charge_kw = float(row.get("charge_kw", 0.0))
        discharge_kw = float(row.get("discharge_kw", 0.0))

        # Apply efficiency
        energy_in = charge_kw * duration_h * eff_one_way
        energy_out = discharge_kw * duration_h / eff_one_way

        current_soc_kwh += energy_in - energy_out

        # Clamp (though simulation should probably show if it violates?)
        # Legacy planner clamps in _pass_6.
        current_soc_kwh = max(0.0, min(current_soc_kwh, capacity_kwh))

        pct = (current_soc_kwh / capacity_kwh * 100.0) if capacity_kwh > 0 else 0.0

        projected_soc_kwh.append(current_soc_kwh)
        projected_soc_pct.append(pct)

    df["projected_soc_kwh"] = projected_soc_kwh
    df["projected_soc_percent"] = projected_soc_pct

    # Calculate costs/revenues if prices available
    if "import_price_sek_kwh" in df.columns:
        df["import_cost"] = df.get("import_kwh", 0.0) * df["import_price_sek_kwh"]
    if "export_price_sek_kwh" in df.columns:
        df["export_revenue"] = df.get("export_kwh", 0.0) * df["export_price_sek_kwh"]

    return df


# ---------------------------------------------------------------------------
# Realism simulation (Predbat-inspired): surface costs the net-node MILP hides
# ---------------------------------------------------------------------------


@dataclass
class RealismSlot:
    """One scheduled slot's flows + state, for the realism replay (all kWh)."""

    pv_kwh: float
    load_kwh: float
    discharge_kwh: float
    grid_import_kwh: float  # planned (from the MILP)
    grid_export_kwh: float  # planned
    import_price_sek_kwh: float
    export_price_sek_kwh: float
    soc_percent: float = 0.0
    soc_target_percent: float = 0.0


@dataclass
class RealismResult:
    """Outcome of the realism replay."""

    predicted_cost_sek: float
    simulated_cost_sek: float
    realism_gap_sek: float
    extra_import_kwh: float
    phase_flagged_slots: list[int] = field(default_factory=lambda: [])
    idle_exposed_slots: list[int] = field(default_factory=lambda: [])


def _normalise_phase_fractions(phase_fractions: dict[str, float] | None) -> list[float] | None:
    """Return load fractions summing to 1, or None when balanced/unset."""
    if not phase_fractions:
        return None
    vals = [max(0.0, float(v)) for v in phase_fractions.values()]
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


def simulate_realistic(
    slots: list[RealismSlot],
    *,
    phase_fractions: dict[str, float] | None = None,
) -> RealismResult:
    """Replay a schedule against real inverter/phase behaviour.

    Phase model: the inverter delivers ``pv + discharge`` balanced across phases,
    while the house load is split by ``phase_fractions`` (e.g. {"A": 0.5, ...}).
    A heavy phase then imports while light phases export, so the grid cost exceeds
    the net-node LP's view by ``realism_gap_sek``. With balanced or no fractions
    the gap is 0 (the LP is already correct).

    Also flags Hold/idle slots during PV surplus, where the executor freezes
    discharge and so cannot absorb a load spike (robustness exposure, not a cost).
    """
    fracs = _normalise_phase_fractions(phase_fractions)
    n_phases = len(fracs) if fracs else 0

    predicted = 0.0
    simulated = 0.0
    extra_import = 0.0
    phase_flagged: list[int] = []
    idle_exposed: list[int] = []

    for i, s in enumerate(slots):
        imp_p = s.import_price_sek_kwh
        exp_p = s.export_price_sek_kwh
        predicted_slot = s.grid_import_kwh * imp_p - s.grid_export_kwh * exp_p
        predicted += predicted_slot
        simulated_slot = predicted_slot

        # Phase-imbalance cost (only when fractions are provided).
        if fracs:
            supply = s.pv_kwh + s.discharge_kwh  # delivered balanced across phases
            supply_per_phase = supply / n_phases
            phase_import = sum(max(0.0, s.load_kwh * f - supply_per_phase) for f in fracs)
            phase_export = sum(max(0.0, supply_per_phase - s.load_kwh * f) for f in fracs)
            net_import = max(0.0, s.load_kwh - supply)
            net_export = max(0.0, supply - s.load_kwh)
            phase_cost = phase_import * imp_p - phase_export * exp_p
            net_cost = net_import * imp_p - net_export * exp_p
            slot_extra = phase_cost - net_cost
            if slot_extra > 1e-9:
                simulated_slot += slot_extra
                extra_import += max(0.0, phase_import - net_import)
                phase_flagged.append(i)

        simulated += simulated_slot

        # Idle-freeze exposure: Hold at/below target during a PV-surplus slot,
        # where the executor freezes discharge (cannot cover a transient spike).
        if (
            s.discharge_kwh <= 1e-6
            and s.soc_percent <= s.soc_target_percent + 0.5
            and s.pv_kwh > s.load_kwh
        ):
            idle_exposed.append(i)

    return RealismResult(
        predicted_cost_sek=round(predicted, 3),
        simulated_cost_sek=round(simulated, 3),
        realism_gap_sek=round(simulated - predicted, 3),
        extra_import_kwh=round(extra_import, 3),
        phase_flagged_slots=phase_flagged,
        idle_exposed_slots=idle_exposed,
    )


def realism_from_schedule(df: pd.DataFrame, config: dict[str, Any]) -> RealismResult:
    """Build RealismSlots from a planner schedule DataFrame and run the replay.

    Reads optional ``phase_load_fractions`` (e.g. {"A": 0.5, "B": 0.3, "C": 0.2})
    from config; absent => balanced => gap 0.
    """
    if df.empty:
        return RealismResult(0.0, 0.0, 0.0, 0.0)

    def col(row: Any, *names: str, default: float = 0.0) -> float:
        for name in names:
            if name in row and pd.notna(row[name]):
                return float(row[name])
        return default

    slots: list[RealismSlot] = []
    for _, row in df.iterrows():
        slots.append(
            RealismSlot(
                pv_kwh=col(row, "pv_kwh", "adjusted_pv_kwh", "pv_forecast_kwh"),
                load_kwh=col(row, "load_kwh", "adjusted_load_kwh", "load_forecast_kwh"),
                discharge_kwh=col(row, "kepler_discharge_kwh", "discharge_kwh"),
                grid_import_kwh=col(row, "kepler_import_kwh", "grid_import_kwh", "import_kwh"),
                grid_export_kwh=col(row, "kepler_export_kwh", "grid_export_kwh", "export_kwh"),
                import_price_sek_kwh=col(row, "import_price_sek_kwh"),
                export_price_sek_kwh=col(row, "export_price_sek_kwh"),
                soc_percent=col(row, "projected_soc_percent"),
                soc_target_percent=col(row, "soc_target_percent"),
            )
        )
    return simulate_realistic(slots, phase_fractions=config.get("phase_load_fractions"))
