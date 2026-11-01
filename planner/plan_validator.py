"""Independent plan validator (solver-agnostic).

Re-checks a KeplerResult against the KeplerInput/KeplerConfig it was solved from,
using only the reported per-slot numbers — never the solver's internals. It exists
because "the solver said Optimal" has burned this project before (GLPK shipping
zero-water garbage labeled Optimal, the heat/boost double-count phantom load, the
tripwire false-rejects): a cheap, dumb second opinion catches bookkeeping drift that
the solver cannot see in itself.

ERRORS (physically impossible or internally inconsistent — enforce mode rejects):
  * non-finite numbers, slot-count mismatch
  * SoC outside [0, capacity]
  * SoC trajectory not following charge/discharge with the configured efficiencies
  * per-slot energy balance not closing: supply - demand must lie in
    [0, pv] (the gap is curtailment); a deficit means unserved load
  * hard power limits: battery charge/discharge, inverter AC headroom, grid
    import/export caps, export while export is disabled
  * reported cost (per slot and total) differing from a recomputation

WARNINGS (legal for the solver but worth seeing — never rejected):
  * SoC below the min-SoC floor (incl. recovery window) or above max SoC — soft
    constraints in Kepler, so a plan can legally dip under them at a high penalty
  * simultaneous charge+discharge or import+export in one slot

Pure computation. ``validate_plan`` never raises on bad plans; it reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from planner.solver.kepler import min_soc_floor_kwh
from planner.solver.types import planned_slot_hours

if TYPE_CHECKING:
    from planner.solver.types import KeplerConfig, KeplerInput, KeplerResult

# Absolute tolerances (kWh / SEK). Solver outputs carry ~1e-7 noise; the expansion
# from the coarse grid and the 15-min arithmetic stay well inside 1e-3.
ENERGY_TOL_KWH = 1e-3
SOC_TOL_KWH = 0.01
COST_TOL_SEK = 0.05
COST_REL_TOL = 1e-3
SIMULTANEOUS_TOL_KWH = 0.01
MAX_MESSAGES = 12


@dataclass
class PlanValidation:
    ok: bool
    errors: list[str] = field(default_factory=lambda: [])
    warnings: list[str] = field(default_factory=lambda: [])
    error_count: int = 0
    warning_count: int = 0
    checked_slots: int = 0
    reported_cost_sek: float | None = None
    recomputed_cost_sek: float | None = None

    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe form for schedule meta / logs."""
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_slots": self.checked_slots,
            "reported_cost_sek": (
                round(self.reported_cost_sek, 3) if self.reported_cost_sek is not None else None
            ),
            "recomputed_cost_sek": (
                round(self.recomputed_cost_sek, 3) if self.recomputed_cost_sek is not None else None
            ),
        }


class _Collector:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.error_count = 0
        self.warning_count = 0

    def error(self, msg: str) -> None:
        self.error_count += 1
        if len(self.errors) < MAX_MESSAGES:
            self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < MAX_MESSAGES:
            self.warnings.append(msg)


def _finite(*values: float) -> bool:
    return all(isinstance(v, int | float) and math.isfinite(v) for v in values)


def _sink_power_kw(config: KeplerConfig) -> dict[str, float]:
    """Sink id -> power, mirroring Kepler's ladder (incl. the legacy synthesized rung)."""
    sinks = {s.id: float(s.power_kw) for s in config.excess_pv_sinks if s.enabled}
    if not config.excess_pv_sinks and (
        config.excess_pv_sink == "custom_entity" or config.excess_pv_custom_entity_enabled
    ):
        sinks["custom_entity"] = float(config.excess_pv_custom_entity_power_kw)
    return sinks


def validate_plan(
    result: KeplerResult, input_data: KeplerInput, config: KeplerConfig
) -> PlanValidation:
    """Check a solved plan against its inputs. See module docstring for the rules."""
    c = _Collector()
    slots = input_data.slots
    rs = result.slots
    if len(rs) != len(slots):
        c.error(f"slot count {len(rs)} != input slot count {len(slots)}")
        return PlanValidation(
            ok=False,
            errors=c.errors,
            error_count=c.error_count,
            checked_slots=0,
            reported_cost_sek=result.total_cost_sek,
        )

    cap = config.capacity_kwh
    eta_c = config.charge_efficiency if config.charge_efficiency > 0 else 1.0
    eta_d = config.discharge_efficiency if config.discharge_efficiency > 0 else 1.0
    # Same per-slot basis as the solver (a partial slot 0 has only its remainder).
    hours = planned_slot_hours(slots, getattr(input_data, "first_slot_remaining_h", None))
    initial = max(0.0, min(cap, input_data.initial_soc_kwh))
    min_soc = cap * config.min_soc_percent / 100.0
    max_soc = cap * config.max_soc_percent / 100.0
    floors = min_soc_floor_kwh(hours, initial, min_soc, cap)
    sink_kw = _sink_power_kw(config)

    prev_soc = initial
    recomputed = 0.0
    below_floor = above_max = both_batt = both_grid = 0
    for t, (s, r) in enumerate(zip(slots, rs, strict=True)):
        h = hours[t]
        label = f"slot {t} ({s.start_time.isoformat()})"
        ch, dis = r.charge_kwh, r.discharge_kwh
        imp, exp = r.grid_import_kwh, r.grid_export_kwh
        soc = r.soc_kwh
        if not _finite(ch, dis, imp, exp, soc, r.cost_sek):
            c.error(f"{label}: non-finite value")
            prev_soc = soc if _finite(soc) else prev_soc
            continue
        if min(ch, dis, imp, exp) < -ENERGY_TOL_KWH:
            c.error(
                f"{label}: negative flow (ch={ch:.3f} dis={dis:.3f} imp={imp:.3f} exp={exp:.3f})"
            )

        # SoC bounds and trajectory.
        if soc < -SOC_TOL_KWH or soc > cap + SOC_TOL_KWH:
            c.error(f"{label}: SoC {soc:.3f} kWh outside [0, {cap:.2f}]")
        expected = prev_soc + ch * eta_c - dis / eta_d
        if abs(soc - expected) > SOC_TOL_KWH:
            c.error(f"{label}: SoC {soc:.3f} kWh does not follow flows (expected {expected:.3f})")
        if soc < floors[t + 1] - SOC_TOL_KWH:
            below_floor += 1
        if soc > max_soc + SOC_TOL_KWH:
            above_max += 1
        prev_soc = soc

        # Energy balance: supply - demand = curtailment (>= 0, <= pv).
        demand = s.load_kwh + ch + exp
        demand += sum(max(0.0, kw) for kw in r.water_heater_results.values()) * h
        demand += sum(max(0.0, kw) for kw in r.ev_charger_results.values()) * h
        demand += max(0.0, r.deferrable_load_kw) * h
        demand += sum(sink_kw.get(sid, 0.0) for sid, on in r.sink_states.items() if on) * h
        residual = s.pv_kwh + dis + imp - demand
        tol = ENERGY_TOL_KWH * max(1.0, abs(demand))
        if residual < -tol:
            c.error(f"{label}: energy balance short by {-residual:.3f} kWh (unserved demand)")
        elif residual > max(0.0, s.pv_kwh) + tol:
            c.error(
                f"{label}: energy balance surplus {residual:.3f} kWh exceeds PV "
                f"{s.pv_kwh:.3f} (energy from nowhere)"
            )

        # Hard power limits.
        if ch > config.max_charge_power_kw * h + ENERGY_TOL_KWH:
            c.error(f"{label}: charge {ch:.3f} kWh > limit {config.max_charge_power_kw * h:.3f}")
        if dis > config.max_discharge_power_kw * h + ENERGY_TOL_KWH:
            c.error(
                f"{label}: discharge {dis:.3f} kWh > limit {config.max_discharge_power_kw * h:.3f}"
            )
        if config.max_inverter_ac_kw is not None:
            frac = config.hybrid_pv_fraction if config.hybrid_pv_fraction is not None else 1.0
            headroom = max(0.0, config.max_inverter_ac_kw * h - s.pv_kwh * frac)
            if dis > headroom + ENERGY_TOL_KWH:
                c.error(f"{label}: discharge {dis:.3f} kWh > inverter AC headroom {headroom:.3f}")
        if config.max_import_power_kw is not None and imp > (
            config.max_import_power_kw * h + ENERGY_TOL_KWH
        ):
            c.error(
                f"{label}: import {imp:.3f} kWh > grid limit {config.max_import_power_kw * h:.3f}"
            )
        if config.max_export_power_kw is not None and exp > (
            config.max_export_power_kw * h + ENERGY_TOL_KWH
        ):
            c.error(
                f"{label}: export {exp:.3f} kWh > grid limit {config.max_export_power_kw * h:.3f}"
            )
        if not config.enable_export and exp > ENERGY_TOL_KWH:
            c.error(f"{label}: export {exp:.3f} kWh while export is disabled")

        if ch > SIMULTANEOUS_TOL_KWH and dis > SIMULTANEOUS_TOL_KWH:
            both_batt += 1
        if imp > SIMULTANEOUS_TOL_KWH and exp > SIMULTANEOUS_TOL_KWH:
            both_grid += 1

        # Cost recomputation (Kepler's energy-only definition incl. half-cycle wear).
        slot_cost = (
            imp * s.import_price_sek_kwh
            - exp * s.export_price_sek_kwh
            + (ch + dis) * config.wear_cost_sek_per_kwh * 0.5
        )
        recomputed += slot_cost
        if abs(slot_cost - r.cost_sek) > max(COST_TOL_SEK, COST_REL_TOL * abs(slot_cost)):
            c.error(f"{label}: reported cost {r.cost_sek:.3f} != recomputed {slot_cost:.3f} SEK")

    if abs(recomputed - result.total_cost_sek) > max(COST_TOL_SEK, COST_REL_TOL * abs(recomputed)):
        c.error(f"total cost {result.total_cost_sek:.3f} SEK != recomputed {recomputed:.3f} SEK")
    if below_floor:
        c.warn(f"{below_floor} slot(s) end below the min-SoC floor (soft constraint)")
    if above_max:
        c.warn(f"{above_max} slot(s) end above max SoC (soft constraint)")
    if both_batt:
        c.warn(f"{both_batt} slot(s) charge and discharge simultaneously")
    if both_grid:
        c.warn(f"{both_grid} slot(s) import and export simultaneously")

    return PlanValidation(
        ok=c.error_count == 0,
        errors=c.errors,
        warnings=c.warnings,
        error_count=c.error_count,
        warning_count=c.warning_count,
        checked_slots=len(rs),
        reported_cost_sek=result.total_cost_sek,
        recomputed_cost_sek=recomputed,
    )
