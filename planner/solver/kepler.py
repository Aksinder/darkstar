"""
Kepler MILP Solver

Mixed-Integer Linear Programming solver for optimal battery scheduling.
Migrated from backend/kepler/solver.py during Rev K13 modularization.
"""

import logging
import os
from collections import defaultdict
from datetime import timedelta  # Rev WH2
from typing import Any

import pulp  # type: ignore[import,no-redef]

from planner.errors import PlannerError, PlannerErrorCode

from .types import (
    DeferrableLoadInput,
    EVChargerInput,
    ExcessPVSinkSpec,
    KeplerConfig,
    KeplerInput,
    KeplerResult,
    KeplerResultSlot,
    WaterHeaterInput,
)

logger = logging.getLogger("darkstar.kepler")

# Solver wall-clock budget (seconds). Replans run every 15 min, so a long solve is
# affordable; a silently time-boxed one is not (see the CBC-first comment at the
# solve call — GLPK at its old 30 s ceiling shipped garbage plans labeled Optimal).
# 120→240 (2026-07-22): the degenerate summer water-heating MILP straddles 120 s on
# the 2-vCPU box (heavy-tailed B&B; the 2026-07-15 seven-hour SOLVER_TIMEOUT freeze) —
# 240 s gives the strong solver headroom and still fits the 15-min cadence even with
# the CBC fallback chained after it (worst case ~8 min). Scheduler backoff already
# tolerates 240 ([60, 120, 240, 300]).
SOLVER_TIME_LIMIT_S = 240

# pulp is untyped; pin the one solution-status constant the guard relies on.
_LP_SOLUTION_OPTIMAL: int = int(pulp.constants.LpSolutionOptimal)  # type: ignore[reportUnknownMemberType]

# Excess-PV sink ladder: rung i earns reward * (1 - i * EPS) per kWh. Small enough
# never to flip a real economic decision (2% of the 0.5 SEK/kWh default), large
# enough that a PROVEN-optimal solve fills earlier rungs first under scarce
# surplus. NOTE: the adjacent-rung differential (~0.01 SEK/kWh at the 0.5 default
# reward) is far below the gapRel=0.01 stop and any time-boxed incumbent gap, so
# rung ordering is best-effort on early-stopped solves — a within-gap "Optimal"
# may permute rungs. Soft by design — no hard chain constraints (the executor-side
# comfort floor can veto a rung at runtime; a hard chain would then wrongly
# starve the rungs below it).
SINK_PRIORITY_EPSILON = 0.02


def solve_is_time_boxed(
    is_optimal: bool,
    sol_status: int,
    used_solver: str,
    solve_duration_s: float,
    limit_s: float = SOLVER_TIME_LIMIT_S,
) -> bool:
    """True when an "Optimal" label is really an unproven time-boxed incumbent.

    PuLP maps CBC's "Stopped on time - objective value X" (time limit hit with an
    incumbent whose optimality gap was NEVER proven <= gapRel) to LpStatusOptimal
    (pulp/apis/coin_api.py, the explicit NotSolved->Optimal promotion). The honest
    discriminator is prob.sol_status: it stays LpSolutionIntegerFeasible (2) for a
    time-boxed incumbent and is LpSolutionOptimal (1) only when CBC itself said
    "Optimal" (including "within gap tolerance"). Empirically verified: a 2s-limited
    solve with a 52,594% gap still reports LpStatus "Optimal" — sol_status is the
    only tell.

    HiGHS (in-process highspy, pulp 3.3.2 apis/highs_api.py status_dict) follows the
    SAME convention, verified empirically: kTimeLimit WITH an incumbent maps to
    (LpStatusOptimal, LpSolutionIntegerFeasible); kTimeLimit with NO incumbent trips
    the +inf-objective minimization check in highs_api and maps to (LpStatusNotSolved,
    LpSolutionNoSolutionFound); a gapRel stop reports kOptimal -> (Optimal, Optimal),
    indistinguishable from exact optimal — same as CBC's gapRel convention. So both
    "cbc" and "highs" use the sol_status test.

    GLPK's wrapper synthesizes sol_status from the status (always looks proven), so
    for that path only the wall clock can tell — and solvers exit just UNDER their
    limit (119.939s of 120), so compare against 95% of the budget.
    """
    if not is_optimal:
        return False
    if used_solver in ("cbc", "highs"):
        return sol_status != _LP_SOLUTION_OPTIMAL
    return solve_duration_s >= 0.95 * limit_s


def _phase_fraction_list(fractions: dict[str, float] | None) -> list[float] | None:
    """Normalise a {"A":..,"B":..,"C":..} split to a list summing to 1, or None."""
    if not fractions:
        return None
    vals = [max(0.0, float(v)) for v in fractions.values()]
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


# Keep only this many failed-solve dumps; each is a few hundred KB.
_SOLVER_DUMP_KEEP = 5


def _dump_failed_solve_instance(
    input_data: KeplerInput,
    config: KeplerConfig,
    *,
    reason: str,
    status: str,
    sol_status: int,
    used_solver: str,
    solve_duration_s: float,
    chain_duration_s: float,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Persist the exact solver input + outcome of a FAILED solve for offline replay.

    The 2026-07-15 SOLVER_TIMEOUT freeze could not be root-caused at the instance
    level because the failing KeplerInput was discarded with the error and the log
    ring had rotated. This dump is the missing evidence: re-running the artifact
    offline distinguishes a no-incumbent timeout from a comfort-violating incumbent
    and validates any solver-tuning fix against the REAL instance. Best-effort —
    a dump failure must never mask the PlannerError being raised.
    """
    try:
        import dataclasses
        import json
        import time as _time
        from pathlib import Path

        # Env-overridable so the test suite (whose solver tests intentionally fail
        # solves) never writes into — or evicts real incident dumps from — the
        # production data/ directory. tests/planner/conftest.py points this at tmp.
        dump_dir = Path(os.environ.get("DARKSTAR_SOLVER_DUMP_DIR", "data/solver_dumps"))
        dump_dir.mkdir(parents=True, exist_ok=True)
        path = dump_dir / f"kepler_fail_{_time.strftime('%Y%m%dT%H%M%S')}.json"
        payload: dict[str, Any] = {
            "reason": reason,
            "status": status,
            "sol_status": sol_status,
            "used_solver": used_solver,
            "solve_duration_s": round(solve_duration_s, 3),
            "chain_duration_s": round(chain_duration_s, 3),
            "time_limit_s": SOLVER_TIME_LIMIT_S,
            **(extra or {}),
            "config": dataclasses.asdict(config),
            "input": dataclasses.asdict(input_data),
        }
        path.write_text(json.dumps(payload, default=str))
        # Retention: newest _SOLVER_DUMP_KEEP only (name-sorted == time-sorted).
        for old in sorted(dump_dir.glob("kepler_fail_*.json"))[:-_SOLVER_DUMP_KEEP]:
            old.unlink(missing_ok=True)
        logger.warning("Failed solve instance persisted to %s for offline replay", path)
        return str(path)
    except Exception as exc:
        logger.warning("Could not persist failed solve instance: %s", exc)
        return None


class KeplerSolver:
    def solve(self, input_data: KeplerInput, config: KeplerConfig) -> KeplerResult:
        """Solve the energy scheduling problem using MILP.

        Args:
            input_data: Input data containing slots, initial SoC, etc.
            config: Solver configuration parameters

        Returns:
            KeplerResult with optimized schedule slots and cost information
        """
        """
        Solve the energy scheduling problem using MILP.
        """
        slots = input_data.slots
        T = len(slots)
        if T == 0:
            return KeplerResult(
                slots=[],
                total_cost_sek=0.0,
                is_optimal=True,
                status_msg="No slots to schedule",
            )

        # Calculate slot duration in hours
        slot_hours: list[float] = []
        for s in slots:
            duration = (s.end_time - s.start_time).total_seconds() / 3600.0
            slot_hours.append(duration)

        # Problem Definition
        prob: Any = pulp.LpProblem("KeplerSchedule", pulp.LpMinimize)

        # Variables (all in kWh per slot)
        charge: dict[int, Any] = pulp.LpVariable.dicts("charge_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
        discharge: dict[int, Any] = pulp.LpVariable.dicts("discharge_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
        grid_import: dict[int, Any] = pulp.LpVariable.dicts("import_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
        grid_export: dict[int, Any] = pulp.LpVariable.dicts("export_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
        curtailment: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "curtailment_kwh", range(T), lowBound=0.0
        )
        load_shedding: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "load_shedding_kwh", range(T), lowBound=0.0
        )

        # Water heating as deferrable load (per-device)
        water_heaters: list[WaterHeaterInput] = config.water_heaters
        water_enabled: bool = len(water_heaters) > 0

        # Per-device indexed variables: water_heat[device_id][t], water_start[device_id][t]
        water_heat: dict[str, dict[int, Any]] = {}
        water_start: dict[str, dict[int, Any]] = {}
        # Per-device boost variables: water_boost[device_id][t]
        water_boost: dict[str, dict[int, Any]] = {}
        boost_enabled = config.excess_pv_sink == "water_heater_boost"
        if water_enabled:
            for heater in water_heaters:
                d = heater.id
                safe_d = d.replace("-", "_").replace(".", "_")
                water_heat[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                    f"water_heat_{safe_d}", range(T), cat="Binary"
                )
                if boost_enabled:
                    water_boost[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                        f"water_boost_{safe_d}", range(T), cat="Binary"
                    )
                needs_start = (
                    heater.min_spacing_hours > 0 or config.water_block_start_penalty_sek > 0
                )
                if needs_start:
                    water_start[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                        f"water_start_{safe_d}", range(T), cat="Binary"
                    )

        # EV Charging as deferrable load (per-device, multi-charger support)
        # Only create variables for plugged-in chargers
        plugged_chargers: list[EVChargerInput] = [c for c in config.ev_chargers if c.plugged_in]
        ev_any_enabled: bool = len(plugged_chargers) > 0

        # Per-device indexed variables: ev_charge[device_id][t], ev_energy[device_id][t]
        # dict[charger_id -> dict[t -> lp_var]]
        ev_charge: dict[str, dict[int, Any]] = {}
        ev_energy: dict[str, dict[int, Any]] = {}
        # Per-device incentive bucket variables: ev_bucket_charged[device_id][bucket_idx]
        ev_bucket_charged: dict[str, dict[int, Any]] = {}

        for charger in plugged_chargers:
            d = charger.id
            safe_d = d.replace("-", "_").replace(".", "_")
            if charger.deadline:
                logger.info(
                    "EV %s: deadline constraint active at %s",
                    d,
                    charger.deadline.strftime("%Y-%m-%d %H:%M"),
                )
            else:
                logger.info("EV %s: no deadline constraint", d)

            ev_charge[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                f"ev_charge_{safe_d}", range(T), cat="Binary"
            )
            ev_energy[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                f"ev_energy_{safe_d}_kwh", range(T), lowBound=0.0
            )
            buckets = charger.incentive_buckets or []
            num_buckets = len(buckets)
            ev_bucket_charged[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                f"ev_bucket_{safe_d}", range(num_buckets), lowBound=0.0
            )

        # any_ev_charging[t]: auxiliary binary - 1 if ANY charger is active in slot t
        any_ev_charging: dict[int, Any]
        if ev_any_enabled:
            any_ev_charging = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                "any_ev_charging", range(T), cat="Binary"
            )
        else:
            any_ev_charging = dict.fromkeys(range(T), 0)

        # Deferrable household loads (dishwasher, washing machine, ...).
        # Each pending run is a single non-interruptible contiguous block that
        # must run exactly once within the horizon. We model only the START slot
        # as a binary; "running" in slot t is the linear expression
        # sum(start[s] for s in [t-N+1 .. t]) which is 0/1 because exactly one
        # start is chosen and the block is contiguous.
        defl_start: dict[str, dict[int, Any]] = {}
        defl_valid: dict[str, list[int]] = {}
        defl_n: dict[str, int] = {}
        defl_energy_per_slot: dict[str, float] = {}
        scheduled_defl: list[DeferrableLoadInput] = []
        for load in config.deferrable_loads:
            n = max(1, int(load.duration_slots))
            e_start = max(0, int(load.earliest_start_slot))
            last_start = T - n  # last slot index at which a full block still fits
            valid = list(range(e_start, last_start + 1))
            if n > T or not valid:
                logger.warning(
                    "Deferrable load %s: cannot fit (dur=%d slots, earliest=%d, T=%d) - skipping",
                    load.id,
                    n,
                    e_start,
                    T,
                )
                continue
            safe_d = load.id.replace("-", "_").replace(".", "_")
            defl_start[load.id] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                f"defl_start_{safe_d}", valid, cat="Binary"
            )
            defl_valid[load.id] = valid
            defl_n[load.id] = n
            defl_energy_per_slot[load.id] = load.energy_kwh / n
            scheduled_defl.append(load)
            # Run within the valid start window. A priority-bearing load (WTP layer
            # enabled) is OPTIONAL — at most once — so it can be skipped when no slot
            # is cheap enough for its reservation price; its urgency ramp pulls it in
            # before the deadline when running is worthwhile. Every other load keeps
            # the legacy mandatory run-once semantics (byte-identical when off).
            if config.load_priority_enabled and load.id in config.load_priorities:
                prob += pulp.lpSum(defl_start[load.id][t] for t in valid) <= 1
            else:
                prob += pulp.lpSum(defl_start[load.id][t] for t in valid) == 1

        def _defl_run_expr(load_id: str, t: int) -> Any:
            """Linear 0/1 expression: is deferrable load `load_id` running in slot t?"""
            n = defl_n[load_id]
            starts = defl_start[load_id]
            return pulp.lpSum(starts[s] for s in defl_valid[load_id] if 0 <= t - s < n)

        # SoC state variables (T+1 states for T slots)

        # SoC state variables (T+1 states for T slots)
        min_soc_kwh: float = config.capacity_kwh * config.min_soc_percent / 100.0
        max_soc_kwh: float = config.capacity_kwh * config.max_soc_percent / 100.0

        soc: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "soc_kwh", range(T + 1), lowBound=0.0, upBound=config.capacity_kwh
        )

        # Slack variables
        soc_violation: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "soc_violation_kwh", range(T + 1), lowBound=0.0
        )
        soc_overshoot: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "soc_overshoot_kwh", range(T + 1), lowBound=0.0
        )
        target_under_violation: Any = pulp.LpVariable(
            "target_under_violation_kwh", lowBound=0.0
        )  # Penalty for being BELOW target at end of horizon
        import_breach: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
            "import_breach_kwh", range(T), lowBound=0.0
        )
        ramp_up: dict[int, Any] = pulp.LpVariable.dicts("ramp_up_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
        ramp_down: dict[int, Any] = pulp.LpVariable.dicts("ramp_down_kwh", range(T), lowBound=0.0)  # type: ignore[reportUnknownMemberType]

        # Export floor SoC constraint (gated on enable_export and export_floor_soc_percent)
        export_floor_active = config.enable_export and config.export_floor_soc_percent is not None
        is_exporting: dict[int, Any]
        export_floor_violation: dict[int, Any]
        export_floor_kwh: float = 0.0
        if export_floor_active:
            assert config.export_floor_soc_percent is not None
            export_floor_kwh = config.capacity_kwh * config.export_floor_soc_percent / 100.0
            is_exporting = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                "is_exporting", range(T), cat="Binary"
            )
            export_floor_violation = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                "export_floor_violation_kwh", range(T), lowBound=0.0
            )
        else:
            is_exporting = dict.fromkeys(range(T), 0)
            export_floor_violation = dict.fromkeys(range(T), 0)

        # Discomfort variable removed.
        # Per-device "Block Overshoot" variables (soft penalty for massive blocks)
        block_overshoot: dict[str, dict[int, Any]] = {}
        if water_enabled:
            for heater in water_heaters:
                d = heater.id
                safe_d = d.replace("-", "_").replace(".", "_")
                block_overshoot[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                    f"block_overshoot_{safe_d}", range(T), lowBound=0.0
                )

        # Per-device incentive bucket setup
        for charger in plugged_chargers:
            d = charger.id
            buckets = charger.incentive_buckets or []
            num_buckets = len(buckets)
            if num_buckets == 0:
                continue

            ev_capacity: float = charger.battery_capacity_kwh
            ev_current_kwh: float = ev_capacity * (charger.current_soc_percent / 100.0)

            prev_threshold_soc: float = 0.0
            accum_energy_cap: float = 0.0

            for i, b in enumerate(buckets):
                bucket_soc_range: float = b.threshold_soc - prev_threshold_soc
                bucket_capacity_kwh: float = max(0.0, ev_capacity * (bucket_soc_range / 100.0))

                already_full: float = max(
                    0.0, min(bucket_capacity_kwh, ev_current_kwh - accum_energy_cap)
                )
                remaining_cap: float = max(0.0, bucket_capacity_kwh - already_full)

                prob += ev_bucket_charged[d][i] <= remaining_cap

                prev_threshold_soc = b.threshold_soc
                accum_energy_cap += bucket_capacity_kwh

            # Total energy for this device must equal sum of its buckets
            prob += pulp.lpSum(ev_energy[d][t] for t in range(T)) == pulp.lpSum(
                ev_bucket_charged[d][i] for i in range(num_buckets)
            )

        # Per-device slack variables for daily minimum soft constraints
        water_min_kwh_violation: dict[str, dict[int, Any]] = {}
        # 2026-08-10 absorption-cap overage slack (SOFT cap — see the constraint below
        # for why hard failed review: hourly-block lattice vs slot-granular floors made
        # the comfort floor unreachable, and misaligned force_on went infeasible).
        water_absorb_overage: dict[str, dict[int, Any]] = {}
        if water_enabled:
            for heater in water_heaters:
                d = heater.id
                safe_d = d.replace("-", "_").replace(".", "_")
                water_min_kwh_violation[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                    f"water_min_kwh_violation_{safe_d}", range(100), lowBound=0.0
                )
                if heater.absorb_cap_kwh_per_day is not None:
                    water_absorb_overage[d] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
                        f"water_absorb_overage_{safe_d}", range(100), lowBound=0.0
                    )

        # Initial SoC Constraint
        initial_soc: float = max(0.0, min(config.capacity_kwh, input_data.initial_soc_kwh))
        prob += soc[0] == initial_soc

        # Excess PV slot flags (pre-calculated from forecasts)
        excess_pv_flags: list[bool] = config.excess_pv_slots
        if excess_pv_flags and len(excess_pv_flags) != T:
            logger.warning(
                "Excess PV flags length (%d) != slot count (%d), ignoring", len(excess_pv_flags), T
            )
            excess_pv_flags = []
        if not excess_pv_flags:
            excess_pv_flags = [False] * T

        # PERF (hourly decision blocks): tie the water binaries together within each
        # wall-clock hour. The overnight cheap band and the midday PV plateau contain
        # dozens of near-identical 15-min slots, so branch-and-bound explores
        # thousands of near-equivalent schedules (the 1e-5 symmetry-breaker in the
        # objective is below CBC's tolerances and prunes nothing). Measured on the
        # live 107-slot model: 31.1s -> 3.8s (8.2x) for +0.03 SEK (~0.1%) objective
        # cost, identical water totals. Slots stay 15-min for battery/EV and for the
        # executor — water is simply constant within the hour. Boost groups split
        # additionally on the excess-PV flag so a flag edge inside an hour cannot
        # force the whole hour to zero (boost is hard-forced 0 on unflagged slots).
        if water_enabled and config.water_hourly_blocks and T > 0:
            _hour_of: list[Any] = [
                slots[t].start_time.replace(minute=0, second=0, microsecond=0) for t in range(T)
            ]
            _heat_groups: defaultdict[Any, list[int]] = defaultdict(list)
            _boost_groups: defaultdict[Any, list[int]] = defaultdict(list)
            for t in range(T):
                _heat_groups[_hour_of[t]].append(t)
                _boost_groups[(_hour_of[t], excess_pv_flags[t])].append(t)
            for d in water_heat:
                for _group in _heat_groups.values():
                    for t in _group[1:]:
                        prob += water_heat[d][t] == water_heat[d][_group[0]]  # type: ignore[operator]
            if boost_enabled:
                for d in water_boost:
                    for _group in _boost_groups.values():
                        for t in _group[1:]:
                            prob += water_boost[d][t] == water_boost[d][_group[0]]  # type: ignore[operator]

        # Prioritized excess-PV sink ladder (ordered: index = priority rung after the
        # water-heater boost). The configured list wins; when the adapter did not
        # populate it, synthesize a single rung from the legacy scalar custom-entity
        # fields so old callers/configs behave byte-identically.
        custom_entity_enabled = (
            config.excess_pv_sink == "custom_entity" or config.excess_pv_custom_entity_enabled
        )
        sinks: list[ExcessPVSinkSpec] = [s for s in config.excess_pv_sinks if s.enabled]
        # Synthesize only when NO ladder list was supplied at all. An all-disabled
        # ladder (observe-first rollout) must win the dual-read: the executor skips
        # disabled rungs and actuates nothing, so the planner must schedule nothing —
        # resurrecting the legacy rung here would make plan and reality diverge.
        if not config.excess_pv_sinks and custom_entity_enabled:
            sinks = [
                ExcessPVSinkSpec(
                    id="custom_entity",
                    power_kw=config.excess_pv_custom_entity_power_kw,
                    price_ceiling_sek_per_kwh=config.excess_pv_price_ceiling_sek_per_kwh,
                    enabled=True,
                )
            ]

        def _sink_price_ok(sink: ExcessPVSinkSpec, t: int) -> bool:
            # Optional per-rung price gate: only soak surplus locally when export
            # pays at or below the ceiling (incl. negative). Sell it otherwise.
            return (
                sink.price_ceiling_sek_per_kwh is None
                or slots[t].export_price_sek_kwh <= sink.price_ceiling_sek_per_kwh
            )

        # SoC threshold binary: 1 when battery SoC >= threshold% (gates all sink activation)
        any_sink_active = boost_enabled or bool(sinks)
        soc_above_threshold: dict[int, Any]
        if any_sink_active:
            # Sinks (boost/ladder) are hard-forced to 0 outside excess-PV slots, so
            # the gating binary is dead weight there — create it only where a sink
            # can actually fire (tonight's live model: 52 of 107 deleted). The
            # constant 0 elsewhere keeps the gate constraints trivially satisfied.
            soc_above_threshold = {
                t: (
                    pulp.LpVariable(  # type: ignore[reportUnknownMemberType]
                        f"soc_above_threshold_{t}", cat="Binary"
                    )
                    if excess_pv_flags[t]
                    else 0
                )
                for t in range(T)
            }
        else:
            soc_above_threshold = dict.fromkeys(range(T), 0)

        # Per-sink activation binaries, created ONLY on flagged + price-ok slots
        # (constant 0 elsewhere keeps the gates trivially satisfied and the model
        # small — this also fixes the old all-T custom_entity_active inefficiency).
        sink_active: dict[str, dict[int, Any]] = {}
        for sink in sinks:
            safe_sink = sink.id.replace("-", "_").replace(".", "_")
            sink_active[sink.id] = {
                t: (
                    pulp.LpVariable(  # type: ignore[reportUnknownMemberType]
                        f"sink_active_{safe_sink}_{t}", cat="Binary"
                    )
                    if excess_pv_flags[t] and _sink_price_ok(sink, t)
                    else 0
                )
                for t in range(T)
            }

        # Sink binaries are deliberately NOT hourly-blocked: build #9 solved the
        # live model with per-slot custom_entity binaries on ALL T slots, and the
        # flagged+price-ok restriction above already makes this model strictly
        # smaller. Hour-tying the sinks held the villavagn AC off until the hour
        # boundary when the SoC gate opened mid-hour (per-slot gate at the
        # soc_above_threshold constraint below is endogenous — it cannot key the
        # groups), a live-config behavior change vs build #9.

        # Objective Function Terms
        total_cost: list[Any] = []

        # Penalty constants
        MIN_SOC_PENALTY = 1000.0  # Hard constraint - don't violate min_soc!
        MAX_SOC_PENALTY = 1000.0  # Soft constraint - prefer to stay below max_soc
        EXPORT_FLOOR_PENALTY = 1000.0  # Soft constraint - don't export below floor
        BOOST_REWARD_SEK = config.excess_pv_reward_sek_per_kwh
        # Target penalty comes from config (derived from risk_appetite in pipeline)
        target_soc_penalty = config.target_soc_penalty_sek
        curtailment_penalty = config.curtailment_penalty_sek
        LOAD_SHEDDING_PENALTY = 10000.0
        IMPORT_BREACH_PENALTY = 5000.0

        for t in range(T):
            s: Any = slots[t]
            h: float = slot_hours[t]

            # Water heating load for this slot (kWh) — sum across all per-device heaters
            # Includes both normal and boost heating
            water_load_kwh: Any = (
                pulp.lpSum(
                    (
                        water_heat[heater.id][t]
                        + (water_boost[heater.id][t] if heater.id in water_boost else 0)
                    )
                    * heater.power_kw
                    * h
                    for heater in water_heaters
                )
                if water_enabled
                else 0
            )

            # Boost constrained to excess PV slots only, gated by SoC threshold
            if water_enabled and boost_enabled:
                for heater in water_heaters:
                    if heater.id not in water_boost:
                        continue
                    if not excess_pv_flags[t]:
                        prob += water_boost[heater.id][t] == 0
                    else:
                        prob += water_boost[heater.id][t] <= soc_above_threshold[t]
                        # 2026-08-10: heat and boost drive the SAME physical element —
                        # they can never both be on. Without this, the energy balance
                        # charged 2x power_kw in double-on slots (both binaries at 1)
                        # while result extraction reported power once: ~5 kWh/day of
                        # hidden phantom load the LP paid for but no one could see.
                        prob += (  # type: ignore[operator]
                            water_heat[heater.id][t] + water_boost[heater.id][t] <= 1
                        )
                        total_cost.append(
                            -BOOST_REWARD_SEK * water_boost[heater.id][t] * heater.power_kw * h
                        )

            # Sink ladder: gate each rung on the shared SoC threshold and pay a
            # soft priority-graded reward — rung i earns reward*(1 - i*EPS) per
            # kWh, so under scarce surplus the solver fills earlier rungs first
            # and under abundant surplus all rungs run. Off-flag / price-gated
            # slots hold constant-0 entries, so no gate constraint is needed there.
            for rung, sink in enumerate(sinks):
                sink_var: Any = sink_active[sink.id][t]
                if isinstance(sink_var, int):
                    continue
                prob += sink_var <= soc_above_threshold[t]
                sink_reward = BOOST_REWARD_SEK * (1.0 - rung * SINK_PRIORITY_EPSILON)
                total_cost.append(-sink_reward * sink_var * sink.power_kw * h)

            # SoC threshold gate (after soc[t] is defined via battery dynamics).
            # Placed here so soc[t] is available, then linked to boost/custom above.
            # M-free formulation: soc[t] >= threshold * b — b=1 forces SoC above the
            # threshold, b=0 is vacuous (soc >= 0). Tighter than the old
            # capacity-sized big-M, whose loose relaxation let fractional binaries
            # harvest the boost reward almost for free at the root node (a major
            # driver of CBC's weak bound / long solves). Only emitted for slots
            # where the gating binary exists (excess-PV flagged).
            if any_sink_active and excess_pv_flags[t]:
                threshold_kwh = config.capacity_kwh * config.excess_pv_soc_threshold_percent / 100.0
                prob += soc[t] >= threshold_kwh * soc_above_threshold[t]  # type: ignore[operator]

            # Per-device EV constraints
            for charger in plugged_chargers:
                d = charger.id
                # Energy coupling: binary ON/OFF at max power
                prob += ev_energy[d][t] == ev_charge[d][t] * charger.max_power_kw * h

                # Deadline constraint: zero charging after deadline
                if charger.deadline is not None and s.end_time > charger.deadline:
                    prob += ev_energy[d][t] == 0.0

                # Grid-only constraint: EV cannot charge from battery discharge
                prob += ev_energy[d][t] <= grid_import[t] + s.pv_kwh + 1e-6

            # any_ev_charging[t] linking constraints
            if ev_any_enabled:
                for charger in plugged_chargers:
                    d = charger.id
                    prob += any_ev_charging[t] >= ev_charge[d][t]
                prob += any_ev_charging[t] <= pulp.lpSum(ev_charge[d][t] for d in ev_charge)
                # Discharge blocking: block when any charger active
                M_discharge = config.max_discharge_power_kw * h
                prob += discharge[t] <= (1 - any_ev_charging[t]) * M_discharge

            # Total EV energy in this slot (sum across all plugged-in chargers)
            total_ev_energy_t: Any = (
                pulp.lpSum(ev_energy[d][t] for d in ev_energy) if ev_any_enabled else 0.0
            )

            # Energy Balance Constraint (water, EV, and sink-ladder loads added to demand side)
            sink_load_kwh: Any = (
                pulp.lpSum(sink_active[sk.id][t] * sk.power_kw * h for sk in sinks)
                if sinks
                else 0.0
            )

            # Deferrable household loads running this slot (energy split evenly
            # across the block's slots), added to the demand side.
            deferrable_load_kwh_t: Any = (
                pulp.lpSum(
                    _defl_run_expr(load.id, t) * defl_energy_per_slot[load.id]
                    for load in scheduled_defl
                )
                if scheduled_defl
                else 0.0
            )

            prob += (
                s.load_kwh
                + water_load_kwh
                + total_ev_energy_t
                + sink_load_kwh
                + deferrable_load_kwh_t
                + charge[t]
                + grid_export[t]
                + curtailment[t]
                == s.pv_kwh + discharge[t] + grid_import[t] + load_shedding[t]
            )

            # Per-device block start detection (task 2.7)
            if water_enabled:
                for heater in water_heaters:
                    d = heater.id
                    if d in water_start:
                        if t == 0:
                            prob += water_start[d][t] == water_heat[d][t]
                        else:
                            prob += water_start[d][t] >= water_heat[d][t] - water_heat[d][t - 1]

            # Per-device mid-block locking (task 2.8)
            if water_enabled:
                for heater in water_heaters:
                    d = heater.id
                    if heater.force_on_slots:
                        for t_idx in heater.force_on_slots:
                            if 0 <= t_idx < T:
                                prob += water_heat[d][t_idx] == 1

            # Battery Dynamics Constraint
            prob += soc[t + 1] == soc[t] + charge[t] * config.charge_efficiency - discharge[t] / (
                config.discharge_efficiency if config.discharge_efficiency > 0 else 1.0
            )

            # Power Limits
            max_chg_kwh: float = config.max_charge_power_kw * h
            max_dis_kwh: float = config.max_discharge_power_kw * h

            prob += charge[t] <= max_chg_kwh
            prob += discharge[t] <= max_dis_kwh

            if config.max_export_power_kw is not None:
                prob += grid_export[t] <= config.max_export_power_kw * h

            if config.max_import_power_kw is not None:
                prob += grid_import[t] <= config.max_import_power_kw * h

            # Inverter AC output limit: battery discharge shares the hybrid inverter's AC bus
            # with that inverter's OWN PV. On a multi-inverter site, PV on a separate
            # AC-coupled inverter (e.g. Fronius) does NOT consume the hybrid inverter's
            # headroom, so subtract only the hybrid inverter's share of PV
            # (hybrid_pv_fraction); None => all PV (legacy single-inverter behaviour).
            if config.max_inverter_ac_kw is not None:
                inverter_ac_kwh = config.max_inverter_ac_kw * h
                hybrid_pv_kwh = s.pv_kwh * (
                    config.hybrid_pv_fraction if config.hybrid_pv_fraction is not None else 1.0
                )
                prob += discharge[t] <= max(0.0, inverter_ac_kwh - hybrid_pv_kwh)

            # Soft Grid Import Limit
            if config.grid_import_limit_kw is not None:
                limit_kwh: float = config.grid_import_limit_kw * h
                prob += grid_import[t] <= limit_kwh + import_breach[t]

            # Rev E4: Strict Export Toggle
            if not config.enable_export:
                prob += grid_export[t] == 0

            # Export floor SoC constraint (big-M formulation)
            if export_floor_active:
                M_export = (
                    config.max_export_power_kw
                    if config.max_export_power_kw is not None
                    else config.max_discharge_power_kw
                ) * h
                prob += grid_export[t] <= M_export * is_exporting[t]
                prob += soc[t + 1] >= (
                    export_floor_kwh * is_exporting[t]
                    + min_soc_kwh * (1 - is_exporting[t])
                    - export_floor_violation[t]
                )

            # Ramping Constraints
            if t > 0:
                prob += (charge[t] - discharge[t]) - (charge[t - 1] - discharge[t - 1]) == ramp_up[
                    t
                ] - ramp_down[t]
            else:
                prob += ramp_up[t] == 0
                prob += ramp_down[t] == 0

            # Objective Terms
            # Wear cost modeling: Apply 50% of config value per action (charge OR discharge)
            # so that a full cycle (charge + discharge) costs exactly config.wear_cost_sek_per_kwh
            slot_wear_cost: Any = (charge[t] + discharge[t]) * config.wear_cost_sek_per_kwh * 0.5
            slot_import_cost: Any = grid_import[t] * s.import_price_sek_kwh
            effective_export_price: float = (
                s.export_price_sek_kwh - config.export_threshold_sek_per_kwh
            )
            slot_export_revenue: Any = grid_export[t] * effective_export_price
            slot_ramping_cost: Any = (
                (ramp_up[t] + ramp_down[t]) / h
            ) * config.ramping_cost_sek_per_kw
            slot_curtailment_cost: Any = curtailment[t] * curtailment_penalty
            slot_shedding_cost: Any = load_shedding[t] * LOAD_SHEDDING_PENALTY
            slot_import_breach_cost: Any = import_breach[t] * IMPORT_BREACH_PENALTY

            # NOTE: Rev K20 stored_energy_cost was removed - it incorrectly made
            # charging unprofitable by adding cost on discharge without offsetting
            # credit on charge. The terminal_value and wear_cost are sufficient
            # for arbitrage decisions.

            slot_ev_cost: float = 0.0  # EV incentive handled in aggregate objective below

            total_cost.append(
                slot_import_cost
                - slot_export_revenue
                + slot_wear_cost
                + slot_ramping_cost
                + slot_curtailment_cost
                + slot_shedding_cost
                + slot_import_breach_cost
                + slot_ev_cost
            )

            # Soft Min/Max SoC Constraints
            prob += soc[t] >= min_soc_kwh - soc_violation[t]
            prob += soc[t] <= max_soc_kwh + soc_overshoot[t]

        # Terminal constraints
        prob += soc[T] >= min_soc_kwh - soc_violation[T]
        prob += soc[T] <= max_soc_kwh + soc_overshoot[T]

        # Deferrable-load deadline penalty (tardiness). A run that starts at slot
        # s finishes at s+N-1; slots beyond the deadline are penalised. Because
        # the lateness of each candidate start is a constant, this stays linear.
        for load in scheduled_defl:
            # A priority-bearing load prices its deadline via the WTP urgency ramp
            # (added below), so skip the legacy tardiness penalty to avoid double-counting.
            if config.load_priority_enabled and load.id in config.load_priorities:
                continue
            n = defl_n[load.id]
            deadline = load.deadline_slot if load.deadline_slot is not None else (T - 1)
            penalty = (
                config.deferrable_hard_deadline_penalty_sek
                if load.deadline_hard
                else config.deferrable_soft_deadline_penalty_sek
            )
            if penalty > 0:
                total_cost.append(
                    penalty
                    * pulp.lpSum(
                        defl_start[load.id][s] * max(0, (s + n - 1) - deadline)
                        for s in defl_valid[load.id]
                    )
                )

        # Load-priority WTP credit (flag-gated). For each priority-bearing deferrable
        # load, running it in slot t earns a credit = WTP(t) * energy_slot, where
        # WTP(t) = base + rank_epsilon + urgency(t); urgency ramps linearly from 0 at
        # the earliest start to urgency_wtp at the deadline. The credit is granted ONLY
        # when the load actually runs (multiplied by the linear run indicator), never
        # for exporting/curtailing, so it cannot distort those decisions. The solver
        # runs the load iff WTP(t) >= the marginal price — WTP is a reservation price.
        # All coefficients are Python constants, so the term is strictly linear (no new
        # variables, no MILP growth).
        if config.load_priority_enabled:
            for load in scheduled_defl:
                lp = config.load_priorities.get(load.id)
                if lp is None:
                    continue
                e_slot = defl_energy_per_slot[load.id]
                earliest = max(0, int(load.earliest_start_slot))
                deadline = load.deadline_slot if load.deadline_slot is not None else (T - 1)
                span = max(1, deadline - earliest)
                base = lp.base_wtp_sek_per_kwh + lp.rank_epsilon_sek_per_kwh
                for t in range(T):
                    ramp = min(1.0, max(0.0, (t - earliest) / span))
                    wtp_t = base + lp.urgency_wtp_sek_per_kwh * ramp
                    if wtp_t == 0.0:
                        continue
                    total_cost.append(-wtp_t * e_slot * _defl_run_expr(load.id, t))

        # Deferrable-load phase balancing (optional): penalise two same-phase
        # loads running concurrently, so large single-phase appliances spread out.
        if config.deferrable_phase_penalty_sek > 0 and len(scheduled_defl) >= 2:
            phase_groups: defaultdict[str, list[DeferrableLoadInput]] = defaultdict(list)
            for load in scheduled_defl:
                if load.phase:
                    phase_groups[str(load.phase)].append(load)
            for phase, loads in phase_groups.items():
                if len(loads) < 2:
                    continue
                safe_p = phase.replace("-", "_").replace(".", "_")
                for t in range(T):
                    overlap: Any = pulp.LpVariable(  # type: ignore[reportUnknownMemberType]
                        f"defl_phase_over_{safe_p}_{t}", lowBound=0
                    )
                    prob += overlap >= (  # type: ignore[operator]
                        pulp.lpSum(_defl_run_expr(load.id, t) for load in loads) - 1
                    )
                    total_cost.append(config.deferrable_phase_penalty_sek * overlap)

        # Phase-aware imbalance cost (flag-gated, default OFF). The single-net-node
        # balance nets a heavy phase's import against the light phases' export to ~zero,
        # hiding the real cost. Under balanced inverter supply (pv + discharge - charge)/n,
        # phase p still imports max(0, load*f_p - supply/n). We price that per-phase import
        # deficit (the part the net view hides) at the slot's import price, so the solver
        # raises discharge to cover the heavy phase — but only WHEN ECONOMIC, since each
        # extra kWh discharged costs terminal battery value + wear and spills as export.
        # Each phase_import var is minimised (positive objective coeff) so it settles at
        # its true max(0, ...) lower bound (no free-variable gaming). Uses the per-hour
        # profile when present so the cost reflects which phase is heavy at the slot's hour.
        if config.phase_aware_enabled and (
            config.phase_load_profile or config.phase_load_fractions
        ):
            static_fracs = _phase_fraction_list(config.phase_load_fractions)
            for t in range(T):
                s_pa: Any = slots[t]
                fracs: list[float] | None = None
                if config.phase_load_profile is not None:
                    fracs = _phase_fraction_list(
                        config.phase_load_profile.get(s_pa.start_time.hour)
                    )
                if fracs is None:
                    fracs = static_fracs
                if not fracs:
                    continue
                n_ph = len(fracs)
                imp_pa: float = s_pa.import_price_sek_kwh
                # Real balanced supply to the house = PV + battery discharge - battery
                # charge (charging consumes from the AC bus). Subtracting charge is
                # essential: with pv+discharge alone the solver fakes phase coverage via a
                # free charge+discharge cycle that nets zero real supply.
                supply_pa: Any = s_pa.pv_kwh + discharge[t] - charge[t]
                for k, f in enumerate(fracs):
                    pi: Any = pulp.LpVariable(  # type: ignore[reportUnknownMemberType]
                        f"phase_imp_{t}_{k}", lowBound=0.0
                    )
                    prob += pi >= s_pa.load_kwh * f - supply_pa / n_ph  # type: ignore[operator]
                    total_cost.append(pi * imp_pa * config.phase_aware_weight)

        # Terminal SoC Target (BIDIRECTIONAL soft constraint)
        # Penalize both being UNDER target (risk) AND OVER target (missed discharge opportunity)
        target_soc_kwh: float = (
            config.target_soc_kwh if config.target_soc_kwh is not None else min_soc_kwh
        )

        # Terminal SoC Target (BIDIRECTIONAL soft constraint)
        # Penalize both being UNDER target (risk) AND OVER target (missed discharge opportunity)
        target_soc_kwh = config.target_soc_kwh if config.target_soc_kwh is not None else min_soc_kwh

        if config.target_soc_kwh is not None:
            # Under target: soc[T] >= target - under_violation
            prob += soc[T] >= target_soc_kwh - target_under_violation

            # Penalize UNDER target (important)
            total_cost.append(target_soc_penalty * target_under_violation)
        else:
            # If no target, we don't care where we end up (within min_soc limits)
            pass

        # Improvement B: continuous stored-energy value (Predbat-inspired).
        # Reward energy remaining at the END of the horizon at its expected forward
        # worth. Applied to soc[T] ONLY -> symmetric (no discharge penalty without an
        # offsetting charge credit), so unlike the removed K20 stored_energy_cost it
        # cannot distort mid-horizon cycling. Lets the planner hold cheaply-charged
        # energy for a genuinely more expensive period without a hard floor forcing
        # grid top-ups. Wear cost still discourages pointless churn.
        if config.battery_value_sek_per_kwh > 0:
            total_cost.append(-config.battery_value_sek_per_kwh * soc[T])

        # Rev // F51: Removed legacy EV target SoC constraint.
        # Replaced by Incentive Buckets in the objective function.

        # Water Heating Constraints — per-device (tasks 2.4-2.6)
        gap_violation_penalty: float = 0.0
        sorted_days: list[Any] = []  # Initialize to avoid unbound error
        # Build #16: collected per-slot anchor REWARD terms (negative cost) for heaters
        # whose previous-plan position survives the price-gate below. Added to the
        # objective next to the symmetry breaker. Empty => no anchor active.
        anchor_reward_terms: list[Any] = []
        if water_enabled:
            avg_slot_hours: float = sum(slot_hours) / len(slot_hours) if slot_hours else 0.25

            # Soft-cap overage price: just above the boost reward so reward-farming
            # beyond the cap nets negative per kWh, yet orders of magnitude below the
            # reliability floor (15-1000 SEK/kWh) and the WTP credits — comfort and
            # force_on run over the cap for pennies, phantom booking cannot.
            absorb_overage_penalty: float = max(0.05, config.excess_pv_reward_sek_per_kwh * 1.2)

            # Build day → slot indices map (shared across all devices, global deferral)
            slots_by_day: defaultdict[Any, list[int]] = defaultdict(list)
            defer_hours: float = config.defer_up_to_hours
            for t in range(T):
                dt: Any = slots[t].start_time
                bucket_date: Any = dt.date()
                if defer_hours > 0 and dt.hour < defer_hours:
                    bucket_date = bucket_date - timedelta(days=1)
                slots_by_day[bucket_date].append(t)
            sorted_days = sorted(slots_by_day.keys())

            for heater in water_heaters:
                d = heater.id
                # Per-device kWh per slot (power differs between heaters)
                kwh_per_slot: float = heater.power_kw * avg_slot_hours

                # Constraint 1: Per-device, per-day daily minimum (task 2.4)
                for i, day in enumerate(sorted_days):
                    day_slot_indices: list[int] = slots_by_day[day]
                    if i == 0:
                        # First day: deduct per-device heated-today progress
                        day_min_kwh: float = max(
                            0.0, heater.min_kwh_per_day - heater.heated_today_kwh
                        )
                    else:
                        day_min_kwh = heater.min_kwh_per_day

                    if day_min_kwh > 0:
                        prob += (  # type: ignore[operator]
                            pulp.lpSum(water_heat[d][t] for t in day_slot_indices) * kwh_per_slot
                            >= day_min_kwh - water_min_kwh_violation[d][i]
                        )

                    # 2026-08-10 absorption cap (SOFT): what the PLAN may book into this
                    # tank per day-bucket (heat + boost together) without paying overage.
                    # The solver has no tank model, and the boost reward (>= export price
                    # everywhere) otherwise pays it to "consume" 30+ kWh/day into tanks
                    # whose measured intake is ~4-5 kWh — phantom load that ate the whole
                    # modeled PV surplus, so the battery never charged.
                    #
                    # WHY SOFT (a hard cap failed adversarial review three ways):
                    #  - hourly_blocks quantizes heat to 4-slot hour groups, so a hard cap
                    #    raised to a slot-granular floor FORBIDS the next attainable group
                    #    and the comfort floor becomes unreachable at almost every replan;
                    #  - force_on slots misaligned with hour groups went hard INFEASIBLE
                    #    (no plan at all);
                    #  - a hard cap deleted boost outright instead of grounding it.
                    # The overage penalty is set just above the boost reward: farming the
                    # reward beyond the cap is unprofitable by construction (net < 0 per
                    # kWh), while the comfort floor (15-1000 SEK/kWh) and force_on run
                    # straight over it for pennies. The executor still commands the
                    # element in every planned slot and the hardware thermostat remains
                    # the physical guard.
                    #
                    # The DEMAND RATCHET lives in the adapter: cap includes
                    # absorbed_today x margin, so a big-draw day (measured absorption
                    # above trailing) EXPANDS the cap and boost turns profitable again
                    # within it — the tank's own measured intake is the probe, and the
                    # thermostat terminates the feedback loop physically.
                    if heater.absorb_cap_kwh_per_day is not None and d in water_absorb_overage:
                        if i == 0:
                            # First bucket: subtract the UNCLAMPED measured absorption
                            # (heated_today_kwh is clamped to min_kwh for the floor and
                            # would leave this too generous after a boost-heavy morning).
                            cap_remaining: float = max(
                                0.0,
                                heater.absorb_cap_kwh_per_day - heater.absorbed_today_kwh,
                            )
                        else:
                            cap_remaining = heater.absorb_cap_kwh_per_day
                        overage: Any = water_absorb_overage[d][i]
                        prob += (  # type: ignore[operator]
                            pulp.lpSum(  # type: ignore[reportUnknownMemberType]
                                water_heat[d][t] + (water_boost[d][t] if d in water_boost else 0)
                                for t in day_slot_indices
                            )
                            * kwh_per_slot
                            <= cap_remaining + overage
                        )
                        total_cost.append(absorb_overage_penalty * overage)

                    # Load-priority WTP credit (increment 2): a priority-bearing heater
                    # earns a credit for meeting its daily comfort need (min_kwh_per_day)
                    # at its reservation price, SATIATED at the need (served <= need) so it
                    # never over-heats. The legacy reliability penalty is suppressed for
                    # these heaters (in the objective below), so a low-priority heater
                    # (e.g. spa) skips when energy costs more than its WTP, while a
                    # high-WTP heater fills its need whenever the marginal price allows.
                    # served = min(heated, need); the credit is linear (no integer vars).
                    if (
                        config.load_priority_enabled
                        and d in config.load_priorities
                        and day_min_kwh > 0
                    ):
                        lp = config.load_priorities[d]
                        wtp_d = lp.base_wtp_sek_per_kwh + lp.rank_epsilon_sek_per_kwh
                        if wtp_d != 0.0:
                            safe_dp = d.replace("-", "_").replace(".", "_")
                            served: Any = pulp.LpVariable(  # type: ignore[reportUnknownMemberType]
                                f"wtp_served_{safe_dp}_{i}", lowBound=0.0
                            )
                            # served = min(heated, need): capped at the daily comfort need
                            # (satiation) and at the energy actually heated.
                            prob += served <= day_min_kwh  # type: ignore[operator]
                            prob += served <= (  # type: ignore[operator]
                                pulp.lpSum(water_heat[d][t] for t in day_slot_indices)
                                * kwh_per_slot
                            )
                            total_cost.append(-wtp_d * served)

                # Constraint 2: Per-device soft block breaker (task 2.5)
                if config.water_block_penalty_sek > 0:
                    max_block_slots: int = max(1, int(config.max_block_hours / avg_slot_hours))
                    window_size: int = max_block_slots + 1
                    for t in range(T - window_size + 1):
                        prob += (  # type: ignore[operator]
                            pulp.lpSum(water_heat[d][j] for j in range(t, t + window_size))
                            <= max_block_slots + block_overshoot[d][t]
                        )

                # Constraint 3: Per-device hard spacing constraint (task 2.6)
                if heater.min_spacing_hours > 0 and d in water_start:
                    spacing_slots: int = max(1, int(heater.min_spacing_hours / avg_slot_hours))
                    M: int = spacing_slots
                    for t in range(T):
                        start_idx: int = max(0, t - spacing_slots)
                        prob += (  # type: ignore[operator]
                            pulp.lpSum(water_heat[d][j] for j in range(start_idx, t))
                            + water_start[d][t] * M  # type: ignore[operator]
                            <= M
                        )

            # Build #16: price-gated previous-plan anchor (the plan-stability root fix).
            # For each heater carrying an anchor (its previous-plan ON slots, already
            # wall-clock-mapped to this future_df by the pipeline), reward keeping the
            # block where it was — UNLESS a genuinely cheaper position exists.
            #
            # WHY it stops the walk: on a flat/degenerate price band many block positions
            # are within the solver's gapRel tolerance, so the time-boxed incumbent lands
            # on a different wall-clock hour each replan (the 1e-5 symmetry-breaker is
            # ~3000x too small to bite). The öre-scale anchor makes "stay put" strictly
            # cheaper among those tied positions, so the block holds.
            #
            # WHY it still relocates on a real price change: the PRICE-GATE below is
            # computed in Python from the import-price vector (deterministic, immune to
            # the 120s time-box). If the n cheapest slots beat the anchored slots by MORE
            # than the total bonus on offer, the anchor is DROPPED entirely for that
            # heater and the block moves freely.
            #
            # COLD-SHOWER SAFETY: this only ever adds a REWARD to water_heat[d][t]==1 at a
            # subset of slots. It cannot reduce the daily-minimum sum, so it NEVER lowers
            # day_min / touches the floor (kepler.py day_min_kwh above). It is strictly
            # below the WTP credit and reliability penalty, so real economics dominate.
            anchor_bonus: float = config.water_anchor_bonus_sek_per_slot
            if anchor_bonus > 0:
                # PV-AWARE effective cost of heating one slot's worth of energy at t:
                # PV surplus (pv - load) is consumed first at its opportunity cost (the
                # export price forgone), the remainder is grid import. This mirrors how
                # the objective actually prices a heating slot, so the gate does NOT
                # mis-judge the midday PV plateau — where import price is high but heating
                # is essentially free from surplus — as "expensive" and wrongly drop the
                # anchor there (that plateau is the DOMINANT walk regime). Deterministic
                # (pure Python), so immune to the 120s HiGHS time-box.
                def _slot_heat_cost(t: int, kwh: float) -> float:
                    surplus = max(0.0, slots[t].pv_kwh - slots[t].load_kwh)
                    pv_used = min(surplus, kwh)
                    grid = kwh - pv_used
                    return (
                        pv_used * slots[t].export_price_sek_kwh
                        + grid * slots[t].import_price_sek_kwh
                    )

                for heater in water_heaters:
                    d = heater.id
                    if not heater.anchor_on_slots:
                        continue
                    anchor_in: list[int] = sorted(t for t in heater.anchor_on_slots if 0 <= t < T)
                    if not anchor_in:
                        continue
                    kwh_per_slot_a: float = heater.power_kw * avg_slot_hours
                    costs_sorted: list[float] = sorted(
                        _slot_heat_cost(t, kwh_per_slot_a) for t in range(T)
                    )
                    # Gate each CONTIGUOUS block independently: one relocatable block
                    # must not drop the anchor for the others (nor couple a cross-day pair).
                    blocks: list[list[int]] = []
                    block: list[int] = [anchor_in[0]]
                    for t in anchor_in[1:]:
                        if t == block[-1] + 1:
                            block.append(t)
                        else:
                            blocks.append(block)
                            block = [t]
                    blocks.append(block)
                    for blk in blocks:
                        k = len(blk)
                        anchored_cost = sum(_slot_heat_cost(t, kwh_per_slot_a) for t in blk)
                        cheapest_cost = sum(costs_sorted[:k])
                        bonus_total = anchor_bonus * k
                        if anchored_cost - cheapest_cost > bonus_total:
                            logger.info(
                                "Plan-stability anchor DROPPED for %s block@slot%d: "
                                "anchored=%.3f vs cheapest=%.3f SEK (delta > bonus %.3f) "
                                "— price moved, block free to relocate",
                                d,
                                blk[0],
                                anchored_cost,
                                cheapest_cost,
                                bonus_total,
                            )
                            continue
                        # KEEP: reward staying ON at the anchored slots (negative cost).
                        anchor_reward_terms.append(
                            -anchor_bonus * pulp.lpSum(water_heat[d][t] for t in blk)
                        )
                        logger.info(
                            "Plan-stability anchor KEPT for %s block@slot%d: %d slots, "
                            "bonus %.3f/slot (anchored=%.3f vs cheapest=%.3f SEK)",
                            d,
                            blk[0],
                            k,
                            anchor_bonus,
                            anchored_cost,
                            cheapest_cost,
                        )

        # Effekttariff: monthly peak demand charge on the hourly-mean grid import
        # (timmedelvärde). One peak variable PER CALENDAR MONTH in the horizon: the
        # current month's is floored at the month-to-date baseline (staying under the
        # existing peak is free), later months start from zero (their billing resets).
        # The billed mean always divides by the FULL hour: the in-progress hour gets
        # its already-elapsed import injected as a constant (mid-hour replans would
        # otherwise both miss real peaks and hallucinate phantom ones), and a trailing
        # partial hour under-counts only its unknown remainder — which the next replan
        # re-prices with that import as elapsed/planned.
        peak_cost_term: Any = 0.0
        if config.peak_power_cost_sek_per_kw > 0.0 and T > 0:
            hour_slot_indices: dict[Any, list[int]] = defaultdict(list)
            for t in range(T):
                hour_slot_indices[
                    input_data.slots[t].start_time.replace(minute=0, second=0, microsecond=0)
                ].append(t)
            first_hour_key: Any = min(hour_slot_indices)
            months: dict[tuple[int, int], list[Any]] = defaultdict(list)
            for hour_key in hour_slot_indices:
                months[(hour_key.year, hour_key.month)].append(hour_key)
            horizon_month: tuple[int, int] = (first_hour_key.year, first_hour_key.month)
            peak_terms: list[Any] = []
            for month_key in sorted(months):
                month_baseline_kw: float = (
                    max(0.0, config.peak_power_baseline_kw) if month_key == horizon_month else 0.0
                )
                month_peak_kw: Any = pulp.LpVariable(
                    f"peak_import_kw_{month_key[0]}_{month_key[1]:02d}",
                    lowBound=month_baseline_kw,
                )
                for hour_key in months[month_key]:
                    hour_import: Any = pulp.lpSum(
                        grid_import[t] for t in hour_slot_indices[hour_key]
                    )
                    if hour_key == first_hour_key:
                        hour_import = hour_import + max(0.0, config.peak_hour_elapsed_import_kwh)
                    # full-hour billed mean: kWh in the clock hour <= peak_kw * 1h
                    prob += hour_import <= month_peak_kw  # type: ignore[operator]
                peak_terms.append(
                    config.peak_power_cost_sek_per_kw * (month_peak_kw - month_baseline_kw)
                )
            peak_cost_term = pulp.lpSum(peak_terms)

        # Terminal SoC Target (BIDIRECTIONAL soft constraint)
        # - min_soc violation: HARD penalty (1000 SEK/kWh)
        # - target violation: SOFT penalty (from config, derived from risk_appetite)
        #   * UNDER target: Risk penalty (configurable)
        #   * OVER target: Opportunity cost penalty (same as under)
        # - gap violation: SOFT comfort penalty (Rev K18)
        prob += (  # type: ignore[operator]
            pulp.lpSum(total_cost)
            + peak_cost_term
            + MIN_SOC_PENALTY * pulp.lpSum(soc_violation)
            + MAX_SOC_PENALTY * pulp.lpSum(soc_overshoot)
            + (
                EXPORT_FLOOR_PENALTY * pulp.lpSum(export_floor_violation[t] for t in range(T))
                if export_floor_active
                else 0.0
            )
            + gap_violation_penalty  # Deprecated in K16 (0.0)
            # Per-device block overshoot penalty (task 2.9)
            + (
                pulp.lpSum(block_overshoot[d][t] for d in block_overshoot for t in range(T))
                * config.water_block_penalty_sek
                if water_enabled
                else 0.0
            )
            # Per-device block start penalty (task 2.9)
            + (
                pulp.lpSum(
                    water_start[d][t]
                    # A STATIC-WTP heater prices its worth purely via its per-kWh credit;
                    # the flat block-start penalty (~3 SEK at comfort L3) exceeds a low
                    # tier's ENTIRE daily credit (0.4 WTP x 6 kWh = 2.4 SEK), silently
                    # turning "heats when energy is cheap/surplus" into "never heats".
                    # Waive it for those heaters — their WTP threshold already gates
                    # starts. Dynamic-percentile heaters keep it (consolidation within
                    # the cheap band is exactly what the penalty is for).
                    for d in water_start
                    if not (
                        config.load_priority_enabled
                        and d in config.load_priorities
                        and config.load_priorities[d].dynamic_percentile is None
                    )
                    for t in range(T)
                )
                * config.water_block_start_penalty_sek
                if water_enabled and water_start and config.water_block_start_penalty_sek > 0
                else 0.0
            )
            # Per-device symmetry breaker (task 2.9)
            + (
                pulp.lpSum(water_heat[d][t] * (t * 1e-5) for d in water_heat for t in range(T))
                if water_enabled
                else 0.0
            )
            # Build #16: price-gated previous-plan anchor reward (negative cost). Öre-scale
            # "stay put" bonus that beats the flat-band ties the 1e-5 term is too weak to
            # settle; already price-gated per heater above, so a genuine price change has
            # dropped it. Never touches the daily-minimum floor (reward-only).
            + (pulp.lpSum(anchor_reward_terms) if anchor_reward_terms else 0.0)
            # Per-device reliability penalties (task 2.9)
            + (
                pulp.lpSum(
                    water_min_kwh_violation[d][i]
                    # A STATIC-WTP priority heater prices its comfort need purely via the WTP
                    # credit, so it's excluded from the legacy reliability penalty — that's
                    # what lets a low-priority heater (e.g. spa) skip a day when energy costs
                    # more than its WTP. A DYNAMIC-percentile heater (e.g. the VVB) KEEPS the
                    # reliability floor: its cap always leaves a cheap band each day, so the
                    # floor forces it to meet the daily minimum *in that cheap band* and it can
                    # never silently defer-forever / skip a day. (Belt-and-suspenders: the cap
                    # steers to the cheapest hours, the floor guarantees it actually heats.)
                    for d in water_min_kwh_violation
                    if not (
                        config.load_priority_enabled
                        and d in config.load_priorities
                        and config.load_priorities[d].dynamic_percentile is None
                    )
                    for i in range(len(sorted_days))
                )
                * config.water_reliability_penalty_sek
                if water_enabled
                else 0.0
            )
            # Per-device incentive bucket values: subtract from objective (negative cost = gain)
            - (
                pulp.lpSum(
                    ev_bucket_charged[charger.id][i] * charger.incentive_buckets[i].value_sek
                    for charger in plugged_chargers
                    for i in range(len(charger.incentive_buckets or []))
                )
                if ev_any_enabled
                else 0.0
            )
        )

        # Solve — HiGHS first, CBC fallback, GLPK last resort
        import time

        build_start: float = time.time()
        # Solver setup is fast, but let's track the overhead of calling the solver command
        # Note: pulp.LpProblem construction happened above, so 'build_time' here is mostly
        # just the overhead of writing the LP file in prob.solve()

        used_solver: str = "highs"
        # Per-attempt timer: solve_is_time_boxed, the SOLVER_TIMEOUT mapping and the
        # degraded-quality WARNING all reason about the solver that produced
        # prob.status, so they must see the LAST attempt's duration — not the
        # cumulative chain (a slow HiGHS malfunction followed by a fast proven GLPK
        # solve must not be labeled time-boxed). Reset AFTER constructing each
        # solver command: on hosts without highspy the exception comes from
        # pulp.HiGHS(...) itself and construction must not count against the clock.
        attempt_start: float = build_start
        try:
            # HiGHS FIRST (in-process highspy). Measured on the live 107-slot model:
            # proves optimality in ~13.6s SINGLE-threaded where CBC needs 12 threads
            # for 15.7s and can't prove in 120s on one thread — decisive on the
            # 2-vCPU production VM. `threads` is deliberately OMITTED: it makes no
            # measured difference for the HiGHS MIP search here, and HiGHS's
            # process-global scheduler returns kNotset (-> NotSolved) if a later
            # solve in the same process passes a DIFFERENT thread count.
            # Missing highspy => prob.solve raises PulpSolverError -> CBC below.
            solver_cmd: Any = pulp.HiGHS(  # type: ignore[reportUnknownMemberType]
                msg=False,
                timeLimit=SOLVER_TIME_LIMIT_S,
                gapRel=0.01,
            )
            attempt_start = time.time()
            prob.solve(solver_cmd)  # type: ignore[reportUnknownMemberType]
            _highs_duration: float = time.time() - attempt_start
            # Any NotSolved falls through to CBC — prob.solve does NOT raise for it.
            # A FAST NotSolved is a solver malfunction (e.g. the global-scheduler
            # thread mismatch, a load error). A SLOW NotSolved is a real no-incumbent
            # timeout; before 2026-07-22 it skipped the fallback and went straight to
            # the SOLVER_TIMEOUT mapping — the 2026-07-15 freeze ran 7 h on a stale
            # plan that way. CBC is the weaker solver on this box, so it is a safety
            # net, not a rescue: its incumbent still passes the comfort-floor
            # tripwire below, so worst case remains "keep the previous plan", never
            # a garbage plan shipped.
            if prob.status == pulp.LpStatusNotSolved:  # type: ignore[reportUnknownMemberType]
                if _highs_duration < 0.9 * SOLVER_TIME_LIMIT_S:
                    raise RuntimeError(
                        f"HiGHS returned NotSolved in {_highs_duration:.1f}s "
                        f"(<0.9x budget) — treating as solver malfunction"
                    )
                logger.warning(
                    "HiGHS found NO incumbent within its %ds budget (%.1fs) — "
                    "retrying with CBC as a safety net",
                    SOLVER_TIME_LIMIT_S,
                    _highs_duration,
                )
                raise RuntimeError(
                    f"HiGHS no-incumbent timeout ({_highs_duration:.1f}s of "
                    f"{SOLVER_TIME_LIMIT_S}s) — retrying with CBC"
                )
        except Exception as highs_exc:
            # Expected on hosts without highspy — INFO, not WARNING.
            logger.info("HiGHS unavailable or failed (%s) — solving with CBC", highs_exc)
            used_solver = "cbc"
            try:
                # CBC before GLPK. GLPK-first caused the 2026-07-05 cold-tank
                # incident: on the grown model (~3.4k vars) GLPK hit its 30 s
                # ceiling every single run and returned an incumbent with ZERO
                # water heating while reporting "Optimal" — silently shipping
                # garbage plans. pulp's bundled CBC is a far stronger MILP solver.
                # GLPK stays as the last resort for images where the bundled CBC
                # binary can't run.
                # Generous limit: we replan every 15 min — a 2-min solve is
                # acceptable, a silently-degraded plan is not. gapRel lets CBC stop
                # at a proven 1%-of-optimal bound instead of chasing the last
                # epsilon. threads: the bundled CBC honors it (measured 4.3x with
                # 8 threads); on a 1-2 vCPU box it is harmless.
                solver_cmd = pulp.PULP_CBC_CMD(
                    msg=False,
                    timeLimit=SOLVER_TIME_LIMIT_S,
                    gapRel=0.01,
                    threads=max(1, os.cpu_count() or 1),
                )
                attempt_start = time.time()
                prob.solve(solver_cmd)  # type: ignore[reportUnknownMemberType]
            except Exception:
                used_solver = "glpk"
                solver_cmd = pulp.GLPK_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT_S)
                attempt_start = time.time()
                prob.solve(solver_cmd)  # type: ignore[reportUnknownMemberType]

        solve_end: float = time.time()

        # Extract Results
        status: str = pulp.LpStatus[prob.status]  # type: ignore[index]
        is_optimal: bool = status == "Optimal"

        # Log Performance Metrics
        # solve_duration covers ONLY the final solver attempt (the one that produced
        # prob.status); total_duration covers the whole fallback chain including the
        # LP-write overhead inside prob.solve().
        solve_duration: float = solve_end - attempt_start
        total_duration: float = solve_end - build_start
        # Count stats
        var_count: int = len(prob.variables())  # type: ignore[reportUnknownMemberType,arg-type]
        const_count: int = len(prob.constraints)  # type: ignore[reportUnknownMemberType,arg-type]

        # A solve that consumed (almost) the whole time budget returned whatever
        # incumbent it had — possibly far from optimal — even when the status reads
        # "Optimal" (GLPK notoriously does this). Never let that pass silently: the
        # 2026-07-05 incident shipped zero-water plans for hours this way.
        if solve_duration >= 0.9 * SOLVER_TIME_LIMIT_S:
            logger.warning(
                "Kepler solve consumed %.1fs of its %ds budget — plan quality may be "
                "degraded (status: %s). Consider a stronger solver or smaller model.",
                solve_duration,
                SOLVER_TIME_LIMIT_S,
                status,
            )

        # "Optimal" from PuLP does NOT mean proven optimal (see solve_is_time_boxed).
        # A time-boxed incumbent may be arbitrarily bad — the trivial "heat nothing"
        # plan is always feasible and is what both the 2026-07-05 GLPK incident and
        # early CBC incumbents ship. Domain tripwire: a sane plan never violates the
        # water comfort floors (they carry water_reliability_penalty_sek ~1000/kWh),
        # so an incumbent that does is garbage — reject it and keep the last good
        # plan rather than shipping cold showers. An incumbent that PASSES the
        # tripwire is economically slightly degraded (~1-3 SEK/day measured) but
        # qualitatively sound: ship it, honestly labeled.
        # getattr: sol_status is standard PuLP, but test doubles / exotic wrappers
        # may lack it — default to "proven" (old behavior) rather than false-flag.
        _sol_status: int = int(getattr(prob, "sol_status", _LP_SOLUTION_OPTIMAL))  # type: ignore[arg-type]
        if solve_is_time_boxed(is_optimal, _sol_status, used_solver, solve_duration):  # type: ignore[arg-type]
            floor_violation_kwh: float = 0.0
            for d in water_min_kwh_violation:
                if (
                    config.load_priority_enabled
                    and d in config.load_priorities
                    and config.load_priorities[d].dynamic_percentile is None
                ):
                    continue  # static-WTP heaters may legitimately skip a day
                for i in range(len(sorted_days)):
                    _v: Any = pulp.value(water_min_kwh_violation[d][i])  # type: ignore[arg-type]
                    if _v is not None:
                        floor_violation_kwh += max(0.0, float(_v))  # type: ignore[arg-type]
            if floor_violation_kwh > 0.5:
                logger.error(
                    "Kepler returned a time-boxed incumbent that violates the water "
                    "comfort floors by %.2f kWh — rejecting the plan (executor keeps "
                    "the previous one).",
                    floor_violation_kwh,
                )
                _dump_path = _dump_failed_solve_instance(
                    input_data,
                    config,
                    reason="tripwire_comfort_floor_violation",
                    status=str(status),  # type: ignore[reportUnknownArgumentType]
                    sol_status=_sol_status,
                    used_solver=used_solver,
                    solve_duration_s=solve_duration,
                    chain_duration_s=total_duration,
                    extra={"floor_violation_kwh": round(floor_violation_kwh, 2)},
                )
                raise PlannerError(
                    code=PlannerErrorCode.SOLVER_TIMEOUT,
                    details={
                        "solver_status": status,
                        "solve_duration_s": round(solve_duration, 3),
                        "chain_duration_s": round(total_duration, 3),
                        "reason": "time-boxed incumbent violates water comfort floors",
                        "floor_violation_kwh": round(floor_violation_kwh, 2),
                        "dump_path": _dump_path,
                    },
                )
            status = f"{status} (time-boxed incumbent, gap unproven)"
            logger.warning(
                "Kepler hit its %ds budget and returned an UNPROVEN incumbent "
                "(passed the comfort-floor tripwire). Shipping it labeled '%s'; "
                "objective may be a few SEK/day off optimal.",
                SOLVER_TIME_LIMIT_S,
                status,
            )

        if not is_optimal:
            prob.writeLP("kepler_debug.lp")  # type: ignore[reportUnknownMemberType]
            print(f"Solver failed: {status}. LP written to kepler_debug.lp")

            # Persist the exact failing instance BEFORE it is discarded with the
            # error, so it can be replayed offline (see _dump_failed_solve_instance).
            _fail_dump_path = _dump_failed_solve_instance(
                input_data,
                config,
                reason="not_optimal",
                status=str(status),  # type: ignore[reportUnknownArgumentType]
                sol_status=_sol_status,
                used_solver=used_solver,
                solve_duration_s=solve_duration,
                chain_duration_s=total_duration,
                extra={"var_count": var_count, "const_count": const_count},
            )

            # Map PuLP status to structured PlannerError. solve_duration_s is the
            # final attempt only; chain_duration_s is the whole fallback chain.
            details = {
                "solver_status": str(status),
                "solve_duration_s": round(solve_duration, 3),
                "chain_duration_s": round(total_duration, 3),
                "dump_path": _fail_dump_path,
            }
            if prob.status == pulp.LpStatusInfeasible:  # type: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                raise PlannerError(code=PlannerErrorCode.SOLVER_INFEASIBLE, details=details)
            elif solve_duration >= 0.95 * SOLVER_TIME_LIMIT_S:
                # Solvers check the clock between nodes and exit just UNDER the
                # limit (119.939s of 120) — a >= limit comparison never fires.
                raise PlannerError(code=PlannerErrorCode.SOLVER_TIMEOUT, details=details)
            elif prob.status == pulp.LpStatusUndefined:  # type: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                raise PlannerError(code=PlannerErrorCode.SOLVER_UNDEFINED, details=details)
            else:
                # A fast NotSolved that survived the whole fallback chain used to
                # slip through here and return a KeplerResult with EMPTY slots
                # (empirically reproduced) — the executor would happily act on a
                # plan that schedules nothing. Fail loud instead. Unconditional
                # else: also catches LpStatusUnbounded and any future status, so
                # no non-optimal outcome can ship an empty plan silently
                # (details carries solver_status for diagnosis).
                raise PlannerError(code=PlannerErrorCode.SOLVER_UNDEFINED, details=details)  # type: ignore[reportUnknownArgumentType]

        result_slots: list[KeplerResultSlot] = []
        final_total_cost: float = 0.0

        if is_optimal:
            for t in range(T):
                s: Any = slots[t]
                h: float = slot_hours[t]

                c_val: float | None = pulp.value(charge[t])  # type: ignore[assignment]
                d_val: float | None = pulp.value(discharge[t])  # type: ignore[assignment]
                i_val: float | None = pulp.value(grid_import[t])  # type: ignore[assignment]
                e_val: float | None = pulp.value(grid_export[t])  # type: ignore[assignment]
                soc_val: float | None = pulp.value(soc[t + 1])  # type: ignore[assignment]

                # Per-device water heating results (task 2.10)
                water_heater_results: dict[str, float] = {}
                total_water_kw: float = 0.0
                water_heating_boost: dict[str, bool] = {}
                if water_enabled:
                    for heater in water_heaters:
                        d_wh = heater.id
                        w_val: float | None = pulp.value(water_heat[d_wh][t])  # type: ignore[assignment]
                        b_val: float | None = (
                            pulp.value(water_boost[d_wh][t])  # type: ignore[assignment]
                            if d_wh in water_boost
                            else None
                        )
                        device_kw: float = (
                            heater.power_kw if w_val is not None and w_val > 0.5 else 0.0
                        )
                        # Boost is active when boost variable is on (even if normal is also on)
                        is_boost: bool = b_val is not None and b_val > 0.5
                        if is_boost:
                            device_kw = heater.power_kw
                            water_heating_boost[d_wh] = True
                        water_heater_results[d_wh] = device_kw
                        total_water_kw += device_kw
                w_kw: float = total_water_kw  # aggregate (backward compat)

                # Per-device EV charging results
                ev_charger_results: dict[str, float] = {}
                total_ev_kw: float = 0.0
                for charger in plugged_chargers:
                    d = charger.id
                    ev_val: float | None = pulp.value(ev_energy[d][t])  # type: ignore[assignment]
                    device_kw: float = ev_val / h if ev_val is not None and h > 0 else 0.0
                    ev_charger_results[d] = device_kw
                    total_ev_kw += device_kw
                ev_kw = total_ev_kw

                # Per-device deferrable-load results
                deferrable_load_results: dict[str, float] = {}
                total_defl_kw: float = 0.0
                for load in scheduled_defl:
                    run_val: float | None = pulp.value(_defl_run_expr(load.id, t))  # type: ignore[assignment]
                    device_kw = (
                        defl_energy_per_slot[load.id] / h
                        if run_val is not None and run_val > 0.5 and h > 0
                        else 0.0
                    )
                    deferrable_load_results[load.id] = device_kw
                    total_defl_kw += device_kw

                # Per-sink ladder results (first sink mirrored into the legacy
                # custom_entity_active flag for old consumers).
                sink_states: dict[str, bool] = {}
                for sink in sinks:
                    _sink_var: Any = sink_active[sink.id][t]
                    if isinstance(_sink_var, int):
                        sink_states[sink.id] = False
                    else:
                        _sink_val: float | None = pulp.value(_sink_var)  # type: ignore[assignment]
                        sink_states[sink.id] = _sink_val is not None and _sink_val > 0.5

                wear: float = (
                    (c_val + d_val) * config.wear_cost_sek_per_kwh * 0.5
                    if c_val is not None and d_val is not None
                    else 0.0
                )
                cost: float = (
                    (i_val * s.import_price_sek_kwh) - (e_val * s.export_price_sek_kwh) + wear
                    if i_val is not None and e_val is not None
                    else 0.0
                )
                final_total_cost += cost

                result_slots.append(
                    KeplerResultSlot(
                        start_time=s.start_time,
                        end_time=s.end_time,
                        charge_kwh=c_val,  # type: ignore[arg-type]
                        discharge_kwh=d_val,  # type: ignore[arg-type]
                        grid_import_kwh=i_val,  # type: ignore[arg-type]
                        grid_export_kwh=e_val,  # type: ignore[arg-type]
                        soc_kwh=soc_val,  # type: ignore[arg-type]
                        cost_sek=cost,
                        import_price_sek_kwh=s.import_price_sek_kwh,
                        export_price_sek_kwh=s.export_price_sek_kwh,
                        water_heat_kw=w_kw,
                        water_heater_results=water_heater_results,
                        water_heating_boost=water_heating_boost,
                        custom_entity_active=(
                            sink_states.get(sinks[0].id, False) if sinks else False
                        ),
                        sink_states=sink_states,
                        ev_charge_kw=ev_kw,
                        ev_charger_results=ev_charger_results,
                        deferrable_load_kw=total_defl_kw,
                        deferrable_load_results=deferrable_load_results,
                        is_optimal=True,
                    )
                )

            # Update the log with correct cost (since we calculated it in the loop)
            logger_perf = logging.getLogger("darkstar.performance")
            logger_perf.setLevel(logging.INFO)  # Ensure we see it
            logger_perf.info(
                "Kepler Solved: %d slots in %.3fs via %s (Vars: %d, Const: %d) | Cost: %.2f SEK",
                T,
                total_duration,
                used_solver,
                var_count,
                const_count,
                final_total_cost,
            )

        objective_cost: float | None = None
        if is_optimal:
            try:
                from typing import cast as _cast

                _obj_val = _cast("float | None", pulp.value(prob.objective))  # type: ignore[arg-type]
                objective_cost = float(_obj_val) if _obj_val is not None else None
            except (TypeError, ValueError):
                objective_cost = None

        return KeplerResult(
            slots=result_slots,
            total_cost_sek=final_total_cost,
            is_optimal=is_optimal,
            status_msg=status,
            objective_cost_sek=objective_cost,
        )
