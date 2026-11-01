"""Battery-only planner: a small MILP over battery charge/discharge and grid flows.

Two users, one model:

* **Fallback planner** (``mode="live"``): when Kepler times out, fails, or its
  time-boxed incumbent is rejected by the comfort-floor tripwire, the planning
  service can ship this plan instead of an hours-old one. Only the battery is
  scheduled — water heaters, EVs, excess-PV sinks and deferrable loads are left OFF
  in the plan (their own executor-side floors/servos keep running). Policy terms
  that are cheap to model are kept so the plan behaves like a degraded Kepler plan,
  not a different strategy: min-SoC floor incl. the below-minimum recovery window,
  soft max SoC, terminal SoC target, stored-energy value, export threshold, export
  toggle and export SoC floor, power/inverter limits, curtailment.
* **Hindsight benchmark** (``mode="benchmark"``): battery-only optimum on recorded
  load/PV/prices for a finished day, with a hard terminal stock (so the comparison
  with what the battery actually did is fair) and no policy terms.

Binaries exist only for mutual exclusion (battery charge vs discharge, grid import
vs export): 2·T of them on ~100-200 slots, which HiGHS proves in well under a second
(measured: 192 slots ~0.1 s on a laptop). That is the whole point — this must solve
in < 2 s on the 2-vCPU box where Kepler can need 240 s.

Pure computation: no I/O, no config reading.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import pulp  # type: ignore[import]

from planner.errors import PlannerError, PlannerErrorCode
from planner.solver.kepler import min_soc_floor_kwh
from planner.solver.types import KeplerConfig, KeplerInput, KeplerResult, KeplerResultSlot

logger = logging.getLogger("darkstar.planner.battery_lp")

# Hard ceiling for the fallback solve. The model is tiny; anything near this means
# something is badly wrong and we would rather fail than stall the planner further.
BATTERY_LP_TIME_LIMIT_S = 10.0

_MIN_SOC_PENALTY = 1000.0  # SEK/kWh below the per-state floor (mirrors Kepler)
_MAX_SOC_PENALTY = 1000.0  # SEK/kWh above max_soc (mirrors Kepler)
_EXPORT_FLOOR_PENALTY = 1000.0
_SHEDDING_PENALTY = 10000.0  # unmet load: only to stay feasible under an import cap
# Tie-breaker: prefer acting earlier among equal-cost schedules (deterministic plans).
_TIME_EPS = 1e-6


@dataclass
class BenchmarkTerms:
    """Hindsight-mode extras: fixed end stock and slots where discharge is forbidden."""

    terminal_soc_kwh: float
    no_discharge_slots: list[bool] | None = None


def _solve(prob: Any, time_limit_s: float) -> str:
    """HiGHS (in-process) first, bundled CBC as fallback. Returns the solver name."""
    try:
        cmd: Any = pulp.HiGHS(msg=False, timeLimit=time_limit_s)  # type: ignore[reportUnknownMemberType]
        if not cmd.available():
            raise RuntimeError("highspy not available")
        prob.solve(cmd)  # type: ignore[reportUnknownMemberType]
        if prob.status == pulp.LpStatusNotSolved:  # type: ignore[reportUnknownMemberType]
            raise RuntimeError("HiGHS returned NotSolved")
        return "highs"
    except Exception as exc:
        logger.debug("battery_lp: HiGHS unavailable/failed (%s) — using CBC", exc)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))  # type: ignore[reportUnknownMemberType]
        return "cbc"


def solve_battery_only(
    input_data: KeplerInput,
    config: KeplerConfig,
    *,
    mode: Literal["live", "benchmark"] = "live",
    benchmark: BenchmarkTerms | None = None,
    time_limit_s: float = BATTERY_LP_TIME_LIMIT_S,
) -> KeplerResult:
    """Solve the battery-only schedule. Raises PlannerError when no plan is found."""
    slots = input_data.slots
    T = len(slots)
    if T == 0:
        return KeplerResult(slots=[], total_cost_sek=0.0, is_optimal=True, status_msg="No slots")
    if mode == "benchmark" and benchmark is None:
        raise ValueError("benchmark mode needs BenchmarkTerms")
    live = mode == "live"

    hours = [max((s.end_time - s.start_time).total_seconds() / 3600.0, 0.0) for s in slots]
    cap = config.capacity_kwh
    eta_c = config.charge_efficiency if config.charge_efficiency > 0 else 1.0
    eta_d = config.discharge_efficiency if config.discharge_efficiency > 0 else 1.0
    min_soc = cap * config.min_soc_percent / 100.0
    max_soc = cap * config.max_soc_percent / 100.0
    initial = max(0.0, min(cap, input_data.initial_soc_kwh))

    if live:
        floors = min_soc_floor_kwh(hours, initial, min_soc, cap)
    else:
        assert benchmark is not None
        # A recorded day may start or end outside the configured window.
        ends = (initial, benchmark.terminal_soc_kwh)
        min_soc = min(min_soc, *ends)
        max_soc = max(max_soc, *ends)
        floors = [min_soc] * (T + 1)

    prob: Any = pulp.LpProblem("BatteryOnly", pulp.LpMinimize)
    R = range(T)
    ch: Any = pulp.LpVariable.dicts("ch", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    dis: Any = pulp.LpVariable.dicts("dis", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    imp: Any = pulp.LpVariable.dicts("imp", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    exp: Any = pulp.LpVariable.dicts("exp", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    curt: Any = pulp.LpVariable.dicts("curt", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    shed: Any = pulp.LpVariable.dicts("shed", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    soc: Any = pulp.LpVariable.dicts("soc", range(T + 1), lowBound=0.0, upBound=cap)  # type: ignore[reportUnknownMemberType]
    charging: Any = pulp.LpVariable.dicts("charging", R, cat="Binary")  # type: ignore[reportUnknownMemberType]
    exporting: Any = pulp.LpVariable.dicts("exporting", R, cat="Binary")  # type: ignore[reportUnknownMemberType]

    soc_under: Any = pulp.LpVariable.dicts("soc_under", range(T + 1), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    soc_over: Any = pulp.LpVariable.dicts("soc_over", range(T + 1), lowBound=0.0)  # type: ignore[reportUnknownMemberType]
    export_floor_viol: Any = pulp.LpVariable.dicts("exp_floor_viol", R, lowBound=0.0)  # type: ignore[reportUnknownMemberType]

    export_floor_active = (
        live and config.enable_export and config.export_floor_soc_percent is not None
    )
    export_floor_kwh = cap * float(config.export_floor_soc_percent or 0.0) / 100.0
    export_threshold = config.export_threshold_sek_per_kwh if live else 0.0
    curtail_penalty = config.curtailment_penalty_sek if live else 0.0

    cost_terms: list[Any] = []
    prob += soc[0] == initial
    for t in R:
        s = slots[t]
        h = hours[t]
        max_ch = config.max_charge_power_kw * h
        max_dis = config.max_discharge_power_kw * h
        if config.max_inverter_ac_kw is not None:
            frac = config.hybrid_pv_fraction if config.hybrid_pv_fraction is not None else 1.0
            max_dis = min(max_dis, max(0.0, config.max_inverter_ac_kw * h - s.pv_kwh * frac))
        grid_cap_imp = (
            config.max_import_power_kw * h if config.max_import_power_kw is not None else None
        )
        grid_cap_exp = (
            config.max_export_power_kw * h if config.max_export_power_kw is not None else None
        )
        # Big-M for the import/export exclusion: the most either can physically be.
        big_m = (
            s.load_kwh
            + s.pv_kwh
            + (config.max_charge_power_kw + config.max_discharge_power_kw) * h
            + 1.0
        )

        prob += s.load_kwh + ch[t] + exp[t] + curt[t] == s.pv_kwh + dis[t] + imp[t] + shed[t]
        prob += soc[t + 1] == soc[t] + ch[t] * eta_c - dis[t] / eta_d
        prob += curt[t] <= s.pv_kwh
        prob += shed[t] <= s.load_kwh
        prob += ch[t] <= max_ch * charging[t]
        prob += dis[t] <= max_dis * (1 - charging[t])
        prob += exp[t] <= (grid_cap_exp if grid_cap_exp is not None else big_m) * exporting[t]
        prob += imp[t] <= (grid_cap_imp if grid_cap_imp is not None else big_m) * (1 - exporting[t])
        if live and not config.enable_export:
            prob += exp[t] == 0
        if not live:
            assert benchmark is not None
            prob += curt[t] == 0
            prob += shed[t] == 0
            if benchmark.no_discharge_slots and benchmark.no_discharge_slots[t]:
                prob += dis[t] == 0
        if export_floor_active:
            prob += soc[t + 1] >= (
                export_floor_kwh * exporting[t]
                + floors[t + 1] * (1 - exporting[t])
                - export_floor_viol[t]
            )

        cost_terms.append(
            imp[t] * s.import_price_sek_kwh
            - exp[t] * (s.export_price_sek_kwh - export_threshold)
            + (ch[t] + dis[t]) * config.wear_cost_sek_per_kwh * 0.5
            + curt[t] * curtail_penalty
            + shed[t] * _SHEDDING_PENALTY
            + (ch[t] + dis[t]) * _TIME_EPS * t
        )

    for k in range(T + 1):
        if live:
            prob += soc[k] >= floors[k] - soc_under[k]
            prob += soc[k] <= max_soc + soc_over[k]
        else:
            prob += soc[k] >= floors[k]
            prob += soc[k] <= max_soc

    objective: Any = pulp.lpSum(cost_terms)
    if live:
        objective += _MIN_SOC_PENALTY * pulp.lpSum(soc_under.values())
        objective += _MAX_SOC_PENALTY * pulp.lpSum(soc_over.values())
        if export_floor_active:
            objective += _EXPORT_FLOOR_PENALTY * pulp.lpSum(export_floor_viol.values())
        if config.target_soc_kwh is not None and config.target_soc_penalty_sek > 0:
            under: Any = pulp.LpVariable("target_under", lowBound=0.0)  # type: ignore[reportUnknownMemberType]
            prob += soc[T] >= config.target_soc_kwh - under
            objective += config.target_soc_penalty_sek * under
        if config.battery_value_sek_per_kwh > 0:
            objective += -config.battery_value_sek_per_kwh * soc[T]
    else:
        assert benchmark is not None
        prob += soc[T] == benchmark.terminal_soc_kwh
    prob += objective

    started = time.monotonic()
    used = _solve(prob, time_limit_s)
    duration = time.monotonic() - started
    status = str(pulp.LpStatus[prob.status])  # type: ignore[index]
    sol_status = int(getattr(prob, "sol_status", pulp.LpSolutionOptimal))  # type: ignore[arg-type]
    if status != "Optimal":
        code = (
            PlannerErrorCode.SOLVER_INFEASIBLE
            if prob.status == pulp.LpStatusInfeasible  # type: ignore[reportUnknownMemberType]
            else PlannerErrorCode.SOLVER_UNDEFINED
        )
        raise PlannerError(
            code=code,
            details={
                "solver_status": status,
                "solver": f"battery_lp/{used}",
                "solve_duration_s": round(duration, 3),
            },
        )

    def val(v: Any) -> float:
        x = pulp.value(v)  # type: ignore[arg-type]
        return max(0.0, float(x)) if x is not None else 0.0  # type: ignore[arg-type]

    out: list[KeplerResultSlot] = []
    total = 0.0
    for t in R:
        s = slots[t]
        c, d, i, e = val(ch[t]), val(dis[t]), val(imp[t]), val(exp[t])
        cost = i * s.import_price_sek_kwh - e * s.export_price_sek_kwh
        cost += (c + d) * config.wear_cost_sek_per_kwh * 0.5
        total += cost
        out.append(
            KeplerResultSlot(
                start_time=s.start_time,
                end_time=s.end_time,
                charge_kwh=c,
                discharge_kwh=d,
                grid_import_kwh=i,
                grid_export_kwh=e,
                soc_kwh=min(cap, val(soc[t + 1])),
                cost_sek=cost,
                import_price_sek_kwh=s.import_price_sek_kwh,
                export_price_sek_kwh=s.export_price_sek_kwh,
                is_optimal=True,
            )
        )
    proven = sol_status == pulp.LpSolutionOptimal
    logger.info(
        "battery_lp (%s) solved %d slots in %.3fs via %s%s | cost %.2f SEK",
        mode,
        T,
        duration,
        used,
        "" if proven else " (unproven incumbent)",
        total,
    )
    try:
        objective_cost: float | None = float(pulp.value(prob.objective))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        objective_cost = None
    return KeplerResult(
        slots=out,
        total_cost_sek=total,
        is_optimal=True,
        status_msg=f"Optimal (battery_lp/{used}{'' if proven else ', unproven'})",
        solve_time_ms=duration * 1000.0,
        objective_cost_sek=objective_cost,
    )
