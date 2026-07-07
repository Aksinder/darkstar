"""Build #9 solver hardening tests.

Covers the two halves of the 2026-07-05 "garbage labeled Optimal" fix:
- solve_is_time_boxed: PuLP relabels CBC time-limit incumbents as "Optimal";
  prob.sol_status is the only honest signal (GLPK needs a wall-clock check).
- The comfort-floor tripwire: a time-boxed incumbent that violates the water
  daily minimums is rejected (PlannerError) instead of shipped.
- water_hourly_blocks: hourly decision blocks keep water constant within each
  wall-clock hour (perf), and stay off by default so sub-hour semantics hold.
"""

from datetime import datetime, timedelta

import pulp
import pytest

from planner.errors import PlannerError, PlannerErrorCode
from planner.solver.kepler import KeplerSolver, solve_is_time_boxed
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    WaterHeaterInput,
)

PROVEN = pulp.constants.LpSolutionOptimal  # 1
INCUMBENT = pulp.constants.LpSolutionIntegerFeasible  # 2


def _quarter_slots(n: int, prices: list[float] | None = None) -> list[KeplerInputSlot]:
    """N 15-minute slots starting at midnight."""
    base = datetime(2026, 1, 15, 0, 0)
    out: list[KeplerInputSlot] = []
    for i in range(n):
        s = base + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=0.1,
                pv_kwh=0.0,
                import_price_sek_kwh=prices[i] if prices else 1.0,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def _cfg(heaters: list[WaterHeaterInput], **overrides) -> KeplerConfig:
    kwargs: dict = {
        "capacity_kwh": 10.0,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "wear_cost_sek_per_kwh": 0.0,
        "water_heaters": heaters,
        "water_reliability_penalty_sek": 1000.0,
    }
    kwargs.update(overrides)
    return KeplerConfig(**kwargs)


class TestSolveIsTimeBoxed:
    def test_cbc_time_limit_incumbent_is_flagged(self):
        # The exact live case: 119.939s of a 120s budget, LpStatus "Optimal",
        # sol_status = IntegerFeasible (CBC said "Stopped on time").
        assert solve_is_time_boxed(True, INCUMBENT, "cbc", 119.939, 120) is True

    def test_cbc_incumbent_flagged_even_when_fast(self):
        # sol_status is authoritative for CBC regardless of the clock.
        assert solve_is_time_boxed(True, INCUMBENT, "cbc", 12.0, 120) is True

    def test_cbc_gap_tolerance_stop_is_trusted(self):
        # "Optimal solution found (within gap tolerance)" -> sol_status proven,
        # even if it took nearly the whole budget.
        assert solve_is_time_boxed(True, PROVEN, "cbc", 119.9, 120) is False

    def test_cbc_fast_proven_is_trusted(self):
        assert solve_is_time_boxed(True, PROVEN, "cbc", 3.8, 120) is False

    def test_highs_time_limit_incumbent_is_flagged(self):
        # HiGHS kTimeLimit with incumbent -> (Optimal, IntegerFeasible), same
        # convention as CBC (verified against pulp 3.3.2 highs_api.py).
        assert solve_is_time_boxed(True, INCUMBENT, "highs", 119.9, 120) is True

    def test_highs_incumbent_flagged_even_when_fast(self):
        assert solve_is_time_boxed(True, INCUMBENT, "highs", 5.0, 120) is True

    def test_highs_gap_tolerance_stop_is_trusted(self):
        # gapRel stop reports kOptimal -> (Optimal, Optimal), trusted even at
        # nearly the full budget — same as CBC's gapRel convention.
        assert solve_is_time_boxed(True, PROVEN, "highs", 119.9, 120) is False

    def test_highs_fast_proven_is_trusted(self):
        assert solve_is_time_boxed(True, PROVEN, "highs", 13.6, 120) is False

    def test_glpk_full_budget_is_flagged(self):
        # GLPK's wrapper synthesizes sol_status from status (always looks
        # proven) — only the wall clock can catch its time-boxed incumbents.
        assert solve_is_time_boxed(True, PROVEN, "glpk", 119.9, 120) is True

    def test_glpk_fast_is_trusted(self):
        assert solve_is_time_boxed(True, PROVEN, "glpk", 12.0, 120) is False

    def test_not_optimal_is_never_time_boxed(self):
        # Non-Optimal statuses flow to the existing PlannerError mapping.
        assert solve_is_time_boxed(False, INCUMBENT, "cbc", 119.9, 120) is False
        assert solve_is_time_boxed(False, INCUMBENT, "highs", 119.9, 120) is False


class TestComfortFloorTripwire:
    """A time-boxed incumbent violating the water floors must be rejected."""

    def _underpowered_setup(self):
        # 8 slots (2h) with a 0.5 kW heater and min 6 kWh/day: even heating
        # every slot yields 0.25 kWh -> a guaranteed ~5.75 kWh floor violation
        # in EVERY solution, including the true optimum.
        heater = WaterHeaterInput(
            id="vvb",
            power_kw=0.5,
            min_kwh_per_day=6.0,
            max_hours_between_heating=0.0,
            min_spacing_hours=0.0,
            heated_today_kwh=0.0,
        )
        return KeplerInput(slots=_quarter_slots(8), initial_soc_kwh=5.0), _cfg([heater])

    def test_time_boxed_incumbent_with_floor_violation_is_rejected(self, monkeypatch):
        input_data, config = self._underpowered_setup()
        monkeypatch.setattr("planner.solver.kepler.solve_is_time_boxed", lambda *a, **k: True)
        with pytest.raises(PlannerError) as exc:
            KeplerSolver().solve(input_data, config)
        assert exc.value.code == PlannerErrorCode.SOLVER_TIMEOUT
        assert "comfort floors" in str(exc.value.details.get("reason", ""))
        assert exc.value.details.get("floor_violation_kwh", 0) > 0.5

    def test_proven_solve_with_same_violation_is_accepted(self):
        # Control: the identical model solved to proven optimality ships fine —
        # the tripwire only applies to unproven time-boxed incumbents.
        input_data, config = self._underpowered_setup()
        result = KeplerSolver().solve(input_data, config)
        assert result.is_optimal
        assert "time-boxed" not in result.status_msg

    def test_time_boxed_incumbent_without_violation_ships_labeled(self, monkeypatch):
        # A feasible-comfort incumbent is shipped, but honestly labeled.
        heater = WaterHeaterInput(
            id="vvb",
            power_kw=3.0,
            min_kwh_per_day=1.5,
            max_hours_between_heating=0.0,
            min_spacing_hours=0.0,
            heated_today_kwh=0.0,
        )
        input_data = KeplerInput(slots=_quarter_slots(8), initial_soc_kwh=5.0)
        config = _cfg([heater])
        monkeypatch.setattr("planner.solver.kepler.solve_is_time_boxed", lambda *a, **k: True)
        result = KeplerSolver().solve(input_data, config)
        assert result.is_optimal
        assert "time-boxed incumbent" in result.status_msg


class TestSolverFallbackChain:
    """Build #10: HiGHS-first chain with honest malfunction handling."""

    def _simple_setup(self):
        heater = WaterHeaterInput(
            id="vvb",
            power_kw=3.0,
            min_kwh_per_day=1.5,
            max_hours_between_heating=0.0,
            min_spacing_hours=0.0,
            heated_today_kwh=0.0,
        )
        return KeplerInput(slots=_quarter_slots(8), initial_soc_kwh=5.0), _cfg([heater])

    def test_highs_fast_not_solved_falls_back_to_cbc(self, monkeypatch):
        # A fast NotSolved from HiGHS (e.g. the global-scheduler thread mismatch)
        # does NOT raise from prob.solve — kepler must detect it and RE-SOLVE with
        # CBC instead of shipping an empty plan.
        class _MalfunctioningHiGHS:
            def __init__(self, *args, **kwargs):
                pass

            def actualSolve(self, lp, **kwargs):
                lp.status = pulp.LpStatusNotSolved
                lp.sol_status = pulp.constants.LpSolutionNoSolutionFound
                return lp.status

        monkeypatch.setattr(pulp, "HiGHS", _MalfunctioningHiGHS)
        input_data, config = self._simple_setup()
        result = KeplerSolver().solve(input_data, config)
        assert result.is_optimal
        assert "time-boxed" not in result.status_msg
        assert len(result.slots) == 8

    def test_fast_not_solved_after_full_chain_raises(self, monkeypatch):
        # Pre-existing hole: a fast NotSolved that survives the whole fallback
        # chain used to return a KeplerResult with EMPTY slots. It must raise.
        def _not_solved(self, solver=None, **kwargs):
            self.status = pulp.LpStatusNotSolved
            self.sol_status = pulp.constants.LpSolutionNoSolutionFound
            return self.status

        monkeypatch.setattr(pulp.LpProblem, "solve", _not_solved)
        monkeypatch.setattr(pulp.LpProblem, "writeLP", lambda self, *a, **k: None)
        input_data, config = self._simple_setup()
        with pytest.raises(PlannerError) as exc:
            KeplerSolver().solve(input_data, config)
        assert exc.value.code == PlannerErrorCode.SOLVER_UNDEFINED


class TestHourlyWaterBlocks:
    def test_off_by_default(self):
        assert (
            KeplerConfig(
                capacity_kwh=10.0,
                min_soc_percent=10.0,
                max_soc_percent=100.0,
                max_charge_power_kw=5.0,
                max_discharge_power_kw=5.0,
                charge_efficiency=1.0,
                discharge_efficiency=1.0,
                wear_cost_sek_per_kwh=0.0,
            ).water_hourly_blocks
            is False
        )

    def test_water_constant_within_each_hour(self):
        # 16 quarter-slots = 4 hours; hour 1 (04-08th slot) and hour 2 cheap.
        prices = [2.0] * 4 + [0.2] * 8 + [2.0] * 4
        heater = WaterHeaterInput(
            id="vvb",
            power_kw=3.0,
            min_kwh_per_day=4.0,
            max_hours_between_heating=0.0,
            min_spacing_hours=0.0,
            heated_today_kwh=0.0,
        )
        input_data = KeplerInput(slots=_quarter_slots(16, prices), initial_soc_kwh=5.0)
        config = _cfg([heater], water_hourly_blocks=True)

        result = KeplerSolver().solve(input_data, config)

        assert result.is_optimal
        heated_total = sum(s.water_heat_kw * 0.25 for s in result.slots)
        assert heated_total >= 4.0 - 0.01
        for h in range(4):
            hour_vals = {round(result.slots[t].water_heat_kw, 6) for t in range(4 * h, 4 * h + 4)}
            assert len(hour_vals) == 1, f"hour {h} not constant: {hour_vals}"

    def test_subhour_precision_without_flag(self):
        # Control: flag off, min 0.75 kWh = exactly one 15-min slot at 3 kW.
        # Sub-hour semantics must remain exact (only one slot heats).
        prices = [2.0] * 4 + [0.2] * 4 + [2.0] * 8
        heater = WaterHeaterInput(
            id="vvb",
            power_kw=3.0,
            min_kwh_per_day=0.75,
            max_hours_between_heating=0.0,
            min_spacing_hours=0.0,
            heated_today_kwh=0.0,
        )
        input_data = KeplerInput(slots=_quarter_slots(16, prices), initial_soc_kwh=5.0)
        config = _cfg([heater], water_hourly_blocks=False)

        result = KeplerSolver().solve(input_data, config)

        assert result.is_optimal
        heating_slots = [s for s in result.slots if s.water_heat_kw > 0]
        assert len(heating_slots) == 1
