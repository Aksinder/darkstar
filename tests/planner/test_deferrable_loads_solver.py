"""Tests for deferrable household loads in the Kepler MILP solver."""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    DeferrableLoadInput,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
)

START = datetime(2025, 1, 1, 0, 0)


def _slots(prices, pv=None):
    """Build 15-min slots from a list of import prices (and optional PV kWh)."""
    out = []
    for i, price in enumerate(prices):
        s = START + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=0.0,
                pv_kwh=(pv[i] if pv else 0.0),
                import_price_sek_kwh=price,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def _config(loads, **kw):
    """KeplerConfig with the battery neutralised (capacity 0) to isolate loads."""
    base = {
        "capacity_kwh": 0.0,  # neutralise battery: soc fixed at 0, charge==discharge==0
        "max_charge_power_kw": 1.0,
        "max_discharge_power_kw": 1.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        "wear_cost_sek_per_kwh": 0.0,
        "deferrable_loads": loads,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _run_slots(result, load_id):
    """Indices where the load is running (kW > 0)."""
    return [
        i for i, s in enumerate(result.slots) if s.deferrable_load_results.get(load_id, 0.0) > 0.5
    ]


class TestDeferrableScheduling:
    def test_runs_once_for_duration_in_cheapest_window(self):
        # 8 slots; slots 4-5 are dirt cheap, everything else expensive.
        prices = [2.0, 2.0, 2.0, 2.0, 0.1, 0.1, 2.0, 2.0]
        load = DeferrableLoadInput(id="dishwasher", energy_kwh=2.0, duration_slots=2)
        result = KeplerSolver().solve(KeplerInput(_slots(prices), 0.0), _config([load]))

        assert result.is_optimal
        running = _run_slots(result, "dishwasher")
        assert running == [4, 5]  # contiguous, in the cheap window
        total_kwh = sum(s.deferrable_load_results["dishwasher"] * 0.25 for s in result.slots)
        assert total_kwh == pytest.approx(2.0, abs=0.01)

    def test_prefers_free_pv_surplus_over_cheap_grid(self):
        # Uniform cheap price, but slots 2-3 have abundant PV (free energy).
        prices = [0.5] * 8
        pv = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
        load = DeferrableLoadInput(id="washer", energy_kwh=2.0, duration_slots=2)
        result = KeplerSolver().solve(KeplerInput(_slots(prices, pv), 0.0), _config([load]))

        running = _run_slots(result, "washer")
        assert running == [2, 3]  # runs under the PV surplus

    def test_hard_deadline_forces_early_run(self):
        # Cheap slots are late (6-7), but a hard deadline requires finishing by slot 2.
        prices = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.1, 0.1]
        load = DeferrableLoadInput(
            id="dishwasher",
            energy_kwh=2.0,
            duration_slots=2,
            deadline_slot=2,
            deadline_hard=True,
        )
        result = KeplerSolver().solve(KeplerInput(_slots(prices), 0.0), _config([load]))

        running = _run_slots(result, "dishwasher")
        assert max(running) <= 2  # finished by the deadline despite cheaper late slots

    def test_soft_deadline_allows_running_late_when_cheaper(self):
        # Same shape, but a soft deadline with a small penalty: economics win,
        # so the load runs in the much cheaper late window.
        prices = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.1, 0.1]
        load = DeferrableLoadInput(
            id="dishwasher",
            energy_kwh=2.0,
            duration_slots=2,
            deadline_slot=2,
            deadline_hard=False,
        )
        result = KeplerSolver().solve(
            KeplerInput(_slots(prices), 0.0),
            _config([load], deferrable_soft_deadline_penalty_sek=0.01),
        )
        running = _run_slots(result, "dishwasher")
        assert min(running) >= 6  # ran late where it was cheap

    def test_earliest_start_respected(self):
        # Cheap early, but the load may not start before slot 3.
        prices = [0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 0.1, 0.1]
        load = DeferrableLoadInput(
            id="washer", energy_kwh=2.0, duration_slots=2, earliest_start_slot=3
        )
        result = KeplerSolver().solve(KeplerInput(_slots(prices), 0.0), _config([load]))
        running = _run_slots(result, "washer")
        assert min(running) >= 3

    def test_two_loads_both_scheduled(self):
        prices = [2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0]
        loads = [
            DeferrableLoadInput(id="dishwasher", energy_kwh=2.0, duration_slots=2),
            DeferrableLoadInput(id="washer", energy_kwh=1.0, duration_slots=1),
        ]
        result = KeplerSolver().solve(KeplerInput(_slots(prices), 0.0), _config(loads))
        assert len(_run_slots(result, "dishwasher")) == 2
        assert len(_run_slots(result, "washer")) == 1

    def test_phase_penalty_separates_same_phase_loads(self):
        # Two 1-slot loads on phase A; only slots 2-3 are cheap. Without a phase
        # penalty both would stack on the single cheapest slot; with one they
        # spread across the two cheap slots.
        prices = [2.0, 2.0, 0.1, 0.1, 2.0, 2.0]
        loads = [
            DeferrableLoadInput(id="a1", energy_kwh=1.0, duration_slots=1, phase="A"),
            DeferrableLoadInput(id="a2", energy_kwh=1.0, duration_slots=1, phase="A"),
        ]
        result = KeplerSolver().solve(
            KeplerInput(_slots(prices), 0.0),
            _config(loads, deferrable_phase_penalty_sek=5.0),
        )
        a1 = _run_slots(result, "a1")
        a2 = _run_slots(result, "a2")
        assert a1 != a2  # not running in the same slot
        assert set(a1 + a2) == {2, 3}

    def test_load_too_long_for_horizon_is_skipped(self):
        # A 10-slot load into a 4-slot horizon cannot fit: solver must stay
        # feasible and simply not schedule it.
        prices = [1.0, 1.0, 1.0, 1.0]
        load = DeferrableLoadInput(id="oversized", energy_kwh=5.0, duration_slots=10)
        result = KeplerSolver().solve(KeplerInput(_slots(prices), 0.0), _config([load]))
        assert result.is_optimal
        assert _run_slots(result, "oversized") == []
