"""Tests for the effekttariff peak-demand-charge term (monthly peak on hourly-mean import).

The tariff economics: raising the month's peak costs the full monthly kr/kW; staying
under the existing month-to-date peak (the baseline) is free. So a high peak cost must
stop battery arbitrage from concentrating import into one cheap hour, while a baseline
above any planned import must make the term inert.
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot


def _two_hour_arbitrage_input() -> KeplerInput:
    """2 hours x 4 slots: cheap hour 1 (0.5 SEK), expensive hour 2 (3.0 SEK), flat 4 kW load."""
    start = datetime(2025, 1, 1, 10, 0)
    slots = []
    for i in range(8):
        price = 0.5 if i < 4 else 3.0
        slots.append(
            KeplerInputSlot(
                start_time=start + timedelta(minutes=15 * i),
                end_time=start + timedelta(minutes=15 * (i + 1)),
                load_kwh=1.0,  # 4 kW flat
                pv_kwh=0.0,
                import_price_sek_kwh=price,
                export_price_sek_kwh=0.0,
            )
        )
    return KeplerInput(slots=slots, initial_soc_kwh=0.0)


def _cfg(**over) -> KeplerConfig:
    base = {
        "capacity_kwh": 10.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 10.0,
        "max_discharge_power_kw": 10.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "wear_cost_sek_per_kwh": 0.1,
        "enable_export": False,
    }
    base.update(over)
    return KeplerConfig(**base)


def _hour_import(result, input_data, hour_index: int) -> float:
    """Total grid import (kWh) in clock hour 0 or 1 of the 2-hour scenario."""
    lo, hi = (0, 4) if hour_index == 0 else (4, 8)
    return sum(result.slots[t].grid_import_kwh for t in range(lo, hi))


def test_no_peak_cost_concentrates_import_in_cheap_hour():
    """Baseline behaviour: arbitrage front-loads import (load + battery charge) into hour 1."""
    inp = _two_hour_arbitrage_input()
    result = KeplerSolver().solve(inp, _cfg())
    assert result.is_optimal
    # Shifting 4 kWh saves (3.0 - 0.5 - 0.1) each => planner charges in the cheap hour.
    assert sum(s.charge_kwh for s in result.slots) >= 3.9
    assert _hour_import(result, inp, 0) >= 7.5  # ~8 kWh => 8 kW hourly mean
    assert _hour_import(result, inp, 1) <= 0.1


def test_peak_cost_prevents_raising_the_hourly_peak():
    """50 SEK/kW demand charge (raise 4->8 kW = 200 SEK) dwarfs the ~9.6 SEK arbitrage gain."""
    inp = _two_hour_arbitrage_input()
    result = KeplerSolver().solve(inp, _cfg(peak_power_cost_sek_per_kw=50.0))
    assert result.is_optimal
    assert sum(s.charge_kwh for s in result.slots) <= 0.1  # arbitrage abandoned
    # Both hours stay at the unavoidable 4 kW load level.
    assert _hour_import(result, inp, 0) <= 4.1
    assert _hour_import(result, inp, 1) >= 3.9


def test_high_baseline_makes_peak_cost_inert():
    """Month-to-date peak of 10 kW already dwarfs any plan here => raising costs nothing."""
    inp = _two_hour_arbitrage_input()
    result = KeplerSolver().solve(
        inp, _cfg(peak_power_cost_sek_per_kw=50.0, peak_power_baseline_kw=10.0)
    )
    assert result.is_optimal
    # Behaves like the no-peak-cost case: arbitrage is free again under the baseline.
    assert sum(s.charge_kwh for s in result.slots) >= 3.9
    assert _hour_import(result, inp, 0) >= 7.5


def test_objective_cost_is_reported_and_matches_energy_when_no_penalties():
    """objective_cost_sek must be populated on optimal solves; in a penalty-free
    scenario (no water/EV/deferrable, ramping 0, no slacks) it equals the
    energy-only recomputation to within rounding."""
    inp = _two_hour_arbitrage_input()
    cfg = _cfg(wear_cost_sek_per_kwh=0.0, ramping_cost_sek_per_kw=0.0)
    result = KeplerSolver().solve(inp, cfg)
    assert result.is_optimal
    assert result.objective_cost_sek is not None
    assert result.objective_cost_sek == pytest.approx(result.total_cost_sek, abs=0.05)


def test_objective_cost_reveals_penalty_wedge():
    """With the demand charge active, the objective includes the peak cost while the
    energy-only figure doesn't — the wedge is exactly what the reporting exposes."""
    inp = _two_hour_arbitrage_input()
    result = KeplerSolver().solve(inp, _cfg(peak_power_cost_sek_per_kw=50.0))
    assert result.is_optimal
    assert result.objective_cost_sek is not None
    # Unavoidable 4 kW load peak x 50 SEK = 200 SEK of peak cost in the objective.
    assert result.objective_cost_sek - result.total_cost_sek == pytest.approx(200.0, abs=1.0)


def test_zero_cost_is_legacy_no_op():
    """peak_power_cost 0.0 (default) must add no variable/constraints — identical plan."""
    inp = _two_hour_arbitrage_input()
    legacy = KeplerSolver().solve(inp, _cfg())
    explicit_zero = KeplerSolver().solve(inp, _cfg(peak_power_cost_sek_per_kw=0.0))
    assert legacy.is_optimal and explicit_zero.is_optimal
    for a, b in zip(legacy.slots, explicit_zero.slots, strict=True):
        assert abs(a.grid_import_kwh - b.grid_import_kwh) < 1e-6
        assert abs(a.charge_kwh - b.charge_kwh) < 1e-6


def _mid_hour_input(start_minute: int, n_slots: int, prices: list[float], loads: list[float]):
    start = datetime(2025, 1, 1, 12, start_minute)
    slots = [
        KeplerInputSlot(
            start_time=start + timedelta(minutes=15 * i),
            end_time=start + timedelta(minutes=15 * (i + 1)),
            load_kwh=loads[i],
            pv_kwh=0.0,
            import_price_sek_kwh=prices[i],
            export_price_sek_kwh=0.0,
        )
        for i in range(n_slots)
    ]
    return KeplerInput(slots=slots, initial_soc_kwh=0.0)


def test_mid_hour_elapsed_import_prices_the_full_billed_hour():
    """Replan at 12:30 with 5.5 kWh already imported: only 0.5 kWh of free headroom
    remains under a 6 kW baseline — the planner must throttle to the battery instead
    of treating the half hour as a fresh 6 kW mean."""
    # price 0.02 < 0.05/kWh battery wear => grid strictly preferred wherever allowed
    inp = _mid_hour_input(30, 6, prices=[0.02] * 6, loads=[1.0] * 6)
    inp = KeplerInput(slots=inp.slots, initial_soc_kwh=10.0)
    cfg = _cfg(
        peak_power_cost_sek_per_kw=50.0,
        peak_power_baseline_kw=6.0,
        peak_hour_elapsed_import_kwh=5.5,
    )
    result = KeplerSolver().solve(inp, cfg)
    assert result.is_optimal
    hour12 = sum(result.slots[t].grid_import_kwh for t in range(2))  # 12:30 + 12:45
    hour13 = sum(result.slots[t].grid_import_kwh for t in range(2, 6))
    assert hour12 <= 0.51, f"expected throttle to the 0.5 kWh headroom, got {hour12}"
    assert hour13 >= 3.9  # next hour is fresh: load served from grid again

    # Control: without the elapsed injection the same replan sees a free 6 kW mean.
    cfg_no_elapsed = _cfg(peak_power_cost_sek_per_kw=50.0, peak_power_baseline_kw=6.0)
    control = KeplerSolver().solve(inp, cfg_no_elapsed)
    assert control.is_optimal
    assert sum(control.slots[t].grid_import_kwh for t in range(2)) >= 1.9


def test_mid_hour_burst_is_not_phantom_priced():
    """Replan at 12:45: charging 2.5 kWh in the last quarter is a 2.5 kW billed hour
    mean (free under a 6 kW baseline) — not a phantom 10 kW peak that blocks arbitrage."""
    prices = [0.5] + [3.0] * 4  # cheap final quarter of hour 12, expensive hour 13
    loads = [0.0] + [1.0] * 4
    inp = _mid_hour_input(45, 5, prices=prices, loads=loads)
    cfg = _cfg(
        peak_power_cost_sek_per_kw=50.0,
        peak_power_baseline_kw=6.0,
        peak_hour_elapsed_import_kwh=0.0,
    )
    result = KeplerSolver().solve(inp, cfg)
    assert result.is_optimal
    # 2.5 kWh charged in 12:45-13:00 => hour-12 mean 2.5 kW <= 6 kW baseline => free.
    assert sum(s.charge_kwh for s in result.slots) >= 2.4


def test_next_month_hours_price_their_peak_from_zero():
    """A month-end horizon must not let next-month hours inherit this month's peak as
    free headroom: February's demand charge bills from zero again."""
    start = datetime(2025, 1, 31, 23, 0)
    slots = []
    for i in range(8):  # Jan 31 23:00-24:00 (0.50 SEK) + Feb 1 00:00-01:00 (0.45 SEK)
        slots.append(
            KeplerInputSlot(
                start_time=start + timedelta(minutes=15 * i),
                end_time=start + timedelta(minutes=15 * (i + 1)),
                load_kwh=0.0,
                pv_kwh=0.0,
                import_price_sek_kwh=0.50 if i < 4 else 0.45,
                export_price_sek_kwh=0.0,
            )
        )
    inp = KeplerInput(slots=slots, initial_soc_kwh=0.0)
    cfg = _cfg(
        peak_power_cost_sek_per_kw=50.0,
        peak_power_baseline_kw=9.0,  # January's month-to-date peak
        target_soc_kwh=8.0,
        target_soc_penalty_sek=1000.0,
    )
    result = KeplerSolver().solve(inp, cfg)
    assert result.is_optimal
    jan_import = sum(result.slots[t].grid_import_kwh for t in range(4))
    feb_import = sum(result.slots[t].grid_import_kwh for t in range(4, 8))
    # Charging in January rides under the existing 9 kW peak for free; the same 8 kW
    # mean in February would cost 8 * 50 SEK — far more than the 0.05 SEK/kWh saving.
    assert jan_import >= 7.5, f"expected charging under January's baseline, got {jan_import}"
    assert feb_import <= 0.5, f"February must not inherit January's headroom, got {feb_import}"
