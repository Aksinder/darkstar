"""Build #15 PART B: measured heated-today crediting flows into kepler's day_min.

The tank-blind solver used to re-insert the FULL min_kwh_per_day floor on every
replan (heated_today hardcoded 0.0), which walked the overnight water block. These
tests prove that a per-tank heated_today_kwh is subtracted from the daily minimum
(heated 6 -> day_min 0) and that over-crediting can never drive the floor negative.
"""

from datetime import datetime, timedelta

from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    WaterHeaterInput,
)


def _slots(n: int = 48, import_price: float = 0.1) -> list[KeplerInputSlot]:
    base = datetime(2026, 1, 15, 0, 0)
    out: list[KeplerInputSlot] = []
    for i in range(n):
        s = base + timedelta(minutes=30 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=30),
                load_kwh=0.5,
                pv_kwh=0.0,
                import_price_sek_kwh=import_price,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def _wh(heated_today_kwh: float, min_kwh_per_day: float = 6.0) -> WaterHeaterInput:
    return WaterHeaterInput(
        id="main_tank",
        power_kw=3.0,
        min_kwh_per_day=min_kwh_per_day,
        max_hours_between_heating=0.0,
        min_spacing_hours=0.0,
        heated_today_kwh=heated_today_kwh,
    )


def _cfg(heater: WaterHeaterInput) -> KeplerConfig:
    return KeplerConfig(
        capacity_kwh=10.0,
        min_soc_percent=10.0,
        max_soc_percent=100.0,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        wear_cost_sek_per_kwh=0.0,
        water_heaters=[heater],
        water_reliability_penalty_sek=100.0,
    )


def _scheduled_water_kwh(result) -> float:
    return sum(s.water_heat_kw * 0.5 for s in result.slots)  # 30-min slots


def test_uncredited_heater_schedules_full_floor():
    """Baseline (heated_today=0): the solver schedules the full 6 kWh daily floor."""
    solver = KeplerSolver()
    result = solver.solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=5.0), _cfg(_wh(0.0))
    )

    assert result.is_optimal
    assert _scheduled_water_kwh(result) >= 6.0 - 0.01


def test_fully_credited_heater_schedules_nothing():
    """heated_today == min_kwh_per_day => day_min drops to 0, solver heats ~0 kWh."""
    solver = KeplerSolver()
    result = solver.solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=5.0), _cfg(_wh(6.0))
    )

    assert result.is_optimal
    assert _scheduled_water_kwh(result) <= 0.01


def test_partial_credit_reduces_remaining_floor():
    """heated_today=4 of 6 => the solver only tops up the remaining ~2 kWh."""
    solver = KeplerSolver()
    result = solver.solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=5.0), _cfg(_wh(4.0))
    )

    assert result.is_optimal
    scheduled = _scheduled_water_kwh(result)
    # Remaining floor is 2 kWh; the solver should not re-heat the whole 6.
    assert scheduled >= 2.0 - 0.01
    assert scheduled < 6.0 - 1.0


def test_overcredit_cannot_drive_day_min_negative():
    """Even an (unclamped) heated_today far above the floor only zeros the day_min,
    never makes it negative or forces phantom heating. (Source clamps to the floor;
    kepler's max(0, min - heated) is the belt-and-braces backstop tested here.)"""
    solver = KeplerSolver()
    result = solver.solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=5.0), _cfg(_wh(100.0))
    )

    assert result.is_optimal
    scheduled = _scheduled_water_kwh(result)
    assert scheduled <= 0.01
    assert scheduled >= 0.0
