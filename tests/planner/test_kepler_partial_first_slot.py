"""In-progress slot 0 is planned over the time left in it (kepler.partial_first_slot).

Replans run on a cadence and a solve can take minutes, so slot 0 is almost always
partly elapsed. Planning it as a whole slot handed the CURRENT SoC a full slot of
PV/load and charge headroom. With ``first_slot_remaining_h`` set, slot 0's energy
and power limits use the remaining hours, its PV/load forecast is scaled to that
share, and the reported kW (what the executor commands) keeps its meaning:
planned energy / remaining hours.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from planner.coarse_tail import expand_result
from planner.plan_validator import validate_plan
from planner.solver.adapter import (
    first_slot_remaining_h,
    kepler_result_to_dataframe,
    scale_first_slot_to_remaining,
)
from planner.solver.battery_lp import solve_battery_only
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    MIN_FIRST_SLOT_H,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    KeplerResult,
    KeplerResultSlot,
    WaterHeaterInput,
    planned_slot_hours,
)

BASE = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)


def _slots(n: int = 8, pv: float = 0.0, load: float = 1.0) -> list[KeplerInputSlot]:
    """Slot 0 cheapest, 1..3 cheap, later slots expensive with more load than the
    battery holds: the solver charges at the limit, slot 0 first."""
    out = []
    for i in range(n):
        start = BASE + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=start,
                end_time=start + timedelta(minutes=15),
                load_kwh=load,
                pv_kwh=pv,
                import_price_sek_kwh=0.05 if i == 0 else (0.2 if i < 4 else 5.0),
                export_price_sek_kwh=0.05,
            )
        )
    return out


def _cfg(**overrides) -> KeplerConfig:
    kwargs: dict = {
        "capacity_kwh": 16.0,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 4.0,
        "max_discharge_power_kw": 4.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "wear_cost_sek_per_kwh": 0.0,
        "enable_export": False,
    }
    kwargs.update(overrides)
    return KeplerConfig(**kwargs)


def _input(remaining_h: float | None, **slot_kw) -> KeplerInput:
    ki = KeplerInput(slots=_slots(**slot_kw), initial_soc_kwh=2.0)
    if remaining_h is None:
        return ki
    ki.first_slot_remaining_h = remaining_h
    return scale_first_slot_to_remaining(ki)


# ------------------------------------------------------------------ adapter helpers


def test_scale_halves_pv_and_load_at_half_elapsed_without_mutating_original():
    slots = _slots(pv=1.0, load=0.4)
    ki = KeplerInput(slots=slots, initial_soc_kwh=0.0, first_slot_remaining_h=0.125)
    out = scale_first_slot_to_remaining(ki)
    assert out.slots[0].pv_kwh == pytest.approx(0.5)
    assert out.slots[0].load_kwh == pytest.approx(0.2)
    assert out.slots[1].pv_kwh == pytest.approx(1.0)  # only slot 0
    # The original slot object may be shared with the coarse tail's fine list.
    assert slots[0].pv_kwh == pytest.approx(1.0)


def test_scale_is_noop_for_boundary_or_unset():
    ki = KeplerInput(slots=_slots(pv=1.0), initial_soc_kwh=0.0)
    assert scale_first_slot_to_remaining(ki).slots[0].pv_kwh == pytest.approx(1.0)
    ki.first_slot_remaining_h = 0.25
    assert scale_first_slot_to_remaining(ki).slots[0].pv_kwh == pytest.approx(1.0)


def test_remaining_hours_clamped_and_floored():
    s, e = BASE, BASE + timedelta(minutes=15)
    assert first_slot_remaining_h(s, e, BASE + timedelta(minutes=5)) == pytest.approx(10 / 60)
    assert first_slot_remaining_h(s, e, BASE - timedelta(minutes=5)) == pytest.approx(0.25)
    assert first_slot_remaining_h(s, e, BASE + timedelta(minutes=20)) == 0.0
    # The solver basis floors an (almost) elapsed slot so the model stays well-posed.
    assert planned_slot_hours(_slots(1), 0.0)[0] == pytest.approx(MIN_FIRST_SLOT_H)


# ------------------------------------------------------------------ solver


def test_half_elapsed_slot0_halves_charge_energy_keeps_kw():
    cfg = _cfg()
    full = KeplerSolver().solve(_input(None), cfg)
    half = KeplerSolver().solve(_input(0.125), cfg)
    assert full.is_optimal and half.is_optimal

    # Charging at the 4 kW limit in the cheap slot either way.
    assert full.slots[0].charge_kwh == pytest.approx(1.0, abs=1e-4)
    assert half.slots[0].charge_kwh == pytest.approx(0.5, abs=1e-4)
    assert half.slots[0].duration_h == pytest.approx(0.125)
    assert full.slots[0].duration_h is None
    assert half.slots[1].duration_h is None

    df_full = kepler_result_to_dataframe(full, 16.0, 2.0)
    df_half = kepler_result_to_dataframe(half, 16.0, 2.0)
    # Commanded kW keeps its meaning: energy planned for the remainder / remainder.
    assert df_full.iloc[0]["battery_charge_kw"] == pytest.approx(4.0, abs=1e-3)
    assert df_half.iloc[0]["battery_charge_kw"] == pytest.approx(4.0, abs=1e-3)
    # The SoC after slot 0 no longer pretends a whole slot is still ahead.
    assert half.slots[0].soc_kwh == pytest.approx(full.slots[0].soc_kwh - 0.5, abs=1e-3)


def test_half_elapsed_slot0_pv_halved_in_balance():
    # Sunny slot 0, battery free to absorb: PV-only charge (no import, export off).
    cfg = _cfg(max_charge_power_kw=10.0, curtailment_penalty_sek=1.0)
    ki_full = KeplerInput(slots=_slots(n=4, pv=2.0, load=0.0), initial_soc_kwh=2.0)
    for s in ki_full.slots:
        s.import_price_sek_kwh = 5.0  # import never pays; only PV charges
    ki_half = scale_first_slot_to_remaining(
        KeplerInput(slots=list(ki_full.slots), initial_soc_kwh=2.0, first_slot_remaining_h=0.125)
    )
    full = KeplerSolver().solve(ki_full, cfg)
    half = KeplerSolver().solve(ki_half, cfg)
    assert full.slots[0].charge_kwh == pytest.approx(2.0, abs=1e-3)
    assert half.slots[0].charge_kwh == pytest.approx(1.0, abs=1e-3)
    assert half.slots[0].grid_import_kwh == pytest.approx(0.0, abs=1e-6)


def test_partial_slot0_water_force_on_reports_heater_power():
    heater = WaterHeaterInput(
        id="vvb",
        power_kw=3.0,
        min_kwh_per_day=0.0,
        max_hours_between_heating=24.0,
        min_spacing_hours=0.0,
        force_on_slots=[0],
    )
    cfg = _cfg(water_heaters=[heater], water_hourly_blocks=False)
    ki = _input(0.05)
    res = KeplerSolver().solve(ki, cfg)
    assert res.slots[0].water_heater_results["vvb"] == pytest.approx(3.0)
    # The validator agrees with the solver about slot 0's energy (3 kW x 0.05 h).
    v = validate_plan(res, ki, cfg)
    assert v.ok, v.errors


def test_validator_accepts_partial_plan_and_rejects_full_slot_limits():
    cfg = _cfg()
    ki = _input(0.125)
    res = KeplerSolver().solve(ki, cfg)
    assert validate_plan(res, ki, cfg).ok
    # A plan that charges a FULL slot's energy into a half-elapsed slot breaches
    # the remaining-time power limit — the validator must see that.
    res.slots[0].charge_kwh = 1.0
    res.slots[0].soc_kwh += 0.5
    assert not validate_plan(res, ki, cfg).ok


def test_battery_only_fallback_uses_remaining_hours():
    cfg = _cfg()
    res = solve_battery_only(_input(0.125), cfg)
    assert res.slots[0].charge_kwh <= 0.5 + 1e-4
    assert res.slots[0].duration_h == pytest.approx(0.125)
    df = kepler_result_to_dataframe(res, 16.0, 2.0)
    assert df.iloc[0]["battery_charge_kw"] <= 4.0 + 1e-3


def test_coarse_split_members_drop_duration():
    start = BASE
    coarse = KeplerResultSlot(
        start_time=start,
        end_time=start + timedelta(hours=1),
        charge_kwh=4.0,
        discharge_kwh=0.0,
        grid_import_kwh=4.0,
        grid_export_kwh=0.0,
        soc_kwh=8.0,
        cost_sek=0.0,
        duration_h=1.0,
    )
    originals = _slots(4)
    out = expand_result(
        KeplerResult(slots=[coarse], total_cost_sek=0.0, is_optimal=True, status_msg="ok"),
        [[0, 1, 2, 3]],
        originals,
    )
    assert all(s.duration_h is None for s in out.slots)
    df = kepler_result_to_dataframe(out, 16.0, 4.0)
    assert df["battery_charge_kw"].tolist() == pytest.approx([4.0] * 4)
