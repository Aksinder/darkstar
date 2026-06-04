"""Tests for the realism replay (phase-imbalance + idle-freeze exposure)."""

import pandas as pd

from planner.simulation import (
    RealismSlot,
    realism_from_schedule,
    simulate_realistic,
)


def _slot(**kw):
    base = {
        "pv_kwh": 0.0,
        "load_kwh": 0.0,
        "discharge_kwh": 0.0,
        "grid_import_kwh": 0.0,
        "grid_export_kwh": 0.0,
        "import_price_sek_kwh": 2.0,
        "export_price_sek_kwh": 0.5,
        "soc_percent": 50.0,
        "soc_target_percent": 50.0,
    }
    base.update(kw)
    return RealismSlot(**base)


def test_no_fractions_gives_zero_gap():
    slots = [_slot(load_kwh=1.0, grid_import_kwh=1.0)]
    r = simulate_realistic(slots)  # no phase model
    assert r.realism_gap_sek == 0.0
    assert r.predicted_cost_sek == r.simulated_cost_sek


def test_balanced_load_gives_zero_gap():
    # Balanced load across 3 phases -> phase import equals net import.
    slots = [_slot(pv_kwh=2.0, load_kwh=1.0, grid_export_kwh=1.0)]
    r = simulate_realistic(slots, phase_fractions={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert abs(r.realism_gap_sek) < 1e-6


def test_single_phase_load_full_battery_exposes_hidden_import():
    # The live bug: PV surplus, battery full so discharge=0, but a large load on
    # ONE phase. The net-node LP sees surplus (export); reality imports on phase A.
    slots = [
        _slot(
            pv_kwh=2.75,  # ~11 kW * 15 min
            load_kwh=1.9,  # ~7.6 kW * 15 min, all on phase A
            discharge_kwh=0.0,
            grid_import_kwh=0.0,  # LP thought: surplus -> no import
            grid_export_kwh=0.85,  # LP: export the surplus
            soc_percent=100.0,
            soc_target_percent=100.0,
        )
    ]
    r = simulate_realistic(slots, phase_fractions={"A": 1.0, "B": 0.0, "C": 0.0})
    assert r.realism_gap_sek > 0.5  # real grid cost the LP missed
    assert r.extra_import_kwh > 0.0
    assert r.phase_flagged_slots == [0]


def test_idle_exposed_flag_on_hold_during_pv_surplus():
    slots = [
        _slot(pv_kwh=2.0, load_kwh=0.5, discharge_kwh=0.0, soc_percent=100, soc_target_percent=100)
    ]
    r = simulate_realistic(slots)
    assert r.idle_exposed_slots == [0]


def test_idle_flag_not_raised_when_discharging():
    slots = [
        _slot(pv_kwh=2.0, load_kwh=0.5, discharge_kwh=1.0, soc_percent=100, soc_target_percent=100)
    ]
    r = simulate_realistic(slots)
    assert r.idle_exposed_slots == []


def test_realism_from_schedule_dataframe():
    # Real planner schedule column names (kepler_* / adjusted_* / *_kwh).
    df = pd.DataFrame(
        [
            {
                "adjusted_pv_kwh": 2.75,
                "adjusted_load_kwh": 1.9,
                "kepler_discharge_kwh": 0.0,
                "import_kwh": 0.0,
                "export_kwh": 0.85,
                "import_price_sek_kwh": 2.0,
                "export_price_sek_kwh": 0.5,
                "projected_soc_percent": 100.0,
                "soc_target_percent": 100.0,
            }
        ]
    )
    r = realism_from_schedule(df, {"phase_load_fractions": {"A": 1.0, "B": 0.0, "C": 0.0}})
    assert r.realism_gap_sek > 0.5

    # No phase config -> balanced assumption -> no gap.
    r2 = realism_from_schedule(df, {})
    assert r2.realism_gap_sek == 0.0


def test_empty_schedule():
    r = realism_from_schedule(pd.DataFrame(), {})
    assert r.realism_gap_sek == 0.0
