"""Tests for the no-battery counterfactual savings metric (backend/learning/savings.py)."""

import pytest

from backend.learning.savings import compute_savings


def _slot(imp=0.0, exp=0.0, pv=0.0, load=0.0, p_imp=2.0, p_exp=0.8, water=0.0, ev=0.0):
    return {
        "import_kwh": imp,
        "export_kwh": exp,
        "pv_kwh": pv,
        "load_kwh": load,
        "water_kwh": water,
        "ev_charging_kwh": ev,
        "import_price_sek_kwh": p_imp,
        "export_price_sek_kwh": p_exp,
    }


def test_empty_rows():
    s = compute_savings([])
    assert s.savings_sek == 0.0
    assert s.n_slots == 0
    assert s.coverage == 0.0


def test_no_battery_house_matches_baseline_exactly():
    """A house whose grid flows equal load-minus-pv has zero savings by definition."""
    rows = [
        _slot(imp=1.0, exp=0.0, pv=0.0, load=1.0),  # night: import == load
        _slot(imp=0.0, exp=2.0, pv=3.0, load=1.0),  # midday: export == surplus
    ]
    s = compute_savings(rows)
    assert s.savings_sek == pytest.approx(0.0)
    assert s.n_priced_slots == 2


def test_stored_solar_self_consumption_saves_money():
    """Battery stores midday surplus (less export revenue) and serves evening load
    (avoids expensive import): savings = avoided import - forgone export revenue."""
    rows = [
        # Midday: pv 3, load 1 => baseline exports 2 @ 0.8 = -1.6; actual charged battery, no export
        _slot(imp=0.0, exp=0.0, pv=3.0, load=1.0, p_imp=1.5, p_exp=0.8),
        # Evening: load 2, pv 0 => baseline imports 2 @ 3.0 = 6.0; actual served from battery
        _slot(imp=0.0, exp=0.0, pv=0.0, load=2.0, p_imp=3.0, p_exp=1.0),
    ]
    s = compute_savings(rows)
    # baseline = -1.6 + 6.0 = 4.4; actual = 0 => savings 4.4
    assert s.savings_sek == pytest.approx(4.4)


def test_arbitrage_import_cheap_serve_expensive():
    rows = [
        # Night: battery grid-charges 2 kWh @ 0.5 on top of 1 kWh load
        _slot(imp=3.0, exp=0.0, pv=0.0, load=1.0, p_imp=0.5, p_exp=0.2),
        # Morning peak: battery serves the 2 kWh load, no import
        _slot(imp=0.0, exp=0.0, pv=0.0, load=2.0, p_imp=3.0, p_exp=1.0),
    ]
    s = compute_savings(rows)
    # actual = 3*0.5 = 1.5 ; baseline = 1*0.5 + 2*3.0 = 6.5 ; savings = 5.0
    assert s.actual_cost_sek == pytest.approx(1.5)
    assert s.baseline_cost_sek == pytest.approx(6.5)
    assert s.savings_sek == pytest.approx(5.0)


def test_bad_battery_day_shows_negative_savings():
    """Honesty requirement: a day where the battery layer lost money must go negative."""
    rows = [
        # Charged 2 kWh from grid at a HIGH price...
        _slot(imp=2.0, exp=0.0, pv=0.0, load=0.0, p_imp=3.0),
        # ...then exported it cheaply.
        _slot(imp=0.0, exp=2.0, pv=0.0, load=0.0, p_imp=3.0, p_exp=0.3),
    ]
    s = compute_savings(rows)
    assert s.savings_sek < 0


def test_unpriced_slots_are_skipped_and_reported():
    rows = [
        _slot(imp=1.0, load=1.0),
        {"import_kwh": 5.0, "load_kwh": 5.0, "import_price_sek_kwh": None},
    ]
    s = compute_savings(rows)
    assert s.n_slots == 2
    assert s.n_priced_slots == 1
    assert s.coverage == pytest.approx(0.5)


def test_vvb_heavy_day_baseline_includes_shifted_loads():
    """Regression: load_kwh is BASE load (recorder subtracts water + EV before storing).

    A house importing 2 kWh to run the VVB (1.5 kWh) on top of 0.5 kWh base load has a
    no-battery baseline of exactly the same 2 kWh import — savings must be 0, not the
    -3.0 SEK the pre-fix code reported (baseline understated by water_kwh * price).
    """
    rows = [_slot(imp=2.0, load=0.5, water=1.5, p_imp=2.0)]
    s = compute_savings(rows)
    assert s.baseline_cost_sek == pytest.approx(4.0)
    assert s.savings_sek == pytest.approx(0.0)


def test_ev_charging_counts_toward_baseline_load():
    """Same regression for the EV column: whole-house load = base + water + EV."""
    rows = [
        # 7 kWh EV charge + 1 kWh base, all imported: baseline == actual == 16 SEK.
        _slot(imp=8.0, load=1.0, ev=7.0, p_imp=2.0),
    ]
    s = compute_savings(rows)
    assert s.actual_cost_sek == pytest.approx(16.0)
    assert s.baseline_cost_sek == pytest.approx(16.0)
    assert s.savings_sek == pytest.approx(0.0)


def test_rows_without_water_ev_columns_still_work():
    """Older callers may pass rows without the water/EV keys — treated as 0."""
    rows = [
        {
            "import_kwh": 1.0,
            "export_kwh": 0.0,
            "pv_kwh": 0.0,
            "load_kwh": 1.0,
            "import_price_sek_kwh": 2.0,
            "export_price_sek_kwh": 0.8,
        }
    ]
    s = compute_savings(rows)
    assert s.savings_sek == pytest.approx(0.0)


def test_missing_export_price_treated_as_zero():
    rows = [
        {
            "import_kwh": 0.0,
            "export_kwh": 2.0,
            "pv_kwh": 2.0,
            "load_kwh": 0.0,
            "import_price_sek_kwh": 2.0,
            "export_price_sek_kwh": None,
        }
    ]
    s = compute_savings(rows)
    # Export valued at 0 on both sides => no phantom savings from missing prices.
    assert s.savings_sek == pytest.approx(0.0)
