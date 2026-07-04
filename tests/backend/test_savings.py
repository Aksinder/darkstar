"""Tests for the no-battery counterfactual savings metric (backend/learning/savings.py)."""

import pytest

from backend.learning.savings import compute_savings


def _slot(imp=0.0, exp=0.0, pv=0.0, load=0.0, p_imp=2.0, p_exp=0.8):
    return {
        "import_kwh": imp,
        "export_kwh": exp,
        "pv_kwh": pv,
        "load_kwh": load,
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
