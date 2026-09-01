"""Tests for the no-battery counterfactual savings metric (backend/learning/savings.py)."""

import pytest

from backend.learning.savings import compute_savings, compute_stored_energy_delta


def _slot(
    imp=0.0,
    exp=0.0,
    pv=0.0,
    load=0.0,
    p_imp=2.0,
    p_exp=0.8,
    water=0.0,
    ev=0.0,
    batt_charge=None,
    batt_discharge=None,
):
    # batt_* default to None (NOT 0.0): the factory must never fabricate a measurement,
    # and the existing assertions below must keep exercising the absent-column path.
    return {
        "import_kwh": imp,
        "export_kwh": exp,
        "pv_kwh": pv,
        "load_kwh": load,
        "water_kwh": water,
        "ev_charging_kwh": ev,
        "import_price_sek_kwh": p_imp,
        "export_price_sek_kwh": p_exp,
        "batt_charge_kwh": batt_charge,
        "batt_discharge_kwh": batt_discharge,
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


# --- compute_savings additivity -------------------------------------------------
# The property the stored-energy design rests on: compute_savings is a pure per-slot
# sum, so it is additive over any partition. An earlier design of the inventory term
# put a window-weighted price INSIDE this function, which would have destroyed it.


def test_compute_savings_is_additive_over_any_partition():
    rows = [
        _slot(imp=1.0, load=1.0, p_imp=2.0),
        _slot(exp=2.0, pv=3.0, load=1.0, p_exp=0.8),
        _slot(imp=0.5, load=2.0, pv=1.5, p_imp=1.4),
        _slot(exp=1.0, pv=1.0, p_exp=-0.2),
    ]
    whole = compute_savings(rows)
    for cut in range(len(rows) + 1):
        left = compute_savings(rows[:cut])
        right = compute_savings(rows[cut:])
        assert left.savings_sek + right.savings_sek == pytest.approx(
            whole.savings_sek, abs=1e-9
        ), f"additivity broken at cut {cut}"


# --- compute_stored_energy_delta ------------------------------------------------


def test_stored_delta_empty_and_batteryless_rows_are_inert():
    """No battery telemetry at all => nothing stored, no basis, zero value."""
    assert compute_stored_energy_delta([]).value_sek == 0.0
    d = compute_stored_energy_delta([_slot(imp=1.0, load=1.0), _slot(exp=2.0, pv=2.0)])
    assert d.n_slots == 2
    assert d.n_battery_slots == 0
    assert d.battery_coverage == 0.0
    assert d.net_stored_kwh == 0.0
    assert d.basis_sek_kwh is None
    assert d.value_sek == 0.0


def test_stored_delta_nulls_are_not_zeros():
    """A NULL flow column must be skipped, never coalesced to 0.0.

    The live DB's battery NULLs cluster in the recorder's daylight gaps, i.e. exactly
    when charging happens. Coalescing them to 0.0 would fabricate a phantom debit while
    priced coverage still read 1.000; battery_coverage is what exposes such a window.
    """
    rows = [
        _slot(pv=4.0, batt_charge=2.0, batt_discharge=0.0),
        _slot(pv=4.0),  # columns absent entirely
        _slot(pv=4.0, batt_charge=None, batt_discharge=1.0),  # half-measured
    ]
    d = compute_stored_energy_delta(rows, roundtrip_efficiency=1.0)
    assert d.n_battery_slots == 1
    assert d.battery_coverage == pytest.approx(1 / 3)
    # Only the fully-measured slot contributes; the half-measured discharge is NOT
    # allowed to subtract 1.0 kWh from the stock.
    assert d.net_stored_kwh == pytest.approx(2.0)


def test_stored_delta_efficiency_removes_the_roundtrip_ratchet():
    """Raw charge-minus-discharge books round-trip loss as stored energy.

    Measured on the live site over 2026-08-04..09-01: raw flow difference +28.49 kWh
    against a real SoC gain of +9.07 kWh. Applying the efficiency to the charge leg
    reconciles the two.
    """
    rows = [
        _slot(pv=10.0, batt_charge=10.0, batt_discharge=0.0),
        _slot(load=10.0, batt_charge=0.0, batt_discharge=9.5),
    ]
    raw = compute_stored_energy_delta(rows, roundtrip_efficiency=1.0)
    adjusted = compute_stored_energy_delta(rows, roundtrip_efficiency=0.95)
    assert raw.net_stored_kwh == pytest.approx(0.5)  # phantom: pure loss
    assert adjusted.net_stored_kwh == pytest.approx(0.0)  # a closed round trip


def test_stored_delta_nets_within_a_slot_that_both_charged_and_discharged():
    """216 real slots carry both flows; netting keeps pass-through out of the basis."""
    d = compute_stored_energy_delta(
        [_slot(pv=5.0, p_exp=1.0, batt_charge=3.0, batt_discharge=2.0)],
        roundtrip_efficiency=1.0,
    )
    assert d.net_stored_kwh == pytest.approx(1.0)
    assert d.charge_kwh == pytest.approx(1.0)  # netted inflow, not the raw 3.0


def test_stored_delta_prices_pv_at_export_and_grid_at_import():
    """Grid-charged energy's foregone alternative is not buying it, at the import price."""
    pv_only = compute_stored_energy_delta(
        [_slot(pv=5.0, load=0.0, p_imp=2.0, p_exp=0.8, batt_charge=2.0, batt_discharge=0.0)],
        roundtrip_efficiency=1.0,
    )
    assert pv_only.basis_sek_kwh == pytest.approx(0.8)  # displaced an export

    grid_only = compute_stored_energy_delta(
        [_slot(pv=0.0, load=0.0, p_imp=2.0, p_exp=0.8, batt_charge=2.0, batt_discharge=0.0)],
        roundtrip_efficiency=1.0,
    )
    assert grid_only.basis_sek_kwh == pytest.approx(2.0)  # displaced a purchase

    # Half from a 1 kWh surplus, half from the grid => the blend.
    mixed = compute_stored_energy_delta(
        [_slot(pv=1.0, load=0.0, p_imp=2.0, p_exp=0.8, batt_charge=2.0, batt_discharge=0.0)],
        roundtrip_efficiency=1.0,
    )
    assert mixed.basis_sek_kwh == pytest.approx((1.0 * 0.8 + 1.0 * 2.0) / 2.0)


def test_stored_delta_negative_export_price_is_not_clamped():
    """SE3 genuinely goes negative; storing then is a real gain, and it must show."""
    d = compute_stored_energy_delta(
        [_slot(pv=5.0, load=0.0, p_exp=-0.5, batt_charge=2.0, batt_discharge=0.0)],
        roundtrip_efficiency=1.0,
    )
    assert d.basis_sek_kwh == pytest.approx(-0.5)
    assert d.value_sek == pytest.approx(-1.0)


def test_stored_delta_unpriced_slot_still_counts_toward_the_stock():
    """Energy moved whether or not a price was recorded — the stock must not lose it.

    This is the one place the function must NOT copy compute_savings, which skips
    unpriced slots outright.
    """
    d = compute_stored_energy_delta(
        [
            _slot(pv=5.0, p_imp=None, p_exp=None, batt_charge=2.0, batt_discharge=0.0),
            _slot(pv=5.0, p_imp=2.0, p_exp=1.0, batt_charge=1.0, batt_discharge=0.0),
        ],
        roundtrip_efficiency=1.0,
    )
    assert d.net_stored_kwh == pytest.approx(3.0)  # both slots
    assert d.charge_kwh == pytest.approx(1.0)  # only the priced one sets the basis
    assert d.basis_sek_kwh == pytest.approx(1.0)


def test_stored_delta_no_inflow_falls_back_to_prior_window_basis():
    """The dominant degenerate case: an early-morning cut that only discharged.

    Without a fallback the basis is 0/0 exactly when the overnight discharge needs
    pricing, and the whole term silently reads zero.
    """
    overnight = [_slot(load=3.0, batt_charge=0.0, batt_discharge=3.0)]
    prior = [_slot(pv=6.0, p_exp=0.9, batt_charge=4.0, batt_discharge=0.0)]

    without = compute_stored_energy_delta(overnight, roundtrip_efficiency=1.0)
    assert without.basis_sek_kwh is None
    assert without.value_sek == 0.0  # no basis => no fabricated value
    assert without.net_stored_kwh == pytest.approx(-3.0)  # stock still moved

    with_prior = compute_stored_energy_delta(
        overnight, roundtrip_efficiency=1.0, basis_rows=prior
    )
    assert with_prior.basis_sek_kwh == pytest.approx(0.9)
    assert with_prior.value_sek == pytest.approx(-2.7)  # the night's draw, priced


def test_stored_delta_basis_fallback_does_not_recurse_past_depth_one():
    """A prior window that itself has no inflow must resolve to None, not recurse."""
    d = compute_stored_energy_delta(
        [_slot(load=1.0, batt_charge=0.0, batt_discharge=1.0)],
        roundtrip_efficiency=1.0,
        basis_rows=[_slot(load=1.0, batt_charge=0.0, batt_discharge=1.0)],
    )
    assert d.basis_sek_kwh is None
    assert d.value_sek == 0.0
