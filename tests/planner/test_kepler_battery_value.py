"""Improvement B: continuous stored-energy value (terminal credit on soc[T]).

Proves the terminal credit (a) makes the planner hold cheaply-stored energy instead
of dumping it for a low export price WITHOUT a hard SoC floor, (b) does not over-reach
into buying grid just to inflate the terminal SoC (the bound that keeps it sane), and
(c) still discharges to avoid an expensive import (no hoarding when using energy is
worth more). The credit is on soc[T] only, so it cannot distort mid-horizon cycling
(the K20 failure mode).
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot
from planner.strategy.s_index import derive_battery_value_sek_per_kwh

START = datetime(2025, 1, 1, 18, 0)


def _slots(specs):
    out = []
    for i, (load, pv, imp, exp) in enumerate(specs):
        s = START + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=load,
                pv_kwh=pv,
                import_price_sek_kwh=imp,
                export_price_sek_kwh=exp,
            )
        )
    return out


def _config(*, battery_value=0.0, target=None):
    return KeplerConfig(
        capacity_kwh=10.0,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        wear_cost_sek_per_kwh=0.01,
        target_soc_kwh=target,
        battery_value_sek_per_kwh=battery_value,
    )


def test_holds_energy_instead_of_cheap_export():
    # 4 idle slots, export worth only 0.05/kWh, NO hard floor (target=None).
    specs = [(0.0, 0.0, 1.0, 0.05)] * 4
    inp = KeplerInput(slots=_slots(specs), initial_soc_kwh=5.0)

    # Without a stored-energy value, exporting at 0.05 beats holding -> it drains.
    no_value = KeplerSolver().solve(inp, _config(battery_value=0.0, target=None))
    assert no_value.is_optimal
    assert no_value.slots[-1].soc_kwh == pytest.approx(0.0, abs=0.2)
    assert sum(s.grid_export_kwh for s in no_value.slots) == pytest.approx(5.0, abs=0.2)

    # With a stored-energy value above the export price, it holds the energy instead.
    with_value = KeplerSolver().solve(inp, _config(battery_value=0.10, target=None))
    assert with_value.is_optimal
    assert with_value.slots[-1].soc_kwh == pytest.approx(5.0, abs=0.2)
    assert sum(s.grid_export_kwh for s in with_value.slots) == pytest.approx(0.0, abs=0.2)


def test_does_not_buy_grid_to_inflate_terminal_soc():
    # Stored-energy value (0.6) is BELOW the import price (1.0): buying grid to hold
    # would be a loss, so the planner must not charge from the grid. It just sits.
    specs = [(0.0, 0.0, 1.0, 0.5)] * 4
    inp = KeplerInput(slots=_slots(specs), initial_soc_kwh=5.0)
    res = KeplerSolver().solve(inp, _config(battery_value=0.6, target=None))
    assert res.is_optimal
    assert sum(s.grid_import_kwh for s in res.slots) == pytest.approx(0.0, abs=0.05)
    assert sum(s.charge_kwh for s in res.slots) == pytest.approx(0.0, abs=0.05)
    # 0.6 (hold) > 0.5 (export) -> it should also not export.
    assert sum(s.grid_export_kwh for s in res.slots) == pytest.approx(0.0, abs=0.2)
    assert res.slots[-1].soc_kwh == pytest.approx(5.0, abs=0.2)


def test_still_discharges_to_avoid_expensive_import():
    # A real load at a high import price: using the battery saves 2.0/kWh, far above
    # the 0.6 hold value, so it must discharge to serve the load (no hoarding). Load
    # is 1 kWh/slot, within the 1.25 kWh/slot discharge cap, so the battery covers it.
    specs = [(1.0, 0.0, 2.0, 0.5)] * 2
    inp = KeplerInput(slots=_slots(specs), initial_soc_kwh=5.0)
    res = KeplerSolver().solve(inp, _config(battery_value=0.6, target=None))
    assert res.is_optimal
    assert sum(s.discharge_kwh for s in res.slots) == pytest.approx(2.0, abs=0.1)
    assert sum(s.grid_import_kwh for s in res.slots) == pytest.approx(0.0, abs=0.1)


class TestDeriveBatteryValue:
    def test_empty_prices_zero(self):
        assert derive_battery_value_sek_per_kwh([]) == 0.0

    def test_uses_min_forward_price_scaled(self):
        # min is 1.0; scale 0.75 -> 0.75.
        assert derive_battery_value_sek_per_kwh([2.0, 1.0, 3.0], scale=0.75) == pytest.approx(0.75)

    def test_never_exceeds_min_forward_price(self):
        # The bound that prevents buying grid to inflate terminal SoC.
        prices = [1.2, 0.8, 2.5, 1.0]
        v = derive_battery_value_sek_per_kwh(prices, scale=1.0, wear_cost_sek_per_kwh=0.1)
        assert v <= min(prices)

    def test_lookahead_window_limits_prices(self):
        # The cheap 0.1 slots are beyond the 2-slot window, so floor stays 1.0.
        prices = [1.0, 1.0, 0.1, 0.1, 0.1]
        v = derive_battery_value_sek_per_kwh(prices, lookahead_slots=2, scale=1.0)
        assert v == pytest.approx(1.0)

    def test_wear_is_subtracted_and_clamped(self):
        assert derive_battery_value_sek_per_kwh([1.0], scale=1.0, wear_cost_sek_per_kwh=0.4) == pytest.approx(0.8)
        assert derive_battery_value_sek_per_kwh([0.1], scale=0.0) == 0.0
