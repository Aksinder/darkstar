"""Kepler must never import and export in the same slot.

plan_validator warns on it (a meter nets one slot's flows, so the plan is
physically meaningless and confuses the executor). Ties are broken by a tiny
gross-flow cost; slots where export pays more than import costs (deep negative
spot: VAT on import spot only) get a grid-direction binary.
"""

from datetime import datetime, timedelta

import pytest

from planner.solver import kepler as kepler_mod
from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot

TOL = 1e-3


def _slots(prices: list[tuple[float, float]], pv: float = 0.0, load: float = 0.5):
    start = datetime(2026, 6, 1, 12, 0)
    return [
        KeplerInputSlot(
            start_time=start + timedelta(minutes=15 * i),
            end_time=start + timedelta(minutes=15 * (i + 1)),
            load_kwh=load,
            pv_kwh=pv,
            import_price_sek_kwh=imp,
            export_price_sek_kwh=exp,
        )
        for i, (imp, exp) in enumerate(prices)
    ]


def _config(**kw) -> KeplerConfig:
    base = {
        "capacity_kwh": 16.0,
        "max_charge_power_kw": 8.0,
        "max_discharge_power_kw": 8.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "wear_cost_sek_per_kwh": 0.2,
        "max_import_power_kw": 17.0,
        "max_export_power_kw": 17.0,
        "enable_export": True,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _overlap_slots(result) -> list[int]:
    return [
        i for i, s in enumerate(result.slots) if s.grid_import_kwh > TOL and s.grid_export_kwh > TOL
    ]


def test_inverted_prices_do_not_loop_import_into_export():
    """Deep negative spot: export -3.95 > import -4.0. The old model imported and
    exported the full grid limit (a fictional ~0.2 SEK/slot "arbitrage")."""
    prices = [(-4.0, -3.95)] * 4 + [(1.5, 0.6)] * 4
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(prices, pv=1.0), initial_soc_kwh=8.0), _config()
    )
    assert result.is_optimal
    assert _overlap_slots(result) == []
    # It still takes the paid-to-import energy (battery + load), just not via a loop.
    assert sum(s.grid_import_kwh for s in result.slots[:4]) > 1.0


def test_inverted_slot_without_import_limit_uses_fallback_big_m():
    prices = [(-4.0, -3.9)] * 2
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(prices, pv=2.0), initial_soc_kwh=8.0),
        _config(max_import_power_kw=None, max_export_power_kw=None),
    )
    assert result.is_optimal
    assert _overlap_slots(result) == []


def test_equal_prices_tie_has_no_overlap():
    """Adapter fallback export=import (no export column): the LP is indifferent;
    the gross-flow epsilon picks the net solution."""
    prices = [(1.0, 1.0)] * 8
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(prices, pv=2.0, load=0.5), initial_soc_kwh=8.0),
        _config(wear_cost_sek_per_kwh=0.0),
    )
    assert result.is_optimal
    assert _overlap_slots(result) == []


def test_epsilon_does_not_move_economics(monkeypatch):
    """Normal prices: the reported plan cost is unchanged to well under 1 öre."""
    prices = [(0.8 + 0.1 * (i % 6), 0.3 + 0.1 * (i % 6)) for i in range(24)]

    def solve() -> float:
        r = KeplerSolver().solve(
            KeplerInput(slots=_slots(prices, pv=1.2, load=0.6), initial_soc_kwh=6.0),
            _config(),
        )
        assert r.is_optimal
        assert _overlap_slots(r) == []
        return r.total_cost_sek

    with_eps = solve()
    monkeypatch.setattr(kepler_mod, "GRID_FLOW_EPSILON_SEK_PER_KWH", 0.0)
    without_eps = solve()
    assert with_eps == pytest.approx(without_eps, abs=0.01)
