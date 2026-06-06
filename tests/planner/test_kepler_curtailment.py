"""C2: price-conditioned curtailment.

When the effective export price is negative the planner must CURTAIL surplus PV (clip it)
rather than export it — exporting at a negative price means paying to export. When the
export price is positive the same surplus is exported instead. This is governed by a
near-zero ``curtailment_penalty_sek`` so that curtailment beats negative-price export, yet
never beats actually using/storing the PV (which carries real value).
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot


def _full_battery_surplus(export_price: float, pv_kwh: float = 1.0) -> KeplerInput:
    """One slot: full battery, no load, `pv_kwh` surplus -> must export or curtail."""
    start = datetime(2025, 6, 1, 12, 0)
    slots = [
        KeplerInputSlot(
            start_time=start,
            end_time=start + timedelta(minutes=15),
            load_kwh=0.0,
            pv_kwh=pv_kwh,
            import_price_sek_kwh=1.0,
            export_price_sek_kwh=export_price,
        )
    ]
    return KeplerInput(slots=slots, initial_soc_kwh=10.0)  # full (== capacity)


def _config(curtailment_penalty: float = 0.001) -> KeplerConfig:
    return KeplerConfig(
        capacity_kwh=10.0,
        max_charge_power_kw=10.0,
        max_discharge_power_kw=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        wear_cost_sek_per_kwh=0.2,  # makes discharge-to-export unprofitable at small prices
        curtailment_penalty_sek=curtailment_penalty,
        enable_export=True,
    )


class TestPriceConditionedCurtailment:
    def test_curtails_instead_of_exporting_at_negative_price(self):
        result = KeplerSolver().solve(_full_battery_surplus(-0.05), _config())
        assert result.is_optimal
        # Surplus PV is clipped, not exported — we won't pay to export.
        assert result.slots[0].grid_export_kwh == pytest.approx(0.0, abs=0.01)
        assert result.slots[0].discharge_kwh == pytest.approx(0.0, abs=0.01)

    def test_curtails_even_at_shallow_negative_price(self):
        # The old penalty (0.1) made the model export at -0.02 because exporting (0.02)
        # was cheaper than curtailing (0.1). With 0.001 it correctly curtails.
        result = KeplerSolver().solve(_full_battery_surplus(-0.02), _config())
        assert result.is_optimal
        assert result.slots[0].grid_export_kwh == pytest.approx(0.0, abs=0.01)

    def test_exports_at_positive_price(self):
        # Small positive price: export the free PV surplus (0.05 > 0.001 curtail cost), but
        # do not discharge the battery to export (0.05 < 0.1 wear per discharged kWh).
        result = KeplerSolver().solve(_full_battery_surplus(0.05), _config())
        assert result.is_optimal
        assert result.slots[0].grid_export_kwh == pytest.approx(1.0, abs=0.01)
        assert result.slots[0].discharge_kwh == pytest.approx(0.0, abs=0.01)

    def test_penalty_threshold_characterization(self):
        # Characterizes the crossover: a penalty above |negative price| makes export win
        # again (the pre-C2 behaviour). Guards the chosen near-zero default.
        result = KeplerSolver().solve(_full_battery_surplus(-0.05), _config(0.1))
        assert result.is_optimal
        assert result.slots[0].grid_export_kwh == pytest.approx(1.0, abs=0.01)
