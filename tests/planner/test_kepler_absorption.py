"""C4: absorption beats cheap export.

With the corrected export price (C1), near-zero midday export compensation makes storing
surplus PV for an expensive evening strictly better than exporting it. This is the same
arbitrage the optimizer already does for the battery; the test pins the integrated
behaviour so a future change can't silently reintroduce "dump surplus at ~0 then buy it
back expensive in the evening" (the user's original complaint).
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot


def test_stores_cheap_midday_surplus_instead_of_exporting():
    start = datetime(2025, 6, 1, 12, 0)
    slots = [
        # Midday: 2 kWh surplus PV, export pays ~nothing (0.02), import cheap.
        KeplerInputSlot(
            start_time=start,
            end_time=start + timedelta(hours=1),
            load_kwh=0.0,
            pv_kwh=2.0,
            import_price_sek_kwh=0.5,
            export_price_sek_kwh=0.02,
        ),
        # Evening: 2 kWh load, expensive import, no PV.
        KeplerInputSlot(
            start_time=start + timedelta(hours=1),
            end_time=start + timedelta(hours=2),
            load_kwh=2.0,
            pv_kwh=0.0,
            import_price_sek_kwh=2.0,
            export_price_sek_kwh=0.5,
        ),
    ]
    config = KeplerConfig(
        capacity_kwh=10.0,
        max_charge_power_kw=10.0,
        max_discharge_power_kw=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        wear_cost_sek_per_kwh=0.05,
        curtailment_penalty_sek=0.001,
        enable_export=True,
    )

    result = KeplerSolver().solve(KeplerInput(slots=slots, initial_soc_kwh=0.0), config)
    assert result.is_optimal

    # Midday: absorb the surplus into the battery rather than export it at ~0.
    assert result.slots[0].charge_kwh == pytest.approx(2.0, abs=0.05)
    assert result.slots[0].grid_export_kwh == pytest.approx(0.0, abs=0.05)

    # Evening: discharge to serve the load instead of importing at 2.0 SEK/kWh.
    assert result.slots[1].discharge_kwh == pytest.approx(2.0, abs=0.05)
    assert result.slots[1].grid_import_kwh == pytest.approx(0.0, abs=0.05)
