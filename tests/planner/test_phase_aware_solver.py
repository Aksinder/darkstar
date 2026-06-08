"""Tests for the phase-aware imbalance cost in the Kepler MILP.

The single-net-node LP nets a heavy phase's import against the light phases' export to
~zero. The phase-aware term prices that hidden cost so the solver discharges to cover the
heavy phase — but only WHEN ECONOMIC (the avoided import must beat the spilled export plus
the battery value spent). Flag-gated; off => byte-identical to before.
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot

START = datetime(2026, 6, 2, 18, 0)
HEAVY_C = {"A": 0.05, "B": 0.05, "C": 0.9}


def _slots(rows):
    """rows: list of (import_price, load_kwh, pv_kwh); export fixed at 0.1."""
    out = []
    for i, (imp, load, pv) in enumerate(rows):
        s = START + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=load,
                pv_kwh=pv,
                import_price_sek_kwh=imp,
                export_price_sek_kwh=0.1,
            )
        )
    return out


def _cfg(**kw):
    base = {
        "capacity_kwh": 10.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        # Small wear so a free charge+discharge cycle isn't a tie-broken artifact.
        "wear_cost_sek_per_kwh": 0.01,
        # Keep the battery from dumping to export for its own sake (export 0.1 < value).
        "battery_value_sek_per_kwh": 0.5,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _discharge(result):
    return sum(s.discharge_kwh for s in result.slots)


class TestPhaseAwareSolver:
    def test_off_parity(self):
        # Disabled (even with fractions populated) == no phase config at all.
        slots = _slots([(2.0, 2.0, 2.0)] * 4)
        base = KeplerSolver().solve(KeplerInput(slots, 8.0), _cfg())
        off = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=False, phase_load_fractions=HEAVY_C),
        )
        assert _discharge(base) == pytest.approx(_discharge(off))
        assert base.total_cost_sek == pytest.approx(off.total_cost_sek)

    def test_covers_heavy_phase_when_economic(self):
        # Net-balanced (pv == load) but phase C carries 90% of the load and import is
        # dear (5.0). With phase-aware on, the battery discharges to raise balanced
        # supply and cover C; with it off it sits idle (nothing to net-cover).
        slots = _slots([(5.0, 2.0, 2.0)] * 4)
        off = KeplerSolver().solve(KeplerInput(slots, 8.0), _cfg())
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=True, phase_load_fractions=HEAVY_C),
        )
        assert _discharge(off) == pytest.approx(0.0, abs=0.05)
        assert _discharge(on) > _discharge(off) + 0.5  # battery covers the heavy phase

    def test_no_over_discharge_when_uneconomic(self):
        # Cheap import (0.15, ~export): covering the phase by spilling export + spending
        # battery value is NOT worth it, so the battery stays put.
        slots = _slots([(0.15, 2.0, 2.0)] * 4)
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=True, phase_load_fractions=HEAVY_C),
        )
        assert _discharge(on) == pytest.approx(0.0, abs=0.1)

    def test_per_hour_profile_overrides_static(self):
        # Profile for hour 18 (the slots' hour) says C is heavy; with a balanced static
        # fallback this still drives coverage, proving the per-hour profile is consulted.
        slots = _slots([(5.0, 2.0, 2.0)] * 4)  # all at hour 18
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(
                phase_aware_enabled=True,
                phase_load_fractions={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},  # balanced static
                phase_load_profile={18: HEAVY_C},  # but hour 18 is heavy on C
            ),
        )
        assert _discharge(on) > 0.5
