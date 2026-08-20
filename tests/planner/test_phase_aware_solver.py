"""Tests for the phase-aware FUSE RELIEF term in the Kepler MILP.

Repurposed 2026-08-20 (owner: "phase_aware borde bara vara ett verktyg for att
hjalpa till att fixa snedbelastning"). Swedish settlement meters net all three
phases momentarily (STAFS 2022:9 2 kap. 7 §), so the original form of this term —
per-phase import priced at the import price — modeled a bill that does not exist,
and double-priced ordinary net import on top of the objective's own import cost.

The term now prices only a phase's grid current ABOVE phase_relief_start_a, in
SEK per ampere-hour: zero for a comfortably balanced house, a steep incentive to
discharge (shaving supply/3 of real amps off the heavy phase) near the fuse.
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot

START = datetime(2026, 6, 2, 18, 0)
HEAVY_C = {"A": 0.05, "B": 0.05, "C": 0.9}
BALANCED = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}


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


# 3.0 kWh per 15-min slot with 90% on C and pv 3.0: C's grid deficit is
# 2.7 - supply/3 = 1.7 kWh => 6.8 kW ~ 30 A, over the 20 A default threshold.
# The same energy balanced leaves every phase's deficit at zero.


class TestFuseRelief:
    def test_off_parity(self):
        """Disabled (even with fractions populated) == no phase config at all."""
        slots = _slots([(2.0, 2.0, 2.0)] * 4)
        base = KeplerSolver().solve(KeplerInput(slots, 8.0), _cfg())
        off = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=False, phase_load_fractions=HEAVY_C),
        )
        assert _discharge(base) == pytest.approx(_discharge(off))
        assert base.total_cost_sek == pytest.approx(off.total_cost_sek)

    def test_discharges_to_shave_amps_off_a_phase_near_the_fuse(self):
        """Net-balanced (pv == load) but phase C carries 90% => ~31 A on C. Extra
        discharge is the only relief the solver owns (supply/3 off each phase)."""
        slots = _slots([(2.0, 3.0, 3.0)] * 4)
        off = KeplerSolver().solve(KeplerInput(slots, 8.0), _cfg())
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=True, phase_load_fractions=HEAVY_C),
        )
        assert _discharge(off) == pytest.approx(0.0, abs=0.05)
        assert _discharge(on) > _discharge(off) + 0.5

    def test_a_balanced_house_pays_nothing_and_does_nothing(self):
        """The same energy spread evenly sits at ~3 A per phase — far inside the
        threshold. The term must be inert: no discharge, no cost."""
        slots = _slots([(2.0, 3.0, 3.0)] * 4)
        off = KeplerSolver().solve(KeplerInput(slots, 8.0), _cfg())
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(phase_aware_enabled=True, phase_load_fractions=BALANCED),
        )
        assert _discharge(on) == pytest.approx(_discharge(off), abs=0.1)
        assert on.total_cost_sek == pytest.approx(off.total_cost_sek, abs=0.05)

    def test_ordinary_import_is_not_double_priced(self):
        """THE regression this rewrite exists for. The old term priced per-phase
        import at the import price; on an all-phases-in-deficit slot the phase
        deficits sum to exactly the net import, so grid import was charged twice.
        A balanced grid-importing night (no PV, no battery) must cost the same
        with the term on as off — the plan's economics, not 2x them."""
        slots = _slots([(3.0, 1.5, 0.0)] * 4)  # pure import, ~7 A per phase
        off = KeplerSolver().solve(KeplerInput(slots, 0.0), _cfg())
        on = KeplerSolver().solve(
            KeplerInput(slots, 0.0),
            _cfg(phase_aware_enabled=True, phase_load_fractions=BALANCED),
        )
        assert on.total_cost_sek == pytest.approx(off.total_cost_sek, abs=1e-6)

    def test_threshold_is_in_amps(self):
        """Raise relief_start_a above the heavy phase's draw and the term goes
        quiet; the amps, not the imbalance ratio, decide."""
        slots = _slots([(2.0, 3.0, 3.0)] * 4)
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(
                phase_aware_enabled=True,
                phase_load_fractions=HEAVY_C,
                phase_relief_start_a=40.0,  # C's deficit peaks ~30 A: under this
            ),
        )
        assert _discharge(on) == pytest.approx(0.0, abs=0.1)

    def test_per_hour_profile_overrides_static(self):
        """Balanced static split but hour 18 heavy on C: the profile decides."""
        slots = _slots([(2.0, 3.0, 3.0)] * 4)  # all at hour 18
        on = KeplerSolver().solve(
            KeplerInput(slots, 8.0),
            _cfg(
                phase_aware_enabled=True,
                phase_load_fractions=BALANCED,
                phase_load_profile={18: HEAVY_C},
            ),
        )
        assert _discharge(on) > 0.5
