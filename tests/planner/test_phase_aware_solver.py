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

import pulp
import pytest

from planner.solver.adapter import config_to_kepler_config, normalize_phase_map
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    EVChargerInput,
    IncentiveBucket,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    WaterHeaterInput,
)

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


# ---------------------------------------------------------------------------
# Per-device phase allocation (EV per charger, water heaters incl. spa)
# ---------------------------------------------------------------------------
HEAVY_A = {"A": 0.9, "B": 0.05, "C": 0.05}
_BATT = {"capacity_kwh": 13.5, "max_charge_a": 100, "max_discharge_a": 100}


def _fmb(phases=None):
    # 3.7 kW single-phase (16 A): ~3.7 kWh wanted = one hour of charging.
    return EVChargerInput(
        id="easee",
        max_power_kw=3.7,
        battery_capacity_kwh=28.0,
        current_soc_percent=50.0,
        plugged_in=True,
        deadline=None,
        incentive_buckets=[IncentiveBucket(threshold_soc=50.0 + 370 / 28, value_sek=5.0)],
        phases=list(phases or []),
    )


def _spa(phases=None):
    return WaterHeaterInput(
        id="spa",
        power_kw=1.8,
        min_kwh_per_day=0.0,
        max_hours_between_heating=48.0,
        min_spacing_hours=0.0,
        phases=list(phases or []),
    )


def _two_hours():
    """18:00-20:00 in quarters, no PV. 0.958 kWh/quarter = 3.83 kW base load;
    hour 18 is 90% on A (~15 A on A), hour 19 is balanced (~5.5 A per phase).
    Hour 18 is slightly CHEAPER, so price alone puts the EV there."""
    return _slots([(1.0, 0.958, 0.0)] * 4 + [(1.2, 0.958, 0.0)] * 4)


def _ev_cfg(**kw):
    return _cfg(
        capacity_kwh=0.0,
        max_charge_power_kw=0.0,
        max_discharge_power_kw=0.0,
        ev_chargers=kw.pop("ev_chargers"),
        phase_load_profile={18: HEAVY_A, 19: BALANCED},
        **kw,
    )


def _ev_kwh_by_hour(result):
    out = {18: 0.0, 19: 0.0}
    for s in result.slots:
        out[s.start_time.hour] += s.ev_charge_kw * 0.25
    return out


class _Capture:
    """Record every LpProblem the solver hands to a backend."""

    def __init__(self, monkeypatch):
        self.probs = []
        orig = pulp.LpProblem.solve

        def _solve(prob, *a, **k):
            self.probs.append(prob)
            return orig(prob, *a, **k)

        monkeypatch.setattr(pulp.LpProblem, "solve", _solve)

    def relief_rows(self):
        prob = self.probs[-1]
        return [
            c
            for c in prob.constraints.values()
            if any(v.name.startswith("phase_over_") for v in c)
        ]


class TestPerDevicePhases:
    def test_ev_on_phase_a_moves_to_the_hour_phase_a_has_headroom(self):
        off = KeplerSolver().solve(KeplerInput(_two_hours(), 0.0), _ev_cfg(ev_chargers=[_fmb()]))
        assert _ev_kwh_by_hour(off)[18] == pytest.approx(3.7, abs=0.05)

        on = KeplerSolver().solve(
            KeplerInput(_two_hours(), 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[_fmb(["A"])]),
        )
        by_hour = _ev_kwh_by_hour(on)
        assert by_hour[19] == pytest.approx(3.7, abs=0.05)
        assert by_hour[18] == pytest.approx(0.0, abs=0.05)

    def test_ev_on_a_light_phase_stays_on_the_cheap_hour(self):
        on = KeplerSolver().solve(
            KeplerInput(_two_hours(), 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[_fmb(["B"])]),
        )
        assert _ev_kwh_by_hour(on)[18] == pytest.approx(3.7, abs=0.05)

    def test_ev_without_phases_is_not_moved(self):
        """Unknown phase = even split: 16 A/3 never crosses the threshold here."""
        on = KeplerSolver().solve(
            KeplerInput(_two_hours(), 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[_fmb()]),
        )
        assert _ev_kwh_by_hour(on)[18] == pytest.approx(3.7, abs=0.05)

    def test_no_device_phases_gives_the_base_load_only_rows(self, monkeypatch):
        """Behaviour parity: with relief ON but no device phases, every relief row is
        exactly pi >= load*f - (pv + dis - chg)/n - start_a*0.23*h — no EV or water
        variable in it, constants identical to the hard-coded 0.23 V form."""
        cap = _Capture(monkeypatch)
        slots = _two_hours()
        cfg = _ev_cfg(phase_aware_enabled=True, ev_chargers=[_fmb()], water_heaters=[_spa()])
        KeplerSolver().solve(KeplerInput(slots, 0.0), cfg)
        rows = cap.relief_rows()
        assert len(rows) == len(slots) * 3
        for c in rows:
            names = {v.name for v in c}
            assert all(
                n.startswith(("phase_over_", "charge", "discharge")) for n in names
            ), names
            pi = next(v for v in c if v.name.startswith("phase_over_"))
            _, t, k = pi.name.rsplit("_", 2)
            s = slots[int(t)]
            f = (HEAVY_A if s.start_time.hour == 18 else BALANCED)["ABC"[int(k)]]
            relief = cfg.phase_relief_start_a * 0.23 * 0.25
            # c: pi - load*f + pv/3 + ... + relief >= 0  -> constant = -(load*f) + relief
            assert c.constant == -(s.load_kwh * f - s.pv_kwh / 3 - relief)

    def test_device_phases_only_change_coefficients(self, monkeypatch):
        cap = _Capture(monkeypatch)
        slots = _two_hours()
        KeplerSolver().solve(
            KeplerInput(slots, 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[_fmb(["A"])], water_heaters=[_spa()]),
        )
        rows = cap.relief_rows()
        assert len(rows) == len(slots) * 3  # same rows, same pi vars
        seen = {}
        for c in rows:
            pi = next(v for v in c if v.name.startswith("phase_over_"))
            _, t, k = pi.name.rsplit("_", 2)
            seen[(int(t), int(k))] = {v.name: a for v, a in c.items()}
        for t in range(len(slots)):
            # Phase A (heavy in hour 18, and the EV's phase): EV fully, spa 1/3.
            a_row = seen[(t, 0)]
            ev = [n for n in a_row if n.startswith("ev_energy_")]
            assert ev and a_row[ev[0]] == pytest.approx(-1.0)
            wh = [n for n in a_row if n.startswith("water_heat_spa")]
            assert wh and a_row[wh[0]] == pytest.approx(-1.8 * 0.25 / 3)
            # Phase B never carries the EV. In hour 18 B is so light that even the
            # spa at full power cannot cross the threshold: the row stays base-only.
            assert not any(n.startswith("ev_energy_") for n in seen[(t, 1)])
            if t < 4:
                assert not any(n.startswith("water_heat_") for n in seen[(t, 1)])

    def test_relief_off_ignores_device_phases(self, monkeypatch):
        cap = _Capture(monkeypatch)
        KeplerSolver().solve(
            KeplerInput(_two_hours(), 0.0),
            _ev_cfg(phase_aware_enabled=False, ev_chargers=[_fmb(["A"])]),
        )
        assert cap.relief_rows() == []

    def test_phase_known_fixed_load_lands_on_its_phase(self, monkeypatch):
        """A cyclic pump pre-scheduled as fixed load with phase_map [c] is taken out
        of the profile split and put on C."""
        cap = _Capture(monkeypatch)
        slots = _two_hours()
        slots[0].phased_load_kwh = {"C": 0.2}
        KeplerSolver().solve(
            KeplerInput(slots, 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[]),
        )
        for c in cap.relief_rows():
            pi = next(v for v in c if v.name.startswith("phase_over_"))
            _, t, k = pi.name.rsplit("_", 2)
            if t != "0":
                continue
            f = HEAVY_A["ABC"[int(k)]]
            extra = 0.2 if k == "2" else 0.0
            relief = 20.0 * 0.23 * 0.25
            assert c.constant == pytest.approx(-((0.958 - 0.2) * f + extra - relief))

    def test_nominal_voltage_scales_the_threshold(self, monkeypatch):
        cap = _Capture(monkeypatch)
        slots = _two_hours()
        KeplerSolver().solve(
            KeplerInput(slots, 0.0),
            _ev_cfg(phase_aware_enabled=True, ev_chargers=[], phase_nominal_voltage_v=240.0),
        )
        for c in cap.relief_rows():
            pi = next(v for v in c if v.name.startswith("phase_over_"))
            _, t, k = pi.name.rsplit("_", 2)
            f = (HEAVY_A if int(t) < 4 else BALANCED)["ABC"[int(k)]]
            assert c.constant == pytest.approx(-(0.958 * f - 20.0 * 0.24 * 0.25))


class TestPhaseConfigParsing:
    def test_normalize_phase_map(self):
        assert normalize_phase_map(["a", "L2", 3, "a"]) == ["A", "B", "C"]
        assert normalize_phase_map(["x"]) == []
        assert normalize_phase_map(None) == []

    def test_adapter_reads_phase_maps_and_voltage(self):
        cfg = {
            "config_version": 2,
            "battery": _BATT,
            "water_heaters": [
                {"id": "main_tank", "power_kw": 3.0, "phase_map": ["a", "b", "c"]},
                {"id": "spa", "power_kw": 1.8},
            ],
            "ev_chargers": [
                {"id": "easee", "max_power_kw": 3.7},
                {"id": "tesla", "max_power_kw": 11.0, "phase_map": ["a", "b", "c"]},
            ],
            "executor": {"ev_surplus": {"chargers": [{"id": "easee", "phase_map": ["a"]}]}},
            "load_balancing": {"nominal_voltage_v": 240},
        }
        kc = config_to_kepler_config(cfg)
        assert {w.id: w.phases for w in kc.water_heaters} == {
            "main_tank": ["A", "B", "C"],
            "spa": [],
        }
        assert {c.id: c.phases for c in kc.ev_chargers} == {
            "easee": ["A"],
            "tesla": ["A", "B", "C"],
        }
        assert kc.phase_nominal_voltage_v == 240.0

    def test_voltage_defaults_to_230(self):
        assert config_to_kepler_config({"battery": _BATT}).phase_nominal_voltage_v == 230.0
