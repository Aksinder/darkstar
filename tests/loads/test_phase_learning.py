"""Tests for phase-aware load modeling (device->phase learning + per-phase load).

Scenarios are physically consistent: we choose the per-phase *house load* and the
balanced inverter output S, then derive each grid phase as ``load_phase - S/3`` (the
inverter supplies its AC output equally across the three phases). This exercises the
exact masking the relative-to-mean method is designed to remove - S can vary wildly
yet a single-phase device must still map to its phase.
"""

from datetime import datetime, timedelta

from backend.learning.cycle_learning import PowerSample
from backend.learning.phase_learning import (
    learn_device_phase,
    phase_imbalance_w,
    reconstruct_phase_fractions,
)

T0 = datetime(2026, 6, 2, 2, 0, 0)


def _series(values):
    """Build a PowerSample series from a list of watts at 1-minute spacing."""
    return [PowerSample(ts=T0 + timedelta(minutes=i), power_w=float(v)) for i, v in enumerate(values)]


def _grid_from_loads(load_a, load_b, load_c, inv):
    """Derive the three grid-phase series from per-phase loads + balanced inverter S.

    grid_phase = load_phase - S/3  (inverter supplies S equally across phases).
    Returns (device-agnostic) grid A/B/C series.
    """
    ga, gb, gc = [], [], []
    for la, lb, lc, s in zip(load_a, load_b, load_c, inv, strict=True):
        share = s / 3.0
        ga.append(la - share)
        gb.append(lb - share)
        gc.append(lc - share)
    return _series(ga), _series(gb), _series(gc)


# A device that toggles off/on every two samples (clean >200 W steps).
_TOGGLE = [0, 0, 3000, 3000, 0, 0, 3000, 3000, 0, 0, 3000, 3000, 0, 0, 3000, 3000]
# Inverter output deliberately varies a lot, to prove the masking is cancelled.
_INV = [0, 1500, 3000, 4500, 6000, 4500, 3000, 1500, 0, 2000, 4000, 6000, 4000, 2000, 0, 2500]


class TestLearnDevicePhase:
    def test_single_phase_load_maps_to_its_phase(self):
        load_a = _TOGGLE
        zeros = [0] * len(_TOGGLE)
        ga, gb, gc = _grid_from_loads(load_a, zeros, zeros, _INV)
        m = learn_device_phase(_series(load_a), ga, gb, gc, device_id="easee")
        assert m.load_type == "single"
        assert m.phase == "A"
        assert m.confidence > 0.8
        # Textbook signature: +2/3 on the true phase, -1/3 on the others.
        assert m.slopes["A"] > 0.5
        assert m.slopes["B"] < 0.0
        assert m.slopes["C"] < 0.0

    def test_inverter_variation_does_not_fool_the_mapping(self):
        # Same as above but with an even more violent inverter swing; the
        # relative-to-mean cancellation must still recover phase B.
        zeros = [0] * len(_TOGGLE)
        wild = [0, 8000, -4000, 9000, -2000, 7000, 0, 8000, -4000, 9000, -2000, 7000, 0, 8000, -4000, 9000]
        ga, gb, gc = _grid_from_loads(zeros, _TOGGLE, zeros, wild)
        m = learn_device_phase(_series(_TOGGLE), ga, gb, gc, device_id="washer")
        assert m.load_type == "single"
        assert m.phase == "B"

    def test_balanced_three_phase_load_is_phase_neutral(self):
        # A 3-phase heater: total device power split equally across phases.
        per_phase = [v / 3.0 for v in _TOGGLE]
        ga, gb, gc = _grid_from_loads(per_phase, per_phase, per_phase, _INV)
        m = learn_device_phase(_series(_TOGGLE), ga, gb, gc, device_id="house_vvb")
        assert m.load_type == "three_phase"
        assert m.phase is None
        assert max(abs(s) for s in m.slopes.values()) < 0.2

    def test_too_few_steps_is_unknown(self):
        load_a = [0, 0, 3000, 3000]  # only ~1 step over threshold
        zeros = [0] * 4
        ga, gb, gc = _grid_from_loads(load_a, zeros, zeros, [0, 0, 0, 0])
        m = learn_device_phase(_series(load_a), ga, gb, gc)
        assert m.load_type == "unknown"
        assert m.confidence == 0.0

    def test_flat_device_is_unknown(self):
        flat = [500] * 8
        ga, gb, gc = _grid_from_loads(flat, [0] * 8, [0] * 8, [0] * 8)
        m = learn_device_phase(_series(flat), ga, gb, gc)
        assert m.load_type == "unknown"


class TestPhaseImbalance:
    def test_balanced_is_zero(self):
        assert phase_imbalance_w(100.0, 100.0, 100.0) == 0.0

    def test_heavy_phase_shows_positive_imbalance(self):
        # 2000 / 1000 / 1000 -> mean 1333.3, max 2000 -> imbalance ~666.7.
        assert abs(phase_imbalance_w(2000.0, 1000.0, 1000.0) - 666.67) < 0.5

    def test_immune_to_balanced_injection(self):
        # Adding the same export on every phase must not change the imbalance.
        base = phase_imbalance_w(2000.0, 1000.0, 1000.0)
        shifted = phase_imbalance_w(2000.0 - 500, 1000.0 - 500, 1000.0 - 500)
        assert abs(base - shifted) < 1e-9


class TestReconstructPhaseFractions:
    def test_recovers_known_fractions(self):
        n = 20
        load_a = [2000.0] * n  # 50%
        load_b = [1000.0] * n  # 25%
        load_c = [1000.0] * n  # 25%
        inv = [float(s) for s in (0, 1500, 3000, 4500, 6000) * 4]
        ga, gb, gc = _grid_from_loads(load_a, load_b, load_c, inv)
        est = reconstruct_phase_fractions(ga, gb, gc, _series(inv))
        assert abs(est.fractions["A"] - 0.5) < 0.02
        assert abs(est.fractions["B"] - 0.25) < 0.02
        assert abs(est.fractions["C"] - 0.25) < 0.02
        assert est.n_samples == n
        assert est.imbalance_w > 600  # 2000 vs 1000/1000

    def test_low_load_samples_are_skipped(self):
        # All total loads below the floor -> nothing usable.
        tiny = [50.0] * 6
        ga, gb, gc = _grid_from_loads(tiny, tiny, tiny, [0] * 6)
        est = reconstruct_phase_fractions(ga, gb, gc, _series([0] * 6), min_total_w=300.0)
        assert est.n_samples == 0
        assert est.fractions == {}
