"""Tests for phase rebalancing recommendations.

Scenarios are physically consistent: grid_phase = (loads on that phase) - (balanced
PV/battery export per phase). Hidden waste appears only when a heavy phase imports
while light phases export - exactly the case a move can fix.
"""

from datetime import datetime, timedelta

from backend.learning.cycle_learning import PowerSample
from backend.learning.phase_learning import PhaseMapping
from backend.learning.phase_recommend import recommend_phase_moves

T0 = datetime(2026, 6, 2, 0, 0, 0)


def _ser(values, step_min=5):
    return [PowerSample(ts=T0 + timedelta(minutes=i * step_min), power_w=float(v)) for i, v in enumerate(values)]


def _const(v, n=24):
    return _ser([v] * n)


def _map(device_id, phase, *, confidence=0.9, load_type="single"):
    return PhaseMapping(
        device_id=device_id, phase=phase, load_type=load_type, confidence=confidence
    )


class TestRecommendPhaseMoves:
    def test_stacked_single_phase_loads_get_a_move(self):
        # Two 2 kW loads both on A, with 1.5 kW PV export per phase.
        # grid: A = 4000-1500 = 2500 (importing), B = C = -1500 (exporting).
        ga, gb, gc = _const(2500), _const(-1500), _const(-1500)
        d1, d2 = _const(2000), _const(2000)
        recs = recommend_phase_moves(
            ga,
            gb,
            gc,
            [(_map("d1", "A"), d1), (_map("d2", "A"), d2)],
            names={"d1": "Diskmaskin", "d2": "Tvätt"},
            import_price_sek_kwh=2.0,
            export_price_sek_kwh=0.5,
        )
        assert recs
        top = recs[0]
        assert top.from_phase == "A"
        assert top.to_phase in {"B", "C"}
        assert top.window_import_avoided_kwh > 0
        assert top.annual_import_avoided_kwh > top.window_import_avoided_kwh
        assert top.annual_saving_sek > 0
        assert top.device_name in {"Diskmaskin", "Tvätt"}
        # Moving reduces the per-phase spread.
        assert top.imbalance_after_w < top.imbalance_before_w

    def test_no_hidden_waste_yields_no_recommendation(self):
        # Single load on A, battery empty (no export anywhere): the phase import
        # equals the net import, so there is nothing to save by moving it.
        ga, gb, gc = _const(2000), _const(0), _const(0)
        d1 = _const(2000)
        recs = recommend_phase_moves(ga, gb, gc, [(_map("d1", "A"), d1)])
        assert recs == []

    def test_already_split_loads_yield_no_recommendation(self):
        # 2 kW on A and 2 kW on B with 1.5 kW export/phase: already balanced as well
        # as a single move can manage, so no positive-benefit move exists.
        ga, gb, gc = _const(500), _const(500), _const(-1500)
        d1, d2 = _const(2000), _const(2000)
        recs = recommend_phase_moves(
            ga, gb, gc, [(_map("d1", "A"), d1), (_map("d2", "B"), d2)]
        )
        assert recs == []

    def test_low_confidence_device_skipped(self):
        ga, gb, gc = _const(2500), _const(-1500), _const(-1500)
        d1 = _const(2000)
        recs = recommend_phase_moves(
            ga, gb, gc, [(_map("d1", "A", confidence=0.2), d1)], min_confidence=0.5
        )
        assert recs == []

    def test_three_phase_device_skipped(self):
        ga, gb, gc = _const(2500), _const(-1500), _const(-1500)
        d1 = _const(2000)
        recs = recommend_phase_moves(
            ga, gb, gc, [(_map("d1", None, load_type="three_phase"), d1)]
        )
        assert recs == []

    def test_picks_the_lighter_target_phase(self):
        # Heavy on A; C is lighter than B, so the device should move to C.
        # grid: A=3000 (2kW load - ... ), B=-500, C=-2500 -> moving load to C balances best.
        ga, gb, gc = _const(3000), _const(-500), _const(-2500)
        d1 = _const(2000)
        recs = recommend_phase_moves(ga, gb, gc, [(_map("d1", "A"), d1)])
        assert recs
        assert recs[0].to_phase == "C"
