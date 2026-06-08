"""Tests for the per-hour phase profile (per-slot phase forecasting).

reconstruct_phase_profile buckets the reconstructed per-phase house load by local hour
of day, so the planner can learn *when* each phase is heavy (e.g. evening cooking on C)
rather than relying on one static split. load_phase_profile / phase_fractions_for_slot
serve it back to the realism simulation.
"""

import json
from datetime import datetime, timedelta

import pytest

from backend.learning.cycle_learning import PowerSample
from backend.learning.phase_learning import (
    load_phase_profile,
    phase_fractions_for_slot,
    reconstruct_phase_profile,
)

BASE = datetime(2026, 6, 2, 0, 0, 0)


def _at_hour(hour, n, watts):
    """n one-minute PowerSamples in the given hour, all at `watts`."""
    return [PowerSample(ts=BASE + timedelta(hours=hour, minutes=i), power_w=float(watts)) for i in range(n)]


class TestReconstructPhaseProfile:
    def test_buckets_fractions_by_hour(self):
        n = 20  # >= min_samples_per_hour
        # Hour 8: phase A heavy. Hour 18: phase C heavy. Inverter off (load == grid).
        ga = _at_hour(8, n, 800) + _at_hour(18, n, 100)
        gb = _at_hour(8, n, 100) + _at_hour(18, n, 100)
        gc = _at_hour(8, n, 100) + _at_hour(18, n, 800)
        inv = _at_hour(8, n, 0) + _at_hour(18, n, 0)

        profile = reconstruct_phase_profile(ga, gb, gc, inv, min_samples_per_hour=8)

        assert profile[8]["A"] == pytest.approx(0.8, abs=0.01)
        assert profile[18]["C"] == pytest.approx(0.8, abs=0.01)
        # The two hours tell genuinely different stories — the whole point.
        assert profile[8]["A"] > profile[18]["A"]
        assert profile[18]["C"] > profile[8]["C"]

    def test_sparse_hours_fall_back_to_overall_average(self):
        n = 20
        ga = _at_hour(8, n, 800) + _at_hour(18, n, 100)
        gb = _at_hour(8, n, 100) + _at_hour(18, n, 100)
        gc = _at_hour(8, n, 100) + _at_hour(18, n, 800)
        inv = _at_hour(8, n, 0) + _at_hour(18, n, 0)

        profile = reconstruct_phase_profile(ga, gb, gc, inv, min_samples_per_hour=8)

        # Hour 3 has no samples -> overall average (A and C symmetric ~0.45, B ~0.1).
        assert profile[3]["A"] == pytest.approx(0.45, abs=0.02)
        assert profile[3]["C"] == pytest.approx(0.45, abs=0.02)
        assert profile[3]["B"] == pytest.approx(0.10, abs=0.02)
        # Every hour is always populated.
        assert set(profile.keys()) == set(range(24))

    def test_no_data_is_balanced(self):
        profile = reconstruct_phase_profile([], [], [], [])
        assert len(profile) == 24
        for h in range(24):
            assert profile[h]["A"] == pytest.approx(1 / 3, abs=0.001)

    def test_min_total_w_filters_noise(self):
        # All samples below the 300 W floor -> nothing counted -> balanced everywhere.
        n = 20
        ga = _at_hour(8, n, 30)
        gb = _at_hour(8, n, 30)
        gc = _at_hour(8, n, 30)
        inv = _at_hour(8, n, 0)
        profile = reconstruct_phase_profile(ga, gb, gc, inv, min_total_w=300.0)
        assert profile[8]["A"] == pytest.approx(1 / 3, abs=0.001)


class TestLoadPhaseProfile:
    def _write(self, tmp_path, payload):
        (tmp_path / "phase_model.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"config_dir": str(tmp_path)}

    def test_reads_fractions_by_hour(self, tmp_path):
        cfg = self._write(
            tmp_path,
            {"fractions_by_hour": {"8": {"A": 0.8, "B": 0.1, "C": 0.1}, "18": {"A": 0.1, "B": 0.1, "C": 0.8}}},
        )
        profile = load_phase_profile(cfg)
        assert profile is not None
        assert profile[8]["A"] == 0.8
        assert profile[18]["C"] == 0.8

    def test_missing_file_returns_none(self, tmp_path):
        assert load_phase_profile({"config_dir": str(tmp_path)}) is None

    def test_no_by_hour_key_returns_none(self, tmp_path):
        cfg = self._write(tmp_path, {"fractions": {"A": 0.5, "B": 0.3, "C": 0.2}})
        assert load_phase_profile(cfg) is None

    def test_malformed_hour_entries_skipped(self, tmp_path):
        cfg = self._write(
            tmp_path,
            {"fractions_by_hour": {"8": {"A": 0.8, "B": 0.1, "C": 0.1}, "bad": {"A": 1}, "9": {"A": 0.5}}},
        )
        profile = load_phase_profile(cfg)
        assert profile is not None
        assert 8 in profile  # valid entry kept
        assert 9 not in profile  # incomplete (missing B/C) skipped
        assert "bad" not in {str(k) for k in profile}


class TestPhaseFractionsForSlot:
    def test_returns_hour_fractions(self):
        profile = {8: {"A": 0.8, "B": 0.1, "C": 0.1}}
        assert phase_fractions_for_slot(profile, 8) == {"A": 0.8, "B": 0.1, "C": 0.1}

    def test_falls_back_when_hour_absent(self):
        profile = {8: {"A": 0.8, "B": 0.1, "C": 0.1}}
        fallback = {"A": 0.34, "B": 0.33, "C": 0.33}
        assert phase_fractions_for_slot(profile, 12, fallback) == fallback

    def test_falls_back_when_profile_none(self):
        fallback = {"A": 0.34, "B": 0.33, "C": 0.33}
        assert phase_fractions_for_slot(None, 8, fallback) == fallback
        assert phase_fractions_for_slot(None, 8) is None


from planner.simulation import RealismSlot, simulate_realistic  # noqa: E402


def _rslot(load, pv=0.0, discharge=0.0, imp=1.0, exp=0.4):
    return RealismSlot(
        pv_kwh=pv,
        load_kwh=load,
        discharge_kwh=discharge,
        grid_import_kwh=max(0.0, load - pv - discharge),
        grid_export_kwh=max(0.0, pv + discharge - load),
        import_price_sek_kwh=imp,
        export_price_sek_kwh=exp,
    )


class TestPerSlotRealism:
    def test_per_slot_fractions_override_static(self):
        # Two identical balanced-net slots (pv=load=1). Per-slot makes slot 1 heavily
        # C-loaded (imbalance cost) while slot 0 is balanced (no cost).
        slots = [_rslot(load=1.0, pv=1.0), _rslot(load=1.0, pv=1.0)]
        per = [{"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, {"A": 0.0, "B": 0.0, "C": 1.0}]
        res = simulate_realistic(slots, phase_fractions=None, phase_fractions_per_slot=per)
        assert res.phase_flagged_slots == [1]
        assert res.realism_gap_sek > 0

    def test_no_per_slot_uses_static(self):
        # Per-slot absent -> static balanced fractions -> no imbalance cost (unchanged).
        slots = [_rslot(load=1.0, pv=1.0)]
        res = simulate_realistic(slots, phase_fractions={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        assert res.phase_flagged_slots == []
        assert res.realism_gap_sek == 0.0
