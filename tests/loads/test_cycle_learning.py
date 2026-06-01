"""Tests for deferrable-load cycle learning (detection + rolling stats)."""

from datetime import datetime, timedelta

import pytest

from backend.learning.cycle_learning import (
    CycleStats,
    PowerSample,
    RunSample,
    StatusSample,
    detect_cycles_from_power,
    detect_cycles_from_runstate,
    detect_cycles_from_status,
    power_samples_from_ha_history,
    run_samples_from_ha_history,
    status_samples_from_ha_history,
)

T0 = datetime(2026, 5, 30, 8, 0, 0)


def _p(samples_min_w):
    """Build PowerSamples from (minute_offset, watts) tuples."""
    return [PowerSample(ts=T0 + timedelta(minutes=m), power_w=w) for m, w in samples_min_w]


def _r(samples_min_bool):
    return [RunSample(ts=T0 + timedelta(minutes=m), running=b) for m, b in samples_min_bool]


def _s(samples_min_status_power):
    """Build StatusSamples from (minute, status, power|None) tuples."""
    return [
        StatusSample(ts=T0 + timedelta(minutes=m), status=st, power_w=pw)
        for m, st, pw in samples_min_status_power
    ]


class TestHaHistoryParsing:
    def test_power_skips_non_numeric_and_sorts(self):
        rows = [
            {"state": "100", "last_changed": "2026-05-30T08:10:00+00:00"},
            {"state": "unavailable", "last_changed": "2026-05-30T08:05:00+00:00"},
            {"state": "50", "last_changed": "2026-05-30T08:00:00+00:00"},
            {"state": "unknown", "last_changed": "2026-05-30T08:07:00+00:00"},
        ]
        out = power_samples_from_ha_history(rows)
        assert [s.power_w for s in out] == [50.0, 100.0]  # sorted, unknowns dropped

    def test_runstate_maps_on_off(self):
        rows = [
            {"state": "on", "last_changed": "2026-05-30T08:00:00+00:00"},
            {"state": "Idle", "last_changed": "2026-05-30T08:30:00+00:00"},
            {"state": "weird", "last_changed": "2026-05-30T08:40:00+00:00"},
        ]
        out = run_samples_from_ha_history(rows)
        assert [s.running for s in out] == [True, False]


class TestDetectFromPower:
    def test_single_clean_cycle(self):
        # 60 min at ~2 kW
        samples = _p([(0, 0), (1, 2000), (30, 2000), (60, 2000), (61, 0)])
        cycles = detect_cycles_from_power(samples)
        assert len(cycles) == 1
        c = cycles[0]
        assert c.complete
        # Cycle spans the first on-sample (min 1) to the last above-off (min 60).
        assert c.duration_min == 59.0
        assert c.peak_w == 2000.0
        assert abs(c.energy_kwh - 2.0) < 0.05  # ~2 kWh over the hour
        # Constant 2 kW → every 15-min bucket reads ~2 kW.
        assert all(abs(p - 2.0) < 0.1 for p in c.profile_kw)

    def test_short_blip_filtered(self):
        # A 1-minute spike below min_on_minutes=3 should be ignored.
        samples = _p([(0, 0), (10, 1500), (11, 0), (60, 0)])
        cycles = detect_cycles_from_power(samples, min_on_minutes=3.0)
        assert cycles == []

    def test_multi_phase_merges_into_one_cycle(self):
        # Heat (10 min) → pause (15 min, below off) → heat (10 min). With a
        # 20-min merge gap these are one dishwasher-like cycle.
        samples = _p(
            [
                (0, 0),
                (1, 2000),
                (10, 2000),
                (11, 5),  # phase 1 then idle
                (25, 5),  # 14-min pause < merge gap
                (26, 1800),
                (35, 1800),
                (36, 0),  # phase 2
                (60, 0),
            ]
        )
        cycles = detect_cycles_from_power(samples, merge_gap_minutes=20.0)
        assert len(cycles) == 1
        assert cycles[0].start == T0 + timedelta(minutes=1)
        assert cycles[0].end == T0 + timedelta(minutes=35)

    def test_long_pause_splits_cycles(self):
        # A 40-min pause exceeds the merge gap → two separate cycles.
        samples = _p(
            [
                (0, 2000),
                (10, 2000),
                (11, 0),
                (51, 0),
                (52, 2000),
                (62, 2000),
                (63, 0),
            ]
        )
        cycles = detect_cycles_from_power(samples, merge_gap_minutes=20.0)
        assert len(cycles) == 2

    def test_energy_not_integrated_across_data_gap(self):
        # Sensor goes quiet (no samples) for 5 h while nominally "on"; the
        # 5-hour interval must not be integrated as energy.
        samples = _p([(0, 2000), (5, 2000), (305, 2000), (306, 0)])
        cycles = detect_cycles_from_power(samples, max_sample_gap_minutes=30.0)
        assert len(cycles) == 1
        # Only the two short ~2 kW spans count, not the 5-hour gap.
        assert cycles[0].energy_kwh < 0.4

    def test_open_tail_marked_incomplete(self):
        samples = _p([(0, 0), (1, 2000), (30, 2000)])  # still running at series end
        cycles = detect_cycles_from_power(samples)
        assert len(cycles) == 1
        assert cycles[0].complete is False

    def test_profile_buckets(self):
        # 30-min cycle, 2 kW then 0.2 kW; 15-min buckets → [~2, ~0.2] kW
        samples = _p([(0, 2000), (14, 2000), (15, 200), (30, 200), (31, 0)])
        cycles = detect_cycles_from_power(samples, profile_bucket_min=15.0)
        prof = cycles[0].profile_kw
        assert len(prof) == 2
        assert prof[0] > 1.5
        assert prof[1] < 0.5


class TestDetectFromRunstate:
    def test_flapping_collapses_to_one_cycle(self):
        # A run-state sensor flapping on/off every minute for ~30 min.
        rows = []
        for m in range(0, 30):
            rows.append((m, True))
            rows.append((m, False))  # immediate flap
        rows.append((31, False))
        cycles = detect_cycles_from_runstate(_r(rows), merge_gap_minutes=20.0)
        assert len(cycles) == 1
        assert cycles[0].duration_min >= 28

    def test_assumed_power_estimates_energy(self):
        rows = _r([(0, True), (90, False)])
        cycles = detect_cycles_from_runstate(rows, assumed_power_kw=1.0)
        assert len(cycles) == 1
        assert abs(cycles[0].energy_kwh - 1.5) < 0.01  # 90 min * 1 kW = 1.5 kWh


class TestCycleStats:
    def test_seed_until_min_cycles(self):
        cycles = detect_cycles_from_power(_p([(0, 2000), (60, 2000), (61, 0)]))  # 1 cycle
        stats = CycleStats.from_cycles(
            cycles, seed_duration_min=110, seed_energy_kwh=1.2, min_cycles=3
        )
        assert stats.learned is False
        assert stats.duration_min == 110
        assert stats.energy_kwh == 1.2

    def test_learns_median_after_enough_cycles(self):
        from backend.learning.cycle_learning import DetectedCycle

        cycles = [
            DetectedCycle(T0, T0 + timedelta(minutes=d), float(d), e)
            for d, e in [(90, 1.0), (100, 1.2), (110, 1.4), (200, 2.0)]
        ]
        stats = CycleStats.from_cycles(
            cycles, seed_duration_min=120, seed_energy_kwh=1.5, min_cycles=3
        )
        assert stats.learned is True
        assert stats.n_cycles == 4
        assert stats.duration_min == 105.0  # median of 90,100,110,200
        assert stats.energy_kwh == 1.3  # median of 1.0,1.2,1.4,2.0
        assert stats.duration_min_p90 >= 110

    def test_incomplete_cycles_excluded(self):
        from backend.learning.cycle_learning import DetectedCycle

        cycles = [
            DetectedCycle(T0, T0 + timedelta(minutes=90), 90.0, 1.0, complete=True),
            DetectedCycle(T0, T0 + timedelta(minutes=999), 999.0, 9.0, complete=False),
        ]
        stats = CycleStats.from_cycles(
            cycles, seed_duration_min=120, seed_energy_kwh=1.5, min_cycles=1
        )
        assert stats.n_cycles == 1
        assert stats.duration_min == 90.0


class TestStatusParsing:
    def test_parses_state_and_power_attribute(self):
        rows = [
            {
                "state": "Filling",
                "last_changed": "2026-05-30T08:00:00+00:00",
                "attributes": {"power": 1850, "icon": "mdi:dishwasher"},
            },
            {
                "state": "Idle",
                "last_changed": "2026-05-30T09:30:00+00:00",
                "attributes": {"power": 0},
            },
        ]
        out = status_samples_from_ha_history(rows)
        assert [s.status for s in out] == ["Filling", "Idle"]
        assert out[0].power_w == 1850.0


class TestDetectFromStatus:
    def test_idle_to_phases_to_idle_is_one_cycle(self):
        samples = _s(
            [
                (0, "Idle", 0),
                (1, "Filling", 200),
                (20, "Washing", 2000),
                (60, "Rinsing", 300),
                (90, "Drying", 50),
                (110, "Idle", 0),
            ]
        )
        cycles = detect_cycles_from_status(samples)
        assert len(cycles) == 1
        c = cycles[0]
        assert c.complete
        assert c.start == T0 + timedelta(minutes=1)
        assert c.end == T0 + timedelta(minutes=110)
        assert c.peak_w == 2000.0

    def test_phase_minutes_breakdown(self):
        samples = _s(
            [
                (0, "Idle", 0),
                (10, "Washing", 2000),  # 30 min washing
                (40, "Rinsing", 300),  # 20 min rinsing
                (60, "Idle", 0),
            ]
        )
        c = detect_cycles_from_status(samples)[0]
        assert c.phase_minutes["Washing"] == 30.0
        assert c.phase_minutes["Rinsing"] == 20.0
        assert "Idle" not in c.phase_minutes  # off-state ends the cycle

    def test_energy_integrated_from_power_attribute(self):
        # 60 min steady at 2 kW (sampled densely) -> ~2 kWh.
        rows = [(0, "Idle", 0)]
        rows += [(m, "Washing", 2000) for m in range(1, 61, 5)]
        rows.append((61, "Idle", 0))
        c = detect_cycles_from_status(_s(rows))[0]
        # ~2 kWh (sampling at the edges trims a little); ballpark is what matters.
        assert 1.6 < c.energy_kwh < 2.1

    def test_pausing_flaps_merge_into_one_cycle(self):
        # Real dishwashers flap Filling<->Pausing; with the merge gap these are
        # one cycle, and Pausing is still counted as part of the run.
        rows = [(0, "Idle", 0)]
        for m in range(1, 40, 2):
            rows.append((m, "Filling", 30))
            rows.append((m + 1, "Pausing", 4))
        rows.append((41, "Idle", 0))
        cycles = detect_cycles_from_status(_s(rows), merge_gap_minutes=20)
        assert len(cycles) == 1
        assert cycles[0].duration_min >= 38

    def test_custom_off_statuses(self):
        samples = _s([(0, "Ready", 0), (5, "Run", 1000), (35, "Ready", 0)])
        cycles = detect_cycles_from_status(samples, off_statuses={"ready"})
        assert len(cycles) == 1
        assert cycles[0].duration_min == 30.0

    def test_open_tail_incomplete(self):
        samples = _s([(0, "Idle", 0), (1, "Washing", 2000), (40, "Washing", 2000)])
        c = detect_cycles_from_status(samples)[0]
        assert c.complete is False


class TestPowerScaleAndRobustStatus:
    def test_power_scale_doubles_energy_from_power(self):
        # Single phase metered at 1 kW; a 2-phase element draws ~2 kW.
        rows = [(0, 0)] + [(m, 1000) for m in range(1, 61, 5)] + [(61, 0)]
        base = detect_cycles_from_power(_p(rows))[0]
        scaled = detect_cycles_from_power(_p(rows), power_scale=2.0)[0]
        assert scaled.energy_kwh == pytest.approx(base.energy_kwh * 2.0, abs=0.01)
        assert scaled.peak_w == pytest.approx(base.peak_w * 2.0, abs=0.1)
        assert scaled.profile_kw[0] == pytest.approx(base.profile_kw[0] * 2.0, abs=0.01)

    def test_status_power_scale(self):
        rows = (
            [(0, "Idle", 0)] + [(m, "Washing", 1000) for m in range(1, 61, 5)] + [(61, "Idle", 0)]
        )
        base = detect_cycles_from_status(_s(rows))[0]
        scaled = detect_cycles_from_status(_s(rows), power_scale=2.0)[0]
        assert scaled.energy_kwh == pytest.approx(base.energy_kwh * 2.0, abs=0.01)

    def test_running_power_overrides_mislabelled_idle(self):
        # Phase sensor wrongly reports "Idle" mid-cycle while power stays high.
        samples = _s(
            [
                (0, "Idle", 0),
                (1, "Washing", 2000),
                (20, "Idle", 2000),  # mislabelled: still drawing 2 kW
                (40, "Washing", 2000),
                (60, "Idle", 0),
            ]
        )
        # Without corroboration the cycle would split at minute 20.
        with_power = detect_cycles_from_status(samples, running_power_w=500)
        assert len(with_power) == 1
        assert with_power[0].duration_min >= 58
