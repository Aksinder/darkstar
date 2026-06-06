"""Tests for the come-home prediction (arrival profile + 3-zone probability)."""

from datetime import datetime, timedelta

from backend.core.ev_arrival import (
    ArrivalProfile,
    build_arrival_profile,
    come_home_probability,
    reserve_kwh,
)

BASE = datetime(2026, 6, 1, 0, 0)  # 00:00 of some day


class TestBuildArrivalProfile:
    def test_fraction_reflects_presence_over_time(self):
        # home 00-08, away 08-20, home 20-08(next).
        events = [
            (BASE, True),
            (BASE + timedelta(hours=8), False),
            (BASE + timedelta(hours=20), True),
            (BASE + timedelta(hours=32), False),
        ]
        prof = build_arrival_profile(events, step_minutes=60)
        assert prof.probability(BASE + timedelta(hours=2)) == 1.0  # home
        assert prof.probability(BASE + timedelta(hours=10)) == 0.0  # away
        assert prof.probability(BASE + timedelta(hours=22)) == 1.0  # home
        assert prof.samples > 0

    def test_empty_events(self):
        prof = build_arrival_profile([], default_p=0.25)
        assert prof.samples == 0
        assert prof.probability(BASE) == 0.25  # falls back to default

    def test_roundtrip(self):
        prof = ArrivalProfile(fraction={"0:2": 0.7}, samples=10, default_p=0.1)
        back = ArrivalProfile.from_dict(prof.to_dict())
        assert back.fraction == prof.fraction
        assert back.samples == 10
        assert back.default_p == 0.1


class TestComeHomeProbability:
    def _prof(self, now: datetime, p: float) -> ArrivalProfile:
        return ArrivalProfile(fraction={ArrivalProfile.key(now.weekday(), now.hour): p})

    def test_near_zone_is_certain(self):
        now = BASE.replace(hour=2)
        p, zone, _ = come_home_probability(
            now, 3.0, self._prof(now, 0.7), near_radius_km=5, extended_radius_km=30
        )
        assert p == 1.0
        assert zone == "near"

    def test_extended_zone_uses_profile(self):
        now = BASE.replace(hour=2)
        p, zone, _ = come_home_probability(
            now, 12.0, self._prof(now, 0.7), near_radius_km=5, extended_radius_km=30
        )
        assert p == 0.7
        assert zone == "extended"

    def test_beyond_extended_is_zero(self):
        now = BASE.replace(hour=2)
        p, zone, _ = come_home_probability(
            now, 50.0, self._prof(now, 0.7), near_radius_km=5, extended_radius_km=30
        )
        assert p == 0.0
        assert zone == "far"

    def test_no_distance_is_zero(self):
        now = BASE.replace(hour=2)
        p, _, reason = come_home_probability(now, None, None, near_radius_km=5, extended_radius_km=30)
        assert p == 0.0
        assert reason == "no_distance"

    def test_override_force_off(self):
        now = BASE.replace(hour=2)
        p, zone, _ = come_home_probability(
            now, 3.0, self._prof(now, 0.7), override="force_off", near_radius_km=5, extended_radius_km=30
        )
        assert p == 0.0
        assert zone == "off"

    def test_override_force_reserve(self):
        now = BASE.replace(hour=2)
        p, zone, _ = come_home_probability(
            now, 99.0, None, override="force_reserve", near_radius_km=5, extended_radius_km=30
        )
        assert p == 1.0
        assert zone == "force"


class TestReserveKwh:
    def test_scaled_and_capped(self):
        assert reserve_kwh(1.0, 5.0, 4.0) == 4.0  # capped at max
        assert reserve_kwh(0.5, 5.0, 4.0) == 2.5
        assert reserve_kwh(0.0, 5.0, 4.0) == 0.0
