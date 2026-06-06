"""Tests for the robust EV home-zone gate (zone + distance + grace)."""

from datetime import UTC, datetime, timedelta

from backend.core.ev_presence import ev_is_home, haversine_km

HOME_LAT = 57.60972
HOME_LON = 18.41464
NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC)


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(HOME_LAT, HOME_LON, HOME_LAT, HOME_LON) < 1e-6

    def test_malmo_is_far(self):
        # The live away-case: car near Malmö/Lund ~ hundreds of km from Gotland.
        assert haversine_km(55.5979, 13.0963, HOME_LAT, HOME_LON) > 300


class TestEvIsHome:
    def test_zone_match(self):
        ok, reason = ev_is_home("home", home_states=["home"])
        assert ok is True
        assert reason == "zone"

    def test_away_with_no_tolerances(self):
        ok, reason = ev_is_home("not_home", home_states=["home"])
        assert ok is False
        assert reason == "away"

    def test_radius_includes_drifting_car_at_home(self):
        # not_home (GPS pushed just outside a tight zone) but physically at the house.
        ok, reason = ev_is_home(
            "not_home",
            home_states=["home"],
            home_lat=HOME_LAT,
            home_lon=HOME_LON,
            car_lat=HOME_LAT + 0.001,  # ~110 m north
            car_lon=HOME_LON,
            radius_km=0.3,
        )
        assert ok is True
        assert reason.startswith("radius=")

    def test_radius_excludes_far_car(self):
        ok, _ = ev_is_home(
            "not_home",
            home_states=["home"],
            home_lat=HOME_LAT,
            home_lon=HOME_LON,
            car_lat=55.5979,
            car_lon=13.0963,
            radius_km=0.3,
        )
        assert ok is False

    def test_grace_keeps_recently_departed_car(self):
        ok, reason = ev_is_home(
            "not_home",
            home_states=["home"],
            last_changed=NOW - timedelta(minutes=3),
            grace_minutes=10,
            now=NOW,
        )
        assert ok is True
        assert reason.startswith("grace=")

    def test_grace_expires(self):
        ok, _ = ev_is_home(
            "not_home",
            home_states=["home"],
            last_changed=NOW - timedelta(minutes=20),
            grace_minutes=10,
            now=NOW,
        )
        assert ok is False

    def test_tolerances_disabled_by_default(self):
        # radius_km=0 and grace_minutes=0 -> plain zone check, away when not in zone.
        ok, _ = ev_is_home(
            "not_home",
            home_states=["home"],
            home_lat=HOME_LAT,
            home_lon=HOME_LON,
            car_lat=HOME_LAT,
            car_lon=HOME_LON,
            radius_km=0.0,
        )
        assert ok is False
