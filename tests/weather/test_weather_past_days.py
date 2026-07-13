"""Tests for window-derived past_days in get_weather_series and the NaN
radiation guard in calculate_physics_pv.

Phase 1b context: training requests weeks of history. With the old hardcoded
past_days=1, every observation older than yesterday 00:00 got NaN weather after
the merge, and NaN radiation slipped past the <=0 guard to produce a 0.0 kWh
physics baseline — training residual = full actual PV. past_days must scale
with the requested window (clamped to the API's 92-day max) and NaN radiation
must yield physics None ("no data"), never 0.0 ("calm night").
"""

import math
import unittest
from datetime import datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytz

import ml.weather as weather_mod
from ml.weather import calculate_physics_pv, get_weather_series

UTC = pytz.UTC
CONFIG = {
    "timezone": "UTC",
    "system": {"location": {"latitude": 59.3, "longitude": 18.1}},
}


def _mock_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "hourly": {
            "time": ["2026-02-15T12:00", "2026-02-15T13:00"],
            "temperature_2m": [10.0, 20.0],
            "cloud_cover": [0.0, 50.0],
            "shortwave_radiation": [0.0, 400.0],
        }
    }
    return resp


class TestPastDaysDerivation(unittest.TestCase):
    def setUp(self):
        weather_mod._weather_cache.clear()

    def _requested_past_days(self, mock_get) -> int:
        _, kwargs = mock_get.call_args
        return int(kwargs["params"]["past_days"])

    @patch("ml.weather.requests.get")
    def test_past_days_covers_requested_window(self, mock_get):
        mock_get.return_value = _mock_response()
        now = datetime.now(UTC)
        start = now - timedelta(days=10)

        get_weather_series(start, now, config=CONFIG)

        assert self._requested_past_days(mock_get) == 10

    @patch("ml.weather.requests.get")
    def test_past_days_clamped_to_api_max(self, mock_get):
        mock_get.return_value = _mock_response()
        now = datetime.now(UTC)
        start = now - timedelta(days=400)

        get_weather_series(start, now, config=CONFIG)

        assert self._requested_past_days(mock_get) == 92

    @patch("ml.weather.requests.get")
    def test_past_days_floor_of_one_for_current_or_future_start(self, mock_get):
        mock_get.return_value = _mock_response()
        now = datetime.now(UTC)

        get_weather_series(now, now + timedelta(days=1), config=CONFIG)
        assert self._requested_past_days(mock_get) == 1

        weather_mod._weather_cache.clear()
        get_weather_series(
            now + timedelta(days=1), now + timedelta(days=2), config=CONFIG
        )
        assert self._requested_past_days(mock_get) == 1


class TestNanRadiationGuard(unittest.TestCase):
    ARRAYS: ClassVar[list[dict[str, Any]]] = [
        {"name": "S", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}
    ]
    # Midday in summer — sun well above horizon, so a 0.0 return could only
    # come from the (former) NaN-through-max() path, not the horizon check.
    SLOT: ClassVar[datetime] = pytz.timezone("Europe/Stockholm").localize(
        datetime(2026, 6, 21, 12, 0)
    )

    def test_nan_radiation_returns_none_not_zero(self):
        physics_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=float("nan"),
            solar_arrays=self.ARRAYS,
            slot_start=self.SLOT,
            latitude=59.3,
            longitude=18.1,
        )
        assert physics_kwh is None, (
            f"NaN radiation must mean 'no data' (None), got {physics_kwh!r} — "
            "0.0 would train residual = full actual PV against zero physics"
        )
        assert per_array == []

    def test_nan_radiation_with_valid_dni_dhi_still_none(self):
        # GHI is required (guard + ground-reflected term); NaN GHI means the
        # weather row is missing, so DNI/DHI would be NaN too in practice.
        physics_kwh, _ = calculate_physics_pv(
            radiation_w_m2=float("nan"),
            solar_arrays=self.ARRAYS,
            slot_start=self.SLOT,
            latitude=59.3,
            longitude=18.1,
            dni_w_m2=500.0,
            dhi_w_m2=100.0,
        )
        assert physics_kwh is None

    def test_valid_radiation_unaffected(self):
        physics_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=800.0,
            solar_arrays=self.ARRAYS,
            slot_start=self.SLOT,
            latitude=59.3,
            longitude=18.1,
        )
        assert physics_kwh is not None and physics_kwh > 0
        assert not math.isnan(physics_kwh)
        assert len(per_array) == 1
