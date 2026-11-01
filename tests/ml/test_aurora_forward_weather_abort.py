"""A failed Open-Meteo weather fetch must not store a near-zero PV forecast.

Without radiation, physics PV is 0 for every slot; storing that would overwrite
the last good forecast the planner reads. Idea ported from upstream 47e557c5.
"""

import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytz

from backend.learning import LearningEngine
from ml.forward import generate_forward_slots

FIXED_NOW = datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)


def _engine() -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.store_forecasts = AsyncMock()
    engine.timezone = pytz.UTC
    engine.db_path = "fake_db.db"
    engine.config = {
        "timezone": "UTC",
        "system": {
            "location": {"latitude": 59.3, "longitude": 18.1},
            "solar_arrays": [{"kwp": 10.0}],
        },
    }
    return engine


class TestForwardWeatherAbort(unittest.IsolatedAsyncioTestCase):
    async def _run(self, weather_df: pd.DataFrame) -> MagicMock:
        engine = _engine()
        index = pd.date_range(FIXED_NOW, periods=8, freq="15min", tz="UTC")
        with (
            patch("ml.forward.datetime") as mock_dt,
            patch("ml.forward.get_learning_engine", return_value=engine),
            patch("ml.forward._load_models", return_value={}),
            patch("ml.forward.async_get_weather_series", AsyncMock(return_value=weather_df)),
            patch("ml.forward.get_vacation_mode_series", return_value=pd.Series(0.0, index=index)),
            patch("ml.forward.get_alarm_armed_series", return_value=pd.Series(0.0, index=index)),
            patch("ml.forward.determine_graduation_level", return_value=(2, "graduate", 30.0)),
            patch("backend.astro.SunCalculator"),
        ):
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.side_effect = datetime
            await generate_forward_slots(horizon_hours=2, forecast_version="test")
        return engine

    async def test_empty_weather_does_not_store(self):
        engine = await self._run(pd.DataFrame(dtype="float64"))
        engine.store_forecasts.assert_not_called()

    async def test_all_nan_radiation_does_not_store(self):
        index = pd.date_range(FIXED_NOW, periods=8, freq="15min", tz="UTC")
        weather = pd.DataFrame(
            {
                "temp_c": [20.0] * 8,
                "cloud_cover_pct": [10.0] * 8,
                "shortwave_radiation_w_m2": [float("nan")] * 8,
            },
            index=index,
        )
        engine = await self._run(weather)
        engine.store_forecasts.assert_not_called()

    async def test_valid_weather_still_stores(self):
        index = pd.date_range(FIXED_NOW, periods=8, freq="15min", tz="UTC")
        weather = pd.DataFrame(
            {
                "temp_c": [20.0] * 8,
                "cloud_cover_pct": [10.0] * 8,
                "shortwave_radiation_w_m2": [800.0] * 8,
            },
            index=index,
        )
        engine = await self._run(weather)
        engine.store_forecasts.assert_called_once()
