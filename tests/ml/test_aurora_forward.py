import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytz

from backend.learning import LearningEngine
from ml.forward import generate_forward_slots


class TestAuroraForward(unittest.IsolatedAsyncioTestCase):
    @patch("ml.forward.datetime")
    @patch("ml.forward.get_learning_engine")
    @patch("ml.forward._load_models")
    @patch("ml.forward.async_get_weather_series")
    @patch("ml.forward.get_vacation_mode_series")
    @patch("ml.forward.get_alarm_armed_series")
    @patch("backend.astro.SunCalculator")
    @patch("ml.forward.determine_graduation_level")
    async def test_forward_pass_multi_array(
        self,
        mock_grad_level,
        mock_sun,
        mock_alarm,
        mock_vacation,
        mock_weather,
        mock_models,
        mock_get_engine,
        mock_datetime,
    ):
        print("\n--- Testing Aurora Forward Pass with Multi-Array ---")

        # 1. Mock datetime to return a fixed midday summer time
        # This ensures sun is above horizon during physics calculation
        fixed_now = datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.side_effect = datetime

        # 2. Mock graduation level
        mock_grad_level.return_value = (2, "graduate", 30.0)

        # 3. Mock LearningEngine
        mock_engine = MagicMock(spec=LearningEngine)
        mock_engine.store_forecasts = AsyncMock()
        mock_engine.timezone = pytz.UTC
        mock_engine.db_path = "fake_db.db"

        # 4. Mock SunCalculator
        mock_sun_instance = MagicMock()
        mock_sun_instance.is_sun_up.return_value = True
        mock_sun.return_value = mock_sun_instance
        mock_engine.config = {
            "timezone": "UTC",
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_arrays": [{"kwp": 10.0}, {"kwp": 5.0}],
            },
        }
        mock_engine.store_forecasts = AsyncMock()
        mock_get_engine.return_value = mock_engine

        # 5. Mock Models (empty to trigger fallback)
        mock_models.return_value = {}

        # 6. Mock Weather data - align with fixed datetime
        slot_start = fixed_now.replace(minute=0, second=0, microsecond=0)
        mock_weather.return_value = pd.DataFrame(
            {
                "temp_c": [20.0] * 8,
                "cloud_cover_pct": [10.0] * 8,
                "shortwave_radiation_w_m2": [800.0] * 8,
            },
            index=pd.date_range(slot_start, periods=8, freq="15min", tz="UTC"),
        )

        # 4. Mock Context flags
        mock_vacation.return_value = pd.Series(0.0, index=mock_weather.return_value.index)
        mock_alarm.return_value = pd.Series(0.0, index=mock_weather.return_value.index)

        # 5. Run forward pass (short horizon)
        await generate_forward_slots(horizon_hours=2, forecast_version="test")

        # 6. Verify physics-based PV forecast in fallback mode
        # With physics model: POA irradiance calculation with panel orientation
        # Radiation = 800 W/m², total capacity = 15 kWp
        # Physics model accounts for solar position and panel orientation
        # At solar noon with south-facing panels: POA ≈ GHI, so
        # PV ≈ (800/1000) * 15 * 0.85 * 0.25 ≈ 2.5 kWh per 15-min slot

        args, _ = mock_engine.store_forecasts.call_args
        forecasts = args[0]

        self.assertGreater(len(forecasts), 0)
        # Check p50 PV forecast for first slot
        first_pv = forecasts[0]["pv_forecast_kwh"]
        # Physics model should give reasonable output (not zero, not huge)
        self.assertGreater(first_pv, 0.0, "PV forecast should be positive during daytime")
        self.assertLess(first_pv, 4.0, "PV forecast should be reasonable for 15kWp system")

        print("✅ Aurora forward pass used physics-based calculation with 15.0 kWp total")
        print(f"✅ Calculated PV forecast: {first_pv:.4f} kWh")
        print("✅ Aurora forward pass multi-array integration verified!")


class TestAuroraForwardPvResidualGate(unittest.IsolatedAsyncioTestCase):
    """forecasting.pv_residual_enabled=false must force physics-only and bypass PV models."""

    def _make_engine(self, pv_residual_enabled: bool) -> MagicMock:
        engine = MagicMock(spec=LearningEngine)
        engine.store_forecasts = AsyncMock()
        engine.timezone = pytz.UTC
        engine.db_path = "fake_db.db"
        engine.config = {
            "timezone": "UTC",
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_arrays": [{"kwp": 10.0, "tilt": 30.0, "azimuth": 180.0}],
            },
            "forecasting": {"pv_residual_enabled": pv_residual_enabled},
        }
        return engine

    def _weather_frame(self, slot_start):
        return pd.DataFrame(
            {
                "temp_c": [20.0] * 8,
                "cloud_cover_pct": [10.0] * 8,
                "shortwave_radiation_w_m2": [800.0] * 8,
                "dni_w_m2": [700.0] * 8,
                "dhi_w_m2": [150.0] * 8,
            },
            index=pd.date_range(slot_start, periods=8, freq="15min", tz="UTC"),
        )

    def _mock_pv_models(self):
        """Return a dict of load+pv booster mocks; pv predict is a spy we assert on."""
        models: dict[str, MagicMock] = {}
        for q in ("p10", "p50", "p90"):
            load_booster = MagicMock()
            load_booster.predict = MagicMock(return_value=[0.5] * 8)
            models[f"load_{q}"] = load_booster
            pv_booster = MagicMock()
            pv_booster.predict = MagicMock(return_value=[0.0] * 8)
            models[f"pv_{q}"] = pv_booster
        return models

    @patch("ml.forward.datetime")
    @patch("ml.forward.get_learning_engine")
    @patch("ml.forward._load_models")
    @patch("ml.forward.async_get_weather_series")
    @patch("ml.forward.get_vacation_mode_series")
    @patch("ml.forward.get_alarm_armed_series")
    @patch("backend.astro.SunCalculator")
    @patch("ml.forward.determine_graduation_level")
    async def test_physics_only_when_disabled(
        self,
        mock_grad_level,
        mock_sun,
        mock_alarm,
        mock_vacation,
        mock_weather,
        mock_models,
        mock_get_engine,
        mock_datetime,
    ):
        fixed_now = datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.side_effect = datetime
        mock_grad_level.return_value = (2, "graduate", 30.0)

        engine = self._make_engine(pv_residual_enabled=False)
        mock_get_engine.return_value = engine

        mock_sun_instance = MagicMock()
        mock_sun_instance.is_sun_up.return_value = True
        mock_sun.return_value = mock_sun_instance

        models = self._mock_pv_models()
        mock_models.return_value = models

        slot_start = fixed_now.replace(minute=0, second=0, microsecond=0)
        mock_weather.return_value = self._weather_frame(slot_start)
        mock_vacation.return_value = pd.Series(0.0, index=mock_weather.return_value.index)
        mock_alarm.return_value = pd.Series(0.0, index=mock_weather.return_value.index)

        await generate_forward_slots(horizon_hours=2, forecast_version="test")

        # PV residual models must NOT be applied in physics-only mode.
        for q in ("p10", "p50", "p90"):
            models[f"pv_{q}"].predict.assert_not_called()

        args, _ = engine.store_forecasts.call_args
        forecasts = args[0]
        self.assertGreater(len(forecasts), 0)
        # Physics-only p50 == physics base; p90 == physics * 1.2 (bands around physics).
        first = forecasts[0]
        self.assertGreater(first["pv_forecast_kwh"], 0.0)
        self.assertAlmostEqual(first["pv_p90"], first["pv_forecast_kwh"] * 1.2, places=4)

    @patch("ml.forward.datetime")
    @patch("ml.forward.get_learning_engine")
    @patch("ml.forward._load_models")
    @patch("ml.forward.async_get_weather_series")
    @patch("ml.forward.get_vacation_mode_series")
    @patch("ml.forward.get_alarm_armed_series")
    @patch("backend.astro.SunCalculator")
    @patch("ml.forward.determine_graduation_level")
    async def test_hybrid_when_enabled(
        self,
        mock_grad_level,
        mock_sun,
        mock_alarm,
        mock_vacation,
        mock_weather,
        mock_models,
        mock_get_engine,
        mock_datetime,
    ):
        fixed_now = datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        mock_datetime.now.return_value = fixed_now
        mock_datetime.side_effect = datetime
        mock_grad_level.return_value = (2, "graduate", 30.0)

        engine = self._make_engine(pv_residual_enabled=True)
        mock_get_engine.return_value = engine

        mock_sun_instance = MagicMock()
        mock_sun_instance.is_sun_up.return_value = True
        mock_sun.return_value = mock_sun_instance

        models = self._mock_pv_models()
        mock_models.return_value = models

        slot_start = fixed_now.replace(minute=0, second=0, microsecond=0)
        mock_weather.return_value = self._weather_frame(slot_start)
        mock_vacation.return_value = pd.Series(0.0, index=mock_weather.return_value.index)
        mock_alarm.return_value = pd.Series(0.0, index=mock_weather.return_value.index)

        await generate_forward_slots(horizon_hours=2, forecast_version="test")

        # Default (enabled) applies the PV residual models.
        models["pv_p50"].predict.assert_called()


if __name__ == "__main__":
    unittest.main()
