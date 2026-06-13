import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# We import from forecasts module. We will patch OpenMeteoSolarForecast where it is USED.
from backend.core.forecasts import _get_forecast_data_async, get_forecast_data


class TestForecastAggregation(unittest.IsolatedAsyncioTestCase):
    async def test_multi_array_aggregation(self):
        print("\n--- Testing Multi-Array Forecast Aggregation ---")

        # 1. Setup mock config with 2 arrays
        config = {
            "timezone": "UTC",
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_arrays": [
                    {"name": "Array 1", "azimuth": 180, "tilt": 35, "kwp": 10.0},
                    {"name": "Array 2", "azimuth": 90, "tilt": 35, "kwp": 5.0},
                ],
            },
        }

        # 2. Setup mock price slots (4 slots = 1 hour)
        price_slots = [
            {"start_time": datetime(2024, 6, 21, 12, 0, tzinfo=UTC)},
            {"start_time": datetime(2024, 6, 21, 12, 15, tzinfo=UTC)},
            {"start_time": datetime(2024, 6, 21, 12, 30, tzinfo=UTC)},
            {"start_time": datetime(2024, 6, 21, 12, 45, tzinfo=UTC)},
        ]

        # 3. Setup mock OpenMeteoSolarForecast estimate response
        mock_estimate = MagicMock()
        mock_estimate.watts = {
            datetime(2024, 6, 21, 12, 0, tzinfo=UTC): 1000.0,
            datetime(2024, 6, 21, 12, 15, tzinfo=UTC): 1000.0,
            datetime(2024, 6, 21, 12, 30, tzinfo=UTC): 1000.0,
            datetime(2024, 6, 21, 12, 45, tzinfo=UTC): 1000.0,
        }

        # 4. Patch OpenMeteoSolarForecast inside inputs
        with patch("backend.core.forecasts.OpenMeteoSolarForecast") as MockForecastClass:
            # Configure the mock instance
            mock_instance = AsyncMock()
            mock_instance.estimate.return_value = mock_estimate
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            MockForecastClass.return_value = mock_instance

            # Mock get_load_profile_from_ha to avoid HA calls
            with patch("backend.core.ha_client.get_load_profile_from_ha", return_value=[0.5] * 96):
                # 5. Call the function
                result = await _get_forecast_data_async(price_slots, config)

            # 6. Verify constructor calls
            MockForecastClass.assert_called_with(
                latitude=[59.3, 59.3],
                longitude=[18.1, 18.1],
                declination=[35.0, 35.0],  # From Array 1 & 2
                azimuth=[0.0, -90.0],  # From Array 1 & 2 (South = 0 for Open-Meteo)
                dc_kwp=[10.0, 5.0],  # From Array 1 & 2
            )
            print("✅ Correct lists passed to OpenMeteoSolarForecast")

            # 7. Verify result aggregation
            # 1000 Watts * 0.25 hours = 250 Wh = 0.25 kWh per slot
            for slot in result["slots"]:
                self.assertEqual(slot["pv_forecast_kwh"], 0.25)

            print("✅ Forecast aggregation result correct (0.25 kWh per slot)")
            print("✅ Multi-array forecast integration verified!")

    async def test_aurora_returns_extended_slots_from_existing_records(self):
        price_slots = [
            {"start_time": datetime(2024, 1, 1, 0, 0, tzinfo=UTC)},
            {"start_time": datetime(2024, 1, 1, 0, 15, tzinfo=UTC)},
        ]
        db_slots = [
            {"pv_forecast_kwh": 0.1, "base_load_forecast_kwh": 0.2},
            {"pv_forecast_kwh": 0.2, "base_load_forecast_kwh": 0.3},
        ]
        extended_records = []
        for idx in range(100):
            slot_start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=15 * idx)
            extended_records.append(
                {
                    "slot_start": slot_start,
                    "final": {"pv_kwh": 0.05 + idx, "load_kwh": 0.1 + idx},
                    "base": {"pv_kwh": 0.04 + idx},
                    "probabilistic": {
                        "pv_p10": 0.01 + idx,
                        "pv_p90": 0.09 + idx,
                        "load_p10": 0.02 + idx,
                        "load_p90": 0.12 + idx,
                    },
                }
            )

        config = {
            "timezone": "UTC",
            "forecasting": {
                "active_forecast_version": "aurora",
                "aurora_load_enabled": True,
                "aurora_pv_enabled": True,
            },
        }

        with (
            patch("backend.core.forecasts.build_db_forecast_for_slots", new=AsyncMock(return_value=db_slots)),
            patch(
                "backend.core.forecasts.get_forecast_slots",
                new=AsyncMock(return_value=extended_records),
            ) as mock_extended_fetch,
            patch("backend.core.ha_client.get_load_profile_from_ha", new=AsyncMock(return_value=[0.4] * 96)),
        ):
            result = await get_forecast_data(price_slots, config)

        assert mock_extended_fetch.await_count == 1
        assert len(result["slots"]) == len(price_slots)
        assert len(result["extended_slots"]) == len(extended_records)
        assert result["extended_slots"][-1]["start_time"] >= price_slots[-1]["start_time"] + timedelta(
            hours=24
        )
        assert result["extended_slots"][0]["pv_forecast_kwh"] == 0.05
        assert result["extended_slots"][0]["load_forecast_kwh"] == 0.1
        assert result["extended_slots"][0]["pv_p10"] == 0.01
        assert result["extended_slots"][0]["pv_p90"] == 0.09
        assert result["extended_slots"][0]["load_p10"] == 0.02
        assert result["extended_slots"][0]["load_p90"] == 0.12

    async def test_aurora_extended_slots_use_load_fallback(self):
        price_slots = [{"start_time": datetime(2024, 1, 1, 0, 0, tzinfo=UTC)}]
        extended_records = [
            {
                "slot_start": datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                "final": {"pv_kwh": 0.0, "load_kwh": 0.0},
                "base": {"pv_kwh": 0.0},
                "probabilistic": {},
            }
        ]
        config = {
            "timezone": "UTC",
            "forecasting": {
                "active_forecast_version": "aurora",
                "aurora_load_enabled": True,
                "aurora_pv_enabled": True,
            },
        }
        ha_profile = [float(idx) for idx in range(96)]

        with (
            patch("backend.core.forecasts.build_db_forecast_for_slots", new=AsyncMock(return_value=[])),
            patch("backend.core.forecasts.get_forecast_slots", new=AsyncMock(return_value=extended_records)),
            patch("backend.core.ha_client.get_load_profile_from_ha", new=AsyncMock(return_value=ha_profile)),
        ):
            result = await get_forecast_data(price_slots, config)

        assert result["extended_slots"][0]["load_forecast_kwh"] == ha_profile[4]
        assert result["daily_load_forecast"]["2024-01-01"] == ha_profile[4]


if __name__ == "__main__":
    unittest.main()
