"""
Tests for REV F55: History Display Bug - Respect Inversion Flags

These tests verify that battery_power_inverted and grid_power_inverted flags
are correctly applied in the recorder and learning engine ETL pipelines.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from backend.learning.engine import LearningEngine
from backend.recorder import record_observation_from_current_state


@pytest.fixture
def mock_config():
    """Base config fixture without inversion flags."""
    return {
        "timezone": "Europe/Stockholm",
        "input_sensors": {
            "battery_power": "sensor.battery_power",
            "battery_soc": "sensor.battery_soc",
            "grid_power": "sensor.grid_power",
            "pv_power": "sensor.pv_power",
            "load_power": "sensor.load_power",
        },
        "system": {"grid_meter_type": "net"},
        "learning": {"sqlite_path": ":memory:"},
    }


@pytest.fixture
def mock_config_battery_inverted(mock_config):
    """Config with battery power inverted."""
    mock_config["input_sensors"]["battery_power_inverted"] = True
    return mock_config


@pytest.fixture
def mock_config_grid_inverted(mock_config):
    """Config with grid power inverted."""
    mock_config["input_sensors"]["grid_power_inverted"] = True
    return mock_config


@pytest.fixture
def mock_store():
    """Create a properly mocked LearningStore."""
    mock = MagicMock()
    mock.store_slot_observations = AsyncMock()
    mock.initialize = AsyncMock()
    mock.get_system_state = AsyncMock(return_value=None)
    mock.set_system_state = AsyncMock()
    mock.close = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


class TestRecorderBatteryInversion:
    """Test battery power inversion in recorder."""

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_inverted_battery_discharge_recorded_correctly(
        self,
        mock_get_kw,
        mock_get_float,
        mock_store_cls,
        mock_load_config,
        mock_config_battery_inverted,
        mock_store,
    ):
        """
        When battery_power_inverted=true, a raw positive value from HA
        should be treated as charging (negative after inversion).

        Sungrow convention: + = charging, - = discharging
        Standard convention: + = discharging, - = charging
        """
        mock_load_config.return_value = mock_config_battery_inverted
        mock_store_cls.return_value = mock_store

        # Sungrow reports +2.0 kW when charging
        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.battery_power":
                return 2.0  # Sungrow: positive = charging
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # After inversion: +2000W becomes -2000W (charging)
        # 2.0 kW * 0.25h = 0.5 kWh charge
        assert df.iloc[0]["batt_charge_kwh"] == 0.5
        assert df.iloc[0]["batt_discharge_kwh"] == 0.0

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_inverted_battery_charge_recorded_correctly(
        self,
        mock_get_kw,
        mock_get_float,
        mock_store_cls,
        mock_load_config,
        mock_config_battery_inverted,
        mock_store,
    ):
        """
        When battery_power_inverted=true, a raw negative value from HA
        should be treated as discharging (positive after inversion).
        """
        mock_load_config.return_value = mock_config_battery_inverted
        mock_store_cls.return_value = mock_store

        # Sungrow reports -2.0 kW when discharging
        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.battery_power":
                return -2.0  # Sungrow: negative = discharging
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # After inversion: -2000W becomes +2000W (discharging)
        # 2.0 kW * 0.25h = 0.5 kWh discharge
        assert df.iloc[0]["batt_discharge_kwh"] == 0.5
        assert df.iloc[0]["batt_charge_kwh"] == 0.0


class TestRecorderGridInversion:
    """Test grid power inversion in recorder with net meter."""

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_inverted_grid_import_recorded_correctly(
        self,
        mock_get_kw,
        mock_get_float,
        mock_store_cls,
        mock_load_config,
        mock_config_grid_inverted,
        mock_store,
    ):
        """
        When grid_power_inverted=true, a raw positive value from HA
        should be treated as export (negative after inversion).

        Some inverters report + = export, - = import
        Standard convention: + = import, - = export
        """
        mock_load_config.return_value = mock_config_grid_inverted
        mock_store_cls.return_value = mock_store

        # Inverted sensor reports +1.0 kW when exporting
        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.grid_power":
                return 1.0  # Inverted: positive = export
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # After inversion: +1000W becomes -1000W (export)
        # 1.0 kW * 0.25h = 0.25 kWh export
        assert df.iloc[0]["export_kwh"] == 0.25
        assert df.iloc[0]["import_kwh"] == 0.0

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_inverted_grid_export_recorded_correctly(
        self,
        mock_get_kw,
        mock_get_float,
        mock_store_cls,
        mock_load_config,
        mock_config_grid_inverted,
        mock_store,
    ):
        """
        When grid_power_inverted=true, a raw negative value from HA
        should be treated as import (positive after inversion).
        """
        mock_load_config.return_value = mock_config_grid_inverted
        mock_store_cls.return_value = mock_store

        # Inverted sensor reports -1.0 kW when importing
        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.grid_power":
                return -1.0  # Inverted: negative = import
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # After inversion: -1000W becomes +1000W (import)
        # 1.0 kW * 0.25h = 0.25 kWh import
        assert df.iloc[0]["import_kwh"] == 0.25
        assert df.iloc[0]["export_kwh"] == 0.0


class TestEnginePowerToSlotsInversion:
    """Test power-to-slots ETL with inversion flags."""

    @pytest.fixture
    def engine_with_inversion(self, tmp_path):
        """Create LearningEngine with inversion flags set."""
        # Create a minimal LearningEngine without needing full initialization
        engine = LearningEngine.__new__(LearningEngine)
        engine.timezone = pytz.timezone("Europe/Stockholm")
        engine.sensor_map = {}
        engine.inversion_flags = {"battery": True, "grid": True}
        engine.config = {"system": {"grid": {"max_power_kw": 8.0}}}
        return engine

    def test_battery_inversion_applied_in_etl(self, engine_with_inversion):
        """Test that battery inversion is applied in etl_power_to_slots."""
        tz = pytz.timezone("Europe/Stockholm")
        # Use exact 15-minute boundary
        now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)

        # Simulate Sungrow data: +2000W (charging in Sungrow convention)
        # Need at least 2 data points to create 1 slot
        power_data = {
            "sensor.battery_power": [
                (now, 2000.0),  # Raw: positive = charging
                (now + timedelta(minutes=15), 2000.0),
            ]
        }

        engine_with_inversion.inversion_flags = {"battery": True, "grid": False}

        df = engine_with_inversion.etl_power_to_slots(power_data, resolution_minutes=15)

        # After inversion: +2000W becomes -2000W, which is charge
        # Energy = 2.0 kW * 0.25h = 0.5 kWh
        assert len(df) >= 1
        assert df.iloc[0]["batt_charge_kwh"] == 0.5
        assert df.iloc[0]["batt_discharge_kwh"] == 0.0

    def test_grid_inversion_applied_in_etl(self, engine_with_inversion):
        """Test that grid inversion is applied in etl_power_to_slots."""
        tz = pytz.timezone("Europe/Stockholm")
        # Use exact 15-minute boundary
        now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)

        # Simulate inverted grid data: +1000W (export in inverted convention)
        # Need at least 2 data points to create 1 slot
        power_data = {
            "sensor.grid_power": [
                (now, 1000.0),  # Raw: positive = export
                (now + timedelta(minutes=15), 1000.0),
            ]
        }

        engine_with_inversion.inversion_flags = {"battery": False, "grid": True}

        df = engine_with_inversion.etl_power_to_slots(power_data, resolution_minutes=15)

        # After inversion: +1000W becomes -1000W, which is export
        # Energy = 1.0 kW * 0.25h = 0.25 kWh
        assert len(df) >= 1
        assert df.iloc[0]["export_kwh"] == 0.25
        assert df.iloc[0]["import_kwh"] == 0.0


class TestNonInvertedSensors:
    """Test that non-inverted sensors still work correctly."""

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_non_inverted_battery_standard_convention(
        self, mock_get_kw, mock_get_float, mock_store_cls, mock_load_config, mock_config, mock_store
    ):
        """Standard battery convention: + = discharge, - = charge."""
        mock_load_config.return_value = mock_config
        mock_store_cls.return_value = mock_store

        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.battery_power":
                return 2.0  # Standard: positive = discharge
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # Standard: +2000W = discharge
        assert df.iloc[0]["batt_discharge_kwh"] == 0.5
        assert df.iloc[0]["batt_charge_kwh"] == 0.0

    @pytest.mark.asyncio
    @patch("backend.recorder._load_config")
    @patch("backend.recorder.LearningStore")
    @patch("backend.recorder.get_ha_sensor_float", new_callable=AsyncMock)
    @patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock)
    async def test_non_inverted_grid_standard_convention(
        self, mock_get_kw, mock_get_float, mock_store_cls, mock_load_config, mock_config, mock_store
    ):
        """Standard grid convention: + = import, - = export."""
        mock_load_config.return_value = mock_config
        mock_store_cls.return_value = mock_store

        async def get_kw_side_effect(entity_id):
            if entity_id == "sensor.grid_power":
                return 1.0  # Standard: positive = import
            if entity_id == "sensor.load_power":
                return 1.5  # realistic house load so the load==0 glitch guard doesn't skip
            return 0.0

        mock_get_kw.side_effect = get_kw_side_effect

        async def get_float_side_effect(entity_id):
            if entity_id == "sensor.battery_soc":
                return 50.0
            return None

        mock_get_float.side_effect = get_float_side_effect

        await record_observation_from_current_state()

        args, _ = mock_store.store_slot_observations.call_args
        df = args[0]

        # Standard: +1000W = import
        assert df.iloc[0]["import_kwh"] == 0.25
        assert df.iloc[0]["export_kwh"] == 0.0
