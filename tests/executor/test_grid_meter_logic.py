from unittest.mock import AsyncMock, patch

import pytest

from backend.ha_socket import HAWebSocketClient
from backend.recorder import record_observation_from_current_state


# Mock inputs.get_ha_sensor_kw_normalized
@pytest.fixture
def mock_get_sensor():
    with patch("backend.recorder.get_ha_sensor_kw_normalized", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_config():
    with patch("backend.recorder._load_config") as mock:
        yield mock


@pytest.fixture
def mock_store():
    with patch("backend.recorder.LearningStore") as mock:
        yield mock


@pytest.mark.asyncio
async def test_recorder_net_meter_mode(mock_config, mock_get_sensor, mock_store):
    # Setup Config
    mock_config.return_value = {
        "system": {"grid_meter_type": "net"},
        "input_sensors": {
            "grid_power": "sensor.grid_net",
            "load_power": "sensor.load",
        },
    }

    # Setup Sensor Values (Grid = 0.5 kW Import). A realistic non-zero house load keeps the
    # recorder's "load==0 is a Sungrow glitch" guard from skipping the observation.
    async def side_effect(entity, **kwargs):
        if entity == "sensor.grid_net":
            return 0.5
        if entity == "sensor.load":
            return 1.0
        return 0.0

    mock_get_sensor.side_effect = side_effect

    # Make store.store_slot_observations an AsyncMock
    mock_store.return_value.store_slot_observations = AsyncMock()
    mock_store.return_value.initialize = AsyncMock()
    mock_store.return_value.get_system_state = AsyncMock(return_value=None)
    mock_store.return_value.set_system_state = AsyncMock()
    mock_store.return_value.close = AsyncMock()
    # Support async context manager
    mock_store.return_value.__aenter__ = AsyncMock(return_value=mock_store.return_value)
    mock_store.return_value.__aexit__ = AsyncMock(return_value=None)

    # Run
    await record_observation_from_current_state()

    # Verify Store call
    df = mock_store.return_value.store_slot_observations.call_args[0][0]
    # Import should be 0.5 kW * 0.25h = 0.125 kWh
    assert df.iloc[0]["import_kwh"] == 0.125
    assert df.iloc[0]["export_kwh"] == 0.0


@pytest.mark.asyncio
async def test_recorder_dual_meter_mode(mock_config, mock_get_sensor, mock_store):
    # Setup Config
    mock_config.return_value = {
        "system": {"grid_meter_type": "dual"},
        "input_sensors": {
            "grid_import_power": "sensor.grid_import",
            "grid_export_power": "sensor.grid_export",
            "load_power": "sensor.load",
        },
    }

    # Setup Sensor Values (Import=1.0 kW, Export=0.2 kW). A realistic non-zero house load keeps
    # the recorder's "load==0 is a Sungrow glitch" guard from skipping the observation.
    async def side_effect(entity, **kwargs):
        if entity == "sensor.grid_import":
            return 1.0
        if entity == "sensor.grid_export":
            return 0.2
        if entity == "sensor.load":
            return 1.0
        return 0.0

    mock_get_sensor.side_effect = side_effect

    # Make store.store_slot_observations an AsyncMock
    mock_store.return_value.store_slot_observations = AsyncMock()
    mock_store.return_value.initialize = AsyncMock()
    mock_store.return_value.get_system_state = AsyncMock(return_value=None)
    mock_store.return_value.set_system_state = AsyncMock()
    mock_store.return_value.close = AsyncMock()
    # Support async context manager
    mock_store.return_value.__aenter__ = AsyncMock(return_value=mock_store.return_value)
    mock_store.return_value.__aexit__ = AsyncMock(return_value=None)

    # Run
    await record_observation_from_current_state()

    # Verify Store call
    df = mock_store.return_value.store_slot_observations.call_args[0][0]
    # Import = 1.0 kW * 0.25 = 0.25 kWh
    # Export = 0.2 kW * 0.25 = 0.05 kWh
    assert df.iloc[0]["import_kwh"] == 0.25
    assert df.iloc[0]["export_kwh"] == 0.05


def test_ha_socket_dual_logic():
    # Test synthetic calculation in HA Socket
    client = HAWebSocketClient()
    # Mock monitored entities
    client.monitored_entities = {
        "sensor.import": "grid_import_kw",
        "sensor.export": "grid_export_kw",
    }

    # Mock emit_live_metrics
    with patch("backend.events.emit_live_metrics") as mock_emit:
        # 1. Receive Import = 2.5 kW
        client._handle_state_change(
            "sensor.import", {"state": "2500", "attributes": {"unit_of_measurement": "W"}}
        )

        # Check emit (import)
        args, _ = mock_emit.call_args
        payload = args[0]
        assert payload["grid_import_kw"] == 2.5
        # Grid kw should be 2.5 - 0 = 2.5
        assert payload["grid_kw"] == 2.5

        # 2. Receive Export = 0.5 kW
        client._handle_state_change(
            "sensor.export", {"state": "500", "attributes": {"unit_of_measurement": "W"}}
        )

        # Check emit (export)
        args, _ = mock_emit.call_args
        payload = args[0]
        assert payload["grid_export_kw"] == 0.5
        # Grid kw should be 2.5 - 0.5 = 2.0
        assert payload["grid_kw"] == 2.0
