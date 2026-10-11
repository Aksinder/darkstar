"""Tests for recorder delta-based energy calculation and state persistence."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from backend.recorder import RecorderStateStore, record_observation_from_current_state


class TestRecorderStateStore:
    """Test suite for RecorderStateStore class."""

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "nonexistent.json"
            store = RecorderStateStore(state_file)
            result = store.load()
            assert result == {}

    def test_load_corrupted_file(self):
        """Test loading corrupted file returns empty dict and removes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "corrupted.json"
            state_file.write_text("invalid json")
            store = RecorderStateStore(state_file)
            result = store.load()
            assert result == {}
            assert not state_file.exists()

    def test_save_and_load(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store._state = {"test_key": {"value": 100.0, "timestamp": "2024-01-01T00:00:00"}}
            store.save()

            # Load in new instance
            store2 = RecorderStateStore(state_file)
            result = store2.load()
            assert result["test_key"]["value"] == 100.0

    def test_get_delta_first_reading(self):
        """Test get_delta returns None on first reading (no previous state)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            now = datetime(2024, 1, 1, 12, 0, 0)
            delta, is_valid = store.get_delta("pv_total", 100.0, now)

            assert delta is None
            assert is_valid is True
            # State should be updated
            assert store._state["pv_total"]["value"] == 100.0

    def test_get_delta_normal_calculation(self):
        """Spec: Persistent Recorder State - Scenario: Recorder resumes after a restart"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            # First reading
            now = datetime(2024, 1, 1, 12, 0, 0)
            store.get_delta("pv_total", 100.0, now)

            # Second reading
            later = datetime(2024, 1, 1, 12, 15, 0)
            delta, is_valid = store.get_delta("pv_total", 150.0, later)

            assert delta == 50.0
            assert is_valid is True

    def test_get_delta_meter_reset_detection(self):
        """Test get_delta detects meter reset (negative delta)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            # First reading
            now = datetime(2024, 1, 1, 12, 0, 0)
            store.get_delta("pv_total", 1000.0, now)

            # Meter reset (reading goes down)
            later = datetime(2024, 1, 1, 12, 15, 0)
            delta, is_valid = store.get_delta("pv_total", 50.0, later)

            assert delta is None
            assert is_valid is False
            # State should be updated with new reading
            assert store._state["pv_total"]["value"] == 50.0

    def test_get_last_timestamp(self):
        """Test get_last_timestamp returns correct datetime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            now = datetime(2024, 1, 1, 12, 0, 0)
            store.get_delta("pv_total", 100.0, now)

            last_ts = store.get_last_timestamp("pv_total")
            assert last_ts == now

    def test_get_last_timestamp_missing_key(self):
        """Test get_last_timestamp returns None for missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            result = store.get_last_timestamp("nonexistent")
            assert result is None

    def test_single_live_recorder_prevents_second_zero_delta_from_becoming_stored(self):
        """Spec: Single Live Recorder Instance - duplicate shared-state cycles zero real data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            previous = datetime(2026, 6, 16, 10, 15, 0)
            slot_time = datetime(2026, 6, 16, 10, 30, 0)

            first_recorder = RecorderStateStore(state_file)
            first_recorder._state = {
                "pv_total": {"value": 100.0, "timestamp": previous.isoformat()}
            }
            first_recorder.save()

            first_delta, first_valid = first_recorder.get_delta("pv_total", 100.45, slot_time)

            second_recorder = RecorderStateStore(state_file)
            second_recorder.load()
            second_delta, second_valid = second_recorder.get_delta("pv_total", 100.45, slot_time)

            # A duplicate process sharing data/recorder_state.json computes zero for the same slot.
            assert first_delta == pytest.approx(0.45)
            assert first_valid is True
            assert second_delta == pytest.approx(0.0)
            assert second_valid is True

            stored_pv_kwh = first_delta
            assert stored_pv_kwh != second_delta


class TestRecorderDeltaLogic:
    """Test suite for recorder delta-based calculation logic."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
                "battery_soc": "sensor.battery_soc",
                # Cumulative sensors
                "total_pv_production": "sensor.total_pv_production",
                "total_load_consumption": "sensor.total_load_consumption",
            },
            "system": {"grid_meter_type": "net", "has_battery": True},
            # ARC15: water heater sensor now in water_heaters array
            "water_heaters": [{"enabled": True, "sensor": "sensor.water_power"}],
            "ev_chargers": [],
        }

    @pytest.mark.asyncio
    async def test_uses_cumulative_sensors_when_available(self, mock_config):
        """Spec: Delta-based Energy Calculation - Scenario: Recorder calculates energy during a continuous run"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"

            # Initialize state store with previous reading
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            # Mock HA sensor responses
            async def mock_get_ha_sensor_kw_normalized(entity):
                power_values = {
                    "sensor.pv_power": 5.0,  # 5 kW
                    "sensor.load_power": 3.0,  # 3 kW
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.5,
                    "sensor.water_power": 0.0,
                }
                return power_values.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                # Cumulative sensors - values increased by expected delta
                cumulative_values = {
                    "sensor.total_pv_production": 101.25,  # +1.25 kWh (5kW * 0.25h)
                    "sensor.total_load_consumption": 50.75,  # +0.75 kWh (3kW * 0.25h)
                    "sensor.battery_soc": 50.0,
                }
                return cumulative_values.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                # Create mock LearningStore
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=mock_config, state_store=state_store
                    )

                    # Verify observations were stored
                    assert mock_store.store_slot_observations.called
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Should use cumulative delta, not power snapshot
                    assert record["pv_kwh"] == pytest.approx(1.25, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_fallback_to_power_snapshot(self, mock_config):
        """Spec: Snapshot Fallback - Scenario: Missing total energy sensor"""
        # Remove cumulative sensors from config
        mock_config["input_sensors"].pop("total_pv_production", None)
        mock_config["input_sensors"].pop("total_load_consumption", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()

            async def mock_get_ha_sensor_kw_normalized(entity):
                power_values = {
                    "sensor.pv_power": 4.0,  # 4 kW
                    "sensor.load_power": 2.0,  # 2 kW
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.5,
                    "sensor.water_power": 0.0,
                }
                return power_values.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None  # No cumulative sensors available

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=mock_config, state_store=state_store
                    )

                    assert mock_store.store_slot_observations.called
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Should use power snapshot (4kW * 0.25h = 1.0 kWh)
                    assert record["pv_kwh"] == pytest.approx(1.0, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_meter_reset_fallback(self, mock_config):
        """Design Risk: Meter Reset/Rollover - Mitigation: Fallback to power snapshot when delta is negative"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"

            # Initialize state with high value
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 10000.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                power_values = {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 3.0,
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.5,
                    "sensor.water_power": 0.0,
                }
                return power_values.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                # Meter reset - value dropped significantly
                cumulative_values = {
                    "sensor.total_pv_production": 10.0,  # Reset from 10000 to 10
                    "sensor.total_load_consumption": 50.0,
                    "sensor.battery_soc": 50.0,
                }
                return cumulative_values.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=mock_config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Should fallback to power snapshot on meter reset
                    assert record["pv_kwh"] == pytest.approx(1.25, abs=0.01)  # 5kW * 0.25h

    @pytest.mark.asyncio
    async def test_dual_meter_cumulative(self):
        """Spec: Support for Cumulative Energy Sensors - Dual meter type support"""
        config = {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_import_power": "sensor.grid_import_power",
                "grid_export_power": "sensor.grid_export_power",
                "battery_power": "sensor.battery_power",
                "water_power": "sensor.water_power",
                "battery_soc": "sensor.battery_soc",
                "total_grid_import": "sensor.grid_import_total",
                "total_grid_export": "sensor.grid_export_total",
            },
            "system": {"grid_meter_type": "dual", "has_battery": True},
            "water_heaters": [],
            "ev_chargers": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"

            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "grid_import_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "grid_export_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return 1.0  # All power sensors return 1 kW

            async def mock_get_ha_sensor_float(entity):
                cumulative_values = {
                    "sensor.grid_import_total": 100.5,  # +0.5 kWh
                    "sensor.grid_export_total": 50.25,  # +0.25 kWh
                    "sensor.battery_soc": 50.0,
                }
                return cumulative_values.get(entity)

            async def mock_get_ha_entity_state(entity):
                # Return state structure with unit_of_measurement for cumulative sensors
                state_values = {
                    "sensor.grid_import_total": {
                        "state": "100.5",
                        "attributes": {"unit_of_measurement": "kWh"},
                    },
                    "sensor.grid_export_total": {
                        "state": "50.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                    },
                    "sensor.battery_soc": {
                        "state": "50.0",
                        "attributes": {"unit_of_measurement": "%"},
                    },
                }
                return state_values.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Should use cumulative deltas
                    assert record["import_kwh"] == pytest.approx(0.5, abs=0.01)
                    assert record["export_kwh"] == pytest.approx(0.25, abs=0.01)


class TestStatePersistence:
    """Test suite for state file persistence."""

    def test_state_file_format(self):
        """Test that state file is saved in expected JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            now = datetime(2024, 1, 1, 12, 0, 0)
            store.get_delta("pv_total", 100.0, now)
            store.get_delta("load_total", 50.0, now)
            store.save()

            # Verify file format
            content = json.loads(state_file.read_text())
            assert "pv_total" in content
            assert "load_total" in content
            assert "value" in content["pv_total"]
            assert "timestamp" in content["pv_total"]
            assert content["pv_total"]["value"] == 100.0

    def test_directory_creation(self):
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "dirs"
            state_file = nested_dir / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            now = datetime.now()
            store.get_delta("test", 100.0, now)
            store.save()

            assert nested_dir.exists()
            assert state_file.exists()


class TestRecorderSpikeValidation:
    """Test suite for recorder spike validation integration."""

    @pytest.mark.asyncio
    async def test_spike_values_zeroed_before_storage(self):
        """Test that spike values are zeroed before storage to database."""

        import pytz

        from backend.learning.store import LearningStore
        from backend.recorder import record_observation_from_current_state

        # Create test config with 8kW grid limit = 4.0 kWh max per slot
        config = {
            "timezone": "Europe/Stockholm",
            "system": {"grid": {"max_power_kw": 8.0}},
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {"battery_soc": "sensor.test_soc"},
        }

        # Mock HA responses with spike values
        with (
            patch("backend.recorder.get_ha_sensor_float") as mock_soc,
            patch("backend.recorder.get_ha_sensor_kw_normalized") as mock_power,
            patch("backend.recorder.get_ha_entity_state") as mock_entity,
        ):
            mock_soc.return_value = 50.0  # Valid SoC
            mock_power.return_value = 100.0  # Spike power value (100kW!)
            mock_entity.return_value = None  # No cumulative sensors

            # Create a mock store to capture what gets stored
            tz = pytz.timezone("Europe/Stockholm")
            store = LearningStore(":memory:", tz)
            try:
                await store.ensure_wal_mode()

                with patch("backend.recorder.LearningStore") as mock_store_class:
                    mock_store = AsyncMock()
                    mock_store_class.return_value = mock_store

                    # Capture the DataFrame passed to store_slot_observations
                    stored_records = []

                    async def capture_store(df):
                        stored_records.extend(df.to_dict("records"))

                    mock_store.store_slot_observations = capture_store
                    mock_store.close = AsyncMock()

                    # Record observation
                    await record_observation_from_current_state(config)

                    # Verify store was called
                    assert len(stored_records) == 1
                    record = stored_records[0]

                    # All energy values should be validated
                    # With 100kW power, the energy would be 100 * 0.25 = 25 kWh
                    # This exceeds the 4.0 kWh threshold, so it should be zeroed
                    assert record["pv_kwh"] == 0.0, (
                        f"PV spike should be zeroed, got {record['pv_kwh']}"
                    )
                    assert record["load_kwh"] == 0.0, (
                        f"Load spike should be zeroed, got {record['load_kwh']}"
                    )
            finally:
                await store.close()

    @pytest.mark.asyncio
    async def test_valid_values_preserved_in_recorder(self):
        """Test that valid values are preserved during recording."""
        config = {
            "timezone": "Europe/Stockholm",
            "system": {"grid": {"max_power_kw": 8.0}},
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "battery_soc": "sensor.test_soc",
                "pv_power": "sensor.pv_power",
            },
        }

        # Mock HA responses with normal values
        with (
            patch("backend.recorder.get_ha_sensor_float") as mock_soc,
            patch("backend.recorder.get_ha_sensor_kw_normalized") as mock_power,
            patch("backend.recorder.get_ha_entity_state") as mock_entity,
        ):
            mock_soc.return_value = 50.0

            def mock_power_side_effect(entity_id, default=0.0):
                if entity_id == "sensor.pv_power":
                    return 2.0  # Normal 2kW PV power
                return default

            mock_power.side_effect = mock_power_side_effect
            mock_entity.return_value = None

            stored_records = []

            with patch("backend.recorder.LearningStore") as mock_store_class:
                mock_store = AsyncMock()
                mock_store_class.return_value = mock_store

                async def capture_store(df):
                    stored_records.extend(df.to_dict("records"))

                mock_store.store_slot_observations = capture_store
                mock_store.close = AsyncMock()

                await record_observation_from_current_state(config)

                assert len(stored_records) == 1
                record = stored_records[0]

                # 2kW * 0.25h = 0.5 kWh, which is under the 4.0 kWh threshold
                assert record["pv_kwh"] == 0.5, (
                    f"Valid PV should be preserved, got {record['pv_kwh']}"
                )

    @pytest.mark.asyncio
    async def test_recorder_handles_missing_config_gracefully(self):
        """Test that recorder handles missing grid config gracefully."""
        config = {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {"battery_soc": "sensor.test_soc"},
            # Missing system.grid.max_power_kw
        }

        with (
            patch("backend.recorder.get_ha_sensor_float") as mock_soc,
            patch("backend.recorder.get_ha_sensor_kw_normalized") as mock_power,
            patch("backend.recorder.get_ha_entity_state") as mock_entity,
            patch("backend.recorder.logger") as mock_logger,
        ):
            mock_soc.return_value = 50.0
            mock_power.return_value = 2.0
            mock_entity.return_value = None

            stored_records = []

            with patch("backend.recorder.LearningStore") as mock_store_class:
                mock_store = AsyncMock()
                mock_store_class.return_value = mock_store

                async def capture_store(df):
                    stored_records.extend(df.to_dict("records"))

                mock_store.store_slot_observations = capture_store
                mock_store.close = AsyncMock()

                await record_observation_from_current_state(config)

                # Should log a warning about missing config
                warning_calls = [
                    c for c in mock_logger.warning.call_args_list if "max_power_kw" in str(c)
                ]
                assert len(warning_calls) > 0 or any(
                    "validate" in str(c).lower() for c in mock_logger.warning.call_args_list
                )

                # Should still store the record
                assert len(stored_records) == 1


class TestRecorderTimeScaling:
    """Test suite for time-proportional scaling to fix sawtooth pattern."""

    def test_time_proportional_scaling_short_interval(self):
        """Task 4.1: Scale up when sensor interval is shorter than 15 min.

        Raw delta covers 10 min, scaled to 15 min.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            # First reading at 10:43
            ts1 = datetime(2024, 1, 1, 10, 43, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # Sensor updated at 10:53 (10 min later); the recorder reads it at 10:58,
            # five minutes after that update — the live shape: counters polled every
            # 10 minutes, read every 15.
            ts2 = datetime(2024, 1, 1, 10, 58, 0)
            sensor_ts2 = datetime(2024, 1, 1, 10, 53, 0)
            delta, is_valid = store.get_delta("pv_total", 102.0, ts2, sensor_timestamp=sensor_ts2)

            # Raw delta: 2.0 kWh over 10 min
            # Scaled: 2.0 * (15/10) = 3.0 kWh — the extra 1.0 is what the counter
            # has gained in the 5 minutes since its own update.
            assert delta == pytest.approx(3.0, abs=0.01)
            assert is_valid is True
            assert store._state["pv_total"]["booked"] == pytest.approx(103.0)

    def test_scaling_up_is_limited_to_what_the_counter_can_have_gained(self):
        """A reading taken the moment the sensor updated has nothing to add: the
        10-minute delta is booked as it is, not stretched to 15 minutes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 43, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            ts2 = datetime(2024, 1, 1, 10, 53, 0)
            delta, is_valid = store.get_delta("pv_total", 102.0, ts2, sensor_timestamp=ts2)

            assert delta == pytest.approx(2.0)
            assert is_valid is True

    def test_time_proportional_scaling_long_interval(self):
        """Task 4.1: Scale down when sensor interval is longer than 15 min.

        Raw delta covers 20 min, scaled to 15 min.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            # First reading at 10:33
            ts1 = datetime(2024, 1, 1, 10, 33, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # Second reading at 10:53 (20 min later)
            ts2 = datetime(2024, 1, 1, 10, 53, 0)
            sensor_ts2 = datetime(2024, 1, 1, 10, 53, 0)
            delta, is_valid = store.get_delta("pv_total", 104.0, ts2, sensor_timestamp=sensor_ts2)

            # Raw delta: 4.0 kWh over 20 min. Scaled it would be 4.0 * (15/20) = 3.0,
            # but nothing of this counter has been booked ahead, so scaling down
            # would simply discard 1.0 kWh the counter has shown.
            assert delta == pytest.approx(4.0, abs=0.01)
            assert is_valid is True

    def test_long_interval_is_scaled_down_only_by_what_was_booked_ahead(self):
        """The sawtooth the scaling exists for: a 10-minute update read 5 minutes
        late books 15 minutes' worth, and the following 20-minute update then
        books 15 — together exactly what the counter moved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            store.load()

            t0 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, t0, sensor_timestamp=t0 - timedelta(minutes=5))
            # +2.0 over the 10 minutes 09:55 -> 10:05, read at 10:15
            first, _ = store.get_delta(
                "pv_total", 102.0, t0 + timedelta(minutes=15),
                sensor_timestamp=t0 + timedelta(minutes=5),
            )
            # +4.0 over the 20 minutes 10:05 -> 10:25, read at 10:30
            second, _ = store.get_delta(
                "pv_total", 106.0, t0 + timedelta(minutes=30),
                sensor_timestamp=t0 + timedelta(minutes=25),
            )

            assert first == pytest.approx(3.0)  # 2.0 * 15/10, one kWh booked ahead
            assert second == pytest.approx(3.0)  # 4.0 * 15/20, the kWh ahead now used up
            assert first + second == pytest.approx(6.0)
            assert "booked" not in store._state["pv_total"]  # level == reading again

    def test_backward_compatibility_missing_sensor_timestamp(self):
        """Task 4.2: No scaling when sensor_timestamp is None.

        Old state files or missing HA timestamps should use raw delta.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            # First reading without sensor timestamp
            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=None)

            # Second reading also without sensor timestamp
            ts2 = datetime(2024, 1, 1, 10, 15, 0)
            delta, is_valid = store.get_delta("pv_total", 103.0, ts2, sensor_timestamp=None)

            # No scaling applied
            assert delta == 3.0
            assert is_valid is True

    def test_backward_compatibility_old_state_file(self):
        """Task 4.2: Old state files without sensor_timestamp field work.

        Simulate loading from an old-format state file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            # Create old-format state file (no sensor_timestamp)
            old_state = {
                "pv_total": {
                    "value": 100.0,
                    "timestamp": "2024-01-01T10:00:00",
                    # No sensor_timestamp field
                }
            }
            state_file.write_text(json.dumps(old_state))

            store = RecorderStateStore(state_file)
            store.load()

            # New reading with sensor timestamp
            ts2 = datetime(2024, 1, 1, 10, 15, 0)
            sensor_ts2 = datetime(2024, 1, 1, 10, 13, 0)  # 13 min gap
            delta, is_valid = store.get_delta("pv_total", 102.6, ts2, sensor_timestamp=sensor_ts2)

            # Old state has no sensor_timestamp, so no scaling
            assert delta == pytest.approx(2.6, abs=0.001)
            assert is_valid is True

            # State file should now have sensor_timestamp for next cycle
            store2 = RecorderStateStore(state_file)
            store2.load()
            assert "sensor_timestamp" in store2._state["pv_total"]

    def test_scaling_window_bounds_too_short(self):
        """Task 4.3: No scaling when interval < 5 minutes.

        Likely sensor glitch or rapid updates.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # Only 3 minutes later (too short)
            ts2 = datetime(2024, 1, 1, 10, 3, 0)
            sensor_ts2 = datetime(2024, 1, 1, 10, 3, 0)
            delta, is_valid = store.get_delta("pv_total", 101.0, ts2, sensor_timestamp=sensor_ts2)

            # No scaling (outside 5-60 min window)
            assert delta == 1.0
            assert is_valid is True

    def test_scaling_window_bounds_too_long(self):
        """Task 4.3: No scaling when interval > 60 minutes.

        Likely restart or long gap.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # The SENSOR was silent for 90 minutes (too long), while the recorder
            # itself kept reading every quarter and booked nothing.
            for minutes in (15, 30, 45, 60, 75):
                quiet, _ = store.get_delta(
                    "pv_total", 100.0, ts1 + timedelta(minutes=minutes), sensor_timestamp=ts1
                )
                assert quiet == 0.0
            ts2 = datetime(2024, 1, 1, 11, 30, 0)
            sensor_ts2 = datetime(2024, 1, 1, 11, 30, 0)
            delta, is_valid = store.get_delta("pv_total", 110.0, ts2, sensor_timestamp=sensor_ts2)

            # No scaling (outside 5-60 min window)
            assert delta == 10.0
            assert is_valid is True
            assert store.multi_slot_deltas == {}

    def test_scaling_window_bounds_exactly_5_min(self):
        """Task 4.3: Scaling applied at exactly 5 minutes.

        Lower boundary of the valid window.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # Sensor updated exactly 5 minutes later, read at the end of the quarter
            ts2 = datetime(2024, 1, 1, 10, 15, 0)
            sensor_ts2 = datetime(2024, 1, 1, 10, 5, 0)
            delta, is_valid = store.get_delta("pv_total", 101.0, ts2, sensor_timestamp=sensor_ts2)

            # Scaling applies (1.0 * 15/5 = 3.0), but the booked level may run ahead
            # of the reading by one sensor interval's energy at most: 1.0 + 1.0.
            assert delta == pytest.approx(2.0, abs=0.01)
            assert is_valid is True

    def test_scaling_window_bounds_exactly_60_min(self):
        """Task 4.3: Scaling applied at exactly 60 minutes.

        Upper boundary of the valid window.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=ts1)

            # The sensor's next update comes exactly 60 minutes later; the recorder
            # read every quarter in between and booked nothing.
            for minutes in (15, 30, 45):
                store.get_delta(
                    "pv_total", 100.0, ts1 + timedelta(minutes=minutes), sensor_timestamp=ts1
                )
            ts2 = datetime(2024, 1, 1, 11, 0, 0)
            sensor_ts2 = datetime(2024, 1, 1, 11, 0, 0)
            delta, is_valid = store.get_delta("pv_total", 104.0, ts2, sensor_timestamp=sensor_ts2)

            # Inside the scaling window (4.0 * 15/60 = 1.0), but the three quiet
            # quarters booked nothing, so all 4.0 kWh are still owed — booking 1.0
            # was the leak that lost 3.0 kWh of every slow counter's tick.
            assert delta == pytest.approx(4.0, abs=0.01)
            assert is_valid is True

    def test_sensor_timestamp_persisted_to_state_file(self):
        """Task 4.2: sensor_timestamp is stored in state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.load()

            ts1 = datetime(2024, 1, 1, 10, 0, 0)
            sensor_ts1 = datetime(2024, 1, 1, 9, 58, 0)
            store.get_delta("pv_total", 100.0, ts1, sensor_timestamp=sensor_ts1)

            # Verify state file includes sensor_timestamp
            content = json.loads(state_file.read_text())
            assert "sensor_timestamp" in content["pv_total"]
            assert content["pv_total"]["sensor_timestamp"] == "2024-01-01T09:58:00"


def _polled_counter(power_kw, hours, *, poll_s=600, phase_s=200, resolution=0.1):
    """Home Assistant style history of a counter: (last_updated, value), one entry
    per CHANGE. The device integrates power_kw (a number or f(seconds)) and is
    polled every poll_s; an unchanged poll leaves last_updated where it was."""
    t0 = datetime(2026, 10, 1, 0, 0, 0)
    out = [(t0, 1000.0)]
    energy, last = 0.0, 1000.0
    for t in range(10, hours * 3600 + 10, 10):
        energy += (power_kw(t) if callable(power_kw) else power_kw) * 10 / 3600
        if t % poll_s == phase_s % poll_s:
            value = round(1000.0 + int(energy / resolution + 1e-9) * resolution, 6)
            if value != last:
                out.append((t0 + timedelta(seconds=t), value))
                last = value
    return out


def _replay(store, history, hours, *, skip=lambda t: False, key="k"):
    """Read the counter 7 s after every quarter, as the recorder does.

    Returns (per-slot deltas, counter value at the first read, at the last read)."""
    t0 = history[0][0]
    deltas, first_value, value = [], None, None
    for q in range(1, hours * 4 + 1):
        now = t0 + timedelta(minutes=15 * q, seconds=7)
        if skip(now):
            continue
        updated, value = [h for h in history if h[0] <= now][-1]
        delta, valid = store.get_delta(key, value, now, sensor_timestamp=updated)
        if first_value is None:
            first_value = value
            continue
        assert valid
        deltas.append(delta)
    return deltas, first_value, value


class TestRecorderBooksWhatTheCounterMoved:
    """Finding (b): the time-proportional scaling discarded energy.

    HA moves ``last_updated`` only when the value changes. A counter that ticks
    0.1 kWh after 40 quiet minutes had the tick scaled by 15/40 although the quiet
    slots had booked nothing; on the 10-minute poll grid even a steady 1 kW lost
    10 % to the same arithmetic. The slot energies must add up to what the counter
    moved, to within the one sensor interval that is still in flight.
    """

    @pytest.mark.parametrize("kw", [0.1, 0.2, 0.3, 0.5, 1.0, 3.0])
    @pytest.mark.parametrize("poll_s", [600, 10])
    def test_constant_load_is_fully_booked(self, kw, poll_s):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            history = _polled_counter(kw, 48, poll_s=poll_s, phase_s=poll_s if poll_s < 600 else 200)
            deltas, first, last = _replay(store, history, 48)

            assert all(d >= 0 for d in deltas)
            # Never behind the counter, never more than one sensor interval ahead.
            assert sum(deltas) >= last - first - 1e-9
            assert sum(deltas) <= last - first + max(0.1, kw * poll_s / 3600) + 1e-9

    def test_start_and_stop_edges_cancel(self):
        """2 kW for 40 minutes every 3 hours: what is booked ahead at a stop edge
        is owed back at the next start, so nothing accumulates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            history = _polled_counter(lambda t: 2.0 if (t % 10800) < 2400 else 0.0, 48)
            deltas, first, last = _replay(store, history, 48)

            assert sum(deltas) == pytest.approx(last - first, abs=0.35)
            assert max(deltas) <= 2.0 * 0.25 + 0.2  # no slot far above 15 min at 2 kW

    def test_a_steady_counter_is_still_smoothed(self):
        """The sawtooth fix must survive: 3 kW polled every 10 minutes books
        0.75 kWh in every quarter, not 0.5 / 1.0 alternating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            deltas, _, _ = _replay(store, _polled_counter(3.0, 12), 12)
            assert deltas == pytest.approx([0.75] * len(deltas), abs=1e-6)

    def test_state_written_by_the_previous_version_still_works(self):
        """No ``booked`` field: the stored reading itself is the booked level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "k": {
                            "value": 100.0,
                            "timestamp": "2026-10-01T10:00:07",
                            "sensor_timestamp": "2026-10-01T09:53:20",
                        }
                    }
                )
            )
            store = RecorderStateStore(state_file)
            store.load()
            delta, valid = store.get_delta(
                "k",
                100.5,
                datetime(2026, 10, 1, 10, 15, 7),
                sensor_timestamp=datetime(2026, 10, 1, 10, 13, 20),
            )
            assert valid
            # 0.5 kWh over 20 min would scale to 0.375, but all 0.5 is still owed.
            assert delta == pytest.approx(0.5)


class TestRecorderBackwardSteps:
    """Finding (a), decided against the real counter.

    The live load counter is a template over several inverter counters and is blind
    to the AC-coupled PV: while that PV charges the battery the template DECLINES,
    quarter after quarter, and does not come back (2026-10-04..11: 83 of 671
    quarters, all in daylight, none at night). A backward step is therefore not
    jitter around a high-water mark. The reading is rebased and the caller falls
    back for that slot; when the counter rises again its increments are real.
    """

    def test_sustained_decline_then_rise_books_the_rise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 10, 0, 7)
            values = [43085.2, 43084.8, 43084.5, 43084.1, 43083.6, 43084.0, 43084.6]
            results = [
                store.get_delta("load_total", v, t0 + timedelta(minutes=15 * i))
                for i, v in enumerate(values)
            ]
            assert results[0] == (None, True)
            assert results[1:5] == [(None, False)] * 4
            # Rising again from the LOWER level: booked, not withheld until 43085.2.
            assert results[5][1] is True and results[5][0] == pytest.approx(0.4)
            assert results[6][1] is True and results[6][0] == pytest.approx(0.6)

    def test_genuine_reset_is_rebased_and_reported_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 10, 0, 7)
            store.get_delta("pv_total", 1000.0, t0)
            with patch("backend.recorder.logger") as mock_logger:
                assert store.get_delta("pv_total", 0.3, t0 + timedelta(minutes=15)) == (None, False)
                assert "Meter reset" in mock_logger.warning.call_args[0][0]
            delta, valid = store.get_delta("pv_total", 0.9, t0 + timedelta(minutes=30))
            assert valid and delta == pytest.approx(0.6)

    def test_small_backward_step_is_not_called_a_meter_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 10, 0, 7)
            store.get_delta("load_total", 43085.2, t0)
            with patch("backend.recorder.logger") as mock_logger:
                store.get_delta("load_total", 43085.0, t0 + timedelta(minutes=15))
                message = mock_logger.warning.call_args[0][0]
            assert "went backwards" in message and "Meter reset" not in message

    def test_a_backward_step_drops_what_was_booked_ahead(self):
        """The booked level must not survive a rebase as a phantom high-water mark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 10, 0, 7)
            store.get_delta("k", 100.0, t0, sensor_timestamp=t0 - timedelta(minutes=5))
            store.get_delta(
                "k", 100.6, t0 + timedelta(minutes=15),
                sensor_timestamp=t0 + timedelta(minutes=5),
            )
            assert store._state["k"]["booked"] == pytest.approx(100.9)  # 0.6 * 15/10
            store.get_delta(
                "k", 100.2, t0 + timedelta(minutes=30),
                sensor_timestamp=t0 + timedelta(minutes=20),
            )
            assert "booked" not in store._state["k"]
            delta, valid = store.get_delta("k", 100.5, t0 + timedelta(minutes=45))
            assert valid and delta == pytest.approx(0.3)


class TestRecorderMultiSlotDelta:
    """Finding (d): a delta that spans several slots.

    After a recorder gap everything the counter moved arrives in one reading. It
    used to land whole in one 15-minute row (3 h of load in one quarter), or be
    zeroed by validation when above the per-slot limit. The row now gets the
    share a uniform spread assigns to it and reports the whole span.
    """

    def test_three_hour_gap_books_this_slots_share(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 9, 0, 7)
            store.get_delta("load_total", 5000.0, t0)
            delta, valid = store.get_delta("load_total", 5012.0, t0 + timedelta(hours=3))

            assert valid
            assert delta == pytest.approx(12.0 * 900 / 10800)  # 1.0 kWh, not 12.0
            assert store.multi_slot_deltas == {"load_total": {"kwh": 12.0, "seconds": 10800}}
            # The whole span is consumed: the next quarter starts from the new reading.
            nxt, _ = store.get_delta("load_total", 5013.0, t0 + timedelta(hours=3, minutes=15))
            assert nxt == pytest.approx(1.0)

    def test_gap_never_books_more_than_was_measured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            history = _polled_counter(1.0, 24)
            nine, noon = datetime(2026, 10, 1, 9, 0), datetime(2026, 10, 1, 12, 0)
            deltas, first, last = _replay(store, history, 24, skip=lambda t: nine <= t < noon)
            assert max(deltas) <= 0.45  # no row carries the 3 kWh of the gap
            assert sum(deltas) <= last - first
            assert sum(deltas) == pytest.approx(21 * 1.0, abs=0.5)  # 24 h minus the 3 h gap

    def test_a_normal_quarter_and_a_late_cycle_are_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 9, 0, 7)
            store.get_delta("k", 10.0, t0)
            store.get_delta("k", 10.5, t0 + timedelta(minutes=15, seconds=30))
            store.get_delta("k", 11.0, t0 + timedelta(minutes=37))  # 21.5 min: one late cycle
            assert store.multi_slot_deltas == {}

    def test_an_idle_counter_across_a_gap_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 9, 0, 7)
            store.get_delta("grid_export_total", 800.0, t0)
            delta, valid = store.get_delta("grid_export_total", 800.0, t0 + timedelta(hours=3))
            assert (delta, valid) == (0.0, True)
            assert store.multi_slot_deltas == {}


class TestRecorderStateTransaction:
    """Finding (c): a retry must not find the counters already consumed."""

    def test_staged_readings_are_not_persisted_until_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            t0 = datetime(2026, 10, 6, 9, 0, 7)
            store = RecorderStateStore(state_file)
            store.get_delta("k", 10.0, t0)

            store.begin()
            assert store.get_delta("k", 10.5, t0 + timedelta(minutes=15))[0] == pytest.approx(0.5)
            assert json.loads(state_file.read_text())["k"]["value"] == 10.0
            store.rollback()
            # Same question, same answer — in this object and in a fresh one.
            assert store.get_delta("k", 10.5, t0 + timedelta(minutes=15))[0] == pytest.approx(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.get_delta("k", 10.0, t0)
            store.begin()
            store.get_delta("k", 10.5, t0 + timedelta(minutes=15))
            store.commit()
            assert json.loads(state_file.read_text())["k"]["value"] == 10.5
            fresh = RecorderStateStore(state_file)
            fresh.load()
            assert fresh.get_delta("k", 10.5, t0 + timedelta(minutes=16))[0] == 0.0

    def test_a_key_read_twice_in_one_cycle_sees_its_own_staged_reading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecorderStateStore(Path(tmpdir) / "state.json")
            t0 = datetime(2026, 10, 6, 9, 0, 7)
            store.get_delta("k", 10.0, t0)
            store.begin()
            assert store.get_delta("k", 10.5, t0 + timedelta(minutes=15))[0] == pytest.approx(0.5)
            assert store.get_delta("k", 10.5, t0 + timedelta(minutes=15))[0] == 0.0

    def test_without_a_transaction_every_reading_is_persisted_at_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            store = RecorderStateStore(state_file)
            store.get_delta("k", 10.0, datetime(2026, 10, 6, 9, 0, 7))
            assert json.loads(state_file.read_text())["k"]["value"] == 10.0


_COUNTER_ENTITIES = {
    "load_total": "sensor.total_load",
    "grid_import_total": "sensor.total_import",
    "grid_export_total": "sensor.total_export",
    "battery_charge_total": "sensor.total_charge",
    "battery_discharge_total": "sensor.total_discharge",
}
_READINGS_BEFORE = {
    "sensor.total_load": 5000.0,
    "sensor.total_import": 3000.0,
    "sensor.total_export": 800.0,
    "sensor.total_charge": 600.0,
    "sensor.total_discharge": 500.0,
}
_READINGS_AFTER = {**_READINGS_BEFORE, "sensor.total_load": 5000.5, "sensor.total_import": 3000.5}


class TestRecorderRetryAndGapEndToEnd:
    """Findings (c) and (d) through record_observation_from_current_state, called
    the way RecorderService._record_with_retry calls it (no state store passed)."""

    @pytest.fixture
    def config(self):
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
                "battery_soc": "sensor.battery_soc",
                "total_load_consumption": "sensor.total_load",
                "total_grid_import": "sensor.total_import",
                "total_grid_export": "sensor.total_export",
                "total_battery_charge": "sensor.total_charge",
                "total_battery_discharge": "sensor.total_discharge",
            },
            "system": {
                "grid_meter_type": "net",
                "has_battery": True,
                "has_water_heater": False,
                "grid": {"max_power_kw": 17},
            },
            "water_heaters": [],
            "ev_chargers": [],
        }

    def _seed(self, state_file, minutes_ago, readings):
        store = RecorderStateStore(state_file)
        prev = (datetime.now(pytz.UTC) - timedelta(minutes=minutes_ago)).isoformat()
        store._state = {
            key: {"value": readings[entity], "timestamp": prev}
            for key, entity in _COUNTER_ENTITIES.items()
        }
        store.save()

    async def _cycle(self, config, state_file, readings, *, fail=False):
        """One recorder cycle; returns the stored record, or None if it raised."""
        now = datetime.now(pytz.UTC)

        async def kw(entity):
            return {"sensor.load_power": 2.0, "sensor.grid_power": 2.0}.get(entity, 0.0)

        async def soc(entity):
            return 50.0 if entity == "sensor.battery_soc" else None

        async def entity_state(entity):
            if entity not in readings:
                return None
            return {
                "state": str(readings[entity]),
                "attributes": {"unit_of_measurement": "kWh"},
                "last_updated": now.isoformat(),
            }

        db = MagicMock()
        db.get_system_state = AsyncMock(return_value=None)
        db.set_system_state = AsyncMock()
        db.store_slot_observations = AsyncMock(
            side_effect=RuntimeError("database is locked") if fail else None
        )
        db.close = AsyncMock()
        with (
            patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=kw),
            patch("backend.recorder.get_ha_sensor_float", side_effect=soc),
            patch("backend.recorder.get_ha_entity_state", side_effect=entity_state),
            patch(
                "backend.recorder.get_energy_from_power_history",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("backend.recorder.get_nordpool_data", return_value=None),
            patch("backend.recorder.LearningStore", return_value=db),
            patch(
                "backend.recorder.RecorderStateStore",
                side_effect=lambda *_a, **_k: RecorderStateStore(state_file),
            ),
        ):
            if fail:
                with pytest.raises(RuntimeError):
                    await record_observation_from_current_state(config)
                return None
            await record_observation_from_current_state(config)
        return db.store_slot_observations.call_args[0][0].iloc[0].to_dict()

    @pytest.mark.asyncio
    async def test_retry_after_a_failed_write_records_the_same_energy(self, config):
        """Attempt 1 raises at the DB write AFTER every delta was taken. Attempt 2
        used to read 0.0 from every counter and store a zero row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            self._seed(state_file, 15, _READINGS_BEFORE)

            assert await self._cycle(config, state_file, _READINGS_AFTER, fail=True) is None
            # Nothing was committed by the failed attempt.
            assert json.loads(state_file.read_text())["load_total"]["value"] == 5000.0

            record = await self._cycle(config, state_file, _READINGS_AFTER)
            assert record["load_kwh"] == pytest.approx(0.5)
            assert record["import_kwh"] == pytest.approx(0.5)
            assert not record.get("quality_flags")
            # ...and now it is.
            assert json.loads(state_file.read_text())["load_total"]["value"] == 5000.5

    @pytest.mark.asyncio
    async def test_a_lost_quarter_is_shared_out_and_flagged(self, config):
        """Both attempts failed for the previous quarter, so this reading spans two.
        The row gets one quarter's share and says so; a normal row says nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            self._seed(state_file, 30, _READINGS_BEFORE)
            two_quarters = {**_READINGS_BEFORE, "sensor.total_load": 5001.0, "sensor.total_import": 3001.0}

            record = await self._cycle(config, state_file, two_quarters)

            assert record["load_kwh"] == pytest.approx(0.5, abs=0.01)
            assert record["import_kwh"] == pytest.approx(0.5, abs=0.01)
            flags = json.loads(record["quality_flags"])
            assert set(flags) == {"multi_slot_delta"}
            assert set(flags["multi_slot_delta"]) == {"load_total", "grid_import_total"}
            assert flags["multi_slot_delta"]["load_total"]["kwh"] == pytest.approx(1.0)
            assert flags["multi_slot_delta"]["load_total"]["seconds"] == pytest.approx(1800, abs=5)

    @pytest.mark.asyncio
    async def test_an_ordinary_quarter_writes_no_flags(self, config):
        """quarantine.py reads NULL / '{}' as "no writer claimed this row"."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            self._seed(state_file, 15, _READINGS_BEFORE)
            record = await self._cycle(config, state_file, _READINGS_AFTER)
            assert "quality_flags" not in record or not record["quality_flags"]


class TestBackfillInterpolation:
    """Test suite for backfill interpolation fix."""

    def test_interpolation_produces_consistent_deltas(self):
        """Task 4.4: Interpolation eliminates sawtooth in backfill.

        Simulates the sawtooth scenario: readings at 10:53 and 11:03
        should produce consistent deltas when interpolated.
        """
        from backend.learning.engine import LearningEngine

        # Simulate readings that would cause sawtooth with ffill
        # Sensor updates at :03, :13, :23, :33, :43, :53
        # Backfill slots at :00, :15, :30, :45
        cumulative_data = {
            "sensor.pv_total": [
                # 10:53 reading (sensor timestamp)
                (datetime(2024, 1, 1, 10, 53, 0), 100.0),
                # 11:03 reading (10 min later)
                (datetime(2024, 1, 1, 11, 3, 0), 102.0),
                # 11:13 reading (10 min later)
                (datetime(2024, 1, 1, 11, 13, 0), 104.0),
                # 11:23 reading (10 min later)
                (datetime(2024, 1, 1, 11, 23, 0), 106.0),
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config for LearningEngine
            config = {
                "timezone": "UTC",
                "learning": {"sqlite_path": str(Path(tmpdir) / "test.db"), "sensor_map": {}},
                "input_sensors": {},
                "system": {"grid": {"max_power_kw": 10.0}},
            }
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(json.dumps(config))

            # Patch config loading
            with patch.object(LearningEngine, "_load_config", return_value=config):
                engine = LearningEngine(str(config_path))
                result = engine.etl_cumulative_to_slots(cumulative_data, resolution_minutes=15)

                # With interpolation, deltas should be consistent
                # 100 kWh at 10:53, 106 kWh at 11:23
                # Total time: 30 min, total delta: 6 kWh
                # Rate: 6 kWh / 30 min = 0.2 kWh per min
                # Per 15-min slot: 3.0 kWh

                if len(result) > 0 and "pv_kwh" in result.columns:
                    pv_deltas = result["pv_kwh"].tolist()
                    # All non-zero deltas should be approximately equal
                    non_zero = [d for d in pv_deltas if d > 0]
                    if len(non_zero) > 1:
                        # Variation should be small (not sawtooth pattern)
                        max_delta = max(non_zero)
                        min_delta = min(non_zero)
                        # Ratio should be close to 1:1 (not 2:1 sawtooth)
                        assert max_delta / min_delta < 1.5, (
                            f"Deltas show sawtooth pattern: {non_zero}"
                        )


class TestLoadIsolationFromDeferrableLoads:
    """Test suite for isolating base load from EV charging and water heating."""

    @pytest.fixture
    def base_config(self):
        """Create a base mock configuration with cumulative sensors."""
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
                "water_power": "sensor.water_power",
                "battery_soc": "sensor.battery_soc",
                "total_pv_production": "sensor.total_pv_production",
                "total_load_consumption": "sensor.total_load_consumption",
            },
            "system": {
                "grid_meter_type": "net",
                "has_battery": True,
                "has_water_heater": True,
                "has_ev_charger": True,
            },
            "water_heaters": [],
            "ev_chargers": [],
        }

    @pytest.mark.asyncio
    async def test_ev_charging_subtracted_from_total_load(self, base_config):
        """Spec: Load Isolation - EV charging subtracted from total load."""
        config = base_config.copy()
        config["ev_chargers"] = [{"sensor": "sensor.ev_power", "enabled": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 7.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 0.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "53.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(1.0, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(2.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_water_heating_subtracted_from_total_load(self, base_config):
        """Spec: Load Isolation - Water heating subtracted from total load."""
        config = base_config.copy()
        # ARC15: water heater sensor now in water_heaters array
        config["water_heaters"] = [{"enabled": True, "sensor": "sensor.water_power"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 5.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 3.0,
                }.get(entity, 0.0)

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "52.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.battery_soc": {
                        "state": "50.0",
                        "attributes": {"unit_of_measurement": "%"},
                    },
                }.get(entity)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch(
                    "backend.recorder.get_ha_entity_state",
                    side_effect=mock_get_ha_entity_state,
                ),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["water_kwh"] == pytest.approx(0.75, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(1.25, abs=0.01)

    @pytest.mark.asyncio
    async def test_both_ev_and_water_subtracted_from_total_load(self, base_config):
        """Spec: Load Isolation - Both EV and water subtracted together."""
        config = base_config.copy()
        config["ev_chargers"] = [{"sensor": "sensor.ev_power", "enabled": True}]
        # ARC15: water heater sensor now in water_heaters array
        config["water_heaters"] = [{"enabled": True, "sensor": "sensor.water_power"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 11.0,
                    "sensor.grid_power": 6.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 3.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "55.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(1.0, abs=0.01)
                    assert record["water_kwh"] == pytest.approx(0.75, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(3.25, abs=0.01)

    @pytest.mark.asyncio
    async def test_negative_base_load_clamped_to_zero(self, base_config):
        """Spec: Load Isolation - Negative base load clamped to zero with warning."""
        config = base_config.copy()
        config["ev_chargers"] = [{"sensor": "sensor.ev_power", "enabled": True}]
        # ARC15: water heater sensor now in water_heaters array
        config["water_heaters"] = [{"enabled": True, "sensor": "sensor.water_power"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 12.0,
                    "sensor.grid_power": 10.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 4.0,
                    "sensor.ev_power": 8.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "52.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.logger") as mock_logger,
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(2.0, abs=0.01)
                    assert record["water_kwh"] == pytest.approx(1.0, abs=0.01)
                    # 2026-08 balance-rescue: a clamped-to-zero base load is exactly the
                    # kind of impossible value the energy-balance rescue replaces — the
                    # record now carries the balance-derived base load instead of a fake
                    # 0.0 (which would poison training and eval like the skip-guard did).
                    assert record["load_kwh"] > 0.0
                    assert "load_rescued" in str(record.get("quality_flags", ""))

                    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                    assert any("Negative base load" in w for w in warning_calls)

    @pytest.mark.asyncio
    async def test_zero_load_glitch_records_with_balance_rescue(self, base_config):
        """REGRESSION (2026-08 incident): a glitched load_power=0 read must NOT skip the
        observation — PV/grid/battery are healthy, so the slot is recorded with a
        balance-derived load instead. The old skip → zero-backfill chain corrupted every
        strong-export midday and produced the phantom forecast-MAE regression."""
        config = base_config.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 6.0,
                    "sensor.load_power": 0.0,  # the Sungrow glitch
                    "sensor.grid_power": -4.0,  # exporting 4 kW
                    "sensor.battery_power": -1.0,  # charging 1 kW
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 80.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.5",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    # The observation was RECORDED (old behavior: skipped entirely).
                    mock_store.store_slot_observations.assert_called_once()
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # PV from the cumulative delta, untouched by the glitch.
                    assert record["pv_kwh"] == pytest.approx(1.5, abs=0.01)
                    # load = pv 1.5 + import 0 - export 1.0 + dis 0 - cha 0.25 = 0.25
                    assert record["load_kwh"] == pytest.approx(0.25, abs=0.01)
                    assert "load_rescued" in str(record.get("quality_flags", ""))

    @pytest.mark.asyncio
    async def test_transient_glitch_keeps_healthy_cumulative_measurement(self, base_config):
        """REVIEW BLOCKER #3: a transient load_power=0 read with a HEALTHY cumulative
        load-counter delta must keep the MEASURED value — the counter is the better
        instrument; the balance estimate must not overwrite it."""
        config = base_config.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 6.0,
                    "sensor.load_power": 0.0,  # transient glitch tick
                    "sensor.grid_power": -4.0,
                    "sensor.battery_power": -1.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 80.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.5",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    # HEALTHY cumulative load counter: 50.0 -> 50.4 = 0.4 kWh measured.
                    "sensor.total_load_consumption": {
                        "state": "50.4",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Measured 0.4 kWh kept — NOT the balance estimate (0.25).
                    assert record["load_kwh"] == pytest.approx(0.4, abs=0.01)
                    assert "load_rescued" not in str(record.get("quality_flags", ""))
                    # But provenance of the invalid power read is still tagged.
                    assert "load_read_invalid" in str(record.get("quality_flags", ""))

    @pytest.mark.asyncio
    async def test_zero_load_glitch_rescue_uses_battery_counter_deltas(self, base_config):
        """REVIEW #4 (2026-08-03): with total_battery_charge/discharge counters configured,
        the recorded batt energies — and therefore the balance-rescue — must come from the
        counter deltas, not the single battery_kw power snapshot. A burst charge inside the
        slot is invisible to an instantaneous read (snapshot says 1 kW ≈ 0.25 kWh here, the
        counters say 0.35 charged + 0.05 discharged)."""
        config = base_config.copy()
        config["input_sensors"] = {
            **config["input_sensors"],
            "total_battery_charge": "sensor.total_battery_charge",
            "total_battery_discharge": "sensor.total_battery_discharge",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "battery_charge_total": {"value": 200.0, "timestamp": prev_time.isoformat()},
                "battery_discharge_total": {"value": 300.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 6.0,
                    "sensor.load_power": 0.0,  # the Sungrow glitch
                    "sensor.grid_power": -4.0,  # exporting 4 kW
                    "sensor.battery_power": -1.0,  # snapshot catches 1 kW of charge
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 80.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.5",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    # Burst charge: counters saw 0.35 kWh in + 0.05 kWh out this slot.
                    "sensor.total_battery_charge": {
                        "state": "200.35",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_battery_discharge": {
                        "state": "300.05",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Recorded batt energies are the counter deltas, not snapshot*0.25.
                    assert record["batt_charge_kwh"] == pytest.approx(0.35, abs=0.01)
                    assert record["batt_discharge_kwh"] == pytest.approx(0.05, abs=0.01)
                    # Rescue balance uses them:
                    # load = pv 1.5 + import 0 - export 1.0 + dis 0.05 - cha 0.35 = 0.2
                    # (snapshot-derived batt would have said 0.25 instead).
                    assert record["load_kwh"] == pytest.approx(0.2, abs=0.01)
                    assert "load_rescued" in str(record.get("quality_flags", ""))

    @pytest.mark.asyncio
    async def test_battery_counter_deltas_run_on_healthy_ticks_too(self, base_config):
        """REVIEW #4 companion: the counter-delta calls consume RecorderStateStore state,
        so they must run in the MAIN path every tick — a healthy (non-rescue) slot must
        also record counter-derived batt energies and advance the stored counter values.
        If the deltas ran only inside the rescue, the next rescue's delta would span
        every slot since the previous rescue."""
        config = base_config.copy()
        config["input_sensors"] = {
            **config["input_sensors"],
            "total_battery_charge": "sensor.total_battery_charge",
            "total_battery_discharge": "sensor.total_battery_discharge",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "battery_charge_total": {"value": 200.0, "timestamp": prev_time.isoformat()},
                "battery_discharge_total": {"value": 300.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 6.0,
                    "sensor.load_power": 2.0,  # healthy read — no rescue
                    "sensor.grid_power": -4.0,
                    "sensor.battery_power": -1.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 80.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_battery_charge": {
                        "state": "200.35",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_battery_discharge": {
                        "state": "300.05",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["batt_charge_kwh"] == pytest.approx(0.35, abs=0.01)
                    assert record["batt_discharge_kwh"] == pytest.approx(0.05, abs=0.01)
                    assert "load_rescued" not in str(record.get("quality_flags", ""))
                    # The stored counter state advanced on this ordinary tick.
                    assert state_store._state["battery_charge_total"]["value"] == pytest.approx(
                        200.35
                    )
                    assert state_store._state["battery_discharge_total"]["value"] == pytest.approx(
                        300.05
                    )

    @pytest.mark.asyncio
    async def test_power_snapshot_fallback_uses_base_load(self, base_config):
        """Spec: Load Isolation - Power snapshot fallback uses base load from disaggregator."""
        config = base_config.copy()
        config["input_sensors"].pop("total_load_consumption", None)
        config["input_sensors"].pop("total_pv_production", None)
        config["ev_chargers"] = [{"sensor": "sensor.ev_power", "enabled": True}]
        # ARC15: water heater sensor now in water_heaters array
        config["water_heaters"] = [{"enabled": True, "sensor": "sensor.water_power"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 4.0,
                    "sensor.load_power": 7.0,
                    "sensor.grid_power": 3.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 1.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            mock_disaggregator = MagicMock()
            mock_disaggregator.update_current_power = AsyncMock(return_value=5.0)
            mock_disaggregator.calculate_base_load = MagicMock(return_value=2.0)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config,
                        disaggregator=mock_disaggregator,
                        state_store=state_store,
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(1.0, abs=0.01)
                    assert record["water_kwh"] == pytest.approx(0.25, abs=0.01)
                    assert record["load_kwh"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_ev_power_history_recording(self, base_config):
        """Task 4.2: EV energy from get_energy_from_power_history, single charger."""
        config = base_config.copy()
        config["ev_chargers"] = [{"id": "ev1", "sensor": "sensor.ev_power", "enabled": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 7.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "53.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            async def mock_history(entity_id, start, end):
                return 0.9

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(0.9, abs=0.01)
                    # Load 3.0 - EV 0.9 = 2.1
                    assert record["load_kwh"] == pytest.approx(2.1, abs=0.01)

    @pytest.mark.asyncio
    async def test_ev_power_history_multiple_chargers_sum(self, base_config):
        """Task 4.2: Multiple EV chargers sum their history-based energy."""
        config = base_config.copy()
        config["ev_chargers"] = [
            {"id": "ev1", "sensor": "sensor.ev1_power", "enabled": True},
            {"id": "ev2", "sensor": "sensor.ev2_power", "enabled": True},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            history_calls: list[str] = []

            async def mock_history(entity_id, start, end):
                history_calls.append(entity_id)
                return {"sensor.ev1_power": 0.8, "sensor.ev2_power": 0.6}.get(entity_id)

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 8.0,
                    "sensor.grid_power": 3.0,
                    "sensor.battery_power": 0.0,
                    "sensor.ev1_power": 4.0,
                    "sensor.ev2_power": 3.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "54.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # 0.8 + 0.6 = 1.4 kWh total
                    assert record["ev_charging_kwh"] == pytest.approx(1.4, abs=0.01)
                    assert "sensor.ev1_power" in history_calls
                    assert "sensor.ev2_power" in history_calls

    @pytest.mark.asyncio
    async def test_water_power_history_recording(self, base_config):
        """Task 4.3: Water heater energy from get_energy_from_power_history."""
        config = base_config.copy()
        config["water_heaters"] = [{"id": "wh1", "sensor": "sensor.water_power", "enabled": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 5.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 3.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "52.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            async def mock_history(entity_id, start, end):
                return 0.7

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["water_kwh"] == pytest.approx(0.7, abs=0.01)
                    # Load 2.0 - water 0.7 = 1.3
                    assert record["load_kwh"] == pytest.approx(1.3, abs=0.01)

    @pytest.mark.asyncio
    async def test_snapshot_fallback_when_history_returns_none(self, base_config):
        """Task 4.4: When get_energy_from_power_history returns None, use power_kw x 0.25."""
        config = base_config.copy()
        config["ev_chargers"] = [{"id": "ev1", "sensor": "sensor.ev_power", "enabled": True}]
        config["water_heaters"] = [{"id": "wh1", "sensor": "sensor.water_power", "enabled": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 8.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 2.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "54.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            async def mock_history_none(entity_id, start, end):
                return None

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    side_effect=mock_history_none,
                ),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Fallback: ev=4.0*0.25=1.0, water=2.0*0.25=0.5
                    assert record["ev_charging_kwh"] == pytest.approx(1.0, abs=0.01)
                    assert record["water_kwh"] == pytest.approx(0.5, abs=0.01)
                    # Snapshot fallback must not write per-device rows (they would
                    # last-write-wins over authoritative history values).
                    assert record["ev_charger_energy"] is None
                    assert record["water_heater_energy"] is None

    @pytest.mark.asyncio
    async def test_load_isolation_always_applied_regardless_of_source(self, base_config):
        """Task 4.5: EV + water subtracted from total load regardless of history or snapshot."""
        config = base_config.copy()
        config["ev_chargers"] = [{"id": "ev1", "sensor": "sensor.ev_power", "enabled": True}]
        config["water_heaters"] = [{"id": "wh1", "sensor": "sensor.water_power", "enabled": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 8.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                    "sensor.water_power": 2.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                if entity == "sensor.battery_soc":
                    return 50.0
                return None

            async def mock_get_ha_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": "101.25",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "54.0",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            for history_value in (1.0, None):
                state_store_iter = RecorderStateStore(Path(tmpdir) / "state_iter.json")
                state_store_iter.load()
                state_store_iter._state = {
                    "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                    "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
                }
                state_store_iter.save()

                async def mock_history(entity_id, start, end, _v=history_value):
                    return _v

                with (
                    patch(
                        "backend.recorder.get_ha_sensor_kw_normalized",
                        side_effect=mock_get_ha_sensor_kw_normalized,
                    ),
                    patch(
                        "backend.recorder.get_ha_sensor_float",
                        side_effect=mock_get_ha_sensor_float,
                    ),
                    patch(
                        "backend.recorder.get_ha_entity_state",
                        side_effect=mock_get_ha_entity_state,
                    ),
                    patch("backend.recorder.get_nordpool_data", return_value=None),
                    patch(
                        "backend.recorder.get_energy_from_power_history",
                        side_effect=mock_history,
                    ),
                ):
                    mock_store = MagicMock()
                    mock_store.get_system_state = AsyncMock(return_value=None)
                    mock_store.set_system_state = AsyncMock()
                    mock_store.store_slot_observations = AsyncMock()
                    mock_store.close = AsyncMock()

                    with patch("backend.recorder.LearningStore", return_value=mock_store):
                        await record_observation_from_current_state(
                            config=config, state_store=state_store_iter
                        )

                        df = mock_store.store_slot_observations.call_args[0][0]
                        record = df.iloc[0].to_dict()

                        # Load isolation always applied regardless of EV/water source.
                        # Total load delta = 54.0 - 50.0 = 4.0 kWh.
                        # EV + water > 0 → base load must be < total load.
                        assert record["load_kwh"] < 4.0, (
                            f"Load isolation not applied (history_value={history_value}): "
                            f"load_kwh={record['load_kwh']}"
                        )


class TestPerDeviceEVRecording:
    """Task 8.3: Per-device EV charging energy recorded alongside aggregate."""

    @pytest.fixture
    def base_config(self):
        return {
            "system": {
                "has_ev_charger": True,
                "has_battery": True,
                "has_solar": False,
                "grid_meter_type": "net",
            },
            "input_sensors": {
                "battery_soc": "sensor.battery_soc",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
            },
            "battery": {"capacity_kwh": 10.0},
            "ev_chargers": [],
            "water_heaters": [],
        }

    def _make_common_patches(self, now):
        """Return common mocks reused across tests."""

        async def mock_sensor_kw(entity):
            return {
                "sensor.load_power": 3.0,
                "sensor.grid_power": 2.0,
                "sensor.battery_power": 0.0,
                "sensor.ev_a_power": 3.2,
                "sensor.ev_b_power": 1.8,
            }.get(entity, 0.0)

        async def mock_float(entity):
            if entity == "sensor.battery_soc":
                return 50.0
            return None

        async def mock_entity_state(entity):
            return None

        async def mock_history(entity_id, start, end):
            return {"sensor.ev_a_power": 0.8, "sensor.ev_b_power": 0.45}.get(entity_id)

        return mock_sensor_kw, mock_float, mock_entity_state, mock_history

    @pytest.mark.asyncio
    async def test_per_device_kwh_recorded_for_two_chargers(self, base_config):
        """Two chargers produce individual ev_charger_energy entries."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["ev_chargers"] = [
            {"id": "ev_a", "sensor": "sensor.ev_a_power", "enabled": True},
            {"id": "ev_b", "sensor": "sensor.ev_b_power", "enabled": True},
        ]

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)
        mock_sensor_kw, mock_float, mock_entity_state, mock_history = self._make_common_patches(now)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Aggregate is sum of both
                    assert record["ev_charging_kwh"] == pytest.approx(0.8 + 0.45, abs=0.01)

                    # Per-device dict has individual values
                    per_device = record["ev_charger_energy"]
                    assert per_device is not None
                    assert per_device["ev_a"] == pytest.approx(0.8, abs=0.01)
                    assert per_device["ev_b"] == pytest.approx(0.45, abs=0.01)

    @pytest.mark.asyncio
    async def test_per_device_kwh_none_when_no_chargers(self, base_config):
        """No chargers → ev_charger_energy is None."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["ev_chargers"] = []

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_sensor_kw(entity):
                return {
                    "sensor.load_power": 2.0,
                    "sensor.grid_power": 2.0,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_float(entity):
                return 50.0 if entity == "sensor.battery_soc" else None

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=lambda e: None),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["ev_charging_kwh"] == pytest.approx(0.0, abs=0.01)
                    assert record["ev_charger_energy"] is None

    @pytest.mark.asyncio
    async def test_aggregate_backward_compat_preserved(self, base_config):
        """ev_charging_kwh aggregate is still present for backward compatibility."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["ev_chargers"] = [{"id": "ev1", "sensor": "sensor.ev_power", "enabled": True}]

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_sensor_kw(entity):
                return {
                    "sensor.load_power": 4.0,
                    "sensor.grid_power": 4.0,
                    "sensor.battery_power": 0.0,
                    "sensor.ev_power": 4.0,
                }.get(entity, 0.0)

            async def mock_float(entity):
                return 50.0 if entity == "sensor.battery_soc" else None

            async def mock_history(entity_id, start, end):
                return 1.0 if entity_id == "sensor.ev_power" else None

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=lambda e: None),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Aggregate still present
                    assert "ev_charging_kwh" in record
                    assert record["ev_charging_kwh"] == pytest.approx(1.0, abs=0.01)
                    # Per-device also present
                    assert record["ev_charger_energy"]["ev1"] == pytest.approx(1.0, abs=0.01)


class TestPerDeviceWaterHeaterRecording:
    """Task 8.4: Per-device water heater energy recorded alongside aggregate."""

    @pytest.fixture
    def base_config(self):
        return {
            "system": {
                "has_water_heater": True,
                "has_battery": True,
                "has_solar": False,
                "grid_meter_type": "net",
            },
            "input_sensors": {
                "battery_soc": "sensor.battery_soc",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
            },
            "battery": {"capacity_kwh": 10.0},
            "ev_chargers": [],
            "water_heaters": [],
        }

    @pytest.mark.asyncio
    async def test_per_device_kwh_recorded_for_two_heaters(self, base_config):
        """Two heaters produce individual water_heater_energy entries."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["water_heaters"] = [
            {"id": "wh1", "sensor": "sensor.wh1_power", "enabled": True, "power_kw": 3.0},
            {"id": "wh2", "sensor": "sensor.wh2_power", "enabled": True, "power_kw": 3.0},
        ]

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)

        async def mock_sensor_kw(entity):
            return {
                "sensor.load_power": 6.0,
                "sensor.grid_power": 6.0,
                "sensor.battery_power": 0.0,
            }.get(entity, 0.0)

        async def mock_float(entity):
            return 50.0 if entity == "sensor.battery_soc" else None

        async def mock_entity_state(entity):
            return None

        async def mock_history(entity_id, start, end):
            return {"sensor.wh1_power": 0.75, "sensor.wh2_power": 0.5}.get(entity_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Aggregate is sum of both heaters
                    assert record["water_kwh"] == pytest.approx(0.75 + 0.5, abs=0.01)

                    # Per-device dict has individual values
                    per_device = record["water_heater_energy"]
                    assert per_device is not None
                    assert per_device["wh1"] == pytest.approx(0.75, abs=0.01)
                    assert per_device["wh2"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_water_heater_energy_none_when_no_heaters(self, base_config):
        """No heaters → water_heater_energy is None."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["water_heaters"] = []

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)

        async def mock_sensor_kw(entity):
            return {
                "sensor.load_power": 2.0,
                "sensor.grid_power": 2.0,
                "sensor.battery_power": 0.0,
            }.get(entity, 0.0)

        async def mock_float(entity):
            return 50.0 if entity == "sensor.battery_soc" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", return_value=None),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["water_heater_energy"] is None

    @pytest.mark.asyncio
    async def test_snapshot_fallback_when_no_history(self, base_config):
        """When history API returns None the AGGREGATE falls back to the power
        snapshot, but NO per-device value is recorded: a snapshot is not a slot
        measurement, and the store's last-write-wins device upsert would let a
        fallback re-record permanently zero an authoritative history value."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytz

        from backend.recorder import RecorderStateStore, record_observation_from_current_state

        config = base_config.copy()
        config["water_heaters"] = [
            {"id": "wh1", "sensor": "sensor.wh1_power", "enabled": True, "power_kw": 2.0},
        ]

        now = datetime.now(pytz.timezone("Europe/Stockholm"))
        prev_time = now - timedelta(minutes=15)

        async def mock_sensor_kw(entity):
            return {
                "sensor.load_power": 2.0,
                "sensor.grid_power": 2.0,
                "sensor.battery_power": 0.0,
                "sensor.wh1_power": 2.0,
            }.get(entity, 0.0)

        async def mock_float(entity):
            return 50.0 if entity == "sensor.battery_soc" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            state_store._state = {
                "pv_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 0.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_sensor_kw),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", return_value=None),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", return_value=None),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Snapshot: 2.0 kW x 0.25 h = 0.5 kWh — aggregate only.
                    assert record["water_kwh"] == pytest.approx(0.5, abs=0.01)
                    # Omitted from the per-device dict => the store writes no row,
                    # so an earlier authoritative value survives a re-record.
                    assert record["water_heater_energy"] is None


class TestSensorGuards:
    """Tests for water heater/EV sensor guards (executor-performance-fixes)."""

    def test_recorder_skips_water_heater_when_disabled(self):
        """Task 3.4: Recorder skips water heater sensor fetch when has_water_heater is false."""
        config = {
            "input_sensors": {
                "pv_power": "sensor.pv",
                "load_power": "sensor.load",
                "battery_power": "sensor.battery",
                "grid_power": "sensor.grid",
            },
            "water_heaters": [{"id": "wh1", "enabled": True, "sensor": "sensor.water_power"}],
            "system": {
                "timezone": "Europe/Stockholm",
                "grid_meter_type": "net",
                "has_water_heater": False,
            },
        }

        power_reads = [
            ("pv_power", lambda: 1.0),
            ("load_power", lambda: 2.0),
            ("battery_power", lambda: 3.0),
        ]

        water_heater_sensors = []
        if config.get("system", {}).get("has_water_heater", True):
            for water_heater in config.get("water_heaters", []):
                if water_heater.get("enabled", True):
                    sensor = water_heater.get("sensor")
                    if sensor:
                        water_heater_sensors.append(str(sensor))
                        power_reads.append((f"wh_{sensor}", lambda s=str(sensor): 4.0))

        assert len(water_heater_sensors) == 0
        assert not any(name.startswith("wh_") for name, _ in power_reads)

    def test_recorder_skips_ev_charger_when_disabled(self):
        """Task 3.5: Recorder skips EV charger sensor fetch when has_ev_charger is false."""
        config = {
            "input_sensors": {
                "pv_power": "sensor.pv",
                "load_power": "sensor.load",
                "battery_power": "sensor.battery",
                "grid_power": "sensor.grid",
            },
            "ev_chargers": [{"id": "ev1", "enabled": True, "sensor": "sensor.ev_power"}],
            "system": {
                "timezone": "Europe/Stockholm",
                "grid_meter_type": "net",
                "has_ev_charger": False,
            },
        }

        power_reads = [
            ("pv_power", lambda: 1.0),
            ("load_power", lambda: 2.0),
            ("battery_power", lambda: 3.0),
        ]

        ev_charger_sensors = []
        if config.get("system", {}).get("has_ev_charger", False):
            ev_chargers = config.get("ev_chargers", [])
            for ev_charger in ev_chargers:
                if ev_charger.get("enabled", True):
                    sensor = ev_charger.get("sensor")
                    if sensor:
                        ev_charger_sensors.append(str(sensor))
                        power_reads.append((f"ev_{sensor}", lambda s=str(sensor): 4.0))

        assert len(ev_charger_sensors) == 0
        assert not any(name.startswith("ev_") for name, _ in power_reads)

    def test_recorder_fetches_water_heater_when_enabled(self):
        """Water heater sensors are fetched when has_water_heater is true."""
        config = {
            "input_sensors": {
                "pv_power": "sensor.pv",
                "load_power": "sensor.load",
                "battery_power": "sensor.battery",
                "grid_power": "sensor.grid",
            },
            "water_heaters": [{"id": "wh1", "enabled": True, "sensor": "sensor.water_power"}],
            "system": {
                "timezone": "Europe/Stockholm",
                "grid_meter_type": "net",
                "has_water_heater": True,
            },
        }

        power_reads = [
            ("pv_power", lambda: 1.0),
            ("load_power", lambda: 2.0),
            ("battery_power", lambda: 3.0),
        ]

        water_heater_sensors = []
        if config.get("system", {}).get("has_water_heater", True):
            for water_heater in config.get("water_heaters", []):
                if water_heater.get("enabled", True):
                    sensor = water_heater.get("sensor")
                    if sensor:
                        water_heater_sensors.append(str(sensor))
                        power_reads.append((f"wh_{sensor}", lambda s=str(sensor): 4.0))

        assert len(water_heater_sensors) == 1
        assert "wh_sensor.water_power" in [name for name, _ in power_reads]

    def test_recorder_fetches_ev_charger_when_enabled(self):
        """EV charger sensors are fetched when has_ev_charger is true."""
        config = {
            "input_sensors": {
                "pv_power": "sensor.pv",
                "load_power": "sensor.load",
                "battery_power": "sensor.battery",
                "grid_power": "sensor.grid",
            },
            "ev_chargers": [{"id": "ev1", "enabled": True, "sensor": "sensor.ev_power"}],
            "system": {
                "timezone": "Europe/Stockholm",
                "grid_meter_type": "net",
                "has_ev_charger": True,
            },
        }

        power_reads = [
            ("pv_power", lambda: 1.0),
            ("load_power", lambda: 2.0),
            ("battery_power", lambda: 3.0),
        ]

        ev_charger_sensors = []
        if config.get("system", {}).get("has_ev_charger", False):
            ev_chargers = config.get("ev_chargers", [])
            for ev_charger in ev_chargers:
                if ev_charger.get("enabled", True):
                    sensor = ev_charger.get("sensor")
                    if sensor:
                        ev_charger_sensors.append(str(sensor))
                        power_reads.append((f"ev_{sensor}", lambda s=str(sensor): 4.0))

        assert len(ev_charger_sensors) == 1
        assert "ev_sensor.ev_power" in [name for name, _ in power_reads]


class TestPVEnergyFromPowerHistory:
    """PV energy prefers pv_power history integration, then cumulative, then snapshot.

    On sites where total_pv_production only counts part of the array (e.g. a Sungrow-only
    kWh counter that omits an AC-coupled Fronius), integrating the combined pv_power sensor
    over the slot window is the only path that captures full production.
    """

    @pytest.fixture
    def pv_config(self):
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
                "battery_soc": "sensor.battery_soc",
                "total_pv_production": "sensor.total_pv_production",
                "total_load_consumption": "sensor.total_load_consumption",
            },
            "system": {
                "grid_meter_type": "net",
                "has_battery": True,
                "grid": {"max_power_kw": 20.0},
            },
            "water_heaters": [],
            "ev_chargers": [],
        }

    @pytest.mark.asyncio
    async def test_pv_uses_power_history_when_available(self, pv_config):
        """(a) pv_power history returns a value -> pv_kwh uses it (over cumulative)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            # Seed prev cumulative so the cumulative path WOULD yield 1.25 kWh — the test
            # proves history (3.0) wins over that competing value.
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 3.0,
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                return {
                    "sensor.total_pv_production": 101.25,  # cumulative would give +1.25
                    "sensor.total_load_consumption": 50.75,
                    "sensor.battery_soc": 50.0,
                }.get(entity)

            async def mock_history(entity_id, start, end):
                # Combined PV (Sungrow DC + Fronius AC) integrated over the slot.
                return 3.0 if entity_id == "sensor.pv_power" else None

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=pv_config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # History wins over the cumulative delta (1.25) and snapshot (1.25).
                    assert record["pv_kwh"] == pytest.approx(3.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_pv_history_zero_is_valid_not_fallback(self, pv_config):
        """(a2) history returns 0.0 (clear night: real zero, not missing data) ->
        pv_kwh records 0.0 and does NOT fall back to the cumulative delta. Guards
        against a future refactor to a truthy check (`if pv_history_kwh:`) that would
        silently record a wrong non-zero PV every night."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            # Seed prev cumulative so the fallback WOULD yield 1.25 kWh — the test proves
            # history's real 0.0 wins over that competing non-zero value.
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,  # snapshot would give 1.25 — must NOT be used
                    "sensor.load_power": 3.0,
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                return {
                    "sensor.total_pv_production": 101.25,  # cumulative would give +1.25
                    "sensor.total_load_consumption": 50.75,
                    "sensor.battery_soc": 50.0,
                }.get(entity)

            async def mock_history(entity_id, start, end):
                # Real integrated zero for PV (night); other sensors have no history.
                return 0.0 if entity_id == "sensor.pv_power" else None

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=pv_config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Real 0.0 is valid data — recorded as 0, NOT the cumulative 1.25.
                    assert record["pv_kwh"] == pytest.approx(0.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_pv_falls_back_to_cumulative_when_history_none(self, pv_config):
        """(b) history returns None -> falls back to cumulative delta."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 5.0,  # snapshot would give 1.25 — must NOT be used here
                    "sensor.load_power": 3.0,
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                return 50.0 if entity == "sensor.battery_soc" else None

            async def mock_get_ha_entity_state(entity):
                # Cumulative counter read path (value + unit) — delta = 101.5 - 100 = 1.5 kWh.
                return {
                    "sensor.total_pv_production": {
                        "state": "101.5",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": "50.75",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            async def mock_history(entity_id, start, end):
                return None  # history unavailable for all entities

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_get_ha_entity_state),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=pv_config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    # Cumulative delta (1.5), not the snapshot (1.25).
                    assert record["pv_kwh"] == pytest.approx(1.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_pv_falls_back_to_snapshot_when_neither_available(self, pv_config):
        """(c) neither history nor cumulative -> terminal power snapshot (pv_kw * 0.25)."""
        # No cumulative PV counter configured.
        pv_config["input_sensors"].pop("total_pv_production", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()

            async def mock_get_ha_sensor_kw_normalized(entity):
                return {
                    "sensor.pv_power": 4.0,  # snapshot: 4.0 * 0.25 = 1.0 kWh
                    "sensor.load_power": 2.0,
                    "sensor.grid_power": 1.0,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_get_ha_sensor_float(entity):
                return 50.0 if entity == "sensor.battery_soc" else None

            async def mock_history(entity_id, start, end):
                return None  # history unavailable

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_get_ha_sensor_kw_normalized,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_get_ha_sensor_float),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()

                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=pv_config, state_store=state_store
                    )

                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

                    assert record["pv_kwh"] == pytest.approx(1.0, abs=0.01)


class TestFinishedSlotBounds:
    """The recorder must meter the just-FINISHED quarter, not the just-started one.

    Live 2026-08-25: the window was [21:00, 21:15) computed at 21:00:09 — a 9 s
    sliver of the future extrapolated to 15 min. Cycling loads (the house VVB's
    5-min 3.5 kW bursts) credited 0.000 kWh, the min_kwh_per_day floor never
    cleared, and the solver re-booked an already-hot tank at peak price nightly.
    """

    def test_seconds_after_the_boundary(self):
        import pytz

        from backend.recorder import finished_slot_bounds

        tz = pytz.timezone("Europe/Stockholm")
        now = tz.localize(datetime(2026, 8, 25, 21, 0, 9))
        start, end = finished_slot_bounds(now)
        assert start == tz.localize(datetime(2026, 8, 25, 20, 45))
        assert end == tz.localize(datetime(2026, 8, 25, 21, 0))

    def test_mid_slot_invocation_still_gets_the_previous_complete_slot(self):
        import pytz

        from backend.recorder import finished_slot_bounds

        tz = pytz.timezone("Europe/Stockholm")
        now = tz.localize(datetime(2026, 8, 25, 21, 7, 30))
        start, end = finished_slot_bounds(now)
        assert start == tz.localize(datetime(2026, 8, 25, 20, 45))
        assert end == tz.localize(datetime(2026, 8, 25, 21, 0))

    def test_midnight_crossing(self):
        import pytz

        from backend.recorder import finished_slot_bounds

        tz = pytz.timezone("Europe/Stockholm")
        now = tz.localize(datetime(2026, 8, 26, 0, 0, 12))
        start, end = finished_slot_bounds(now)
        assert start == tz.localize(datetime(2026, 8, 25, 23, 45))
        assert end == tz.localize(datetime(2026, 8, 26, 0, 0))


class TestUnmeasuredWhenConfigIsAbsent:
    """An all-zero row written with no configuration is a fabrication, not a quiet hour.

    2026-09-02/03: run.sh replaced config.yaml with a 27-byte stub after a parse failure,
    and the recorder wrote 44 consecutive all-zero observations across eleven hours,
    logging "Could not validate energy values ... Proceeding with raw values" each time.
    They landed with quality_flags "{}" — indistinguishable from a real quarter in which
    nothing happened, and from the price-mint rows quarantine.py exists to label.

    The row is still stored, because the PRICE on it is real and feeds price_outlook and
    the ML lags. It is stored TAGGED.
    """

    @pytest.fixture
    def base_config(self):
        # Deliberately missing system.grid.max_power_kw — the shape a failed load has.
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {"pv_power": "sensor.pv", "load_power": "sensor.load"},
            "system": {"grid_meter_type": "net"},
            "water_heaters": [],
            "ev_chargers": [],
        }

    @pytest.mark.asyncio
    async def test_an_unconfigured_observation_is_tagged(self, base_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = RecorderStateStore(Path(tmpdir) / "s.json")
            state_store.load()

            async def zero(_entity):
                return 0.0

            async def none_f(_entity):
                return None

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=zero),
                patch("backend.recorder.get_ha_sensor_float", side_effect=none_f),
                patch("backend.recorder.get_ha_entity_state", side_effect=none_f),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock, return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.logger"),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()
                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=base_config, state_store=state_store
                    )
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

        flags = json.loads(str(record["quality_flags"]))
        assert flags.get("no_config") is True, (
            "an observation taken without configuration must say so, or it is "
            "indistinguishable from a genuinely idle quarter"
        )

    @pytest.mark.asyncio
    async def test_a_configured_observation_is_not_tagged(self, base_config):
        """The healthy path must stay untouched — no tag on a normal quarter."""
        cfg = base_config.copy()
        cfg["system"] = {"grid_meter_type": "net", "grid": {"max_power_kw": 25.0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = RecorderStateStore(Path(tmpdir) / "s.json")
            state_store.load()

            async def zero(_entity):
                return 0.0

            async def none_f(_entity):
                return None

            with (
                patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=zero),
                patch("backend.recorder.get_ha_sensor_float", side_effect=none_f),
                patch("backend.recorder.get_ha_entity_state", side_effect=none_f),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock, return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.logger"),
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()
                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=cfg, state_store=state_store
                    )
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()

        raw = record.get("quality_flags")
        flags = json.loads(str(raw)) if raw else {}
        assert "no_config" not in flags


class TestEnergyBalanceTripwire:
    """The recorder flags quarters where energy conservation does not close.

    Identity: (import + pv + discharge) - (export + charge + base_load + water + EV
    + cyclic). The tripwire is computed BEFORE the balance-rescue, because the rescue
    assigns load from this same identity and would drive the residual to exactly zero.
    """

    @pytest.fixture
    def base_config(self):
        return {
            "timezone": "Europe/Stockholm",
            "learning": {"sqlite_path": ":memory:"},
            "input_sensors": {
                "pv_power": "sensor.pv_power",
                "load_power": "sensor.load_power",
                "grid_power": "sensor.grid_power",
                "battery_power": "sensor.battery_power",
                "battery_soc": "sensor.battery_soc",
                "total_pv_production": "sensor.total_pv_production",
                "total_load_consumption": "sensor.total_load_consumption",
            },
            "system": {
                "grid_meter_type": "net",
                "has_battery": True,
                "has_water_heater": False,
                "has_ev_charger": False,
            },
            "water_heaters": [],
            "ev_chargers": [],
        }

    async def _run(self, config, *, pv_delta, load_delta, grid_kw):
        """Drive one observation and return (record, warning_calls).

        PV and load energy come from the cumulative counters (the recorder prefers
        them); grid comes from the power snapshot over the quarter.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "recorder_state.json"
            state_store = RecorderStateStore(state_file)
            state_store.load()
            now = datetime.now(pytz.timezone("Europe/Stockholm"))
            prev_time = now - timedelta(minutes=15)
            state_store._state = {
                "pv_total": {"value": 100.0, "timestamp": prev_time.isoformat()},
                "load_total": {"value": 50.0, "timestamp": prev_time.isoformat()},
            }
            state_store.save()

            async def mock_kw(entity):
                return {
                    "sensor.pv_power": 5.0,
                    "sensor.load_power": 7.0,
                    "sensor.grid_power": grid_kw,
                    "sensor.battery_power": 0.0,
                }.get(entity, 0.0)

            async def mock_float(entity):
                return 50.0 if entity == "sensor.battery_soc" else None

            async def mock_entity_state(entity):
                return {
                    "sensor.total_pv_production": {
                        "state": f"{100.0 + pv_delta}",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                    "sensor.total_load_consumption": {
                        "state": f"{50.0 + load_delta}",
                        "attributes": {"unit_of_measurement": "kWh"},
                        "last_updated": now.isoformat(),
                    },
                }.get(entity)

            with (
                patch(
                    "backend.recorder.get_ha_sensor_kw_normalized",
                    side_effect=mock_kw,
                ),
                patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
                patch("backend.recorder.get_ha_entity_state", side_effect=mock_entity_state),
                patch(
                    "backend.recorder.get_energy_from_power_history",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("backend.recorder.get_nordpool_data", return_value=None),
                patch("backend.recorder.logger") as mock_logger,
            ):
                mock_store = MagicMock()
                mock_store.get_system_state = AsyncMock(return_value=None)
                mock_store.set_system_state = AsyncMock()
                mock_store.store_slot_observations = AsyncMock()
                mock_store.close = AsyncMock()
                with patch("backend.recorder.LearningStore", return_value=mock_store):
                    await record_observation_from_current_state(
                        config=config, state_store=state_store
                    )
                    df = mock_store.store_slot_observations.call_args[0][0]
                    record = df.iloc[0].to_dict()
                    warnings = [str(c) for c in mock_logger.warning.call_args_list]
                    return record, warnings

    @staticmethod
    def _flags(record):
        raw = record.get("quality_flags")
        return json.loads(raw) if raw else {}

    @pytest.mark.asyncio
    async def test_balanced_slot_does_not_flag(self, base_config):
        """import 1.75 + pv 1.25 - load 3.0 == 0: nothing to report."""
        record, _ = await self._run(
            base_config.copy(), pv_delta=1.25, load_delta=3.0, grid_kw=7.0
        )
        assert "balance_residual_kwh" not in self._flags(record)

    @pytest.mark.asyncio
    async def test_positive_imbalance_flags_with_the_residual_value(self, base_config):
        """More energy arrived than left: 1.75 + 1.25 - 1.0 = +2.0 kWh unaccounted."""
        record, warnings = await self._run(
            base_config.copy(), pv_delta=1.25, load_delta=1.0, grid_kw=7.0
        )
        flags = self._flags(record)
        assert flags["balance_residual_kwh"] == pytest.approx(2.0, abs=0.01)
        assert any("energy balance did not close" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_negative_imbalance_flags_too(self, base_config):
        """The clamp trap: balance_base_kwh is max(0.0, ...), so a residual taken from
        it could only ever show one sign. The two largest real imbalances on this site
        were both negative."""
        record, _ = await self._run(
            base_config.copy(), pv_delta=1.25, load_delta=5.0, grid_kw=7.0
        )
        flags = self._flags(record)
        assert flags["balance_residual_kwh"] == pytest.approx(-2.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_rescued_slot_abstains_instead_of_reporting_balanced(self, base_config):
        """The vacuous-flag trap.

        A frozen load counter triggers the balance-rescue, which sets load_kwh from the
        very identity the tripwire tests — making the residual exactly zero. Such a slot
        must NOT be recorded as balanced; it is unevaluable, and it says so by carrying
        load_rescued and no residual. On live data this is 28.6% of all quarters and 90%
        of quarters above 2.5 kWh PV.
        """
        record, _ = await self._run(
            base_config.copy(), pv_delta=1.25, load_delta=0.0, grid_kw=7.0
        )
        flags = self._flags(record)
        assert "load_rescued" in flags
        assert "balance_residual_kwh" not in flags

    @pytest.mark.asyncio
    async def test_tolerance_is_configurable(self, base_config):
        """A +2.0 kWh residual passes a 3.0 tolerance and trips a 0.5 one."""
        loose = base_config.copy()
        loose["recorder"] = {"balance_tolerance_kwh": 3.0}
        record, _ = await self._run(loose, pv_delta=1.25, load_delta=1.0, grid_kw=7.0)
        assert "balance_residual_kwh" not in self._flags(record)

        tight = base_config.copy()
        tight["recorder"] = {"balance_tolerance_kwh": 0.5}
        record, _ = await self._run(tight, pv_delta=1.25, load_delta=1.0, grid_kw=7.0)
        assert "balance_residual_kwh" in self._flags(record)
