import logging
from pathlib import Path

import pytest

# Setup logging
logging.basicConfig(level=logging.INFO)


class TestDeprecatedKeyRemoval:
    """Test that deprecated keys are actually deleted."""

    def test_remove_deprecated_keys_root_level(self):
        from backend.config_migration import remove_deprecated_keys

        config = {
            "config_version": 2,
            "deferrable_loads": [{"id": "washer", "power_sensor": "sensor.w"}],
            "ev_charger": {},
            "solar_array": {},
        }
        result, changed = remove_deprecated_keys(config)
        assert changed
        # deferrable_loads is a LIVE key again (smart-appliance controller) and must
        # survive the deprecated-key sweep — stripping it made the feature
        # unconfigurable (legacy-shaped ENTRIES are pruned separately).
        assert result["deferrable_loads"] == [{"id": "washer", "power_sensor": "sensor.w"}]
        assert "ev_charger" not in result
        assert "solar_array" not in result

    def test_prune_legacy_deferrable_entries(self):
        from backend.config_migration import _prune_legacy_deferrable_entries

        # Mixed: legacy entries (no power_sensor) pruned, new-schema kept.
        config = {
            "deferrable_loads": [
                {"id": "my_tesla", "power": 11.0},
                {"id": "washer", "power_sensor": "sensor.w", "switch_entity": "switch.w"},
            ]
        }
        result, changed = _prune_legacy_deferrable_entries(config)
        assert changed
        assert [e["id"] for e in result["deferrable_loads"]] == ["washer"]

        # Pure legacy: key dropped entirely (old cleanup behaviour preserved).
        config2 = {"deferrable_loads": [{"id": "my_pool", "power": 3.0}]}
        result2, changed2 = _prune_legacy_deferrable_entries(config2)
        assert changed2 and "deferrable_loads" not in result2

        # Pure new-schema: untouched, no change flagged.
        config3 = {"deferrable_loads": [{"id": "dw", "power_sensor": "sensor.d"}]}
        result3, changed3 = _prune_legacy_deferrable_entries(config3)
        assert not changed3
        assert result3["deferrable_loads"] == [{"id": "dw", "power_sensor": "sensor.d"}]

    def test_remove_deprecated_keys_nested(self):
        from backend.config_migration import remove_deprecated_keys

        config = {
            "executor": {
                "inverter": {
                    "work_mode": "select.ems_mode",
                    "work_mode_entity": "select.ems_mode",  # Deprecated
                    "soc_target": "",
                    "soc_target_entity": "",  # Deprecated
                }
            }
        }
        result, changed = remove_deprecated_keys(config)
        assert changed
        assert "work_mode" in result["executor"]["inverter"]
        assert "work_mode_entity" not in result["executor"]["inverter"]
        assert "soc_target_entity" not in result["executor"]["inverter"]


class TestInverterKeyMigration:
    """Test migration of system.inverter.max_power_kw to max_ac_power_kw."""

    def test_migrate_inverter_max_power_kw_to_max_ac_power_kw(self):
        """Old key should be migrated to new key when new key doesn't exist."""
        from backend.config_migration import _migrate_inverter_keys

        config = {
            "system": {
                "inverter": {
                    "max_power_kw": 8.8,  # Old key
                }
            }
        }

        result, changed = _migrate_inverter_keys(config)

        assert changed is True
        assert result["system"]["inverter"]["max_ac_power_kw"] == 8.8

    def test_migrate_inverter_does_not_overwrite_existing(self):
        """Should NOT overwrite existing max_ac_power_kw when both keys exist."""
        from backend.config_migration import _migrate_inverter_keys

        config = {
            "system": {
                "inverter": {
                    "max_power_kw": 8.8,  # Old key
                    "max_ac_power_kw": 10.0,  # New key already exists
                }
            }
        }

        result, changed = _migrate_inverter_keys(config)

        assert changed is False
        assert result["system"]["inverter"]["max_ac_power_kw"] == 10.0

    def test_migrate_inverter_no_change_when_no_old_key(self):
        """Should not change anything when old key doesn't exist."""
        from backend.config_migration import _migrate_inverter_keys

        config = {"system": {"inverter": {}}}

        result, changed = _migrate_inverter_keys(config)

        assert changed is False
        assert "max_ac_power_kw" not in result["system"]["inverter"]

    def test_migrate_inverter_creates_inverter_section_if_missing(self):
        """Should handle missing inverter section gracefully."""
        from backend.config_migration import _migrate_inverter_keys

        config = {"system": {}}

        _result, changed = _migrate_inverter_keys(config)

        assert changed is False


class TestBackendSave:
    """Test backend save logic with template merge."""

    @pytest.mark.asyncio
    async def test_backend_save_removes_deprecated_keys(self, tmp_path, monkeypatch):
        from ruamel.yaml import YAML

        from backend.api.routers.config import save_config

        yaml_loader = YAML()

        config_file = tmp_path / "config.yaml"
        default_file = tmp_path / "config.default.yaml"

        user_config = {
            "config_version": 2,
            "deferrable_loads": [{"id": "old_load"}],
            "timezone": "Europe/London",
            "system": {
                "system_id": "test",
                "inverter_profile": "test",
                "has_solar": False,
                "has_battery": False,
            },
            "battery": {},
            "executor": {},
            "input_sensors": {},
        }

        default_config = {
            "config_version": 2,
            "timezone": "Europe/London",
            "system": {
                "system_id": "test",
                "inverter_profile": "test",
                "has_solar": False,
                "has_battery": False,
            },
            "battery": {},
            "executor": {},
            "input_sensors": {},
        }

        with config_file.open("w") as f:
            yaml_loader.dump(user_config, f)
        with default_file.open("w") as f:
            yaml_loader.dump(default_config, f)

        import backend.api.routers.config as config_router

        monkeypatch.setattr(
            config_router,
            "Path",
            lambda p: tmp_path / p if p in ["config.yaml", "config.default.yaml"] else Path(p),
        )
        monkeypatch.setattr(config_router, "get_executor_instance", lambda: None)
        monkeypatch.setattr(config_router, "_validate_config_for_save", lambda x: [])

        await save_config({"timezone": "Europe/Stockholm"})

        with config_file.open() as f:
            saved_data = yaml_loader.load(f)
        assert saved_data["timezone"] == "Europe/Stockholm"
        # {"id": "old_load"} is legacy-shaped (no power_sensor) => pruned on save; the
        # template's empty default may merge in. No legacy content may survive.
        assert list(saved_data.get("deferrable_loads") or []) == []


class TestBackupSystem:
    """Test timestamped backup system."""

    def test_create_timestamped_backup(self, tmp_path):
        from backend.config_migration import create_timestamped_backup

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test content")

        backup_path = create_timestamped_backup(config_file)
        assert backup_path is not None
        assert backup_path.exists()
        assert "config.yaml_" in backup_path.name


class TestTemplateAwareMerge:
    """Test template_aware_merge functionality."""

    def test_merge_preserves_comments(self):
        from ruamel.yaml import YAML

        from backend.config_migration import template_aware_merge

        yaml_loader = YAML()
        template_str = """
# Global Comment
system:
  # Section Comment
  id: "default_id"
"""
        user_cfg = {"system": {"id": "user_id"}}
        template_cfg = yaml_loader.load(template_str)
        template_aware_merge(template_cfg, user_cfg)
        assert template_cfg["system"]["id"] == "user_id"


class TestFullMigrationFlow:
    """Test the complete migration pipeline."""

    @pytest.mark.asyncio
    async def test_migrate_config_idempotent_for_clean_config(self, tmp_path, monkeypatch):
        """Verify that a clean config_version=2 config with no deprecated keys is not rewritten."""
        from ruamel.yaml import YAML

        import backend.config_migration as cm

        yaml_loader = YAML()
        config_file = tmp_path / "config.yaml"
        default_file = tmp_path / "config.default.yaml"

        clean_config = {
            "config_version": 2,
            "system": {
                "system_id": "test",
                "inverter_profile": "test",
                "has_solar": False,
                "has_battery": True,
            },
            "battery": {"min_soc_percent": 20},
            "executor": {},
            "input_sensors": {},
        }

        with config_file.open("w") as f:
            yaml_loader.dump(clean_config, f)
        with default_file.open("w") as f:
            yaml_loader.dump(clean_config, f)

        monkeypatch.setattr(
            cm,
            "Path",
            lambda p: tmp_path / p if p in ["config.yaml", "config.default.yaml"] else Path(p),
        )

        write_calls: list = []

        def mock_write(*args, **kwargs):
            write_calls.append(args)

        monkeypatch.setattr(cm, "_write_config", mock_write)

        await cm.migrate_config(strict_validation=False)

        assert len(write_calls) == 0, (
            f"_write_config was called {len(write_calls)} time(s) but should not have been "
            f"for a clean config with no deprecated keys"
        )


class TestWaterHeaterMigration:
    """Test water heater field migration from legacy locations to water_heaters[] array."""

    def test_migrate_all_water_heater_fields(self):
        """Legacy config with all three keys → values migrate into water_heaters[0], old keys removed."""
        from backend.config_migration import _migrate_water_heater_fields

        config = {
            "config_version": 2,
            "input_sensors": {
                "water_power": "sensor.boiler_power",
                "water_heater_consumption": "sensor.boiler_energy",
                "grid_power": "sensor.grid",
            },
            "executor": {
                "water_heater": {
                    "target_entity": "climate.boiler",
                    "temp_off": 40,
                    "temp_normal": 60,
                }
            },
            "water_heaters": [
                {
                    "id": "main_tank",
                    "sensor": "",
                    "energy_sensor": "",
                    "target_entity": "",
                    "power_kw": 3.0,
                }
            ],
        }

        result, changed = _migrate_water_heater_fields(config)

        assert changed is True
        # Values should be migrated
        assert result["water_heaters"][0]["sensor"] == "sensor.boiler_power"
        assert result["water_heaters"][0]["target_entity"] == "climate.boiler"
        # Old keys should be removed
        assert "water_power" not in result["input_sensors"]
        assert "water_heater_consumption" not in result["input_sensors"]
        assert "target_entity" not in result["executor"]["water_heater"]
        # Other keys should remain
        assert result["input_sensors"]["grid_power"] == "sensor.grid"
        assert result["executor"]["water_heater"]["temp_off"] == 40

    def test_migrate_preserves_existing_sensor(self):
        """Config with sensor already set → input_sensors.water_power is NOT copied, old key still removed."""
        from backend.config_migration import _migrate_water_heater_fields

        config = {
            "config_version": 2,
            "input_sensors": {
                "water_power": "sensor.boiler_power",
                "water_heater_consumption": "",
            },
            "executor": {
                "water_heater": {
                    "target_entity": "",
                }
            },
            "water_heaters": [
                {
                    "id": "main_tank",
                    "sensor": "sensor.existing_power",  # Already set
                    "energy_sensor": "",
                    "target_entity": "",
                }
            ],
        }

        result, changed = _migrate_water_heater_fields(config)

        assert changed is True  # Still changed because old keys are removed
        # Existing value should be preserved
        assert result["water_heaters"][0]["sensor"] == "sensor.existing_power"
        # Old key should still be removed
        assert "water_power" not in result["input_sensors"]

    def test_migrate_no_water_heaters(self):
        """Config with no water_heaters array should not crash."""
        from backend.config_migration import _migrate_water_heater_fields

        config = {
            "config_version": 2,
            "input_sensors": {
                "water_power": "sensor.boiler_power",
            },
        }

        result, changed = _migrate_water_heater_fields(config)

        # Should not crash, no changes made
        assert changed is False
        assert "input_sensors" in result

    def test_migrate_empty_water_heaters(self):
        """Config with empty water_heaters array should not crash."""
        from backend.config_migration import _migrate_water_heater_fields

        config = {
            "config_version": 2,
            "input_sensors": {
                "water_power": "sensor.boiler_power",
            },
            "water_heaters": [],
        }

        result, changed = _migrate_water_heater_fields(config)

        # Should not crash, no changes made to array
        assert changed is False
        assert result["water_heaters"] == []


class TestRemoveEnergySensorFields:
    """Tests for _remove_energy_sensor_fields migration step."""

    def test_removes_energy_sensor_from_ev_chargers(self):
        """energy_sensor is removed from all ev_chargers[] items."""
        from backend.config_migration import _remove_energy_sensor_fields

        config = {
            "ev_chargers": [
                {"id": "ev1", "sensor": "sensor.ev_power", "energy_sensor": "sensor.ev_energy"},
                {"id": "ev2", "sensor": "sensor.ev2_power", "energy_sensor": "sensor.ev2_energy"},
            ],
        }

        result, changed = _remove_energy_sensor_fields(config)

        assert changed is True
        assert "energy_sensor" not in result["ev_chargers"][0]
        assert "energy_sensor" not in result["ev_chargers"][1]
        assert result["ev_chargers"][0]["sensor"] == "sensor.ev_power"

    def test_removes_energy_sensor_from_water_heaters(self):
        """energy_sensor is removed from all water_heaters[] items."""
        from backend.config_migration import _remove_energy_sensor_fields

        config = {
            "water_heaters": [
                {"id": "wh1", "sensor": "sensor.wh_power", "energy_sensor": "sensor.wh_energy"},
            ],
        }

        result, changed = _remove_energy_sensor_fields(config)

        assert changed is True
        assert "energy_sensor" not in result["water_heaters"][0]
        assert result["water_heaters"][0]["sensor"] == "sensor.wh_power"

    def test_other_fields_untouched(self):
        """Other fields on the item are preserved."""
        from backend.config_migration import _remove_energy_sensor_fields

        config = {
            "ev_chargers": [
                {
                    "id": "ev1",
                    "name": "My EV",
                    "enabled": True,
                    "sensor": "sensor.ev",
                    "energy_sensor": "sensor.ev_energy",
                    "soc_sensor": "sensor.ev_soc",
                },
            ],
        }

        result, changed = _remove_energy_sensor_fields(config)

        assert changed is True
        item = result["ev_chargers"][0]
        assert item["id"] == "ev1"
        assert item["name"] == "My EV"
        assert item["enabled"] is True
        assert item["sensor"] == "sensor.ev"
        assert item["soc_sensor"] == "sensor.ev_soc"
        assert "energy_sensor" not in item

    def test_idempotent_no_error_if_field_absent(self):
        """No error if energy_sensor already absent; changed=False."""
        from backend.config_migration import _remove_energy_sensor_fields

        config = {
            "ev_chargers": [{"id": "ev1", "sensor": "sensor.ev"}],
            "water_heaters": [{"id": "wh1", "sensor": "sensor.wh"}],
        }

        result, changed = _remove_energy_sensor_fields(config)

        assert changed is False
        assert result["ev_chargers"][0] == {"id": "ev1", "sensor": "sensor.ev"}

    def test_no_arrays_no_error(self):
        """Config without ev_chargers or water_heaters doesn't crash."""
        from backend.config_migration import _remove_energy_sensor_fields

        config = {"system": {"has_battery": True}}

        _, changed = _remove_energy_sensor_fields(config)

        assert changed is False


class TestMigrateEvChargerFields:
    """Tests for _migrate_ev_charger_fields: moves global EV settings into ev_chargers[0]."""

    def _base_config(self, extra_charger_fields=None):
        charger = {"id": "main", "enabled": True, "name": "My EV"}
        if extra_charger_fields:
            charger.update(extra_charger_fields)
        return {
            "ev_chargers": [charger],
        }

    def test_departure_time_migrated_to_first_enabled_charger(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config()
        config["ev_departure_time"] = "07:30"

        result, changed = _migrate_ev_charger_fields(config)

        assert changed is True
        assert result["ev_chargers"][0]["departure_time"] == "07:30"

    def test_switch_entity_migrated_from_executor_ev_charger(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config()
        config["executor"] = {"ev_charger": {"switch_entity": "switch.tesla"}}

        result, changed = _migrate_ev_charger_fields(config)

        assert changed is True
        assert result["ev_chargers"][0]["switch_entity"] == "switch.tesla"

    def test_replan_on_plugin_migrated(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config()
        config["executor"] = {"ev_charger": {"replan_on_plugin": True}}

        result, changed = _migrate_ev_charger_fields(config)

        assert changed is True
        assert result["ev_chargers"][0]["replan_on_plugin"] is True

    def test_replan_on_unplug_migrated(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config()
        config["executor"] = {"ev_charger": {"replan_on_unplug": True}}

        result, changed = _migrate_ev_charger_fields(config)

        assert changed is True
        assert result["ev_chargers"][0]["replan_on_unplug"] is True

    def test_idempotent_departure_time_not_overwritten(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config(extra_charger_fields={"departure_time": "08:00"})
        config["ev_departure_time"] = "07:30"

        result, _ = _migrate_ev_charger_fields(config)

        # departure_time already present — should NOT be overwritten
        assert result["ev_chargers"][0]["departure_time"] == "08:00"

    def test_idempotent_switch_entity_not_overwritten(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config(extra_charger_fields={"switch_entity": "switch.existing"})
        config["executor"] = {"ev_charger": {"switch_entity": "switch.old"}}

        result, _ = _migrate_ev_charger_fields(config)

        assert result["ev_chargers"][0]["switch_entity"] == "switch.existing"

    def test_no_op_when_already_migrated(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = self._base_config(
            extra_charger_fields={
                "departure_time": "07:00",
                "switch_entity": "switch.ev",
                "replan_on_plugin": True,
                "replan_on_unplug": False,
            }
        )
        # No old-style fields present either
        _, changed = _migrate_ev_charger_fields(config)

        assert changed is False

    def test_no_enabled_charger_returns_unchanged(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = {
            "ev_chargers": [{"id": "main", "enabled": False, "name": "My EV"}],
            "ev_departure_time": "07:00",
            "executor": {"ev_charger": {"switch_entity": "switch.ev"}},
        }

        result, changed = _migrate_ev_charger_fields(config)

        assert changed is False
        # The disabled charger should not have departure_time added
        assert "departure_time" not in result["ev_chargers"][0]

    def test_empty_ev_chargers_returns_unchanged(self):
        from backend.config_migration import _migrate_ev_charger_fields

        config = {
            "ev_chargers": [],
            "ev_departure_time": "07:00",
        }

        _, changed = _migrate_ev_charger_fields(config)

        assert changed is False
