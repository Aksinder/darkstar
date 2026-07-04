import asyncio
import shutil
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from backend.config_migration import migrate_config


class TestRegressionComplex(unittest.TestCase):
    def setUp(self):
        self.yaml = YAML()
        self.test_dir = Path("tests/temp_regression")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        self.user_path = self.test_dir / "config.yaml"
        self.default_path = self.test_dir / "config.default.yaml"
        self.maxDiff = None

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    async def run_migration(self):
        await migrate_config(str(self.user_path), str(self.default_path), strict_validation=False)

    def test_complex_migration_and_structure(self):
        """
        Scenario:
        1. User has an OLD config (v1 style) with battery keys in executor.controller.
        2. Default has v2 style with battery section.
        3. Default has comments and specific order.
        4. User has custom list of deferrable loads.
        """

        default_content = """# Darkstar Default
version: "2.0.0"

# Core System
system:
  id: prod
  # Inverter settings
  inverter:
    power: 5.0
    brand: generic

# Battery Section (New)
battery:
  capacity_kwh: 5.0
  min_voltage_v: 46.0

# Load Management
deferrable_loads:
  - id: default_load
    power: 1.0

# Future Section
future_stuff:
  enabled: false
"""

        user_content = """# User config with legacy keys
version: "1.9.0"

executor:
  controller:
    battery_capacity_kwh: 12.5  # Legacy key!
    worst_case_voltage_v: 44.0  # Legacy key!

# User has custom loads
deferrable_loads:
  - id: my_tesla
    power: 11.0
  - id: my_pool
    power: 3.0

system:
  inverter:
    brand: deye
    power: 8.0
  id: my_home

custom_top_level: "special"
"""

        with self.default_path.open("w") as f:
            f.write(default_content)
        with self.user_path.open("w") as f:
            f.write(user_content)

        # Run migration
        asyncio.run(self.run_migration())

        # Load result
        with self.user_path.open() as f:
            result_str = f.read()
            f.seek(0)
            result_cfg = self.yaml.load(f)

        # VERIFICATIONS

        # 1. Check Key Presence
        # version is removed by deprecated key sweep. deferrable_loads is a LIVE key
        # again (smart appliances): the user's LEGACY entries (my_tesla/my_pool, no
        # power_sensor) must be pruned; the template's empty default may merge in.
        self.assertNotIn("version", result_cfg)
        self.assertIn("system", result_cfg)
        self.assertIn("battery", result_cfg)
        self.assertIn("future_stuff", result_cfg)
        self.assertIn("custom_top_level", result_cfg)

        # 2. Check legacy deferrable content removed (empty/absent both fine)
        self.assertEqual(list(result_cfg.get("deferrable_loads") or []), [])

        # 4. Check Comment Preservation from Default (lenient mode may not preserve all comments)
        self.assertIn("# Darkstar Default", result_str)
        self.assertIn("# Battery Section (New)", result_str)

        # 5. Check Values
        self.assertEqual(result_cfg["system"]["inverter"]["brand"], "deye")
        self.assertEqual(result_cfg["system"]["inverter"]["power"], 8.0)
        self.assertEqual(result_cfg["custom_top_level"], "special")

        # 6. Verify Backup
        backup_path = self.user_path.with_suffix(".yaml.bak")
        self.assertTrue(backup_path.exists())

    def test_missing_nested_keys(self):
        """
        Verify that missing nested keys in user config are filled from default.
        """
        default_content = """
system:
  location:
    lat: 50.0
    lon: 10.0
  mode: "auto"
"""
        user_content = """
system:
  location:
    lat: 55.5
  # lon is missing!
"""
        with self.default_path.open("w") as f:
            f.write(default_content)
        with self.user_path.open("w") as f:
            f.write(user_content)

        asyncio.run(self.run_migration())

        with self.user_path.open() as f:
            result_cfg = self.yaml.load(f)

        self.assertEqual(result_cfg["system"]["location"]["lat"], 55.5)
        self.assertEqual(result_cfg["system"]["location"]["lon"], 10.0)  # From default
        self.assertEqual(result_cfg["system"]["mode"], "auto")  # From default


if __name__ == "__main__":
    unittest.main()
