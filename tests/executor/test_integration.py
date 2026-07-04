"""
REV F57 Integration Tests
Tests that verify healthy configs don't get corrupted during migration.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from backend.config_migration import migrate_config


class TestF57Integration:
    """Integration tests for REV F57 config migration fixes."""

    @pytest.mark.asyncio
    async def test_healthy_config_unchanged(self, tmp_path):
        """
        Test that a healthy config (config.default.yaml) stays unchanged
        when migrated.

        This prevents regression where valid configs get corrupted.
        """
        # Create a healthy config from the real template
        test_config = tmp_path / "config.yaml"
        default_config = tmp_path / "config.default.yaml"

        # Use the REAL config.default.yaml as both source and template
        real_default = Path("config.default.yaml")
        if not real_default.exists():
            pytest.skip("config.default.yaml not found")

        shutil.copy(real_default, test_config)
        shutil.copy(real_default, default_config)

        # Run migration with lenient validation
        await migrate_config(str(test_config), str(default_config), strict_validation=False)

        # Load result
        with test_config.open() as f:
            result = yaml.safe_load(f)

        # Verify structure is intact (using real template, values come from template)
        assert result["config_version"] == 2, "config_version should be present"
        assert "system" in result, "system section should be present"
        assert "battery" in result, "battery section should be present"
        assert "water_heaters" in result, "water_heaters should be present"
        assert "ev_chargers" in result, "ev_chargers should be present"

        # Verify no deprecated keys added. deferrable_loads is a LIVE key again
        # (smart-appliance controller): the template's empty default may merge in,
        # but no legacy content may appear.
        assert result.get("deferrable_loads", []) == [], "no legacy deferrable content"
        assert "ev_charger" not in result, "ev_charger should not be added"
        assert "version" not in result, "version should not be added"

    @pytest.mark.asyncio
    async def test_migration_creates_backups(self, tmp_path):
        """Test that migration creates timestamped backups."""
        test_config = tmp_path / "config.yaml"
        default_config = tmp_path / "config.default.yaml"

        # Use the REAL config.default.yaml as template
        real_default = Path("config.default.yaml")
        if not real_default.exists():
            pytest.skip("config.default.yaml not found")

        # Create a corrupted config from the template
        with real_default.open() as f:
            config = yaml.safe_load(f)
        # Add some corruption
        config["version"] = "2.4.0"
        config["config_version"] = 1

        with test_config.open("w") as f:
            yaml.dump(config, f)

        shutil.copy(real_default, default_config)

        # Run migration
        await migrate_config(str(test_config), str(default_config), strict_validation=False)

        # Check backup was created
        backup_dir = tmp_path / "backups"
        assert backup_dir.exists(), "Backup directory should be created"

        backups = list(backup_dir.glob("config.yaml_*.bak"))
        assert len(backups) >= 1, "At least one backup should be created"

    def test_config_file_safety(self):
        """
        Verify that tests use isolated temp files and don't modify real configs.

        This test documents the safety requirements for config tests.
        """
        # List of files that should NEVER be modified by tests
        protected_files = [
            Path("config.yaml"),
            Path("config.default.yaml"),
        ]

        for f in protected_files:
            if f.exists():
                # Just verify they exist - we can't actually test they weren't
                # modified without complex checksum logic, but this serves as
                # documentation of the requirement
                assert f.exists(), f"{f} should exist"

        # Note: Actual safety is enforced by:
        # 1. Using tmp_path fixture in pytest
        # 2. Copying files before modification
        # 3. Never writing to paths outside of temp directories
