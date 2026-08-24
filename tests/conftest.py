from pathlib import Path
from unittest.mock import patch

import pytest

from backend.core import secrets


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up a clean test environment with mocked config for CI."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    db_path = data_dir / "test_planner.db"

    test_config = {
        "version": "2.5.1-beta",
        "timezone": "Europe/Stockholm",
        "learning": {
            "enable": True,
            "sqlite_path": "data/test_planner.db",
            "horizon_days": 2,
        },
        "forecasting": {
            "active_forecast_version": "aurora",
        },
        "input_sensors": {
            "battery_soc": "sensor.test_soc",
        },
    }

    original_load_yaml = secrets.load_yaml

    def mock_load_yaml(path: str) -> dict:
        if path == "config.yaml":
            return test_config
        return original_load_yaml(path)

    with patch.object(secrets, "load_yaml", mock_load_yaml):
        yield

    try:
        if db_path.exists():
            db_path.unlink()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_nordpool_fail_memo():
    """The negative-fetch memo is a module global; a test that exercises a failed
    fetch must not silence get_nordpool_data for every later test."""
    from backend.core import prices

    prices._fetch_fail_until = 0.0
    yield
    prices._fetch_fail_until = 0.0
