"""A non-numeric power state must not abort the state gather.

An unguarded float() on pv/load/import/export raised inside the gather's try and
skipped every read below it — manual_override included. Now each entity parses
on its own, with a rate-limited warning (upstream 42716df2 idea).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from executor.actions import HAClient
from executor.config import ExecutorConfig, InverterConfig
from executor.engine import ExecutorEngine

if TYPE_CHECKING:
    from pathlib import Path

ENTITIES = {
    "sensor.soc": "55",
    "sensor.pv": "garbage",
    "sensor.load": "1500",
    "sensor.imp": "nan",
    "sensor.exp": "250",
    "input_boolean.manual": "on",
}


def _engine(tmp_path: Path) -> ExecutorEngine:
    config = ExecutorConfig(
        enabled=True,
        schedule_path=str(tmp_path / "schedule.json"),
        timezone="Europe/Stockholm",
        inverter=InverterConfig(),
        has_water_heater=False,
        manual_override_entity="input_boolean.manual",
    )
    full = {
        "input_sensors": {
            "battery_soc": "sensor.soc",
            "pv_power": "sensor.pv",
            "load_power": "sensor.load",
            "grid_import_power": "sensor.imp",
            "grid_export_power": "sensor.exp",
        },
        "system": {"grid_meter_type": "dual"},
        "battery": {"min_soc_percent": 10},
    }
    with (
        patch("executor.engine.load_executor_config", return_value=config),
        patch("executor.engine.load_yaml", return_value=full),
        patch.object(ExecutorEngine, "_get_db_path", return_value=str(tmp_path / "t.db")),
    ):
        eng = ExecutorEngine("config.yaml")
    ha = MagicMock(spec=HAClient)
    ha.get_state_value.side_effect = lambda e: ENTITIES.get(e)
    eng.ha_client = ha
    return eng


@pytest.mark.asyncio
async def test_bad_power_state_does_not_skip_the_rest(tmp_path) -> None:
    eng = _engine(tmp_path)
    state = await eng._gather_system_state()
    assert state.manual_override_active is True  # read AFTER the bad pv value
    assert state.current_soc_percent == 55.0
    assert state.current_load_kw == pytest.approx(1.5)
    assert state.current_export_kw == pytest.approx(0.25)
    assert state.current_pv_kw == 0.0  # bad value keeps the default
    assert state.current_import_kw == 0.0  # NaN is not a reading either


@pytest.mark.asyncio
async def test_bad_sensor_warning_is_rate_limited(tmp_path, caplog) -> None:
    eng = _engine(tmp_path)
    with caplog.at_level(logging.WARNING):
        await eng._gather_system_state()
        await eng._gather_system_state()
    pv_warnings = [r for r in caplog.records if "sensor.pv" in r.getMessage()]
    assert len(pv_warnings) == 1


def test_sensor_kw_parses_and_rejects() -> None:
    eng = object.__new__(ExecutorEngine)
    assert eng._sensor_kw("2000", "sensor.x") == pytest.approx(2.0)
    assert eng._sensor_kw("0", "sensor.x") == 0.0
    for bad in (None, "", "unknown", "unavailable", "on", "inf", object()):
        assert eng._sensor_kw(bad, "sensor.x") is None
