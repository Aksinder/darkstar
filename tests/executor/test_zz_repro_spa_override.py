"""Repro: slot-failure fallback with a Burgbyn10-shaped water config."""
from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from sqlalchemy import create_engine

from backend.learning.models import Base
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterDeviceConfig,
    WaterHeaterGlobalConfig,
)
from executor.engine import ExecutorEngine
from executor.actions import HAClient


@pytest.fixture
def temp_schedule():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        p = f.name
    yield p
    with contextlib.suppress(OSError):
        Path(p).unlink()


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        p = f.name
    Base.metadata.create_all(create_engine(f"sqlite:///{p}"))
    yield p
    with contextlib.suppress(OSError):
        Path(p).unlink()


@pytest.fixture
def engine(temp_schedule, temp_db):
    from executor.actions import ActionDispatcher

    with patch("executor.engine.load_executor_config") as mock_config:
        config = ExecutorConfig(
            enabled=True,
            schedule_path=temp_schedule,
            timezone="Europe/Stockholm",
            automation_toggle_entity="input_boolean.automation",
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            notifications=NotificationConfig(),
            # Site: the tanks' globals
            water_heater=WaterHeaterGlobalConfig(temp_normal=60, temp_off=40, temp_boost=70,
                                                 temp_max=85),
            water_heater_devices=[
                WaterHeaterDeviceConfig(
                    id="main_tank", name="VVB", target_entity="switch.vvb", power_kw=3.0,
                ),
                # Site: the spa, 20-40 bridge helper
                WaterHeaterDeviceConfig(
                    id="spa", name="Spa", target_entity="input_number.spa_darkstar_target_temp",
                    power_kw=1.8, temp_off=20, temp_normal=38, temp_boost=40, temp_max=40,
                    idle_hold=True, state_entity="climate.layzspa_temperature_control",
                    climate_heat_mode="heat",
                ),
            ],
        )
        mock_config.return_value = config
        with patch("executor.engine.load_yaml") as mock_yaml:
            mock_yaml.return_value = {"input_sensors": {}}
            with patch.object(ExecutorEngine, "_get_db_path", return_value=temp_db):
                eng = ExecutorEngine("config.yaml")
                mock_ha = MagicMock(spec=HAClient)

                def get_state_value(entity_id):
                    if "input_boolean" in entity_id or "automation" in entity_id:
                        return "on"
                    if "soc" in entity_id:
                        return "50"
                    if entity_id == "input_number.spa_darkstar_target_temp":
                        return "20"          # spa currently OFF/fan_only per the plan
                    if entity_id == "switch.vvb":
                        return "off"
                    return "0.0"

                mock_ha.get_state_value.side_effect = get_state_value
                mock_ha.get_state = AsyncMock(
                    return_value={"state": "fan_only", "attributes": {"temperature": 20}}
                )
                mock_ha.set_select_option.return_value = True
                mock_ha.set_switch.return_value = True
                mock_ha.set_number.return_value = True
                mock_ha.set_input_number = AsyncMock(return_value=True)
                mock_ha.call_service = AsyncMock(return_value=True)
                eng.ha_client = mock_ha
                eng.dispatcher = ActionDispatcher(mock_ha, config, shadow_mode=False)
                eng._has_water_heater = True
                yield eng, mock_ha


@pytest.mark.asyncio
async def test_stale_schedule_commands_the_spa(engine, temp_schedule):
    eng, ha = engine
    tz = pytz.timezone("Europe/Stockholm")
    now = datetime.now(tz)
    stale_start = now - timedelta(hours=3)
    stale_end = stale_start + timedelta(minutes=15)
    slot = {
        "start_time": stale_start.isoformat(),
        "end_time": stale_end.isoformat(),
        "end_time_kepler": stale_end.isoformat(),
        "battery_charge_kw": 0.0,
        "battery_discharge_kw": 0.0,
        "export_kwh": 0.0,
        "water_heating_kw": 0.0,
        "soc_target_percent": 50,
        "projected_soc_percent": 45,
    }
    with Path(temp_schedule).open("w", encoding="utf-8") as f:
        json.dump({"schedule": [slot], "meta": {"generated_at": stale_start.isoformat()}}, f)

    await eng.run_once()

    print("OVERRIDE:", eng.status.override_type)
    print("set_input_number calls:", ha.set_input_number.call_args_list)
    print("set_switch calls:", ha.set_switch.call_args_list)
    print("climate call_service calls:", ha.call_service.call_args_list)


@pytest.mark.asyncio
async def test_manual_override_also_commands_the_spa(engine, temp_schedule):
    """MANUAL_OVERRIDE ('executor will not change settings') has actions={} ->
    controller defaults water_temp to 40 -> same branch."""
    eng, ha = engine
    tz = pytz.timezone("Europe/Stockholm")
    now = datetime.now(tz)
    start = now - timedelta(minutes=5)
    end = start + timedelta(minutes=15)
    slot = {
        "start_time": start.isoformat(), "end_time": end.isoformat(),
        "end_time_kepler": end.isoformat(), "battery_charge_kw": 0.0,
        "battery_discharge_kw": 0.0, "export_kwh": 0.0, "water_heating_kw": 0.0,
        "soc_target_percent": 50, "projected_soc_percent": 45,
    }
    with Path(temp_schedule).open("w", encoding="utf-8") as f:
        json.dump({"schedule": [slot], "meta": {}}, f)

    orig = eng._gather_system_state

    async def _with_manual():
        st = await orig()
        st.manual_override_active = True
        return st

    eng._gather_system_state = _with_manual
    await eng.run_once()
    print("MANUAL override type:", eng.status.override_type)
    print("MANUAL set_input_number:", ha.set_input_number.call_args_list)


@pytest.mark.asyncio
async def test_water_boost_writes_global_70_to_the_2040_helper(engine, temp_schedule):
    eng, ha = engine
    tz = pytz.timezone("Europe/Stockholm")
    now = datetime.now(tz)
    start = now - timedelta(minutes=5)
    end = start + timedelta(minutes=15)
    slot = {
        "start_time": start.isoformat(), "end_time": end.isoformat(),
        "end_time_kepler": end.isoformat(), "battery_charge_kw": 0.0,
        "battery_discharge_kw": 0.0, "export_kwh": 0.0, "water_heating_kw": 0.0,
        "soc_target_percent": 50, "projected_soc_percent": 45,
    }
    with Path(temp_schedule).open("w", encoding="utf-8") as f:
        json.dump({"schedule": [slot], "meta": {}}, f)
    eng._water_boost_until = datetime.now(tz) + timedelta(minutes=60)
    await eng.run_once()
    print("BOOST set_input_number:", ha.set_input_number.call_args_list)
    print("BOOST set_switch:", ha.set_switch.call_args_list)
