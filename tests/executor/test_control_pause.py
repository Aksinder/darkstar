"""Build #17: per-device control-pause (rent-out hands-off) tests.

Covers three layers:
1. Config parse — control_pause_entities lands on WaterHeaterDeviceConfig,
   ExcessPVSinkSpec, and the shared normalize_excess_pv_sinks dict.
2. Dispatcher helper — is_control_paused / control_pause_entity, including the
   FAIL-SAFE direction (unreadable => NOT paused) and per-tick caching.
3. Engine actuation — a paused device is left ALONE (set_water_temp / set_sink
   not called) across the plan, boost, and safety forced-OFF branches, while an
   unpaused sibling (main_tank) is unaffected.
"""

import contextlib
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
import yaml
from sqlalchemy import create_engine

from backend.learning.models import Base
from executor.actions import ActionDispatcher, ActionResult, HAClient
from executor.config import (
    ControllerConfig,
    ExcessPVConfig,
    ExcessPVSinkSpec,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterConfig,
    WaterHeaterDeviceConfig,
    load_executor_config,
    normalize_excess_pv_sinks,
)
from executor.engine import ExecutorEngine

TZ = pytz.timezone("Europe/Stockholm")

MASTER = "input_boolean.darkstar_pausa_villavagn"
PAUSE_VVB = "input_boolean.darkstar_pausa_villavagn_vvb"
PAUSE_AC = "input_boolean.darkstar_pausa_villavagn_ac"


# --------------------------------------------------------------------------- #
# 1. Config parsing
# --------------------------------------------------------------------------- #
class TestControlPauseConfigParsing:
    def test_water_heater_device_parses_pause_list(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_data = {
            "executor": {"enabled": True},
            "water_heaters": [
                {"id": "main_tank", "target_entity": "switch.vvb"},
                {
                    "id": "villavagn_tank",
                    "target_entity": "switch.villavagn_vvb",
                    "control_pause_entities": [PAUSE_VVB, MASTER],
                },
            ],
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        config = load_executor_config(str(config_file))

        by_id = {d.id: d for d in config.water_heater_devices}
        # main_tank has no pause entities => always managed
        assert by_id["main_tank"].control_pause_entities == []
        assert by_id["villavagn_tank"].control_pause_entities == [PAUSE_VVB, MASTER]

    def test_water_heater_device_accepts_scalar_pause_entity(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_data = {
            "executor": {"enabled": True},
            "water_heaters": [
                {
                    "id": "villavagn_tank",
                    "target_entity": "switch.villavagn_vvb",
                    "control_pause_entities": MASTER,  # scalar, not a list
                }
            ],
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        config = load_executor_config(str(config_file))
        assert config.water_heater_devices[0].control_pause_entities == [MASTER]

    def test_normalize_sinks_carries_pause_entities(self):
        entries = normalize_excess_pv_sinks(
            {
                "sinks": [
                    {
                        "id": "villavagn_ac",
                        "entity": "climate.villavagn",
                        "enabled": True,
                        "control_pause_entities": [PAUSE_AC, MASTER],
                    }
                ]
            }
        )
        assert entries[0]["control_pause_entities"] == [PAUSE_AC, MASTER]

    def test_normalize_sinks_defaults_pause_entities_empty(self):
        entries = normalize_excess_pv_sinks(
            {"sinks": [{"id": "poolpump", "entity": "switch.poolpump"}]}
        )
        assert entries[0]["control_pause_entities"] == []

    def test_full_loader_wires_sink_pause_entities(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_data = {
            "executor": {
                "excess_pv": {
                    "sinks": [
                        {
                            "id": "villavagn_ac",
                            "entity": "climate.villavagn",
                            "enabled": True,
                            "control_pause_entities": [PAUSE_AC, MASTER],
                        }
                    ]
                }
            }
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        config = load_executor_config(str(config_file))
        assert config.excess_pv.sinks[0].control_pause_entities == [PAUSE_AC, MASTER]


# --------------------------------------------------------------------------- #
# 2. Dispatcher helper (fail-safe + caching)
# --------------------------------------------------------------------------- #
def _dispatcher(state_map: dict[str, str | None], *, raise_on: set[str] | None = None):
    """Build a real ActionDispatcher over a mock HAClient with a canned state map."""
    raise_on = raise_on or set()

    async def _get_state_value(entity_id: str):
        if entity_id in raise_on:
            raise RuntimeError("boom")
        return state_map.get(entity_id)

    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(side_effect=_get_state_value)
    cfg = ExecutorConfig()
    return ActionDispatcher(ha, cfg, shadow_mode=False), ha


@pytest.mark.asyncio
class TestDispatcherControlPause:
    async def test_empty_or_none_is_not_paused(self):
        disp, _ = _dispatcher({})
        assert await disp.is_control_paused([]) is False
        assert await disp.is_control_paused(None) is False

    async def test_on_is_paused(self):
        disp, _ = _dispatcher({MASTER: "on"})
        assert await disp.is_control_paused([MASTER]) is True
        assert await disp.control_pause_entity([MASTER]) == MASTER

    async def test_off_is_not_paused(self):
        disp, _ = _dispatcher({MASTER: "off"})
        assert await disp.is_control_paused([MASTER]) is False

    async def test_master_on_while_individual_off_is_paused(self):
        # ANY entity on => paused. Master gates even when the per-device toggle is off.
        disp, _ = _dispatcher({PAUSE_VVB: "off", MASTER: "on"})
        assert await disp.control_pause_entity([PAUSE_VVB, MASTER]) == MASTER

    @pytest.mark.parametrize("bad", ["unavailable", "unknown", None])
    async def test_unreadable_state_is_failsafe_not_paused(self, bad):
        disp, _ = _dispatcher({MASTER: bad})
        assert await disp.is_control_paused([MASTER]) is False

    async def test_read_error_is_failsafe_not_paused(self):
        disp, _ = _dispatcher({MASTER: "on"}, raise_on={MASTER})
        assert await disp.is_control_paused([MASTER]) is False

    async def test_cache_reads_each_entity_once_per_tick(self):
        disp, ha = _dispatcher({MASTER: "on", PAUSE_VVB: "off"})
        cache: dict[str, bool] = {}
        # Two devices share the master toggle; with a shared cache the master is
        # read exactly once even across both lookups.
        await disp.is_control_paused([PAUSE_VVB, MASTER], cache)
        await disp.is_control_paused([PAUSE_AC, MASTER], cache)
        reads = [c.args[0] for c in ha.get_state_value.call_args_list]
        assert reads.count(MASTER) == 1


# --------------------------------------------------------------------------- #
# 3. Engine actuation
# --------------------------------------------------------------------------- #
@pytest.fixture
def temp_schedule():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        path = f.name
    yield path
    with contextlib.suppress(OSError):
        Path(path).unlink()


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))
    yield db_path
    with contextlib.suppress(OSError):
        Path(db_path).unlink()


def _make_config(temp_schedule: str) -> ExecutorConfig:
    return ExecutorConfig(
        enabled=True,
        schedule_path=temp_schedule,
        timezone="Europe/Stockholm",
        automation_toggle_entity=None,
        manual_override_entity=None,
        inverter=InverterConfig(),
        water_heater=WaterHeaterConfig(),
        water_heater_devices=[
            WaterHeaterDeviceConfig(
                id="main_tank", name="main_tank", target_entity="switch.vvb"
            ),
            WaterHeaterDeviceConfig(
                id="villavagn_tank",
                name="villavagn_tank",
                target_entity="switch.villavagn_vvb",
                control_pause_entities=[PAUSE_VVB, MASTER],
            ),
        ],
        notifications=NotificationConfig(),
        controller=ControllerConfig(),
        excess_pv=ExcessPVConfig(
            sinks=[
                ExcessPVSinkSpec(
                    id="villavagn_ac",
                    entity="climate.villavagn",
                    enabled=True,
                    control_pause_entities=[PAUSE_AC, MASTER],
                )
            ]
        ),
    )


def _build_engine(temp_schedule, temp_db, paused_on: set[str]):
    """Engine with the real control-pause path; only set_water_temp/set_sink mocked.

    ``paused_on`` is the set of pause entity ids currently reading 'on'.
    """
    config = _make_config(temp_schedule)
    with patch("executor.engine.load_executor_config", return_value=config), patch(
        "executor.engine.load_yaml", return_value={"input_sensors": {}}
    ), patch.object(ExecutorEngine, "_get_db_path", return_value=temp_db):
        engine = ExecutorEngine("config.yaml")

    engine._has_water_heater = True
    engine._has_ev_charger = False

    def _state(entity_id: str):
        if entity_id in (MASTER, PAUSE_VVB, PAUSE_AC):
            return "on" if entity_id in paused_on else "off"
        if "soc" in entity_id:
            return "50"
        if "temp" in entity_id or "target" in entity_id:
            return "55"
        return "0.0"

    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(side_effect=_state)
    ha.set_switch = AsyncMock(return_value=True)
    engine.ha_client = ha

    engine.dispatcher = ActionDispatcher(ha, config, shadow_mode=False)
    engine.dispatcher.set_water_temp = AsyncMock(
        return_value=ActionResult(action_type="water_temp", success=True, message="ok")
    )
    engine.dispatcher.set_sink = AsyncMock(
        return_value=ActionResult(action_type="sink:villavagn_ac", success=True, message="ok")
    )
    return engine


def _write_plan_schedule(temp_schedule: str, *, sink_on: bool = True):
    """A valid current slot with both tanks planned to heat and the AC sink ON."""
    now = datetime.now(TZ)
    start = now - timedelta(minutes=5)
    end = start + timedelta(minutes=15)
    slot = {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "end_time_kepler": end.isoformat(),
        "battery_charge_kw": 0,
        "battery_discharge_kw": 0,
        "export_kwh": 0,
        "water_heating_kw": 3.0,
        "soc_target_percent": 50,
        "projected_soc_percent": 45,
        "water_heaters": {
            "main_tank": {"heating_kw": 3.0},
            "villavagn_tank": {"heating_kw": 3.0},
        },
        "sinks": {"villavagn_ac": sink_on},
    }
    payload = {"schedule": [slot], "meta": {"generated_at": now.isoformat()}}
    Path(temp_schedule).write_text(json.dumps(payload), encoding="utf-8")


def _water_entities_called(engine) -> set[str]:
    return {
        c.args[1] if len(c.args) > 1 else c.kwargs.get("target_entity")
        for c in engine.dispatcher.set_water_temp.call_args_list
    }


@pytest.mark.asyncio
class TestEnginePlanBranch:
    async def test_unpaused_actuates_both_tanks(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on=set())
        _write_plan_schedule(temp_schedule)
        await engine.run_once()
        called = _water_entities_called(engine)
        assert "switch.vvb" in called
        assert "switch.villavagn_vvb" in called

    async def test_individual_pause_skips_only_villavagn(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on={PAUSE_VVB})
        _write_plan_schedule(temp_schedule)
        await engine.run_once()
        called = _water_entities_called(engine)
        # main_tank still managed; villavagn left alone (not commanded on OR off)
        assert "switch.vvb" in called
        assert "switch.villavagn_vvb" not in called

    async def test_master_pause_skips_only_villavagn(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on={MASTER})
        _write_plan_schedule(temp_schedule)
        await engine.run_once()
        called = _water_entities_called(engine)
        assert "switch.vvb" in called
        assert "switch.villavagn_vvb" not in called

    async def test_unavailable_pause_entity_is_failsafe_actuates(self, temp_schedule, temp_db):
        # Pause entity reads 'unavailable' -> treated as NOT paused -> actuated.
        engine = _build_engine(temp_schedule, temp_db, paused_on=set())
        engine.ha_client.get_state_value = AsyncMock(
            side_effect=lambda e: "unavailable"
            if e in (MASTER, PAUSE_VVB, PAUSE_AC)
            else ("50" if "soc" in e else "0.0")
        )
        engine.dispatcher.ha = engine.ha_client
        _write_plan_schedule(temp_schedule)
        await engine.run_once()
        assert "switch.villavagn_vvb" in _water_entities_called(engine)


@pytest.mark.asyncio
class TestEngineSinkBranch:
    async def test_unpaused_sink_actuated(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on=set())
        _write_plan_schedule(temp_schedule, sink_on=True)
        await engine.run_once()
        engine.dispatcher.set_sink.assert_called()

    async def test_paused_sink_not_actuated(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on={PAUSE_AC})
        _write_plan_schedule(temp_schedule, sink_on=True)
        await engine.run_once()
        engine.dispatcher.set_sink.assert_not_called()

    async def test_master_pause_also_skips_sink(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on={MASTER})
        _write_plan_schedule(temp_schedule, sink_on=True)
        await engine.run_once()
        engine.dispatcher.set_sink.assert_not_called()


@pytest.mark.asyncio
class TestEngineBoostBranch:
    async def test_boost_skips_paused_device(self, temp_schedule, temp_db):
        engine = _build_engine(temp_schedule, temp_db, paused_on={MASTER})
        # Stub the notification coroutine the low-SoC boost-guard may fire (unawaited
        # by design in engine code) so it does not leak a RuntimeWarning here.
        engine.dispatcher._send_notification = MagicMock()
        # Activate manual water boost so the boost branch runs this tick.
        engine._water_boost_until = datetime.now(TZ) + timedelta(minutes=30)
        _write_plan_schedule(temp_schedule)
        await engine.run_once()
        called = _water_entities_called(engine)
        # Boost still hits main_tank, but a paused villavagn is hands-off.
        assert "switch.vvb" in called
        assert "switch.villavagn_vvb" not in called


@pytest.mark.asyncio
class TestEngineForcedOffBranch:
    async def test_slot_failure_forced_off_skips_paused_device(self, temp_schedule, temp_db):
        # No valid slot => SLOT_FAILURE_FALLBACK forces water OFF per-device. A paused
        # villavagn must NOT be forced off either (hands-off), main_tank still commanded.
        engine = _build_engine(temp_schedule, temp_db, paused_on={MASTER})
        Path(temp_schedule).write_text(
            json.dumps({"schedule": [], "meta": {}}), encoding="utf-8"
        )
        await engine.run_once()
        called = _water_entities_called(engine)
        assert "switch.vvb" in called
        assert "switch.villavagn_vvb" not in called
