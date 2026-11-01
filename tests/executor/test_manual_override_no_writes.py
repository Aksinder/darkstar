"""Manual override (executor.manual_override_entity "on") stops ALL writes.

override.py promises "executor will not change settings", yet the controller's
_apply_override turned it into idle / soc_target 10 / water OFF and the dispatcher
still executed it. Contract now (upstream dc1595c6 skip_writes idea): no inverter,
EV, water, sink, balancer or fuse-cap writes; telemetry and the execution record
continue; one notification on entering and one on leaving manual mode.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from sqlalchemy import create_engine

from backend.learning.models import Base
from executor.actions import ActionDispatcher, HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterConfig,
)
from executor.engine import ExecutorEngine

if TYPE_CHECKING:
    from collections.abc import Iterator

STHLM = pytz.timezone("Europe/Stockholm")
MANUAL = "input_boolean.darkstar_manual"


@pytest.fixture
def eng_ha() -> Iterator[tuple[ExecutorEngine, MagicMock, dict[str, str]]]:
    with tempfile.TemporaryDirectory() as d:
        sched = Path(d) / "schedule.json"
        db = Path(d) / "t.db"
        Base.metadata.create_all(create_engine(f"sqlite:///{db}"))
        now = datetime.now(STHLM)
        start = now - timedelta(minutes=5)
        end = start + timedelta(minutes=15)
        sched.write_text(
            json.dumps(
                {
                    "schedule": [
                        {
                            "start_time": start.isoformat(),
                            "end_time": end.isoformat(),
                            "battery_charge_kw": 5.0,
                            "battery_discharge_kw": 0,
                            "export_kwh": 0,
                            "soc_target_percent": 80,
                        }
                    ],
                    "meta": {"planned_at": now.isoformat()},
                }
            ),
            encoding="utf-8",
        )
        config = ExecutorConfig(
            enabled=True,
            schedule_path=str(sched),
            timezone="Europe/Stockholm",
            inverter=InverterConfig(),
            water_heater=WaterHeaterConfig(),
            notifications=NotificationConfig(),
            controller=ControllerConfig(),
            has_water_heater=False,
            manual_override_entity=MANUAL,
        )
        with (
            patch("executor.engine.load_executor_config", return_value=config),
            patch(
                "executor.engine.load_yaml",
                return_value={"input_sensors": {}, "battery": {"min_soc_percent": 10}},
            ),
            patch.object(ExecutorEngine, "_get_db_path", return_value=str(db)),
        ):
            eng = ExecutorEngine("config.yaml")
        states = {MANUAL: "on"}
        ha = MagicMock(spec=HAClient)
        ha.get_state_value.side_effect = lambda e: states.get(
            e, "on" if "input_boolean" in e else "0.0"
        )
        eng.ha_client = ha
        eng.dispatcher = ActionDispatcher(ha, config, shadow_mode=False)
        eng.dispatcher.execute = AsyncMock(return_value=[])  # type: ignore[method-assign]
        eng.dispatcher._send_notification = AsyncMock()  # type: ignore[method-assign]
        eng.dispatcher.notify_override = AsyncMock()  # type: ignore[method-assign]
        yield eng, ha, states


@pytest.mark.asyncio
async def test_manual_override_skips_every_write_and_records(eng_ha):
    eng, _ha, _states = eng_ha
    with (
        patch.object(eng, "_control_ev_charger", AsyncMock()) as ev,
        patch.object(eng, "_apply_fuse_battery_cap") as fuse,
    ):
        result = await eng.run_once()
    assert result["success"] is True, result
    eng.dispatcher.execute.assert_not_awaited()
    ev.assert_not_awaited()
    fuse.assert_not_called()
    # Heartbeat IS published on manual (attribute manual: true): the HA dead-man
    # watchdogs keyed on it must not undo a deliberate manual session.
    assert eng._dispatched_this_tick is True
    assert result["override"]["type"] == "manual_override"
    action = result["actions"][0]
    assert action["type"] == "manual_override"
    assert action["skipped"] is True
    assert "would have" in action["message"]
    # One manual-mode notice, and NOT a second generic override notice.
    assert eng.dispatcher._send_notification.await_count == 1
    eng.dispatcher.notify_override.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_override_notifies_once_on_enter_and_once_on_leave(eng_ha):
    eng, _ha, states = eng_ha
    await eng.run_once()
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 1
    titles = [c.kwargs.get("title") for c in eng.dispatcher._send_notification.await_args_list]
    assert titles == ["Darkstar: manuellt läge"]

    states[MANUAL] = "off"
    await eng.run_once()
    assert eng.dispatcher.execute.await_count == 1  # control resumes on the next tick
    titles = [c.kwargs.get("title") for c in eng.dispatcher._send_notification.await_args_list]
    assert titles == ["Darkstar: manuellt läge", "Darkstar: manuellt läge avslutat"]


@pytest.mark.asyncio
async def test_manual_notice_respects_override_notification_flag(eng_ha):
    eng, _ha, _states = eng_ha
    eng.config.notifications.on_override_activated = False
    await eng.run_once()
    eng.dispatcher._send_notification.assert_not_awaited()
    eng.dispatcher.execute.assert_not_awaited()


# --- review fixes: safety layers, heartbeat, reminder, unreadable blips -------------


@pytest.mark.asyncio
async def test_manual_heartbeat_carries_manual_attribute(eng_ha):
    eng, ha, _states = eng_ha
    await eng.run_once()
    await eng._publish_heartbeat(datetime.now(STHLM))
    ((entity, _state, attrs), _kw) = ha.set_state.await_args
    assert entity == "sensor.darkstar_executor_heartbeat"
    assert attrs["manual"] is True


@pytest.mark.asyncio
async def test_enter_notice_says_fuse_relief_stays(eng_ha):
    eng, _ha, _states = eng_ha
    await eng.run_once()
    msg = eng.dispatcher._send_notification.await_args.args[0]
    assert "Säkringsvakten är kvar" in msg
    assert "SÄNKA" in msg
    assert "30:e minut" in msg


@pytest.mark.asyncio
async def test_unreadable_manual_entity_holds_last_state(eng_ha):
    eng, _ha, states = eng_ha
    await eng.run_once()
    states[MANUAL] = "unavailable"
    await eng.run_once()
    states[MANUAL] = "unknown"
    await eng.run_once()
    eng.dispatcher.execute.assert_not_awaited()  # still manual: no writes resumed
    titles = [c.kwargs.get("title") for c in eng.dispatcher._send_notification.await_args_list]
    assert titles == ["Darkstar: manuellt läge"]  # no leave/enter pair per blip
    states[MANUAL] = "off"
    await eng.run_once()
    assert eng.dispatcher.execute.await_count == 1


@pytest.mark.asyncio
async def test_first_read_unreadable_means_not_manual(eng_ha):
    eng, _ha, states = eng_ha
    states[MANUAL] = "unavailable"
    await eng.run_once()
    assert eng.dispatcher.execute.await_count == 1
    eng.dispatcher._send_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_reminder_after_n_minutes_then_rate_limited(eng_ha):
    eng, _ha, _states = eng_ha
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 1  # enter notice
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 1  # not yet 30 min

    eng._manual_override_since -= 31 * 60
    await eng.run_once()
    titles = [c.kwargs.get("title") for c in eng.dispatcher._send_notification.await_args_list]
    assert titles[-1] == "Darkstar: manuellt läge fortfarande på"
    assert eng.dispatcher._send_notification.await_count == 2
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 2  # rate-limited

    eng._manual_reminder_last -= 30 * 60
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 3


@pytest.mark.asyncio
async def test_manual_reminder_disabled_by_zero(eng_ha):
    eng, _ha, _states = eng_ha
    eng.config.pause_reminder_minutes = 0
    await eng.run_once()
    eng._manual_override_since -= 120 * 60
    await eng.run_once()
    assert eng.dispatcher._send_notification.await_count == 1


# --- manual-mode EV fuse relief (reduce-only) ----------------------------------------

EASEE = "dev-easee"


def _servo(phases, *, last_a=None, chargers=None, readback=None):
    from executor.ev_surplus_runtime import EVSurplusChargerCfg, EVSurplusRuntimeConfig

    chargers = chargers or [
        EVSurplusChargerCfg(
            id="easee", easee_device_id=EASEE, power_entity="sensor.easee_power",
            phases=3, min_current_a=6.0, max_current_a=16.0,
        )
    ]
    cfg = EVSurplusRuntimeConfig(
        enabled=True, grid_power_entity="sensor.grid", fuse_guard_enabled=True,
        fuse_phase_entities={"a": "s.a", "b": "s.b", "c": "s.c"}, chargers=chargers,
    )
    srv = MagicMock()
    srv.cfg = cfg
    srv.fuse_budget_a = 23.0
    srv.read_phase_currents = AsyncMock(return_value=phases)
    srv._failsafe_stop_all = AsyncMock()
    srv.last_phase_ok_ts = None
    srv._last_a = dict(last_a or {})
    srv._last_ts = {}
    srv._last_switch = {}
    return srv


def _ha(values):
    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(side_effect=lambda e: values.get(e))
    ha.call_service = AsyncMock()
    return ha


def _engine_for_relief(srv, ha):
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._ev_surplus = srv
    eng.ha_client = ha
    eng.config = MagicMock(shadow_mode=False)
    eng._manual_fuse_cap_a = {}
    return eng


@pytest.mark.asyncio
async def test_manual_relief_lowers_easee_dynamic_limit_on_overload():
    # L1 at 27 A vs budget 23 A: Easee at 16 A x 3ph must drop 4 A -> 12 A.
    srv = _servo({"a": 27.0, "b": 18.0, "c": 18.0}, last_a={"easee": 16.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    res = await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once_with(
        "easee", "set_charger_dynamic_limit", None,
        {"device_id": EASEE, "current": 12, "time_to_live": 0},
    )
    assert res and res[0].success
    assert srv._last_a["easee"] == 12.0  # servo memory kept in step
    # Next tick, overload gone: nothing is raised back.
    srv.read_phase_currents.return_value = {"a": 20.0, "b": 15.0, "c": 15.0}
    ha.call_service.reset_mock()
    assert await eng._manual_ev_fuse_relief() == []
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_relief_sheds_a_car_the_human_started_despite_servo_memory_zero():
    # Servo had the charger at 0 before manual; the human started it at 16 A. The
    # remembered 0 is not "in force" against a measured 16 A draw: shed 16 -> 12.
    srv = _servo({"a": 27.0, "b": 18.0, "c": 18.0}, last_a={"easee": 0.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once_with(
        "easee", "set_charger_dynamic_limit", None,
        {"device_id": EASEE, "current": 12, "time_to_live": 0},
    )


@pytest.mark.asyncio
async def test_manual_relief_sheds_again_after_human_restarts_a_paused_car():
    srv = _servo({"a": 27.0, "b": 18.0, "c": 18.0}, last_a={"easee": 0.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    eng._manual_fuse_cap_a["easee"] = 0.0  # relief paused it earlier this session
    await eng._manual_ev_fuse_relief()
    assert ha.call_service.await_args.args[3]["current"] == 12


@pytest.mark.asyncio
async def test_manual_relief_cut_is_proportional_to_measured_draw():
    # Stale servo memory of 8 A must not turn a 2 A cut on a measured 16 A draw into
    # a full pause: basis is the measurement -> 14 A.
    srv = _servo({"a": 25.0, "b": 18.0, "c": 18.0}, last_a={"easee": 8.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    assert ha.call_service.await_args.args[3]["current"] == 14


@pytest.mark.asyncio
async def test_manual_relief_no_draw_and_no_readback_sheds_nothing():
    srv = _servo({"a": 27.0, "b": 18.0, "c": 18.0}, last_a={"easee": 16.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": "0"})
    eng = _engine_for_relief(srv, ha)
    assert await eng._manual_ev_fuse_relief() == []
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_relief_lagging_tesla_sensor_uses_readback():
    from executor.ev_surplus_runtime import EVSurplusChargerCfg

    tesla = EVSurplusChargerCfg(
        id="tesla", current_entity="number.tesla_current", switch_entity="switch.tesla",
        power_entity="sensor.tesla_power", phases=3, min_current_a=5.0, max_current_a=16.0,
    )
    srv = _servo({"a": 26.0, "b": 20.0, "c": 20.0}, chargers=[tesla])
    ha = _ha({"sensor.grid": "9000", "sensor.tesla_power": "0", "number.tesla_current": "16"})
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once_with(
        "number", "set_value", "number.tesla_current", {"value": 13.0}
    )


@pytest.mark.asyncio
async def test_manual_relief_tesla_number_and_readback():
    from executor.ev_surplus_runtime import EVSurplusChargerCfg

    tesla = EVSurplusChargerCfg(
        id="tesla", current_entity="number.tesla_current", switch_entity="switch.tesla",
        power_entity="sensor.tesla_power", phases=3, min_current_a=5.0, max_current_a=16.0,
    )
    srv = _servo({"a": 25.0, "b": 20.0, "c": 20.0}, chargers=[tesla])
    vals = {
        "sensor.grid": "9000",
        "sensor.tesla_power": str(16 * 230 * 3),
        "number.tesla_current": "16",
    }
    ha = _ha(vals)
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once_with(
        "number", "set_value", "number.tesla_current", {"value": 14.0}
    )
    # Readback already below what the overload asks for: no write (never a raise).
    ha.call_service.reset_mock()
    eng._manual_fuse_cap_a.clear()
    srv._last_a.clear()
    vals["number.tesla_current"] = "10"
    await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once()  # own = min(16, 10) = 10 -> 8 A
    assert ha.call_service.await_args.args[3] == {"value": 8.0}


@pytest.mark.asyncio
async def test_manual_relief_switch_pause_below_floor():
    from executor.ev_surplus_runtime import EVSurplusChargerCfg

    tesla = EVSurplusChargerCfg(
        id="tesla", current_entity="number.tesla_current", switch_entity="switch.tesla",
        power_entity="sensor.tesla_power", phases=1, min_current_a=5.0, max_current_a=16.0,
        phase_map=("a",),
    )
    srv = _servo({"a": 33.0, "b": 10.0, "c": 10.0}, chargers=[tesla])
    ha = _ha({"sensor.grid": "9000", "sensor.tesla_power": str(12 * 230)})
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    ha.call_service.assert_awaited_once_with("switch", "turn_off", "switch.tesla")
    assert srv._last_switch["tesla"] is False


@pytest.mark.asyncio
async def test_manual_relief_nothing_when_within_budget_or_exporting():
    srv = _servo({"a": 22.0, "b": 18.0, "c": 18.0}, last_a={"easee": 16.0})
    ha = _ha({"sensor.grid": "5000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    assert await eng._manual_ev_fuse_relief() == []
    # 25 A on L1 but exporting 6.9 kW: 10 A/phase export credit -> 15 A, fine.
    srv.read_phase_currents.return_value = {"a": 25.0, "b": 12.0, "c": 12.0}
    ha.get_state_value.side_effect = lambda e: {"sensor.grid": "-6900"}.get(e)
    assert await eng._manual_ev_fuse_relief() == []
    ha.call_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_relief_blind_guard_uses_servo_failsafe():
    srv = _servo(None)
    ha = _ha({})
    eng = _engine_for_relief(srv, ha)
    await eng._manual_ev_fuse_relief()
    srv._failsafe_stop_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_relief_respects_shadow():
    srv = _servo({"a": 27.0, "b": 18.0, "c": 18.0}, last_a={"easee": 16.0})
    ha = _ha({"sensor.grid": "12000", "sensor.easee_power": str(16 * 230 * 3)})
    eng = _engine_for_relief(srv, ha)
    eng.config.shadow_mode = True
    res = await eng._manual_ev_fuse_relief()
    ha.call_service.assert_not_awaited()
    assert "shadow" in res[0].message


@pytest.mark.asyncio
async def test_manual_tick_runs_fuse_relief_but_no_dispatch(eng_ha):
    eng, _ha, _states = eng_ha
    with patch.object(eng, "_manual_ev_fuse_relief", AsyncMock(return_value=[])) as relief:
        await eng.run_once()
    relief.assert_awaited_once()
    eng.dispatcher.execute.assert_not_awaited()


# --- load balancer under manual: reductions only ------------------------------------


@pytest.mark.asyncio
async def test_balancer_reductions_only_lower():
    from executor.load_balancer import EVBalancerOutput
    from executor.load_balancer_actuator import LoadBalancerActuator

    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng.ha_client = MagicMock()
    eng.config = MagicMock(shadow_mode=False)
    act = LoadBalancerActuator()
    act._apply_ev = AsyncMock(return_value="capped")  # type: ignore[method-assign]
    eng._balancer_actuator = act
    ccfg = MagicMock(id="easee")
    status = MagicMock(
        ev_outputs=[
            EVBalancerOutput("easee", 10, "throttling"),
        ]
    )
    done = await eng._apply_balancer_reductions(status, {"easee": ccfg}, {"easee": 16.0})
    assert done == ["capped"]
    # A release (target >= servo) is never applied on manual.
    act._apply_ev.reset_mock()
    status.ev_outputs = [EVBalancerOutput("easee", 16, "idle")]
    assert await eng._apply_balancer_reductions(status, {"easee": ccfg}, {"easee": 16.0}) == []
    # Nor is raising an existing cap (8 A held -> 10 A wanted).
    act._ev_cap["easee"] = 8
    act._ev_path["easee"] = "easee"
    act._ev_servo_at_cap["easee"] = 16.0
    status.ev_outputs = [EVBalancerOutput("easee", 10, "throttling")]
    assert await eng._apply_balancer_reductions(status, {"easee": ccfg}, {"easee": 16.0}) == []
    act._apply_ev.assert_not_awaited()


@pytest.mark.asyncio
async def test_balancer_plan_targets_are_ignored_on_manual():
    from executor.load_balancer import EVBalancerOutput
    from executor.load_balancer_actuator import LoadBalancerActuator

    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng.ha_client = MagicMock()
    eng.config = MagicMock(shadow_mode=False)
    act = LoadBalancerActuator()
    act._apply_ev = AsyncMock(return_value="x")  # type: ignore[method-assign]
    eng._balancer_actuator = act
    ccfg = MagicMock(id="easee")
    for out in (
        EVBalancerOutput("easee", None, "idle", reason="Plan ended"),
        EVBalancerOutput("easee", 10, "idle", reason="At target"),
    ):
        status = MagicMock(ev_outputs=[out])
        assert await eng._apply_balancer_reductions(status, {"easee": ccfg}, {"easee": 16.0}) == []
    act._apply_ev.assert_not_awaited()
    # A fuse-driven pause still applies.
    status = MagicMock(ev_outputs=[EVBalancerOutput("easee", None, "paused")])
    assert await eng._apply_balancer_reductions(status, {"easee": ccfg}, {"easee": 16.0}) == ["x"]


@pytest.mark.asyncio
async def test_heartbeat_withheld_when_manual_safety_fails(eng_ha):
    eng, _ha, _states = eng_ha
    with patch.object(eng, "_manual_safety_tick", AsyncMock(side_effect=RuntimeError("boom"))):
        result = await eng.run_once()
    assert result["success"] is True
    assert eng._dispatched_this_tick is False
    assert any(a["type"] == "manual_safety" and not a["success"] for a in result["actions"])


@pytest.mark.asyncio
async def test_heartbeat_withheld_when_a_relief_write_failed(eng_ha):
    from executor.actions import ActionResult

    eng, _ha, _states = eng_ha
    bad = ActionResult(action_type="manual_fuse_relief:x", success=False, message="no")
    with patch.object(eng, "_manual_safety_tick", AsyncMock(return_value=[bad])):
        await eng.run_once()
    assert eng._dispatched_this_tick is False


@pytest.mark.asyncio
async def test_unreadable_manual_hold_is_capped_then_dropped_with_notice(eng_ha):
    eng, _ha, states = eng_ha
    await eng.run_once()  # manual on
    states[MANUAL] = "unavailable"
    await eng.run_once()
    eng.dispatcher.execute.assert_not_awaited()
    assert eng._manual_unreadable_since is not None
    eng._manual_unreadable_since -= 61 * 60
    await eng.run_once()
    assert eng.dispatcher.execute.await_count == 1  # control resumed
    titles = [c.kwargs.get("title") for c in eng.dispatcher._send_notification.await_args_list]
    assert titles == ["Darkstar: manuellt läge", "Darkstar: manuellt läge släppt (oläsbar)"]
    await eng.run_once()  # still unreadable: stays off, no new notices
    assert eng.dispatcher.execute.await_count == 2
    assert eng.dispatcher._send_notification.await_count == 2
    states[MANUAL] = "on"
    await eng.run_once()  # readable ON again re-enters manual
    assert eng.dispatcher.execute.await_count == 2
