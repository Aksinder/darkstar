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
    assert eng._dispatched_this_tick is False  # no heartbeat: nothing was commanded
    assert result["override"]["type"] == "manual_override"
    (action,) = result["actions"]
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
