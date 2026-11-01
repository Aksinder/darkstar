"""Repeated command failures notify once per streak (upstream b7cddc26 idea).

A command type that fails ACTION_FAILURE_NOTIFY_STREAK ticks in a row sends ONE
notify_error; a success resets the streak. Refused (skipped) writes have their
own notices and neither count nor reset. The execution record carries a
top-level error_message naming the failed commands.
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
from executor.actions import ActionDispatcher, ActionResult, HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterConfig,
)
from executor.engine import ACTION_FAILURE_NOTIFY_STREAK, ExecutorEngine

if TYPE_CHECKING:
    from collections.abc import Iterator

STHLM = pytz.timezone("Europe/Stockholm")


def _fail(t: str = "work_mode") -> ActionResult:
    return ActionResult(action_type=t, success=False, message="HA 500", entity_id="select.ems")


def _ok(t: str = "work_mode") -> ActionResult:
    return ActionResult(action_type=t, success=True, message="ok")


def _refused(t: str = "work_mode") -> ActionResult:
    return ActionResult(action_type=t, success=False, message="unreachable", skipped=True)


def _bare() -> ExecutorEngine:
    eng = object.__new__(ExecutorEngine)
    eng.dispatcher = MagicMock()
    eng.dispatcher.notify_error = AsyncMock()
    return eng


@pytest.mark.asyncio
async def test_notifies_once_after_streak_and_rearms_after_success() -> None:
    eng = _bare()
    for _ in range(ACTION_FAILURE_NOTIFY_STREAK - 1):
        await eng._track_command_failures([_fail()])
    eng.dispatcher.notify_error.assert_not_awaited()
    await eng._track_command_failures([_fail()])
    assert eng.dispatcher.notify_error.await_count == 1
    assert "work_mode" in eng.dispatcher.notify_error.await_args.args[0]
    for _ in range(5):  # streak continues: still exactly one notice
        await eng._track_command_failures([_fail()])
    assert eng.dispatcher.notify_error.await_count == 1

    await eng._track_command_failures([_ok()])  # reset
    for _ in range(ACTION_FAILURE_NOTIFY_STREAK):
        await eng._track_command_failures([_fail()])
    assert eng.dispatcher.notify_error.await_count == 2


@pytest.mark.asyncio
async def test_streaks_are_per_type_and_refusals_do_not_count_or_reset() -> None:
    eng = _bare()
    await eng._track_command_failures([_fail("work_mode"), _fail("water_temp")])
    await eng._track_command_failures([_fail("work_mode"), _ok("water_temp")])
    await eng._track_command_failures([_refused("work_mode"), _fail("water_temp")])
    eng.dispatcher.notify_error.assert_not_awaited()
    await eng._track_command_failures([_fail("work_mode")])  # third real failure
    assert eng.dispatcher.notify_error.await_count == 1
    assert "work_mode" in eng.dispatcher.notify_error.await_args.args[0]


@pytest.fixture
def tick_eng() -> Iterator[ExecutorEngine]:
    with tempfile.TemporaryDirectory() as d:
        sched = Path(d) / "schedule.json"
        db = Path(d) / "t.db"
        Base.metadata.create_all(create_engine(f"sqlite:///{db}"))
        now = datetime.now(STHLM)
        start = now - timedelta(minutes=5)
        sched.write_text(
            json.dumps(
                {
                    "schedule": [
                        {
                            "start_time": start.isoformat(),
                            "end_time": (start + timedelta(minutes=15)).isoformat(),
                            "battery_charge_kw": 5.0,
                            "soc_target_percent": 80,
                        }
                    ],
                    "meta": {},
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
        ha = MagicMock(spec=HAClient)
        ha.get_state_value.side_effect = lambda e: "0.0"
        eng.ha_client = ha
        eng.dispatcher = ActionDispatcher(ha, config, shadow_mode=True)
        eng.dispatcher.execute = AsyncMock(return_value=[_fail()])  # type: ignore[method-assign]
        eng.dispatcher.notify_error = AsyncMock()  # type: ignore[method-assign]
        yield eng


@pytest.mark.asyncio
async def test_tick_notifies_and_records_error_message(tick_eng) -> None:
    records = []
    tick_eng.history.log_execution = records.append  # type: ignore[method-assign]
    for _ in range(ACTION_FAILURE_NOTIFY_STREAK):
        await tick_eng.run_once()
    assert tick_eng.dispatcher.notify_error.await_count == 1
    assert records[-1].success == 0
    assert records[-1].error_message == "work_mode: HA 500"
