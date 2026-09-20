"""A write to an entity that is not answering is a lie, not a command.

2026-09-18 18:16 the Sungrow modbus link dropped (10.100.100.2:502, "Not
connected") and stayed down for 50 hours. Home Assistant still ACCEPTS a service
call against an unavailable entity and returns success, so every tick:

    executor.actions - Executing mode 'self_consumption' ... for profile 'sungrow'
    executor.actions - Mode 'self_consumption' executed: 6/6 actions successful

...while nothing reached the inverter — plus one "Set work_mode to
Self-consumption mode (default)" notification per minute, ~3000 of them, which is
what the owner noticed.

Two rules follow: don't write to an entity that reads unavailable/unknown (say so
once, and once more when it comes back), and only notify on a change that was
verified by a read-back.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from executor.actions import ActionDispatcher
from executor.config import ExecutorConfig
from executor.controller import ControllerDecision
from executor.profiles import EntityDefinition, InverterProfile, ModeAction, ModeDefinition


def _profile():
    profile = MagicMock(spec=InverterProfile)
    profile.modes = {
        "self_consumption": ModeDefinition(
            description="Self consumption",
            actions=[ModeAction(entity="work_mode", value="Self-consumption mode (default)")],
        )
    }
    profile.metadata = MagicMock()
    profile.metadata.name = "sungrow"
    profile.entities = {
        "work_mode": EntityDefinition(
            default_entity="select.ems_mode",
            domain="select",
            category="system",
            description="EMS mode",
            required=True,
        )
    }
    profile.get_mode = lambda name: profile.modes[name]
    profile.behavior = MagicMock()
    profile.behavior.requires_mode_settling = False
    profile.behavior.mode_settling_ms = 0
    return profile


def _dispatcher(states):
    """A dispatcher whose HA returns `states.pop(0)` (or the last value) per read."""
    ha = AsyncMock()
    queue = list(states)

    async def _read(_entity):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    ha.get_state_value = AsyncMock(side_effect=_read)
    ha.set_select_option = AsyncMock(return_value=True)
    ha.set_number = AsyncMock(return_value=True)
    ha.set_switch = AsyncMock(return_value=True)
    ha.send_notification = AsyncMock(return_value=True)
    cfg = ExecutorConfig()
    cfg.inverter.work_mode = "select.ems_mode"
    cfg.notifications.on_export_start = True  # work_mode notifications gate on this
    cfg.notifications.on_write_unverified = True
    return ActionDispatcher(ha, cfg, profile=_profile()), ha


DECISION = ControllerDecision(
    mode_intent="self_consumption", charge_value=0.0, discharge_value=0.0,
    soc_target=10, water_temp=40, control_unit="W",
)
TARGET = "Self-consumption mode (default)"


class TestUnreachableEntityIsNotWritten:
    @pytest.mark.parametrize("state", ["unavailable", "unknown", "", None])
    @pytest.mark.asyncio
    async def test_no_write_when_the_entity_does_not_answer(self, state):
        disp, ha = _dispatcher([state])
        results = await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()
        work = next(r for r in results if r.action_type == "work_mode")
        assert work.success is False
        assert work.skipped is True
        assert "unreachable" in work.message

    @pytest.mark.asyncio
    async def test_the_outage_notice_fires_once_not_once_per_tick(self):
        disp, ha = _dispatcher(["unavailable"])
        for _ in range(5):
            await disp.execute(DECISION)
        titles = [c.args[1] for c in ha.send_notification.call_args_list]
        assert sum("svarar inte" in t for t in titles) == 1, titles

    @pytest.mark.asyncio
    async def test_recovery_resumes_writes_and_says_so_once(self):
        # tick 1 unavailable, then the device answers with a stale value twice.
        disp, ha = _dispatcher(["unavailable", "Forced mode", TARGET, "Forced mode", TARGET])
        await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with("select.ems_mode", TARGET)
        titles = [c.args[1] for c in ha.send_notification.call_args_list]
        assert sum("svarar igen" in t for t in titles) == 1, titles
        # ...and the recovery notice does not repeat on the next healthy tick.
        await disp.execute(DECISION)
        titles = [c.args[1] for c in ha.send_notification.call_args_list]
        assert sum("svarar igen" in t for t in titles) == 1, titles

    @pytest.mark.asyncio
    async def test_a_healthy_entity_at_target_is_untouched(self):
        disp, ha = _dispatcher([TARGET])
        results = await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()
        work = next(r for r in results if r.action_type == "work_mode")
        assert work.success is True and work.skipped is True
        assert not ha.send_notification.called


class TestNotifyOnlyOnAVerifiedChange:
    @pytest.mark.asyncio
    async def test_verified_change_notifies(self):
        disp, ha = _dispatcher(["Forced mode", TARGET])
        await disp.execute(DECISION)
        msgs = [c.args[2] for c in ha.send_notification.call_args_list]
        assert any("Set work_mode to" in m for m in msgs), msgs

    @pytest.mark.asyncio
    async def test_unverified_write_stays_silent(self):
        """The device took the call but did not change: no cheerful notification."""
        disp, ha = _dispatcher(["Forced mode", "Forced mode"])
        results = await disp.execute(DECISION)
        msgs = [c.args[2] for c in ha.send_notification.call_args_list]
        assert not any("Set work_mode to" in m for m in msgs), msgs
        work = next(r for r in results if r.action_type == "work_mode")
        assert work.verification_success is False

    @pytest.mark.asyncio
    async def test_unreadable_readback_stays_silent(self):
        """Wrote, then the device stopped answering — no evidence, no notification."""
        disp, ha = _dispatcher(["Forced mode", None])
        await disp.execute(DECISION)
        msgs = [c.args[2] for c in ha.send_notification.call_args_list]
        assert not any("Set work_mode to" in m for m in msgs), msgs
