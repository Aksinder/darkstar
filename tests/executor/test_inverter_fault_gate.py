"""An inverter that answers "Fault" is not an inverter that takes commands.

2026-09-16 17:31 the SH10RT went to Fault (system-state register 0x0100) and
stayed there until the owner power-cycled it on 09-23 20:31. All week every
control entity was readable and every write was accepted — the inverter simply
did nothing with them: no charge, no discharge, no PV from its own strings. The
executor logged "5/5 actions successful" per tick, the fuse guard capped charge
power on a battery that could not charge, and the unreachable guard never fired
because nothing was unreachable.

Rule: read the profile's ``inverter_state`` before applying a mode; while it is
in ``behavior.fault_states`` apply nothing, say so once, and say once more when
it runs again. An unreadable state sensor leaves the gate open (that is the link
being down, which the per-entity guard covers). Shadow mode is exempt.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from executor.actions import ActionDispatcher
from executor.config import ExecutorConfig
from executor.controller import ControllerDecision
from executor.profiles import (
    EntityDefinition,
    InverterProfile,
    ModeAction,
    ModeDefinition,
    ProfileBehavior,
    load_profile,
)

STATE_ENTITY = "sensor.sungrow_inverter_state"
WORK_MODE = "select.ems_mode"
TARGET = "Self-consumption mode (default)"
DECISION = ControllerDecision(
    mode_intent="self_consumption",
    charge_value=0.0,
    discharge_value=0.0,
    soc_target=10,
    water_temp=40,
    control_unit="W",
)


def _profile(fault_states=("Fault",), declare_state_entity=True):
    profile = MagicMock(spec=InverterProfile)
    profile.modes = {
        "self_consumption": ModeDefinition(
            description="Self consumption",
            actions=[ModeAction(entity="work_mode", value=TARGET)],
        )
    }
    profile.metadata = MagicMock()
    profile.metadata.name = "sungrow"
    profile.entities = {
        "work_mode": EntityDefinition(
            default_entity=WORK_MODE,
            domain="select",
            category="system",
            description="EMS mode",
            required=True,
        ),
    }
    if declare_state_entity:
        profile.entities["inverter_state"] = EntityDefinition(
            default_entity=STATE_ENTITY,
            domain="sensor",
            category="system",
            description="Inverter state",
            required=False,
        )
    profile.get_mode = lambda name: profile.modes[name]
    profile.behavior = ProfileBehavior(control_unit="W", fault_states=list(fault_states))
    return profile


def _dispatcher(inverter_states, work_mode_states=("Forced mode", TARGET), **profile_kw):
    """HA answers per entity from its own queue (last value repeats)."""
    queues = {STATE_ENTITY: list(inverter_states), WORK_MODE: list(work_mode_states)}
    reads: list[str] = []

    async def _read(entity):
        reads.append(entity)
        q = queues[entity]
        return q.pop(0) if len(q) > 1 else q[0]

    ha = AsyncMock()
    ha.get_state_value = AsyncMock(side_effect=_read)
    ha.set_select_option = AsyncMock(return_value=True)
    ha.send_notification = AsyncMock(return_value=True)
    cfg = ExecutorConfig()
    cfg.inverter.work_mode = WORK_MODE
    cfg.notifications.on_export_start = True
    cfg.notifications.on_write_unverified = True
    disp = ActionDispatcher(ha, cfg, profile=_profile(**profile_kw))
    return disp, ha, reads


def _titles(ha):
    return [c.args[1] for c in ha.send_notification.call_args_list]


class TestFaultedInverterGetsNoCommands:
    @pytest.mark.asyncio
    async def test_no_mode_is_applied_while_faulted(self):
        disp, ha, _ = _dispatcher(["Fault"])
        results = await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()
        assert len(results) == 1
        (r,) = results
        assert r.action_type == "inverter_state"
        assert r.success is False and r.skipped is True
        assert "Fault" in r.message and "self_consumption" in r.message

    @pytest.mark.asyncio
    async def test_fault_states_match_case_insensitively(self):
        disp, ha, _ = _dispatcher([" fault "])
        await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_fault_notice_fires_once_not_once_per_tick(self):
        disp, ha, _ = _dispatcher(["Fault"])
        for _ in range(5):
            await disp.execute(DECISION)
        assert sum("rapporterar fel" in t for t in _titles(ha)) == 1, _titles(ha)

    @pytest.mark.asyncio
    async def test_recovery_resumes_control_and_says_so_once(self):
        disp, ha, _ = _dispatcher(["Fault", "Running", "Running"])
        await disp.execute(DECISION)
        ha.set_select_option.assert_not_called()
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with(WORK_MODE, TARGET)
        assert sum("kör igen" in t for t in _titles(ha)) == 1, _titles(ha)
        await disp.execute(DECISION)
        assert sum("kör igen" in t for t in _titles(ha)) == 1, _titles(ha)

    @pytest.mark.asyncio
    async def test_a_running_inverter_never_hears_about_recovery(self):
        disp, ha, _ = _dispatcher(["Running"])
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with(WORK_MODE, TARGET)
        assert not any("växelriktaren" in t for t in _titles(ha)), _titles(ha)


class TestGateStaysOpenWhenItCannotJudge:
    @pytest.mark.parametrize("state", ["unavailable", "unknown", None])
    @pytest.mark.asyncio
    async def test_unreadable_state_sensor_does_not_block(self, state):
        """The link being down is the per-entity guard's job, not this one's."""
        disp, ha, _ = _dispatcher([state])
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with(WORK_MODE, TARGET)
        assert not any("växelriktaren" in t for t in _titles(ha))

    @pytest.mark.asyncio
    async def test_a_profile_without_fault_states_is_not_gated(self):
        disp, ha, reads = _dispatcher(["Fault"], fault_states=())
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with(WORK_MODE, TARGET)
        assert STATE_ENTITY not in reads  # not even read

    @pytest.mark.asyncio
    async def test_a_profile_without_the_state_entity_is_not_gated(self):
        disp, ha, reads = _dispatcher(["Fault"], declare_state_entity=False)
        await disp.execute(DECISION)
        ha.set_select_option.assert_awaited_once_with(WORK_MODE, TARGET)
        assert STATE_ENTITY not in reads

    @pytest.mark.asyncio
    async def test_shadow_mode_reports_intent_regardless(self):
        disp, _ha, reads = _dispatcher(["Fault"])
        disp.shadow_mode = True
        results = await disp.execute(DECISION)
        assert STATE_ENTITY not in reads
        work = next(r for r in results if r.action_type == "work_mode")
        assert "SHADOW" in work.message or work.skipped


class TestSungrowProfileDeclaresTheGate:
    def test_live_profile_has_state_entity_and_fault_states(self):
        profile = load_profile("sungrow")
        ok, errors = profile.validate()
        assert ok, errors
        ent = profile.entities["inverter_state"]
        assert ent.default_entity == STATE_ENTITY
        assert ent.domain == "sensor" and ent.required is False
        assert profile.behavior.fault_states == ["Fault"]

    def test_fault_states_without_the_entity_is_invalid(self):
        profile = load_profile("sungrow")
        del profile.entities["inverter_state"]
        ok, errors = profile.validate()
        assert not ok and any("inverter_state" in e for e in errors), errors

    def test_a_mode_may_not_write_to_a_sensor(self):
        profile = load_profile("sungrow")
        profile.modes["idle"].actions.append(ModeAction(entity="inverter_state", value="x"))
        ok, errors = profile.validate()
        assert not ok and any("read-only" in e for e in errors), errors

    @pytest.mark.parametrize("name", ["generic", "deye", "fronius"])
    def test_other_profiles_are_untouched(self, name):
        try:
            profile = load_profile(name)
        except FileNotFoundError:
            pytest.skip(f"no {name} profile")
        assert profile.behavior.fault_states == []
