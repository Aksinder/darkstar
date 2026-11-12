"""``mode: window`` cyclic loads in the executor: a room heater on a smart plug.

The plan only books these loads; the plug follows a LIVE gate. Tested here against
the REAL dispatcher (set_cyclic_load with its read-before-write and read-back) and a
fake Home Assistant, so "zero writes" means zero service calls, not zero calls to a
mock that stood in for the writer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from executor import window_load as wl
from executor.actions import ActionDispatcher
from executor.config import (
    CyclicLoadConfig,
    ExecutorConfig,
    NotificationConfig,
    load_executor_config,
)
from executor.engine import ExecutorEngine

PLUG = "switch.kims_uttag"
CAL = "calendar.kim"
PAUSE = "input_boolean.kim_heater_paused"
OVERRIDE = "input_select.kim_heater_override"
TEMP = "sensor.kims_rum_temperature"
STATUS = "sensor.darkstar_kim_heater_state"
MIN = 60.0

# Friday 2026-11-13 12:00 UTC (13:00 in Stockholm).
T0 = datetime(2026, 11, 13, 12, 0, tzinfo=UTC).timestamp()


# 30th percentile 0.8, median 1.2, 85th percentile 1.9.
PRICE_WINDOW = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)


def _at(minutes: float) -> datetime:
    return datetime.fromtimestamp(T0 + minutes * MIN, UTC)


class FakeHA:
    """Just enough Home Assistant: states, service calls that move the switch,
    and the sensors Darkstar publishes."""

    def __init__(self, states: dict[str, Any]):
        self.states = dict(states)
        self.attrs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.published: list[tuple[str, str, dict[str, Any]]] = []
        self.reads: list[str] = []
        self.plug_obeys = True
        self.service_error: Exception | None = None

    async def get_state_value(self, entity_id: str):
        self.reads.append(entity_id)
        value = self.states.get(entity_id)
        if isinstance(value, Exception):
            return None  # the real client swallows read errors into None
        return value

    async def get_state(self, entity_id: str):
        value = await self.get_state_value(entity_id)
        if value is None:
            return None
        return {"state": value, **self.attrs.get(entity_id, {})}

    async def call_service(self, domain: str, service: str, entity_id: str | None = None, **_):
        self.calls.append((domain, service, str(entity_id)))
        if self.service_error is not None:
            raise self.service_error
        if self.plug_obeys and entity_id:
            self.states[entity_id] = "on" if service == "turn_on" else "off"
        return True

    async def set_state(self, entity_id: str, state: str, attributes=None):
        self.published.append((entity_id, state, dict(attributes or {})))
        return True

    @property
    def status(self) -> tuple[str, dict[str, Any]]:
        _, state, attrs = [p for p in self.published if p[0] == STATUS][-1]
        return state, attrs


def _load(**over: Any) -> CyclicLoadConfig:
    base: dict[str, Any] = {
        "id": "kim_heater",
        "name": "Kims element",
        "switch_entity": PLUG,
        "power_kw": 1.0,
        "mode": "window",
        "window_entities": [CAL],
        "control_pause_entities": [PAUSE],
        "override_entity": OVERRIDE,
    }
    base.update(over)
    return CyclicLoadConfig(**base)


class Rig:
    def __init__(self, tmp_path, load: CyclicLoadConfig, states: dict[str, Any], *,
                 shadow: bool = False, events: Any = None):
        config = ExecutorConfig(
            cyclic_loads=[load], shadow_mode=shadow, notifications=NotificationConfig()
        )
        with patch("executor.engine.load_executor_config", return_value=config), patch(
            "executor.engine.load_yaml", return_value={"input_sensors": {}}
        ), patch.object(ExecutorEngine, "_get_db_path", return_value=str(tmp_path / "x.db")):
            self.engine = ExecutorEngine("config.yaml")
        self.ha = FakeHA({PLUG: "off", CAL: "off", PAUSE: "off", OVERRIDE: "auto", **states})
        self.engine.ha_client = self.ha  # type: ignore[assignment]
        self.engine.dispatcher = ActionDispatcher(
            ha_client=self.ha, config=config, shadow_mode=shadow  # type: ignore[arg-type]
        )
        self.engine.dispatcher.notify_unverified = AsyncMock()  # type: ignore[method-assign]
        # None = the calendar's event list cannot be read; a list = its windows.
        self.events: Any = events if events is not None else []
        self.fetches: list[float] = []
        self.now = T0

        async def fetch(entity, start, end, tz, *, match=None):
            self.fetches.append(self.now)
            assert start.tzinfo is not None and end.tzinfo is not None
            if isinstance(self.events, Exception):
                raise self.events
            return self.events

        self.engine.__dict__["_window_loads"] = wl.WindowLoadRuntime(fetch=fetch)
        # The executor's price source: None = unknown price.
        self.price: float | None = None
        self.price_window: list[float] = list(PRICE_WINDOW)
        self.engine._current_import_price = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda: self.price
        )
        self.engine._price_window = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda hours: list(self.price_window)
        )

    def set_price(self, price: float | None) -> None:
        """A new hour's price (the runtime keeps a known price for the clock hour)."""
        self.price = price
        self.engine.__dict__["_window_loads"]._price_memo.clear()

    async def tick(self, minutes: float = 0.0, *, plan: dict[str, float] | None = None,
                   ctx: dict[str, Any] | None = None) -> list[Any]:
        """One executor pass over the cyclic loads at T0 + ``minutes``."""
        self.now = T0 + minutes * MIN
        results: list[Any] = []
        slot = SimpleNamespace(water_heater_plans=plan if plan is not None else {})
        with patch("executor.engine.time.time", return_value=self.now), patch(
            "executor.actions.VERIFY_MAX_WAIT_MS", 0
        ):
            await self.engine._actuate_cyclic_loads(slot, ctx, {}, results)
        return results

    @property
    def writes(self) -> list[tuple[str, str, str]]:
        return self.ha.calls

    @property
    def plug(self) -> str:
        return self.ha.states[PLUG]


class TestLiveGate:
    @pytest.mark.asyncio
    async def test_calendar_on_switches_the_plug_on_and_off_again(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        assert rig.writes == [("switch", "turn_on", PLUG)]
        assert rig.ha.status[0] == "on"
        await rig.tick(1)
        assert len(rig.writes) == 1  # already on: no second write
        rig.ha.states[CAL] = "off"
        await rig.tick(10)
        assert rig.writes[-1] == ("switch", "turn_off", PLUG)
        assert rig.ha.status[0] == "off"

    @pytest.mark.asyncio
    async def test_live_gate_beats_a_stale_plan_that_says_off(self, tmp_path):
        """Calendar edited after planning (or a kept-previous plan): the event is
        active NOW, the plan slot still says 0."""
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0, plan={"kim_heater": 0.0})
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_live_gate_beats_a_stale_plan_that_says_on(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "off", PLUG: "on"})
        await rig.tick(0, plan={"kim_heater": 0.5})
        assert rig.plug == "off"

    @pytest.mark.asyncio
    async def test_a_load_absent_from_the_plan_is_still_gated_live(self, tmp_path):
        """Classic loads go hands-off without a plan entry; a heater must not."""
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0, plan={})
        assert rig.plug == "on"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gate", ["input_boolean.kim_here", "schedule.kim", "binary_sensor.x", "switch.y"]
    )
    async def test_any_on_off_entity_can_be_the_gate(self, tmp_path, gate):
        rig = Rig(tmp_path, _load(window_entities=[gate]), {gate: "on"})
        await rig.tick(0)
        assert rig.plug == "on"
        assert rig.fetches == []  # no calendar, no event list
        rig.ha.states[gate] = "off"
        await rig.tick(10)
        assert rig.plug == "off"

    @pytest.mark.asyncio
    async def test_any_of_several_gates_opens_the_window(self, tmp_path):
        rig = Rig(tmp_path, _load(window_entities=[CAL, "input_boolean.guest"]),
                  {CAL: "off", "input_boolean.guest": "on"})
        await rig.tick(0)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_a_definite_on_wins_over_an_unreadable_sibling(self, tmp_path):
        rig = Rig(tmp_path, _load(window_entities=[CAL, "input_boolean.guest"]),
                  {CAL: "unavailable", "input_boolean.guest": "on"})
        await rig.tick(0)
        assert rig.plug == "on"


class TestEventList:
    @pytest.mark.asyncio
    async def test_summary_match_gates_on_the_event_list_not_the_bare_state(self, tmp_path):
        """A shared calendar is "on" for Alex's event too; only the list can tell."""
        rig = Rig(tmp_path, _load(window_event_match="kim"), {CAL: "on"}, events=[])
        await rig.tick(0)
        assert rig.plug == "off"
        assert CAL not in rig.ha.reads  # the bare state was not even asked for
        rig.events = [(_at(-60), _at(600))]
        await rig.tick(6)  # past the refresh interval
        assert rig.plug == "on"
        state, attrs = rig.ha.status
        assert state == "on" and attrs["window_active"] is True
        assert attrs["next_window_end"] == "2026-11-13T23:00+01:00"

    @pytest.mark.asyncio
    async def test_event_list_is_refreshed_every_few_minutes_not_every_tick(self, tmp_path):
        rig = Rig(tmp_path, _load(window_event_match="kim"), {}, events=[])
        for minute in range(11):
            await rig.tick(minute)
        assert rig.fetches == [T0, T0 + 5 * MIN, T0 + 10 * MIN]

    @pytest.mark.asyncio
    async def test_preheat_switches_on_before_the_window(self, tmp_path):
        rig = Rig(tmp_path, _load(preheat_minutes=45), {CAL: "off"},
                  events=[(_at(60), _at(600))])
        await rig.tick(0)  # 60 min before
        assert rig.plug == "off" and rig.ha.status[0] == "off"
        assert rig.ha.status[1]["next_window_start"] == "2026-11-13T14:00+01:00"
        await rig.tick(14)  # 46 min before
        assert rig.plug == "off"
        await rig.tick(15)  # 45 min before
        assert rig.plug == "on" and rig.ha.status[0] == "preheat"
        rig.ha.states[CAL] = "on"
        await rig.tick(60)
        assert rig.plug == "on" and rig.ha.status[0] == "on"
        assert len(rig.writes) == 1

    @pytest.mark.asyncio
    async def test_no_preheat_without_a_readable_event_list(self, tmp_path):
        """The bare state still gates; preheat simply cannot happen."""
        rig = Rig(tmp_path, _load(preheat_minutes=45), {CAL: "off"}, events=None)
        rig.events = None
        await rig.tick(0)
        assert rig.plug == "off" and rig.ha.status[0] == "off"

    @pytest.mark.asyncio
    async def test_a_failed_refresh_keeps_following_the_last_list_for_a_while(self, tmp_path):
        rig = Rig(tmp_path, _load(window_event_match="kim"), {},
                  events=[(_at(20), _at(600))])
        await rig.tick(0)
        assert rig.plug == "off"
        rig.events = RuntimeError("calendar integration down")
        await rig.tick(21)  # the cached timetable still says: started a minute ago
        assert rig.plug == "on"
        await rig.tick(29)
        assert rig.ha.status[0] == "on"
        # retried once a minute at most, never more often
        assert all(b - a >= 60 for a, b in zip(rig.fetches, rig.fetches[1:], strict=False))


class TestUnknownGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["unavailable", "unknown", None, RuntimeError("HA down")])
    async def test_an_unreadable_gate_holds_a_warm_room_warm(self, tmp_path, bad):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        rig.ha.states[CAL] = bad
        for minute in (1, 10, 29):
            await rig.tick(minute, plan={"kim_heater": 0.0})
        assert rig.writes == [("switch", "turn_on", PLUG)]  # nothing since
        assert rig.plug == "on"
        state, attrs = rig.ha.status
        assert state == "gate_unknown" and attrs["window_active"] is None

    @pytest.mark.asyncio
    async def test_an_unreadable_gate_does_not_heat_an_empty_room(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "unavailable"})
        for minute in (0, 15, 29):
            await rig.tick(minute, plan={"kim_heater": 0.5})
        assert rig.writes == []

    @pytest.mark.asyncio
    async def test_after_the_hold_the_plan_slot_decides(self, tmp_path, caplog):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        rig.ha.states[CAL] = "unavailable"
        await rig.tick(1, plan={"kim_heater": 0.0})  # the hold starts here
        await rig.tick(30, plan={"kim_heater": 0.0})
        assert rig.plug == "on"  # 29 min held
        with caplog.at_level("WARNING"):
            await rig.tick(31, plan={"kim_heater": 0.0})
            await rig.tick(32, plan={"kim_heater": 0.0})
        assert rig.plug == "off"
        assert rig.ha.status[0] == "off"
        assert sum("following the plan slot" in r.message for r in caplog.records) == 1
        await rig.tick(40, plan={"kim_heater": 0.5})
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_after_the_hold_a_load_missing_from_the_plan_is_left_alone(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        rig.ha.states[CAL] = "unavailable"
        for minute in (1, 31, 45, 120):
            await rig.tick(minute, plan={})
        assert rig.writes == [("switch", "turn_on", PLUG)]
        assert rig.ha.status[0] == "gate_unknown"

    @pytest.mark.asyncio
    async def test_the_gate_coming_back_ends_the_episode(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "unavailable", PLUG: "on"})
        await rig.tick(0)
        rig.ha.states[CAL] = "off"
        await rig.tick(5)
        assert rig.plug == "off"
        # A new outage gets a full hold again, not the remainder of the old one.
        rig.ha.states[CAL] = "unavailable"
        await rig.tick(40, plan={"kim_heater": 0.5})
        await rig.tick(69, plan={"kim_heater": 0.5})
        assert rig.plug == "off"

    @pytest.mark.asyncio
    async def test_a_match_gate_with_no_event_list_is_unknown(self, tmp_path):
        rig = Rig(tmp_path, _load(window_event_match="kim"), {PLUG: "on"},
                  events=RuntimeError("down"))
        await rig.tick(0)
        assert rig.writes == [] and rig.ha.status[0] == "gate_unknown"

    @pytest.mark.asyncio
    async def test_an_event_list_too_old_to_trust_becomes_unknown(self, tmp_path):
        rig = Rig(tmp_path, _load(window_event_match="kim"), {},
                  events=[(_at(-10), _at(600))])
        await rig.tick(0)
        assert rig.plug == "on"
        rig.events = RuntimeError("down")
        await rig.tick(30)
        assert rig.ha.status[0] == "on"
        await rig.tick(31)
        assert rig.ha.status[0] == "gate_unknown"
        assert rig.plug == "on" and len(rig.writes) == 1


class TestPrecedence:
    """disabled -> control pause -> fuse shed -> override -> live window."""

    @pytest.mark.asyncio
    async def test_disabled_is_not_touched_or_published(self, tmp_path):
        rig = Rig(tmp_path, _load(enabled=False), {CAL: "on"})
        await rig.tick(0)
        assert rig.writes == [] and rig.ha.published == [] and rig.ha.reads == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("plug", ["on", "off"])
    async def test_pause_is_zero_writes_whatever_else_is_true(self, tmp_path, plug):
        """Paused beats the window, the override AND the fuse guard: hands off."""
        rig = Rig(tmp_path, _load(fuse_shed=True),
                  {PAUSE: "on", CAL: "on", OVERRIDE: "force_off", PLUG: plug})
        overloaded = {"phase_currents": {"l1": 30.0, "l2": 30.0, "l3": 30.0},
                      "fuse_budget_a": 25.0, "grid_w": 9000.0}
        for minute in range(3):
            await rig.tick(minute, plan={"kim_heater": 0.5}, ctx=overloaded)
        assert rig.writes == []
        assert rig.plug == plug
        assert PLUG not in rig.ha.reads  # not even looked at
        assert rig.ha.status[0] == "paused"

    @pytest.mark.asyncio
    async def test_control_resumes_after_a_pause_like_a_fresh_process(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)  # on, and the dwell clock starts
        rig.ha.states.update({PAUSE: "on", PLUG: "off"})  # a human switches it off
        await rig.tick(1)
        state, attrs = rig.ha.status
        assert state == "paused" and attrs["window_active"] is None  # not read while paused
        rig.ha.states[PAUSE] = "off"
        await rig.tick(2)  # no dwell owed for a switch we did not make
        assert rig.plug == "on" and len(rig.writes) == 2

    @pytest.mark.asyncio
    async def test_an_unreadable_pause_helper_is_not_a_pause(self, tmp_path):
        rig = Rig(tmp_path, _load(), {PAUSE: "unavailable", CAL: "on"})
        await rig.tick(0)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_fuse_shed_outranks_the_window_and_force_on(self, tmp_path):
        rig = Rig(tmp_path, _load(fuse_shed=True, phase_map=("l2",)),
                  {CAL: "on", OVERRIDE: "force_on", PLUG: "on"})
        overloaded = {"phase_currents": {"l1": 10.0, "l2": 27.0, "l3": 10.0},
                      "fuse_budget_a": 25.0, "grid_w": 9000.0}
        await rig.tick(0, ctx=overloaded)
        assert rig.plug == "off" and rig.ha.status[0] == "fuse_shed"
        calm = {**overloaded, "phase_currents": {"l1": 10.0, "l2": 12.0, "l3": 10.0}}
        await rig.tick(1, ctx=calm)
        assert rig.plug == "on" and rig.ha.status[0] == "override_on"

    @pytest.mark.asyncio
    async def test_fuse_shed_needs_opting_in(self, tmp_path):
        rig = Rig(tmp_path, _load(fuse_shed=False), {CAL: "on"})
        overloaded = {"phase_currents": {"l1": 30.0, "l2": 30.0, "l3": 30.0},
                      "fuse_budget_a": 25.0, "grid_w": 9000.0}
        await rig.tick(0, ctx=overloaded)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_force_on_outside_a_window_and_force_off_inside_one(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "off", OVERRIDE: "force_on"})
        await rig.tick(0)
        assert rig.plug == "on" and rig.ha.status[0] == "override_on"
        rig.ha.states.update({CAL: "on", OVERRIDE: "force_off"})
        await rig.tick(1)
        assert rig.plug == "off" and rig.ha.status[0] == "override_off"
        # Back to auto: no dwell is owed for a switch the human asked for.
        rig.ha.states[OVERRIDE] = "auto"
        await rig.tick(2)
        assert rig.plug == "on" and rig.ha.status[0] == "on"

    @pytest.mark.asyncio
    async def test_override_does_not_need_a_readable_gate(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "unavailable", OVERRIDE: "force_on"})
        await rig.tick(0)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_classic_loads_next_to_a_heater_still_follow_the_plan(self, tmp_path):
        pump = CyclicLoadConfig(id="poolpump", switch_entity="switch.poolpump", power_kw=0.26)
        rig = Rig(tmp_path, _load(), {CAL: "on", "switch.poolpump": "off"})
        rig.engine.config.cyclic_loads.insert(0, pump)
        await rig.tick(0, plan={"poolpump": 0.26, "kim_heater": 0.0})
        assert rig.ha.states["switch.poolpump"] == "on" and rig.plug == "on"
        await rig.tick(1, plan={"kim_heater": 0.0})  # pump absent: left alone
        assert rig.ha.states["switch.poolpump"] == "on"
        await rig.tick(2, plan={"poolpump": 0.0, "kim_heater": 0.0})
        assert rig.ha.states["switch.poolpump"] == "off" and rig.plug == "on"


def _temp(rig: Rig, value: Any, *, age_min: float = 1.0, minutes: float = 0.0) -> None:
    rig.ha.states[TEMP] = value
    stamp = datetime.fromtimestamp(T0 + (minutes - age_min) * MIN, UTC)
    rig.ha.attrs[TEMP] = {"last_reported": stamp.isoformat()}


THERMO = {"temp_sensor": TEMP, "target_temp_c": 20.0, "hysteresis_c": 0.5}


class TestThermostat:
    def test_hysteresis_band(self):
        assert wl.thermostat_wants_heat(19.5, 20.0, 0.5, False) is True
        assert wl.thermostat_wants_heat(19.6, 20.0, 0.5, False) is False
        assert wl.thermostat_wants_heat(19.6, 20.0, 0.5, True) is True
        assert wl.thermostat_wants_heat(20.4, 20.0, 0.5, True) is True
        assert wl.thermostat_wants_heat(20.5, 20.0, 0.5, True) is False
        assert wl.thermostat_wants_heat(19.9, 20.0, 0.5, None) is True
        assert wl.thermostat_wants_heat(20.1, 20.0, 0.5, None) is False

    @pytest.mark.asyncio
    async def test_inside_a_window_the_plug_follows_the_room(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO), {CAL: "on"})
        _temp(rig, "18.2")
        await rig.tick(0)
        assert rig.plug == "on"
        state, attrs = rig.ha.status
        assert (state, attrs["room_temp"], attrs["target_temp"]) == ("on", 18.2, 20.0)
        _temp(rig, "20.1", minutes=10)  # inside the band: unchanged
        await rig.tick(10)
        assert rig.plug == "on"
        _temp(rig, "20.6", minutes=20)
        await rig.tick(20)
        assert rig.plug == "off" and rig.ha.status[0] == "temp_hold"
        _temp(rig, "19.8", minutes=30)  # inside the band again: stays off
        await rig.tick(30)
        assert rig.plug == "off"
        _temp(rig, "19.4", minutes=40)
        await rig.tick(40)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_outside_a_window_the_thermostat_does_not_heat(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO), {CAL: "off"})
        _temp(rig, "12.0")
        await rig.tick(0)
        assert rig.plug == "off" and rig.ha.status[0] == "off"
        assert TEMP not in rig.ha.reads  # no target outside => not even read

    @pytest.mark.asyncio
    async def test_setback_keeps_the_room_from_going_cold_outside_a_window(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO, setback_temp_c=15.0), {CAL: "off"})
        _temp(rig, "14.2")
        await rig.tick(0)
        assert rig.plug == "on"
        state, attrs = rig.ha.status
        assert state == "setback" and attrs["target_temp"] == 15.0
        _temp(rig, "15.6", minutes=10)
        await rig.tick(10)
        assert rig.plug == "off" and rig.ha.status[0] == "off"

    @pytest.mark.asyncio
    async def test_setback_alone_without_a_comfort_target(self, tmp_path):
        rig = Rig(tmp_path, _load(temp_sensor=TEMP, setback_temp_c=15.0), {CAL: "on"})
        _temp(rig, "25.0")
        await rig.tick(0)
        assert rig.plug == "on"  # in a window with no target: plain window

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reading", ["unknown", "unavailable", None, "nan", "warm"])
    async def test_a_dead_sensor_falls_back_to_plain_window_control(self, tmp_path, reading):
        """Both room sensors read "unknown" today: the heater must simply follow
        the calendar, inside AND outside the window."""
        rig = Rig(tmp_path, _load(**THERMO, setback_temp_c=15.0), {CAL: "on"})
        _temp(rig, reading)
        await rig.tick(0)
        assert rig.plug == "on" and rig.ha.status[0] == "on"
        assert rig.ha.status[1]["room_temp"] is None
        rig.ha.states[CAL] = "off"
        await rig.tick(10)
        assert rig.plug == "off" and rig.ha.status[0] == "off"

    @pytest.mark.asyncio
    async def test_a_stale_reading_is_a_missing_reading(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO, temp_max_age_minutes=30), {CAL: "on", PLUG: "on"})
        _temp(rig, "23.0", age_min=29)
        await rig.tick(0)
        assert rig.plug == "off"  # fresh and too warm
        _temp(rig, "23.0", age_min=31, minutes=10)
        await rig.tick(10)
        assert rig.plug == "on"  # stale: the window decides
        assert "stale" in rig.ha.status[1]["reason"]

    @pytest.mark.asyncio
    async def test_a_reading_without_any_timestamp_is_not_trusted(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO), {CAL: "on", TEMP: "23.0"})
        await rig.tick(0)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_a_noisy_sensor_cannot_chatter_the_relay(self, tmp_path):
        rig = Rig(tmp_path, _load(**THERMO), {CAL: "on"})
        for minute in range(12):
            _temp(rig, "19.0" if minute % 2 == 0 else "21.0", minutes=minute)
            await rig.tick(minute)
        # on at 0, off at 5 (the first "too warm" reading 5 min later), on at 10.
        assert [c[1] for c in rig.writes] == ["turn_on", "turn_off", "turn_on"]
        assert "switched under 5 min ago" in rig.ha.published[1][2]["reason"]

    @pytest.mark.asyncio
    async def test_dwell_also_spares_the_relay_from_a_flapping_gate(self, tmp_path):
        gate = "binary_sensor.flaky"
        rig = Rig(tmp_path, _load(window_entities=[gate]), {gate: "on"})
        for minute in range(5):
            rig.ha.states[gate] = "on" if minute % 2 == 0 else "off"
            await rig.tick(minute)
        assert rig.writes == [("switch", "turn_on", PLUG)]


class TestRestart:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("calendar", "plug", "writes"),
        [("on", "on", 0), ("on", "off", 1), ("off", "on", 1), ("off", "off", 0)],
    )
    async def test_first_tick_reaches_the_right_state_at_once(
        self, tmp_path, calendar, plug, writes
    ):
        """A fresh process has no dwell clock and no memory: one write at most,
        none when the plug is already right, and nothing after that."""
        rig = Rig(tmp_path, _load(preheat_minutes=30), {CAL: calendar, PLUG: plug})
        for minute in range(4):
            await rig.tick(minute)
        assert len(rig.writes) == writes
        assert rig.plug == calendar

    @pytest.mark.asyncio
    @pytest.mark.parametrize("plug", ["on", "off"])
    async def test_inside_the_thermostat_band_a_fresh_process_keeps_the_relay(
        self, tmp_path, plug
    ):
        rig = Rig(tmp_path, _load(**THERMO), {CAL: "on", PLUG: plug})
        _temp(rig, "20.2")
        await rig.tick(0)
        assert rig.writes == [] and rig.plug == plug

    @pytest.mark.asyncio
    async def test_a_restart_during_an_outage_holds_first(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "unavailable", PLUG: "on"})
        await rig.tick(0, plan={"kim_heater": 0.0})
        assert rig.writes == []


class TestWrites:
    @pytest.mark.asyncio
    async def test_shadow_mode_is_zero_writes(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"}, shadow=True)
        for minute in range(3):
            results = await rig.tick(minute)
        assert rig.writes == [] and rig.plug == "off"
        assert results[0].skipped and "SHADOW" in results[0].message
        state, attrs = rig.ha.status  # the status sensor still says what it would do
        assert state == "on" and attrs["shadow_mode"] is True

    @pytest.mark.asyncio
    async def test_shadow_mode_override_and_fuse_are_zero_writes_too(self, tmp_path):
        rig = Rig(tmp_path, _load(fuse_shed=True), {CAL: "off", OVERRIDE: "force_on"},
                  shadow=True)
        await rig.tick(0)
        rig.ha.states[PLUG] = "on"
        overloaded = {"phase_currents": {"l1": 30.0, "l2": 30.0, "l3": 30.0},
                      "fuse_budget_a": 25.0, "grid_w": 9000.0}
        await rig.tick(1, ctx=overloaded)
        assert rig.writes == []

    @pytest.mark.asyncio
    async def test_a_failed_service_call_is_a_failed_result_and_isolated(self, tmp_path):
        from executor.actions import HACallError

        rig = Rig(tmp_path, _load(), {CAL: "on"})
        rig.ha.service_error = HACallError(message="boom")
        (result,) = await rig.tick(0)
        assert result.success is False and not result.skipped
        assert result.action_type == "cyclic:kim_heater"
        assert result.entity_id == PLUG

    @pytest.mark.asyncio
    async def test_a_plug_that_ignores_us_is_escalated_once_and_recovers(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        rig.ha.plug_obeys = False
        notify = rig.engine.dispatcher.notify_unverified
        for minute in range(2):
            await rig.tick(minute)
        notify.assert_not_awaited()
        for minute in range(2, 6):
            await rig.tick(minute)
        assert notify.await_count == 1
        assert notify.await_args.args[0] == "Kims element"
        assert len(rig.writes) == 6  # it keeps trying every tick
        rig.ha.plug_obeys = True
        await rig.tick(6)
        assert notify.await_count == 2
        assert notify.await_args.kwargs == {"recovered": True}

    @pytest.mark.asyncio
    async def test_steady_state_costs_four_reads_and_no_publish(self, tmp_path):
        """Per tick: pause helper, override, calendar state, and the switch read
        inside the writer. No event fetch, no status publish while nothing changes."""
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        rig.ha.reads.clear()
        published = len(rig.ha.published)
        fetches = len(rig.fetches)
        await rig.tick(1)
        assert sorted(rig.ha.reads) == sorted([PAUSE, OVERRIDE, CAL, PLUG])
        assert len(rig.ha.published) == published and len(rig.fetches) == fetches

    @pytest.mark.asyncio
    async def test_status_is_republished_periodically_even_when_unchanged(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        await rig.tick(0)
        await rig.tick(9)
        assert len(rig.ha.published) == 1
        await rig.tick(10)
        assert len(rig.ha.published) == 2

    @pytest.mark.asyncio
    async def test_a_failing_status_publish_does_not_stop_the_plug(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"})
        rig.ha.set_state = AsyncMock(side_effect=RuntimeError("HA down"))  # type: ignore[method-assign]
        results = await rig.tick(0)
        assert rig.plug == "on" and all(r.success for r in results)


class TestStatusSensor:
    @pytest.mark.asyncio
    async def test_entity_id_and_attributes(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "off"}, events=[(_at(120), _at(900))])
        await rig.tick(0)
        entity, state, attrs = rig.ha.published[-1]
        assert entity == STATUS and state == "off"
        assert attrs == {
            "friendly_name": "Kims element status",
            "icon": "mdi:radiator",
            "load_id": "kim_heater",
            "window_active": False,
            "next_window_start": "2026-11-13T15:00+01:00",
            "next_window_end": "2026-11-14T04:00+01:00",
            "room_temp": None,
            "target_temp": None,
            "reason": "no window active",
            "shadow_mode": False,
            "price_now": None,
            "early_start_threshold": None,
            "peak_off_threshold": None,
            "early_start_latched": False,
            "peak_off_since": None,
            "heat_since": None,
        }

    def test_ids_are_slugged(self):
        assert wl.status_entity_id("Kim's heater") == "sensor.darkstar_kim_s_heater_state"


EARLY = {"early_start_hours": 14}
PEAK = {"peak_off_price_percentile": 85}
OWNER = {**EARLY, **PEAK}
HOURS = 60.0


def _ruled(**keys: Any) -> CyclicLoadConfig:
    from executor.window_price import PriceRules

    rule_keys = {k: keys.pop(k) for k in list(keys) if k in PriceRules.__dataclass_fields__}
    return _load(price_rules=PriceRules.from_mapping(rule_keys), **keys)


class TestPriceRulesLive:
    """Cheap early start and peak off, live. The decision itself is pinned down in
    test_window_price.py; here: the wiring to the executor's price source, the
    latch and clocks across ticks, the dwell, restarts, and the status sensor."""

    @pytest.mark.asyncio
    async def test_a_load_without_the_keys_never_asks_for_a_price(self, tmp_path):
        rig = Rig(tmp_path, _load(), {CAL: "on"}, events=[(_at(-60), _at(600))])
        rig.set_price(99.0)
        await rig.tick(0)
        assert rig.plug == "on" and rig.ha.status[0] == "on"
        rig.engine._current_import_price.assert_not_awaited()
        rig.engine._price_window.assert_not_awaited()
        attrs = rig.ha.status[1]
        assert attrs["price_now"] is None and attrs["peak_off_threshold"] is None

    @pytest.mark.asyncio
    async def test_early_start_latches_on_a_cheap_price_and_stays_on(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**EARLY), {}, events=[(_at(10 * HOURS), _at(99 * HOURS))])
        rig.set_price(1.0)  # above the 30th percentile (0.8)
        await rig.tick(0)
        assert rig.plug == "off" and rig.ha.status[0] == "off"
        rig.set_price(0.8)
        await rig.tick(60)
        state, attrs = rig.ha.status
        assert rig.plug == "on" and state == "early_start"
        assert (attrs["price_now"], attrs["early_start_threshold"]) == (0.8, 0.8)
        assert attrs["peak_off_threshold"] is None  # that rule is off
        assert attrs["early_start_latched"] is True and attrs["window_active"] is False
        rig.set_price(2.2)  # dear again: latched, stays on
        await rig.tick(120)
        assert rig.plug == "on" and rig.ha.status[0] == "early_start"
        rig.ha.states[CAL] = "on"  # the window begins: no gap, no extra write
        await rig.tick(600)
        assert rig.ha.status[0] == "on" and len(rig.writes) == 1

    @pytest.mark.asyncio
    async def test_early_start_only_inside_its_span(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**EARLY), {}, events=[(_at(20 * HOURS), _at(99 * HOURS))])
        rig.set_price(0.2)
        await rig.tick(0)  # 20 h before
        assert rig.plug == "off"
        await rig.tick(6 * HOURS)  # 14 h before
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_the_span_counts_from_when_preheat_would_switch_on(self, tmp_path):
        rig = Rig(tmp_path, _ruled(early_start_hours=2, preheat_minutes=60), {},
                  events=[(_at(4 * HOURS), _at(99 * HOURS))])
        rig.set_price(0.2)
        await rig.tick(59)  # 3 h 01 before the window: preheat at -1 h, span from -3 h
        assert rig.plug == "off"
        await rig.tick(60)
        assert rig.plug == "on" and rig.ha.status[0] == "early_start"
        rig.set_price(2.2)
        await rig.tick(3 * HOURS)  # preheat takes over, still on
        assert rig.plug == "on" and rig.ha.status[0] == "preheat"

    @pytest.mark.asyncio
    async def test_a_non_calendar_gate_has_no_early_start(self, tmp_path):
        gate = "input_boolean.kim_here"
        rig = Rig(tmp_path, _ruled(window_entities=[gate], **EARLY), {gate: "off"})
        rig.set_price(0.2)
        await rig.tick(0)
        assert rig.plug == "off"

    @pytest.mark.asyncio
    async def test_an_unknown_price_never_starts_early_and_never_switches_off(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**OWNER), {}, events=[(_at(HOURS), _at(99 * HOURS))])
        rig.set_price(None)
        await rig.tick(0)
        assert rig.plug == "off"
        rig.ha.states[CAL] = "on"
        for minute in (60, 200, 400):
            await rig.tick(minute)
            assert rig.plug == "on" and rig.ha.status[0] == "on"
        assert rig.ha.status[1]["price_now"] is None
        # The price source failing outright is the same thing.
        rig.engine._current_import_price.side_effect = RuntimeError("nordpool down")
        await rig.tick(460)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_a_price_window_without_prices_disables_the_rules(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "off"})
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        rig.price_window = []
        rig.set_price(99.0)
        await rig.tick(10)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_peak_off_inside_a_window_with_its_time_bounds(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "off"})
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        rig.set_price(1.0)
        await rig.tick(10)
        assert rig.plug == "on"
        rig.set_price(1.9)  # the 85th percentile
        await rig.tick(100)  # on for 90 min: not yet
        assert rig.plug == "on" and "on for less than 120 min" in rig.ha.status[1]["reason"]
        await rig.tick(130)  # 120 min on
        state, attrs = rig.ha.status
        assert rig.plug == "off" and state == "peak_off"
        assert (attrs["price_now"], attrs["peak_off_threshold"]) == (1.9, 1.9)
        assert attrs["peak_off_since"] == "2026-11-13T15:10+01:00"
        assert attrs["window_active"] is True
        await rig.tick(309)  # 179 min off
        assert rig.plug == "off"
        await rig.tick(310)  # 180 min: must run again, although still dear
        assert rig.plug == "on" and rig.ha.status[0] == "on"
        await rig.tick(429)
        assert rig.plug == "on"
        await rig.tick(430)  # 120 min on: the next stretch may begin
        assert rig.plug == "off"
        rig.set_price(1.0)
        await rig.tick(440)
        assert rig.plug == "on"
        assert [c[1] for c in rig.writes] == ["turn_on", "turn_off", "turn_on", "turn_off",
                                              "turn_on"]

    @pytest.mark.asyncio
    async def test_a_window_that_opens_at_a_peak_waits_it_out(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "off"})
        rig.set_price(2.2)
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        await rig.tick(10)
        assert rig.plug == "off" and rig.ha.status[0] == "peak_off" and rig.writes == []
        await rig.tick(190)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_a_peak_interrupts_an_early_start_and_the_latch_survives(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**OWNER), {}, events=[(_at(12 * HOURS), _at(99 * HOURS))])
        rig.set_price(0.4)
        await rig.tick(0)
        assert rig.ha.status[0] == "early_start"
        rig.set_price(2.0)
        await rig.tick(120)
        assert rig.plug == "off" and rig.ha.status[0] == "peak_off"
        assert rig.ha.status[1]["early_start_latched"] is True
        rig.set_price(1.4)  # no longer a peak, not cheap either: latched, back on
        await rig.tick(180)
        assert rig.plug == "on" and rig.ha.status[0] == "early_start"

    @pytest.mark.asyncio
    async def test_the_dwell_applies_to_price_driven_switches(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "off"})
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        rig.set_price(1.0)
        await rig.tick(10)
        rig.set_price(2.2)
        await rig.tick(130)
        assert rig.plug == "off"
        rig.set_price(1.0)  # the peak is over a minute later
        await rig.tick(131)
        assert rig.plug == "off" and "switched under 5 min ago" in rig.ha.status[1]["reason"]
        await rig.tick(135)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_fuse_shed_pause_and_override_still_outrank_the_price_rules(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**OWNER), {OVERRIDE: "force_on", CAL: "on"})
        rig.set_price(2.2)
        await rig.tick(0)
        assert rig.plug == "on" and rig.ha.status[0] == "override_on"
        rig.ha.states.update({OVERRIDE: "auto", PAUSE: "on"})
        await rig.tick(1)
        assert rig.ha.status[0] == "paused" and len(rig.writes) == 1
        rig.engine._current_import_price.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_price_is_read_once_per_hour_not_once_per_tick(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "on"})
        rig.set_price(1.0)
        for minute in range(0, 60, 5):
            await rig.tick(minute)
        assert rig.engine._current_import_price.await_count == 1
        await rig.tick(60)
        assert rig.engine._current_import_price.await_count == 2
        # ...but an unknown price is asked for again every tick.
        rig.set_price(None)
        for minute in (120, 121, 122):
            await rig.tick(minute)
        assert rig.engine._current_import_price.await_count == 5


class TestPriceRulesRestart:
    """Nothing is persisted. A fresh process (and a load just un-paused) instead
    starts from a SAFE reading of the plug: the min-on clock starts now, and a plug
    that is on inside the early-start span is taken to be latched."""

    @pytest.mark.asyncio
    async def test_no_fresh_off_stretch_right_after_a_restart(self, tmp_path):
        """In a window, at a peak, plug on (it may have just come back from a
        peak-off): it must run the full min-on time first."""
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "on", PLUG: "on"})
        rig.set_price(2.2)
        for minute in (0, 60, 119):
            await rig.tick(minute)
        assert rig.writes == []
        await rig.tick(120)
        assert rig.plug == "off" and rig.ha.status[0] == "peak_off"

    @pytest.mark.asyncio
    async def test_a_restart_during_a_peak_off_ends_it(self, tmp_path):
        """The off-clock is gone, so the stretch cannot be bounded any more: on."""
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "on", PLUG: "off"})
        rig.set_price(2.2)
        await rig.tick(0)
        assert rig.plug == "on"
        await rig.tick(60)
        assert rig.plug == "on" and len(rig.writes) == 1

    @pytest.mark.asyncio
    async def test_a_restart_keeps_an_early_start_that_is_running(self, tmp_path):
        """Started on last night's cheap price; restarted at noon when it is dear.
        Switching it off now would throw the stored heat away."""
        rig = Rig(tmp_path, _ruled(**EARLY), {PLUG: "on"},
                  events=[(_at(3 * HOURS), _at(99 * HOURS))])
        rig.set_price(2.0)
        for minute in (0, 1, 30):
            await rig.tick(minute)
        assert rig.writes == [] and rig.plug == "on"
        assert rig.ha.status[0] == "early_start"

    @pytest.mark.asyncio
    async def test_a_restart_does_not_invent_an_early_start(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**EARLY), {PLUG: "off"},
                  events=[(_at(3 * HOURS), _at(99 * HOURS))])
        rig.set_price(2.0)
        await rig.tick(0)
        assert rig.writes == [] and rig.ha.status[0] == "off"

    @pytest.mark.asyncio
    async def test_a_plug_left_on_outside_the_span_is_not_a_latch(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**EARLY), {PLUG: "on"},
                  events=[(_at(20 * HOURS), _at(99 * HOURS))])
        rig.set_price(2.0)
        await rig.tick(0)
        assert rig.plug == "off"

    @pytest.mark.asyncio
    async def test_un_pausing_starts_like_a_fresh_process(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK), {CAL: "off"})
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        rig.set_price(1.0)
        await rig.tick(10)
        rig.ha.states[PAUSE] = "on"
        await rig.tick(200)
        rig.ha.states[PAUSE] = "off"
        rig.set_price(2.2)
        await rig.tick(201)  # on for 190 min by the old clock — which is forgotten
        assert rig.plug == "on"


class TestPeakOffThermostatGuard:
    """With a working thermostat there is no peak-off while the room is below
    target - hysteresis - margin (20 - 0.5 - 1.0 = 18.5)."""

    async def _window_opening_at_a_peak(self, tmp_path, temp: str) -> Rig:
        rig = Rig(tmp_path, _ruled(**PEAK, **THERMO), {CAL: "off"})
        rig.set_price(2.2)
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        _temp(rig, temp, minutes=10)
        await rig.tick(10)
        return rig

    @pytest.mark.asyncio
    async def test_a_cold_room_is_heated_through_the_peak(self, tmp_path):
        rig = await self._window_opening_at_a_peak(tmp_path, "18.4")
        assert rig.plug == "on" and rig.ha.status[0] == "on"

    @pytest.mark.asyncio
    async def test_a_room_above_the_margin_may_wait_out_the_peak(self, tmp_path):
        rig = await self._window_opening_at_a_peak(tmp_path, "18.5")
        assert rig.plug == "off" and rig.ha.status[0] == "peak_off"
        _temp(rig, "18.4", minutes=40)  # cools below the margin: the peak-off ends
        await rig.tick(40)
        assert rig.plug == "on" and "too cold" in rig.ha.status[1]["reason"]

    @pytest.mark.asyncio
    async def test_a_dead_sensor_leaves_only_the_time_bounds(self, tmp_path):
        rig = await self._window_opening_at_a_peak(tmp_path, "unknown")
        assert rig.plug == "off" and rig.ha.status[0] == "peak_off"
        await rig.tick(190)
        assert rig.plug == "on"

    @pytest.mark.asyncio
    async def test_the_setback_still_protects_the_room_during_a_peak_off(self, tmp_path):
        rig = Rig(tmp_path, _ruled(**PEAK, temp_sensor=TEMP, setback_temp_c=15.0),
                  {CAL: "off"})
        rig.set_price(2.2)
        _temp(rig, "16.0")
        await rig.tick(0)
        rig.ha.states[CAL] = "on"
        _temp(rig, "14.0", minutes=10)
        await rig.tick(10)
        assert rig.plug == "on" and rig.ha.status[0] == "setback"


class TestGateFromWindows:
    def test_dst_night_is_compared_by_instant(self):
        """02:30 happens twice on 25 October 2026. A window ending at the SECOND
        02:30 is still active at the first one."""
        import pytz

        tz = pytz.timezone("Europe/Stockholm")
        end = tz.localize(datetime(2026, 10, 25, 2, 30), is_dst=False)
        windows = [(end - timedelta(hours=6), end)]
        first = tz.localize(datetime(2026, 10, 25, 2, 30), is_dst=True)
        assert wl.gate_from_windows(windows, first, 0).active is True
        assert wl.gate_from_windows(windows, end - timedelta(minutes=1), 0).active is True
        assert wl.gate_from_windows(windows, end, 0).active is False

    def test_end_is_exclusive_and_the_next_window_is_reported(self):
        a = (_at(0), _at(60))
        b = (_at(120), _at(180))
        assert wl.gate_from_windows([a, b], _at(59), 0) == wl.Gate(True, False, *a)
        assert wl.gate_from_windows([a, b], _at(60), 30) == wl.Gate(False, False, *b)
        assert wl.gate_from_windows([a, b], _at(90), 30) == wl.Gate(False, True, *b)
        assert wl.gate_from_windows([a, b], _at(200), 30) == wl.Gate(False)


class TestConfig:
    def _load(self, tmp_path, cyclic_loads):
        import yaml

        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"executor": {"enabled": True}, "cyclic_loads": cyclic_loads})
        )
        return load_executor_config(str(path)).cyclic_loads

    def test_window_keys_are_parsed_with_their_defaults(self, tmp_path):
        (load,) = self._load(tmp_path, [{
            "id": "kim_heater", "mode": "window", "switch_entity": PLUG,
            "power_sensor": "sensor.kims_uttag_power", "power_kw": 1.0,
            "window_entities": CAL,
        }])
        assert load.mode == "window" and load.window_entities == [CAL]
        assert load.power_entity == "sensor.kims_uttag_power"
        assert (load.window_event_match, load.preheat_minutes) == (None, 0.0)
        assert (load.temp_sensor, load.target_temp_c, load.setback_temp_c) == (None, None, None)
        assert (load.hysteresis_c, load.temp_max_age_minutes) == (0.5, 30.0)

    def test_every_window_key_round_trips(self, tmp_path):
        (load,) = self._load(tmp_path, [{
            "id": "kim_heater", "mode": "Window", "switch_entity": PLUG,
            "window_entities": [CAL, "input_boolean.guest"], "window_event_match": "Kim",
            "preheat_minutes": 45, "temp_sensor": TEMP, "target_temp_c": 20,
            "hysteresis_c": -0.3, "temp_max_age_minutes": 20, "setback_temp_c": 15,
            "control_pause_entities": [PAUSE], "fuse_shed": True, "phase_map": ["L2"],
        }])
        assert load.mode == "window"
        assert load.window_entities == [CAL, "input_boolean.guest"]
        assert (load.window_event_match, load.preheat_minutes) == ("Kim", 45.0)
        assert (load.temp_sensor, load.target_temp_c, load.setback_temp_c) == (TEMP, 20.0, 15.0)
        assert (load.hysteresis_c, load.temp_max_age_minutes) == (0.3, 20.0)
        assert load.control_pause_entities == [PAUSE] and load.fuse_shed is True

    def test_price_rule_keys(self, tmp_path):
        from executor.window_price import PriceRules

        base = {"id": "h", "mode": "window", "switch_entity": PLUG, "window_entities": [CAL]}
        plain, ruled = self._load(tmp_path, [
            base,
            {**base, "id": "r", "early_start_hours": 14, "early_start_price_percentile": 30,
             "peak_off_price_percentile": 85, "peak_off_max_minutes": 180,
             "peak_off_min_on_minutes": 120, "price_window_hours": 48},
        ])
        assert plain.price_rules == PriceRules() and not plain.price_rules.enabled
        assert ruled.price_rules == PriceRules(
            early_start_hours=14.0, early_start_price_percentile=30.0,
            peak_off_price_percentile=85.0, peak_off_max_minutes=180.0,
            peak_off_min_on_minutes=120.0, peak_off_temp_margin_c=1.0,
            price_window_hours=48.0,
        )

    def test_an_existing_pump_config_is_a_daily_load(self, tmp_path):
        (load,) = self._load(tmp_path, [{
            "id": "poolpump", "switch_entity": "switch.poolpump", "power_kw": 0.26,
            "min_kwh_per_day": 0.78,
        }])
        assert load.mode == "daily" and load.window_entities == []

    def test_a_mistyped_mode_or_a_window_without_a_gate_is_skipped(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            loads = self._load(tmp_path, [
                {"id": "a", "switch_entity": "switch.a", "mode": "windows",
                 "window_entities": [CAL]},
                {"id": "b", "switch_entity": "switch.b", "mode": "window"},
                {"id": "c", "switch_entity": "switch.c"},
            ])
        assert [load.id for load in loads] == ["c"]
        assert "unknown mode 'windows'" in caplog.text
        assert "mode: window needs window_entities" in caplog.text
