"""Verify before command, and a rejected start on a drawing car is an adoption.

2026-09-13 06:00 and 2026-09-14 03:15: the Tesla woke and started charging by itself.
HA's power sensor still read 0.0 from the poll ten minutes earlier; the servo's memory
said OFF; it sent switch.turn_on to a car already pulling 5-11 kW, the vehicle API
answered 500 three times, the backoff climbed to 480 s and the engine notified an
"EV charge failure" while the car charged. Also the servo-side plug/home hold, the
sibling of tests/test_ev_plug_hold.py.
"""

from __future__ import annotations

import pytest

from executor.ev_surplus import ChargerCommand
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import _cfg_dict, _states

NOW = 1_700_000_000.0
POWER = "sensor.tesla_power"
# 10 kW of export: enough for the FMB (3.7 kW, prio 0) AND a Tesla start (>= 3.45 kW).
SURPLUS = {"sensor.pv": "12000", "sensor.grid": "-10000"}


class RecordingHA:
    """get_state carries last_reported; a fresh read can be armed to return a new value."""

    def __init__(self, states, *, report_ts=None, fresh_power=None, reject_turn_on=False):
        self.states = dict(states)
        self.report_ts = report_ts
        self.fresh_power = fresh_power
        self.reject_turn_on = reject_turn_on
        self.calls: list[tuple] = []

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        if entity not in self.states:
            return None
        d = {"state": self.states[entity], "attributes": {}}
        if self.report_ts is not None and entity == POWER:
            from datetime import UTC, datetime
            d["last_reported"] = datetime.fromtimestamp(self.report_ts, UTC).isoformat()
            d["last_updated"] = d["last_reported"]
        return d

    async def call_service(self, domain, service, entity_id=None, data=None, **kw):
        self.calls.append((domain, service, entity_id))
        if (domain, service) == ("homeassistant", "update_entity") and self.fresh_power is not None:
            self.states[POWER] = str(self.fresh_power)
        if (domain, service) == ("switch", "turn_on") and self.reject_turn_on:
            raise RuntimeError("HTTP 500: charging")
        return True


def _ctrl():
    cfg = parse_ev_surplus_config(_cfg_dict())
    assert cfg is not None
    return EVSurplusController(cfg), cfg


def _tesla(cfg):
    return next(c for c in cfg.chargers if c.id == "tesla")


def _start_cmd():
    return ChargerCommand(id="tesla", switch_on=True, set_current_a=6.0, target_power_w=4140.0,
                          reason="test")


class TestVerifyBeforeCommand:
    @pytest.mark.asyncio
    async def test_stale_zero_is_refetched_and_a_drawing_car_is_adopted(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0"}), report_ts=NOW - 540.0, fresh_power=5000.0)
        ctrl._power_report_ts["tesla"] = NOW - 540.0
        await ctrl._actuate(ha, _tesla(cfg), _start_cmd(), NOW, False, drawing_w=0.0)
        assert ("homeassistant", "update_entity", POWER) in ha.calls
        assert ("switch", "turn_on", "switch.tesla") not in ha.calls
        assert ctrl._last_switch["tesla"] is True

    @pytest.mark.asyncio
    async def test_fresh_zero_is_trusted_and_the_start_goes_out(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0"}), report_ts=NOW - 30.0, fresh_power=5000.0)
        ctrl._power_report_ts["tesla"] = NOW - 30.0
        await ctrl._actuate(ha, _tesla(cfg), _start_cmd(), NOW, False, drawing_w=0.0)
        assert ("homeassistant", "update_entity", POWER) not in ha.calls
        assert ("switch", "turn_on", "switch.tesla") in ha.calls

    @pytest.mark.asyncio
    async def test_stale_zero_that_stays_zero_still_starts(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0"}), report_ts=NOW - 540.0, fresh_power=0.0)
        ctrl._power_report_ts["tesla"] = NOW - 540.0
        await ctrl._actuate(ha, _tesla(cfg), _start_cmd(), NOW, False, drawing_w=0.0)
        assert ("switch", "turn_on", "switch.tesla") in ha.calls

    @pytest.mark.asyncio
    async def test_shadow_never_refetches(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0"}), report_ts=NOW - 540.0, fresh_power=5000.0)
        ctrl._power_report_ts["tesla"] = NOW - 540.0
        await ctrl._actuate(ha, _tesla(cfg), _start_cmd(), NOW, True, drawing_w=0.0)
        assert not ha.calls


class TestRejectedStartOnADrawingCar:
    @pytest.mark.asyncio
    async def test_the_500_becomes_an_adoption_not_a_failure(self):
        """The 2026-09-14 03:17 shape: fresh-enough zero, turn_on rejected, car charging."""
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0", **SURPLUS}), report_ts=NOW - 30.0,
                         fresh_power=5000.0, reject_turn_on=True)
        ctrl._power_report_ts["tesla"] = NOW - 30.0
        # Drive the run-loop's actuation section directly via run().
        res = await ctrl.run(ha, NOW)
        assert res is not None
        assert "tesla" not in ctrl._act_fail
        assert ctrl._last_switch.get("tesla") is True

    @pytest.mark.asyncio
    async def test_a_genuinely_asleep_car_still_counts_a_failure(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0", **SURPLUS}), report_ts=NOW - 30.0,
                         fresh_power=0.0, reject_turn_on=True)
        ctrl._power_report_ts["tesla"] = NOW - 30.0
        await ctrl.run(ha, NOW)
        assert "tesla" in ctrl._act_fail


class TestServoPlugAndHomeHold:
    @pytest.mark.asyncio
    async def test_unknown_plug_holds_plugged(self):
        ctrl, cfg = _ctrl()
        c = _tesla(cfg)
        ha = RecordingHA(_states())
        st = await ctrl._read_charger(ha, c, NOW, vacation=False)
        assert st.plugged is True
        ha.states["binary_sensor.tesla_plug"] = "unknown"
        st = await ctrl._read_charger(ha, c, NOW + 60.0, vacation=False)
        assert st.plugged is True

    @pytest.mark.asyncio
    async def test_off_plug_still_unplugs(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{"binary_sensor.tesla_plug": "off"}))
        st = await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert st.plugged is False

    @pytest.mark.asyncio
    async def test_unknown_with_nothing_held_is_unplugged(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{"binary_sensor.tesla_plug": "unavailable"}))
        st = await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert st.plugged is False

    @pytest.mark.asyncio
    async def test_power_report_age_is_published(self):
        ctrl, cfg = _ctrl()
        ha = RecordingHA(_states(**{POWER: "0"}), report_ts=NOW - 420.0)
        await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert ctrl.last_power_report_age_s["tesla"] == pytest.approx(420.0)
