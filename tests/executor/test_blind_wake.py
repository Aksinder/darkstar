"""Wake a car we cannot see when a deadline is coming.

2026-09-28 21:04 Home Assistant restarted with the Tesla asleep at 13 % and a
07:30 departure. Every white_betty_* entity read `unknown` and stayed that way:
Tesla Fleet has nothing to show until the car next comes online, and the only
thing that would ever wake it was a start command failing at 02:00 — the moment
charging was already due. The wake button's last press was three days old.

Rule: an hour into such a blackout, with the car (held as) plugged in and at home
and a deadline inside twelve hours, press wake once.
"""

from __future__ import annotations

import pytest

from executor.ev_surplus import ChargerState
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config

T0 = 1_790_000_000.0
HOUR = 3600.0
WAKE = ("button", "press", "button.tesla_wake", None)


class FakeHA:
    def __init__(self, states=None, attrs=None, fail=False):
        self.states = states or {}
        self.attrs = attrs or {}
        self.calls: list[tuple] = []
        self.fail = fail

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        if entity not in self.states and entity not in self.attrs:
            return None
        return {"state": self.states.get(entity), "attributes": self.attrs.get(entity, {})}

    async def call_service(self, domain, service, entity_id=None, data=None, **_kw):
        self.calls.append((domain, service, entity_id, data))
        if self.fail:
            raise RuntimeError("HTTP 500")
        return True


def _ctrl(**charger):
    base = {
        "id": "tesla",
        "priority": 0,
        "min_current_a": 5,
        "max_current_a": 16,
        "phases": 3,
        "switch_entity": "switch.tesla",
        "current_entity": "number.tesla_amps",
        "power_entity": "sensor.tesla_power",
        "plug_entity": "binary_sensor.tesla_plug",
        "soc_entity": "sensor.tesla_soc",
        "wake_entity": "button.tesla_wake",
    }
    base.update(charger)
    return EVSurplusController(
        parse_ev_surplus_config(
            {
                "ev_surplus": {
                    "enabled": True,
                    "pv_power_entity": "sensor.pv",
                    "grid_power_entity": "sensor.grid",
                    "battery_power_entity": "sensor.batt",
                    "battery_soc_entity": "sensor.soc",
                    "chargers": [base],
                }
            }
        )
    )


def _car(*, soc=None, plugged=True, at_home=True, deadline_hours=8.0):
    return ChargerState(
        id="tesla",
        plugged=plugged,
        at_home=at_home,
        enabled=True,
        current_power_w=0.0,
        max_current_a=16.0,
        min_current_a=5.0,
        soc_percent=soc,
        deadline_hours=deadline_hours,
    )


async def _tick(ctrl, ha, now_ts, shadow=False, **car):
    await ctrl._wake_blind_cars(ha, [_car(**car)], now_ts, shadow)


class TestTheIncident:
    @pytest.mark.asyncio
    async def test_an_hour_blind_with_a_deadline_wakes_the_car(self):
        ctrl, ha = _ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)  # 21:15 — goes blind
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + 0.5 * HOUR)  # too early to spend a wake
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + HOUR)  # 22:15
        assert ha.calls == [WAKE]

    @pytest.mark.asyncio
    async def test_it_does_not_hammer(self):
        ctrl, ha = _ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)
        for minute in range(60, 240, 1):
            await _tick(ctrl, ha, T0 + minute * 60.0, deadline_hours=11.0)
        assert ha.calls == [WAKE]  # one press in three hours
        await _tick(ctrl, ha, T0 + 4 * HOUR + 1, deadline_hours=7.0)
        assert ha.calls == [WAKE, WAKE]

    @pytest.mark.asyncio
    async def test_a_reading_ends_the_episode(self):
        ctrl, ha = _ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + 0.9 * HOUR, soc=13.0)  # the car spoke
        await _tick(ctrl, ha, T0 + 1.0 * HOUR)  # blind again: clock restarts
        await _tick(ctrl, ha, T0 + 1.5 * HOUR)
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + 2.0 * HOUR)
        assert ha.calls == [WAKE]


class TestItStaysNarrow:
    @pytest.mark.parametrize(
        "car",
        [
            {"soc": 40.0},  # readable
            {"plugged": False},  # not in its cable
            {"at_home": False},  # somewhere else
            {"deadline_hours": None},  # nothing to be ready for (also: vacation)
            {"deadline_hours": 0.0},
            {"deadline_hours": 13.0},  # too far off to spend a wake on
        ],
    )
    @pytest.mark.asyncio
    async def test_no_wake(self, car):
        ctrl, ha = _ctrl(), FakeHA()
        await _tick(ctrl, ha, T0, **car)
        await _tick(ctrl, ha, T0 + 2 * HOUR, **car)
        assert ha.calls == []

    @pytest.mark.asyncio
    async def test_no_wake_button_no_wake(self):
        ctrl, ha = _ctrl(wake_entity=None), FakeHA()
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + 2 * HOUR)
        assert ha.calls == []

    @pytest.mark.asyncio
    async def test_a_car_without_an_soc_sensor_is_not_blind(self):
        ctrl, ha = _ctrl(soc_entity=None), FakeHA()
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + 2 * HOUR)
        assert ha.calls == []

    @pytest.mark.parametrize("how", ["run", "charger"])
    @pytest.mark.asyncio
    async def test_shadow_mode_presses_nothing(self, how):
        ctrl = _ctrl(shadow=True) if how == "charger" else _ctrl()
        ha = FakeHA()
        await _tick(ctrl, ha, T0, shadow=(how == "run"))
        await _tick(ctrl, ha, T0 + 2 * HOUR, shadow=(how == "run"))
        assert ha.calls == []

    @pytest.mark.asyncio
    async def test_it_shares_the_failed_start_cooldown(self):
        """The failed-start path just pressed wake: do not press again on its heels."""
        ctrl, ha = _ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)
        ctrl._last_wake_ts["tesla"] = T0 + HOUR - 60.0
        await _tick(ctrl, ha, T0 + HOUR)
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + HOUR + 300.0)
        assert ha.calls == [WAKE]

    @pytest.mark.asyncio
    async def test_a_failed_press_is_swallowed_and_still_paced(self):
        ctrl, ha = _ctrl(), FakeHA(fail=True)
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + HOUR)
        await _tick(ctrl, ha, T0 + HOUR + 60.0)
        assert ha.calls == [WAKE]


class TestWiredIntoTheRunLoop:
    @pytest.mark.asyncio
    async def test_run_wakes_a_blind_car(self):
        ctrl = _ctrl(departure_entity="input_datetime.tesla_departure")
        states = {
            "sensor.pv": "0",
            "sensor.grid": "1000",
            "sensor.batt": "0",
            "sensor.soc": "50",
            "sensor.tesla_power": "0",
            "binary_sensor.tesla_plug": "on",
            # sensor.tesla_soc absent: the blackout
        }
        attrs = {"input_datetime.tesla_departure": {"timestamp": T0 + 9 * HOUR}}
        ha = FakeHA(states, attrs)
        await ctrl.run(ha, T0)
        assert WAKE not in ha.calls
        await ctrl.run(ha, T0 + HOUR)
        assert ha.calls.count(WAKE) == 1
