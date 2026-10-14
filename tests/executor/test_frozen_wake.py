"""Wake a sleeping car whose frozen snapshot hides that it came home.

2026-10-14: last poll 19:37 on the road, asleep 19:47 in its cable at home. Tesla
Fleet republished "away, no cable, 21 %" all night; the 07:30 floor of 40 % was
never attempted.
"""

from __future__ import annotations

import pytest

from executor.ev_surplus import ChargerState
from tests.executor.test_blind_wake import HOUR, T0, WAKE, FakeHA, _ctrl


def _car(*, soc=21.0, floor=40.0, plugged=False, at_home=False, deadline_hours=11.0):
    return ChargerState(
        id="tesla", plugged=plugged, at_home=at_home, enabled=True,
        current_power_w=0.0, max_current_a=16.0, min_current_a=5.0,
        soc_percent=soc, floor_soc_percent=floor, deadline_hours=deadline_hours,
    )


async def _tick(ctrl, ha, now_ts, online=False, shadow=False, **car):
    ctrl._online["tesla"] = online
    await ctrl._wake_frozen_cars(ha, [_car(**car)], now_ts, shadow)


def _frozen_ctrl(**over):
    return _ctrl(online_entity="binary_sensor.tesla_online", **over)


class TestTheIncident:
    @pytest.mark.asyncio
    async def test_half_an_hour_asleep_on_an_away_snapshot_wakes_it(self):
        ctrl, ha = _frozen_ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)  # 19:47 asleep
        await _tick(ctrl, ha, T0 + 0.4 * HOUR)
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + 0.5 * HOUR)
        assert ha.calls == [WAKE]

    @pytest.mark.asyncio
    async def test_a_car_that_really_is_away_is_not_hammered(self):
        ctrl, ha = _frozen_ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)
        for minute in range(30, 150):
            await _tick(ctrl, ha, T0 + minute * 60.0)
        assert ha.calls == [WAKE]
        await _tick(ctrl, ha, T0 + 2.5 * HOUR + 1, deadline_hours=8.0)
        assert ha.calls == [WAKE, WAKE]

    @pytest.mark.asyncio
    async def test_coming_online_restarts_the_clock(self):
        ctrl, ha = _frozen_ctrl(), FakeHA()
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + 0.3 * HOUR, online=True)
        await _tick(ctrl, ha, T0 + 0.4 * HOUR)
        await _tick(ctrl, ha, T0 + 0.8 * HOUR)
        assert ha.calls == []
        await _tick(ctrl, ha, T0 + 0.9 * HOUR)
        assert ha.calls == [WAKE]


class TestItStaysNarrow:
    @pytest.mark.parametrize(
        "car",
        [
            {"soc": 40.0},  # floor already met
            {"soc": None},  # unreadable: the blind wake's case
            {"floor": None},  # no guarantee to keep
            {"plugged": True, "at_home": True},  # visible: a failed start wakes it
            {"deadline_hours": None},  # nothing to be ready for (also: vacation)
            {"deadline_hours": 13.0},
        ],
    )
    @pytest.mark.asyncio
    async def test_no_press(self, car):
        ctrl, ha = _frozen_ctrl(), FakeHA()
        await _tick(ctrl, ha, T0, **car)
        await _tick(ctrl, ha, T0 + HOUR, **car)
        assert ha.calls == []

    @pytest.mark.asyncio
    async def test_needs_an_online_sensor_and_a_live_executor(self):
        ctrl, ha = _ctrl(), FakeHA()  # no online_entity configured
        await _tick(ctrl, ha, T0)
        await _tick(ctrl, ha, T0 + HOUR)
        ctrl2, ha2 = _frozen_ctrl(), FakeHA()
        await _tick(ctrl2, ha2, T0, shadow=True)
        await _tick(ctrl2, ha2, T0 + HOUR, shadow=True)
        assert ha.calls == [] and ha2.calls == []

    @pytest.mark.asyncio
    async def test_unplugged_at_home_also_counts(self):
        ctrl, ha = _frozen_ctrl(), FakeHA()
        await _tick(ctrl, ha, T0, at_home=True)
        await _tick(ctrl, ha, T0 + HOUR, at_home=True)
        assert ha.calls == [WAKE]
