"""A satisfied thermostat must not hold the group's power budget.

The gate built on 2026-09-05 judges the COMMAND, not the power, and that was right: a
load the phase guard has just cut also reads 0 W, and it must keep its claim because it
resumes the instant the guard lets go.

It was also blind to a second way of reading 0 W. On 2026-09-06 the villavagn tank
latched a 120-minute committed block at 12:39, was drawing 1.16 W by 13:33, and held the
whole 2.3 kW group until 16:01 — through the cheapest hours of the day (2-8 ore) with
12 kW on the roof and the house battery already full. The spa began heating four minutes
after it let go. Queuing them was correct. Letting a satisfied thermostat hold the queue
was not.

    SHED      relay OPEN,   ~0 W  -> keeps its claim
    SATURATED relay CLOSED, ~0 W  -> releases it

Everything here is about telling those two apart, and about refusing to guess when the
evidence is missing.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from executor.actions import ActionDispatcher, HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    LoadGroupConfig,
    WaterHeaterDeviceConfig,
)
from executor.engine import ExecutorEngine

SPA_ENT = "input_number.spa_darkstar_target_temp"
TANK_ENT = "switch.villavagn_vvb"

RELEASE_S = 600.0


def _engine(*, relay: str = "on", release_after: float = RELEASE_S):
    """The live shape: spa (1.8 kW) + villavagn tank (1.6 kW) under a 2.3 kW cap."""
    cfg = ExecutorConfig(
        inverter=InverterConfig(),
        controller=ControllerConfig(),
        group_release_after_s=release_after,
        water_heater_devices=[
            WaterHeaterDeviceConfig(
                id="spa", target_entity=SPA_ENT, power_kw=1.8, surplus_boost=True
            ),
            WaterHeaterDeviceConfig(
                id="villavagn_tank",
                target_entity=TANK_ENT,
                power_kw=1.6,
                power_entity="sensor.villavagn_vvb_power",
            ),
        ],
        load_groups=[
            LoadGroupConfig(
                id="villavagn", max_power_kw=2.3, members=["spa", "villavagn_tank"]
            )
        ],
    )
    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(return_value=relay)
    dispatcher = ActionDispatcher(ha_client=ha, config=cfg, profile=None, shadow_mode=True)
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng.config = cfg
    eng.dispatcher = dispatcher
    eng.ha_client = ha
    eng._group_idle_since = {}
    by = {d.id: d for d in cfg.water_heater_devices}
    dispatcher._last_water_cmd[TANK_ENT] = "on"  # the tank holds the supply
    return eng, dispatcher, by


class TestTellingTheTwoZeroesApart:
    @pytest.mark.asyncio
    async def test_saturated_long_enough_releases_the_budget(self):
        """THE case: relay closed, 1.16 W, ten minutes. The spa may heat."""
        eng, _d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - RELEASE_S)
        assert eng._group_member_released("villavagn_tank", now) is True
        assert eng._group_sibling_claiming_supply(by["spa"]) is None

    @pytest.mark.asyncio
    async def test_a_shed_sibling_keeps_its_claim(self):
        """The 2026-09-02 case, and the one this must never break: the phase guard cut
        the relay, so 0 W means interrupted, not finished."""
        eng, _d, by = _engine(relay="off")
        now = time.time()
        for _ in range(3):
            await eng._note_group_saturation(by["villavagn_tank"], 0.0, now - 3600)
        assert eng._group_member_released("villavagn_tank", now) is False
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_saturation_must_be_sustained(self):
        eng, _d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - 60)
        assert eng._group_member_released("villavagn_tank", now) is False
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_a_drawing_sibling_keeps_its_claim(self):
        eng, _d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1600.0, now - 3600)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_resuming_draw_reclaims_the_budget_immediately(self):
        """A tank that starts re-heating must not wait out a second dwell — the fuse
        does not care why the two loads overlapped."""
        eng, _d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - RELEASE_S)
        assert eng._group_sibling_claiming_supply(by["spa"]) is None
        await eng._note_group_saturation(by["villavagn_tank"], 1600.0, now)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None


class TestRefusingToGuess:
    """Every one of these leaves the claim standing. A member stops holding the budget
    only on positive evidence."""

    @pytest.mark.asyncio
    async def test_unreadable_power_holds(self):
        eng, _d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], None, now - 3600)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_unreadable_relay_holds(self):
        eng, _d, by = _engine(relay="unavailable")
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - 3600)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_a_raising_relay_read_holds(self):
        eng, _d, by = _engine()
        eng.ha_client.get_state_value = AsyncMock(side_effect=RuntimeError("HA said no"))
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - 3600)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_a_non_switch_target_holds(self):
        """The spa is driven through an input_number, so its 'relay' cannot be read as
        on/off at all. Guessing there would re-create the collision."""
        eng, _d, by = _engine()
        eng.dispatcher._last_water_cmd[SPA_ENT] = "on"
        now = time.time()
        await eng._note_group_saturation(by["spa"], 1.0, now - 3600)
        assert eng._group_member_released("spa", now) is False

    @pytest.mark.asyncio
    async def test_power_scale_is_applied_when_present(self):
        """A kW-denominated sensor reading 1.6 is a tank at full tilt, not an idle one.

        WaterHeaterDeviceConfig carries no power_scale field today — the executor reads
        power_entity raw and every threshold in this file (idle_power_w included) assumes
        watts, an assumption that predates this change. The multiplier is honoured if the
        attribute ever appears, so the day it is parsed this keeps working."""
        eng, _d, by = _engine()
        by["villavagn_tank"].power_scale = 1000.0
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.6, now - 3600)
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None


class TestItStaysInertWhereItShould:
    @pytest.mark.asyncio
    async def test_zero_disables_the_release(self):
        eng, _d, by = _engine(release_after=0.0)
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - 86400)
        assert eng._group_member_released("villavagn_tank", now) is False
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    @pytest.mark.asyncio
    async def test_a_sibling_no_longer_commanded_on_is_forgotten(self):
        eng, d, by = _engine()
        now = time.time()
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now - 3600)
        d._last_water_cmd[TANK_ENT] = "off"
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, now)
        assert "villavagn_tank" not in eng._group_idle_since

    @pytest.mark.asyncio
    async def test_the_relay_is_not_read_while_the_load_is_drawing(self):
        """A tick is 100% HTTP-bound. The extra round-trip is only worth paying once the
        power says it might matter."""
        eng, _d, by = _engine()
        await eng._note_group_saturation(by["villavagn_tank"], 1600.0, time.time())
        eng.ha_client.get_state_value.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ungrouped_device_is_never_tracked(self):
        eng, d, by = _engine()
        eng.config.load_groups = []
        d._last_water_cmd[TANK_ENT] = "on"
        await eng._note_group_saturation(by["villavagn_tank"], 1.16, time.time())
        assert eng._group_idle_since == {}
