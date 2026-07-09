"""
Build #16 PART C: executor block-commit latch on switch water targets.

Extends the build #15 min-on/min-off dwell into a committed-block latch: on a rising
edge (OFF->ON) the relay is committed ON for the planned block length so a momentary
mid-block plan OFF cannot chop the block (the planner-walk safety net). Guards:
  * min_off is still enforced before the next ON,
  * NO new commit once measured heated_today >= min_kwh_per_day (over-heat guard),
  * boost/safety (bypass_dwell) still break the commit — the floor is never held OFF.
"""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestWaterBlockCommit:
    def _config(self, min_on: float = 30.0, min_off: float = 15.0):
        from executor.config import (
            ControllerConfig,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
            WaterHeaterDeviceConfig,
        )

        return ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(
                temp_normal=60,
                temp_off=40,
                temp_boost=70,
                min_on_minutes=min_on,
                min_off_minutes=min_off,
                manual_on_respect_minutes=90.0,
            ),
            water_heater_devices=[
                WaterHeaterDeviceConfig(
                    id="main",
                    name="Main Heater",
                    target_entity="switch.vvb",
                    power_kw=3.0,
                )
            ],
            notifications=NotificationConfig(),
        )

    def _dispatcher(self, current_state: str, **cfg_over):
        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value=current_state)
        ha_client.set_switch = AsyncMock(return_value=True)
        dispatcher = ActionDispatcher(
            ha_client=ha_client, config=self._config(**cfg_over), shadow_mode=False
        )
        return dispatcher, ha_client

    @pytest.mark.asyncio
    async def test_rising_edge_latches_planned_block_length(self):
        """OFF->ON with commit_minutes=45 latches a 45-min commitment."""
        dispatcher, _ = self._dispatcher("off")
        t0 = time.time()
        await dispatcher.set_water_temp(
            60, "switch.vvb", commit_minutes=45, heated_today_kwh=0.0, min_kwh_per_day=6.0
        )
        cu = dispatcher._water_commit_until["switch.vvb"]
        assert 44 * 60 < (cu - t0) < 46 * 60

    @pytest.mark.asyncio
    async def test_no_extended_commit_when_length_unavailable(self):
        """commit_minutes=None => no separate latch; the min_on dwell IS the fallback
        hold (latching a second min_on-length commit would just duplicate it)."""
        dispatcher, _ = self._dispatcher("off", min_on=30.0)
        await dispatcher.set_water_temp(
            60, "switch.vvb", commit_minutes=None, heated_today_kwh=0.0, min_kwh_per_day=6.0
        )
        assert "switch.vvb" not in dispatcher._water_commit_until

    @pytest.mark.asyncio
    async def test_no_extended_commit_when_block_shorter_than_min_on(self):
        """A planned block shorter than min_on does not latch — min_on already covers it."""
        dispatcher, _ = self._dispatcher("off", min_on=30.0)
        await dispatcher.set_water_temp(
            60, "switch.vvb", commit_minutes=15, heated_today_kwh=0.0, min_kwh_per_day=6.0
        )
        assert "switch.vvb" not in dispatcher._water_commit_until

    @pytest.mark.asyncio
    async def test_block_commit_holds_on_across_mid_block_plan_off(self):
        """A committed block holds ON through a mid-block plan OFF, even after min_on
        has elapsed (so it is the block-commit, not the dwell, doing the holding)."""
        dispatcher, ha = self._dispatcher("off", min_on=30.0)
        # Rising edge: 90-min planned block, floor NOT met.
        await dispatcher.set_water_temp(
            60, "switch.vvb", commit_minutes=90, heated_today_kwh=0.0, min_kwh_per_day=6.0
        )
        assert "switch.vvb" in dispatcher._water_commit_until

        # Age the last switch past min_on so the dwell alone would ALLOW the OFF.
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 31 * 60
        ha.get_state_value = AsyncMock(return_value="on")
        ha.set_switch.reset_mock()

        result = await dispatcher.set_water_temp(
            40, "switch.vvb", commit_minutes=90, heated_today_kwh=0.0, min_kwh_per_day=6.0
        )
        assert result.success is True
        assert result.skipped is True
        assert "Block-commit hold" in result.message
        assert result.new_value == "on"
        ha.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_min_off_enforced_after_block_ends(self):
        """Once the commit expires the OFF is honored (commit cleared), and an immediate
        re-ON is then held by min_off."""
        dispatcher, ha = self._dispatcher("on", min_on=30.0, min_off=15.0)
        dispatcher._last_water_cmd["switch.vvb"] = "on"
        dispatcher._water_commit_until["switch.vvb"] = time.time() - 60  # expired
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 31 * 60  # min_on ok

        off = await dispatcher.set_water_temp(40, "switch.vvb")
        assert off.new_value == "off"
        ha.set_switch.assert_awaited_once_with("switch.vvb", False)
        assert "switch.vvb" not in dispatcher._water_commit_until  # cleared on OFF

        # Immediate re-ON is held by min_off (the OFF just happened).
        ha.get_state_value = AsyncMock(return_value="off")
        ha.set_switch.reset_mock()
        on = await dispatcher.set_water_temp(60, "switch.vvb")
        assert on.skipped is True
        assert "Dwell hold" in on.message
        ha.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_new_commit_once_floor_met(self):
        """Over-heat guard: a rising edge does NOT latch a commit once measured
        heated_today >= min_kwh_per_day, and a subsequent plan OFF is not block-held."""
        dispatcher, ha = self._dispatcher("off", min_on=30.0)
        on = await dispatcher.set_water_temp(
            60, "switch.vvb", commit_minutes=90, heated_today_kwh=6.5, min_kwh_per_day=6.0
        )
        assert on.new_value == "on"  # ON still actuates (plan may still want a top-up)
        assert "switch.vvb" not in dispatcher._water_commit_until  # but NO commit latched

        # After min_on the plan OFF is allowed — no over-heat hold.
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 31 * 60
        ha.get_state_value = AsyncMock(return_value="on")
        ha.set_switch.reset_mock()
        off = await dispatcher.set_water_temp(
            40, "switch.vvb", commit_minutes=90, heated_today_kwh=6.5, min_kwh_per_day=6.0
        )
        assert off.new_value == "off"
        ha.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_boost_safety_bypass_breaks_commit(self):
        """A boost/safety forced OFF (bypass_dwell=True) breaks an active commit — the
        floor is never held OFF and safety always wins."""
        dispatcher, ha = self._dispatcher("on")
        dispatcher._last_water_cmd["switch.vvb"] = "on"
        dispatcher._water_commit_until["switch.vvb"] = time.time() + 3600  # active commit

        result = await dispatcher.set_water_temp(40, "switch.vvb", bypass_dwell=True)
        assert result.new_value == "off"
        ha.set_switch.assert_awaited_once_with("switch.vvb", False)
        assert "switch.vvb" not in dispatcher._water_commit_until

    @pytest.mark.asyncio
    async def test_boost_on_does_not_latch_commit(self):
        """A bypass_dwell ON (boost) actuates but does not trap the relay in a commit —
        boost is re-asserted every tick and must never be governed by the block latch."""
        dispatcher, _ = self._dispatcher("off")
        await dispatcher.set_water_temp(70, "switch.vvb", bypass_dwell=True)
        assert "switch.vvb" not in dispatcher._water_commit_until
