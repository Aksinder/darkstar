"""C3: executor real-time export curtailment.

When the current slot's effective export price is below the threshold (you would pay to
export), the executor forces the inverter export-power limit to 0 W; above the threshold it
restores the feed-in limit. Driven by ``ControllerDecision.export_price_sek_kwh`` and gated by
``executor.export_curtailment``. Uses the Sungrow profile's ``number.export_power_limit``
entity automatically.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from executor.actions import ActionDispatcher, HAClient
from executor.config import ExecutorConfig, ExportCurtailmentConfig, InverterConfig
from executor.controller import ControllerDecision
from executor.profiles import load_profile

EXPORT_ENTITY = "number.export_power_limit"
SWITCH_ENTITY = "switch.export_power_limit"


def _dispatcher(curtailment: ExportCurtailmentConfig, *, current_limit: str = "3720", shadow=False):
    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(return_value=current_limit)
    ha.set_select = AsyncMock(return_value=True)
    ha.set_switch = AsyncMock(return_value=True)
    ha.set_number = AsyncMock(return_value=True)
    ha.set_input_number = AsyncMock(return_value=True)
    config = ExecutorConfig(inverter=InverterConfig(), export_curtailment=curtailment)
    dispatcher = ActionDispatcher(
        ha_client=ha, config=config, profile=load_profile("sungrow"), shadow_mode=shadow
    )
    return dispatcher, ha


def _export_writes(ha) -> list[float]:
    return [c.args[1] for c in ha.set_number.call_args_list if c.args and c.args[0] == EXPORT_ENTITY]


class TestApplyExportCurtailment:
    @pytest.mark.asyncio
    async def test_negative_price_clamps_to_zero(self):
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=3720))
        await d._apply_export_curtailment(-0.05)
        assert 0.0 in _export_writes(ha)
        ha.set_switch.assert_awaited()  # switch enabled so the 0 W limit is enforced

    @pytest.mark.asyncio
    async def test_positive_price_restores_configured_limit(self):
        # Currently clamped at 0; a profitable price must restore the feed-in limit.
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, restore_limit_w=3720), current_limit="0"
        )
        await d._apply_export_curtailment(0.50)
        assert 3720.0 in _export_writes(ha)

    @pytest.mark.asyncio
    async def test_auto_captures_restore_limit_before_first_clamp(self):
        # restore_limit_w=0 => capture the resting limit (3720) right before clamping.
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=0))
        await d._apply_export_curtailment(-0.05)
        assert d._restore_export_limit_w == pytest.approx(3720.0)
        assert 0.0 in _export_writes(ha)

    @pytest.mark.asyncio
    async def test_threshold_respected(self):
        # threshold -0.10: a -0.05 price is NOT below it -> no clamp (restore path).
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, threshold_sek_per_kwh=-0.10, restore_limit_w=3720),
            current_limit="0",
        )
        await d._apply_export_curtailment(-0.05)
        assert 0.0 not in _export_writes(ha)
        assert 3720.0 in _export_writes(ha)

    @pytest.mark.asyncio
    async def test_no_restore_value_does_nothing_when_not_curtailing(self):
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=0))
        result = await d._apply_export_curtailment(0.50)
        assert result is None
        assert _export_writes(ha) == []

    @pytest.mark.asyncio
    async def test_shadow_mode_does_not_write(self):
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=3720), shadow=True)
        result = await d._apply_export_curtailment(-0.05)
        assert result is not None and result.skipped
        ha.set_number.assert_not_awaited()


class TestExecuteGating:
    @pytest.mark.asyncio
    async def test_enabled_self_consumption_clamps_at_negative_price(self):
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=3720))
        await d.execute(
            ControllerDecision(mode_intent="self_consumption", export_price_sek_kwh=-0.05)
        )
        assert 0.0 in _export_writes(ha)

    @pytest.mark.asyncio
    async def test_disabled_never_clamps(self):
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=False))
        await d.execute(
            ControllerDecision(mode_intent="self_consumption", export_price_sek_kwh=-0.05)
        )
        assert _export_writes(ha) == []

    @pytest.mark.asyncio
    async def test_export_mode_is_not_overridden_by_curtailment(self):
        # In explicit grid-export mode the mode manages the limit itself; the curtailment
        # guard must not run (the planner only picks export at a profitable price).
        d, _ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=3720))
        d._apply_export_curtailment = AsyncMock()  # type: ignore[method-assign]
        await d.execute(ControllerDecision(mode_intent="export", export_price_sek_kwh=-0.05))
        d._apply_export_curtailment.assert_not_called()


class TestSwitchMethod:
    """method='switch' (2026-08-04): curtail = clamp_limit_w + mode switch ON;
    restore = mode switch OFF (unlimited, no number write — the SH10RT rejects
    out-of-range register writes, so restoring via a high number is fragile)."""

    @pytest.mark.asyncio
    async def test_negative_price_writes_clamp_level_and_enables_switch(self):
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, method="switch", clamp_limit_w=400)
        )
        await d._apply_export_curtailment(-0.05)
        assert 400.0 in _export_writes(ha)
        # F49 inside _set_max_export_power turns the mode switch ON.
        assert any(c.args == (SWITCH_ENTITY, True) for c in ha.set_switch.call_args_list)

    @pytest.mark.asyncio
    async def test_positive_price_turns_switch_off_without_number_write(self):
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, method="switch", clamp_limit_w=400),
            current_limit="on",  # get_state_value returns switch state in this path
        )
        result = await d._apply_export_curtailment(0.50)
        assert result is not None and result.success
        assert result.new_value == "off"
        assert _export_writes(ha) == []  # no number write on restore
        assert any(c.args == (SWITCH_ENTITY, False) for c in ha.set_switch.call_args_list)

    @pytest.mark.asyncio
    async def test_restore_is_idempotent_when_switch_already_off(self):
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, method="switch", clamp_limit_w=400),
            current_limit="off",
        )
        result = await d._apply_export_curtailment(0.50)
        assert result is None  # already off — no write, no EEPROM churn
        ha.set_switch.assert_not_awaited()
        assert _export_writes(ha) == []


class TestUnknownPrice:
    """None = price unknown (stale/missing schedule). C3 fails OPEN: restore normal
    export. A coerced 0.0 used to clamp export on stale schedules (fail-closed for
    revenue); holding an existing clamp on unknown data would forfeit ~1 SEK/kWh for
    the whole outage while a genuine negative slot is rare and shallow."""

    @pytest.mark.asyncio
    async def test_none_price_restores_normal_export(self):
        # Device currently clamped (limit reads 0): unknown price must restore.
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, restore_limit_w=3720), current_limit="0"
        )
        await d._apply_export_curtailment(None)
        assert 3720.0 in _export_writes(ha)

    @pytest.mark.asyncio
    async def test_none_price_releases_active_clamp(self):
        # Clamp active, then the price goes unknown: fail open — the clamp is
        # released so a planner freeze cannot silently zero export for hours.
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, restore_limit_w=3720), current_limit="0"
        )
        await d._apply_export_curtailment(-0.05)
        await d._apply_export_curtailment(None)
        assert _export_writes(ha)[-1] == 3720.0

    @pytest.mark.asyncio
    async def test_none_price_never_clamps(self):
        # Whatever else it does, unknown price must not CREATE a clamp.
        d, ha = _dispatcher(ExportCurtailmentConfig(enabled=True, restore_limit_w=0))
        await d._apply_export_curtailment(None)
        assert 0.0 not in _export_writes(ha)


class TestRejectedRestoreSelfHeals:
    """The number path is the dangerous half: _set_max_export_power turns the limit MODE
    SWITCH ON whenever the write reports success, so a value the DEVICE refuses leaves the
    switch enforcing whatever stale low limit the register still holds. The site then sits
    curtailed with nothing trying to lift it.

    Not hypothetical: Sungrow SH10RT register 13073 accepts 8500 and 400 but rejects 10000
    with a pymodbus isError — and 10000 is exactly the restore_limit_w configured live.
    """

    @staticmethod
    def _rejecting(written_but_holds: str, restore_w: float):
        """A device that accepts the service call but whose register keeps its old value."""
        d, ha = _dispatcher(
            ExportCurtailmentConfig(enabled=True, restore_limit_w=restore_w),
            current_limit=written_but_holds,
        )
        return d, ha

    @staticmethod
    def _switch_writes(ha) -> list[bool]:
        return [
            c.args[1]
            for c in ha.set_switch.call_args_list
            if c.args and c.args[0] == SWITCH_ENTITY
        ]

    @pytest.mark.asyncio
    async def test_a_rejected_restore_falls_back_to_switch_off(self):
        # The register holds 400 W and refuses 10000: the readback never becomes 10000.
        d, ha = self._rejecting("400", 10000.0)
        await d._apply_export_curtailment(0.5)  # price above threshold => restore
        assert 10000.0 in _export_writes(ha), "it must still attempt the configured restore"
        # ...and, seeing the register did not take it, must end unlimited rather than
        # leaving the mode switch enforcing 400 W.
        assert self._switch_writes(ha)[-1] is False

    @pytest.mark.asyncio
    async def test_an_accepted_restore_does_not_touch_the_switch_off(self):
        """The healthy path must be untouched: a device that TAKES the value keeps the
        limit enforced at it, exactly as before."""
        d, ha = self._rejecting("400", 8500.0)

        # A device that accepts: the register reads 400 until we write, then 8500.
        state = {"v": "400"}

        async def _accepting_write(entity, value):
            if entity == EXPORT_ENTITY:
                state["v"] = str(value)
            return True

        ha.set_number = AsyncMock(side_effect=_accepting_write)
        ha.get_state_value = AsyncMock(side_effect=lambda _e: state["v"])

        await d._apply_export_curtailment(0.5)
        assert 8500.0 in _export_writes(ha)
        assert False not in self._switch_writes(ha)
