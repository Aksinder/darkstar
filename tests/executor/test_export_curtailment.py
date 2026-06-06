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
