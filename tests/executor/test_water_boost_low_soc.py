"""A manual water boost is never cancelled because the battery is low.

The old guard cancelled the boost when SoC < min_soc + 10. It protected nothing:
the FORCE_HEAT override runs the inverter in idle (no battery discharge), so the
guard only made the boost button dead on a low battery (every winter evening).
"""

import contextlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from executor.controller import make_decision
from tests.executor.test_control_pause import (  # noqa: F401  (fixtures)
    TZ,
    _build_engine,
    _water_entities_called,
    _write_plan_schedule,
    temp_db,
    temp_schedule,
)


def _engine_at_soc(temp_schedule, temp_db, soc: float):  # noqa: F811
    engine = _build_engine(temp_schedule, temp_db, paused_on=set())
    engine._full_config = {**engine._full_config, "battery": {"min_soc_percent": 12}}
    engine.dispatcher._send_notification = AsyncMock()
    engine.dispatcher.notify_override = AsyncMock()

    real_gather = engine._gather_system_state

    async def _gather():
        state = await real_gather()
        state.current_soc_percent = soc
        return state

    engine._gather_system_state = _gather
    return engine


@contextlib.contextmanager
def _capture_decision():
    """Record the ControllerDecision the tick acts on (the first call; the status
    snapshot emitted afterwards re-derives a plan-only decision)."""
    seen = {}

    def _decide(*a, **k):
        decision = make_decision(*a, **k)
        seen.setdefault("decision", decision)
        return decision

    with patch("executor.engine.make_decision", side_effect=_decide):
        yield seen


@pytest.mark.asyncio
class TestWaterBoostNotCancelledBySoc:
    @pytest.mark.parametrize("soc", [5.0, 12.0, 21.9, 22.0, 80.0])
    async def test_boost_survives_any_soc(self, temp_schedule, temp_db, soc):  # noqa: F811
        engine = _engine_at_soc(temp_schedule, temp_db, soc)
        until = datetime.now(TZ) + timedelta(minutes=30)
        engine._water_boost_until = until
        _write_plan_schedule(temp_schedule)

        result = await engine.run_once()

        # Boost is still armed and drives the FORCE_HEAT override this tick.
        assert engine._water_boost_until == until
        assert result["override"]["type"] == "force_heat"
        # No "battery too low" notification.
        engine.dispatcher._send_notification.assert_not_called()
        # The tanks are written at the boost temperature.
        boost = engine.config.water_heater.temp_boost
        calls = engine.dispatcher.set_water_temp.call_args_list
        assert {"switch.vvb", "switch.villavagn_vvb"} <= _water_entities_called(engine)
        assert all(c.args[0] == boost for c in calls)

    async def test_boost_keeps_battery_idle(self, temp_schedule, temp_db):  # noqa: F811
        """The override stays an idle hold: no charge / discharge is commanded."""
        engine = _engine_at_soc(temp_schedule, temp_db, 13.0)
        engine._water_boost_until = datetime.now(TZ) + timedelta(minutes=30)
        _write_plan_schedule(temp_schedule)
        with _capture_decision() as seen:
            await engine.run_once()

        d = seen["decision"]
        assert d.mode_intent == "idle"
        assert not d.write_charge_current
        assert not d.write_discharge_current
        # soc_target is the plan slot's, not the old SoC-10 / min+10 formula.
        assert d.soc_target == 50

    async def test_boost_without_slot_holds_current_soc(self, temp_schedule, temp_db):  # noqa: F811
        """No plan slot: soc_target = live SoC, never below min_soc."""
        engine = _engine_at_soc(temp_schedule, temp_db, 7.4)
        engine._water_boost_until = datetime.now(TZ) + timedelta(minutes=30)
        Path(temp_schedule).write_text(
            json.dumps({"schedule": [], "meta": {}}), encoding="utf-8"
        )
        with _capture_decision() as seen:
            result = await engine.run_once()

        assert engine._water_boost_until is not None
        assert result["override"]["type"] == "force_heat"
        assert seen["decision"].soc_target == 12  # max(round(7.4), min_soc 12)
