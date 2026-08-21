"""The stateful half of the opportunistic pump gates: the daily extra-hours budget.

The decision itself is pure and covered by test_cyclic_run.py. What lives only in
the engine is the CLOCK — how much opportunistic runtime a load has spent today —
and its interaction with the early exits (control pause, fuse shed), which return
before the gate is ever consulted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from executor.actions import ActionResult
from executor.config import CyclicLoadConfig, ExecutorConfig
from executor.engine import ExecutorEngine

FILTER = "bogfilter"
HOUR = 3600.0


def _load(**over) -> CyclicLoadConfig:
    base = {
        "id": FILTER,
        "name": "Bogfilter",
        "switch_entity": "switch.bogfilter",
        "power_kw": 0.39,
        "surplus_run": True,
        "max_price_percentile": 80.0,
        "presence_entities": ["person.robert_niska"],
        "presence_max_price_percentile": 50.0,
        "max_extra_hours_per_day": 4.0,
    }
    base.update(over)
    return CyclicLoadConfig(**base)


def _engine(loads, tmp_path) -> ExecutorEngine:
    config = ExecutorConfig(cyclic_loads=loads)
    with patch("executor.engine.load_executor_config", return_value=config), patch(
        "executor.engine.load_yaml", return_value={"input_sensors": {}}
    ), patch.object(ExecutorEngine, "_get_db_path", return_value=str(tmp_path / "x.db")):
        engine = ExecutorEngine("config.yaml")
    ha = MagicMock()
    ha.get_state_value = AsyncMock(return_value="not_home")
    engine.ha_client = ha
    engine.dispatcher = MagicMock()
    engine.dispatcher.control_pause_entity = AsyncMock(return_value=None)
    engine.dispatcher.set_cyclic_load = AsyncMock(
        return_value=ActionResult(action_type="cyclic", success=True, message="ok")
    )
    return engine


def _ctx(**over) -> dict:
    """Exporting, cheap, so the surplus gate says yes unless told otherwise."""
    base = {
        "grid_w": -4000.0,
        "battery_w": 0.0,
        "import_price": 0.60,
        "export_price": 0.40,
        "price_window": [0.20, 0.40, 0.60, 0.90, 1.30, 1.80, 2.40, 3.60],
        # Export runs roughly a krona under import; the gate takes its surplus
        # percentile from THIS series, not the import one.
        "export_price_window": [0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20, 1.80],
        "phase_currents": {"l1": 24.0, "l2": 24.0, "l3": 24.0},
        "fuse_budget_a": 25.0,
    }
    base.update(over)
    return base


def _slot(plan_kw: float = 0.0):
    return SimpleNamespace(water_heater_plans={FILTER: plan_kw})


async def _tick(engine, ctx, now, *, plan_kw=0.0):
    results: list = []
    with patch("executor.engine.time.time", return_value=now):
        await engine._actuate_cyclic_loads(_slot(plan_kw), ctx, {}, results)
    return engine.dispatcher.set_cyclic_load.call_args[0][1]


class TestTheBudgetClock:
    @pytest.mark.asyncio
    async def test_extra_runtime_accumulates_between_ticks(self, tmp_path):
        engine = _engine([_load()], tmp_path)
        assert await _tick(engine, _ctx(), 1000.0) is True
        # The first tick has no previous timestamp, so nothing is billed yet.
        assert engine._cyclic_extra_hours(FILTER) == 0.0
        await _tick(engine, _ctx(), 1000.0 + HOUR)
        assert engine._cyclic_extra_hours(FILTER) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_a_spent_budget_stops_the_extra_run(self, tmp_path):
        engine = _engine([_load(max_extra_hours_per_day=2.0)], tmp_path)
        now = 1000.0
        for _ in range(3):
            await _tick(engine, _ctx(), now)
            now += HOUR
        assert await _tick(engine, _ctx(), now) is False

    @pytest.mark.asyncio
    async def test_a_clock_jump_cannot_swallow_the_budget(self, tmp_path):
        """A day-long delta is a clock jump, not a day of pumping."""
        engine = _engine([_load()], tmp_path)
        await _tick(engine, _ctx(), 1000.0)
        await _tick(engine, _ctx(), 1000.0 + 24 * HOUR)
        assert engine._cyclic_extra_hours(FILTER) == 0.0

    @pytest.mark.asyncio
    async def test_planned_runtime_is_not_billed_to_the_extra_budget(self, tmp_path):
        """The plan's own 6 h must not eat the opportunistic allowance."""
        engine = _engine([_load()], tmp_path)
        assert await _tick(engine, _ctx(), 1000.0, plan_kw=0.39) is True
        await _tick(engine, _ctx(), 1000.0 + HOUR, plan_kw=0.39)
        assert engine._cyclic_extra_hours(FILTER) == 0.0


class TestEarlyExitsStopTheClock:
    """Every path that returns before the gate must also stop billing — otherwise a
    shed or a pause is charged to the budget as if the pump had been running."""

    @pytest.mark.asyncio
    async def test_a_fuse_shed_is_not_billed(self, tmp_path):
        # Driven by the PRESENCE gate: the fuse guard credits export against the
        # phase reading, so an overload and a 4 kW surplus cannot coexist.
        engine = _engine([_load(fuse_shed=True)], tmp_path)
        engine.ha_client.get_state_value = AsyncMock(return_value="home")
        home = _ctx(grid_w=300.0)
        assert await _tick(engine, home, 1000.0) is True
        # An hour of shedding on an overloaded phase...
        shed_ctx = _ctx(grid_w=300.0, phase_currents={"l1": 26.0, "l2": 26.0, "l3": 26.0})
        assert await _tick(engine, shed_ctx, 1000.0 + HOUR) is False
        # ...then back to normal: the shed hour must not appear on the bill.
        await _tick(engine, home, 1000.0 + 2 * HOUR)
        assert engine._cyclic_extra_hours(FILTER) == 0.0

    @pytest.mark.asyncio
    async def test_a_control_pause_is_not_billed(self, tmp_path):
        engine = _engine(
            [_load(control_pause_entities=["input_boolean.uthyrning"])], tmp_path
        )
        await _tick(engine, _ctx(), 1000.0)
        engine.dispatcher.control_pause_entity = AsyncMock(
            return_value="input_boolean.uthyrning"
        )
        await _tick(engine, _ctx(), 1000.0 + HOUR)
        engine.dispatcher.control_pause_entity = AsyncMock(return_value=None)
        await _tick(engine, _ctx(), 1000.0 + 2 * HOUR)
        assert engine._cyclic_extra_hours(FILTER) == 0.0

    @pytest.mark.asyncio
    async def test_an_idle_hour_is_not_billed(self, tmp_path):
        """Surplus disappears, the pump stops, and the budget stops with it."""
        engine = _engine([_load()], tmp_path)
        await _tick(engine, _ctx(), 1000.0)
        assert await _tick(engine, _ctx(grid_w=3000.0), 1000.0 + HOUR) is False
        await _tick(engine, _ctx(), 1000.0 + 2 * HOUR)
        assert engine._cyclic_extra_hours(FILTER) == 0.0


class TestPrecedence:
    @pytest.mark.asyncio
    async def test_force_off_outranks_sunshine(self, tmp_path):
        engine = _engine(
            [_load(override_entity="input_select.bogfilter_override")], tmp_path
        )
        engine.ha_client.get_state_value = AsyncMock(return_value="force_off")
        assert await _tick(engine, _ctx(), 1000.0) is False

    @pytest.mark.asyncio
    async def test_no_context_means_no_extras(self, tmp_path):
        """Blind on grid and price: the plan still runs, the luxury does not."""
        engine = _engine([_load()], tmp_path)
        results: list = []
        with patch("executor.engine.time.time", return_value=1000.0):
            await engine._actuate_cyclic_loads(_slot(0.0), None, {}, results)
        assert engine.dispatcher.set_cyclic_load.call_args[0][1] is False


class TestAbsentIsNotOff:
    """2026-08-21 07:26: the pumps' plan timed out, the kept-previous plan had no entry
    for them, and the executor read that as 0 kW — switching both off, filter
    included, for the rest of the day. A plan with no opinion must leave the load
    alone; only an explicit 0.0 means "off on purpose"."""

    @pytest.mark.asyncio
    async def test_a_load_missing_from_the_plan_is_left_alone(self, tmp_path):
        engine = _engine([_load()], tmp_path)
        results: list = []
        with patch("executor.engine.time.time", return_value=1000.0):
            await engine._actuate_cyclic_loads(
                SimpleNamespace(water_heater_plans={"some_other_load": 1.0}),
                _ctx(grid_w=2000.0),  # no surplus either: nothing may turn it on
                {},
                results,
            )
        engine.dispatcher.set_cyclic_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_explicit_zero_still_means_off(self, tmp_path):
        engine = _engine([_load()], tmp_path)
        assert await _tick(engine, _ctx(grid_w=2000.0), 1000.0, plan_kw=0.0) is False

    @pytest.mark.asyncio
    async def test_an_empty_plan_dict_is_absent_too(self, tmp_path):
        engine = _engine([_load()], tmp_path)
        results: list = []
        with patch("executor.engine.time.time", return_value=1000.0):
            await engine._actuate_cyclic_loads(
                SimpleNamespace(water_heater_plans={}), _ctx(grid_w=2000.0), {}, results
            )
        engine.dispatcher.set_cyclic_load.assert_not_called()
