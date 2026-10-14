"""Regression tests for the 2026-10-11 audit, batch 3."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import pytz
from sqlalchemy import text

from backend.learning.models import Base
from backend.learning.store import LearningStore, _slot_is_over
from executor.engine import PLAN_STALE_SECONDS, ExecutorEngine
from tests.executor.test_blind_wake import T0, FakeHA, _ctrl

TZ = pytz.timezone("Europe/Stockholm")


# --- EV switch memory ------------------------------------------------------


class _SwitchHA(FakeHA):
    """FakeHA whose switch entity carries a last_changed time."""

    def __init__(self, state: str, changed_ts: float) -> None:
        super().__init__({"switch.tesla": state})
        self._changed = datetime.fromtimestamp(changed_ts, UTC).isoformat()

    async def get_state(self, entity: str) -> dict[str, Any] | None:
        base = await super().get_state(entity)
        if base is not None and entity == "switch.tesla":
            base["last_changed"] = self._changed
        return base


def _reconcile(ctrl: Any, ha: Any) -> None:
    asyncio.run(ctrl._reconcile_switch(ha, ctrl.cfg.chargers[0]))


def test_car_that_stopped_itself_is_noticed() -> None:
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = True
    ctrl._last_a["tesla"] = 12.0
    ctrl._last_cmd_ts["tesla"] = T0
    _reconcile(ctrl, _SwitchHA("off", T0 + 600))
    assert ctrl._last_switch["tesla"] is False, "memory must follow the device"
    assert ctrl._last_a["tesla"] == 0.0


def test_car_that_started_itself_is_noticed() -> None:
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = False
    ctrl._last_cmd_ts["tesla"] = T0
    _reconcile(ctrl, _SwitchHA("on", T0 + 600))
    assert ctrl._last_switch["tesla"] is True, "so the next stop is actually sent"


def test_state_older_than_our_command_is_not_believed() -> None:
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = True
    ctrl._last_cmd_ts["tesla"] = T0
    _reconcile(ctrl, _SwitchHA("off", T0 - 300))  # has not caught up with the command
    assert ctrl._last_switch["tesla"] is True
    _reconcile(ctrl, _SwitchHA("off", T0 + 5))  # inside the echo margin
    assert ctrl._last_switch["tesla"] is True


def test_unreadable_or_agreeing_switch_changes_nothing() -> None:
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = True
    ctrl._last_cmd_ts["tesla"] = T0
    _reconcile(ctrl, _SwitchHA("unavailable", T0 + 600))
    _reconcile(ctrl, _SwitchHA("on", T0 + 600))
    _reconcile(ctrl, FakeHA({}))
    assert ctrl._last_switch["tesla"] is True


def _reconcile_at(ctrl: Any, ha: Any, now_ts: float) -> None:
    asyncio.run(ctrl._reconcile_switch(ha, ctrl.cfg.chargers[0], now_ts))


def test_persistent_disagreement_is_believed_despite_newer_commands() -> None:
    # Live 2026-10-14: switch off at T0 (our own stop), memory flipped to ON by a stale
    # reading, then amp writes kept the command clock ahead of the switch's off-time.
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = True
    ctrl._last_a["tesla"] = 5.0
    ctrl._last_cmd_ts["tesla"] = T0 + 3600  # an amp write long after the switch went off
    ha = _SwitchHA("off", T0)
    _reconcile_at(ctrl, ha, T0 + 3600)
    _reconcile_at(ctrl, ha, T0 + 3600 + 600)
    assert ctrl._last_switch["tesla"] is True, "ten minutes may still be poll lag"
    _reconcile_at(ctrl, ha, T0 + 3600 + 900)
    assert ctrl._last_switch["tesla"] is False, "fifteen minutes of 'off' is a fact"
    assert ctrl._last_a["tesla"] == 0.0, "so the next tick sends a real turn_on"


def test_agreement_resets_the_disagreement_clock() -> None:
    ctrl = _ctrl()
    ctrl._last_switch["tesla"] = True
    ctrl._last_cmd_ts["tesla"] = T0 + 3600
    _reconcile_at(ctrl, _SwitchHA("off", T0), T0 + 3600)
    _reconcile_at(ctrl, _SwitchHA("on", T0 + 3700), T0 + 3600 + 600)
    _reconcile_at(ctrl, _SwitchHA("off", T0), T0 + 3600 + 900)
    assert ctrl._last_switch["tesla"] is True, "the clock restarted when they agreed"


def test_persistence_needs_a_known_belief() -> None:
    ctrl = _ctrl()
    ha = _SwitchHA("off", T0)
    _reconcile_at(ctrl, ha, T0)
    _reconcile_at(ctrl, ha, T0 + 2000)
    assert "tesla" not in ctrl._last_switch, "no memory => nothing to correct"


# --- stale plan ------------------------------------------------------------


def _engine(tmp_path: Path) -> ExecutorEngine:
    sched = tmp_path / "schedule.json"
    sched.write_text('{"schedule": [], "meta": {}}', encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"timezone: Europe/Stockholm\nexecutor:\n  enabled: true\n  schedule_path: {sched}\n",
        encoding="utf-8",
    )
    eng = ExecutorEngine(config_path=str(cfg))
    eng.dispatcher = MagicMock()
    eng.dispatcher.notify_error = AsyncMock()
    return eng


def test_stale_plan_notifies_once_and_clears(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    sched = Path(eng.config.schedule_path)

    asyncio.run(eng._check_plan_age())
    eng.dispatcher.notify_error.assert_not_called()

    old = time.time() - PLAN_STALE_SECONDS - 60
    os.utime(sched, (old, old))
    asyncio.run(eng._check_plan_age())
    asyncio.run(eng._check_plan_age())
    assert eng.dispatcher.notify_error.await_count == 1, "once, not every tick"

    os.utime(sched, None)
    asyncio.run(eng._check_plan_age())
    assert eng._plan_stale_notified is False


# --- single-flight tick ----------------------------------------------------


def test_a_second_tick_is_refused_while_one_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = 0

    async def slow_inner() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"success": True}

    eng._tick_inner = slow_inner  # type: ignore[method-assign]

    async def both() -> list[dict[str, Any]]:
        return list(await asyncio.gather(eng._tick(), eng._tick()))

    first, second = asyncio.run(both())
    assert calls == 1
    assert {first["success"], second["success"]} == {True, False}
    assert asyncio.run(eng._tick())["success"] is True, "the lock is released afterwards"


# --- "PAST slot" classification --------------------------------------------


def test_the_running_slot_is_not_past() -> None:
    now = datetime(2026, 10, 11, 12, 7, tzinfo=TZ.localize(datetime(2026, 10, 11)).tzinfo)
    iso = lambda h, m: now.replace(hour=h, minute=m).isoformat()  # noqa: E731
    assert not _slot_is_over(iso(12, 0), iso(12, 15), now)  # running
    assert not _slot_is_over(iso(11, 45), iso(12, 6), now)  # inside grace
    assert _slot_is_over(iso(11, 30), iso(11, 45), now)  # genuinely past
    assert not _slot_is_over(iso(12, 0), None, now)
    assert _slot_is_over(iso(11, 0), None, now)


# --- batched upserts -------------------------------------------------------


async def _store() -> LearningStore:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return store


@pytest.mark.asyncio
async def test_forecasts_and_plan_upsert_in_bulk() -> None:
    store = await _store()
    idx = pd.date_range("2026-10-11", periods=96, freq="15min", tz="Europe/Stockholm")
    rows = [
        {"slot_start": t.isoformat(), "pv_forecast_kwh": 1.0, "load_forecast_kwh": 0.5, "pv_p10": 0.4}
        for t in idx
    ]
    rows.append({"pv_forecast_kwh": 9.0})  # no slot_start: skipped, as before
    await store.store_forecasts(rows, "aurora")
    await store.store_forecasts([{**r, "pv_forecast_kwh": 2.0} for r in rows], "aurora")

    plan = pd.DataFrame({"start_time": idx, "kepler_charge_kwh": 0.1, "soc_target_percent": 50.0})
    await store.store_plan(plan)
    await store.store_plan(plan.assign(soc_target_percent=60.0))

    async with store.AsyncSession() as s:
        n, pv, p10 = (
            await s.execute(text("select count(*), sum(pv_forecast_kwh), sum(pv_p10) from slot_forecasts"))
        ).one()
        m, soc = (await s.execute(text("select count(*), avg(planned_soc_percent) from slot_plans"))).one()
    assert (n, pv, round(p10, 6)) == (96, 192.0, 38.4), "second write updates, never duplicates"
    assert (m, soc) == (96, 60.0)
    await store.close()
