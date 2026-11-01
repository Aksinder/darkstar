"""Executor safety: opt-in plan-age cutoff + min-SoC floor latch with hysteresis.

Plan-age cutoff (executor.max_plan_age_minutes, default OFF): a plan older than the
limit is dropped and the executor runs PLAN_STALE safe mode (self-consumption, no
grid charge, no export, water at temp_off) until a fresh plan lands. Age is the
schedule file's mtime against an AWARE now — epoch arithmetic, so DST and the
container's local timezone cannot shift it (the naive-datetime trap class).

Min-SoC latch (upstream de8b93ee idea): the floor hold engages at SoC <= min_soc and
releases only at min_soc + margin, so the mode stops flapping on 1 % sensor steps.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from sqlalchemy import create_engine

from backend.learning.models import Base
from executor.actions import ActionDispatcher, HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterConfig,
    load_executor_config,
)
from executor.controller import Controller, MinSocFloorLatch
from executor.engine import ExecutorEngine
from executor.override import OverrideEvaluator, OverrideType, SlotPlan, SystemState

if TYPE_CHECKING:
    from collections.abc import Iterator


STHLM = pytz.timezone("Europe/Stockholm")


# --- helpers -------------------------------------------------------------------


@contextlib.contextmanager
def _process_tz(name: str) -> Iterator[None]:
    """Run with the PROCESS local timezone set, like the add-on container (TZ env)."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def _engine(tmp_path: Path, max_plan_age_minutes: float | None = None) -> ExecutorEngine:
    sched = tmp_path / "schedule.json"
    sched.write_text('{"schedule": [], "meta": {}}', encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    extra = (
        "" if max_plan_age_minutes is None else f"  max_plan_age_minutes: {max_plan_age_minutes}\n"
    )
    cfg.write_text(
        "timezone: Europe/Stockholm\nexecutor:\n  enabled: true\n"
        f"  schedule_path: {sched}\n{extra}",
        encoding="utf-8",
    )
    eng = ExecutorEngine(config_path=str(cfg))
    eng.dispatcher = MagicMock()
    eng.dispatcher.notify_error = AsyncMock()
    return eng


def _set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


# --- config --------------------------------------------------------------------


class TestConfig:
    def _load(self, tmp_path: Path, body: str) -> ExecutorConfig:
        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        return load_executor_config(str(p))

    def test_defaults(self, tmp_path):
        cfg = self._load(tmp_path, "executor:\n  enabled: true\n")
        assert cfg.max_plan_age_minutes is None, "plan-age cutoff must default OFF"
        assert cfg.controller.min_soc_release_margin_percent == 2.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("90", 90.0),
            ("120.5", 120.5),
            ("0", None),
            ("-5", None),
            ("null", None),
            ("abc", None),
            ("true", None),
        ],
    )
    def test_max_plan_age_parsing(self, tmp_path, raw, expected):
        cfg = self._load(tmp_path, f"executor:\n  max_plan_age_minutes: {raw}\n")
        assert cfg.max_plan_age_minutes == expected

    @pytest.mark.parametrize(("raw", "expected"), [("0", 0.0), ("3", 3.0), ("-1", 2.0), ("x", 2.0)])
    def test_release_margin_parsing(self, tmp_path, raw, expected):
        cfg = self._load(
            tmp_path, f"executor:\n  controller:\n    min_soc_release_margin_percent: {raw}\n"
        )
        assert cfg.controller.min_soc_release_margin_percent == expected

    def test_default_yaml_ships_cutoff_120(self):
        # Owner decision 2026-11-01: the shipped default is 120 min (never below 60).
        cfg = load_executor_config(str(Path(__file__).parents[2] / "config.default.yaml"))
        assert cfg.max_plan_age_minutes == 120.0
        assert cfg.load_balancing.enabled is True
        assert cfg.load_balancing.actuate is False
        assert cfg.controller.min_soc_release_margin_percent == 2.0


# --- plan-age cutoff: engine -----------------------------------------------------


class TestPlanAgeCutoff:
    @pytest.mark.asyncio
    async def test_off_by_default_follows_any_age(self, tmp_path):
        eng = _engine(tmp_path)
        sched = Path(eng.config.schedule_path)
        now = datetime.now(UTC)
        _set_mtime(sched, now - timedelta(hours=30))
        age = await eng._check_plan_age(now)
        assert await eng._apply_plan_age_cutoff(age) is None
        assert eng._plan_cutoff_active is False

    @pytest.mark.asyncio
    async def test_cutoff_enters_once_and_recovers(self, tmp_path):
        eng = _engine(tmp_path, max_plan_age_minutes=60)
        sched = Path(eng.config.schedule_path)
        now = datetime.now(STHLM)

        _set_mtime(sched, now - timedelta(minutes=30))
        assert await eng._apply_plan_age_cutoff(eng._plan_age_seconds(now)) is None

        _set_mtime(sched, now - timedelta(minutes=61))
        reason = await eng._apply_plan_age_cutoff(eng._plan_age_seconds(now))
        assert reason is not None and "PLAN_STALE" in reason
        assert await eng._apply_plan_age_cutoff(eng._plan_age_seconds(now)) is not None
        assert eng.dispatcher.notify_error.await_count == 1, "notify once per transition"

        _set_mtime(sched, now)  # the planner wrote a fresh plan
        assert await eng._apply_plan_age_cutoff(eng._plan_age_seconds(now)) is None
        assert eng._plan_cutoff_active is False

        # A second episode notifies again.
        _set_mtime(sched, now - timedelta(minutes=90))
        assert await eng._apply_plan_age_cutoff(eng._plan_age_seconds(now)) is not None
        assert eng.dispatcher.notify_error.await_count == 2

    @pytest.mark.asyncio
    async def test_missing_file_is_not_stale(self, tmp_path):
        eng = _engine(tmp_path, max_plan_age_minutes=60)
        Path(eng.config.schedule_path).unlink()
        assert eng._plan_age_seconds(datetime.now(UTC)) is None
        assert await eng._apply_plan_age_cutoff(None) is None

    def test_naive_now_is_rejected(self, tmp_path):
        eng = _engine(tmp_path, max_plan_age_minutes=60)
        with pytest.raises(ValueError):
            eng._plan_age_seconds(datetime.now())

    @pytest.mark.parametrize("process_tz", ["Europe/Stockholm", "UTC", "America/New_York"])
    def test_autumn_dst_fallback_counts_real_minutes(self, tmp_path, process_tz):
        """2026-10-25: 03:00 CEST -> 02:00 CET. Plan written 01:50 CEST, now 02:20 CET:
        the wall clock says 30 min, 90 real minutes have passed. A 60-min cutoff must
        trip — whatever the container's local zone is."""
        with _process_tz(process_tz):
            eng = _engine(tmp_path, max_plan_age_minutes=60)
            sched = Path(eng.config.schedule_path)
            written = STHLM.localize(datetime(2026, 10, 25, 1, 50), is_dst=True)
            now = STHLM.localize(datetime(2026, 10, 25, 2, 20), is_dst=False)
            assert now.replace(tzinfo=None) - written.replace(tzinfo=None) == timedelta(minutes=30)
            _set_mtime(sched, written)
            age = eng._plan_age_seconds(now)
            assert age == pytest.approx(90 * 60)
            # Same instant expressed in UTC gives the same age.
            assert eng._plan_age_seconds(now.astimezone(UTC)) == pytest.approx(90 * 60)
            assert eng._plan_cutoff_s() == 3600.0
            assert age > eng._plan_cutoff_s()

    @pytest.mark.parametrize("process_tz", ["Europe/Stockholm", "UTC"])
    def test_spring_dst_forward_does_not_false_trip(self, tmp_path, process_tz):
        """2026-03-29: 02:00 CET -> 03:00 CEST. Written 01:50 CET, now 03:10 CEST: the
        wall clock says 80 min, only 20 real minutes passed. No safe mode."""
        with _process_tz(process_tz):
            eng = _engine(tmp_path, max_plan_age_minutes=60)
            sched = Path(eng.config.schedule_path)
            written = STHLM.localize(datetime(2026, 3, 29, 1, 50))
            now = STHLM.localize(datetime(2026, 3, 29, 3, 10))
            _set_mtime(sched, written)
            assert eng._plan_age_seconds(now) == pytest.approx(20 * 60)

    @pytest.mark.asyncio
    async def test_two_hour_notice_says_safe_mode_when_cutoff_tripped(self, tmp_path):
        eng = _engine(tmp_path, max_plan_age_minutes=60)
        sched = Path(eng.config.schedule_path)
        now = datetime.now(UTC)
        _set_mtime(sched, now - timedelta(hours=3))
        # Entering safe mode on this tick: the cutoff's own notice follows, so the
        # 2 h notifier stays quiet (one message, not two).
        await eng._check_plan_age(now)
        eng.dispatcher.notify_error.assert_not_awaited()
        # Already in safe mode (cutoff tripped earlier): the 2 h escalation says so.
        eng._plan_stale_notified = False
        eng._plan_cutoff_active = True
        await eng._check_plan_age(now)
        msg = eng.dispatcher.notify_error.await_args.args[0]
        assert "säkert läge" in msg


# --- plan-age cutoff: override + controller ------------------------------------------


class TestPlanStaleSafeMode:
    REASON = "Plan is 95 min old"

    def _override(self, **state_kw):
        state = SystemState(
            plan_stale_reason=self.REASON, slot_exists=False, slot_valid=False, **state_kw
        )
        return state, OverrideEvaluator(water_temp_off=40).evaluate(state, None)

    def test_override_actions(self):
        _state, ovr = self._override(current_soc_percent=55.0, min_soc_percent=12.0)
        assert ovr.override_type == OverrideType.PLAN_STALE
        assert ovr.reason == self.REASON
        assert ovr.actions == {"grid_charging": False, "soc_target": 12, "water_temp": 40}

    def test_beats_slot_failure_but_not_manual_override(self):
        state = SystemState(
            plan_stale_reason=self.REASON, slot_exists=False, manual_override_active=True
        )
        assert OverrideEvaluator().evaluate(state).override_type == OverrideType.MANUAL_OVERRIDE

    def test_no_reason_keeps_slot_failure_fallback(self):
        state = SystemState(slot_exists=False, slot_valid=False)
        assert OverrideEvaluator().evaluate(state).override_type == (
            OverrideType.SLOT_FAILURE_FALLBACK
        )

    def _decide(self, slot: SlotPlan | None = None, **state_kw):
        state, ovr = self._override(**state_kw)
        ctrl = Controller(ControllerConfig(max_charge_w=9000.0), InverterConfig(control_unit="W"))
        return ctrl.decide(slot or SlotPlan(), state, ovr)

    def test_self_consumption_no_grid_no_export(self):
        d = self._decide(current_soc_percent=60.0)
        assert d.mode_intent == "self_consumption"
        assert d.source == "override"
        assert d.export_discharge_w == 0.0
        assert d.charge_value == 9000.0, "PV must still be able to charge the pack"

    def test_holds_at_the_latched_floor(self):
        d = self._decide(current_soc_percent=11.0, min_soc_percent=10.0, min_soc_latched=True)
        assert d.mode_intent == "idle"

    def test_holds_while_a_car_draws(self):
        d = self._decide(SlotPlan(ev_isolation=True), current_soc_percent=60.0)
        assert d.mode_intent == "idle"

    def test_unknown_soc_still_self_consumption(self):
        d = self._decide(soc_known=False)
        assert d.mode_intent == "self_consumption"


# --- plan-age cutoff: full tick ---------------------------------------------------


@pytest.fixture
def tick_engine() -> Iterator[tuple[ExecutorEngine, Path, ExecutorConfig]]:
    with tempfile.TemporaryDirectory() as d:
        sched = Path(d) / "schedule.json"
        db = Path(d) / "t.db"
        Base.metadata.create_all(create_engine(f"sqlite:///{db}"))
        config = ExecutorConfig(
            enabled=True,
            schedule_path=str(sched),
            timezone="Europe/Stockholm",
            inverter=InverterConfig(),
            water_heater=WaterHeaterConfig(),
            notifications=NotificationConfig(),
            controller=ControllerConfig(),
            has_water_heater=False,
        )
        with (
            patch("executor.engine.load_executor_config", return_value=config),
            patch(
                "executor.engine.load_yaml",
                return_value={"input_sensors": {}, "battery": {"min_soc_percent": 10}},
            ),
            patch.object(ExecutorEngine, "_get_db_path", return_value=str(db)),
        ):
            eng = ExecutorEngine("config.yaml")
        ha = MagicMock(spec=HAClient)
        ha.get_state_value.side_effect = lambda e: "on" if "input_boolean" in e else "0.0"
        ha.set_select_option.return_value = True
        ha.set_switch.return_value = True
        ha.set_number.return_value = True
        ha.set_input_number.return_value = True
        eng.ha_client = ha
        eng.dispatcher = ActionDispatcher(ha, config, shadow_mode=True)
        eng.dispatcher.notify_error = AsyncMock()  # type: ignore[method-assign]
        eng.dispatcher.notify_override = AsyncMock()  # type: ignore[method-assign]
        yield eng, sched, config


def _write_grid_charge_plan(sched: Path) -> None:
    now = datetime.now(STHLM)
    start = now - timedelta(minutes=5)
    end = start + timedelta(minutes=15)
    sched.write_text(
        json.dumps(
            {
                "schedule": [
                    {
                        "start_time": start.isoformat(),
                        "end_time": end.isoformat(),
                        "end_time_kepler": end.isoformat(),
                        "battery_charge_kw": 5.0,
                        "battery_discharge_kw": 0,
                        "export_kwh": 0,
                        "pv_forecast_kwh": 0,
                        "load_forecast_kwh": 0.25,
                        "soc_target_percent": 80,
                        "projected_soc_percent": 75,
                    }
                ],
                "meta": {"planned_at": now.isoformat()},
            }
        ),
        encoding="utf-8",
    )


class TestTick:
    @pytest.mark.asyncio
    async def test_stale_plan_tick_runs_safe_mode_then_recovers(self, tick_engine):
        eng, sched, config = tick_engine
        config.max_plan_age_minutes = 60
        _write_grid_charge_plan(sched)
        _set_mtime(sched, datetime.now(UTC) - timedelta(minutes=120))

        result = await eng.run_once()
        assert result.get("success") is True, result
        assert result.get("override", {}).get("type") == "plan_stale"
        assert eng._last_system_state.plan_stale_reason is not None
        assert eng.status.override_type == "plan_stale"
        assert eng.dispatcher.notify_error.await_count == 1

        _set_mtime(sched, datetime.now(UTC))  # fresh plan
        result = await eng.run_once()
        assert result.get("success") is True, result
        assert result.get("override") is None
        assert eng._plan_cutoff_active is False
        assert eng.status.current_slot is not None

    @pytest.mark.asyncio
    async def test_cutoff_off_follows_the_old_plan(self, tick_engine):
        eng, sched, _config = tick_engine
        _write_grid_charge_plan(sched)
        _set_mtime(sched, datetime.now(UTC) - timedelta(hours=10))
        result = await eng.run_once()
        assert result.get("success") is True, result
        assert result.get("override") is None
        assert eng.status.current_slot is not None

    @pytest.mark.asyncio
    async def test_min_soc_latch_survives_ticks(self, tick_engine):
        eng, sched, _config = tick_engine
        _write_grid_charge_plan(sched)
        seen: list[bool | None] = []
        for soc in (10.0, 11.0, 11.4, 12.0, 11.0):
            with patch.object(
                eng,
                "_gather_system_state",
                AsyncMock(return_value=SystemState(current_soc_percent=soc, min_soc_percent=10.0)),
            ):
                await eng.run_once()
            seen.append(eng._last_system_state.min_soc_latched)
        assert seen == [True, True, True, False, False]


# --- min-SoC latch -----------------------------------------------------------------


def _hold_slot() -> SlotPlan:
    # Hold at target 10 with forecast PV > load: self_consumption above the floor.
    return SlotPlan(soc_target=10, pv_kw=2.5, load_kw=1.2)


class TestMinSocLatch:
    def _run(self, socs, margin=2.0, min_soc=10.0, known=None):
        latch = MinSocFloorLatch()
        ctrl = Controller(ControllerConfig(), InverterConfig())
        modes = []
        for i, soc in enumerate(socs):
            state = SystemState(
                current_soc_percent=soc,
                min_soc_percent=min_soc,
                soc_known=True if known is None else known[i],
            )
            state.min_soc_latched = latch.update(state, margin)
            modes.append(ctrl.decide(_hold_slot(), state).mode_intent)
        return modes, latch

    def test_hysteresis_stops_the_flap(self):
        modes, _ = self._run([12.0, 10.0, 11.0, 10.0, 11.0, 12.0, 11.0])
        assert modes == [
            "self_consumption",
            "idle",
            "idle",
            "idle",
            "idle",
            "self_consumption",
            "self_consumption",
        ]

    def test_stateless_rule_used_to_flap(self):
        """Without the latch (direct callers) the old rule still applies unchanged."""
        ctrl = Controller(ControllerConfig(), InverterConfig())
        modes = [
            ctrl.decide(_hold_slot(), SystemState(current_soc_percent=s)).mode_intent
            for s in (10.0, 11.0, 10.0)
        ]
        assert modes == ["idle", "self_consumption", "idle"]

    def test_margin_zero_reproduces_the_old_rule(self):
        modes, _ = self._run([10.0, 11.0, 10.0, 10.4], margin=0.0)
        assert modes == ["idle", "self_consumption", "idle", "idle"]

    def test_custom_margin(self):
        modes, _ = self._run([10.0, 12.0, 14.0], margin=4.0)
        assert modes == ["idle", "idle", "self_consumption"]

    def test_unknown_soc_neither_engages_nor_releases(self):
        modes, latch = self._run([10.0, 50.0, 50.0], known=[True, False, False])
        assert modes == ["idle", "idle", "idle"], "a held floor stays held"
        assert latch.engaged is True
        modes, latch = self._run([50.0, 5.0], known=[True, False])
        assert modes[1] == "self_consumption", "an unlatched floor is not engaged by junk"
        assert latch.engaged is False

    def test_fresh_latch_starts_released(self):
        """Restart: no remembered state; the first reading applies SoC <= min."""
        assert MinSocFloorLatch().engaged is False
        _, latch = self._run([11.0])
        assert latch.engaged is False
        _, latch = self._run([9.6])
        assert latch.engaged is True

    def test_engine_owns_one_latch(self, tmp_path):
        eng = _engine(tmp_path)
        assert isinstance(eng._min_soc_latch, MinSocFloorLatch)
