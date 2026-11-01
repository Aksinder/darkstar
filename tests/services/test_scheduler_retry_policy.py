"""Scheduler retry decisions and skip logging (port of upstream 5acfad38 extras).

- HA unreachable is a typed, transient error (backoff, not UNKNOWN)
- a config-blocking suspension has a 30-min ceiling instead of lasting until a save
- every non-run decision is logged (rate-limited), so "planning stopped" leaves a trace
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import backend.services.scheduler_service as scheduler_module
from backend.services.planner_service import (
    _SUSPENSION_CEILING_S,
    PlannerResult,
    PlannerService,
)
from backend.services.scheduler_service import SchedulerService
from planner.errors import PlannerError, PlannerErrorCode, is_transient

_CFG = {"enabled": True, "every_minutes": 30, "jitter_minutes": 0}


def _scheduler_with(monkeypatch, svc: PlannerService, tick) -> SchedulerService:
    scheduler = SchedulerService()
    monkeypatch.setattr(scheduler_module, "planner_service", svc)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", tick)
    scheduler._load_config = lambda: dict(_CFG)
    scheduler._running = True
    return scheduler


def test_skip_reason_change_logs_immediately(caplog):
    scheduler = SchedulerService()
    due_at = datetime.now(UTC) + timedelta(seconds=30)
    with caplog.at_level("INFO", logger="darkstar.services.scheduler"):
        scheduler._log_skip("retry_pending", due_at)
        scheduler._log_skip("suspended", None)
    assert len(caplog.records) == 2
    assert "retry_pending" in caplog.records[0].message
    assert "suspended" in caplog.records[1].message


def test_repeated_skip_reason_is_rate_limited(caplog):
    scheduler = SchedulerService()
    due_at = datetime.now(UTC) + timedelta(seconds=30)
    with caplog.at_level("INFO", logger="darkstar.services.scheduler"):
        for _ in range(10):
            scheduler._log_skip("retry_pending", due_at)
    assert len(caplog.records) == 1


def test_suspension_expired_only_after_ceiling():
    svc = PlannerService()
    svc._consecutive_failures = 1
    svc._apply_retry_policy(PlannerErrorCode.CONFIG_INVALID)
    assert svc.retry_suspended is True
    assert svc.suspension_expired is False
    svc._suspended_since = datetime.now(UTC) - timedelta(seconds=_SUSPENSION_CEILING_S + 1)
    assert svc.suspension_expired is True
    svc.clear_retry_suspension()
    assert svc.suspension_expired is False
    assert svc._suspended_since is None


@pytest.mark.asyncio
async def test_suspension_retries_after_ceiling_and_resuspends(monkeypatch):
    svc = PlannerService()
    svc._retry_suspended = True
    svc._suspended_since = datetime.now(UTC) - timedelta(seconds=_SUSPENSION_CEILING_S + 1)
    attempts = 0

    async def fail_again(**_kwargs):
        nonlocal attempts
        attempts += 1
        svc._consecutive_failures += 1
        svc._apply_retry_policy(PlannerErrorCode.CONFIG_INVALID)
        scheduler._running = False
        return PlannerResult(
            success=False,
            planned_at=datetime.now(UTC),
            error="configuration remains invalid",
            error_code=PlannerErrorCode.CONFIG_INVALID.value,
        )

    async def tick(_seconds):
        return None

    svc.run_once = fail_again
    scheduler = _scheduler_with(monkeypatch, svc, tick)
    await scheduler._loop()

    assert attempts == 1
    assert svc.retry_suspended is True
    assert svc.suspension_expired is False  # fresh suspension window


@pytest.mark.asyncio
async def test_active_suspension_is_logged_without_running(monkeypatch, caplog):
    svc = PlannerService()
    svc._retry_suspended = True
    svc._suspended_since = datetime.now(UTC)
    svc.run_once = AsyncMock(side_effect=AssertionError("must not run while suspended"))

    async def tick(_seconds):
        scheduler._running = False

    scheduler = _scheduler_with(monkeypatch, svc, tick)
    with caplog.at_level("INFO", logger="darkstar.services.scheduler"):
        await scheduler._loop()
    assert any("Planning skipped: suspended" in r.message for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("when", "reason"), [("future", "retry_pending"), ("due", "cadence_pending")]
)
async def test_scheduler_logs_non_run_reason(monkeypatch, caplog, when, reason):
    svc = PlannerService()
    now = datetime.now(UTC)
    svc._next_retry_at = now + timedelta(minutes=5) if when == "future" else now
    svc.run_once = AsyncMock(side_effect=AssertionError("must not run"))

    async def tick(_seconds):
        scheduler._running = False

    scheduler = _scheduler_with(monkeypatch, svc, tick)
    with caplog.at_level("INFO", logger="darkstar.services.scheduler"):
        await scheduler._loop()
    assert any(reason in r.message for r in caplog.records)


def test_ha_unavailable_is_transient_backoff():
    assert is_transient(PlannerErrorCode.HA_UNAVAILABLE)
    svc = PlannerService()
    svc._consecutive_failures = 3
    before = datetime.now(UTC)
    svc._apply_retry_policy(PlannerErrorCode.HA_UNAVAILABLE)
    assert svc.retry_suspended is False
    assert 230 <= (svc._next_retry_at - before).total_seconds() <= 245  # 3rd step = 240 s


@pytest.mark.asyncio
async def test_unreadable_battery_soc_raises_ha_unavailable(tmp_path):
    from backend.core.ha_client import get_initial_state

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "system": {"battery": {"capacity_kwh": 10.0}},
                "input_sensors": {"battery_soc": "sensor.soc"},
            }
        )
    )
    with (
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
        patch(
            "backend.core.ha_client.get_ha_entity_state",
            new_callable=AsyncMock,
            return_value={"state": "unavailable"},
        ),
        pytest.raises(PlannerError) as exc,
    ):
        await get_initial_state(str(cfg))
    assert exc.value.code == PlannerErrorCode.HA_UNAVAILABLE
    assert exc.value.details == {"entity_id": "sensor.soc"}
