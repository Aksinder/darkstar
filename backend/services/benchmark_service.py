"""Nightly battery hindsight benchmark service.

Once a night at ``learning.hindsight_benchmark.run_at`` LOCAL time (default 03:30 —
after the 03:00 ML training, and away from the evening/dawn replans that strain the
2-vCPU box), score every finished day of the last ``window_days`` that has no stored
non-failed result for the current BENCHMARK_VERSION. Oldest first, one small LP at a
time off the event loop. If the live planner is mid-solve, wait for it (bounded) so
the two never compete for the CPU. Results go to system_state
(``hindsight_benchmark:<date>``) and are served by GET /api/energy/battery-benchmark.

Read-only on observations; never touches the plan, executor or HA.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import pytz

from backend.core.secrets import load_yaml
from backend.hindsight_benchmark import (
    BENCHMARK_VERSION,
    base_record,
    benchmark_day,
    rows_by_local_day,
    state_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.learning.store import LearningStore

logger = logging.getLogger("darkstar.services.benchmark")

DEFAULT_RUN_AT = time(3, 30)
DEFAULT_WINDOW_DAYS = 14
PLANNER_WAIT_POLL_S = 15.0
PLANNER_WAIT_MAX_S = 15 * 60.0


def benchmark_settings(config: dict[str, Any]) -> tuple[bool, time, int]:
    """(enabled, run_at local time, window_days) from learning.hindsight_benchmark."""
    cfg = (config.get("learning") or {}).get("hindsight_benchmark") or {}
    enabled = bool(cfg.get("enabled", True))
    run_at = DEFAULT_RUN_AT
    raw = cfg.get("run_at")
    if raw:
        try:
            hh, mm = (int(x) for x in str(raw).split(":", 1))
            run_at = time(hh, mm)
        except (ValueError, TypeError):
            logger.warning("hindsight_benchmark.run_at %r invalid — using 03:30", raw)
    try:
        window = max(1, min(60, int(cfg.get("window_days", DEFAULT_WINDOW_DAYS))))
    except (TypeError, ValueError):
        window = DEFAULT_WINDOW_DAYS
    return enabled, run_at, window


def next_run_after(now: datetime, run_at: time, tz: Any) -> datetime:
    """Next tz-aware local occurrence of run_at strictly after ``now`` (DST-safe)."""
    local = now.astimezone(tz)
    candidate = tz.localize(datetime.combine(local.date(), run_at))
    if candidate <= local:
        candidate = tz.localize(datetime.combine(local.date() + timedelta(days=1), run_at))
    return candidate


def _planner_busy() -> bool:
    try:
        from backend.services.planner_service import planner_service

        return planner_service.get_status().get("is_running", False) is True
    except Exception:
        return False


class BenchmarkService:
    def __init__(
        self,
        config_loader: Callable[[], dict[str, Any]] | None = None,
        now: Callable[[], datetime] | None = None,
        planner_busy: Callable[[], bool] | None = None,
    ) -> None:
        self._config_loader = config_loader or (lambda: load_yaml("config.yaml") or {})
        self._now = now or (lambda: datetime.now(UTC))
        self._planner_busy = planner_busy or _planner_busy
        self._task: asyncio.Task[None] | None = None

    async def start(self, store: LearningStore) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(store), name="hindsight_benchmark")
        logger.info("Hindsight benchmark service started")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self, store: LearningStore) -> None:
        while True:
            config = self._config_loader()
            tz = pytz.timezone(str(config.get("timezone", "Europe/Stockholm")))
            _enabled, run_at, _window = benchmark_settings(config)
            now = self._now()
            wake = next_run_after(now, run_at, tz)
            await asyncio.sleep(max(1.0, (wake - now.astimezone(tz)).total_seconds()))
            try:
                await self.run_once(store)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Hindsight benchmark run failed")

    async def _wait_for_planner(self) -> None:
        waited = 0.0
        while self._planner_busy() and waited < PLANNER_WAIT_MAX_S:
            await asyncio.sleep(PLANNER_WAIT_POLL_S)
            waited += PLANNER_WAIT_POLL_S

    async def run_once(self, store: LearningStore) -> list[date]:
        """Score pending days; returns the days processed."""
        from planner.solver.adapter import config_to_kepler_config

        config = self._config_loader()
        enabled, _run_at, window = benchmark_settings(config)
        if not enabled:
            return []
        kepler_config = config_to_kepler_config(config)
        if kepler_config.capacity_kwh <= 0:
            return []
        tz = pytz.timezone(str(config.get("timezone", "Europe/Stockholm")))
        now = self._now().astimezone(tz)
        today = now.date()
        days = [today - timedelta(days=k) for k in range(window, 0, -1)]
        stored = await store.get_system_states([state_key(d) for d in days])
        pending = [d for d in days if not _is_done(stored.get(state_key(d)))]
        if not pending:
            return []

        start = tz.localize(datetime.combine(pending[0] - timedelta(days=1), time()))
        end = tz.localize(datetime.combine(today, time()))
        by_day = rows_by_local_day(await store.get_slot_observation_rows(start, end), tz)
        for day in pending:
            await self._wait_for_planner()
            try:
                record = await asyncio.to_thread(benchmark_day, day, by_day, kepler_config, tz, now)
            except Exception as exc:
                logger.exception("Hindsight benchmark failed for %s", day)
                record = base_record(day, kepler_config)
                record.update(status="failed", reason="error", solver_status=type(exc).__name__)
            await store.set_system_state(state_key(day), json.dumps(record))
            logger.info(
                "Hindsight benchmark %s: %s %s (actual %s, hindsight %s, grid-only %s SEK)",
                day,
                record["status"],
                record.get("reason") or "",
                record.get("actual_cost_sek"),
                record.get("hindsight_cost_sek"),
                record.get("grid_only_cost_sek"),
            )
        return pending


def _is_done(raw: str | None) -> bool:
    """Stored with the current version and not failed (failed days are retried)."""
    if raw is None:
        return False
    try:
        record = json.loads(raw)
        return record.get("version") == BENCHMARK_VERSION and record.get("status") != "failed"
    except (ValueError, AttributeError):
        return False


benchmark_service = BenchmarkService()
