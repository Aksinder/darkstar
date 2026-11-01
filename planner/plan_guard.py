"""Plan guard: independent validation + battery-only fallback around the Kepler solve.

The pipeline calls ``solve_guarded`` instead of ``KeplerSolver.solve``. Behaviour:

1. Kepler solves. On success the plan goes through ``validate_plan``:
   * ``plan_validator.mode: log`` (default) — findings are logged and recorded in the
     schedule meta, the plan ships regardless.
   * ``enforce`` — a plan with validation ERRORS is rejected exactly like a solver
     failure (step 2). Warnings never reject.
   * ``off`` — skipped.
2. Kepler failed (SOLVER_TIMEOUT incl. the comfort-floor tripwire reject,
   SOLVER_INFEASIBLE, SOLVER_UNDEFINED) or was rejected by the validator:
   * ``fallback.enabled: false`` (default) — re-raise; the executor keeps the last good
     plan (pre-existing behaviour, unchanged).
   * enabled, and the last good plan on disk is YOUNGER than
     ``fallback.min_plan_age_minutes`` and was itself a Kepler plan — re-raise (a fresh
     Kepler plan beats a battery-only one).
   * otherwise solve the battery-only model (``planner.solver.battery_lp``), validate
     it (enforce mode can reject it too, then the ORIGINAL error is raised), and ship
     it with ``plan_source: "fallback"`` in the schedule meta. Water heaters, EVs,
     sinks and deferrable loads are OFF in that plan. A previous fallback plan is
     always refreshed by a new one regardless of age (both are cheap; newer inputs win).

Any exception inside the guard's own machinery falls back to the original behaviour
(raise the Kepler error / ship the Kepler plan) — the guard must never make the
planner less available than it was without it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from planner.errors import PlannerError, PlannerErrorCode
from planner.plan_validator import PlanValidation, validate_plan
from planner.solver.battery_lp import solve_battery_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from planner.solver.types import KeplerConfig, KeplerInput, KeplerResult

logger = logging.getLogger("darkstar.planner.plan_guard")

VALIDATOR_MODES = ("off", "log", "enforce")
DEFAULT_MIN_PLAN_AGE_MINUTES = 30.0
FALLBACK_CODES = frozenset(
    {
        PlannerErrorCode.SOLVER_TIMEOUT,
        PlannerErrorCode.SOLVER_INFEASIBLE,
        PlannerErrorCode.SOLVER_UNDEFINED,
        PlannerErrorCode.INVALID_SCHEDULE,
    }
)


@dataclass(frozen=True)
class PlanGuardSettings:
    fallback_enabled: bool = False
    fallback_min_plan_age_minutes: float = DEFAULT_MIN_PLAN_AGE_MINUTES
    validator_mode: str = "log"


def settings_from_config(config: dict[str, Any]) -> PlanGuardSettings:
    """Read ``kepler.fallback`` / ``kepler.plan_validator`` (defaults: off / log)."""
    kepler = config.get("kepler") or {}
    fb = kepler.get("fallback") or {}
    pv = kepler.get("plan_validator") or {}
    mode = str(pv.get("mode", "log") or "log").strip().lower()
    if mode not in VALIDATOR_MODES:
        logger.warning("kepler.plan_validator.mode %r unknown — using 'log'", mode)
        mode = "log"
    try:
        min_age = float(fb.get("min_plan_age_minutes", DEFAULT_MIN_PLAN_AGE_MINUTES))
    except (TypeError, ValueError):
        min_age = DEFAULT_MIN_PLAN_AGE_MINUTES
    return PlanGuardSettings(
        fallback_enabled=bool(fb.get("enabled", False)),
        fallback_min_plan_age_minutes=max(0.0, min_age),
        validator_mode=mode,
    )


def read_last_plan(schedule_path: Path, now: datetime | None = None) -> tuple[float | None, str]:
    """(age in minutes, plan_source) of the schedule on disk; (None, "none") if unreadable.

    ``meta.planned_at`` is written as a NAIVE LOCAL timestamp (planner/output/schedule.py);
    compare naive with naive local now, aware with aware — never mix (the recurring
    naive-datetime trap in this codebase).
    """
    try:
        with schedule_path.open(encoding="utf-8") as f:
            meta = (json.load(f) or {}).get("meta") or {}
        planned = datetime.fromisoformat(str(meta["planned_at"]))
    except Exception:
        return None, "none"
    source = str(meta.get("plan_source") or "kepler")
    if planned.tzinfo is None:
        ref = now.replace(tzinfo=None) if now is not None else datetime.now()
    else:
        ref = now if now is not None and now.tzinfo is not None else datetime.now(UTC)
    return max(0.0, (ref - planned).total_seconds() / 60.0), source


def _log_validation(source: str, report: PlanValidation, mode: str) -> None:
    if report.ok and not report.warning_count:
        logger.info(
            "Plan validator (%s, %s): OK — %d slots, cost %.2f SEK",
            source,
            mode,
            report.checked_slots,
            report.recomputed_cost_sek or 0.0,
        )
        return
    level = logging.WARNING if report.error_count else logging.INFO
    logger.log(
        level,
        "Plan validator (%s, %s): %d error(s), %d warning(s). errors=%s warnings=%s",
        source,
        mode,
        report.error_count,
        report.warning_count,
        report.errors,
        report.warnings,
    )


def _trim_details(details: dict[str, Any] | None) -> dict[str, Any]:
    keep = ("solver_status", "reason", "solve_duration_s", "chain_duration_s", "dump_path")
    return {k: v for k, v in (details or {}).items() if k in keep}


def solve_guarded(
    solve: Callable[[KeplerInput, KeplerConfig], KeplerResult],
    input_data: KeplerInput,
    config: KeplerConfig,
    settings: PlanGuardSettings,
    schedule_path: Path,
    now: datetime | None = None,
) -> tuple[KeplerResult, dict[str, Any]]:
    """Run Kepler with validation and optional fallback. Returns (result, plan_meta).

    plan_meta is merged into schedule.json ``meta``; it always carries ``plan_source``.
    """
    primary_error: PlannerError | None = None
    validation_meta: dict[str, Any] | None = None
    try:
        result = solve(input_data, config)
    except PlannerError as exc:
        if exc.code not in FALLBACK_CODES:
            raise
        primary_error = exc
    else:
        if settings.validator_mode == "off":
            return result, {"plan_source": "kepler"}
        try:
            report = validate_plan(result, input_data, config)
        except Exception as exc:  # validator bug must never block a plan
            logger.warning("Plan validator crashed (%s) — shipping Kepler plan unvalidated", exc)
            return result, {"plan_source": "kepler"}
        _log_validation("kepler", report, settings.validator_mode)
        validation_meta = report.summary()
        if report.ok or settings.validator_mode != "enforce":
            return result, {"plan_source": "kepler", "plan_validation": validation_meta}
        primary_error = PlannerError(
            code=PlannerErrorCode.INVALID_SCHEDULE,
            details={
                "reason": "plan_validator",
                "solver_status": result.status_msg,
                "validation_errors": report.errors,
            },
        )
        logger.error("Plan validator REJECTED the Kepler plan (enforce mode): %s", report.errors)

    # --- Kepler failed or was rejected ---
    assert primary_error is not None
    if not settings.fallback_enabled:
        raise primary_error

    try:
        age_min, last_source = read_last_plan(schedule_path, now)
        if (
            last_source != "fallback"
            and age_min is not None
            and age_min < settings.fallback_min_plan_age_minutes
        ):
            logger.warning(
                "Kepler failed (%s) but the last good plan is only %.0f min old "
                "(< %.0f) — keeping it, no fallback",
                primary_error.code.value,
                age_min,
                settings.fallback_min_plan_age_minutes,
            )
            raise primary_error

        fallback = solve_battery_only(input_data, config, mode="live")
        report = validate_plan(fallback, input_data, config)
        _log_validation("fallback", report, settings.validator_mode)
        if not report.ok and settings.validator_mode == "enforce":
            logger.error(
                "Fallback plan failed validation in enforce mode — keeping the last plan: %s",
                report.errors,
            )
            raise primary_error
    except PlannerError as exc:
        if exc is not primary_error:
            logger.error("Fallback planner failed (%s) — keeping the last plan", exc.code.value)
        raise primary_error from None
    except Exception as exc:
        logger.exception("Fallback planner crashed (%s) — keeping the last plan", exc)
        raise primary_error from None

    logger.warning(
        "PLAN SOURCE = FALLBACK: Kepler %s (%s); last plan was %s min old (%s). Shipping a "
        "battery-only plan (water/EV/sinks/deferrables OFF), cost %.2f SEK over %d slots.",
        primary_error.code.value,
        _trim_details(primary_error.details).get("reason")
        or _trim_details(primary_error.details).get("solver_status"),
        f"{age_min:.0f}" if age_min is not None else "?",
        last_source,
        fallback.total_cost_sek,
        len(fallback.slots),
    )
    meta: dict[str, Any] = {
        "plan_source": "fallback",
        "fallback": {
            "reason": primary_error.code.value,
            "kepler_error": _trim_details(primary_error.details),
            "last_plan_age_minutes": round(age_min, 1) if age_min is not None else None,
            "last_plan_source": last_source,
            "devices_scheduled": False,
            "solve_time_ms": round(fallback.solve_time_ms, 1),
        },
        "plan_validation": report.summary(),
    }
    if validation_meta is not None:
        meta["fallback"]["rejected_kepler_validation"] = validation_meta
    return fallback, meta
