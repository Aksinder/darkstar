import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from backend.learning.models import ConfigVersion, LearningDailyMetric, LearningRun

if TYPE_CHECKING:
    from backend.learning.store import LearningStore

logger = logging.getLogger("darkstar.api.learning")

router = APIRouter(tags=["learning"])


def _get_learning_engine() -> Any:
    """Get the learning engine instance."""
    from backend.learning import get_learning_engine

    return get_learning_engine()


def _parse_min_date(raw: str) -> datetime:
    """Parse a training-window lower bound (ISO date or datetime).

    A bare date ("2026-07-09") or a naive datetime is localized to the learning
    engine's timezone; an explicitly tz-aware value is CONVERTED to the engine
    timezone — slot_start is stored as local-offset ISO strings and filtered by
    lexical comparison, so the bound must be rendered in the same offset to be
    chronologically correct. Raises HTTPException(400) on unparseable input
    (including the empty string).
    """
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid min_date {raw!r}; expected ISO date/datetime like '2026-07-09'.",
        ) from exc

    tz = getattr(_get_learning_engine(), "timezone", None)
    if parsed.tzinfo is None:
        if tz is not None:
            # pytz tzinfo exposes localize(); stdlib tzinfo does not.
            localize = getattr(tz, "localize", None)
            parsed = localize(parsed) if callable(localize) else parsed.replace(tzinfo=tz)
    elif tz is not None:
        parsed = parsed.astimezone(tz)
    return parsed


@router.get(
    "/api/learning/status",
    summary="Get Learning Status",
    description="Return learning engine status and metrics.",
)
async def learning_status() -> dict[str, Any]:
    """Return learning engine status and metrics."""
    try:
        engine = _get_learning_engine()
        # get_status is now async
        status = await engine.get_status()
        return cast("dict[str, Any]", status)
    except Exception as e:
        logger.exception("Failed to get learning status")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/api/learning/history",
    summary="Get Learning History",
    description="Return learning engine run history.",
)
async def learning_history(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    """Return learning engine run history using Async SQLAlchemy."""
    try:
        engine = _get_learning_engine()
        store: LearningStore = engine.store

        async with store.AsyncSession() as session:
            stmt = select(LearningRun).order_by(desc(LearningRun.started_at)).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()

            results: list[dict[str, Any]] = []
            for run in rows:
                results.append(
                    {
                        "id": run.id,
                        "run_date": run.started_at.isoformat() if run.started_at else None,
                        "status": run.status,
                        "training_type": run.training_type,
                        "models_trained": json.loads(run.models_trained)
                        if run.models_trained
                        else [],
                        "training_duration_seconds": run.training_duration_seconds,
                        "partial_failure": run.partial_failure,
                        "metrics": json.loads(run.result_metrics_json)
                        if run.result_metrics_json
                        else None,
                        "config_changes": json.loads(run.params_json) if run.params_json else None,
                    }
                )
            return {"runs": results, "count": len(results)}
    except Exception as e:
        logger.warning(f"Failed to get learning history (DB may be uninitialized): {e}")
        return {"runs": [], "count": 0, "message": f"Learning history unavailable: {e!s}"}


@router.post(
    "/api/learning/train",
    summary="Trigger ML Training",
    description="Trigger manual ML model retraining now.",
)
async def learning_train(
    min_date: str | None = Query(
        None,
        description=(
            "Optional GLOBAL lower bound (ISO date or datetime, e.g. '2026-07-09') "
            "for the training window — applies to load AND PV models alike; use it "
            "for experiments only. For the persistent PV clean-data floor use the "
            "forecasting.pv_training_min_date config key instead: it bounds PV "
            "residual training on every path (nightly, UI, this endpoint) while "
            "load models keep their full clean history. Omit for the default run."
        ),
    ),
) -> dict[str, Any]:
    """Trigger ML model retraining manually using the unified orchestrator."""
    try:
        from ml.training_orchestrator import train_all_models

        # `is not None` (not truthiness): an empty-but-present value
        # (?min_date= — blank UI field, unset shell variable) must 400 loudly,
        # not silently fall back to a full-history retrain.
        parsed_min_date = _parse_min_date(min_date) if min_date is not None else None

        logger.info(
            "Manual training triggered via API (min_date=%s)",
            parsed_min_date.isoformat() if parsed_min_date else None,
        )

        # train_all_models is async and handles locking/logging
        raw_result = await train_all_models(
            training_type="manual", min_date=parsed_min_date
        )
        result: dict[str, Any] = raw_result

        if result.get("status") == "busy":
            raise HTTPException(status_code=409, detail="Training already in progress")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to train models")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/api/learning/run",
    summary="Trigger Learning Run (Full)",
    description="Trigger full learning suite (Reflex + Training).",
)
async def learning_run() -> dict[str, Any]:
    """Trigger full learning run (Sync Reflex + Async Training)."""
    try:
        from backend.learning.reflex import AuroraReflex
        from ml.training_orchestrator import train_all_models

        logger.info("Full learning run triggered via API")

        reflex_report: Any = await AuroraReflex().run(dry_run=False)

        # Training is async
        training_result: dict[str, Any] = await train_all_models(training_type="manual")

        return {
            "status": "success",
            "reflex_report": reflex_report,
            "training_result": training_result,
            "message": "Full learning run completed ",
        }
    except Exception as e:
        logger.exception("Failed to run full learning cycle")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/api/learning/loops",
    summary="Get Learning Loops",
    description="Get status of individual learning loops.",
)
async def learning_loops() -> dict[str, Any]:
    """Real status per learning loop (previously hardcoded 'active' for UI compat).

    - pv_forecast / load_forecast are driven by the nightly LightGBM retraining
      (LearningRun rows): 'active' = a successful run within 8 days, 'stale' =
      older, 'never_ran' = none.
    - s_index / arbitrage are driven by Aurora Reflex parameter tuning
      (ReflexState rows): they report 'never_ran' until Reflex has actually
      applied a change — honesty over reassurance.
    """
    from datetime import datetime, timedelta

    from backend.learning.models import ReflexState

    def _freshness(last_iso: str | None) -> str:
        if not last_iso:
            return "never_ran"
        try:
            last = datetime.fromisoformat(last_iso)
            now = datetime.now(last.tzinfo) if last.tzinfo else datetime.now()
            return "active" if (now - last) <= timedelta(days=8) else "stale"
        except (ValueError, TypeError):
            return "stale"

    try:
        engine = _get_learning_engine()
        store: LearningStore = engine.store
        async with store.AsyncSession() as session:
            run_stmt = (
                select(LearningRun)
                .where(LearningRun.status == "success")
                .order_by(desc(LearningRun.started_at))
                .limit(1)
            )
            last_run = (await session.execute(run_stmt)).scalars().first()
            last_train = (
                last_run.started_at.isoformat() if last_run and last_run.started_at else None
            )

            reflex_rows = (await session.execute(select(ReflexState))).scalars().all()
            s_index_last = max(
                (r.last_updated for r in reflex_rows if r.last_updated and "s_index" in r.param_path),
                default=None,
            )
            other_last = max(
                (
                    r.last_updated
                    for r in reflex_rows
                    if r.last_updated and "s_index" not in r.param_path
                ),
                default=None,
            )

        train_status = _freshness(last_train)
        result: dict[str, Any] = {
            "pv_forecast": {"status": train_status, "last_run": last_train, "error": None},
            "load_forecast": {"status": train_status, "last_run": last_train, "error": None},
            "s_index": {"status": _freshness(s_index_last), "last_run": s_index_last, "error": None},
            "arbitrage": {"status": _freshness(other_last), "last_run": other_last, "error": None},
        }
        return {"loops": result}
    except Exception as e:
        logger.exception("Failed to get learning loop status")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/api/learning/daily_metrics",
    summary="Get Daily Metrics",
    description="Get latest daily metrics from learning engine.",
)
async def learning_daily_metrics():
    """Get latest daily metrics from learning engine using Async SQLAlchemy."""
    try:
        engine = _get_learning_engine()
        store: LearningStore = engine.store

        async with store.AsyncSession() as session:
            stmt = select(LearningDailyMetric).order_by(desc(LearningDailyMetric.date)).limit(1)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if not row:
                return {"message": "No daily metrics yet"}

            return {
                "date": row.date,
                "pv_error_mean_abs_kwh": row.pv_error_mean_abs_kwh,
                "load_error_mean_abs_kwh": row.load_error_mean_abs_kwh,
                "s_index_base_factor": row.s_index_base_factor,
            }
    except Exception as e:
        logger.warning(f"Failed to get daily metrics: {e}")
        return {"message": f"Daily metrics unavailable: {e!s}"}


@router.get(
    "/api/learning/changes",
    summary="Get Learning Changes",
    description="Return recent learning configuration changes.",
)
async def learning_changes(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """Return recent learning configuration changes using Async SQLAlchemy."""
    try:
        engine = _get_learning_engine()
        store: LearningStore = engine.store

        async with store.AsyncSession() as session:
            stmt = select(ConfigVersion).order_by(desc(ConfigVersion.created_at)).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()

            changes: list[dict[str, Any]] = []
            for change in rows:
                changes.append(
                    {
                        "id": change.id,
                        "created_at": change.created_at.isoformat() if change.created_at else None,
                        "reason": change.reason,
                        "applied": change.applied,
                        "metrics": json.loads(change.metrics_json) if change.metrics_json else None,
                    }
                )
            return {"changes": changes}
    except Exception as e:
        logger.warning(f"Failed to get learning changes: {e}")
        return {"changes": [], "message": f"Learning changes unavailable: {e!s}"}


@router.post(
    "/api/learning/record_observation",
    summary="Record Observation",
    description="Trigger observation recording from current system state.",
)
async def record_observation() -> dict[str, str]:
    """Trigger observation recording from current system state."""
    try:
        from backend.recorder import record_observation_from_current_state

        # record_observation_from_current_state is now async
        await record_observation_from_current_state()
        return {"status": "success", "message": "Observation recorded"}
    except Exception as e:
        logger.exception("Failed to record observation")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/api/learning/training-status",
    summary="Get Training Status",
    description="Get current ML training status and model details.",
)
async def learning_training_status() -> dict[str, Any]:
    """Get current ML training status."""
    try:
        import time
        from pathlib import Path

        from ml.api import get_engine
        from ml.forward import determine_graduation_level
        from ml.training_orchestrator import get_training_status

        LOCK_FILE = Path("data/ml/models/.training.lock")

        def _is_lock_stale() -> bool:
            """Check if training lock is stale (older than 1 hour)."""
            if not LOCK_FILE.exists():
                return False
            return time.time() - LOCK_FILE.stat().st_mtime > 3600

        status: dict[str, Any] = await asyncio.to_thread(get_training_status)

        # ARC11 Fix: Explicitly add lock status structure
        # Even if get_training_status has it, we ensure it matches the requested format
        status["lock_status"] = {
            "locked": LOCK_FILE.exists(),
            "stale": _is_lock_stale() if LOCK_FILE.exists() else False,
            "lock_age_seconds": time.time() - LOCK_FILE.stat().st_mtime
            if LOCK_FILE.exists()
            else None,
        }

        # Add graduation level
        try:
            # We recreate engine here to fetch fresh stats
            # In a long running process `get_learning_engine` is cached lru,
            # but determine_graduation_level queries DB directly so it's fine.
            engine = get_engine()
            level, label, days = determine_graduation_level(engine)
            status["graduation_level"] = {
                "level": level,
                "label": label,
                "days_of_data": days,
            }
        except Exception as e:
            logger.warning(f"Failed to determine graduation level: {e}")
            status["graduation_level"] = {
                "level": 0,
                "label": "unknown",
                "days_of_data": 0,
            }

        return status
    except Exception as e:
        logger.exception("Failed to get training status")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/api/learning/training-history",
    summary="Get Training History",
    description="Get recent ML training attempts.",
)
async def learning_training_history(limit: int = Query(5, ge=1, le=20)) -> dict[str, Any]:
    """Get recent ML training attempts (specialized view of learning history)."""
    # Reuse learning_history logic but filter/format specifically for training view if needed
    # For now, it's a wrapper with a smaller default limit
    return await learning_history(limit=limit)
