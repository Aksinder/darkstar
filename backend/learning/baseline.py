"""Naive 7-day-average baseline forecasts — the missing half of the ML A/B.

The eval endpoints have always been ABLE to compare "aurora" against
"baseline_7_day_avg", but nothing ever wrote baseline rows, so mae_*_baseline was
forever null and the ML's value was unfalsifiable. This module writes them: for every
slot Aurora forecasts, the baseline prediction is simply the mean of the OBSERVED
pv/load at the same time-of-day over the trailing week. If LightGBM can't beat that,
the training CPU isn't earning its keep — now we can see it either way.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.learning.store import LearningStore

logger = logging.getLogger("darkstar.learning.baseline")

BASELINE_VERSION = "baseline_7_day_avg"


def _time_of_day_key(iso_ts: str) -> str | None:
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return None
    return f"{dt.hour:02d}:{dt.minute:02d}"


async def store_baseline_forecasts(store: LearningStore, slot_starts: list[str]) -> int:
    """Upsert baseline_7_day_avg SlotForecast rows for the given target slots.

    Baseline per slot = mean observed pv/load at the same time-of-day over the 7 days
    before the earliest target slot. Slots whose time-of-day has no observations are
    skipped (a thin history must not fabricate zeros the eval would score).
    Returns the number of rows written; never raises (best-effort observability).
    """
    if not slot_starts:
        return 0
    try:
        anchor = min(
            (d for d in (_parse(s) for s in slot_starts) if d is not None),
            default=None,
        )
        if anchor is None:
            return 0
        history = await store.get_observation_rows_between(
            (anchor - timedelta(days=7)).isoformat(), anchor.isoformat()
        )
        sums: dict[str, list[float]] = {}
        for row in history:
            key = _time_of_day_key(str(row.get("slot_start")))
            if key is None:
                continue
            bucket = sums.setdefault(key, [0.0, 0.0, 0.0])
            bucket[0] += float(row.get("pv_kwh") or 0.0)
            bucket[1] += float(row.get("load_kwh") or 0.0)
            bucket[2] += 1.0

        forecasts: list[dict[str, Any]] = []
        for slot in slot_starts:
            key = _time_of_day_key(slot)
            bucket = sums.get(key) if key else None
            if not bucket or bucket[2] <= 0:
                continue
            forecasts.append(
                {
                    "slot_start": slot,
                    "pv_forecast_kwh": bucket[0] / bucket[2],
                    "load_forecast_kwh": bucket[1] / bucket[2],
                }
            )
        if forecasts:
            await store.store_forecasts(forecasts, forecast_version=BASELINE_VERSION)
        return len(forecasts)
    except Exception as exc:
        logger.warning("Baseline forecast write skipped: %s", exc)
        return 0


def _parse(iso_ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return None
