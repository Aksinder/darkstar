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

# A slot mean used for PLANNING (model fallback / blend) needs at least this many
# usable observations of that time-of-day in the trailing week.
MIN_SLOT_MEAN_OBSERVATIONS = 3


LOAD_BLEND_WEIGHT_KEY = "load_blend_slot_mean_weight"
_warned_blend_weights: set[str] = set()


def load_blend_weight_issue(raw: Any) -> str | None:
    """Why ``raw`` is not a usable forecasting.load_blend_slot_mean_weight (None = fine)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "must be a number between 0 and 1, not true/false"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return f"must be a number between 0 and 1, got {raw!r}"
    if value != value or not 0.0 <= value <= 1.0:
        return f"must be between 0 and 1 (0.6 = 60 % slot mean), got {raw!r}"
    return None


def resolve_load_blend_weight(config: dict[str, Any]) -> float:
    """forecasting.load_blend_slot_mean_weight as a weight in [0, 1].

    Share of the 7-day per-slot mean in the stored load forecast; the model gets the
    rest. Unset/0 = off (pure model, the behaviour without this option). An invalid
    or out-of-range value is NOT clamped — it is reported and treated as 0, because a
    typo such as 40 must not silently turn the forecast into a pure trailing mean.
    """
    forecasting_cfg: dict[str, Any] = config.get("forecasting", {}) or {}
    raw: Any = forecasting_cfg.get(LOAD_BLEND_WEIGHT_KEY)
    issue = load_blend_weight_issue(raw)
    if issue is not None:
        # Read on every forecast run: warn once per distinct bad value, not per run.
        if repr(raw) not in _warned_blend_weights:
            _warned_blend_weights.add(repr(raw))
            logger.warning(
                "Ignoring forecasting.%s: %s; blend disabled", LOAD_BLEND_WEIGHT_KEY, issue
            )
        return 0.0
    return float(raw) if raw is not None else 0.0


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


def time_of_day_key(ts: datetime) -> str:
    """Slot-of-day key ("HH:MM", local wall clock) matching load_slot_means()."""
    return f"{ts.hour:02d}:{ts.minute:02d}"


def load_slot_means(
    rows: list[dict[str, Any]],
    min_observations: int = MIN_SLOT_MEAN_OBSERVATIONS,
    apply_cleaning: bool = True,
) -> dict[str, float]:
    """Mean observed BASE load per time-of-day ("HH:MM") from observation rows.

    Unlike the stored baseline_7_day_avg rows (an eval comparator that counts a
    NULL/zero load as 0.0), this is fit to plan on: only slots with a real load
    (> 0.001 kWh, the trainer's own filter) that the recorder did not flag as
    unmeasured for the load target (ml/cleaning.py) are averaged, and a time-of-day
    with fewer than ``min_observations`` such slots gets no value.

    The flag rules are the trainer's, including its offline-device exemption and its
    starvation guard: if they would remove more than half of the week, the unfiltered
    rows are averaged instead (with a warning) rather than leaving no mean at all.
    ``apply_cleaning=False`` (forecasting.training_data_cleaning_enabled: false)
    skips the flag rules.
    """
    if not rows:
        return {}
    import pandas as pd

    from ml.cleaning import clean_training_rows

    frame = pd.DataFrame(rows)
    if "load_kwh" not in frame.columns or "slot_start" not in frame.columns:
        return {}
    frame["load_kwh"] = pd.to_numeric(frame["load_kwh"], errors="coerce")
    if apply_cleaning:
        # On the whole window (before the load filter) so device outages are judged
        # on every slot that was recorded.
        frame, _ = clean_training_rows(frame, "load", log=logger.debug)
    frame = frame[frame["load_kwh"] > 0.001]
    if frame.empty:
        return {}

    buckets: dict[str, list[float]] = {}
    for slot_start, load_kwh in zip(frame["slot_start"], frame["load_kwh"], strict=True):
        key = _time_of_day_key(str(slot_start))
        if key is not None:
            buckets.setdefault(key, []).append(float(load_kwh))
    return {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if len(values) >= min_observations
    }


async def get_load_slot_means_7d(
    store: LearningStore, before: datetime, apply_cleaning: bool = True
) -> dict[str, float]:
    """load_slot_means() over the 7 days of observations before ``before``.

    Never raises: any failure yields {} so callers fall through to their next tier.
    """
    try:
        history = await store.get_observation_rows_between(
            (before - timedelta(days=7)).isoformat(), before.isoformat()
        )
        return load_slot_means(history, apply_cleaning=apply_cleaning)
    except Exception as exc:
        logger.warning("7-day load slot means unavailable: %s", exc)
        return {}


def _parse(iso_ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return None
