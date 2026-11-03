"""Intraday PV bias correction.

The Aurora PV forecast (physics + ML residual) is produced from a weather forecast and
has no feedback from what the panels are doing *right now*. On an overcast day that the
weather model called sunny (2026-11-01: ~3-4x over-forecast at midday) the planner keeps
planning to cover load and charge the battery from PV that never arrives.

This module compares today's recently finished slots — actual PV from the recorder
(``slot_observations.pv_kwh``) against the forecast that was valid for those slots
(``slot_forecasts.pv_forecast_kwh``, the row the planner stored while the slot was still
current/upcoming) — and derives one energy-weighted ratio::

    factor = clamp(sum(actual) / sum(forecast), min_factor, max_factor)

which is then applied to the upcoming slots' PV forecast (p10/p50/p90 alike), decaying
toward 1.0 with lead time (exponential, ``half_life_hours``) and never past the end of
today's local day, so tomorrow's forecast is untouched.

Idea ported from srcfl/ftw ``pvmodel/residual.go`` (AGPL, like Darkstar); own code.

Fail-safe by construction: every error path yields factor 1.0 (a no-op) and a warning.
The correction is applied only to the in-memory planner frame — the DB keeps the raw
forecast, so the ratio never feeds back on itself.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from utils.config_parsing import strict_bool
from utils.time_utils import dst_safe_localize

if TYPE_CHECKING:
    import pytz

logger = logging.getLogger("darkstar.planner.inputs.pv_intraday")

SLOT = timedelta(minutes=15)

# Observation quality flags that make the slot's pv_kwh untrustworthy as a measurement.
# "pv_unmeasured": the recorder fell back to the counter/snapshot (pv_power history
# unavailable, e.g. after an HA restart) — partial or ~0 PV that is not weather.
_BAD_PV_FLAGS = ("multi_slot_delta", "no_config", "pv_unmeasured")

PV_COLUMNS = ("pv_forecast_kwh", "pv_p10", "pv_p90")


@dataclass(frozen=True)
class PvIntradayConfig:
    enabled: bool = True
    lookback_hours: float = 2.0
    half_life_hours: float = 2.0
    min_factor: float = 0.2
    max_factor: float = 1.3
    # Upward (under-forecast) deviations are scaled by this before the clamp: planning
    # on PV that may not come is the expensive mistake, so "more sun than forecast" is
    # believed only partially. 1.0 = symmetric.
    upward_gain: float = 0.5
    # Per-slot: ignore slots whose forecast is below this (night, dawn/dusk, where a
    # ratio is noise). 0.05 kWh per 15 min = 0.2 kW average.
    min_slot_forecast_kwh: float = 0.05
    # Evidence: at least this many qualifying slots AND this much forecast energy.
    min_slots: int = 4
    min_forecast_kwh: float = 1.0
    # Dead-feed guard: a slot producing < dead_feed_ratio of a forecast of at least
    # dead_feed_min_forecast_kwh is treated as an unmeasured slot (a broken/stalled PV
    # reading), not as clouds — heavy overcast still yields ~10-30% of clear-sky.
    dead_feed_ratio: float = 0.05
    dead_feed_min_forecast_kwh: float = 0.15
    # And most evidence slots must show some production at all, else no correction.
    min_nonzero_fraction: float = 0.75
    # Slots with the battery this full may have had PV throttled (nowhere to put it):
    # actual < forecast there is not weather. Excluded from the evidence.
    max_soc_percent: float = 98.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PvIntradayConfig:
        raw = (config.get("forecasting") or {}).get("pv_intraday_correction")
        if not isinstance(raw, dict):
            return cls()
        d = cls()

        def f(key: str, default: float) -> float:
            v = raw.get(key, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return v if math.isfinite(v) else default

        min_factor = max(0.0, f("min_factor", d.min_factor))
        max_factor = max(min_factor, f("max_factor", d.max_factor))
        return cls(
            enabled=strict_bool(raw.get("enabled", d.enabled), default=d.enabled),
            lookback_hours=max(0.25, f("lookback_hours", d.lookback_hours)),
            half_life_hours=max(0.0, f("half_life_hours", d.half_life_hours)),
            min_factor=min_factor,
            max_factor=max_factor,
            upward_gain=min(1.0, max(0.0, f("upward_gain", d.upward_gain))),
            min_slot_forecast_kwh=max(0.0, f("min_slot_forecast_kwh", d.min_slot_forecast_kwh)),
            min_slots=max(1, int(f("min_slots", d.min_slots))),
            min_forecast_kwh=max(0.0, f("min_forecast_kwh", d.min_forecast_kwh)),
            max_soc_percent=f("max_soc_percent", d.max_soc_percent),
            dead_feed_ratio=max(0.0, f("dead_feed_ratio", d.dead_feed_ratio)),
            dead_feed_min_forecast_kwh=max(
                0.0, f("dead_feed_min_forecast_kwh", d.dead_feed_min_forecast_kwh)
            ),
            min_nonzero_fraction=min(
                1.0, max(0.0, f("min_nonzero_fraction", d.min_nonzero_fraction))
            ),
        )


@dataclass
class PvCorrection:
    factor: float = 1.0
    raw_ratio: float | None = None
    evidence_slots: int = 0
    actual_kwh: float = 0.0
    forecast_kwh: float = 0.0
    window_start: str | None = None
    window_end: str | None = None
    applied: bool = False
    reason: str = "disabled"
    half_life_hours: float = 2.0
    excluded: dict[str, int] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "factor": round(self.factor, 4),
            "raw_ratio": None if self.raw_ratio is None else round(self.raw_ratio, 4),
            "evidence_slots": self.evidence_slots,
            "actual_kwh": round(self.actual_kwh, 3),
            "forecast_kwh": round(self.forecast_kwh, 3),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "applied": self.applied,
            "reason": self.reason,
            "half_life_hours": self.half_life_hours,
            "excluded": dict(self.excluded),
        }


def _to_utc(value: Any, tz: pytz.BaseTzInfo) -> pd.Timestamp | None:
    """Parse a stored slot_start to an aware UTC timestamp (naive = site-local)."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = pd.Timestamp(dst_safe_localize(ts.to_pydatetime(), tz))
    return ts.tz_convert("UTC")


def local_day_bounds(now: datetime, tz: pytz.BaseTzInfo) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the site-local calendar day containing ``now``, in UTC.

    Built from the local DATE (not now - 24h), so the 23 h / 25 h DST days come out right.
    """
    now_utc = pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    local_date = now_utc.tz_convert(tz).date()
    start = dst_safe_localize(datetime.combine(local_date, time()), tz)
    end = dst_safe_localize(datetime.combine(local_date + timedelta(days=1), time()), tz)
    return pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")


def compute_pv_correction(
    rows: list[dict[str, Any]],
    now: datetime,
    cfg: PvIntradayConfig,
    tz: pytz.BaseTzInfo,
) -> PvCorrection:
    """Derive the intraday PV factor from joined forecast/actual rows.

    Each row: ``slot_start``, ``pv_kwh`` (actual), ``pv_forecast_kwh`` (forecast valid
    for that slot), optionally ``quality_flags``, ``soc_end_percent``,
    ``export_price_sek_kwh``. Pure function — no I/O.
    """
    corr = PvCorrection(half_life_hours=cfg.half_life_hours)
    if not cfg.enabled:
        return corr

    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    now_ts = now_ts.tz_convert("UTC")
    day_start, _ = local_day_bounds(now_ts.to_pydatetime(), tz)
    # Slot-aligned (UTC floor == local floor for quarter boundaries).
    window_start = max(day_start, (now_ts - timedelta(hours=cfg.lookback_hours)).floor("15min"))
    corr.window_start = window_start.isoformat()
    corr.window_end = now_ts.isoformat()

    excluded: dict[str, int] = {}

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    actual_sum = 0.0
    forecast_sum = 0.0
    n = 0
    nonzero = 0
    for row in rows:
        start = _to_utc(row.get("slot_start"), tz)
        if start is None:
            continue
        # Only FINISHED slots of today inside the lookback window.
        if start < window_start or start + SLOT > now_ts:
            continue
        try:
            fc = float(row.get("pv_forecast_kwh"))  # type: ignore[arg-type]
            act = float(row.get("pv_kwh"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            skip("missing")
            continue
        if not (math.isfinite(fc) and math.isfinite(act)) or act < 0:
            skip("missing")
            continue
        if fc < cfg.min_slot_forecast_kwh:
            skip("low_forecast")
            continue
        if fc >= cfg.dead_feed_min_forecast_kwh and act < cfg.dead_feed_ratio * fc:
            skip("pv_dead_feed")
            continue
        flags_raw = row.get("quality_flags")
        if flags_raw:
            try:
                flags = json.loads(str(flags_raw))
            except (ValueError, TypeError):
                flags = {}
            if isinstance(flags, dict) and any(k in flags for k in _BAD_PV_FLAGS):
                skip("bad_quality")
                continue
        soc = row.get("soc_end_percent")
        if soc is not None:
            try:
                if float(soc) >= cfg.max_soc_percent:
                    skip("battery_full")
                    continue
            except (TypeError, ValueError):
                pass
        # The executor's export curtailment (C3) clamps PV on negative export prices;
        # a slot whose price is unknown (recorder lookup failed → NULL) may have been
        # curtailed just the same, so unknown counts as suspect too.
        exp_price = row.get("export_price_sek_kwh")
        try:
            exp_val = float(exp_price)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            exp_val = math.nan
        if not math.isfinite(exp_val):
            skip("export_price_unknown")
            continue
        if exp_val < 0:
            skip("negative_export_price")
            continue
        actual_sum += act
        forecast_sum += fc
        n += 1
        if act > 0.001:
            nonzero += 1

    corr.evidence_slots = n
    corr.actual_kwh = actual_sum
    corr.forecast_kwh = forecast_sum
    corr.excluded = excluded

    if n < cfg.min_slots or forecast_sum < cfg.min_forecast_kwh or forecast_sum <= 0:
        corr.reason = "insufficient_evidence"
        return corr
    if nonzero < cfg.min_nonzero_fraction * n:
        # Mostly-zero "production" under a real forecast is a feed problem, not weather.
        corr.reason = "pv_feed_suspect"
        return corr

    ratio = actual_sum / forecast_sum
    corr.raw_ratio = ratio
    shaped = 1.0 + (ratio - 1.0) * cfg.upward_gain if ratio > 1.0 else ratio
    corr.factor = min(cfg.max_factor, max(cfg.min_factor, shaped))
    corr.applied = abs(corr.factor - 1.0) > 1e-6
    corr.reason = "ok"
    return corr


def decay_factors(
    index: pd.DatetimeIndex,
    factor: float,
    now: datetime,
    half_life_hours: float,
    tz: pytz.BaseTzInfo,
) -> pd.Series:
    """Per-slot multiplier: ``1 + (factor-1) * 0.5**(lead_h / half_life)``.

    Lead time is measured from ``now`` to the slot's midpoint (0 for slots already
    started). Slots at/after the end of today's local day get exactly 1.0.
    ``half_life_hours == 0`` applies the factor only to slots already started.
    """
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    now_ts = now_ts.tz_convert("UTC")
    _, day_end = local_day_bounds(now_ts.to_pydatetime(), tz)

    idx_utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    mid = idx_utc + SLOT / 2
    lead_h = ((mid - now_ts).total_seconds() / 3600.0).to_numpy()
    lead_h = lead_h.clip(min=0.0)
    if half_life_hours > 0:
        weight = 0.5 ** (lead_h / half_life_hours)
    else:
        weight = (idx_utc <= now_ts).astype(float)
    mult = 1.0 + (factor - 1.0) * weight
    mult[(idx_utc >= day_end)] = 1.0
    return pd.Series(mult, index=index)


def apply_pv_correction(
    df: pd.DataFrame,
    corr: PvCorrection,
    now: datetime,
    tz: pytz.BaseTzInfo,
) -> pd.DataFrame:
    """Scale the PV columns of ``df`` (p50 + p10/p90 when present) in place-safe copy."""
    if not corr.applied or df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    mult = decay_factors(df.index, corr.factor, now, corr.half_life_hours, tz)
    out = df.copy()
    for col in PV_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float) * mult
    return out


async def load_pv_correction(
    config: dict[str, Any],
    now: datetime,
    tz: pytz.BaseTzInfo,
) -> PvCorrection:
    """Read today's forecast/actual rows from the learning DB and compute the factor.

    Never raises: any failure → factor 1.0 with a warning.
    """
    try:
        cfg = PvIntradayConfig.from_config(config)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("PV intraday correction: bad config (%s) — factor 1.0", e)
        return PvCorrection(reason="error")
    if not cfg.enabled:
        return PvCorrection(half_life_hours=cfg.half_life_hours, reason="disabled")

    try:
        from backend.learning.store import LearningStore

        path = (config.get("learning") or {}).get("sqlite_path", "data/planner_learning.db")
        if not Path(path).is_file():
            # Never let a read create an empty DB (tests, fresh installs).
            return PvCorrection(half_life_hours=cfg.half_life_hours, reason="no_db")
        version = (config.get("forecasting") or {}).get("active_forecast_version", "aurora")
        day_start, _ = local_day_bounds(now, tz)
        store = LearningStore(path, tz)
        try:
            # String bound one day early: stored ISO strings carry a local offset, so a
            # lexicographic cut exactly at the window would be offset-sensitive. The
            # precise filter is done on parsed aware timestamps.
            lower = (day_start - timedelta(days=1)).tz_convert(tz).isoformat()
            rows = await store.get_pv_forecast_vs_actual(lower, version)
        finally:
            await store.close()
        corr = compute_pv_correction(rows, now, cfg, tz)
        if corr.applied and corr.factor <= 0.4:
            logger.warning(
                "PV intraday correction: factor %.2f (actual %.2f vs forecast %.2f kWh over "
                "%d slots) — heavy overcast, or a broken PV feed? Check the pv_power sensor.",
                corr.factor,
                corr.actual_kwh,
                corr.forecast_kwh,
                corr.evidence_slots,
            )
    except Exception as e:
        logger.warning("PV intraday correction failed (%s) — using factor 1.0", e)
        return PvCorrection(half_life_hours=2.0, reason="error")
    return corr
