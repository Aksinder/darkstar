"""
Main entry point for calculating forecasted states for the next window (Aurora).

Supports hybrid PV forecasting: physics base + ML residual + corrector.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import pandas as pd

from backend.health import clear_load_forecast_status, set_load_forecast_status
from backend.learning import LearningEngine, get_learning_engine
from ml.context_features import get_alarm_armed_series, get_vacation_mode_series
from ml.train import _build_time_features  # type: ignore[reportPrivateUsage]
from ml.weather import async_get_weather_series, calculate_physics_pv
from utils.time_utils import dst_safe_date_range


def determine_graduation_level(engine: LearningEngine) -> tuple[int, str, float]:
    """Determine graduation level based on available training data."""
    import sqlite3
    from datetime import datetime, timedelta

    tz = engine.timezone
    cutoff = (datetime.now(tz) - timedelta(days=90)).date().isoformat()

    try:
        with sqlite3.connect(engine.db_path, timeout=30.0) as conn:
            query = """
                SELECT COUNT(DISTINCT DATE(o.slot_start))
                FROM slot_observations o
                JOIN slot_forecasts f ON o.slot_start = f.slot_start
                WHERE DATE(o.slot_start) >= ?
                  AND o.load_kwh IS NOT NULL
                  AND f.load_forecast_kwh IS NOT NULL
            """
            row = conn.execute(query, (cutoff,)).fetchone()
            days = int(row[0] or 0) if row else 0
    except Exception:
        days = 0

    if days < 4:
        return 0, "infant", float(days)
    elif days < 14:
        return 1, "statistician", float(days)
    else:
        return 2, "graduate", float(days)


logger = logging.getLogger("darkstar.ml.forward")


_EXPECTED_LOAD_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "temp_c",
    "cloud_cover_pct",
    "shortwave_radiation_w_m2",
    "vacation_mode_flag",
    "alarm_armed_flag",
]
_EXPECTED_PV_FEATURES = [*_EXPECTED_LOAD_FEATURES, "physics_forecast_kwh"]


def _load_models(models_dir: str = "data/ml/models") -> dict[str, lgb.Booster]:
    """Load trained LightGBM models for AURORA forward inference (Probabilistic).

    Returns:
        dict mapping model names to Booster objects.
        Empty dict if no models could be loaded.
    """
    import json
    from pathlib import Path as _Path

    models: dict[str, lgb.Booster] = {}

    def _validate_features(model_path: str, model_key: str) -> bool:
        features_file = _Path(model_path).with_suffix(".features.json")
        if not features_file.exists():
            return True
        try:
            stored = json.loads(features_file.read_text()).get("feature_names", [])
        except Exception:
            return True
        expected = _EXPECTED_PV_FEATURES if model_key.startswith("pv") else _EXPECTED_LOAD_FEATURES
        if sorted(stored) != sorted(expected):
            logger.warning(
                "Feature mismatch for %s: stored=%s expected=%s; falling back to baseline",
                model_path,
                stored,
                expected,
            )
            return False
        return True

    # Quantiles to load
    quantiles = ["p10", "p50", "p90"]

    # Load Load Models
    for q in quantiles:
        # Try specific quantile file first
        path = f"{models_dir}/load_model_{q}.lgb"
        model_key = f"load_{q}"
        try:
            booster = lgb.Booster(model_file=path)
            if _validate_features(path, model_key):
                models[model_key] = booster
        except Exception:
            # Fallback for p50: try legacy name
            if q == "p50":
                fallback_path = f"{models_dir}/load_model.lgb"
                try:
                    booster = lgb.Booster(model_file=fallback_path)
                    if _validate_features(fallback_path, model_key):
                        models[model_key] = booster
                except Exception as exc:
                    logger.debug(f"Could not load load_model ({q}): {exc}")
            else:
                logger.debug(f"Could not load load_model_{q}.lgb")

    # Load PV Models
    for q in quantiles:
        path = f"{models_dir}/pv_model_{q}.lgb"
        model_key = f"pv_{q}"
        try:
            booster = lgb.Booster(model_file=path)
            if _validate_features(path, model_key):
                models[model_key] = booster
        except Exception:
            if q == "p50":
                fallback_path = f"{models_dir}/pv_model.lgb"
                try:
                    booster = lgb.Booster(model_file=fallback_path)
                    if _validate_features(fallback_path, model_key):
                        models[model_key] = booster
                except Exception as exc:
                    logger.debug(f"Could not load pv_model ({q}): {exc}")
            else:
                logger.debug(f"Could not load pv_model_{q}.lgb")

    # REV PERS2: Log CRITICAL if no models loaded (planner will fail silently otherwise)
    if not models:
        logger.critical(
            "❌ NO ML MODELS LOADED from %s! "
            "Forecasting will use fallback (Open-Meteo for PV, baseline avg for Load). "
            "Train models or ensure baseline models are deployed.",
            models_dir,
        )
    else:
        logger.info(f"✅ Loaded {len(models)} ML models from {models_dir}")

    return models


async def generate_forward_slots(
    horizon_hours: int = 168,
    forecast_version: str = "aurora",
) -> None:
    """
    Generate forward AURORA forecasts for the next horizon_hours.
    Includes probabilistic bands (p10, p50, p90).
    """
    engine = get_learning_engine()
    assert isinstance(engine, LearningEngine)

    tz = engine.timezone
    now = datetime.now(tz)

    # Align to current 15-minute slot boundary (matches price slot timestamps)
    # Critical fix: previously aligned to NEXT boundary, causing first forecast
    # slot to have no data. Part of "belt and suspenders" approach:
    # 1. This fix aligns ML output to current boundary
    # 2. recorder_service.py retries with 5s delay on observation gaps
    # 3. inputs.py interpolates small gaps (1-2 slots) as defensive fallback
    minutes = (now.minute // 15) * 15
    slot_start = now.replace(minute=minutes, second=0, microsecond=0)

    horizon_end = slot_start + timedelta(hours=horizon_hours)

    print(f"🔮 Generating AURORA Forecast: {slot_start} -> {horizon_end} ({horizon_hours}h)")

    slots = dst_safe_date_range(
        start=slot_start,
        end=horizon_end,
        freq="15min",
        tz=tz,
        inclusive="left",
    )
    if len(slots) == 0:
        print("No future slots to forecast.")
        return

    df = pd.DataFrame({"slot_start": slots})

    # Enrich with forecast weather
    print("   Fetching weather data...")
    weather_df = await async_get_weather_series(slot_start, horizon_end, config=engine.config)
    if not weather_df.empty:
        df = df.merge(weather_df, left_on="slot_start", right_index=True, how="left")

    # Ensure ALL weather columns exist (even if empty) to match trained model feature count.
    # dni_w_m2/dhi_w_m2 are PHYSICS inputs (not ML features) — included here only so a missing
    # weather fetch degrades to the legacy GHI transposition instead of raising KeyError below.
    for col in (
        "temp_c",
        "cloud_cover_pct",
        "shortwave_radiation_w_m2",
        "dni_w_m2",
        "dhi_w_m2",
    ):
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = df[col].astype("float64")

    # Context flags
    vac_series = get_vacation_mode_series(
        slot_start - timedelta(days=7), horizon_end, config=engine.config
    )
    if not vac_series.empty:
        df = df.merge(
            vac_series.to_frame(name="vacation_mode_flag"),
            left_on="slot_start",
            right_index=True,
            how="left",
        )
    else:
        df["vacation_mode_flag"] = 0.0

    alarm_series = get_alarm_armed_series(
        slot_start - timedelta(days=7), horizon_end, config=engine.config
    )
    if not alarm_series.empty:
        df = df.merge(
            alarm_series.to_frame(name="alarm_armed_flag"),
            left_on="slot_start",
            right_index=True,
            how="left",
        )
    else:
        df["alarm_armed_flag"] = 0.0

    df = _build_time_features(df)

    # All 11 features required by trained models - we ensure all columns exist above
    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "temp_c",
        "cloud_cover_pct",
        "shortwave_radiation_w_m2",
        "vacation_mode_flag",
        "alarm_armed_flag",
    ]

    logger.info("   Running LightGBM inference (Probabilistic)...")
    X = df[feature_cols]
    models = _load_models()

    quantiles = ["p10", "p50", "p90"]
    predictions: dict[str, pd.Series] = {}

    # Initialize prediction series map
    for q in quantiles:
        predictions[f"load_{q}"] = pd.Series(0.0, index=df.index)
        predictions[f"pv_{q}"] = pd.Series(0.0, index=df.index)

    # REV PERS2: Fallback logic when no ML models available
    has_load_models = any(f"load_{q}" in models for q in quantiles)
    has_pv_models = any(f"pv_{q}" in models for q in quantiles)

    # Config-gated physics-only PV (atomicity fix for the DNI/DHI physics correction).
    # The live PV residual models were trained with residual = actual - physics_OLD
    # (the under-forecasting GHI transposition). Once physics is corrected but before a
    # min-date retrain re-learns the residual, adding the stale positive residual on top
    # of the raised physics would DOUBLE-correct and over-forecast mornings. Setting
    # forecasting.pv_residual_enabled: false forces the physics-only band on the corrected
    # physics until that retrain. Default true preserves the upstream hybrid behaviour.
    forecasting_cfg: dict[str, Any] = engine.config.get("forecasting", {}) or {}
    pv_residual_enabled = bool(forecasting_cfg.get("pv_residual_enabled", True))
    pv_forced_physics_only = False
    if not pv_residual_enabled and has_pv_models:
        logger.info(
            "⚠️ PV residual disabled (forecasting.pv_residual_enabled=false): "
            "running physics-only on corrected physics; PV residual models NOT applied "
            "(interim until min-date retrain)."
        )
        has_pv_models = False
        pv_forced_physics_only = True

    # --- LOAD INFERENCE (or fallback) ---
    if has_load_models:
        for q in quantiles:
            model_key = f"load_{q}"
            if model_key in models:
                raw_pred: Any = models[model_key].predict(X)  # type: ignore[reportUnknownMemberType]
                # Apply guardrails (same for all bands)
                # Floor at 0.01, Ceiling at 16kW
                cleaned = [max(0.01, min(float(x), 16.0)) for x in raw_pred]
                predictions[model_key] = pd.Series(cleaned, index=df.index)
        # REV F65 Phase 5b: Clear degraded status when ML models working
        clear_load_forecast_status()
    else:
        # Fallback: Write 0.0 to DB so inputs.py applies HA 7-day profile fallback
        # Only use 0.5 flat as last resort when even HA fetch fails
        logger.warning(
            "⚠️ Load models not available, using 0.0 (inputs.py will apply HA profile fallback)"
        )

        # REV F65 Phase 5e: Distinguish new setup vs ML failure
        level, _, _ = determine_graduation_level(engine)
        if level == 0:
            # New setup (< 4 days) - expected state, info level
            set_load_forecast_status("degraded", "baseline")
        else:
            # Level 1+ but no ML models - warning, should have models
            set_load_forecast_status("degraded", "no_ml")

        baseline_load = 0.0  # Let inputs.py apply HA profile fallback
        for q in quantiles:
            if q == "p10":
                predictions[f"load_{q}"] = pd.Series(baseline_load * 0.7, index=df.index)
            elif q == "p50":
                predictions[f"load_{q}"] = pd.Series(baseline_load, index=df.index)
            else:  # p90
                predictions[f"load_{q}"] = pd.Series(baseline_load * 1.3, index=df.index)

    # --- PV INFERENCE (Hybrid: Physics + ML Residual) ---
    # Setup Astro Clamping
    sun_calc = None
    try:
        from backend.astro import SunCalculator

        system_cfg: dict[str, Any] = engine.config.get("system", {})
        location_cfg: dict[str, Any] = system_cfg.get("location", {})
        lat: float = location_cfg.get("latitude", 59.3293)
        lon: float = location_cfg.get("longitude", 18.0686)
        sun_calc = SunCalculator(latitude=lat, longitude=lon, timezone=str(tz))
    except Exception as e:
        logger.warning(f"⚠️ Astro init failed: {e}")

    # Get solar arrays config for physics calculation
    system_config: dict[str, Any] = engine.config.get("system", {})
    solar_arrays: list[Any] = system_config.get("solar_arrays", [])
    loc_cfg: dict[str, Any] = system_config.get("location", {}) or {}
    physics_lat = float(loc_cfg.get("latitude", 59.3))
    physics_lon = float(loc_cfg.get("longitude", 18.1))

    # Fallback to legacy single array
    if not solar_arrays:
        legacy_cfg: dict[str, Any] = system_config.get("solar_array", {}) or {}
        if legacy_cfg:
            solar_arrays = [legacy_cfg]

    # Calculate physics base forecast for all slots
    physics_series = pd.Series(0.0, index=df.index)
    for idx, row in df.iterrows():
        slot_ts = row["slot_start"]
        radiation = row.get("shortwave_radiation_w_m2")

        physics_kwh, _ = calculate_physics_pv(
            radiation_w_m2=radiation,
            solar_arrays=solar_arrays,  # type: ignore[arg-type]
            slot_start=slot_ts,
            latitude=physics_lat,
            longitude=physics_lon,
            dni_w_m2=row.get("dni_w_m2"),
            dhi_w_m2=row.get("dhi_w_m2"),
        )
        physics_series.loc[idx] = physics_kwh if physics_kwh is not None else 0.0  # type: ignore[reportIndexIssue]

    # Store physics for output
    predictions["physics_kwh"] = physics_series

    if has_pv_models:
        # HYBRID MODE: ML predicts residual, final = physics + residual
        # Add physics as feature for ML model
        df["physics_forecast_kwh"] = physics_series

        # Feature columns for PV residual model (includes physics)
        pv_feature_cols = feature_cols.copy()
        if "physics_forecast_kwh" not in pv_feature_cols:
            pv_feature_cols.append("physics_forecast_kwh")

        X_pv = df[pv_feature_cols]

        for q in quantiles:
            model_key = f"pv_{q}"
            if model_key in models:
                raw_pred: Any = models[model_key].predict(X_pv)  # type: ignore[reportUnknownMemberType]

                series: pd.Series = pd.Series(0.0, index=df.index)
                for pos_idx, (idx, row) in enumerate(df.iterrows()):
                    # ML predicts residual (could be negative)
                    ml_residual = float(raw_pred[pos_idx])
                    physics = float(physics_series.iloc[pos_idx])

                    # Final = physics + residual
                    val = physics + ml_residual

                    # 1. Astro Clamp
                    is_sun_up = False
                    if sun_calc:
                        is_sun_up = sun_calc.is_sun_up(row["slot_start"], buffer_minutes=30)
                    else:
                        h = row["slot_start"].hour
                        is_sun_up = 5 <= h < 22

                    if not is_sun_up:
                        val = 0.0

                    # 2. Radiation Clamp
                    rad = row.get("shortwave_radiation_w_m2")
                    if rad is not None and rad < 1.0:
                        val = 0.0

                    # Floor at 0
                    val = max(0.0, val)
                    series.loc[idx] = val  # type: ignore[reportIndexIssue]

                # 3. Smoothing
                predictions[model_key] = (
                    series.rolling(window=3, center=True, min_periods=1).mean().fillna(0.0)
                )

                # Store ML residual for transparency
                predictions[f"ml_residual_{q}"] = pd.Series(raw_pred, index=df.index)

        logger.info("✅ PV: Using hybrid mode (physics + ML residual)")
    else:
        # PHYSICS-ONLY MODE: use physics directly (either no ML models, or the
        # pv_residual_enabled gate intentionally disabled them — already logged above).
        if not pv_forced_physics_only:
            logger.warning("⚠️ PV models not available, using physics-only mode")
        for q in quantiles:
            # Apply uncertainty bands around physics
            if q == "p10":
                predictions[f"pv_{q}"] = physics_series * 0.8
            elif q == "p50":
                predictions[f"pv_{q}"] = physics_series
            else:  # p90
                predictions[f"pv_{q}"] = physics_series * 1.2

            # Apply smoothing
            predictions[f"pv_{q}"] = (
                predictions[f"pv_{q}"]
                .rolling(window=3, center=True, min_periods=1)
                .mean()
                .fillna(0.0)  # type: ignore[union-attr]
            )

            # Zero out nighttime
            for idx, row in df.iterrows():
                is_sun_up = False
                if sun_calc:
                    is_sun_up = sun_calc.is_sun_up(row["slot_start"], buffer_minutes=30)
                else:
                    h = row["slot_start"].hour
                    is_sun_up = 5 <= h < 22
                if not is_sun_up:
                    predictions[f"pv_{q}"].loc[idx] = 0.0  # type: ignore[reportIndexIssue]

            predictions[f"ml_residual_{q}"] = pd.Series(0.0, index=df.index)

    # Repair applies only when all 3 quantile models were predicted (not default-zero placeholders)
    _load_all_q = all(f"load_{q}" in models for q in quantiles)
    # has_pv_models may have been forced False by the physics-only gate above, in which
    # case the physics-only bands (already monotonic) are used rather than model output.
    _pv_all_q = has_pv_models and all(f"pv_{q}" in models for q in quantiles)

    # --- STORE RESULTS ---
    forecasts: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        # Repair: ensure p10 ≤ p50 ≤ p90 before storage (only when all 3 were predicted)
        if _load_all_q:
            load_p10_v, load_p50_v, load_p90_v = sorted(
                [
                    float(predictions["load_p10"][idx]),  # type: ignore[reportUnknownArgumentType]
                    float(predictions["load_p50"][idx]),  # type: ignore[reportUnknownArgumentType]
                    float(predictions["load_p90"][idx]),  # type: ignore[reportUnknownArgumentType]
                ]
            )
        else:
            load_p10_v = float(predictions["load_p10"][idx])  # type: ignore[reportUnknownArgumentType]
            load_p50_v = float(predictions["load_p50"][idx])  # type: ignore[reportUnknownArgumentType]
            load_p90_v = float(predictions["load_p90"][idx])  # type: ignore[reportUnknownArgumentType]

        if _pv_all_q:
            pv_p10_v, pv_p50_v, pv_p90_v = sorted(
                [
                    float(predictions["pv_p10"][idx]),  # type: ignore[reportUnknownArgumentType]
                    float(predictions["pv_p50"][idx]),  # type: ignore[reportUnknownArgumentType]
                    float(predictions["pv_p90"][idx]),  # type: ignore[reportUnknownArgumentType]
                ]
            )
        else:
            pv_p10_v = float(predictions["pv_p10"][idx])  # type: ignore[reportUnknownArgumentType]
            pv_p50_v = float(predictions["pv_p50"][idx])  # type: ignore[reportUnknownArgumentType]
            pv_p90_v = float(predictions["pv_p90"][idx])  # type: ignore[reportUnknownArgumentType]
        item = {
            "slot_start": row["slot_start"].isoformat(),
            "temp_c": row.get("temp_c"),
            # Primary (Legacy/p50)
            "pv_forecast_kwh": pv_p50_v,
            "load_forecast_kwh": load_p50_v,
            "base_load_forecast_kwh": load_p50_v,
            # Probabilistic Bands (monotonic after repair)
            "pv_p10": pv_p10_v,
            "pv_p90": pv_p90_v,
            "load_p10": load_p10_v,
            "load_p90": load_p90_v,
            "base_load_p10": load_p10_v,
            "base_load_p90": load_p90_v,
        }
        forecasts.append(item)

    if forecasts:
        await engine.store_forecasts(forecasts, forecast_version=forecast_version)
        print(f"✅ Stored {len(forecasts)} forward AURORA forecasts ({forecast_version}).")

        # Write the naive 7-day-average baseline for the SAME slots so the eval
        # endpoints' aurora-vs-baseline A/B finally has a comparator (best-effort;
        # never blocks the aurora path).
        if forecast_version == "aurora":
            from backend.learning.baseline import store_baseline_forecasts

            baseline_n = await store_baseline_forecasts(
                engine.store, [str(f["slot_start"]) for f in forecasts]
            )
            if baseline_n:
                print(f"✅ Stored {baseline_n} baseline_7_day_avg forecasts.")

        # Log physics vs ML breakdown for monitoring
        if "physics_kwh" in predictions:
            total_physics = float(predictions["physics_kwh"].sum())
            total_pv_p50 = float(predictions["pv_p50"].sum())
            if total_physics > 0:
                ml_residual_total = total_pv_p50 - total_physics
                physics_pct = (total_physics / total_pv_p50 * 100) if total_pv_p50 > 0 else 0
                logger.info(
                    f"📊 PV Forecast Breakdown: Physics={total_physics:.2f}kWh ({physics_pct:.1f}%), "
                    f"ML Residual={ml_residual_total:.2f}kWh"
                )


if __name__ == "__main__":
    import asyncio

    asyncio.run(generate_forward_slots())
