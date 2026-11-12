#!/usr/bin/env python3
"""Compare PV residual training on the two weather sources before switching.

forecasting.pv_training_weather_source can be `forecast_past_days` (default: the
forecast endpoint's hindcast, max 92 days) or `historical_forecast` (Open-Meteo's
historical-forecast archive). The archive has been measured to give 8-18 % more
radiation, which shifts the physics baseline the residual is learned against, so
this script answers "does a model trained on the archive forecast better?" on the
site's own data before the option is enabled.

What it does, for each source on the SAME rows:
  1. reads slot_observations from the learning DB (opened READ-ONLY, mode=ro),
  2. fetches that source's weather for the window, computes the physics baseline,
     applies the trainer's sun-up filter and quality-flag cleaning,
  3. trains the p50 residual model on everything before the held-out tail,
  4. scores the tail the way production inference works: physics + residual, both
     fed with FORECAST-endpoint weather (inference never sees the archive).
It prints MAE and bias (forecast - actual, kWh per 15-min slot) overall, per month
and per hour, next to the physics-only baseline.

Usage:
    compare_pv_training_sources.py --db data/planner_learning.db
        [--config config.yaml] [--days 92] [--holdout-days 14] [--min-samples 100]

Limits (read them before deciding):
  - The forecast endpoint's hindcast is a proxy for the forecast that was actually
    issued for a slot; real day-ahead error is larger for every variant.
  - No HA access: vacation/alarm flags are 0 for all rows. The astro clamp and the
    3-slot smoothing of ml/forward.py are not applied.
  - Rows are compared only where BOTH sources have a physics baseline. With the
    default 92 days the archive's extra history is not used; the last line shows a
    third variant trained on all archive history since forecasting.pv_training_min_date.
  - Needs network access to Open-Meteo (two or three requests). Writes nothing:
    no models, no DB rows.
  - Nothing here addresses the known winter PV over-forecast (the weather model
    calling an overcast day sunny); a tail in a dark month mostly measures that.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytz
import yaml

from ml.cleaning import clean_training_rows
from ml.weather import (
    PV_WEATHER_SOURCE_FORECAST,
    PV_WEATHER_SOURCE_HISTORICAL,
    calculate_physics_pv,
    get_historical_weather_series,
    get_weather_series,
)

WEATHER_COLS = ("temp_c", "cloud_cover_pct", "shortwave_radiation_w_m2", "dni_w_m2", "dhi_w_m2")
BASE_FEATURES = [
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
PV_FEATURES = [*BASE_FEATURES, "physics_forecast_kwh"]


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open the learning DB so that no statement can modify it."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def load_pv_observations(
    db_path: str, tz: Any, start: datetime | None, max_kwh: float | None
) -> pd.DataFrame:
    """Read the rows ml/train.py would consider for PV (same SQL filters)."""
    query = (
        "SELECT slot_start, load_kwh, pv_kwh, quality_flags FROM slot_observations "
        "WHERE load_kwh > 0.001 AND pv_kwh IS NOT NULL"
    )
    params: list[Any] = []
    if max_kwh is not None:
        query += " AND load_kwh <= ? AND pv_kwh <= ?"
        params += [max_kwh, max_kwh]
    with open_readonly(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["slot_start"] = pd.to_datetime(df["slot_start"], format="ISO8601", utc=True, errors="coerce")
    df = df.dropna(subset=["slot_start"])
    df["slot_start"] = df["slot_start"].dt.tz_convert(tz)
    if start is not None:
        df = df[df["slot_start"] >= start]
    return df.sort_values("slot_start").drop_duplicates("slot_start").reset_index(drop=True)


def solar_setup(config: dict[str, Any]) -> tuple[list[dict[str, Any]], float, float]:
    system_cfg: dict[str, Any] = config.get("system", {}) or {}
    arrays: list[dict[str, Any]] = list(system_cfg.get("solar_arrays") or [])
    if not arrays and system_cfg.get("solar_array"):
        arrays = [system_cfg["solar_array"]]
    loc: dict[str, Any] = system_cfg.get("location", {}) or {}
    if not arrays or loc.get("latitude") is None or loc.get("longitude") is None:
        raise SystemExit("config needs system.solar_arrays and system.location.latitude/longitude")
    return arrays, float(loc["latitude"]), float(loc["longitude"])


def build_pv_frame(
    obs: pd.DataFrame, weather: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Observations + one source's weather -> features, physics and residual target.

    Mirrors ml/train.py: time features, sun-up filter, quality-flag cleaning, physics
    from DNI/DHI, rows without a physics baseline dropped. Indexed by slot_start.
    """
    from ml.train import _build_time_features

    arrays, latitude, longitude = solar_setup(config)
    df = obs.merge(weather, left_on="slot_start", right_index=True, how="left")
    for col in WEATHER_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vacation_mode_flag"] = 0.0
    df["alarm_armed_flag"] = 0.0
    df = _build_time_features(df)

    df = df[(df["shortwave_radiation_w_m2"] > 10) | (df["pv_kwh"] > 0.01)].copy()
    df, _ = clean_training_rows(df, "pv", log=lambda _msg: None)

    physics: list[float | None] = []
    for _, row in df.iterrows():
        kwh, _ = calculate_physics_pv(
            radiation_w_m2=row.get("shortwave_radiation_w_m2"),
            solar_arrays=arrays,  # type: ignore[arg-type]
            slot_start=row["slot_start"],
            latitude=latitude,
            longitude=longitude,
            dni_w_m2=row.get("dni_w_m2"),
            dhi_w_m2=row.get("dhi_w_m2"),
        )
        physics.append(kwh)
    df["physics_forecast_kwh"] = pd.Series(physics, index=df.index, dtype="float64")
    df = df[df["physics_forecast_kwh"].notna()].copy()
    df["pv_residual"] = df["pv_kwh"] - df["physics_forecast_kwh"]
    df = df.set_index("slot_start", drop=False)
    df.index.name = "slot"
    return df


def train_p50(frame: pd.DataFrame, min_samples: int) -> Any:
    from ml.train import _compute_sample_weights, _train_regressor

    return _train_regressor(
        frame[PV_FEATURES],
        frame["pv_residual"].astype(float),
        min_samples,
        alpha=0.5,
        sample_weight=_compute_sample_weights(frame),
    )


def predict(model: Any, inference_frame: pd.DataFrame) -> pd.Series:
    """physics + residual, floored at 0 — fed with the inference-side weather."""
    residual = np.asarray(model.predict(inference_frame[PV_FEATURES]), dtype=float)
    values = np.maximum(0.0, inference_frame["physics_forecast_kwh"].to_numpy() + residual)
    return pd.Series(values, index=inference_frame.index)


def score(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    err = pred - actual
    return {
        "n": float(len(err)),
        "mae": float(err.abs().mean()) if len(err) else float("nan"),
        "bias": float(err.mean()) if len(err) else float("nan"),
        "sum_pred": float(pred.sum()),
        "sum_actual": float(actual.sum()),
    }


def score_table(
    preds: dict[str, pd.Series], actual: pd.Series, bucket: pd.Series | None, label: str
) -> str:
    """One line per bucket, MAE and bias per variant."""
    names = list(preds)
    header = f"{label:>9} {'n':>6} {'actual':>8} " + " ".join(f"{name[:22]:>22}" for name in names)
    sub = f"{'':>9} {'':>6} {'kWh':>8} " + " ".join(f"{'MAE / bias':>22}" for _ in names)
    lines = [header, sub]
    groups: list[tuple[str, pd.Index]]
    if bucket is None:
        groups = [("all", actual.index)]
    else:
        groups = [(str(key), idx) for key, idx in bucket.groupby(bucket).groups.items()]
    for key, idx in groups:
        act = actual.loc[idx]
        cells = []
        for name in names:
            s = score(preds[name].loc[idx], act)
            cells.append(f"{s['mae']:>10.4f} / {s['bias']:>+9.4f}")
        lines.append(f"{key:>9} {len(act):>6} {act.sum():>8.1f} " + " ".join(cells))
    return "\n".join(lines)


def compare(
    obs: pd.DataFrame,
    weather: dict[str, pd.DataFrame],
    config: dict[str, Any],
    holdout_start: datetime,
    min_samples: int,
    extra_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train one p50 residual model per source on identical rows; score the tail.

    ``weather`` maps source name -> weather frame. ``extra_history`` optionally holds
    archive-weather rows OLDER than the comparison window for a third variant.
    Returns {"preds": {...}, "actual": Series, "rows": {...}} for the held-out tail.
    """
    frames = {name: build_pv_frame(obs, wx, config) for name, wx in weather.items()}
    common = frames[PV_WEATHER_SOURCE_FORECAST].index
    for frame in frames.values():
        common = common.intersection(frame.index)
    rows = {name: len(frame) for name, frame in frames.items()}
    rows["common"] = len(common)

    inference = frames[PV_WEATHER_SOURCE_FORECAST].loc[common]
    tail = inference[inference["slot_start"] >= holdout_start]
    actual = tail["pv_kwh"].astype(float)
    preds: dict[str, pd.Series] = {"physics_only": tail["physics_forecast_kwh"].clip(lower=0.0)}
    rows["train"] = int((inference["slot_start"] < holdout_start).sum())
    rows["holdout"] = len(tail)

    for name, frame in frames.items():
        train = frame.loc[common]
        train = train[train["slot_start"] < holdout_start]
        model = train_p50(train, min_samples)
        if model is None:
            continue
        preds[name] = predict(model, tail)

    if extra_history is not None and not extra_history.empty:
        hist = frames[PV_WEATHER_SOURCE_HISTORICAL].loc[common]
        full = pd.concat([extra_history, hist[hist["slot_start"] < holdout_start]])
        rows["train_archive_all_history"] = len(full)
        model = train_p50(full, min_samples)
        if model is not None:
            preds["archive_all_history"] = predict(model, tail)

    return {"preds": preds, "actual": actual, "rows": rows, "tail": tail}


def _parse_min_date(config: dict[str, Any], tz: Any) -> datetime | None:
    raw = (config.get("forecasting", {}) or {}).get("pv_training_min_date")
    if raw in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(raw))
    return tz.localize(parsed) if parsed.tzinfo is None else parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="Path to the learning SQLite DB (read-only)")
    parser.add_argument("--config", default="config.yaml", help="Config with location + arrays")
    parser.add_argument("--days", type=int, default=92, help="Comparison window (max 92)")
    parser.add_argument("--holdout-days", type=int, default=14, help="Held-out tail length")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args(argv)

    with Path(args.config).open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}
    tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
    max_kwh: float | None
    try:
        from backend.validation import get_max_energy_per_slot

        max_kwh = get_max_energy_per_slot(config)
    except Exception:
        max_kwh = None

    floor = _parse_min_date(config, tz)
    all_obs = load_pv_observations(args.db, tz, floor, max_kwh)
    if all_obs.empty:
        print("No usable PV observations in the DB.")
        return 1

    now = datetime.now(tz)
    end = min(all_obs["slot_start"].max().to_pydatetime() + timedelta(minutes=15), now)
    window_start = end - timedelta(days=min(args.days, 92))
    holdout_start = end - timedelta(days=args.holdout_days)
    obs = all_obs[all_obs["slot_start"] >= window_start].reset_index(drop=True)
    older = all_obs[all_obs["slot_start"] < window_start].reset_index(drop=True)

    print(f"DB (read-only): {args.db}")
    print(
        f"Window: {window_start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M}  "
        f"held-out tail from {holdout_start:%Y-%m-%d %H:%M}"
    )
    print(f"PV floor (forecasting.pv_training_min_date): {floor.date() if floor else 'none'}")

    weather = {
        PV_WEATHER_SOURCE_FORECAST: get_weather_series(window_start, end, config=config),
        PV_WEATHER_SOURCE_HISTORICAL: get_historical_weather_series(
            window_start, end, config=config
        ),
    }
    for name, frame in weather.items():
        if frame.empty:
            print(f"Weather source {name} returned nothing; cannot compare.")
            return 1

    extra: pd.DataFrame | None = None
    if not older.empty:
        older_wx = get_historical_weather_series(
            older["slot_start"].min().to_pydatetime(), window_start, config=config
        )
        if not older_wx.empty:
            extra = build_pv_frame(older, older_wx, config)

    both = weather[PV_WEATHER_SOURCE_FORECAST].join(
        weather[PV_WEATHER_SOURCE_HISTORICAL], lsuffix="_fc", rsuffix="_hist", how="inner"
    )
    print("\nRadiation totals over the window (archive / hindcast):")
    for col in ("shortwave_radiation_w_m2", "dni_w_m2", "dhi_w_m2"):
        pair = both[[f"{col}_fc", f"{col}_hist"]].dropna()
        total_fc = float(pair[f"{col}_fc"].sum())
        ratio = float(pair[f"{col}_hist"].sum()) / total_fc if total_fc > 0 else float("nan")
        print(f"  {col:<26} {ratio:6.3f}  ({len(pair)} slots)")

    result = compare(obs, weather, config, holdout_start, args.min_samples, extra)
    rows = result["rows"]
    print("\nRows with a physics baseline: " + ", ".join(f"{k}={v}" for k, v in rows.items()))
    preds: dict[str, pd.Series] = result["preds"]
    actual: pd.Series = result["actual"]
    if len(preds) < 3 or actual.empty:
        print(
            "Not enough rows to train both variants and score a tail "
            "(try a shorter --holdout-days or a lower --min-samples)."
        )
        return 1

    tail: pd.DataFrame = result["tail"]
    print("\nUnits: kWh per 15-min slot. bias = forecast - actual (positive = over-forecast).")
    print("\n== Overall ==")
    print(score_table(preds, actual, None, "bucket"))
    print(
        "\n   total kWh over the tail: actual "
        f"{actual.sum():.1f}, " + ", ".join(f"{k} {v.sum():.1f}" for k, v in preds.items())
    )
    print("\n== Per month ==")
    print(score_table(preds, actual, tail["slot_start"].dt.strftime("%Y-%m"), "month"))
    print("\n== Per hour (local) ==")
    print(score_table(preds, actual, tail["slot_start"].dt.strftime("%H"), "hour"))
    print(
        "\nReading it: enable historical_forecast only if its column beats "
        "forecast_past_days on MAE without a worse bias, in the hours that carry "
        "the energy. Neither column fixes the winter over-forecast."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
