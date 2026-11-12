#!/usr/bin/env python3
"""Compare the stored load forecasts against what was actually recorded.

Answers, on the site's own learning DB, the questions behind
forecasting.load_blend_slot_mean_weight and forecasting.load_safety_margin_percent:
  - is the Aurora load model biased, and at which hours / weekdays?
  - would a naive 7-day slot mean do better, and would a blend of the two?
  - how much of the load floor (117 % here) is compensating a low model, i.e. what
    floor goes with which blend weight?

Compared per 15-min slot, against recorded base load (slot_observations.load_kwh):
  aurora          stored forecast_version 'aurora' (base_load_forecast_kwh)
  baseline_7d     stored forecast_version 'baseline_7_day_avg' (eval comparator; it
                  counts zero/NULL load as 0, so it reads slightly low)
  slot_mean       the mean the blend really uses: same quarter-hour over the 7 local
                  days before the slot's day, only real (> 0.001 kWh) unflagged loads,
                  at least 3 of them — recomputed here from the observations
  blend w         (1-w) * aurora + w * slot_mean  (aurora alone where no slot mean)
Each is shown as stored ("raw") and multiplied by the floor ("x1.17"): the planner
never plans for less than floor % of the forecast.

Usage:
    compare_load_forecasts.py --db data/planner_learning.db
        [--config config.yaml] [--days 60] [--floor 117] [--weights 0.2,0.4,0.6,0.8]
        [--timezone Europe/Stockholm] [--immutable]

The DB is opened read-only (mode=ro) and closed again; its contents are never
modified and no other file is written by this script. But mode=ro on a WAL-mode
database still makes SQLite create -shm/-wal files beside it. To avoid even that, run
against a COPY of the DB with --immutable (never use --immutable on the live DB).

Limits:
  - 'aurora' rows are whatever the forecaster STORED, which is not always the pure
    model: while the load model was unusable they hold the 7-day slot mean (fallback
    tier), and once load_blend_slot_mean_weight is > 0 they hold the BLENDED value. So
    run this BEFORE enabling the blend; afterwards "aurora" already contains the slot
    mean and the "blend w" columns blend it a second time.
  - 'aurora' rows are the LAST forecast stored for a slot (minutes old), not the
    day-ahead one the plan was built on; real planning error is larger.
  - slot_mean is recomputed per calendar day, the live blend per forecast run.
  - Slots with no real recorded load, or flagged by the recorder as unmeasured for
    the load target, are excluded. Slots with a balance-derived load
    (load_rescued/repaired, common on sunny quarters) are kept and counted below.
  - The floor is the MINIMUM margin; the S-index can plan above it.
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from backend.learning.baseline import BASELINE_VERSION, MIN_SLOT_MEAN_OBSERVATIONS
from ml.cleaning import DERIVED_LOAD_FLAG_KEYS, bad_row_mask, flag_mask

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def open_readonly(db_path: str, immutable: bool = False) -> sqlite3.Connection:
    """Open the learning DB so that no statement can modify its contents.

    mode=ro forbids every write to the database, but on a WAL-mode database SQLite
    still creates (and may leave) ``-shm``/``-wal`` files beside it. ``immutable``
    adds immutable=1: no locks, no sidecar files at all. That is only safe on a COPY
    nobody else writes to — on the live DB it can return inconsistent data — and it
    ignores an un-checkpointed ``-wal`` file. The caller must close the connection.
    """
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def _to_local(series: pd.Series, tz: str) -> pd.Series:
    parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    return parsed.dt.tz_convert(tz)


def load_frames(
    db_path: str, tz: str, immutable: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (observations, forecasts-wide) with tz-aware local slot_start."""
    with contextlib.closing(open_readonly(db_path, immutable)) as conn:
        obs = pd.read_sql_query(
            "SELECT slot_start, load_kwh, quality_flags FROM slot_observations", conn
        )
        fc = pd.read_sql_query(
            "SELECT slot_start, forecast_version, load_forecast_kwh, base_load_forecast_kwh "
            "FROM slot_forecasts WHERE forecast_version IN (?, ?)",
            conn,
            params=("aurora", BASELINE_VERSION),
        )
    obs["slot_start"] = _to_local(obs["slot_start"], tz)
    obs = obs.dropna(subset=["slot_start"]).drop_duplicates("slot_start")
    obs["load_kwh"] = pd.to_numeric(obs["load_kwh"], errors="coerce")

    fc["slot_start"] = _to_local(fc["slot_start"], tz)
    fc = fc.dropna(subset=["slot_start"])
    base = pd.to_numeric(fc["base_load_forecast_kwh"], errors="coerce")
    total = pd.to_numeric(fc["load_forecast_kwh"], errors="coerce")
    # Same preference as backend/core/forecasts.py: base load, else the legacy column.
    fc["value"] = base.where(base > 0, total)
    wide = fc.pivot_table(index="slot_start", columns="forecast_version", values="value")
    wide = wide.rename(columns={"aurora": "aurora", BASELINE_VERSION: "baseline_7d"})
    for col in ("aurora", "baseline_7d"):
        if col not in wide.columns:
            wide[col] = float("nan")
    return obs.sort_values("slot_start").reset_index(drop=True), wide.reset_index()


def usable_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """Rows fit to score against / average: real load, not flagged for the load target."""
    good = obs[obs["load_kwh"] > 0.001]
    return good[~bad_row_mask(good, "load")].copy()


def rolling_slot_means(
    good: pd.DataFrame, min_observations: int = MIN_SLOT_MEAN_OBSERVATIONS
) -> pd.Series:
    """Per slot: mean load of the same quarter-hour over the 7 local days before its day.

    Returns a Series indexed like ``good`` (NaN where fewer than ``min_observations``
    earlier samples exist). Mirrors backend.learning.baseline.load_slot_means.
    """
    frame = good[["slot_start", "load_kwh"]].copy()
    frame["day"] = frame["slot_start"].dt.tz_localize(None).dt.normalize()
    frame["key"] = frame["slot_start"].dt.strftime("%H:%M")
    daily = frame.groupby(["key", "day"])["load_kwh"].agg(["sum", "count"]).reset_index()

    out = pd.Series(float("nan"), index=good.index, dtype="float64")
    by_key = {key: grp.set_index("day") for key, grp in daily.groupby("key")}
    for idx, key, day in zip(frame.index, frame["key"], frame["day"], strict=True):
        hist = by_key[key]
        window = hist[(hist.index >= day - timedelta(days=7)) & (hist.index < day)]
        n = int(window["count"].sum())
        if n >= min_observations:
            out.loc[idx] = float(window["sum"].sum()) / n
    return out


def build_comparison(
    obs: pd.DataFrame, forecasts: pd.DataFrame, weights: list[float]
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Join everything into one scored frame. Returns (frame, variant columns, counts)."""
    good = usable_observations(obs)
    good["slot_mean"] = rolling_slot_means(good)
    frame = good.merge(forecasts, on="slot_start", how="inner")
    counts = {
        "observations": len(obs),
        "usable_observations": len(good),
        "with_aurora_forecast": int(frame["aurora"].notna().sum()),
    }
    frame = frame[frame["aurora"] > 0.001].copy()
    counts["scored"] = len(frame)
    counts["scored_with_slot_mean"] = int(frame["slot_mean"].notna().sum())
    counts["scored_balance_derived_load"] = int(flag_mask(frame, DERIVED_LOAD_FLAG_KEYS).sum())

    variants = ["aurora", "baseline_7d", "slot_mean"]
    for w in weights:
        name = f"blend {w:g}"
        mean = frame["slot_mean"].fillna(frame["aurora"])
        frame[name] = ((1 - w) * frame["aurora"] + w * mean).clip(lower=0.01, upper=16.0)
        variants.append(name)
    frame["hour"] = frame["slot_start"].dt.strftime("%H")
    frame["weekday"] = frame["slot_start"].dt.dayofweek
    return frame, variants, counts


def stats(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    """MAE, bias (forecast - actual), bias in % of actual energy, share of slots under."""
    mask = pred.notna()
    pred, actual = pred[mask], actual[mask]
    if pred.empty:
        return {
            "n": 0,
            "mae": float("nan"),
            "bias": float("nan"),
            "bias_pct": float("nan"),
            "under": float("nan"),
            "zero_bias_floor": float("nan"),
        }
    err = pred - actual
    total_actual = float(actual.sum())
    total_pred = float(pred.sum())
    return {
        "n": len(err),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "bias_pct": 100.0 * float(err.sum()) / total_actual if total_actual > 0 else float("nan"),
        "under": 100.0 * float((err < 0).mean()),
        # floor % at which forecast energy equals recorded energy
        "zero_bias_floor": 100.0 * total_actual / total_pred if total_pred > 0 else float("nan"),
    }


def overall_table(frame: pd.DataFrame, variants: list[str], floor_pct: float) -> str:
    factor = floor_pct / 100.0
    head = (
        f"{'variant':<12} {'n':>6} | {'MAE':>7} {'bias':>8} {'bias%':>7} {'under%':>7} | "
        f"x{factor:.2f}: {'MAE':>7} {'bias':>8} {'bias%':>7} {'under%':>7} | {'0-bias floor':>12}"
    )
    lines = [head, "-" * len(head)]
    for name in variants:
        raw = stats(frame[name], frame["load_kwh"])
        fl = stats(frame[name] * factor, frame["load_kwh"])
        lines.append(
            f"{name:<12} {raw['n']:>6} | {raw['mae']:>7.4f} {raw['bias']:>+8.4f} "
            f"{raw['bias_pct']:>+6.1f}% {raw['under']:>6.1f}% |        "
            f"{fl['mae']:>7.4f} {fl['bias']:>+8.4f} {fl['bias_pct']:>+6.1f}% {fl['under']:>6.1f}% | "
            f"{raw['zero_bias_floor']:>11.1f}%"
        )
    return "\n".join(lines)


def bucket_table(
    frame: pd.DataFrame,
    variants: list[str],
    bucket: str,
    labels: dict[Any, str] | None,
    factor: float = 1.0,
) -> str:
    head = f"{bucket:>8} {'n':>6} {'actual':>8} " + " ".join(f"{name:>17}" for name in variants)
    sub = f"{'':>8} {'':>6} {'kWh/slot':>8} " + " ".join(f"{'MAE / bias':>17}" for _ in variants)
    lines = [head, sub]
    for key, group in frame.groupby(bucket):
        cells = []
        for name in variants:
            s = stats(group[name] * factor, group["load_kwh"])
            cells.append(f"{s['mae']:>7.3f} / {s['bias']:>+7.3f}")
        label = labels[key] if labels else str(key)
        lines.append(
            f"{label:>8} {len(group):>6} {group['load_kwh'].mean():>8.3f} " + " ".join(cells)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="Path to the learning SQLite DB (read-only)")
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="Open with immutable=1 (no -shm/-wal files created beside the DB). ONLY for "
        "a COPY of the database that nothing else is writing to; never the live DB.",
    )
    parser.add_argument("--config", default=None, help="Optional config for timezone and floor")
    parser.add_argument("--days", type=int, default=60, help="Score the last N days (0 = all)")
    parser.add_argument(
        "--floor", type=float, default=None, help="Load floor in %% (default: config, else 117)"
    )
    parser.add_argument(
        "--weights", default="0.2,0.4,0.6,0.8", help="Blend weights of the slot mean"
    )
    parser.add_argument("--timezone", default=None)
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    if args.config:
        with Path(args.config).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    tz = args.timezone or config.get("timezone") or "Europe/Stockholm"
    floor_pct = args.floor
    if floor_pct is None:
        floor_pct = float(
            (config.get("forecasting", {}) or {}).get("load_safety_margin_percent") or 117.0
        )
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    if any(not 0.0 <= w <= 1.0 for w in weights):
        parser.error("--weights must be between 0 and 1")

    obs, forecasts = load_frames(args.db, tz, args.immutable)
    frame, variants, counts = build_comparison(obs, forecasts, weights)
    if args.days > 0 and not frame.empty:
        frame = frame[frame["slot_start"] >= frame["slot_start"].max() - timedelta(days=args.days)]
    if frame.empty:
        print("No slots with both a recorded load and a stored aurora forecast.")
        return 1

    print(f"DB (read-only): {args.db}")
    print(
        f"Scored window: {frame['slot_start'].min():%Y-%m-%d} .. "
        f"{frame['slot_start'].max():%Y-%m-%d} ({len(frame)} slots, timezone {tz})"
    )
    print("Row counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(
        "NOTE: 'aurora' is the stored forecast, not always the pure model: it holds the "
        "7-day slot mean\nfor periods without a usable load model, and the blended value "
        "once the blend is enabled."
    )
    print("\nUnits: kWh per 15-min slot. bias = forecast - actual (negative = under-forecast).")
    print(
        "under% = share of slots forecast below the recorded load. "
        "0-bias floor = floor % at which forecast energy equals recorded energy."
    )

    print(f"\n== Overall (raw, and with the {floor_pct:g} % floor) ==")
    print(overall_table(frame, variants, floor_pct))
    print("\n== Per hour (local), raw ==")
    print(bucket_table(frame, variants, "hour", None))
    print(f"\n== Per hour (local), with the {floor_pct:g} % floor ==")
    print(bucket_table(frame, variants, "hour", None, floor_pct / 100.0))
    weekday_labels = dict(enumerate(WEEKDAYS))
    print("\n== Per weekday, raw ==")
    print(bucket_table(frame, variants, "weekday", weekday_labels))
    print(f"\n== Per weekday, with the {floor_pct:g} % floor ==")
    print(bucket_table(frame, variants, "weekday", weekday_labels, floor_pct / 100.0))
    print(
        "\nReading it: a blend is worth enabling only if its raw MAE beats aurora's. "
        "If you enable it, move the floor toward that blend's '0-bias floor' plus the "
        "safety margin you want, instead of stacking the old floor on a corrected forecast."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
