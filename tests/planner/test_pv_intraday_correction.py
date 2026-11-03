"""Intraday PV bias correction (planner/inputs/pv_intraday.py)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import pytz

from planner.inputs.data_prep import prepare_df
from planner.inputs.pv_intraday import (
    PvCorrection,
    PvIntradayConfig,
    apply_pv_correction,
    compute_pv_correction,
    decay_factors,
    load_pv_correction,
    local_day_bounds,
)

TZ = pytz.timezone("Europe/Stockholm")
CFG = PvIntradayConfig()


def _local(y, m, d, hh, mm=0):
    return TZ.localize(datetime(y, m, d, hh, mm))


def _rows(start_local, n, fc, act, **extra):
    """n consecutive 15-min rows starting at start_local."""
    out = []
    for i in range(n):
        st = start_local + timedelta(minutes=15 * i)
        out.append(
            {
                "slot_start": TZ.normalize(st).isoformat(),
                "pv_forecast_kwh": fc,
                "pv_kwh": act,
                **extra,
            }
        )
    return out


# --- ratio / clamp / evidence -------------------------------------------------


def test_overcast_midday_gives_low_factor():
    now = _local(2026, 11, 1, 13, 5)
    rows = _rows(_local(2026, 11, 1, 11), 8, fc=0.8, act=0.2)
    c = compute_pv_correction(rows, now, CFG, TZ)
    assert c.reason == "ok" and c.applied
    assert c.evidence_slots == 8
    assert c.raw_ratio == pytest.approx(0.25)
    assert c.factor == pytest.approx(0.25)


def test_clamped_low_and_high():
    now = _local(2026, 6, 1, 13, 0)
    low = compute_pv_correction(_rows(_local(2026, 6, 1, 11), 8, 1.0, 0.0), now, CFG, TZ)
    assert low.factor == pytest.approx(CFG.min_factor)
    high = compute_pv_correction(_rows(_local(2026, 6, 1, 11), 8, 0.5, 2.0), now, CFG, TZ)
    assert high.raw_ratio == pytest.approx(4.0)
    assert high.factor == pytest.approx(CFG.max_factor)


def test_upward_correction_is_damped():
    now = _local(2026, 6, 1, 13, 0)
    c = compute_pv_correction(_rows(_local(2026, 6, 1, 11), 8, 1.0, 1.2), now, CFG, TZ)
    assert c.raw_ratio == pytest.approx(1.2)
    assert c.factor == pytest.approx(1.1)  # half the upside
    sym = PvIntradayConfig(upward_gain=1.0)
    assert compute_pv_correction(
        _rows(_local(2026, 6, 1, 11), 8, 1.0, 1.2), now, sym, TZ
    ).factor == pytest.approx(1.2)


def test_energy_weighted_not_mean_of_ratios():
    now = _local(2026, 6, 1, 13, 0)
    rows = _rows(_local(2026, 6, 1, 11), 4, 1.0, 1.0) + _rows(_local(2026, 6, 1, 12), 4, 0.1, 0.0)
    c = compute_pv_correction(rows, now, CFG, TZ)
    assert c.factor == pytest.approx(4.0 / 4.4)


def test_insufficient_slots_is_noop():
    now = _local(2026, 6, 1, 13, 0)
    c = compute_pv_correction(_rows(_local(2026, 6, 1, 12, 15), 3, 1.0, 0.1), now, CFG, TZ)
    assert c.factor == 1.0 and not c.applied and c.reason == "insufficient_evidence"


def test_insufficient_energy_is_noop():
    now = _local(2026, 6, 1, 13, 0)
    c = compute_pv_correction(_rows(_local(2026, 6, 1, 12), 4, 0.06, 0.0), now, CFG, TZ)
    assert c.factor == 1.0 and c.reason == "insufficient_evidence"


def test_no_data_is_noop():
    c = compute_pv_correction([], _local(2026, 6, 1, 13), CFG, TZ)
    assert c.factor == 1.0 and not c.applied


def test_night_slots_are_ignored():
    now = _local(2026, 11, 1, 22, 0)
    rows = _rows(_local(2026, 11, 1, 20), 8, fc=0.0, act=0.0)
    c = compute_pv_correction(rows, now, CFG, TZ)
    assert c.factor == 1.0 and c.evidence_slots == 0
    assert c.excluded.get("low_forecast") == 8


def test_disabled_is_noop():
    now = _local(2026, 11, 1, 13, 5)
    rows = _rows(_local(2026, 11, 1, 11), 8, fc=0.8, act=0.2)
    c = compute_pv_correction(rows, now, PvIntradayConfig(enabled=False), TZ)
    assert c.factor == 1.0 and c.reason == "disabled"


def test_only_finished_slots_in_lookback_and_today():
    now = _local(2026, 6, 2, 0, 40)
    # Yesterday's slots are outside "today" even though inside the 2h lookback.
    rows = _rows(_local(2026, 6, 1, 23), 4, 1.0, 0.1)
    assert compute_pv_correction(rows, now, CFG, TZ).evidence_slots == 0

    now = _local(2026, 6, 1, 13, 5)
    rows = (
        _rows(_local(2026, 6, 1, 9), 8, 1.0, 0.0)  # older than lookback
        + _rows(_local(2026, 6, 1, 11, 15), 7, 1.0, 0.5)  # in window, finished
        + _rows(_local(2026, 6, 1, 13), 1, 1.0, 0.0)  # in progress — excluded
    )
    c = compute_pv_correction(rows, now, CFG, TZ)
    assert c.evidence_slots == 7
    assert c.factor == pytest.approx(0.5)


def test_suspect_slots_excluded():
    now = _local(2026, 6, 1, 13, 0)
    good = _rows(_local(2026, 6, 1, 11), 4, 1.0, 0.5)
    full = _rows(_local(2026, 6, 1, 12), 2, 1.0, 0.0, soc_end_percent=100.0)
    neg = _rows(_local(2026, 6, 1, 12, 30), 1, 1.0, 0.0, export_price_sek_kwh=-0.2)
    bad = _rows(
        _local(2026, 6, 1, 12, 45), 1, 1.0, 0.0, quality_flags=json.dumps({"multi_slot_delta": {}})
    )
    c = compute_pv_correction(good + full + neg + bad, now, CFG, TZ)
    assert c.evidence_slots == 4 and c.factor == pytest.approx(0.5)
    assert c.excluded == {"battery_full": 2, "negative_export_price": 1, "bad_quality": 1}


def test_garbage_rows_do_not_raise():
    now = _local(2026, 6, 1, 13, 0)
    rows = [
        {"slot_start": "not-a-date", "pv_forecast_kwh": 1.0, "pv_kwh": 0.0},
        {"slot_start": _local(2026, 6, 1, 12).isoformat(), "pv_forecast_kwh": None, "pv_kwh": 0.0},
        {
            "slot_start": _local(2026, 6, 1, 12).isoformat(),
            "pv_forecast_kwh": 1.0,
            "pv_kwh": float("nan"),
        },
    ]
    c = compute_pv_correction(rows, now, CFG, TZ)
    assert c.factor == 1.0


def test_config_parsing_and_sanitising():
    cfg = PvIntradayConfig.from_config(
        {
            "forecasting": {
                "pv_intraday_correction": {"enabled": False, "min_factor": "x", "max_factor": 0.1}
            }
        }
    )
    assert cfg.enabled is False
    assert cfg.min_factor == 0.2  # unparseable → default
    assert cfg.max_factor == 0.2  # never below min_factor
    assert PvIntradayConfig.from_config({}).enabled is True


# --- decay ---------------------------------------------------------------------


def _idx(start_local, n):
    return pd.DatetimeIndex(
        [TZ.normalize(start_local + timedelta(minutes=15 * i)) for i in range(n)]
    )


def test_decay_halves_per_half_life_and_stops_at_midnight():
    now = _local(2026, 6, 1, 12, 0)
    idx = _idx(_local(2026, 6, 1, 12, 0), 4 * 14)  # to 02:00 next day
    m = decay_factors(idx, 0.2, now, 2.0, TZ)
    # slot 12:00 midpoint 12:07:30 → lead 0.125h
    assert m.iloc[0] == pytest.approx(1 - 0.8 * 0.5 ** (0.125 / 2))
    # two hours (one half-life) later the deviation from 1.0 has halved
    dev0 = 1 - m.iloc[0]
    dev2h = 1 - m[TZ.localize(datetime(2026, 6, 1, 14, 0))]
    assert dev2h == pytest.approx(dev0 / 2)
    # monotone toward 1.0
    today = m[m.index < TZ.localize(datetime(2026, 6, 2))]
    assert (today.diff().dropna() >= -1e-12).all()
    # tomorrow untouched
    assert (m[m.index >= TZ.localize(datetime(2026, 6, 2))] == 1.0).all()


def test_decay_dst_autumn_day():
    """2026-10-25 is the 25-hour day: midnight cut must use the LOCAL date."""
    now = TZ.localize(datetime(2026, 10, 25, 1, 30), is_dst=True)
    start, end = local_day_bounds(now, TZ)
    assert (end - start) == pd.Timedelta(hours=25)
    idx = pd.date_range(
        start, end + pd.Timedelta(hours=2), freq="15min", inclusive="left"
    ).tz_convert(TZ)
    m = decay_factors(idx, 0.5, now, 2.0, TZ)
    assert (m[idx >= end.tz_convert(TZ)] == 1.0).all()
    assert (m[idx < end.tz_convert(TZ)] < 1.0).all()


def test_decay_dst_spring_day_bounds():
    now = _local(2026, 3, 29, 12, 0)
    start, end = local_day_bounds(now, TZ)
    assert (end - start) == pd.Timedelta(hours=23)


def test_zero_half_life_only_current_slot():
    now = _local(2026, 6, 1, 12, 5)
    idx = _idx(_local(2026, 6, 1, 12, 0), 3)
    m = decay_factors(idx, 0.5, now, 0.0, TZ)
    assert m.tolist() == [0.5, 1.0, 1.0]


# --- apply ---------------------------------------------------------------------


def test_apply_scales_all_quantiles_consistently():
    now = _local(2026, 6, 1, 12, 0)
    idx = _idx(now, 4)
    df = pd.DataFrame(
        {"pv_forecast_kwh": 1.0, "pv_p10": 0.5, "pv_p90": 1.5, "load_forecast_kwh": 0.3}, index=idx
    )
    corr = PvCorrection(factor=0.4, applied=True, half_life_hours=2.0)
    out = apply_pv_correction(df, corr, now, TZ)
    m = decay_factors(idx, 0.4, now, 2.0, TZ)
    assert out["pv_forecast_kwh"].tolist() == pytest.approx(m.tolist())
    assert out["pv_p10"].tolist() == pytest.approx((0.5 * m).tolist())
    assert out["pv_p90"].tolist() == pytest.approx((1.5 * m).tolist())
    assert out["load_forecast_kwh"].tolist() == [0.3] * 4
    assert df["pv_forecast_kwh"].tolist() == [1.0] * 4  # input not mutated


def test_apply_noop_when_not_applied():
    df = pd.DataFrame({"pv_forecast_kwh": [1.0]}, index=_idx(_local(2026, 6, 1, 12), 1))
    assert apply_pv_correction(df, PvCorrection(), _local(2026, 6, 1, 12), TZ) is df


# --- integration: real learning DB → planner input frame -------------------------


@pytest.mark.asyncio
async def test_db_to_planner_input_is_scaled(tmp_path):
    from backend.learning.models import Base
    from backend.learning.store import LearningStore

    db = tmp_path / "planner_learning.db"
    store = LearningStore(str(db), TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    now = _local(2026, 11, 1, 13, 5)
    day = _local(2026, 11, 1, 0)
    slots = [day + timedelta(minutes=15 * i) for i in range(96)]
    # Forecast: 0.8 kWh/slot 09-15, else 0. Actual: a quarter of it.
    fc_rows = [
        {
            "slot_start": s.isoformat(),
            "pv_forecast_kwh": 0.8 if 9 <= s.hour < 15 else 0.0,
            "load_forecast_kwh": 0.3,
        }
        for s in slots
    ]
    await store.store_forecasts(fc_rows, "aurora")
    obs = pd.DataFrame(
        [
            {
                "slot_start": s,
                "slot_end": s + timedelta(minutes=15),
                "pv_kwh": (0.2 if 9 <= s.hour < 15 else 0.0),
                "load_kwh": 0.3,
                "soc_end_percent": 50.0,
            }
            for s in slots
            if s + timedelta(minutes=15) <= now
        ]
    )
    await store.store_slot_observations(obs)
    await store.close()

    config = {
        "learning": {"sqlite_path": str(db)},
        "forecasting": {"pv_intraday_correction": {"enabled": True}},
    }
    corr = await load_pv_correction(config, now.astimezone(UTC), TZ)
    assert corr.reason == "ok"
    assert corr.evidence_slots == 8  # 11:00-12:45 inside 2 h lookback
    assert corr.factor == pytest.approx(0.25)

    future = [s for s in slots if s >= _local(2026, 11, 1, 13)]
    input_data = {
        "price_data": [
            {
                "start_time": s.isoformat(),
                "end_time": (s + timedelta(minutes=15)).isoformat(),
                "import_price_sek_kwh": 1.0,
                "export_price_sek_kwh": 0.5,
            }
            for s in future
        ],
        # The planner feed keys rows by start_time (backend/core/forecasts.py).
        "forecast_data": [
            {"start_time": r["slot_start"], **r}
            for r in fc_rows
            if pd.Timestamp(r["slot_start"]) >= future[0]
        ],
    }
    df = prepare_df(input_data, "Europe/Stockholm")
    out = apply_pv_correction(df, corr, now.astimezone(UTC), TZ)
    first = out["pv_forecast_kwh"].iloc[0]
    assert first == pytest.approx(0.8 * (1 - 0.75 * 0.5 ** ((2.5 / 60) / 2)), rel=1e-6)
    assert out["pv_forecast_kwh"].iloc[0] < out["pv_forecast_kwh"].iloc[4] < 0.8
    assert (out["pv_forecast_kwh"] <= df["pv_forecast_kwh"] + 1e-12).all()


@pytest.mark.asyncio
async def test_load_missing_db_is_noop(tmp_path):
    cfg = {"learning": {"sqlite_path": str(tmp_path / "nope.db")}}
    corr = await load_pv_correction(cfg, _local(2026, 6, 1, 12), TZ)
    assert corr.factor == 1.0 and corr.reason == "no_db"
    assert not (tmp_path / "nope.db").exists()


@pytest.mark.asyncio
async def test_load_broken_db_is_noop(tmp_path):
    p = tmp_path / "broken.db"
    p.write_text("not sqlite")
    corr = await load_pv_correction(
        {"learning": {"sqlite_path": str(p)}}, _local(2026, 6, 1, 12), TZ
    )
    assert corr.factor == 1.0 and corr.reason == "error"
