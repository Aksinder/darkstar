"""Load forecast fallback order when no load model is usable.

model -> learning DB 7-day per-slot mean -> (stored 0.0 =) HA profile -> demo.
Before, a missing model went straight to the HA profile, and a failed HA fetch on
top of that planned on an invented flat 0.5 kWh/slot (48 kWh/day).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

import backend.health as health
from backend.learning import LearningEngine
from backend.learning.baseline import (
    MIN_SLOT_MEAN_OBSERVATIONS,
    get_load_slot_means_7d,
    load_slot_means,
    time_of_day_key,
)
from ml.forward import generate_forward_slots

FIXED_NOW = datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
SLOTS = pd.date_range(FIXED_NOW, periods=8, freq="15min", tz="UTC")


def _history(
    load_by_key: dict[str, float], days: int = 7, flags: str | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in range(1, days + 1):
        for key, load in load_by_key.items():
            hour, minute = (int(part) for part in key.split(":"))
            ts = (FIXED_NOW - timedelta(days=day)).replace(hour=hour, minute=minute)
            rows.append({"slot_start": ts.isoformat(), "load_kwh": load, "quality_flags": flags})
    return rows


# --- the slot means ----------------------------------------------------------------


def test_slot_means_average_same_time_of_day():
    rows = _history({"12:00": 0.4, "12:15": 0.8})
    rows[0]["load_kwh"] = 1.1  # one 12:00 sample differs
    means = load_slot_means(rows)
    assert means["12:00"] == pytest.approx((1.1 + 6 * 0.4) / 7)
    assert means["12:15"] == pytest.approx(0.8)
    assert time_of_day_key(FIXED_NOW) == "12:00"


def test_slot_means_ignore_zero_null_and_flagged_rows():
    rows = _history({"12:00": 0.4})
    rows += [
        {"slot_start": rows[0]["slot_start"], "load_kwh": 0.0, "quality_flags": None},
        {"slot_start": rows[0]["slot_start"], "load_kwh": None, "quality_flags": None},
        {
            "slot_start": rows[0]["slot_start"],
            "load_kwh": 9.0,
            "quality_flags": json.dumps({"ev_unmeasured": {"easee": "unavailable"}}),
        },
        {
            "slot_start": rows[0]["slot_start"],
            "load_kwh": 9.0,
            "quality_flags": json.dumps({"load_spike_zeroed": ["load_kwh"]}),
        },
    ]
    assert load_slot_means(rows) == {"12:00": pytest.approx(0.4)}


def test_slot_means_need_a_minimum_of_observations():
    assert MIN_SLOT_MEAN_OBSERVATIONS == 3
    assert load_slot_means(_history({"12:00": 0.4}, days=2)) == {}
    assert load_slot_means(_history({"12:00": 0.4}, days=3)) == {"12:00": pytest.approx(0.4)}
    assert load_slot_means([]) == {}


@pytest.mark.asyncio
async def test_slot_means_query_the_trailing_week_and_never_raise():
    store = MagicMock()
    store.get_observation_rows_between = AsyncMock(return_value=_history({"12:00": 0.4}))
    assert await get_load_slot_means_7d(store, FIXED_NOW) == {"12:00": pytest.approx(0.4)}
    store.get_observation_rows_between.assert_awaited_once_with(
        (FIXED_NOW - timedelta(days=7)).isoformat(), FIXED_NOW.isoformat()
    )

    store.get_observation_rows_between = AsyncMock(side_effect=RuntimeError("db locked"))
    assert await get_load_slot_means_7d(store, FIXED_NOW) == {}


# --- ml.forward tier selection -------------------------------------------------------


def _engine(history: list[dict[str, Any]] | Exception) -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.store_forecasts = AsyncMock()
    engine.timezone = pytz.UTC
    engine.db_path = "fake_db.db"
    engine.config = {
        "timezone": "UTC",
        "system": {
            "location": {"latitude": 59.3, "longitude": 18.1},
            "solar_arrays": [{"kwp": 10.0}],
        },
    }
    engine.store = MagicMock()
    if isinstance(history, Exception):
        engine.store.get_observation_rows_between = AsyncMock(side_effect=history)
    else:
        engine.store.get_observation_rows_between = AsyncMock(return_value=history)
    return engine


class _Booster:
    def __init__(self, value: float | Exception):
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if isinstance(self.value, Exception):
            raise self.value
        return np.full(len(features), self.value)


async def _forward(engine: MagicMock, models: dict[str, Any]) -> list[dict[str, Any]]:
    weather = pd.DataFrame(
        {"temp_c": 20.0, "cloud_cover_pct": 10.0, "shortwave_radiation_w_m2": 500.0}, index=SLOTS
    )
    health.set_load_forecast_tier("")
    with (
        patch("ml.forward.datetime") as mock_dt,
        patch("ml.forward.get_learning_engine", return_value=engine),
        patch("ml.forward._load_models", return_value=models),
        patch("ml.forward.async_get_weather_series", AsyncMock(return_value=weather)),
        patch("ml.forward.get_vacation_mode_series", return_value=pd.Series(0.0, index=SLOTS)),
        patch("ml.forward.get_alarm_armed_series", return_value=pd.Series(0.0, index=SLOTS)),
        patch("ml.forward.determine_graduation_level", return_value=(2, "graduate", 30.0)),
        patch("backend.astro.SunCalculator"),
    ):
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.side_effect = datetime
        await generate_forward_slots(horizon_hours=2, forecast_version="test")
    args, _ = engine.store_forecasts.call_args
    return args[0]


ALL_KEYS = {time_of_day_key(ts): 0.2 + 0.1 * i for i, ts in enumerate(SLOTS)}
LOAD_MODELS = {"load_p10": _Booster(0.3), "load_p50": _Booster(0.5), "load_p90": _Booster(0.9)}


@pytest.fixture(autouse=True)
def _reset_health():
    yield
    health.clear_load_forecast_status()
    health.set_load_forecast_tier("")


@pytest.mark.asyncio
async def test_working_model_wins_and_is_reported():
    engine = _engine(_history(ALL_KEYS))
    forecasts = await _forward(engine, dict(LOAD_MODELS))

    assert [f["load_forecast_kwh"] for f in forecasts] == [0.5] * 8
    assert [f["load_p10"] for f in forecasts] == [0.3] * 8
    engine.store.get_observation_rows_between.assert_not_awaited()
    status = health.get_load_forecast_status()
    assert status["tier"] == "model" and status["status"] == "ok"


@pytest.mark.asyncio
async def test_missing_model_uses_db_slot_means_before_ha_profile(caplog):
    engine = _engine(_history(ALL_KEYS))
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        forecasts = await _forward(engine, {})

    expected = [0.2 + 0.1 * i for i in range(8)]
    assert [f["load_forecast_kwh"] for f in forecasts] == pytest.approx(expected)
    assert [f["base_load_forecast_kwh"] for f in forecasts] == pytest.approx(expected)
    assert [f["load_p10"] for f in forecasts] == pytest.approx([0.7 * v for v in expected])
    assert [f["load_p90"] for f in forecasts] == pytest.approx([1.3 * v for v in expected])
    # every value is above the planner's 0.001 "no forecast" threshold, so
    # backend/core/forecasts.py will NOT replace it with the HA/demo profile
    assert all(f["load_forecast_kwh"] > 0.001 for f in forecasts)

    status = health.get_load_forecast_status()
    assert status == {
        "status": "degraded",
        "reason": "no_ml",
        "tier": "slot_mean_7d",
        "detail": "",
    }
    assert "Load forecast source: slot_mean_7d (8/8 slots" in caplog.text


@pytest.mark.asyncio
async def test_slots_without_recorded_data_fall_through_to_ha_profile():
    partial = {key: value for i, (key, value) in enumerate(ALL_KEYS.items()) if i < 3}
    forecasts = await _forward(_engine(_history(partial)), {})

    loads = [f["load_forecast_kwh"] for f in forecasts]
    assert loads[:3] == pytest.approx([0.2, 0.3, 0.4])
    assert loads[3:] == [0.0] * 5  # 0.0 = "planner, use the HA profile here"
    assert health.get_load_forecast_status()["tier"] == "slot_mean_7d+ha_profile"


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [[], RuntimeError("db locked"), _history(ALL_KEYS, days=2)])
async def test_no_usable_history_reproduces_the_old_zero_fallback(history):
    forecasts = await _forward(_engine(history), {})

    for f in forecasts:
        assert f["load_forecast_kwh"] == 0.0
        assert f["load_p10"] == 0.0 and f["load_p90"] == 0.0
    assert health.get_load_forecast_status()["tier"] == "ha_profile"


@pytest.mark.asyncio
async def test_model_that_fails_to_predict_falls_back_instead_of_aborting(caplog):
    models = {**LOAD_MODELS, "load_p50": _Booster(ValueError("feature count mismatch"))}
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        forecasts = await _forward(_engine(_history(ALL_KEYS)), models)

    expected = [0.2 + 0.1 * i for i in range(8)]
    assert [f["load_forecast_kwh"] for f in forecasts] == pytest.approx(expected)
    # no half-applied model output: the bands come from the fallback too
    assert [f["load_p10"] for f in forecasts] == pytest.approx([0.7 * v for v in expected])
    assert "Load model inference failed (ValueError: feature count mismatch)" in caplog.text
    assert "Load models failed to predict" in caplog.text
    status = health.get_load_forecast_status()
    assert status["tier"] == "slot_mean_7d"
    # not reported as "models not trained": the reason and the exception class are kept
    assert status["reason"] == "predict_failed"
    assert status["detail"] == "ValueError: feature count mismatch"
    checker = health.HealthChecker.__new__(health.HealthChecker)
    (issue,) = checker.check_load_forecast()
    assert issue.message == (
        "Load forecast model failed to predict (ValueError: feature count mismatch)"
    )
    assert "not trained" not in issue.guidance
    assert "7-day per-slot mean of recorded load" in issue.guidance


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
async def test_non_finite_model_output_is_a_predict_failure_not_a_healthy_001(bad_value, caplog):
    """NaN used to pass the guardrails as 0.01 kWh with status ok / tier model."""
    models = {**LOAD_MODELS, "load_p50": _Booster(bad_value)}
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        forecasts = await _forward(_engine(_history(ALL_KEYS)), models)

    expected = [0.2 + 0.1 * i for i in range(8)]
    assert [f["load_forecast_kwh"] for f in forecasts] == pytest.approx(expected)
    assert [f["load_p90"] for f in forecasts] == pytest.approx([1.3 * v for v in expected])
    status = health.get_load_forecast_status()
    assert status["status"] == "degraded" and status["reason"] == "predict_failed"
    assert status["tier"] == "slot_mean_7d"
    assert "load_p50 returned 8/8 non-finite values" in status["detail"]
    assert "Load model inference failed (ValueError" in caplog.text


# --- kill switch: forecasting.load_slot_mean_fallback_enabled ------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("models", [{}, {**LOAD_MODELS, "load_p50": _Booster(ValueError("x"))}])
async def test_fallback_disabled_stores_zero_exactly_like_before(models, caplog):
    engine = _engine(_history(ALL_KEYS))  # the DB HAS usable slot means
    engine.config["forecasting"] = {"load_slot_mean_fallback_enabled": False}
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        forecasts = await _forward(engine, models)

    no_history = await _forward(_engine([]), {})  # == the pre-change output
    assert [{k: f[k] for k in f if k.startswith(("load", "base_load"))} for f in forecasts] == [
        {k: f[k] for k in f if k.startswith(("load", "base_load"))} for f in no_history
    ]
    for f in forecasts:
        assert f["load_forecast_kwh"] == 0.0 and f["load_p10"] == 0.0 and f["load_p90"] == 0.0
    engine.store.get_observation_rows_between.assert_not_awaited()
    assert "7-day slot-mean fallback is DISABLED" in caplog.text


@pytest.mark.asyncio
async def test_fallback_disabled_reports_ha_profile_tier():
    engine = _engine(_history(ALL_KEYS))
    engine.config["forecasting"] = {"load_slot_mean_fallback_enabled": "false"}
    await _forward(engine, {})
    assert health.get_load_forecast_status()["tier"] == "ha_profile"


# --- flags in the slot-mean path -----------------------------------------------------


def test_slot_means_survive_a_week_of_an_offline_device():
    """Winterised heater 'unavailable' on every slot: the base load is still right."""
    offline = json.dumps({"water_unmeasured": {"spa": "unavailable"}})
    rows = [
        {
            "slot_start": (FIXED_NOW - timedelta(days=7) + timedelta(minutes=15 * i)).isoformat(),
            "load_kwh": 0.4,
            "quality_flags": offline,
        }
        for i in range(7 * 96)
    ]
    means = load_slot_means(rows)
    assert len(means) == 96 and all(v == pytest.approx(0.4) for v in means.values())


def test_slot_means_drop_a_short_gap_but_not_the_rest():
    gap = json.dumps({"ev_unmeasured": {"easee": "unavailable"}})
    rows = _history({"12:00": 0.4, "12:15": 0.4})
    for row in rows:
        if row["slot_start"].endswith("12:15:00+00:00"):
            continue
        row["quality_flags"] = gap  # 12:00 flagged on all 7 days: 7 isolated slots
    assert load_slot_means(rows) == {"12:15": pytest.approx(0.4)}


def test_slot_means_have_the_starvation_guard(caplog):
    """A whole week flagged by a NON-device rule must not leave zero means."""
    rows = _history({"12:00": 0.4, "12:15": 0.6}, flags=json.dumps({"base_load_clamped": True}))
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.cleaning"):
        means = load_slot_means(rows)
    assert means == {"12:00": pytest.approx(0.4), "12:15": pytest.approx(0.6)}
    assert "starvation guard" in caplog.text


def test_slot_means_without_cleaning_average_flagged_rows_too():
    rows = _history({"12:00": 0.4})
    rows.append(
        {
            "slot_start": rows[0]["slot_start"],
            "load_kwh": 1.2,
            "quality_flags": json.dumps({"ev_unmeasured": {"easee": "snapshot"}}),
        }
    )
    assert load_slot_means(rows)["12:00"] == pytest.approx(0.4)
    assert load_slot_means(rows, apply_cleaning=False)["12:00"] == pytest.approx(0.5)


def test_health_issue_names_the_tier_in_use():
    checker = health.HealthChecker.__new__(health.HealthChecker)
    health.set_load_forecast_status("degraded", "no_ml")

    health.set_load_forecast_tier("slot_mean_7d")
    (issue,) = checker.check_load_forecast()
    assert "7-day per-slot mean of recorded load" in issue.guidance

    health.set_load_forecast_tier("")  # unknown tier keeps the old wording
    (issue,) = checker.check_load_forecast()
    assert "Current forecast uses HA historical profile." in issue.guidance
