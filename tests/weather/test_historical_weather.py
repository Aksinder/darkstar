"""PV training weather from Open-Meteo's historical-forecast archive (opt-in).

forecasting.pv_training_weather_source selects the source; the archive gives 8-18 %
more radiation than the forecast endpoint's hindcast, so the choice is explicit, the
default is the old source, gaps are never interpolated across, and the source a PV
model was trained on is stored with the model. All network access is mocked.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

import ml.forward as forward_mod
import ml.train as train_mod
import ml.weather as weather_mod
from backend.learning import LearningEngine
from ml.weather import (
    PV_WEATHER_SOURCE_FORECAST,
    PV_WEATHER_SOURCE_HISTORICAL,
    get_historical_weather_series,
    interpolate_capped,
    resolve_pv_training_weather_source,
)

UTC = pytz.UTC
TZ = pytz.timezone("Europe/Stockholm")
CONFIG: dict[str, Any] = {
    "timezone": "UTC",
    "system": {"location": {"latitude": 10.0, "longitude": 20.0}},
}


@pytest.fixture(autouse=True)
def _clear_cache():
    weather_mod._weather_cache.clear()
    yield
    weather_mod._weather_cache.clear()


def _payload(start: datetime, hours: int, nulls: set[int] | None = None) -> dict[str, Any]:
    nulls = nulls or set()
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(hours)]

    def col(scale: float) -> list[float | None]:
        return [None if h in nulls else scale * h for h in range(hours)]

    return {
        "hourly": {
            "time": times,
            "temperature_2m": col(1.0),
            "cloud_cover": col(2.0),
            "shortwave_radiation": col(10.0),
            "direct_normal_irradiance": col(20.0),
            "diffuse_radiation": col(5.0),
        }
    }


def _response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


# --- source selection -------------------------------------------------------------


def test_source_defaults_to_forecast_past_days():
    assert resolve_pv_training_weather_source({}) == PV_WEATHER_SOURCE_FORECAST
    assert resolve_pv_training_weather_source(None) == PV_WEATHER_SOURCE_FORECAST
    for raw in (None, "", "  "):
        cfg = {"forecasting": {"pv_training_weather_source": raw}}
        assert resolve_pv_training_weather_source(cfg) == PV_WEATHER_SOURCE_FORECAST


def test_source_accepts_historical_and_rejects_unknown(caplog):
    cfg = {"forecasting": {"pv_training_weather_source": " Historical_Forecast "}}
    assert resolve_pv_training_weather_source(cfg) == PV_WEATHER_SOURCE_HISTORICAL

    with caplog.at_level(logging.WARNING, logger="darkstar.ml.weather"):
        cfg = {"forecasting": {"pv_training_weather_source": "archive"}}
        assert resolve_pv_training_weather_source(cfg) == PV_WEATHER_SOURCE_FORECAST
    assert "Unknown forecasting.pv_training_weather_source" in caplog.text


def test_default_config_ships_the_current_source():
    import yaml

    default_path = Path(__file__).resolve().parents[2] / "config.default.yaml"
    default_cfg = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    assert default_cfg["forecasting"]["pv_training_weather_source"] == "forecast_past_days"
    assert resolve_pv_training_weather_source(default_cfg) == PV_WEATHER_SOURCE_FORECAST


# --- the fetch ---------------------------------------------------------------------


@patch("ml.weather.requests.get")
def test_historical_fetch_uses_archive_endpoint_dates_and_config_coordinates(mock_get):
    start = UTC.localize(datetime(2026, 7, 10, 0, 0))
    end = UTC.localize(datetime(2026, 7, 11, 0, 0))
    mock_get.return_value = _response(_payload(datetime(2026, 7, 9), 72))

    df = get_historical_weather_series(start, end, config=CONFIG)

    args, kwargs = mock_get.call_args
    assert args[0] == "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = kwargs["params"]
    assert params["latitude"] == 10.0 and params["longitude"] == 20.0
    assert params["start_date"] == "2026-07-09"  # one day of padding
    assert params["end_date"] == "2026-07-12"
    assert params["timezone"] == "UTC"
    assert "past_days" not in params
    for name in ("direct_normal_irradiance", "diffuse_radiation", "shortwave_radiation"):
        assert name in params["hourly"].split(",")

    assert list(df.columns) == [
        "temp_c",
        "cloud_cover_pct",
        "shortwave_radiation_w_m2",
        "dni_w_m2",
        "dhi_w_m2",
    ]
    assert len(df) == 96  # 15-min slots of the requested day only
    assert df.index.min() == start and df.index.max() == end - timedelta(minutes=15)
    assert not df.isna().any().any()
    # hour 24 of the payload is 2026-07-10 00:00; 00:30 is halfway to hour 25
    assert df.loc[start, "shortwave_radiation_w_m2"] == pytest.approx(240.0)
    assert df.loc[start + timedelta(minutes=30), "dni_w_m2"] == pytest.approx(490.0)


@patch("ml.weather.requests.get")
def test_historical_fetch_converts_to_planner_timezone(mock_get):
    cfg = {**CONFIG, "timezone": "Europe/Stockholm"}
    start = TZ.localize(datetime(2026, 7, 10, 0, 0))
    mock_get.return_value = _response(_payload(datetime(2026, 7, 9), 72))

    df = get_historical_weather_series(start, start + timedelta(hours=2), config=cfg)

    assert str(df.index.tz) == "Europe/Stockholm"
    # 00:00 local = 22:00 UTC the day before = payload hour 22
    assert df.loc[start, "temp_c"] == pytest.approx(22.0)


@patch("ml.weather.requests.get")
def test_historical_fetch_without_coordinates_fetches_nothing(mock_get):
    start = UTC.localize(datetime(2026, 7, 10))
    df = get_historical_weather_series(
        start, start + timedelta(days=1), config={"timezone": "UTC", "system": {}}
    )
    assert df.empty
    mock_get.assert_not_called()


@patch("ml.weather.retry_sync", side_effect=RuntimeError("boom"))
def test_historical_fetch_failure_returns_empty(_retry):
    start = UTC.localize(datetime(2026, 7, 10))
    assert get_historical_weather_series(start, start + timedelta(days=1), config=CONFIG).empty


# --- interpolation cap ---------------------------------------------------------------


def test_interpolate_capped_fills_between_adjacent_hours_only():
    idx = pd.date_range("2026-07-10 00:00", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"x": [0.0, 4.0, np.nan, 12.0, 16.0, 20.0]}, index=idx)

    out = interpolate_capped(df)

    assert out.loc["2026-07-10 00:30+00:00", "x"] == pytest.approx(2.0)
    # the missing 02:00 value leaves a 7-slot hole: nothing in it is invented
    hole = out.loc["2026-07-10 01:15+00:00":"2026-07-10 02:45+00:00", "x"]
    assert len(hole) == 7 and hole.isna().all()
    assert out.loc["2026-07-10 03:15+00:00", "x"] == pytest.approx(13.0)


def test_interpolate_capped_does_not_bridge_a_missing_day():
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-07-10 00:00", periods=24, freq="1h", tz="UTC"))
        + list(pd.date_range("2026-07-12 00:00", periods=24, freq="1h", tz="UTC"))
    )
    df = pd.DataFrame({"x": np.arange(48, dtype=float)}, index=idx)

    out = interpolate_capped(df)

    missing_day = out.loc["2026-07-11 00:00+00:00":"2026-07-11 23:45+00:00", "x"]
    assert len(missing_day) == 96 and missing_day.isna().all()
    # pandas' unbounded interpolate (what the forecast-endpoint path does) WOULD fill it
    assert not df.resample("15min").interpolate(method="linear")["x"].isna().any()


@patch("ml.weather.requests.get")
def test_historical_fetch_leaves_null_hours_empty(mock_get, caplog):
    start = UTC.localize(datetime(2026, 7, 10, 0, 0))
    # payload hours 30..35 (06:00-11:00 on the 10th) are null
    mock_get.return_value = _response(_payload(datetime(2026, 7, 9), 72, nulls=set(range(30, 36))))

    with caplog.at_level(logging.WARNING, logger="darkstar.ml.weather"):
        df = get_historical_weather_series(start, start + timedelta(days=1), config=CONFIG)

    gap = df.loc[start + timedelta(hours=5, minutes=15) : start + timedelta(hours=11, minutes=45)]
    assert gap["shortwave_radiation_w_m2"].isna().all()
    assert df.loc[start + timedelta(hours=5), "shortwave_radiation_w_m2"] == pytest.approx(290.0)
    assert df.loc[start + timedelta(hours=12), "shortwave_radiation_w_m2"] == pytest.approx(360.0)
    assert "no radiation after capped interpolation" in caplog.text


# --- training wiring + model metadata ---------------------------------------------------


def _engine(source: str | None) -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.timezone = TZ
    engine.db_path = ":memory:"
    engine.config = {
        "system": {
            "location": {"latitude": 10.0, "longitude": 20.0},
            "solar_arrays": [{"name": "A", "kwp": 5.0, "tilt": 30, "azimuth": 180}],
        },
        "forecasting": {} if source is None else {"pv_training_weather_source": source},
    }
    return engine


def _obs() -> pd.DataFrame:
    slots = pd.date_range(start=TZ.localize(datetime(2026, 7, 9, 0, 0)), periods=192, freq="15min")
    return pd.DataFrame(
        {"slot_start": slots, "load_kwh": 0.3, "pv_kwh": 0.5, "quality_flags": None}
    )


def _weather(obs: pd.DataFrame, radiation: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temp_c": 15.0,
            "cloud_cover_pct": 10.0,
            "shortwave_radiation_w_m2": radiation,
            "dni_w_m2": radiation,
            "dhi_w_m2": 50.0,
        },
        index=pd.DatetimeIndex(obs["slot_start"]),
    )


def _train(source: str | None, hist: pd.DataFrame | None = None):
    obs = _obs()
    calls: list[dict[str, Any]] = []
    saved: list[dict[str, Any]] = []

    def _regressor(features, target, min_samples, alpha=0.5, sample_weight=None):
        calls.append({"features": features.copy(), "n": len(target)})
        return MagicMock()

    def _save(model, path, feature_names=None):
        saved.append({"path": str(path), "metadata": None})

    def _annotate(path, metadata):
        for entry in saved:
            if entry["path"] == str(path):
                entry["metadata"] = metadata

    hist_mock = MagicMock(return_value=_weather(obs, 800.0) if hist is None else hist)
    with (
        patch("ml.train.get_learning_engine", return_value=_engine(source)),
        patch("ml.train._load_slot_observations", return_value=obs.copy()),
        patch("ml.train.get_weather_series", return_value=_weather(obs, 400.0)),
        patch("ml.train.get_historical_weather_series", hist_mock),
        patch("ml.train.get_vacation_mode_series", return_value=pd.Series(dtype=float)),
        patch("ml.train.get_alarm_armed_series", return_value=pd.Series(dtype=float)),
        patch("ml.train._train_regressor", side_effect=_regressor),
        patch("ml.train._save_model", side_effect=_save),
        patch("ml.train._annotate_model_sidecar", side_effect=_annotate),
        patch("ml.train.sqlite3.connect", side_effect=RuntimeError("no db in test")),
    ):
        result = train_mod.train_models()
    return result, calls, saved, hist_mock


def test_training_default_source_never_calls_the_archive():
    result, calls, saved, hist_mock = _train(None)

    hist_mock.assert_not_called()
    assert result["pv_weather_source"] == PV_WEATHER_SOURCE_FORECAST
    load_calls, pv_calls = calls[:3], calls[3:]
    assert all((c["features"]["shortwave_radiation_w_m2"] == 400.0).all() for c in pv_calls)
    assert all((c["features"]["shortwave_radiation_w_m2"] == 400.0).all() for c in load_calls)
    pv_saved = [s for s in saved if "pv_model" in s["path"]]
    assert pv_saved and all(
        s["metadata"] == {"weather_source": PV_WEATHER_SOURCE_FORECAST} for s in pv_saved
    )
    assert all(s["metadata"] is None for s in saved if "load_model" in s["path"])


def test_training_historical_source_swaps_pv_weather_only():
    result, calls, saved, hist_mock = _train(PV_WEATHER_SOURCE_HISTORICAL)

    hist_mock.assert_called_once()
    assert result["pv_weather_source"] == PV_WEATHER_SOURCE_HISTORICAL
    assert result["error"] is None
    load_calls, pv_calls = calls[:3], calls[3:]
    assert len(pv_calls) == 3
    # load models keep the forecast-endpoint weather, PV rows carry the archive's
    assert all((c["features"]["shortwave_radiation_w_m2"] == 400.0).all() for c in load_calls)
    assert all((c["features"]["shortwave_radiation_w_m2"] == 800.0).all() for c in pv_calls)
    assert all(c["features"]["physics_forecast_kwh"].notna().all() for c in pv_calls)
    pv_saved = [s for s in saved if "pv_model" in s["path"]]
    assert pv_saved and all(
        s["metadata"] == {"weather_source": PV_WEATHER_SOURCE_HISTORICAL} for s in pv_saved
    )


def test_training_historical_source_unavailable_skips_pv_instead_of_mixing():
    result, calls, saved, _ = _train(PV_WEATHER_SOURCE_HISTORICAL, hist=pd.DataFrame())

    assert len(calls) == 3  # load quantiles only
    assert result["pv_samples"] == 0
    assert "historical weather unavailable" in result["error"]
    assert not [s for s in saved if "pv_model" in s["path"]]
    assert [s for s in saved if "load_model" in s["path"]]


def test_model_sidecar_carries_the_weather_source(tmp_path):
    model = MagicMock()
    path = tmp_path / "pv_model_p50.lgb"
    train_mod._save_model(model, path, feature_names=["a", "b"])
    train_mod._annotate_model_sidecar(path, {"weather_source": "historical_forecast"})
    stored = json.loads(path.with_suffix(".features.json").read_text())
    assert stored == {"feature_names": ["a", "b"], "weather_source": "historical_forecast"}
    assert forward_mod.pv_model_weather_source(str(tmp_path)) == "historical_forecast"

    # no sidecar (model saved without feature names): nothing is created
    train_mod._annotate_model_sidecar(tmp_path / "other.lgb", {"weather_source": "x"})
    assert not (tmp_path / "other.features.json").exists()


# --- inference mismatch warning -----------------------------------------------------------


def _write_sidecar(models_dir, payload: dict[str, Any]) -> None:
    (models_dir / "pv_model_p50.features.json").write_text(json.dumps(payload))


def test_legacy_sidecar_counts_as_forecast_source(tmp_path):
    _write_sidecar(tmp_path, {"feature_names": ["hour"]})
    assert forward_mod.pv_model_weather_source(str(tmp_path)) == PV_WEATHER_SOURCE_FORECAST
    assert forward_mod.pv_model_weather_source(str(tmp_path / "missing")) is None


@pytest.mark.parametrize(
    ("trained", "configured", "expect_warning"),
    [
        (None, None, False),  # legacy model, key unset
        (PV_WEATHER_SOURCE_FORECAST, PV_WEATHER_SOURCE_FORECAST, False),
        (PV_WEATHER_SOURCE_HISTORICAL, PV_WEATHER_SOURCE_HISTORICAL, False),
        (None, PV_WEATHER_SOURCE_HISTORICAL, True),  # config switched, not retrained yet
        (PV_WEATHER_SOURCE_HISTORICAL, None, True),  # config reverted, model still archive
        (PV_WEATHER_SOURCE_HISTORICAL, PV_WEATHER_SOURCE_FORECAST, True),
    ],
)
def test_inference_warns_on_weather_source_mismatch(
    tmp_path, caplog, trained, configured, expect_warning
):
    payload: dict[str, Any] = {"feature_names": ["hour"]}
    if trained is not None:
        payload["weather_source"] = trained
    _write_sidecar(tmp_path, payload)
    cfg = {"forecasting": {} if configured is None else {"pv_training_weather_source": configured}}

    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        mismatch = forward_mod.warn_on_pv_weather_source_mismatch(cfg, str(tmp_path))

    assert mismatch is expect_warning
    assert ("trained on weather source" in caplog.text) is expect_warning


def test_no_pv_model_no_warning(tmp_path, caplog):
    cfg = {"forecasting": {"pv_training_weather_source": PV_WEATHER_SOURCE_HISTORICAL}}
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.forward"):
        assert forward_mod.warn_on_pv_weather_source_mismatch(cfg, str(tmp_path)) is False
    assert caplog.text == ""
