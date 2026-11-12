"""scripts/compare_pv_training_sources.py: read-only DB access and the comparison core.

The script lets the owner judge the historical-forecast training source on real data
before enabling it, so it must never write to the learning DB and must score both
sources on identical rows with inference-side (forecast endpoint) weather.
"""

from __future__ import annotations

import importlib.util
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import pytz

from ml.weather import PV_WEATHER_SOURCE_FORECAST, PV_WEATHER_SOURCE_HISTORICAL

TZ = pytz.timezone("Europe/Stockholm")
CONFIG: dict[str, Any] = {
    "timezone": "Europe/Stockholm",
    "system": {
        "location": {"latitude": 10.0, "longitude": 20.0},
        "solar_arrays": [{"name": "A", "kwp": 5.0, "tilt": 20, "azimuth": 180}],
    },
}


def _load_script() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_pv_training_sources.py"
    spec = importlib.util.spec_from_file_location("compare_pv_training_sources", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _slots(days: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=TZ.localize(datetime(2026, 6, 1, 0, 0)), periods=days * 96, freq="15min"
    )


def _weather(slots: pd.DatetimeIndex, scale: float) -> pd.DataFrame:
    hours = slots.hour + slots.minute / 60.0
    sun = [max(0.0, math.sin(math.pi * (h - 5.0) / 14.0)) if 5 <= h <= 19 else 0.0 for h in hours]
    # alternate bright and dull days so the model has something to learn
    day_factor = [1.0 if d % 2 == 0 else 0.4 for d in slots.dayofyear]
    ghi = [800.0 * s * f * scale for s, f in zip(sun, day_factor, strict=True)]
    return pd.DataFrame(
        {
            "temp_c": 18.0,
            "cloud_cover_pct": [0.0 if f == 1.0 else 80.0 for f in day_factor],
            "shortwave_radiation_w_m2": ghi,
            "dni_w_m2": [g * 0.8 for g in ghi],
            "dhi_w_m2": [g * 0.2 for g in ghi],
        },
        index=slots,
    )


def _make_db(tmp_path: Path, slots: pd.DatetimeIndex, pv: list[float]) -> Path:
    db = tmp_path / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE slot_observations (slot_start TEXT PRIMARY KEY, load_kwh REAL, "
            "pv_kwh REAL, quality_flags TEXT)"
        )
        conn.executemany(
            "INSERT INTO slot_observations VALUES (?, ?, ?, ?)",
            [(ts.isoformat(), 0.3, value, None) for ts, value in zip(slots, pv, strict=True)],
        )
    return db


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    slots = _slots(30)
    fc = _weather(slots, 1.0)
    hist = _weather(slots, 1.15)  # the archive runs hotter
    obs_template = pd.DataFrame({"slot_start": slots, "load_kwh": 0.3, "pv_kwh": 0.0})
    truth = script.build_pv_frame(obs_template.assign(pv_kwh=1.0, quality_flags=None), fc, CONFIG)
    pv = (0.8 * truth["physics_forecast_kwh"]).reindex(slots).fillna(0.0).tolist()
    db = _make_db(tmp_path_factory.mktemp("db"), slots, pv)
    return {"db": db, "slots": slots, "fc": fc, "hist": hist}


def test_db_is_opened_read_only(dataset):
    for immutable in (False, True):
        conn = script.open_readonly(str(dataset["db"]), immutable)
        try:
            count = conn.execute("SELECT COUNT(*) FROM slot_observations").fetchone()[0]
            assert count == 30 * 96
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM slot_observations")
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE x (a)")
        finally:
            conn.close()


def test_loader_closes_its_connection(dataset, monkeypatch):
    opened: list[sqlite3.Connection] = []
    real_open = script.open_readonly

    def _tracking_open(path: str, immutable: bool = False) -> sqlite3.Connection:
        opened.append(real_open(path, immutable))
        return opened[-1]

    monkeypatch.setattr(script, "open_readonly", _tracking_open)
    script.load_pv_observations(str(dataset["db"]), TZ, None, None, immutable=True)
    (conn,) = opened
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_load_pv_observations_applies_trainer_filters(dataset):
    obs = script.load_pv_observations(str(dataset["db"]), TZ, None, max_kwh=5.0)
    assert len(obs) == 30 * 96
    assert str(obs["slot_start"].dt.tz) == "Europe/Stockholm"
    floor = TZ.localize(datetime(2026, 6, 21))
    assert len(script.load_pv_observations(str(dataset["db"]), TZ, floor, None)) == 10 * 96
    assert script.load_pv_observations(str(dataset["db"]), TZ, None, max_kwh=0.1).empty


def test_compare_trains_both_sources_on_identical_rows(dataset):
    mtime_before = dataset["db"].stat().st_mtime_ns
    obs = script.load_pv_observations(str(dataset["db"]), TZ, None, None)
    holdout_start = dataset["slots"][-1].to_pydatetime() - timedelta(days=5)

    result = script.compare(
        obs,
        {PV_WEATHER_SOURCE_FORECAST: dataset["fc"], PV_WEATHER_SOURCE_HISTORICAL: dataset["hist"]},
        CONFIG,
        holdout_start,
        min_samples=100,
    )

    preds, actual, rows = result["preds"], result["actual"], result["rows"]
    assert set(preds) == {"physics_only", PV_WEATHER_SOURCE_FORECAST, PV_WEATHER_SOURCE_HISTORICAL}
    assert rows["common"] == rows[PV_WEATHER_SOURCE_FORECAST] == rows[PV_WEATHER_SOURCE_HISTORICAL]
    assert rows["train"] + rows["holdout"] == rows["common"]
    assert all(series.index.equals(actual.index) for series in preds.values())
    assert (actual.index >= holdout_start).all()

    # Truth is 0.8 x forecast-source physics: physics-only over-forecasts by 25 %,
    # the model trained on the SAME source as inference learns that away, and the
    # model trained against the hotter archive learns too large a correction.
    physics = script.score(preds["physics_only"], actual)
    same = script.score(preds[PV_WEATHER_SOURCE_FORECAST], actual)
    hotter = script.score(preds[PV_WEATHER_SOURCE_HISTORICAL], actual)
    assert physics["bias"] > 0
    assert same["mae"] < physics["mae"]
    assert hotter["bias"] < same["bias"]

    table = script.score_table(preds, actual, None, "bucket")
    assert "MAE / bias" in table and "physics_only" in table
    hourly = script.score_table(
        preds, actual, result["tail"]["slot_start"].dt.strftime("%H"), "hour"
    )
    assert len(hourly.splitlines()) > 5

    assert dataset["db"].stat().st_mtime_ns == mtime_before
    assert not list(dataset["db"].parent.glob("*-wal")) and not list(
        dataset["db"].parent.glob("*-journal")
    )


def test_main_prints_overall_month_and_hour_tables(dataset, tmp_path, monkeypatch, capsys):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    window = lambda frame, start, end: frame[(frame.index >= start) & (frame.index < end)]  # noqa: E731
    monkeypatch.setattr(
        script, "get_weather_series", lambda s, e, config=None: window(dataset["fc"], s, e)
    )
    monkeypatch.setattr(
        script,
        "get_historical_weather_series",
        lambda s, e, config=None: window(dataset["hist"], s, e),
    )

    code = script.main(
        [
            "--db",
            str(dataset["db"]),
            "--config",
            str(cfg_path),
            "--days",
            "20",
            "--holdout-days",
            "5",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    for heading in ("== Overall ==", "== Per month ==", "== Per hour (local) =="):
        assert heading in out
    assert "2026-06" in out
    assert "archive_all_history" in out  # the 10 days before the window feed variant 3
    assert " 1.150  (" in out  # archive/hindcast radiation ratio
