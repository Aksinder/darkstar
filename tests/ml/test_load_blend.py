"""Opt-in blend of the load model with the 7-day per-slot mean.

forecasting.load_blend_slot_mean_weight defaults to 0 = off, and off must be
bit-identical to a build without the option. The 117 % load floor key is untouched.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import pytz
import yaml

import backend.health as health
import backend.learning.baseline as baseline_mod
from backend.api.routers.config import _validate_config_for_save
from backend.config_migration import remove_deprecated_keys, template_aware_merge
from backend.learning.baseline import resolve_load_blend_weight
from tests.ml.test_load_fallback_tiers import (
    ALL_KEYS,
    FIXED_NOW,
    LOAD_MODELS,
    _engine,
    _forward,
    _history,
)

REPO = Path(__file__).resolve().parents[2]
MODEL = {"load_p10": 0.3, "load_p50": 0.5, "load_p90": 0.9}
MEANS = [0.2 + 0.1 * i for i in range(8)]


@pytest.fixture(autouse=True)
def _reset_health():
    baseline_mod._warned_blend_weights.clear()
    yield
    health.clear_load_forecast_status()
    health.set_load_forecast_tier("")


async def _run(weight: Any, *, set_key: bool = True, history: Any = None):
    engine = _engine(_history(ALL_KEYS) if history is None else history)
    if set_key:
        engine.config["forecasting"] = {"load_blend_slot_mean_weight": weight}
    forecasts = await _forward(engine, dict(LOAD_MODELS))
    return engine, forecasts


# --- weight resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 0.0), (0, 0.0), (0.0, 0.0), ("0.6", 0.6), (0.6, 0.6), (1, 1.0), (1.0, 1.0)],
)
def test_weight_accepts_shares_between_0_and_1(raw, expected):
    cfg = {"forecasting": {"load_blend_slot_mean_weight": raw}}
    assert resolve_load_blend_weight(cfg) == expected


def test_weight_defaults_to_off_without_the_key():
    assert resolve_load_blend_weight({}) == 0.0
    assert resolve_load_blend_weight({"forecasting": None}) == 0.0


@pytest.mark.parametrize("raw", [40, 60.0, 1.01, -0.1, "abc", True, float("nan"), [0.5]])
def test_weight_out_of_range_is_rejected_not_clamped(raw, caplog):
    cfg = {"forecasting": {"load_blend_slot_mean_weight": raw}}
    with caplog.at_level(logging.WARNING, logger="darkstar.learning.baseline"):
        assert resolve_load_blend_weight(cfg) == 0.0
    assert "blend disabled" in caplog.text


def test_invalid_weight_is_logged_once_per_value_not_per_forecast_run(caplog):
    cfg = {"forecasting": {"load_blend_slot_mean_weight": 40}}
    with caplog.at_level(logging.WARNING, logger="darkstar.learning.baseline"):
        assert [resolve_load_blend_weight(cfg) for _ in range(5)] == [0.0] * 5
        assert caplog.text.count("blend disabled") == 1
        cfg["forecasting"]["load_blend_slot_mean_weight"] = 60  # a changed value warns again
        resolve_load_blend_weight(cfg)
        resolve_load_blend_weight(cfg)
        assert caplog.text.count("blend disabled") == 2


def test_config_save_rejects_out_of_range_weight():
    def errors(value: Any) -> list[str]:
        cfg = {"forecasting": {"load_blend_slot_mean_weight": value}}
        return [
            i["message"]
            for i in _validate_config_for_save(cfg)
            if i["severity"] == "error" and "load_blend_slot_mean_weight" in i["message"]
        ]

    assert errors(40) and errors(-0.2) and errors("abc")
    assert not errors(0) and not errors(0.6) and not errors(1) and not errors(None)
    assert not [
        i for i in _validate_config_for_save({}) if "load_blend_slot_mean_weight" in i["message"]
    ]


# --- config plumbing -----------------------------------------------------------------


def test_default_config_ships_both_options_off_and_keeps_the_load_floor():
    default_cfg = yaml.safe_load((REPO / "config.default.yaml").read_text(encoding="utf-8"))
    forecasting = default_cfg["forecasting"]
    assert forecasting["load_blend_slot_mean_weight"] == 0.0
    assert forecasting["pv_training_weather_source"] == "forecast_past_days"
    # kill switches ship ON (= the new default behaviour)
    assert forecasting["training_data_cleaning_enabled"] is True
    assert forecasting["load_slot_mean_fallback_enabled"] is True
    # this site's live load floor: must never be removed or changed by this work
    assert forecasting["load_safety_margin_percent"] == 117.0
    assert default_cfg["learning"]["max_daily_param_change"]["load_safety_margin_percent"] == 1.0
    assert not [i for i in _validate_config_for_save(default_cfg) if "load_blend" in i["message"]]


def test_new_keys_survive_migration_and_merge():
    from ruamel.yaml import YAML

    ruamel = YAML()
    default_cfg = ruamel.load((REPO / "config.default.yaml").read_text(encoding="utf-8"))

    # an existing install without the new keys gets the (off) defaults ...
    old_user: dict[str, Any] = {"forecasting": {"load_safety_margin_percent": 117.0}}
    old_user, _ = remove_deprecated_keys(old_user)
    merged = ruamel.load((REPO / "config.default.yaml").read_text(encoding="utf-8"))
    template_aware_merge(merged, old_user)
    assert merged["forecasting"]["load_blend_slot_mean_weight"] == 0.0
    assert merged["forecasting"]["training_data_cleaning_enabled"] is True
    assert merged["forecasting"]["load_slot_mean_fallback_enabled"] is True
    assert merged["forecasting"]["pv_training_weather_source"] == "forecast_past_days"
    assert merged["forecasting"]["load_safety_margin_percent"] == 117.0

    # ... and values the owner sets are kept, floor included
    user: dict[str, Any] = {
        "forecasting": {
            "load_blend_slot_mean_weight": 0.4,
            "training_data_cleaning_enabled": False,
            "load_slot_mean_fallback_enabled": False,
            "pv_training_weather_source": "historical_forecast",
            "load_safety_margin_percent": 109.0,
        }
    }
    user, _ = remove_deprecated_keys(user)
    template_aware_merge(default_cfg, user)
    assert default_cfg["forecasting"]["load_blend_slot_mean_weight"] == 0.4
    assert default_cfg["forecasting"]["training_data_cleaning_enabled"] is False
    assert default_cfg["forecasting"]["load_slot_mean_fallback_enabled"] is False
    assert default_cfg["forecasting"]["pv_training_weather_source"] == "historical_forecast"
    assert default_cfg["forecasting"]["load_safety_margin_percent"] == 109.0


# --- the blend in ml.forward ------------------------------------------------------------


@pytest.mark.asyncio
async def test_weight_zero_is_bit_identical_and_never_touches_the_db():
    _, reference = await _run(None, set_key=False)
    for weight in (0, 0.0, None, "0"):
        engine, forecasts = await _run(weight)
        assert forecasts == reference  # exact equality, every field of every slot
        engine.store.get_observation_rows_between.assert_not_awaited()
        assert health.get_load_forecast_status()["tier"] == "model"
    assert [f["load_forecast_kwh"] for f in reference] == [MODEL["load_p50"]] * 8


@pytest.mark.asyncio
async def test_invalid_weight_leaves_the_model_output_untouched():
    _, reference = await _run(None, set_key=False)
    engine, forecasts = await _run(40)
    assert forecasts == reference
    engine.store.get_observation_rows_between.assert_not_awaited()


@pytest.mark.asyncio
async def test_blend_moves_p50_toward_the_slot_mean_and_shifts_the_band(caplog):
    with caplog.at_level(logging.INFO, logger="darkstar.ml.forward"):
        _, forecasts = await _run(0.6)

    for f, mean in zip(forecasts, MEANS, strict=True):
        delta = 0.6 * (mean - MODEL["load_p50"])
        assert f["load_forecast_kwh"] == pytest.approx(0.4 * MODEL["load_p50"] + 0.6 * mean)
        assert f["base_load_forecast_kwh"] == f["load_forecast_kwh"]
        assert f["load_p10"] == pytest.approx(max(0.01, MODEL["load_p10"] + delta))
        assert f["load_p90"] == pytest.approx(MODEL["load_p90"] + delta)
        assert f["load_p10"] <= f["load_forecast_kwh"] <= f["load_p90"]
    assert health.get_load_forecast_status()["tier"] == "model+slot_mean_7d"
    assert "Load forecast blend: 60% 7-day slot mean / 40% model on 8/8 slots" in caplog.text


@pytest.mark.asyncio
async def test_blend_weight_one_is_the_slot_mean():
    _, forecasts = await _run(1.0)
    assert [f["load_forecast_kwh"] for f in forecasts] == pytest.approx(MEANS)


@pytest.mark.asyncio
async def test_blend_on_under_half_the_slots_is_a_warning_and_visible(caplog):
    partial = {key: value for i, (key, value) in enumerate(ALL_KEYS.items()) if i < 3}
    with caplog.at_level(logging.INFO, logger="darkstar.ml.forward"):
        _, forecasts = await _run(0.5, history=_history(partial))
    loads = [f["load_forecast_kwh"] for f in forecasts]
    assert loads[:3] == pytest.approx([0.35, 0.4, 0.45])
    assert loads[3:] == [MODEL["load_p50"]] * 5  # no slot mean -> pure model

    (record,) = [r for r in caplog.records if "Load forecast blend" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "on 3/8 slots" in record.getMessage() and "ENABLED" in record.getMessage()
    assert health.get_load_forecast_status()["tier"] == "model+slot_mean_7d_partial"
    checker = health.HealthChecker.__new__(health.HealthChecker)
    (issue,) = checker.check_load_forecast()
    assert issue.severity == "warning"
    assert issue.message == "Load forecast blend applied to fewer than half the slots"


@pytest.mark.asyncio
async def test_enabled_blend_that_reaches_no_slot_is_not_a_silent_noop(caplog):
    with caplog.at_level(logging.INFO, logger="darkstar.ml.forward"):
        _, forecasts = await _run(0.5, history=[])
    assert [f["load_forecast_kwh"] for f in forecasts] == [MODEL["load_p50"]] * 8

    (record,) = [r for r in caplog.records if "Load forecast blend" in r.getMessage()]
    assert record.levelno == logging.WARNING and "on 0/8 slots" in record.getMessage()
    assert health.get_load_forecast_status()["tier"] == "model_blend_inactive"
    checker = health.HealthChecker.__new__(health.HealthChecker)
    (issue,) = checker.check_load_forecast()
    assert issue.message == "Load forecast blend is enabled but not applied to any slot"


@pytest.mark.asyncio
async def test_blend_on_half_or_more_of_the_slots_stays_info_and_raises_no_issue(caplog):
    half = {key: value for i, (key, value) in enumerate(ALL_KEYS.items()) if i < 4}
    with caplog.at_level(logging.INFO, logger="darkstar.ml.forward"):
        await _run(0.5, history=_history(half))
    (record,) = [r for r in caplog.records if "Load forecast blend" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert health.get_load_forecast_status()["tier"] == "model+slot_mean_7d"
    assert health.HealthChecker.__new__(health.HealthChecker).check_load_forecast() == []


@pytest.mark.asyncio
async def test_blend_still_works_through_a_week_of_an_offline_device():
    """The reviewer's case: every slot of the week flagged 'unavailable' for one heater."""
    offline = '{"water_unmeasured": {"spa": "unavailable"}}'
    rows = [
        {
            "slot_start": (FIXED_NOW - timedelta(days=7) + timedelta(minutes=15 * i)).isoformat(),
            "load_kwh": 0.8,
            "quality_flags": offline,
        }
        for i in range(7 * 96)
    ]
    _, forecasts = await _run(0.5, history=rows)
    assert [f["load_forecast_kwh"] for f in forecasts] == pytest.approx([0.65] * 8)
    assert health.get_load_forecast_status()["tier"] == "model+slot_mean_7d"


# --- scripts/compare_load_forecasts.py ----------------------------------------------------


def _load_script() -> Any:
    path = REPO / "scripts" / "compare_load_forecasts.py"
    spec = importlib.util.spec_from_file_location("compare_load_forecasts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def load_db(tmp_path_factory) -> Path:
    """14 days, actual load 0.5 flat; aurora says 0.4 (20 % low), baseline says 0.5."""
    tz = pytz.timezone("Europe/Stockholm")
    slots = pd.date_range(tz.localize(datetime(2026, 9, 1)), periods=14 * 96, freq="15min")
    db = tmp_path_factory.mktemp("loaddb") / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE slot_observations (slot_start TEXT PRIMARY KEY, load_kwh REAL, "
            "quality_flags TEXT)"
        )
        conn.execute(
            "CREATE TABLE slot_forecasts (slot_start TEXT, forecast_version TEXT, "
            "load_forecast_kwh REAL, base_load_forecast_kwh REAL)"
        )
        for i, ts in enumerate(slots):
            iso = ts.isoformat()
            flags = '{"ev_unmeasured": {"easee": "snapshot"}}' if i == 5 else None
            conn.execute(
                "INSERT INTO slot_observations VALUES (?, ?, ?)",
                (iso, 9.0 if i == 5 else 0.5, flags),
            )
            conn.execute("INSERT INTO slot_forecasts VALUES (?, 'aurora', 0.4, 0.4)", (iso,))
            conn.execute(
                "INSERT INTO slot_forecasts VALUES (?, 'baseline_7_day_avg', 0.5, 0.0)", (iso,)
            )
    return db


def test_compare_load_script_scores_variants_and_blends(load_db):
    script = _load_script()
    obs, forecasts = script.load_frames(str(load_db), "Europe/Stockholm")
    frame, variants, counts = script.build_comparison(obs, forecasts, [0.5, 1.0])

    assert variants == ["aurora", "baseline_7d", "slot_mean", "blend 0.5", "blend 1"]
    assert counts["observations"] == 14 * 96
    assert counts["usable_observations"] == 14 * 96 - 1  # the flagged slot is excluded
    assert frame["baseline_7d"].eq(0.5).all()  # legacy column used when base is 0

    aurora = script.stats(frame["aurora"], frame["load_kwh"])
    assert aurora["bias"] == pytest.approx(-0.1)
    assert aurora["bias_pct"] == pytest.approx(-20.0)
    assert aurora["under"] == pytest.approx(100.0)
    assert aurora["zero_bias_floor"] == pytest.approx(125.0)

    # slot mean needs 3 earlier days; from then on it equals the flat 0.5 load
    late = frame[frame["slot_start"] >= frame["slot_start"].min() + timedelta(days=3)]
    # (the quarter-hour whose day-0 sample was flagged has only 2 samples on day 3)
    assert late["slot_mean"].isna().sum() == 1
    late = late[late["slot_mean"].notna()]
    assert late["slot_mean"].eq(0.5).all()
    assert frame["slot_mean"].isna().sum() == 3 * 96
    assert late["blend 0.5"].eq(0.45).all()
    assert script.stats(late["blend 1"], late["load_kwh"])["mae"] == pytest.approx(0.0)
    # no slot mean yet -> the blend is the pure model, as in production
    early = frame[frame["slot_mean"].isna()]
    assert early["blend 1"].eq(0.4).all()


def test_compare_load_script_main_is_read_only_and_prints_all_tables(load_db, capsys):
    script = _load_script()
    before = load_db.read_bytes()

    code = script.main(["--db", str(load_db), "--floor", "117", "--weights", "0.4,0.6"])

    out = capsys.readouterr().out
    assert code == 0
    for heading in (
        "== Overall (raw, and with the 117 % floor) ==",
        "== Per hour (local), raw ==",
        "== Per hour (local), with the 117 % floor ==",
        "== Per weekday, raw ==",
        "== Per weekday, with the 117 % floor ==",
    ):
        assert heading in out
    assert "blend 0.4" in out and "blend 0.6" in out and "Mon" in out and "Sun" in out
    assert load_db.read_bytes() == before
    assert "'aurora' is the stored forecast, not always the pure model" in out
    conn = script.open_readonly(str(load_db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM slot_forecasts")
    finally:
        conn.close()


def test_scripts_close_the_db_and_immutable_leaves_no_sidecar_files(load_db, tmp_path, capsys):
    """A WAL-mode COPY opened with --immutable gets no -shm/-wal beside it."""
    import shutil

    copy = tmp_path / "copy.db"
    shutil.copy(load_db, copy)
    with sqlite3.connect(copy) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    for leftover in tmp_path.glob("copy.db-*"):
        leftover.unlink()
    before = copy.read_bytes()

    script = _load_script()
    opened: list[sqlite3.Connection] = []
    real_open = script.open_readonly

    def _tracking_open(path: str, immutable: bool = False) -> sqlite3.Connection:
        opened.append(real_open(path, immutable))
        return opened[-1]

    script.open_readonly = _tracking_open
    assert script.main(["--db", str(copy), "--immutable"]) == 0
    capsys.readouterr()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy.db"]
    assert copy.read_bytes() == before
    assert opened
    for conn in opened:  # closed: any use now raises
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
