"""Regression guards for Phase 1b training correctness.

1. TRAINING physics must use the same Open-Meteo native DNI/DHI transposition as
   INFERENCE physics (ml/forward.py). The ML residual target is
   `actual - physics`; if training computes `physics` from GHI-only transposition
   while inference computes it from DNI/DHI, the residual is fit against one
   physics and applied on another, re-introducing the morning over-forecast the
   DNI/DHI fix removed. The assertions pin the EXACT per-slot values forwarded
   (not just non-None), so swapped dni/dhi arguments or forwarding GHI into the
   DNI slot also fail.

2. PV slots without weather coverage (physics None) must be DROPPED from PV
   training, not trained as residual = actual - 0.0.

3. The PV clean-data floor (forecasting.pv_training_min_date) must bound PV
   training on every path while leaving load training untouched.

4. train_models must report what it actually saved (models_saved) so callers
   don't glob stale files.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytz

import ml.train as train_mod
from backend.learning import LearningEngine
from ml.train import _resolve_pv_training_min_date
from ml.weather import calculate_physics_pv as _real_calculate_physics_pv

TZ = pytz.timezone("Europe/Stockholm")


def _mock_engine(tmp_db: str, forecasting: dict | None = None) -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.timezone = TZ
    engine.db_path = tmp_db
    engine.config = {
        "system": {
            "location": {"latitude": 57.6097, "longitude": 18.4146},
            "solar_arrays": [
                {"name": "Fronius SW", "kwp": 11.4, "azimuth": 230, "tilt": 30},
                {"name": "Sungrow NE", "kwp": 4.0, "azimuth": 50, "tilt": 30},
            ],
        },
        "forecasting": forecasting or {},
    }
    return engine


def _synthetic_clean_window() -> pd.DataFrame:
    """~4 clean days (2026-07-09..07-12) of 15-min slots, daytime PV/radiation."""
    slots = pd.date_range(
        start=TZ.localize(datetime(2026, 7, 9, 0, 0)),
        end=TZ.localize(datetime(2026, 7, 12, 23, 45)),
        freq="15min",
    )
    rows = []
    for ts in slots:
        h = ts.hour + ts.minute / 60.0
        # Daylight roughly 04:00-22:00 on Gotland in July.
        day = 4.0 <= h <= 22.0
        # Bell-ish radiation peaking at solar noon.
        ghi = max(0.0, 850.0 * np.cos((h - 13.0) / 9.0)) if day else 0.0
        rows.append(
            {
                "slot_start": ts,
                "load_kwh": 0.25,
                "pv_kwh": round(ghi / 850.0 * 3.0, 3),  # up to ~3 kWh/slot
            }
        )
    return pd.DataFrame(rows)


def _weather_series(index: pd.DatetimeIndex) -> pd.DataFrame:
    """15-min weather frame carrying GHI plus native DNI/DHI for every slot.

    DNI, DHI and GHI are constructed to be pairwise DIFFERENT on daytime slots
    so assertions can detect swapped or mis-sourced forwarding.
    """
    data = []
    for ts in index:
        h = ts.hour + ts.minute / 60.0
        day = 4.0 <= h <= 22.0
        ghi = max(0.0, 850.0 * np.cos((h - 13.0) / 9.0)) if day else 0.0
        # Realistic-ish clear-sky split; DNI high midday, DHI a smaller share.
        dni = max(0.0, 780.0 * np.cos((h - 13.0) / 8.0)) if day else 0.0
        dhi = ghi * 0.25
        data.append(
            {
                "temp_c": 18.0,
                "cloud_cover_pct": 10.0,
                "shortwave_radiation_w_m2": ghi,
                "dni_w_m2": dni,
                "dhi_w_m2": dhi,
            }
        )
    return pd.DataFrame(data, index=index)


def _run_train(
    obs: pd.DataFrame,
    weather: pd.DataFrame,
    tmp_path,
    forecasting: dict | None = None,
    spy_physics: list | None = None,
    spy_regressor: list | None = None,
    spy_saves: list | None = None,
):
    """Run train_models with mocked data sources and optional spies."""

    def _physics(**kwargs):
        if spy_physics is not None:
            spy_physics.append(kwargs)
        return _real_calculate_physics_pv(**kwargs)

    def _regressor(features, target, min_samples, alpha=0.5, sample_weight=None):
        if spy_regressor is not None:
            spy_regressor.append(
                {"columns": list(features.columns), "n": len(target), "alpha": alpha}
            )
        return None  # skip actual LightGBM fit + save

    def _save(booster, path, feature_names=None):
        if spy_saves is not None:
            spy_saves.append(str(path))

    from contextlib import ExitStack

    with ExitStack() as stack:
        mock_engine = stack.enter_context(patch("ml.train.get_learning_engine"))
        mock_load = stack.enter_context(patch("ml.train._load_slot_observations"))
        mock_weather = stack.enter_context(patch("ml.train.get_weather_series"))
        stack.enter_context(
            patch("ml.train.get_vacation_mode_series", return_value=pd.Series(dtype=float))
        )
        stack.enter_context(
            patch("ml.train.get_alarm_armed_series", return_value=pd.Series(dtype=float))
        )
        stack.enter_context(
            patch("ml.train.calculate_physics_pv", side_effect=_physics)
        )
        if spy_regressor is not None:
            # Skip real LightGBM fits when the test only inspects sample sets.
            stack.enter_context(
                patch("ml.train._train_regressor", side_effect=_regressor)
            )
        stack.enter_context(patch("ml.train._save_model", side_effect=_save))

        mock_engine.return_value = _mock_engine(
            str(tmp_path / "learning.db"), forecasting=forecasting
        )
        mock_load.return_value = obs.copy()
        mock_weather.return_value = weather
        return train_mod.train_models(
            min_date=TZ.localize(datetime(2026, 7, 9, 0, 0)),
        )


class TestTrainDniDhiWiring:
    def test_training_forwards_exact_dni_dhi_per_slot(self, tmp_path):
        """Every physics call must carry the weather frame's OWN dni/dhi values
        for that slot — not swapped, not GHI, not a constant."""
        obs = _synthetic_clean_window()
        weather = _weather_series(pd.DatetimeIndex(obs["slot_start"]))
        physics_calls: list = []
        saves: list = []

        result = _run_train(
            obs, weather, tmp_path, spy_physics=physics_calls, spy_saves=saves
        )

        assert physics_calls, "calculate_physics_pv was never called during training"

        checked_positive_dni = 0
        for call in physics_calls:
            ts = call["slot_start"]
            expected_dni = float(weather.loc[ts, "dni_w_m2"])
            expected_dhi = float(weather.loc[ts, "dhi_w_m2"])
            got_dni = call.get("dni_w_m2")
            got_dhi = call.get("dhi_w_m2")
            assert got_dni is not None and got_dhi is not None, (
                f"slot {ts}: physics called without DNI/DHI — residual would be "
                "trained against GHI-only physics but applied on DNI/DHI physics."
            )
            assert float(got_dni) == expected_dni, (
                f"slot {ts}: dni_w_m2={got_dni} != weather frame {expected_dni} "
                "(swapped arguments or wrong source column?)"
            )
            assert float(got_dhi) == expected_dhi, (
                f"slot {ts}: dhi_w_m2={got_dhi} != weather frame {expected_dhi} "
                "(swapped arguments or wrong source column?)"
            )
            if expected_dni > 0:
                # On daytime slots dni != dhi != ghi by construction, so the
                # equality checks above genuinely discriminate the sources.
                assert expected_dni != expected_dhi
                checked_positive_dni += 1
        assert checked_positive_dni > 0, "no daytime slot exercised the DNI path"

        # (b) PV models were actually written (>= min_samples sun-up slots).
        pv_saves = [p for p in saves if "pv_model" in p]
        assert pv_saves, "No PV models were written — trainer produced nothing"
        assert sorted(set(result["models_saved"])) == sorted(
            {p.rsplit("/", 1)[-1] for p in saves}
        ), "models_saved must report exactly the files written"


class TestMissingWeatherRowsDropped:
    def test_uncovered_slots_excluded_from_pv_training(self, tmp_path):
        """Slots the weather fetch cannot cover (NaN radiation -> physics None)
        must be dropped, NOT trained as residual = actual - 0.0."""
        obs = _synthetic_clean_window()
        full_index = pd.DatetimeIndex(obs["slot_start"])
        # Weather covers only the last 2 of 4 days (mimics a past_days-limited fetch).
        covered = full_index[full_index >= TZ.localize(datetime(2026, 7, 11, 0, 0))]
        weather = _weather_series(covered)

        regressor_calls: list = []
        result = _run_train(obs, weather, tmp_path, spy_regressor=regressor_calls)

        pv_calls = [c for c in regressor_calls if "physics_forecast_kwh" in c["columns"]]
        load_calls = [
            c for c in regressor_calls if "physics_forecast_kwh" not in c["columns"]
        ]
        assert pv_calls and load_calls

        # Expected PV samples: sun-up slots INSIDE the covered window only.
        covered_obs = obs[obs["slot_start"].isin(covered)]
        expected_pv = int((covered_obs["pv_kwh"] > 0.01).sum())
        # Sun-up slots outside coverage exist (they'd have been trained before the fix).
        uncovered_sunup = int(
            (obs[~obs["slot_start"].isin(covered)]["pv_kwh"] > 0.01).sum()
        )
        assert uncovered_sunup > 0, "test setup: need uncovered daytime slots"

        for c in pv_calls:
            assert c["n"] == expected_pv, (
                f"PV trained on {c['n']} samples, expected {expected_pv} — "
                "uncovered slots (no physics baseline) must be dropped"
            )
        # Load models keep the FULL window (they don't need PV physics).
        for c in load_calls:
            assert c["n"] == len(obs)

        assert result["pv_rows_dropped_no_physics"] == uncovered_sunup
        assert result["pv_samples"] == expected_pv


class TestPvTrainingMinDateFloor:
    def test_floor_bounds_pv_but_not_load(self, tmp_path):
        """forecasting.pv_training_min_date excludes pre-floor rows from PV
        training while load training keeps the full window."""
        obs = _synthetic_clean_window()  # 2026-07-09 .. 2026-07-12
        weather = _weather_series(pd.DatetimeIndex(obs["slot_start"]))
        floor = TZ.localize(datetime(2026, 7, 11, 0, 0))

        regressor_calls: list = []
        result = _run_train(
            obs,
            weather,
            tmp_path,
            forecasting={"pv_training_min_date": "2026-07-11"},
            spy_regressor=regressor_calls,
        )

        pv_calls = [c for c in regressor_calls if "physics_forecast_kwh" in c["columns"]]
        load_calls = [
            c for c in regressor_calls if "physics_forecast_kwh" not in c["columns"]
        ]
        assert pv_calls and load_calls

        post_floor = obs[obs["slot_start"] >= floor]
        # In the synthetic data radiation > 10 implies pv > 0.01, so the sun-up
        # mask reduces to the pv_kwh test.
        expected_pv = int((post_floor["pv_kwh"] > 0.01).sum())
        for c in pv_calls:
            assert c["n"] == expected_pv, (
                f"PV trained on {c['n']} samples, expected {expected_pv} "
                "(only post-floor sun-up slots)"
            )
        for c in load_calls:
            assert c["n"] == len(obs), "load training must NOT be bounded by the PV floor"

        assert result["pv_training_min_date"] == floor.isoformat()

    def test_resolve_floor_variants(self, tmp_path):
        db = str(tmp_path / "x.db")
        # Unset / empty -> None
        assert _resolve_pv_training_min_date(_mock_engine(db)) is None
        assert (
            _resolve_pv_training_min_date(
                _mock_engine(db, forecasting={"pv_training_min_date": ""})
            )
            is None
        )
        # ISO string -> tz-aware midnight local
        got = _resolve_pv_training_min_date(
            _mock_engine(db, forecasting={"pv_training_min_date": "2026-07-09"})
        )
        assert got == TZ.localize(datetime(2026, 7, 9, 0, 0))
        # YAML date object (unquoted YAML dates parse as datetime.date)
        from datetime import date

        got = _resolve_pv_training_min_date(
            _mock_engine(db, forecasting={"pv_training_min_date": date(2026, 7, 9)})
        )
        assert got == TZ.localize(datetime(2026, 7, 9, 0, 0))
        # Garbage must raise (fail loud, never silently train on dirty data)
        import pytest

        with pytest.raises(ValueError):
            _resolve_pv_training_min_date(
                _mock_engine(db, forecasting={"pv_training_min_date": "not-a-date"})
            )
