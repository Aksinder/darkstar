"""Load training must not treat balance-derived load as measured load.

The recorder writes ``quality_flags = {"load_rescued": "energy_balance"}`` when the
load register glitched and load_kwh was computed from the energy-balance identity,
and the statistics repair writes ``{"repaired": "statistics_backfill"}`` (flat
within-hour balance estimate). On Burgbyn10 the rescue fires on ~90% of high-PV
quarters, so DROPPING those rows would gut sunny-daytime load coverage. They are
instead down-weighted. PV training must be unaffected.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytz

import ml.train as train_mod
from backend.learning import LearningEngine

TZ = pytz.timezone("Europe/Stockholm")


def _engine() -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.timezone = TZ
    engine.db_path = ":memory:"
    engine.config = {}
    return engine


def _obs() -> pd.DataFrame:
    slots = pd.date_range(start=TZ.localize(datetime(2026, 7, 9, 0, 0)), periods=400, freq="15min")
    rescued = json.dumps({"load_rescued": "energy_balance"})
    repaired = json.dumps({"repaired": "statistics_backfill"})
    other = json.dumps({"soc_stale": True})
    flags: list[str | None] = []
    for i in range(len(slots)):
        flags.append([None, rescued, repaired, other, "{}"][i % 5])
    return pd.DataFrame(
        {
            "slot_start": slots,
            "load_kwh": 0.3,
            "pv_kwh": 0.5,
            "quality_flags": flags,
        }
    )


def _run(obs: pd.DataFrame) -> tuple[dict, list[dict]]:
    calls: list[dict] = []

    def _regressor(features, target, min_samples, alpha=0.5, sample_weight=None):
        calls.append(
            {"n": len(target), "w": np.asarray(sample_weight), "cols": list(features.columns)}
        )
        return None

    with (
        patch("ml.train.get_learning_engine", return_value=_engine()),
        patch("ml.train._load_slot_observations", return_value=obs.copy()),
        patch("ml.train.get_weather_series", return_value=pd.DataFrame()),
        patch("ml.train.get_vacation_mode_series", return_value=pd.Series(dtype=float)),
        patch("ml.train.get_alarm_armed_series", return_value=pd.Series(dtype=float)),
        patch(
            "ml.train._compute_sample_weights", side_effect=lambda df, config=None: np.ones(len(df))
        ),
        patch("ml.train._train_regressor", side_effect=_regressor),
    ):
        result = train_mod.train_models()
    return result, calls


def test_derived_load_mask_reads_recorder_and_repair_flags():
    obs = _obs()
    mask = train_mod._derived_load_mask(obs)
    expected = [i % 5 in (1, 2) for i in range(len(obs))]
    assert mask.tolist() == expected


def test_derived_load_mask_tolerates_missing_column_and_bad_json():
    df = pd.DataFrame({"load_kwh": [1.0, 1.0, 1.0]})
    assert not train_mod._derived_load_mask(df).any()
    df["quality_flags"] = ["not json", '{"load_rescued": "energy_balance"', None]
    # Truncated JSON still carries the key: substring fallback catches it.
    assert train_mod._derived_load_mask(df).tolist() == [False, True, False]


def test_load_training_downweights_rescued_and_repaired_rows():
    obs = _obs()
    result, calls = _run(obs)

    load_calls = [c for c in calls if "physics_forecast_kwh" not in c["cols"]]
    assert len(load_calls) == 3  # p10/p50/p90
    weights = load_calls[0]["w"]
    # Nothing is dropped: coverage of sunny quarters is preserved.
    assert load_calls[0]["n"] == len(obs)

    derived = np.array([i % 5 in (1, 2) for i in range(len(obs))])
    assert np.allclose(weights[derived], train_mod.DERIVED_LOAD_WEIGHT)
    assert np.allclose(weights[~derived], 1.0)
    assert 0.0 < train_mod.DERIVED_LOAD_WEIGHT < 1.0

    assert result["load_derived_rows"] == int(derived.sum())
