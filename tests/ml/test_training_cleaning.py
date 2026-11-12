"""Training must not learn from slots the recorder flagged as not measured.

ml/cleaning.py is the shared rule set: PV drops ``pv_unmeasured``; load drops
``water_unmeasured`` / ``ev_unmeasured`` / ``base_load_clamped`` /
``load_spike_zeroed`` (and a balance-derived load computed from unmeasured PV).
Tokens that never occur must be an exact no-op, and a rule that would eat most of
the data must not starve training.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

import ml.train as train_mod
from backend.learning import LearningEngine
from ml.cleaning import (
    CLEANING_ENABLED_KEY,
    DERIVED_FROM_UNMEASURED_PV,
    DERIVED_FROM_ZEROED_COLUMN,
    LOAD_BAD_FLAGS,
    PV_BAD_FLAGS,
    PV_SPIKE_ZEROED,
    bad_row_mask,
    clean_training_rows,
    forecasting_switch,
    has_flag,
    spike_zeroed_column,
)

TZ = pytz.timezone("Europe/Stockholm")


def _frame(flags: list[object]) -> pd.DataFrame:
    slots = pd.date_range(
        start=TZ.localize(datetime(2026, 7, 9, 0, 0)), periods=len(flags), freq="15min"
    )
    return pd.DataFrame(
        {"slot_start": slots, "load_kwh": 0.3, "pv_kwh": 0.5, "quality_flags": flags}
    )


def _silent(_msg: str) -> None:
    return None


def test_flag_contract_tokens_are_fixed():
    # Contract with backend/recorder.py — renaming a token silently disables a rule.
    assert PV_BAD_FLAGS == ("pv_unmeasured",)
    assert LOAD_BAD_FLAGS == (
        "water_unmeasured",
        "ev_unmeasured",
        "base_load_clamped",
        "load_spike_zeroed",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (json.dumps({"pv_unmeasured": "snapshot"}), True),
        (json.dumps({"pv_unmeasured": "counter", "soc_stale": True}), True),
        (json.dumps(["pv_unmeasured"]), True),
        ({"pv_unmeasured": "snapshot"}, True),
        ('{"pv_unmeasured": "snap', True),  # truncated JSON -> substring match
        (json.dumps({"pv_unmeasured": False}), False),
        (json.dumps({"pv_unmeasured": None}), False),
        (json.dumps({"note": "pv_unmeasured"}), False),  # token as a VALUE is not a flag
        (json.dumps({"soc_stale": True}), False),
        ("{}", False),
        ("", False),
        (None, False),
        (float("nan"), False),
    ],
)
def test_has_flag_parses_recorder_formats(raw, expected):
    assert has_flag(raw, ("pv_unmeasured",)) is expected


@pytest.mark.parametrize(
    ("raw", "column", "expected"),
    [
        (json.dumps({"load_spike_zeroed": ["import_kwh", "load_kwh", "pv_kwh"]}), "load_kwh", True),
        (json.dumps({"load_spike_zeroed": ["import_kwh", "load_kwh", "pv_kwh"]}), "pv_kwh", True),
        (json.dumps({"load_spike_zeroed": ["import_kwh"]}), "load_kwh", False),
        (json.dumps({"load_spike_zeroed": ["import_kwh"]}), "pv_kwh", False),
        (json.dumps({"load_spike_zeroed": ["pv_kwh"]}), "load_kwh", False),
        (json.dumps({"load_spike_zeroed": []}), "load_kwh", False),
        ({"load_spike_zeroed": ["load_kwh"]}, "load_kwh", True),  # already parsed
        (json.dumps({"load_spike_zeroed": "pv_kwh"}), "pv_kwh", True),  # bare string
        (json.dumps({"load_spike_zeroed": True}), "load_kwh", True),  # no list: name says load
        (json.dumps({"load_spike_zeroed": True}), "pv_kwh", False),
        (json.dumps({"load_spike_zeroed": False}), "load_kwh", False),
        ('{"load_spike_zeroed": ["load_kwh", "pv_', "load_kwh", True),  # truncated JSON
        ('{"load_spike_zeroed": ["load_kwh", "pv_', "pv_kwh", False),
        (json.dumps({"soc_stale": True}), "load_kwh", False),
        (None, "load_kwh", False),
        ("not json", "load_kwh", False),
        (17, "load_kwh", False),
    ],
)
def test_spike_zeroed_is_matched_per_column(raw, column, expected):
    assert spike_zeroed_column(raw, column) is expected


def test_pv_rule_drops_pv_unmeasured_and_zeroed_pv_only():
    df = _frame(
        [
            None,
            json.dumps({"pv_unmeasured": "snapshot"}),
            json.dumps({"water_unmeasured": {"main": "snapshot"}}),
            json.dumps({"load_rescued": "energy_balance"}),
            "{}",
            json.dumps({"load_spike_zeroed": ["import_kwh", "pv_kwh"]}),
            json.dumps({"load_spike_zeroed": ["load_kwh"]}),  # PV untouched -> kept
            json.dumps({"ev_unmeasured": {"easee": "unavailable"}}),
        ]
        * 4
    )
    out, report = clean_training_rows(df, "pv", log=_silent)
    assert len(out) == 24
    assert report.dropped_by_rule == {"pv_unmeasured": 4, PV_SPIKE_ZEROED: 4}
    assert not report.starved
    assert not bad_row_mask(out, "pv").any()


def test_load_rules_drop_each_contract_flag_and_keep_plain_pv_unmeasured():
    flags = [
        None,
        json.dumps({"water_unmeasured": {"main": "unavailable"}}),
        json.dumps({"ev_unmeasured": {"easee": "snapshot"}}),
        json.dumps({"base_load_clamped": True}),
        json.dumps({"load_spike_zeroed": ["load_kwh", "pv_kwh"]}),
        # another column was zeroed, the METERED load is fine -> kept
        json.dumps({"load_spike_zeroed": ["export_kwh"]}),
        # ... but a balance-derived load was computed from the zeroed column -> dropped
        json.dumps({"load_spike_zeroed": ["import_kwh"], "load_rescued": "energy_balance"}),
        # PV fallback with a METERED load: load target unaffected -> kept
        json.dumps({"pv_unmeasured": "snapshot"}),
        # ... but a balance-derived load was computed from that PV -> dropped
        json.dumps({"pv_unmeasured": "snapshot", "load_rescued": "energy_balance"}),
        json.dumps({"load_rescued": "energy_balance"}),  # down-weighted elsewhere, kept
        "{}",
        json.dumps({"soc_stale": True}),
    ] * 3
    out, report = clean_training_rows(_frame(flags), "load", log=_silent)
    assert report.dropped_by_rule == {
        "water_unmeasured": 3,
        "ev_unmeasured": 3,
        "base_load_clamped": 3,
        "load_spike_zeroed": 3,
        DERIVED_FROM_UNMEASURED_PV: 3,
        DERIVED_FROM_ZEROED_COLUMN: 3,
    }
    assert len(out) == 36 - 18
    assert report.rows_dropped == 18


def test_absent_flags_are_an_exact_noop():
    """Historical rows carry none of the tokens: same object back, nothing logged."""
    df = _frame([None, "{}", json.dumps({"soc_stale": True}), json.dumps({"repaired": "x"})] * 5)
    logged: list[str] = []
    for target in ("pv", "load"):
        out, report = clean_training_rows(df, target, min_rows=1000, log=logged.append)
        assert out is df
        assert report.dropped_by_rule == {}
        assert report.rows_in == report.rows_out == len(df)
        assert not report.starved
    assert logged == []

    no_col = df.drop(columns=["quality_flags"])
    out, _ = clean_training_rows(no_col, "load", log=logged.append)
    assert out is no_col


def test_starvation_guard_majority_removed_falls_back_unfiltered(caplog):
    bad = json.dumps({"pv_unmeasured": "counter"})
    df = _frame([bad] * 60 + [None] * 40)
    with caplog.at_level(logging.WARNING, logger="darkstar.ml.cleaning"):
        out, report = clean_training_rows(df, "pv", log=_silent)
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert out is df
    assert report.starved
    assert report.rows_out == 100
    assert report.dropped_by_rule == {"pv_unmeasured": 60}
    assert len(logged) == 1 and "starvation guard" in logged[0]


def test_starvation_guard_below_min_rows_falls_back_unfiltered():
    bad = json.dumps({"ev_unmeasured": True})
    df = _frame([bad] * 30 + [None] * 90)  # 25% removed, but 90 < min_rows <= 120
    out, report = clean_training_rows(df, "load", min_rows=100, log=_silent)
    assert out is df and report.starved

    # Unfiltered set was already too small: the guard has nothing to save -> clean.
    out, report = clean_training_rows(df, "load", min_rows=500, log=_silent)
    assert len(out) == 90 and not report.starved


# --- offline devices (per-device "unavailable" runs) -------------------------------


def _unavailable(*devices: str, flag: str = "water_unmeasured") -> str:
    return json.dumps({flag: dict.fromkeys(devices, "unavailable")})


def test_device_offline_for_weeks_keeps_the_newest_rows():
    """Winterised spa: 'unavailable' on every slot of the 44 newest days of 90."""
    flags: list[object] = [None] * (46 * 96) + [_unavailable("spa")] * (44 * 96)
    df = _frame(flags)
    logged: list[str] = []

    out, report = clean_training_rows(df, "load", min_rows=100, log=logged.append)

    assert out is df  # nothing dropped at all
    assert report.dropped_by_rule == {}
    assert report.kept_offline_device == {"water_unmeasured:spa": 44 * 96}
    assert not report.starved
    assert any("offline devices" in line and "water_unmeasured:spa=4224" in line for line in logged)
    assert not bad_row_mask(df, "load").any()


def test_short_unavailable_gap_is_still_dropped():
    flags: list[object] = [None] * 400
    flags[100:108] = [_unavailable("spa")] * 8  # 2 hours
    out, report = clean_training_rows(_frame(flags), "load", log=_silent)

    assert len(out) == 392
    assert report.dropped_by_rule == {"water_unmeasured": 8}
    assert report.kept_offline_device == {}


def test_run_just_under_a_day_is_a_gap_and_a_full_day_is_offline():
    almost = [None] * 10 + [_unavailable("spa")] * 95 + [None] * 10
    _, report = clean_training_rows(_frame(almost), "load", log=_silent)
    assert report.dropped_by_rule == {"water_unmeasured": 95}

    full = [None] * 10 + [_unavailable("spa")] * 96 + [None] * 10
    out, report = clean_training_rows(_frame(full), "load", log=_silent)
    assert len(out) == 116 and report.kept_offline_device == {"water_unmeasured:spa": 96}


def test_a_measured_slot_in_between_splits_the_run():
    """Two 20 h outages separated by one measured slot are two gaps, not one day."""
    flags = [None] * 200 + [_unavailable("spa")] * 80 + [None] + [_unavailable("spa")] * 80
    out, report = clean_training_rows(_frame(flags), "load", log=_silent)
    assert report.kept_offline_device == {}
    assert report.dropped_by_rule == {"water_unmeasured": 160}
    assert len(out) == 201


def test_two_devices_only_the_short_gap_slots_are_dropped():
    """Spa offline all day; the caravan heater has a 1-hour gap inside that day."""
    flags: list[object] = [None] * 96 + [_unavailable("spa")] * 96 + [None] * 96
    gap = range(96 + 40, 96 + 44)
    for i in gap:
        flags[i] = _unavailable("spa", "villavagn")
    df = _frame(flags)

    out, report = clean_training_rows(df, "load", log=_silent)

    dropped = df.index.difference(out.index)
    assert list(dropped) == list(gap)
    assert report.dropped_by_rule == {"water_unmeasured": 4}
    assert report.kept_offline_device == {"water_unmeasured:spa": 96}


def test_snapshot_and_unknown_values_never_count_as_offline():
    snapshot = json.dumps({"ev_unmeasured": {"easee": "snapshot"}})
    legacy = json.dumps({"ev_unmeasured": True})
    for value in (snapshot, legacy):
        flags = [None] * 150 + [value] * 100  # 25 h, but not "unavailable"
        _, report = clean_training_rows(_frame(flags), "load", log=_silent)
        assert report.dropped_by_rule == {"ev_unmeasured": 100}
        assert report.kept_offline_device == {}


def test_offline_device_does_not_shield_other_rules():
    offline_and_clamped = json.dumps(
        {"ev_unmeasured": {"easee": "unavailable"}, "base_load_clamped": True}
    )
    flags = [None] * 100 + [_unavailable("easee", flag="ev_unmeasured")] * 100
    flags[150] = offline_and_clamped
    out, report = clean_training_rows(_frame(flags), "load", log=_silent)
    assert len(out) == 199
    assert report.dropped_by_rule == {"base_load_clamped": 1}
    assert report.kept_offline_device == {"ev_unmeasured:easee": 100}


def test_offline_rule_works_on_iso_string_slot_starts_and_unsorted_rows():
    """The slot-mean path passes store rows: ISO strings, any order."""
    df = _frame([_unavailable("spa")] * 200 + [None] * 50)
    df["slot_start"] = df["slot_start"].map(lambda ts: ts.isoformat())
    shuffled = df.sample(frac=1.0, random_state=1)
    assert not bad_row_mask(shuffled, "load").any()

    no_time = df.drop(columns=["slot_start"])  # cannot measure a run -> conservative
    assert bad_row_mask(no_time, "load").sum() == 200


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        clean_training_rows(_frame([None]), "price", log=_silent)


# --- wiring into ml.train.train_models -------------------------------------------


def _engine(config: dict | None = None) -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.timezone = TZ
    engine.db_path = ":memory:"
    engine.config = config or {}
    return engine


def _run(obs: pd.DataFrame, config: dict | None = None) -> tuple[dict, list[dict]]:
    calls: list[dict] = []

    def _regressor(features, target, min_samples, alpha=0.5, sample_weight=None):
        calls.append({"n": len(target), "w": np.asarray(sample_weight)})
        return None

    with (
        patch("ml.train.get_learning_engine", return_value=_engine(config)),
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


def test_train_models_applies_rules_per_target():
    flags: list[str | None] = []
    for i in range(400):
        flags.append(
            [
                None,
                json.dumps({"pv_unmeasured": "snapshot"}),
                json.dumps({"water_unmeasured": {"main": "snapshot"}}),
                "{}",
            ][i % 4]
        )
    result, calls = _run(_frame(flags))

    # 3 load quantiles then 3 PV quantiles
    assert [c["n"] for c in calls] == [300, 300, 300, 300, 300, 300]
    assert result["load_samples"] == 300
    assert result["pv_samples"] == 300
    assert result["cleaning"]["load"]["dropped_by_rule"] == {"water_unmeasured": 100}
    assert result["cleaning"]["pv"]["dropped_by_rule"] == {"pv_unmeasured": 100}
    # sample weights stay aligned with the surviving rows
    assert all(len(c["w"]) == c["n"] for c in calls)


def test_train_models_unflagged_history_trains_on_every_row():
    result, calls = _run(_frame([None, "{}"] * 200))
    assert [c["n"] for c in calls] == [400] * 6
    assert result["cleaning"]["load"]["dropped_by_rule"] == {}
    assert result["cleaning"]["pv"]["dropped_by_rule"] == {}


# --- kill switch: forecasting.training_data_cleaning_enabled -------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),
        (True, True),
        ("true", True),
        ("on", True),
        (1, True),
        ("garbage", True),  # unreadable -> default
        (False, False),
        ("false", False),
        ("Off", False),
        ("no", False),
        (0, False),
    ],
)
def test_forecasting_switch_parsing(raw, expected):
    cfg = {"forecasting": {CLEANING_ENABLED_KEY: raw}}
    assert forecasting_switch(cfg, CLEANING_ENABLED_KEY) is expected
    assert forecasting_switch({}, CLEANING_ENABLED_KEY) is True
    assert forecasting_switch({"forecasting": None}, CLEANING_ENABLED_KEY) is True
    assert forecasting_switch(None, CLEANING_ENABLED_KEY, default=False) is False


def _mixed_flags() -> list[str | None]:
    return [
        [
            None,
            json.dumps({"pv_unmeasured": "snapshot"}),
            json.dumps({"water_unmeasured": {"main": "snapshot"}}),
            json.dumps({"load_spike_zeroed": ["load_kwh", "pv_kwh"]}),
        ][i % 4]
        for i in range(400)
    ]


def test_cleaning_disabled_trains_on_exactly_the_rows_of_unflagged_history():
    """Off = the pre-cleaning trainer: every row, same weights, nothing recorded."""
    off = {"forecasting": {CLEANING_ENABLED_KEY: False}}
    flagged_result, flagged_calls = _run(_frame(_mixed_flags()), off)
    plain_result, plain_calls = _run(_frame([None] * 400), off)

    assert [c["n"] for c in flagged_calls] == [400] * 6
    assert flagged_result["cleaning"] == {}
    assert flagged_result["load_samples"] == flagged_result["pv_samples"] == 400
    for with_flags, without in zip(flagged_calls, plain_calls, strict=True):
        assert with_flags["n"] == without["n"]
        assert np.array_equal(with_flags["w"], without["w"])
    assert plain_result["cleaning"] == {}


def test_cleaning_enabled_by_default_on_the_same_data():
    result, calls = _run(_frame(_mixed_flags()))
    # load loses the water and the zeroed-load rows; PV the unmeasured and zeroed-PV rows
    assert [c["n"] for c in calls] == [200] * 6
    assert set(result["cleaning"]) == {"load", "pv"}
