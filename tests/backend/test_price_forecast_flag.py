"""Per-slot ``price_is_forecast``: ML-forecast D+1 prices are marked, published ones not.

The flag flows prices.py -> planner input frame -> schedule.json and must not
change planner economics (no solver term reads it).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytz

from backend.core import prices as P
from planner.inputs.data_prep import build_price_dataframe, prepare_df
from planner.output.formatter import dataframe_to_json_response
from planner.solver.adapter import planner_to_kepler_input

TZ = pytz.timezone("Europe/Stockholm")


def _entry(day: datetime, hour: int, value: float, **extra) -> dict:
    start = TZ.localize(day.replace(hour=hour))
    return {"start": start, "end": start + timedelta(hours=1), "value": value, **extra}


def test_process_marks_only_forecast_entries():
    day = datetime(2026, 4, 28)
    result = P._process_nordpool_data(
        [
            _entry(day, 10, 500.0),
            _entry(day + timedelta(days=1), 10, 300.0, price_is_forecast=True),
        ],
        {"timezone": "Europe/Stockholm"},
    )
    assert [s["price_is_forecast"] for s in result] == [False, True]
    assert all(isinstance(s["price_is_forecast"], bool) for s in result)


class _Morning(datetime):
    """10:00 local: D+1 not yet auctioned, so the ML fallback fills it."""

    @classmethod
    def now(cls, tz=None):
        return TZ.localize(datetime(2026, 4, 28, 10, 0))


@pytest.mark.asyncio
async def test_get_nordpool_data_flags_forecast_fallback(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "timezone: Europe/Stockholm\nnordpool:\n  price_area: SE3\n  resolution_minutes: 15\n"
    )
    today = TZ.localize(datetime(2026, 4, 28))
    raw_today = {
        "areas": {
            "SE3": {
                "values": [
                    {
                        "start": today + timedelta(minutes=15 * i),
                        "end": today + timedelta(minutes=15 * (i + 1)),
                        "value": 400.0,
                    }
                    for i in range(96)
                ]
            }
        }
    }
    tomorrow = today + timedelta(days=1)
    forecast = [
        {"slot_start": (tomorrow + timedelta(minutes=15 * i)).isoformat(), "spot_p50": 0.25}
        for i in range(96)
    ]

    P._fetch_fail_until = 0.0
    with (
        patch.object(P, "datetime", _Morning),
        patch.object(P.cache_sync, "get", return_value=None),
        patch.object(P.cache_sync, "set"),
        patch.object(P.Prices, "fetch", return_value=raw_today),
        patch("ml.price_forecast.get_d1_price_forecast_fallback", return_value=forecast),
    ):
        slots = await P.get_nordpool_data(str(cfg))

    published = [s for s in slots if s["start_time"] < tomorrow]
    forecasted = [s for s in slots if s["start_time"] >= tomorrow]
    assert len(published) == 96 and len(forecasted) == 96
    assert not any(s["price_is_forecast"] for s in published)
    assert all(s["price_is_forecast"] for s in forecasted)


def _price_slots() -> list[dict]:
    start = TZ.localize(datetime(2026, 4, 28, 23, 0))
    out = []
    for i in range(8):
        st = start + timedelta(minutes=15 * i)
        out.append(
            {
                "start_time": st,
                "end_time": st + timedelta(minutes=15),
                "import_price_sek_kwh": 1.0 + i / 10,
                "export_price_sek_kwh": 0.5 + i / 10,
                "price_is_forecast": st.day == 29,
            }
        )
    return out


def test_flag_reaches_planner_frame_and_schedule_json():
    df = prepare_df({"price_data": _price_slots(), "forecast_data": []}, "Europe/Stockholm")
    assert df["price_is_forecast"].tolist() == [False] * 4 + [True] * 4

    # Same shape as the pipeline's final merge: solver columns joined onto the input.
    result_df = pd.DataFrame({"battery_charge_kw": np.zeros(len(df))}, index=df.index)
    final_df = df.join(result_df)
    records = dataframe_to_json_response(
        final_df, now_override=datetime(2026, 4, 28, 22, 0), timezone_name="Europe/Stockholm"
    )
    assert [r["price_is_forecast"] for r in records] == [False] * 4 + [True] * 4
    json.dumps(records, default=str)  # plain JSON bools, no numpy leak
    assert all(type(r["price_is_forecast"]) is bool for r in records)


def test_formatter_treats_missing_or_nan_flag_as_published():
    idx = pd.date_range("2026-04-28 10:00", periods=2, freq="15min", tz="Europe/Stockholm")
    df = pd.DataFrame(
        {
            "end_time": idx + pd.Timedelta(minutes=15),
            "import_price_sek_kwh": [1.0, 1.0],
            "price_is_forecast": [np.nan, np.bool_(True)],
        },
        index=idx,
    )
    records = dataframe_to_json_response(
        df, now_override=datetime(2026, 4, 28, 9, 0), timezone_name="Europe/Stockholm"
    )
    assert [r["price_is_forecast"] for r in records] == [False, True]

    legacy = dataframe_to_json_response(
        df.drop(columns=["price_is_forecast"]),
        now_override=datetime(2026, 4, 28, 9, 0),
        timezone_name="Europe/Stockholm",
    )
    assert [r["price_is_forecast"] for r in legacy] == [False, False]


def test_legacy_price_slots_without_flag_default_to_published():
    slots = _price_slots()
    for s in slots:
        del s["price_is_forecast"]
    df = build_price_dataframe(slots, "Europe/Stockholm")
    assert not df["price_is_forecast"].any()


def test_flag_does_not_change_solver_input():
    """Economics unchanged: the Kepler input is identical with or without the flag."""
    flagged = prepare_df({"price_data": _price_slots(), "forecast_data": []}, "Europe/Stockholm")
    unflagged = flagged.drop(columns=["price_is_forecast"])
    a = planner_to_kepler_input(flagged.copy(), initial_soc_kwh=5.0)
    b = planner_to_kepler_input(unflagged.copy(), initial_soc_kwh=5.0)
    assert a == b
