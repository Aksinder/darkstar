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
        known = await P.get_known_spot_by_slot(str(cfg))

    published = [s for s in slots if s["start_time"] < tomorrow]
    forecasted = [s for s in slots if s["start_time"] >= tomorrow]
    assert len(published) == 96 and len(forecasted) == 96
    assert not any(s["price_is_forecast"] for s in published)
    assert all(s["price_is_forecast"] for s in forecasted)
    # Published-price map (used for price-forecast lags) never includes forecasts.
    assert len(known) == 96
    assert max(known) < tomorrow


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


# --- forecast units / resolution -------------------------------------------------

_ADDER_CFG = {
    "timezone": "Europe/Stockholm",
    "pricing": {
        "export_premium_sek_kwh": 0.10,
        "export_grid_benefit_sek_kwh": 0.05,
        "export_fee_sek_kwh": 0.02,
        "vat_percent": 25.0,
        "grid_transfer_fee_sek": 0.25,
        "energy_tax_sek": 0.44,
    },
}


def _forecast_rows(p50: float = 0.80) -> list[dict]:
    tomorrow = TZ.localize(datetime(2026, 4, 29))
    return [
        {
            "slot_start": (tomorrow + timedelta(minutes=15 * i)).isoformat(),
            "spot_p50": p50 + i / 1000,
        }
        for i in range(96)
    ]


def test_forecast_fallback_does_not_double_count_export_adders():
    """The model predicts the EFFECTIVE export price (its training target). The
    fallback must reproduce it as the slot's export price, not add the adders again."""
    entries = P.forecast_fallback_entries(_forecast_rows(), _ADDER_CFG, TZ, 15)
    slots = P._process_nordpool_data(entries, _ADDER_CFG)
    first = slots[0]
    assert first["export_price_sek_kwh"] == pytest.approx(0.80)
    spot = 0.80 - (0.10 + 0.05 - 0.02)
    assert first["spot_price_sek_kwh"] == pytest.approx(spot)
    assert first["import_price_sek_kwh"] == pytest.approx((spot + 0.25 + 0.44) * 1.25)


def test_derive_consumer_prices_export_equals_model_output():
    from ml.price_forecast import derive_consumer_prices

    out = derive_consumer_prices(0.5, 0.8, 1.1, _ADDER_CFG)
    assert out["export_p10"] == pytest.approx(0.5)
    assert out["export_p50"] == pytest.approx(0.8)
    assert out["export_p90"] == pytest.approx(1.1)


def test_hourly_resolution_fallback_has_no_overlapping_slots():
    entries = P.forecast_fallback_entries(
        _forecast_rows(), {"timezone": "Europe/Stockholm"}, TZ, 60
    )
    assert len(entries) == 24
    for a, b in zip(entries, entries[1:], strict=False):
        assert a["end"] == b["start"]
    # Hour 0 = mean of its four quarters (0.800..0.803).
    assert entries[0]["value"] == pytest.approx(801.5)


class _Afternoon(datetime):
    """13:20 local: the D+1 auction is out, so a cached forecast is stale."""

    @classmethod
    def now(cls, tz=None):
        return TZ.localize(datetime(2026, 4, 28, 13, 20))


@pytest.mark.asyncio
async def test_cached_forecast_is_not_served_after_the_auction(tmp_path):
    """A cache filled before 13:00 with ML-forecast D+1 slots must be refetched after
    13:00; forecast slots do not count as published tomorrow prices."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "timezone: Europe/Stockholm\nnordpool:\n  price_area: SE3\n  resolution_minutes: 15\n"
    )
    today = TZ.localize(datetime(2026, 4, 28))
    cached = [
        {
            "start_time": today + timedelta(minutes=15 * i),
            "end_time": today + timedelta(minutes=15 * (i + 1)),
            "import_price_sek_kwh": 1.0,
            "export_price_sek_kwh": 0.5,
            "price_is_forecast": i >= 96,
        }
        for i in range(192)
    ]
    P._fetch_fail_until = 0.0
    with (
        patch.object(P, "datetime", _Afternoon),
        patch.object(P.cache_sync, "get", return_value=cached),
        patch.object(P.cache_sync, "set"),
        patch.object(P.Prices, "fetch", return_value=None) as fetch,
    ):
        await P.get_nordpool_data(str(cfg))
    assert fetch.called  # cache rejected -> fresh fetch for published prices


@pytest.mark.asyncio
async def test_no_forecast_fallback_without_spot_in_export(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "timezone: Europe/Stockholm\nnordpool:\n  price_area: SE3\n  resolution_minutes: 15\n"
        "pricing:\n  export_includes_spot: false\n"
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
    P._fetch_fail_until = 0.0
    with (
        patch.object(P, "datetime", _Morning),
        patch.object(P.cache_sync, "get", return_value=None),
        patch.object(P.cache_sync, "set"),
        patch.object(P.Prices, "fetch", return_value=raw_today),
        patch("ml.price_forecast.get_d1_price_forecast_fallback") as fallback,
    ):
        slots = await P.get_nordpool_data(str(cfg))
    fallback.assert_not_called()
    assert not any(s["price_is_forecast"] for s in slots)
