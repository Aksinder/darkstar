"""Time-of-use grid transfer fee (pricing.grid_transfer_fee_schedule).

Real tariff under test: GEAB / Gotlands Elnät T4, 2026, 25 A, incl. VAT —
höglast 0.65 SEK/kWh Mon-Fri 06-22 in Jan-Mar + Nov-Dec except holidays, else 0.37.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import ClassVar

import pandas as pd
import pytest
import pytz

from backend.core import grid_tariff
from backend.core.prices import _process_nordpool_data, calculate_import_export_prices
from backend.core.time_windows import parse_window, swedish_public_holidays
from ml.price_forecast import derive_consumer_prices
from planner.solver.adapter import planner_to_kepler_input

TZ = pytz.timezone("Europe/Stockholm")

T4 = {
    "enabled": True,
    "default_sek": 0.37,
    "windows": [
        {
            "name": "hoglast",
            "fee_sek": 0.65,
            "months": [1, 2, 3, 11, 12],
            "weekdays": [0, 1, 2, 3, 4],
            "hours": "06-22",
            "exclude_holidays": True,
        }
    ],
}


def _pricing(schedule=T4, **extra):
    return {
        "vat_percent": 25.0,
        "fees_include_vat": True,
        "grid_transfer_fee_sek": 0.525,
        "energy_tax_sek": 0.535,
        "grid_transfer_fee_schedule": schedule,
        **extra,
    }


def _fee(local_naive: datetime, pricing=None) -> float:
    return grid_tariff.transfer_fee_for(
        TZ.localize(local_naive), pricing or _pricing(), 0.525, "Europe/Stockholm"
    )


@pytest.fixture(autouse=True)
def _fresh_caches():
    grid_tariff._compiled_cache.clear()
    grid_tariff._logged_once.clear()


class TestT4Windows:
    def test_november_weekday_morning_is_peak(self):
        assert _fee(datetime(2026, 11, 3, 7, 0)) == pytest.approx(0.65)  # Tuesday

    def test_peak_window_boundaries(self):
        assert _fee(datetime(2026, 11, 3, 5, 45)) == pytest.approx(0.37)
        assert _fee(datetime(2026, 11, 3, 6, 0)) == pytest.approx(0.65)
        assert _fee(datetime(2026, 11, 3, 21, 45)) == pytest.approx(0.65)
        assert _fee(datetime(2026, 11, 3, 22, 0)) == pytest.approx(0.37)

    def test_saturday_is_offpeak(self):
        assert _fee(datetime(2026, 11, 7, 12, 0)) == pytest.approx(0.37)

    def test_christmas_eve_weekday_is_offpeak(self):
        assert date(2026, 12, 24).weekday() == 3  # Thursday
        assert _fee(datetime(2026, 12, 24, 10, 0)) == pytest.approx(0.37)
        assert _fee(datetime(2026, 12, 23, 10, 0)) == pytest.approx(0.65)

    def test_epiphany_and_good_friday_offpeak(self):
        assert _fee(datetime(2026, 1, 6, 10, 0)) == pytest.approx(0.37)  # Trettondedag (Tue)
        assert _fee(datetime(2027, 3, 26, 10, 0)) == pytest.approx(0.37)  # Långfredag 2027

    def test_april_weekday_is_offpeak(self):
        assert _fee(datetime(2026, 4, 15, 10, 0)) == pytest.approx(0.37)

    def test_utc_input_converted_to_local(self):
        # 06:00 UTC on a November Tuesday = 07:00 CET -> peak; 05:30 UTC = 06:30 CET -> peak;
        # 04:30 UTC = 05:30 CET -> off-peak (a UTC comparison would get these wrong).
        p = _pricing()
        utc = pytz.UTC
        assert grid_tariff.transfer_fee_for(
            datetime(2026, 11, 3, 5, 30, tzinfo=utc), p, 0.525
        ) == pytest.approx(0.65)
        assert grid_tariff.transfer_fee_for(
            datetime(2026, 11, 3, 4, 30, tzinfo=utc), p, 0.525
        ) == pytest.approx(0.37)

    def test_naive_and_string_are_local_wall_time(self):
        p = _pricing()
        assert grid_tariff.transfer_fee_for(datetime(2026, 11, 3, 7), p, 0.525) == 0.65
        assert grid_tariff.transfer_fee_for("2026-11-03T07:00:00", p, 0.525) == 0.65
        assert grid_tariff.transfer_fee_for("2026-11-03T07:00:00+01:00", p, 0.525) == 0.65


class TestDST:
    # Spring-forward (2026-03-29) and fall-back (2026-10-25) are Sundays,
    # so use an all-week window to exercise the transition hours.
    ALL_WEEK: ClassVar[dict] = {
        "enabled": True,
        "default_sek": 0.10,
        "windows": [{"fee_sek": 0.90, "hours": "02-04"}],
    }

    def _walk(self, day: date) -> list[tuple[datetime, float]]:
        start = TZ.localize(datetime(day.year, day.month, day.day))
        end = TZ.localize(datetime.combine(day + timedelta(days=1), datetime.min.time()))
        out = []
        t = start.astimezone(pytz.UTC)
        while t < end.astimezone(pytz.UTC):
            out.append(
                (
                    t.astimezone(TZ),
                    grid_tariff.transfer_fee_for(t, _pricing(self.ALL_WEEK), 0.5),
                )
            )
            t += timedelta(minutes=15)
        return out

    def test_spring_forward(self):
        slots = self._walk(date(2026, 3, 29))
        assert len(slots) == 92  # 23-hour day
        peak = [s for s, f in slots if f == pytest.approx(0.90)]
        # 02:00-03:00 does not exist: only 03:00-03:45 local matches.
        assert [s.hour for s in peak] == [3, 3, 3, 3]

    def test_fall_back(self):
        slots = self._walk(date(2026, 10, 25))
        assert len(slots) == 100  # 25-hour day
        peak = [s for s, f in slots if f == pytest.approx(0.90)]
        # 02:xx happens twice (CEST then CET) + 03:xx once = 12 quarters.
        assert len(peak) == 12

    def test_weekday_peak_on_dst_monday(self):
        # Monday after spring-forward (March is a peak month): 06:00 CEST = 04:00 UTC is peak.
        assert grid_tariff.transfer_fee_for(
            datetime(2026, 3, 30, 4, 0, tzinfo=pytz.UTC), _pricing(), 0.525
        ) == pytest.approx(0.65)
        assert grid_tariff.transfer_fee_for(
            datetime(2026, 3, 30, 3, 45, tzinfo=pytz.UTC), _pricing(), 0.525
        ) == pytest.approx(0.37)


class TestDisabledAndInvalid:
    @pytest.mark.parametrize(
        "schedule",
        [None, {}, {**T4, "enabled": False}, {"enabled": "yes", "windows": []}],
    )
    def test_disabled_returns_flat(self, schedule):
        p = _pricing(schedule)
        assert _fee(datetime(2026, 11, 3, 7), p) == 0.525

    def test_disabled_prices_identical_to_today(self):
        base = {
            "vat_percent": 25.0,
            "grid_transfer_fee_sek": 0.30,
            "energy_tax_sek": 0.40,
            "export_premium_sek_kwh": 0.1,
        }
        disabled = {**base, "grid_transfer_fee_schedule": {**T4, "enabled": False}}
        slot = TZ.localize(datetime(2026, 11, 3, 7))
        for spot in (-50.0, 0.0, 800.0, 3500.0):
            legacy = calculate_import_export_prices(spot, {"pricing": base})
            assert calculate_import_export_prices(spot, {"pricing": base}, slot) == legacy
            assert calculate_import_export_prices(spot, {"pricing": disabled}, slot) == legacy

    @pytest.mark.parametrize(
        "bad_window",
        [
            {"fee_sek": 0.65, "months": [0, 13]},
            {"fee_sek": 0.65, "months": "jan"},
            {"fee_sek": 0.65, "weekdays": [7]},
            {"fee_sek": 0.65, "hours": "6 to 22"},
            {"fee_sek": 0.65, "hours": "06-25"},
            {"fee_sek": 0.65, "hours": "06-06"},
            {"fee_sek": "high"},
            {"fee_sek": 65},  # öre, not SEK
            {"fee_sek": 0.65, "exclude_holidays": "yes"},
        ],
    )
    def test_invalid_falls_back_to_flat_and_logs_error_once(self, bad_window, caplog):
        p = _pricing({"enabled": True, "default_sek": 0.37, "windows": [bad_window]})
        with caplog.at_level(logging.ERROR, logger="darkstar.core.grid_tariff"):
            assert _fee(datetime(2026, 11, 3, 7), p) == 0.525
            assert _fee(datetime(2026, 11, 3, 23), p) == 0.525
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1

    def test_invalid_default_falls_back(self):
        p = _pricing({**T4, "default_sek": -1})
        assert _fee(datetime(2026, 11, 3, 23), p) == 0.525

    def test_missing_default_uses_flat_outside_windows(self):
        p = _pricing({k: v for k, v in T4.items() if k != "default_sek"})
        assert _fee(datetime(2026, 11, 3, 23), p) == 0.525
        assert _fee(datetime(2026, 11, 3, 7), p) == 0.65

    def test_no_slot_time_returns_flat(self):
        assert grid_tariff.transfer_fee_for(None, _pricing(), 0.525) == 0.525


class TestPrecedenceAndOrdering:
    def test_schedule_overrides_entity_resolved_flat(self, caplog):
        # The entity override is resolved into grid_transfer_fee_sek upstream; when the
        # schedule is enabled that value is ignored (logged once).
        p = _pricing(grid_transfer_fee_entity="input_number.natavgift")
        with caplog.at_level(logging.INFO, logger="darkstar.core.grid_tariff"):
            assert grid_tariff.transfer_fee_for(datetime(2026, 11, 3, 7), p, 0.99) == 0.65
            assert grid_tariff.transfer_fee_for(datetime(2026, 11, 3, 23), p, 0.99) == 0.37
        assert sum("ignored" in r.message for r in caplog.records) == 1

    def test_first_matching_window_wins(self):
        sched = {
            "enabled": True,
            "default_sek": 0.1,
            "windows": [
                {"fee_sek": 0.8, "hours": "17-20"},
                {"fee_sek": 0.5, "hours": "06-22"},
            ],
        }
        p = _pricing(sched)
        assert _fee(datetime(2026, 6, 1, 18), p) == 0.8
        assert _fee(datetime(2026, 6, 1, 10), p) == 0.5
        assert _fee(datetime(2026, 6, 1, 23), p) == 0.1

    def test_wrapping_hours(self):
        p = _pricing(
            {"enabled": True, "default_sek": 0.1, "windows": [{"fee_sek": 0.2, "hours": "22-06"}]}
        )
        assert _fee(datetime(2026, 6, 1, 23), p) == 0.2
        assert _fee(datetime(2026, 6, 1, 5), p) == 0.2
        assert _fee(datetime(2026, 6, 1, 6), p) == 0.1

    def test_hours_forms_equivalent(self):
        for hours in ("06-22", [6, 22], {"start": 6, "end": 22}):
            w = parse_window({"hours": hours})
            assert (w.hour_start, w.hour_end) == (6, 22)


class TestHolidays:
    def test_known_dates_2026(self):
        h = swedish_public_holidays(2026)
        for d in [
            date(2026, 4, 3),  # Långfredag
            date(2026, 4, 6),  # Annandag påsk
            date(2026, 5, 14),  # Kristi himmelsfärd
            date(2026, 6, 19),  # Midsommarafton
            date(2026, 12, 31),
        ]:
            assert d in h
        assert date(2026, 12, 23) not in h


class TestPriceConsumers:
    def _entries(self, day: date, spot_mwh: float = 500.0):
        start = TZ.localize(datetime(day.year, day.month, day.day))
        return [
            {
                "start": (start + timedelta(minutes=15 * i)).astimezone(pytz.UTC),
                "end": (start + timedelta(minutes=15 * (i + 1))).astimezone(pytz.UTC),
                "value": spot_mwh,
            }
            for i in range(96)
        ]

    def _config(self, schedule=T4):
        return {"timezone": "Europe/Stockholm", "pricing": _pricing(schedule)}

    def test_processed_slots_and_planner_input_vary_when_enabled(self):
        slots = _process_nordpool_data(self._entries(date(2026, 11, 3)), self._config())
        by_hour = {s["start_time"].hour: s["import_price_sek_kwh"] for s in slots}
        # Same spot all day; only the transfer fee moves: 0.65 - 0.37 = 0.28.
        assert by_hour[7] - by_hour[3] == pytest.approx(0.28)
        assert by_hour[22] == pytest.approx(by_hour[3])
        assert by_hour[3] == pytest.approx(0.5 * 1.25 + 0.37 + 0.535)

        df = pd.DataFrame(slots).set_index("start_time")
        df["load_forecast_kwh"] = 0.5
        df["pv_forecast_kwh"] = 0.0
        kin = planner_to_kepler_input(df, initial_soc_kwh=5.0)
        prices = {round(s.import_price_sek_kwh, 6) for s in kin.slots}
        assert len(prices) == 2

    def test_processed_slots_flat_when_disabled(self):
        slots = _process_nordpool_data(
            self._entries(date(2026, 11, 3)), self._config({**T4, "enabled": False})
        )
        assert len({s["import_price_sek_kwh"] for s in slots}) == 1
        assert slots[0]["import_price_sek_kwh"] == pytest.approx(0.5 * 1.25 + 0.525 + 0.535)

    def test_ml_derive_consumer_prices_uses_slot_fee(self):
        cfg = self._config()
        peak = derive_consumer_prices(0.5, 0.5, 0.5, cfg, slot_start="2026-11-03T07:00:00")
        off = derive_consumer_prices(0.5, 0.5, 0.5, cfg, slot_start="2026-11-03T23:00:00")
        flat = derive_consumer_prices(0.5, 0.5, 0.5, cfg)
        assert peak["import_p50"] - off["import_p50"] == pytest.approx(0.28)
        assert flat["import_p50"] - off["import_p50"] == pytest.approx(0.525 - 0.37)
        assert peak["export_p50"] == off["export_p50"] == flat["export_p50"]

    @pytest.mark.asyncio
    async def test_ml_d1_fallback_prices_each_slot_at_its_local_tou_fee(self, monkeypatch):
        """The D+1 fallback path (get_d1_price_forecast_fallback -> derive_consumer_prices)
        uses the per-slot ToU fee, matched on LOCAL wall time even for aware-UTC rows:
        05:30Z is 06:30 local (peak window 06-22), 04:30Z is 05:30 local (off-peak)."""
        import ml.price_forecast as pf

        rows = [
            {"slot_start": "2026-11-03T05:30:00+00:00", "days_ahead": 1, "spot_p50": 0.5},
            {"slot_start": "2026-11-03T04:30:00+00:00", "days_ahead": 1, "spot_p50": 0.5},
        ]
        monkeypatch.setattr(pf, "_latest_forecasts_for_local_date", lambda *_a: rows)
        out = await pf.get_d1_price_forecast_fallback(self._config())
        assert out is not None
        assert out[0]["import_p50"] - out[1]["import_p50"] == pytest.approx(0.28)
