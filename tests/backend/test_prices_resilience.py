"""get_nordpool_data horizon + outage resilience, and price-override last-good memory.

- The D+1 forecast fallback applies whenever tomorrow's real prices are missing
  (was gated on hour < 13: a late publication left the evening without D+1).
- A failing D+1 fetch keeps today's prices (it used to return []).
- A failing whole fetch serves the last good series while it still covers now.
- An unreadable price-override entity uses its last good value, not the literal.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import pytz

from backend.core import ha_client, prices as P

TZ = pytz.timezone("Europe/Stockholm")


def local(y, m, d, h=0, mi=0):
    return TZ.localize(datetime(y, m, d, h, mi))


def _clock(now_local: datetime):
    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_local.astimezone(tz) if tz else now_local.replace(tzinfo=None)

    return _Now


def _quarters(day_start: datetime, value: float = 400.0) -> list[dict]:
    nxt = TZ.localize(
        datetime.combine((day_start + timedelta(hours=26)).date(), datetime.min.time())
    )
    out, cur = [], day_start.astimezone(pytz.UTC)
    while cur < nxt:
        out.append({"start": cur, "end": cur + timedelta(minutes=15), "value": value})
        cur += timedelta(minutes=15)
    return out


def _forecast_rows(day_start: datetime, p50: float = 0.25) -> list[dict]:
    return [
        {"slot_start": q["start"].astimezone(TZ).isoformat(), "spot_p50": p50}
        for q in _quarters(day_start)
    ]


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "timezone: Europe/Stockholm\nnordpool:\n  price_area: SE3\n  resolution_minutes: 15\n"
    )
    return str(path)


def _fetcher(by_day: dict):
    """Prices.fetch stand-in: returns/raises per requested local day."""
    calls: list = []

    def fetch(end_date=None, areas=None, resolution=None):
        calls.append(end_date)
        result = by_day.get(end_date)
        if isinstance(result, Exception):
            raise result
        return {"areas": {"SE3": {"values": result or []}}}

    return fetch, calls


async def _run(cfg, now, by_day, forecast=None, cache=None):
    fetch, calls = _fetcher(by_day)
    with (
        patch.object(P, "datetime", _clock(now)),
        patch.object(P.cache_sync, "get", return_value=cache),
        patch.object(P.cache_sync, "set"),
        patch.object(P.Prices, "fetch", side_effect=fetch),
        patch("ml.price_forecast.get_d1_price_forecast_fallback", return_value=forecast) as fb,
    ):
        slots = await P.get_nordpool_data(cfg)
    return slots, calls, fb


# --- Task 1: fallback no longer gated on hour < 13 -------------------------------


@pytest.mark.asyncio
async def test_late_publication_evening_still_gets_forecast_d1(cfg):
    today, tomorrow = local(2026, 11, 10), local(2026, 11, 11)
    slots, calls, fb = await _run(
        cfg,
        local(2026, 11, 10, 18),
        {today.date(): _quarters(today), tomorrow.date(): []},  # not published yet
        forecast=_forecast_rows(tomorrow),
    )
    fb.assert_called_once()
    assert len(calls) == 2
    real = [s for s in slots if not s["price_is_forecast"]]
    fc = [s for s in slots if s["price_is_forecast"]]
    assert len(real) == 96 and len(fc) == 96
    assert min(s["start_time"] for s in fc) == tomorrow


@pytest.mark.asyncio
async def test_published_tomorrow_skips_fallback(cfg):
    today, tomorrow = local(2026, 11, 10), local(2026, 11, 11)
    slots, _calls, fb = await _run(
        cfg,
        local(2026, 11, 10, 15),
        {today.date(): _quarters(today), tomorrow.date(): _quarters(tomorrow, 100.0)},
        forecast=_forecast_rows(tomorrow),
    )
    fb.assert_not_called()
    assert len(slots) == 192 and not any(s["price_is_forecast"] for s in slots)


@pytest.mark.asyncio
async def test_forecast_never_overrides_real_prices(cfg):
    """A forecast row overlapping published slots (hourly bucket straddling, or a
    stray row for today) is dropped: forecasts only extend past the last real slot."""
    today, tomorrow = local(2026, 11, 10), local(2026, 11, 11)
    forecast = _forecast_rows(today - timedelta(hours=0), 9.9)[-8:] + _forecast_rows(tomorrow)
    slots, _calls, _fb = await _run(
        cfg, local(2026, 11, 10, 9), {today.date(): _quarters(today)}, forecast=forecast
    )
    todays = [s for s in slots if s["start_time"] < tomorrow]
    assert len(todays) == 96 and not any(s["price_is_forecast"] for s in todays)
    assert all(s["spot_price_sek_kwh"] == pytest.approx(0.4) for s in todays)
    assert len([s for s in slots if s["price_is_forecast"]]) == 96


@pytest.mark.asyncio
async def test_dst_day_forecast_fills_100_quarters(cfg):
    today, tomorrow = local(2026, 10, 24), local(2026, 10, 25)
    slots, _calls, _fb = await _run(
        cfg,
        local(2026, 10, 24, 20),
        {today.date(): _quarters(today), tomorrow.date(): []},
        forecast=_forecast_rows(tomorrow),
    )
    fc = [s for s in slots if s["price_is_forecast"]]
    assert len(fc) == 100
    assert len({s["start_time"] for s in slots}) == len(slots)


# --- Task 2: outages ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_d1_fetch_failure_keeps_today_and_uses_forecast(cfg):
    today, tomorrow = local(2026, 11, 10), local(2026, 11, 11)
    slots, _calls, fb = await _run(
        cfg,
        local(2026, 11, 10, 14),
        {today.date(): _quarters(today), tomorrow.date(): RuntimeError("D+1 endpoint down")},
        forecast=_forecast_rows(tomorrow),
    )
    fb.assert_called_once()
    assert len([s for s in slots if not s["price_is_forecast"]]) == 96
    assert len([s for s in slots if s["price_is_forecast"]]) == 96


@pytest.mark.asyncio
async def test_failed_d1_fetch_is_not_retried_on_every_tick(cfg):
    """After a D+1 failure the (today + forecast) series is served from cache until the
    retry memo expires, instead of blocking every executor tick on the dead endpoint."""
    today, tomorrow = local(2026, 11, 10), local(2026, 11, 11)
    now = local(2026, 11, 10, 14)
    by_day = {today.date(): _quarters(today), tomorrow.date(): RuntimeError("down")}
    first, calls1, _ = await _run(cfg, now, by_day, forecast=_forecast_rows(tomorrow))
    second, calls2, _ = await _run(cfg, now, by_day, forecast=None, cache=first)
    assert len(calls1) == 2 and calls2 == []
    assert second is first


@pytest.mark.asyncio
async def test_whole_fetch_failure_serves_last_good_series(cfg):
    """Yesterday's fetch (today + tomorrow) is still valid at 00:30 when Nordpool is down."""
    d1, d2 = local(2026, 11, 10), local(2026, 11, 11)
    good, _c, _f = await _run(
        cfg, local(2026, 11, 10, 15), {d1.date(): _quarters(d1), d2.date(): _quarters(d2, 200.0)}
    )
    assert len(good) == 192

    stale, _c, _f = await _run(
        cfg, local(2026, 11, 11, 0, 30), {d2.date(): RuntimeError("Nordpool down")}
    )
    assert stale, "never [] while a valid cached series covers now"
    assert stale[0]["start_time"] == local(2026, 11, 11, 0, 30)
    assert stale[-1]["end_time"] == local(2026, 11, 12)
    assert all(s["spot_price_sek_kwh"] == pytest.approx(0.2) for s in stale)

    # Memo window: no new fetch, still the stale-but-valid series.
    again, calls, _f = await _run(cfg, local(2026, 11, 11, 0, 35), {})
    assert calls == [] and again == stale


@pytest.mark.asyncio
async def test_last_good_series_not_served_once_it_no_longer_covers_now(cfg):
    d1 = local(2026, 11, 10)
    await _run(cfg, local(2026, 11, 10, 9), {d1.date(): _quarters(d1)})
    out, _c, _f = await _run(
        cfg, local(2026, 11, 11, 1), {local(2026, 11, 11).date(): RuntimeError("x")}
    )
    assert out == []


# --- Task 3: pricing override last-good memory --------------------------------------


@pytest.mark.asyncio
async def test_unreadable_override_uses_last_good_value(monkeypatch, caplog):
    states = {
        "input_number.natavgift": {"state": "25", "attributes": {"unit_of_measurement": "öre/kWh"}}
    }

    async def fake_get(entity_id):
        return states.get(entity_id)

    monkeypatch.setattr(ha_client, "get_ha_entity_state", fake_get)
    cfg = {
        "pricing": {
            "grid_transfer_fee_entity": "input_number.natavgift",
            "grid_transfer_fee_sek": 0.9,
        }
    }

    first = await P._resolve_pricing_overrides(cfg)
    assert first["grid_transfer_fee_sek"] == pytest.approx(0.25)

    states["input_number.natavgift"] = {"state": "unavailable"}
    with caplog.at_level("WARNING", logger="darkstar.core.prices"):
        second = await P._resolve_pricing_overrides(cfg)
        third = await P._resolve_pricing_overrides(cfg)
    # Same value -> same cache key; NOT the 0.9 literal.
    assert second["grid_transfer_fee_sek"] == pytest.approx(0.25)
    assert third["grid_transfer_fee_sek"] == pytest.approx(0.25)
    assert sum("last good value" in r.message for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_override_read_exception_uses_last_good(monkeypatch):
    calls = {"n": 0}

    async def flaky(entity_id):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("HA restarting")
        return {"state": "0.10"}

    monkeypatch.setattr(ha_client, "get_ha_entity_state", flaky)
    cfg = {"pricing": {"export_premium_entity": "sensor.premium"}}
    assert (await P._resolve_pricing_overrides(cfg))["export_premium_sek_kwh"] == pytest.approx(
        0.10
    )
    assert (await P._resolve_pricing_overrides(cfg))["export_premium_sek_kwh"] == pytest.approx(
        0.10
    )


@pytest.mark.asyncio
async def test_override_never_read_falls_back_to_literal(monkeypatch):
    async def down(entity_id):
        raise ConnectionError("HA down")

    monkeypatch.setattr(ha_client, "get_ha_entity_state", down)
    cfg = {"pricing": {"export_premium_entity": "sensor.premium", "export_premium_sek_kwh": 0.07}}
    assert await P._resolve_pricing_overrides(cfg) is None
