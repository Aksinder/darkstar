from datetime import datetime, timedelta

import pytest
import pytz

from backend.core.prices import _process_nordpool_data

local_tz = pytz.timezone("Europe/Stockholm")


def _make_entry(hour: int, value: float, base_date: datetime | None = None) -> dict:
    if base_date is None:
        base_date = datetime(2026, 4, 28)
    start = local_tz.localize(base_date.replace(hour=hour, minute=0, second=0, microsecond=0))
    end = start + timedelta(hours=1)
    return {"start": start, "end": end, "value": value}


def test_dedup_keeps_nordpool_over_fallback():
    """Duplicate start_time values: first occurrence (Nordpool) wins."""
    nordpool_entry = _make_entry(10, 500.0)
    fallback_entry = _make_entry(10, 300.0)

    all_entries = [nordpool_entry, fallback_entry, _make_entry(11, 600.0)]

    result = _process_nordpool_data(all_entries, {"timezone": "Europe/Stockholm"})

    assert len(result) == 2

    slot_10 = [s for s in result if s["start_time"].hour == 10]
    assert len(slot_10) == 1
    assert slot_10[0]["export_price_sek_kwh"] == pytest.approx(500.0 / 1000.0)


def test_spot_price_preserved_in_slots():
    """Raw spot survives into the slot dict — the EV servo's tiers are written in
    spot terms and must not have to invert the fee-and-VAT import formula."""
    result = _process_nordpool_data(
        [_make_entry(10, 500.0), _make_entry(11, -12.3)],
        {"timezone": "Europe/Stockholm"},
    )
    by_hour = {s["start_time"].hour: s for s in result}
    assert by_hour[10]["spot_price_sek_kwh"] == pytest.approx(0.5)
    # Negative spot is legitimate and must pass through signed.
    assert by_hour[11]["spot_price_sek_kwh"] == pytest.approx(-0.0123)
    # Spot excludes VAT/fees: import must be strictly above it for positive spot.
    assert by_hour[10]["import_price_sek_kwh"] > by_hour[10]["spot_price_sek_kwh"]


class TestFetchFailMemo:
    """A failed/empty Nordpool fetch is memoized for _FETCH_FAIL_MEMO_S so executor
    ticks don't block on a doomed ~10 s fetch inside the actuation path."""

    @pytest.mark.asyncio
    async def test_failure_short_circuits_next_call(self, tmp_path, monkeypatch):
        from unittest.mock import patch as mpatch

        from backend.core import prices as P

        cfg = tmp_path / "config.yaml"
        cfg.write_text("timezone: Europe/Stockholm\nnordpool:\n  price_area: SE3\n")
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("api down")

        with mpatch.object(P.Prices, "fetch", side_effect=boom):
            assert await P.get_nordpool_data(str(cfg)) == []
            assert calls["n"] == 1
            # Second call within the memo window: no new fetch attempt.
            assert await P.get_nordpool_data(str(cfg)) == []
            assert calls["n"] == 1
        # Memo expiry re-enables fetching.
        P._fetch_fail_until = 0.0
        with mpatch.object(P.Prices, "fetch", side_effect=boom):
            assert await P.get_nordpool_data(str(cfg)) == []
            assert calls["n"] == 2
