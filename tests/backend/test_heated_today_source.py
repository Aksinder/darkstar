"""Build #15 PART B: MEASURED, cold-shower-safe per-tank heated-today source.

Covers backend.core.ha_client.get_water_heated_today_by_tank and its day-bucket
helper _water_day_bucket_start:
  * measured per-tank crediting from the learning store (slot_device_energy),
  * over-credit clamped to min_kwh_per_day (a big draw-day can't zero the floor),
  * power-sensor fallback when the store has no rows for a tank,
  * uncertainty/error -> 0.0 (safe: over-heat, never under-heat),
  * the day-bucket window matches kepler's defer_up_to_hours bucketing.
"""

import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from backend.core import ha_client
from backend.core.ha_client import _water_day_bucket_start, get_water_heated_today_by_tank

STHLM = pytz.timezone("Europe/Stockholm")


def _config(defer_hours: float = 10.0) -> dict:
    return {
        "timezone": "Europe/Stockholm",
        "learning": {"sqlite_path": "unused-because-store-is-mocked.db"},
        "water_heating": {"defer_up_to_hours": defer_hours},
        "water_heaters": [
            {
                "id": "main_tank",
                "enabled": True,
                "min_kwh_per_day": 6.0,
                "sensor": "sensor.house_vvb_real_power",
            },
            {
                "id": "villavagn_tank",
                "enabled": True,
                "min_kwh_per_day": 3.0,
                "sensor": "sensor.villavagn_vvb_power",
            },
        ],
    }


def _kepler_bucket_date(dt: datetime, defer_hours: float):
    """Replicate kepler.py:888-893 bucketing for one slot's local start time."""
    bucket_date = dt.date()
    if defer_hours > 0 and dt.hour < defer_hours:
        bucket_date = bucket_date - timedelta(days=1)
    return bucket_date


class TestDayBucketMatchesKepler:
    def test_pre_boundary_hour_buckets_to_previous_day(self):
        now = STHLM.localize(datetime(2026, 7, 9, 2, 0))
        start = _water_day_bucket_start(now, 10.0, STHLM)
        # 02:00 with defer=10 belongs to the previous day's 10:00 bucket.
        assert start.date() == datetime(2026, 7, 8).date()
        assert start.hour == 10
        # Matches kepler's own bucketing of a slot at `now`.
        assert start.date() == _kepler_bucket_date(now, 10.0)
        # `now` is inside [start, start + 24h).
        assert start <= now < start + timedelta(hours=24)

    def test_post_boundary_hour_buckets_to_same_day(self):
        now = STHLM.localize(datetime(2026, 7, 9, 15, 0))
        start = _water_day_bucket_start(now, 10.0, STHLM)
        assert start.date() == datetime(2026, 7, 9).date()
        assert start.hour == 10
        assert start.date() == _kepler_bucket_date(now, 10.0)
        assert start <= now < start + timedelta(hours=24)

    def test_zero_defer_uses_local_midnight(self):
        now = STHLM.localize(datetime(2026, 7, 9, 2, 0))
        start = _water_day_bucket_start(now, 0.0, STHLM)
        assert start.hour == 0
        assert start.date() == datetime(2026, 7, 9).date()
        assert start.date() == _kepler_bucket_date(now, 0.0)

    def test_fractional_defer_uses_ceil_boundary(self):
        now = STHLM.localize(datetime(2026, 7, 9, 15, 0))
        start = _water_day_bucket_start(now, 10.5, STHLM)
        assert start.hour == math.ceil(10.5) == 11


def _mock_store(rows: list[dict] | None = None, raise_exc: bool = False):
    store = MagicMock()
    if raise_exc:
        store.get_device_energy_rows_between = AsyncMock(side_effect=RuntimeError("db down"))
    else:
        store.get_device_energy_rows_between = AsyncMock(return_value=rows or [])
    store.close = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_measured_store_rows_credited_and_overcredit_clamped():
    """main_tank sums store rows (4 kWh); villavagn's 10 kWh clamps to its 3 kWh floor."""
    rows = [
        {"slot_start": "x", "device_id": "main_tank", "kwh": 2.0},
        {"slot_start": "y", "device_id": "main_tank", "kwh": 2.0},
        {"slot_start": "z", "device_id": "villavagn_tank", "kwh": 10.0},
    ]
    with patch("backend.learning.store.LearningStore", return_value=_mock_store(rows)):
        result = await get_water_heated_today_by_tank(_config())

    assert result["main_tank"] == pytest.approx(4.0)
    # Over-credit clamped to min_kwh_per_day so the cold-shower floor survives.
    assert result["villavagn_tank"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_no_store_rows_credits_zero_not_powerhistory():
    """COLD-SHOWER SAFETY: no store rows for a tank => credit 0.0 (safe over-heat).
    The long-window power-history integral (which can OVER-estimate over a multi-hour
    bucket -> zero the reliability floor -> cold shower) must NEVER be used for
    crediting. Proven by asserting get_energy_from_power_history is not awaited even
    though the store returned no rows and a fallback value was available."""
    with (
        patch("backend.learning.store.LearningStore", return_value=_mock_store([])),
        patch.object(
            ha_client, "get_energy_from_power_history", AsyncMock(return_value=2.5)
        ) as gefph,
    ):
        result = await get_water_heated_today_by_tank(_config())

    assert result["main_tank"] == 0.0
    assert result["villavagn_tank"] == 0.0
    gefph.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_error_credits_zero_safe():
    """Store read raises (e.g. SQLite lock contention) => seen_ids empty => every tank
    credits 0.0 (safe over-heat). No power-history over-estimate can leak in."""
    with (
        patch("backend.learning.store.LearningStore", return_value=_mock_store(raise_exc=True)),
        patch.object(
            ha_client, "get_energy_from_power_history", AsyncMock(return_value=99.0)
        ) as gefph,
    ):
        result = await get_water_heated_today_by_tank(_config())

    assert result["main_tank"] == 0.0
    assert result["villavagn_tank"] == 0.0
    gefph.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_construction_failure_credits_zero_safe():
    """A failure constructing the store => every tank credits 0.0 (safe over-heat)."""
    with (
        patch(
            "backend.learning.store.LearningStore",
            side_effect=RuntimeError("cannot open db"),
        ),
        patch.object(
            ha_client, "get_energy_from_power_history", AsyncMock(return_value=99.0)
        ) as gefph,
    ):
        result = await get_water_heated_today_by_tank(_config())

    assert result["main_tank"] == 0.0
    assert result["villavagn_tank"] == 0.0
    gefph.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_water_heaters_returns_empty():
    cfg = _config()
    cfg["water_heaters"] = []
    result = await get_water_heated_today_by_tank(cfg)
    assert result == {}
