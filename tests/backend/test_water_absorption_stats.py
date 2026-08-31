"""get_water_absorption_stats: the measured feed behind the plan's absorption cap."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import pytz

from backend.core.ha_client import get_water_absorption_stats
from backend.learning.models import Base, SlotDeviceEnergy
from backend.learning.store import LearningStore

TZ = pytz.timezone("Europe/Stockholm")

# A FIXED wall clock, deliberately well past the 10:00 water-day boundary.
# get_water_absorption_stats reads rows in [window_start, now] and correctly drops
# future-timestamped ones, so seeding the current bucket against the REAL clock made
# this file fail for 20 minutes every day: between 10:00 and 10:20 local the rows at
# bucket + 5/20 min are still in the future. Pinning the clock the production module
# reads keeps every seeded timestamp unambiguously in the past at any time of day.
NOW = TZ.localize(datetime(2026, 7, 9, 15, 0, 0))


class _FrozenDatetime(datetime):
    """A real datetime in every respect except now(), which is pinned to NOW."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _frozen_clock():
    """Freeze the clock get_water_absorption_stats reads, not the tests' own."""
    with patch("backend.core.ha_client.datetime", _FrozenDatetime):
        yield


def _config(db_path: str) -> dict:
    return {
        "timezone": "Europe/Stockholm",
        "learning": {"sqlite_path": db_path},
        "water_heating": {"defer_up_to_hours": 10},
        "water_heaters": [
            {"id": "main_tank", "enabled": True, "power_kw": 3.4},
            {"id": "villavagn_tank", "enabled": True, "power_kw": 1.6},
        ],
    }


async def _seed(db_path: str, rows: list[tuple[datetime, str, float]]) -> None:
    store = LearningStore(db_path, TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with store.AsyncSession() as session:
        for slot, device, kwh in rows:
            session.add(
                SlotDeviceEnergy(
                    slot_start=slot.isoformat(),
                    device_id=device,
                    kwh=kwh,
                )
            )
        await session.commit()
    await store.close()


def _bucket_start(now: datetime) -> datetime:
    """Mirror _water_day_bucket_start for defer=10: boundary at 10:00 local."""
    d = now.date()
    if now.hour < 10:
        d = d - timedelta(days=1)
    return TZ.localize(datetime(d.year, d.month, d.day, 10, 0, 0))


@pytest.mark.asyncio
async def test_today_window_split_and_average(tmp_path):
    db = str(tmp_path / "stats.db")
    bucket = _bucket_start(NOW)

    rows: list[tuple[datetime, str, float]] = []
    # 7 complete buckets of history: 2.0 kWh/day for main.
    for day in range(1, 8):
        rows.append((bucket - timedelta(days=day) + timedelta(hours=2), "main_tank", 2.0))
    # Current bucket: 1.4 kWh absorbed so far, both rows before the frozen now.
    rows.append((bucket + timedelta(minutes=5), "main_tank", 0.9))
    rows.append((bucket + timedelta(minutes=20), "main_tank", 0.5))

    await _seed(db, rows)
    stats = await get_water_absorption_stats(_config(db))

    assert abs(stats["main_tank"]["absorbed_today_kwh"] - 1.4) < 1e-9
    assert abs(stats["main_tank"]["absorbed_daily_avg_kwh"] - 2.0) < 1e-9


@pytest.mark.asyncio
async def test_device_without_history_gets_none_average(tmp_path):
    """No window rows => None => the adapter applies NO cap (fail-open)."""
    db = str(tmp_path / "stats.db")
    bucket = _bucket_start(NOW)
    # villavagn has only a current-bucket row, no window history.
    await _seed(db, [(bucket + timedelta(minutes=5), "villavagn_tank", 0.4)])

    stats = await get_water_absorption_stats(_config(db))
    assert stats["villavagn_tank"]["absorbed_daily_avg_kwh"] is None
    assert abs(stats["villavagn_tank"]["absorbed_today_kwh"] - 0.4) < 1e-9
    assert stats["main_tank"]["absorbed_daily_avg_kwh"] is None
    assert stats["main_tank"]["absorbed_today_kwh"] == 0.0


@pytest.mark.asyncio
async def test_average_divides_by_covered_days_zero_rows_count(tmp_path):
    """A COVERED day where the tank absorbed nothing (recorder wrote 0.0 rows —
    thermostat satisfied) is a real zero and dilutes the average: 7.0 kWh across
    7 covered days averages to 1.0."""
    db = str(tmp_path / "stats.db")
    bucket = _bucket_start(NOW)
    rows: list[tuple[datetime, str, float]] = [
        (bucket - timedelta(days=3) + timedelta(hours=1), "main_tank", 7.0)
    ]
    for day in (1, 2, 4, 5, 6, 7):
        rows.append((bucket - timedelta(days=day) + timedelta(hours=1), "main_tank", 0.0))
    await _seed(db, rows)

    stats = await get_water_absorption_stats(_config(db))
    assert abs(stats["main_tank"]["absorbed_daily_avg_kwh"] - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_recorder_outage_days_do_not_bias_the_average_low(tmp_path):
    """An UNCOVERED day (no rows at all = recorder outage) is missing data, not a
    zero: 10.0 kWh over 5 covered days averages to 2.0, not 10/7."""
    db = str(tmp_path / "stats.db")
    bucket = _bucket_start(NOW)
    rows: list[tuple[datetime, str, float]] = []
    for day in (1, 2, 3, 4, 5):  # days 6-7: outage, no rows
        rows.append((bucket - timedelta(days=day) + timedelta(hours=1), "main_tank", 2.0))
    await _seed(db, rows)

    stats = await get_water_absorption_stats(_config(db))
    assert abs(stats["main_tank"]["absorbed_daily_avg_kwh"] - 2.0) < 1e-9


@pytest.mark.asyncio
async def test_bucket_date_is_reported(tmp_path):
    """The planner needs to know WHICH bucket the today-numbers belong to, so a
    solve crossing the 10:00 boundary never charges the new day with them."""
    db = str(tmp_path / "stats.db")
    bucket = _bucket_start(NOW)
    await _seed(db, [(bucket + timedelta(minutes=5), "main_tank", 0.4)])

    stats = await get_water_absorption_stats(_config(db))
    assert stats["main_tank"]["absorbed_bucket_date"] == bucket.date().isoformat()


@pytest.mark.asyncio
async def test_failure_returns_empty_dict(tmp_path):
    """Any store failure => {} => callers must run uncapped, never crash."""
    db = str(tmp_path / "empty.db")  # file will be created but has NO tables
    stats = await get_water_absorption_stats(_config(db))
    assert stats == {}
