"""Tests for the 7-day-average baseline forecast writer (backend/learning/baseline.py)."""

from datetime import datetime, timedelta

import pytest
import pytz
from sqlalchemy import select

from backend.learning.baseline import BASELINE_VERSION, store_baseline_forecasts
from backend.learning.models import Base, SlotForecast, SlotObservation
from backend.learning.store import LearningStore

TZ = pytz.timezone("Europe/Stockholm")


async def _make_store() -> LearningStore:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return store


async def _seed_noon_history(store: LearningStore, anchor: datetime, pv_values: list[float]):
    """One observation per day at 12:00 for the len(pv_values) days before anchor."""
    async with store.AsyncSession() as session:
        for i, pv in enumerate(pv_values, start=1):
            slot = (anchor - timedelta(days=i)).replace(hour=12, minute=0)
            session.add(
                SlotObservation(
                    slot_start=slot.isoformat(),
                    slot_end=(slot + timedelta(minutes=15)).isoformat(),
                    pv_kwh=pv,
                    load_kwh=1.0,
                )
            )
        await session.commit()


@pytest.mark.asyncio
async def test_baseline_is_mean_of_same_time_of_day():
    store = await _make_store()
    try:
        anchor = TZ.localize(datetime(2026, 7, 8, 12, 0))
        await _seed_noon_history(store, anchor, [3.0, 4.0, 5.0])  # mean 4.0

        written = await store_baseline_forecasts(store, [anchor.isoformat()])
        assert written == 1

        async with store.AsyncSession() as session:
            row = (
                (
                    await session.execute(
                        select(SlotForecast).where(
                            SlotForecast.forecast_version == BASELINE_VERSION
                        )
                    )
                )
                .scalars()
                .one()
            )
        assert row.pv_forecast_kwh == pytest.approx(4.0)
        assert row.load_forecast_kwh == pytest.approx(1.0)
        assert row.slot_start == anchor.isoformat()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_slots_without_history_are_skipped_not_zeroed():
    """A time-of-day with no observations must be skipped — fabricated zeros would
    hand the baseline free wins/losses in the eval."""
    store = await _make_store()
    try:
        anchor = TZ.localize(datetime(2026, 7, 8, 12, 0))
        await _seed_noon_history(store, anchor, [4.0])

        thirteen = anchor.replace(hour=13)
        written = await store_baseline_forecasts(
            store, [anchor.isoformat(), thirteen.isoformat()]
        )
        assert written == 1  # only the 12:00 slot has history

        async with store.AsyncSession() as session:
            rows = (
                (
                    await session.execute(
                        select(SlotForecast).where(
                            SlotForecast.forecast_version == BASELINE_VERSION
                        )
                    )
                )
                .scalars()
                .all()
            )
        assert [r.slot_start for r in rows] == [anchor.isoformat()]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_empty_and_garbage_inputs_are_safe():
    store = await _make_store()
    try:
        assert await store_baseline_forecasts(store, []) == 0
        assert await store_baseline_forecasts(store, ["not-a-timestamp"]) == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rerun_upserts_instead_of_duplicating():
    store = await _make_store()
    try:
        anchor = TZ.localize(datetime(2026, 7, 8, 12, 0))
        await _seed_noon_history(store, anchor, [4.0])
        await store_baseline_forecasts(store, [anchor.isoformat()])
        await store_baseline_forecasts(store, [anchor.isoformat()])

        async with store.AsyncSession() as session:
            rows = (
                (
                    await session.execute(
                        select(SlotForecast).where(
                            SlotForecast.forecast_version == BASELINE_VERSION
                        )
                    )
                )
                .scalars()
                .all()
            )
        assert len(rows) == 1
    finally:
        await store.close()
