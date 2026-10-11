"""The price repair must reach past days, not only today (2026-10-11)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
import pytz
from sqlalchemy import select

from backend import recorder
from backend.learning.models import Base, SlotObservation
from backend.learning.store import LearningStore

TZ = pytz.timezone("Europe/Stockholm")


@pytest.mark.asyncio
async def test_backfill_prices_a_slot_from_an_earlier_day(monkeypatch: pytest.MonkeyPatch) -> None:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    now = datetime.now(TZ)
    old = TZ.localize(datetime.combine((now - timedelta(days=3)).date(), datetime.min.time())) + timedelta(
        hours=23, minutes=45
    )
    priced = old - timedelta(minutes=15)
    async with store.AsyncSession() as s:
        s.add(SlotObservation(slot_start=old.isoformat(), slot_end=(old + timedelta(minutes=15)).isoformat()))
        s.add(
            SlotObservation(
                slot_start=priced.isoformat(),
                slot_end=old.isoformat(),
                import_price_sek_kwh=9.99,
                export_price_sek_kwh=8.88,
            )
        )
        await s.commit()

    asked: list[Any] = []

    async def fake_for_date(day: Any, *a: Any, **k: Any) -> list[dict[str, Any]]:
        asked.append(day)
        return [
            {"start_time": old, "end_time": old + timedelta(minutes=15),
             "import_price_sek_kwh": 1.5, "export_price_sek_kwh": 0.4},
            {"start_time": priced, "end_time": old,
             "import_price_sek_kwh": 1.1, "export_price_sek_kwh": 0.3},
        ]

    async def fake_today(*a: Any, **k: Any) -> list[dict[str, Any]]:
        return []

    async def _noop() -> None:
        return None

    monkeypatch.setattr("backend.core.prices.get_nordpool_data_for_date", fake_for_date)
    monkeypatch.setattr("backend.core.prices.get_nordpool_data", fake_today)
    monkeypatch.setattr(recorder, "LearningStore", lambda *a, **k: store)
    monkeypatch.setattr(store, "close", _noop)

    await recorder.backfill_missing_prices()

    async with store.AsyncSession() as s:
        rows = {r.slot_start: r for r in (await s.execute(select(SlotObservation))).scalars().all()}
    assert asked == [old.date()], "only the day that has a gap is fetched"
    assert rows[old.isoformat()].import_price_sek_kwh == 1.5
    assert rows[priced.isoformat()].import_price_sek_kwh == 9.99, "an existing price is not rewritten"
