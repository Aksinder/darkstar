"""An overwritten forecast row must carry the time of the overwrite (upstream 0f7e0407).

store_forecasts upserts on (slot_start, forecast_version). created_at was only
set on INSERT, so a slot re-forecast every replan kept the timestamp of its very
first write and any "how old is this forecast" check read it as stale.
"""

from __future__ import annotations

import pytest
import pytz
from sqlalchemy import text

from backend.learning.models import Base
from backend.learning.store import LearningStore

TZ = pytz.timezone("Europe/Stockholm")
SLOT = "2026-10-11T12:00:00+02:00"


@pytest.mark.asyncio
async def test_overwrite_refreshes_created_at() -> None:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    row = {"slot_start": SLOT, "pv_forecast_kwh": 1.0, "load_forecast_kwh": 0.5}
    await store.store_forecasts([row], "aurora")
    async with store.AsyncSession() as s:
        await s.execute(text("update slot_forecasts set created_at = '2000-01-01 00:00:00'"))
        await s.commit()

    await store.store_forecasts([{**row, "pv_forecast_kwh": 2.0}], "aurora")

    async with store.AsyncSession() as s:
        n, pv, created = (
            await s.execute(
                text("select count(*), max(pv_forecast_kwh), max(created_at) from slot_forecasts")
            )
        ).one()
    await store.close()

    assert (n, pv) == (1, 2.0)
    assert not str(created).startswith("2000-"), created
