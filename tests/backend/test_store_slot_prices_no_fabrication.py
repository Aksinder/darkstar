"""store_slot_prices must never CREATE an observation row (2026-08-08).

Regression guard for the zero-fabrication bug: store_slot_prices used to upsert price
rows into slot_observations, and because models.py declares default=0.0 +
server_default on the six energy columns, every future slot was pre-minted as a
complete, provenance-free ZERO observation. "Not measured" and "produced 0.0 kWh"
became identical on disk, and a month of missing observations masqueraded as a bad PV
forecast.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz
from sqlalchemy import select

from backend.learning.models import Base, SlotObservation
from backend.learning.store import OBS_FIX_APPLIED_KEY, LearningStore

TZ = pytz.timezone("Europe/Stockholm")


async def _make_store() -> LearningStore:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return store


async def _all_rows(store: LearningStore) -> list[SlotObservation]:
    async with store.AsyncSession() as session:
        return list((await session.execute(select(SlotObservation))).scalars().all())


def _price_row(slot: datetime) -> dict:
    return {
        "slot_start": slot,
        "slot_end": slot + timedelta(minutes=15),
        "import_price_sek_kwh": 2.0,
        "export_price_sek_kwh": 0.8,
    }


@pytest.mark.asyncio
async def test_prices_for_unobserved_slot_create_no_row():
    """THE regression: a future slot with no observation must stay absent."""
    store = await _make_store()
    try:
        future = TZ.localize(datetime(2026, 8, 9, 12, 0))
        await store.store_slot_prices([_price_row(future)])

        assert await _all_rows(store) == [], (
            "store_slot_prices fabricated an observation row for an unobserved slot"
        )
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_prices_land_on_an_existing_observation():
    """The legitimate case must keep working: prices attach to a real observation."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 8, 7, 12, 0))
        await store.store_slot_observations(
            pd.DataFrame(
                [
                    {
                        "slot_start": slot,
                        "slot_end": slot + timedelta(minutes=15),
                        "pv_kwh": 3.6,
                        "load_kwh": 0.5,
                        "import_kwh": 0.0,
                        "export_kwh": 3.1,
                        "water_kwh": 0.0,
                        "ev_charging_kwh": 0.0,
                    }
                ]
            )
        )

        await store.store_slot_prices([_price_row(slot)])

        rows = await _all_rows(store)
        assert len(rows) == 1
        assert rows[0].import_price_sek_kwh == 2.0
        assert rows[0].export_price_sek_kwh == 0.8
        # Energy must be untouched -- UPDATE writes exactly its SET list.
        assert rows[0].pv_kwh == 3.6
        assert rows[0].load_kwh == 0.5
        assert rows[0].export_kwh == 3.1
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_incoming_null_price_never_clobbers_existing():
    """Preserve the old coalesce semantics: an incoming NULL must not overwrite."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 8, 7, 13, 0))
        await store.store_slot_observations(
            pd.DataFrame(
                [
                    {
                        "slot_start": slot,
                        "slot_end": slot + timedelta(minutes=15),
                        "pv_kwh": 1.0,
                        "load_kwh": 0.4,
                        "import_kwh": 0.0,
                        "export_kwh": 0.6,
                        "water_kwh": 0.0,
                        "ev_charging_kwh": 0.0,
                        "import_price_sek_kwh": 1.5,
                        "export_price_sek_kwh": 0.5,
                    }
                ]
            )
        )

        await store.store_slot_prices(
            [
                {
                    "slot_start": slot,
                    "slot_end": slot + timedelta(minutes=15),
                    "import_price_sek_kwh": None,
                    "export_price_sek_kwh": 0.9,
                }
            ]
        )

        rows = await _all_rows(store)
        assert rows[0].import_price_sek_kwh == 1.5, "incoming NULL clobbered a real price"
        assert rows[0].export_price_sek_kwh == 0.9

    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_fix_marker_is_self_arming():
    """The coverage tripwire's pre/post boundary stamps itself on first price write."""
    store = await _make_store()
    try:
        assert await store.get_system_state(OBS_FIX_APPLIED_KEY) is None

        slot = TZ.localize(datetime(2026, 8, 9, 14, 0))
        await store.store_slot_prices([_price_row(slot)])

        stamped = await store.get_system_state(OBS_FIX_APPLIED_KEY)
        assert stamped is not None
        first = stamped

        await store.store_slot_prices([_price_row(slot + timedelta(minutes=15))])
        assert await store.get_system_state(OBS_FIX_APPLIED_KEY) == first, (
            "marker must record the FIRST application, not the latest write"
        )
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_marker_not_armed_by_a_no_op_call():
    """A call where every row is malformed offers nothing and must not arm the marker."""
    store = await _make_store()
    try:
        await store.store_slot_prices([{"slot_start": None}])
        assert await store.get_system_state(OBS_FIX_APPLIED_KEY) is None
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_nan_price_does_not_clobber_an_existing_price():
    """NaN is not None but SQLite stores it as NULL -- it must be treated as absent."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 8, 7, 14, 0))
        await store.store_slot_observations(
            pd.DataFrame(
                [
                    {
                        "slot_start": slot,
                        "slot_end": slot + timedelta(minutes=15),
                        "pv_kwh": 1.0,
                        "load_kwh": 0.4,
                        "import_kwh": 0.0,
                        "export_kwh": 0.6,
                        "water_kwh": 0.0,
                        "ev_charging_kwh": 0.0,
                        "import_price_sek_kwh": 1.5,
                        "export_price_sek_kwh": 0.5,
                    }
                ]
            )
        )

        await store.store_slot_prices(
            [
                {
                    "slot_start": slot,
                    "slot_end": slot + timedelta(minutes=15),
                    "import_price_sek_kwh": float("nan"),
                    "export_price_sek_kwh": 0.9,
                }
            ]
        )

        rows = await _all_rows(store)
        assert rows[0].import_price_sek_kwh == 1.5, "NaN clobbered a real price"
        assert rows[0].export_price_sek_kwh == 0.9
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_zero_price_is_written_not_treated_as_absent():
    """0.0 SEK/kWh is a real Nordpool price and must not be swallowed by truthiness."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 8, 7, 15, 0))
        await store.store_slot_observations(
            pd.DataFrame(
                [
                    {
                        "slot_start": slot,
                        "slot_end": slot + timedelta(minutes=15),
                        "pv_kwh": 1.0,
                        "load_kwh": 0.4,
                        "import_kwh": 0.0,
                        "export_kwh": 0.6,
                        "water_kwh": 0.0,
                        "ev_charging_kwh": 0.0,
                        "import_price_sek_kwh": 1.5,
                        "export_price_sek_kwh": 0.5,
                    }
                ]
            )
        )

        await store.store_slot_prices(
            [
                {
                    "slot_start": slot,
                    "slot_end": slot + timedelta(minutes=15),
                    "import_price_sek_kwh": 0.0,
                    "export_price_sek_kwh": 0.0,
                }
            ]
        )

        rows = await _all_rows(store)
        assert rows[0].import_price_sek_kwh == 0.0
        assert rows[0].export_price_sek_kwh == 0.0
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_unplaced_past_slot_warns(caplog):
    """A PAST slot with no row is silent-degradation, not the expected future case."""
    import logging

    store = await _make_store()
    try:
        past = TZ.localize(datetime(2020, 1, 1, 12, 0))
        with caplog.at_level(logging.WARNING):
            await store.store_slot_prices([_price_row(past)])
        assert "PAST slot" in caplog.text

        caplog.clear()
        future = TZ.localize(datetime(2099, 1, 1, 12, 0))
        with caplog.at_level(logging.WARNING):
            await store.store_slot_prices([_price_row(future)])
        assert "PAST slot" not in caplog.text, "a future slot must not warn"
    finally:
        await store.async_engine.dispose()


@pytest.mark.asyncio
async def test_empty_input_is_a_noop():
    store = await _make_store()
    try:
        await store.store_slot_prices([])
        assert await _all_rows(store) == []
        assert await store.get_system_state(OBS_FIX_APPLIED_KEY) is None
    finally:
        await store.async_engine.dispose()
