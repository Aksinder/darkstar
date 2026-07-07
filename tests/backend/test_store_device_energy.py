"""Tests for per-device slot energy persistence (slot_device_energy table)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from backend.learning.models import Base
from backend.learning.store import LearningStore

TZ = pytz.timezone("Europe/Stockholm")


async def _make_store() -> LearningStore:
    store = LearningStore(":memory:", TZ)
    async with store.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return store


def _obs_record(slot: datetime, **over):
    rec = {
        "slot_start": slot,
        "slot_end": slot + timedelta(minutes=15),
        "pv_kwh": 0.0,
        "load_kwh": 0.5,
        "import_kwh": 1.0,
        "export_kwh": 0.0,
        "water_kwh": 0.0,
        "ev_charging_kwh": 0.0,
        "import_price_sek_kwh": 2.0,
        "export_price_sek_kwh": 0.8,
    }
    rec.update(over)
    return rec


@pytest.mark.asyncio
async def test_per_device_dicts_are_persisted():
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 7, 6, 12, 0))
        df = pd.DataFrame(
            [
                _obs_record(
                    slot,
                    water_kwh=0.9,
                    water_heater_energy={"main_tank": 0.6, "villavagn_tank": 0.3},
                    ev_charging_kwh=2.0,
                    ev_charger_energy={"easee": 2.0},
                )
            ]
        )
        await store.store_slot_observations(df)

        rows = await store.get_device_energy_rows_between("2026-07-06", "2026-07-07")
        by_dev = {r["device_id"]: r["kwh"] for r in rows}
        assert by_dev == {"main_tank": 0.6, "villavagn_tank": 0.3, "easee": 2.0}
        assert all(r["slot_start"] == slot.isoformat() for r in rows)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_zero_overwrites_instead_of_resurrecting_stale_value():
    """A legitimately-zero per-device slot must replace a stale non-zero value
    (deliberately unlike the aggregate columns' >0 backfill guard)."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 7, 6, 12, 0))
        await store.store_slot_observations(
            pd.DataFrame([_obs_record(slot, water_heater_energy={"main_tank": 0.5})])
        )
        await store.store_slot_observations(
            pd.DataFrame([_obs_record(slot, water_heater_energy={"main_tank": 0.0})])
        )
        rows = await store.get_device_energy_rows_between("2026-07-06", "2026-07-07")
        assert rows == [{"slot_start": slot.isoformat(), "device_id": "main_tank", "kwh": 0.0}]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_records_without_device_dicts_write_no_device_rows():
    """Backfill/legacy records (no per-device columns) must not crash or wipe rows."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 7, 6, 12, 0))
        await store.store_slot_observations(
            pd.DataFrame([_obs_record(slot, water_heater_energy={"main_tank": 0.5})])
        )
        # Re-store the same slot without the dict columns (e.g. backfill): the
        # device row stays — absent-from-dict means "not computed", not zero.
        await store.store_slot_observations(pd.DataFrame([_obs_record(slot)]))
        rows = await store.get_device_energy_rows_between("2026-07-06", "2026-07-07")
        assert rows == [{"slot_start": slot.isoformat(), "device_id": "main_tank", "kwh": 0.5}]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_observation_rows_include_water_and_ev_columns():
    """Regression for the savings v1 bias: the row reader must return the columns
    the whole-house baseline reconstruction needs."""
    store = await _make_store()
    try:
        slot = TZ.localize(datetime(2026, 7, 6, 12, 0))
        await store.store_slot_observations(
            pd.DataFrame([_obs_record(slot, water_kwh=1.5, ev_charging_kwh=7.0)])
        )
        rows = await store.get_observation_rows_between("2026-07-06", "2026-07-07")
        assert len(rows) == 1
        assert rows[0]["water_kwh"] == 1.5
        assert rows[0]["ev_charging_kwh"] == 7.0
    finally:
        await store.close()
