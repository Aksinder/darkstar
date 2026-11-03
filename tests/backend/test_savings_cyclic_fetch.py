"""Savings sensors read whole-house consumption: cyclic loads are added back."""

import asyncio
from datetime import datetime, timedelta

import pytest
import pytz

from backend.learning.cycle_publisher_service import fetch_savings_rows
from backend.learning.savings import compute_savings

TZ = pytz.timezone("Europe/Stockholm")
NOW = TZ.localize(datetime(2026, 11, 3, 1, 0))
MIDNIGHT = TZ.localize(datetime(2026, 11, 3, 0, 0))


def _row(t: datetime, cyclic=None) -> dict:
    # Night, idle battery: the meter imported base 0.3 + pump 0.1.
    return {
        "slot_start": t.isoformat(),
        "import_kwh": 0.4,
        "export_kwh": 0.0,
        "pv_kwh": 0.0,
        "load_kwh": 0.3,
        "water_kwh": 0.0,
        "ev_charging_kwh": 0.0,
        "cyclic_kwh": cyclic,
        "import_price_sek_kwh": 2.0,
        "export_price_sek_kwh": 0.5,
        "quality_flags": None,
    }


class _Store:
    def __init__(self, rows, dev):
        self.rows, self.dev, self.dev_calls = rows, dev, 0

    async def get_observation_rows_between(self, start_iso, end_iso):
        return [dict(r) for r in self.rows if start_iso <= r["slot_start"] < end_iso]

    async def get_device_energy_rows_between(self, start_iso, end_iso):
        self.dev_calls += 1
        return self.dev


def test_legacy_and_new_rows_both_get_cyclic_back():
    legacy_t = MIDNIGHT - timedelta(hours=2)
    new_t = MIDNIGHT + timedelta(minutes=15)
    rows = [_row(legacy_t), _row(new_t, cyclic=0.1)]
    dev = [{"slot_start": legacy_t.isoformat(), "device_id": "poolpump", "kwh": 0.1}]
    store = _Store(rows, dev)
    cfg = {"cyclic_loads": [{"id": "poolpump", "enabled": True}]}

    today, d30 = asyncio.run(fetch_savings_rows(store, cfg, MIDNIGHT, NOW))

    assert [r["cyclic_kwh"] for r in d30] == pytest.approx([0.1, 0.1])
    assert [r["cyclic_kwh"] for r in today] == pytest.approx([0.1])
    # Idle battery: the baseline house imports exactly what the meter did.
    assert compute_savings(d30).savings_sek == pytest.approx(0.0)


def test_no_device_query_when_every_row_carries_the_column():
    store = _Store([_row(MIDNIGHT + timedelta(minutes=15), cyclic=0.1)], [])
    cfg = {"cyclic_loads": [{"id": "poolpump"}]}
    asyncio.run(fetch_savings_rows(store, cfg, MIDNIGHT, NOW))
    assert store.dev_calls == 0
