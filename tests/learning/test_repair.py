"""Observation repair from HA statistics (the 2026-08 recorder-corruption healer)."""

from datetime import UTC, datetime, timedelta

import pytest
import pytz

import backend.learning.repair as repair_mod
from backend.learning.repair import (
    _hourly_kwh,
    _ws_url,
    compute_slot_record,
    repair_observations,
)


def _epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


HOUR = datetime(2026, 7, 20, 10, 0, tzinfo=UTC)


def test_ws_url_derivation():
    assert _ws_url("http://supervisor/core") == "ws://supervisor/core/api/websocket"
    assert _ws_url("https://ha.example.se/") == "wss://ha.example.se/api/websocket"


def test_hourly_kwh_mean_and_change():
    stats = {
        # units={"power": "kW"} in the WS request => mean arrives in kW
        "sensor.solpaneler": [{"start": _epoch_ms(HOUR), "mean": 8.0}],
        "sensor.total_imported_energy": [{"start": _epoch_ms(HOUR), "change": 1.2}],
    }
    assert _hourly_kwh(stats, "sensor.solpaneler", "mean_kw") == {HOUR: 8.0}
    assert _hourly_kwh(stats, "sensor.total_imported_energy", "change") == {HOUR: 1.2}
    assert _hourly_kwh(stats, None, "mean_kw") == {}
    assert _hourly_kwh(stats, "sensor.absent", "change") == {}


def test_compute_slot_record_balance_and_quarter_split():
    """8 kW PV hour + 0.4 kWh import - 0.2 export + battery ±, minus water/EV."""
    hourly = {
        "pv": {HOUR: 8.0},
        "imp": {HOUR: 0.4},
        "exp": {HOUR: 0.2},
        "cha": {HOUR: 4.0},
        "dis": {HOUR: 0.0},
        "water": {HOUR: 0.8},
        "ev": {HOUR: 0.4},
    }
    rec = compute_slot_record(HOUR + timedelta(minutes=15), hourly, full=True)
    assert rec is not None
    assert rec["pv_kwh"] == pytest.approx(2.0)  # 8.0 / 4
    assert rec["import_kwh"] == pytest.approx(0.1)
    assert rec["batt_charge_kwh"] == pytest.approx(1.0)
    # load = 2.0 + 0.1 - 0.05 + 0 - 1.0 - 0.2 - 0.1 = 0.75
    assert rec["load_kwh"] == pytest.approx(0.75)
    assert "repaired" in rec["quality_flags"]


def test_compute_slot_record_clamps_negative_load_and_needs_data():
    hourly = {"pv": {HOUR: 0.4}, "imp": {}, "exp": {HOUR: 8.0}, "cha": {}, "dis": {}}
    rec = compute_slot_record(HOUR, hourly, full=True)
    assert rec is not None
    assert rec["load_kwh"] == 0.0  # balance negative -> clamped
    # No pv AND no import data at all -> nothing to write.
    assert compute_slot_record(HOUR, {"pv": {}, "imp": {}}, full=True) is None


class _FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.written = None

    async def get_observation_rows_between(self, start_iso, end_iso):
        return self.rows

    async def store_slot_observations(self, df):
        self.written = df


class _FakeEngine:
    def __init__(self, rows):
        self.timezone = pytz.timezone("Europe/Stockholm")
        self.store = _FakeStore(rows)


def _config():
    return {
        "input_sensors": {
            "pv_power": "sensor.solpaneler",
            "total_grid_import": "sensor.total_imported_energy",
            "total_grid_export": "sensor.total_exported_energy",
            "total_battery_charge": "sensor.total_battery_charge",
            "total_battery_discharge": "sensor.total_battery_discharge",
        },
        "water_heaters": [],
        "ev_chargers": [],
    }


@pytest.mark.asyncio
async def test_repair_targets_artifacts_only_and_respects_dry_run(monkeypatch):
    """Artifact rows (load=0) get repaired; healthy rows are left alone; dry_run writes nothing."""
    start = HOUR
    end = HOUR + timedelta(hours=1)
    tz = pytz.timezone("Europe/Stockholm")
    rows = [
        # slot :00 healthy (load 0.4) -> untouched
        {"slot_start": start.astimezone(tz).isoformat(), "load_kwh": 0.4, "pv_kwh": 2.0},
        # slot :15 artifact (load 0.0) -> repaired
        {
            "slot_start": (start + timedelta(minutes=15)).astimezone(tz).isoformat(),
            "load_kwh": 0.0,
            "pv_kwh": 0.0,
        },
        # slots :30/:45 missing entirely -> repaired
    ]
    engine = _FakeEngine(rows)
    monkeypatch.setattr("backend.learning.get_learning_engine", lambda: engine)

    async def fake_stats(entity_ids, s, e, **kw):
        return {
            "sensor.solpaneler": [{"start": _epoch_ms(HOUR), "mean": 4.0}],
            "sensor.total_imported_energy": [{"start": _epoch_ms(HOUR), "change": 0.8}],
        }

    monkeypatch.setattr(repair_mod, "fetch_statistics_during_period", fake_stats)

    dry = await repair_observations(_config(), start, end, dry_run=True)
    assert dry["slots_scanned"] == 4
    assert dry["artifact_rows"] == 1
    assert dry["missing_rows"] == 2
    assert dry["records_to_write"] == 3
    assert dry["repaired"] == 0
    assert engine.store.written is None

    live = await repair_observations(_config(), start, end, dry_run=False)
    assert live["repaired"] == 3
    assert engine.store.written is not None
    written = engine.store.written
    assert len(written) == 3
    assert (written["pv_kwh"] - 1.0).abs().max() < 1e-6  # 4 kW hour / 4


@pytest.mark.asyncio
async def test_repair_clamps_live_edge(monkeypatch):
    """The window never reaches into the recorder's live edge."""
    engine = _FakeEngine([])
    monkeypatch.setattr("backend.learning.get_learning_engine", lambda: engine)

    async def fake_stats(entity_ids, s, e, **kw):
        return {}

    monkeypatch.setattr(repair_mod, "fetch_statistics_during_period", fake_stats)

    now = datetime.now(UTC)
    result = await repair_observations(_config(), now - timedelta(minutes=30), now, dry_run=True)
    assert result.get("error") == "empty window after live-edge clamp"


# -- review blocker fixes (2026-08-03 FIX-FIRST verdict) ---------------------


def test_artifact_update_record_preserves_measured_fields():
    """full=False (artifact update): only pv+load carried; other energies 0.0 and
    batt None so the F35 upsert preserves measured values on the existing row."""
    hourly = {
        "pv": {HOUR: 8.0},
        "imp": {HOUR: 0.4},
        "exp": {HOUR: 0.2},
        "cha": {HOUR: 4.0},
        "dis": {HOUR: 1.0},
        "water": {HOUR: 0.8},
        "ev": {HOUR: 0.4},
    }
    rec = compute_slot_record(HOUR, hourly, full=False)
    assert rec is not None
    assert rec["pv_kwh"] == pytest.approx(2.0)
    assert rec["load_kwh"] > 0
    # Preservation shape: F35 ">0" fails for 0.0 energies; coalesce keeps batt.
    assert rec["import_kwh"] == 0.0
    assert rec["export_kwh"] == 0.0
    assert rec["water_kwh"] == 0.0
    assert rec["ev_charging_kwh"] == 0.0
    assert rec["batt_charge_kwh"] is None
    assert rec["batt_discharge_kwh"] is None


def test_insert_record_omits_series_without_statistics():
    """full=True: series with NO statistics stay 0.0/None — never fabricated."""
    hourly = {"pv": {HOUR: 8.0}, "imp": {HOUR: 0.4}}  # no exp/cha/dis/water/ev data
    rec = compute_slot_record(HOUR, hourly, full=True)
    assert rec is not None
    assert rec["import_kwh"] == pytest.approx(0.1)
    assert rec["export_kwh"] == 0.0
    assert rec["batt_charge_kwh"] is None
    assert rec["batt_discharge_kwh"] is None


@pytest.mark.asyncio
async def test_clamped_live_row_with_real_pv_is_not_touched(monkeypatch):
    """A row with load=0 but REAL pv (live clamp case) is NOT an artifact — the
    repair must leave it entirely alone (flat estimates never replace measured PV)."""
    start = HOUR
    end = HOUR + timedelta(minutes=15)
    tz = pytz.timezone("Europe/Stockholm")
    rows = [
        {"slot_start": start.astimezone(tz).isoformat(), "load_kwh": 0.0, "pv_kwh": 3.1},
    ]
    engine = _FakeEngine(rows)
    monkeypatch.setattr("backend.learning.get_learning_engine", lambda: engine)

    async def fake_stats(entity_ids, s, e, **kw):
        return {"sensor.solpaneler": [{"start": _epoch_ms(HOUR), "mean": 4.0}]}

    monkeypatch.setattr(repair_mod, "fetch_statistics_during_period", fake_stats)

    result = await repair_observations(_config(), start, end, dry_run=False)
    assert result["artifact_rows"] == 0
    assert result["repaired"] == 0
    assert engine.store.written is None
