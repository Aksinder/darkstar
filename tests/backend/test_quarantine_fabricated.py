"""Quarantine of historical price-mint observation rows (2026-08-08).

The two rules that must never regress:
  1. PAST rows only -- store_slot_observations PRESERVES an existing quality_flags when
     the incoming record carries "{}", so labelling a future minted row would brand real
     data as fabricated forever once the recorder fills that slot in.
  2. A real observation must never be labelled, however boring it looks (a genuine
     all-zero night slot still carries SoC and battery).
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz

from backend.learning.quarantine import QUARANTINE_FLAG, quarantine_fabricated

TZ = pytz.timezone("Europe/Stockholm")

_DDL = """
CREATE TABLE slot_observations (
    slot_start TEXT PRIMARY KEY,
    slot_end TEXT,
    import_kwh REAL DEFAULT 0.0,
    export_kwh REAL DEFAULT 0.0,
    pv_kwh REAL DEFAULT 0.0,
    load_kwh REAL DEFAULT 0.0,
    water_kwh REAL DEFAULT 0.0,
    ev_charging_kwh REAL DEFAULT 0.0,
    batt_charge_kwh REAL,
    batt_discharge_kwh REAL,
    soc_start_percent REAL,
    soc_end_percent REAL,
    import_price_sek_kwh REAL,
    export_price_sek_kwh REAL,
    quality_flags TEXT
)
"""


def _db(tmp_path) -> str:
    p = str(tmp_path / "q.db")
    with sqlite3.connect(p) as conn:
        conn.execute(_DDL)
    return p


def _minted(conn, slot: datetime):
    conn.execute(
        "INSERT INTO slot_observations (slot_start, slot_end, import_price_sek_kwh) VALUES (?,?,?)",
        (slot.isoformat(), (slot + timedelta(minutes=15)).isoformat(), 2.0),
    )


def _measured(conn, slot: datetime, pv=1.0, load=0.4, flags=None):
    conn.execute(
        "INSERT INTO slot_observations (slot_start, slot_end, pv_kwh, load_kwh, "
        "batt_charge_kwh, batt_discharge_kwh, soc_end_percent, quality_flags) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            slot.isoformat(),
            (slot + timedelta(minutes=15)).isoformat(),
            pv,
            load,
            0.0,
            0.0,
            55.0,
            flags,
        ),
    )


def _flags(db: str, slot: datetime):
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "SELECT quality_flags FROM slot_observations WHERE slot_start = ?",
            (slot.isoformat(),),
        )
        return (cur.fetchone() or [None])[0]


def test_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        for i in range(10):
            _minted(conn, past + timedelta(minutes=15 * i))

    res = quarantine_fabricated(db, TZ, dry_run=True)
    assert res["candidates"] == 10
    assert res["quarantined"] == 0
    assert _flags(db, past) is None


def test_past_minted_rows_are_labelled(tmp_path):
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        for i in range(10):
            _minted(conn, past + timedelta(minutes=15 * i))

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["quarantined"] == 10

    flags = json.loads(_flags(db, past))
    assert flags[QUARANTINE_FLAG] is True
    assert "quarantined_at" in flags


def test_future_minted_rows_are_left_alone(tmp_path):
    """THE trap: a labelled future row keeps its label after the recorder fills it in."""
    db = _db(tmp_path)
    future = datetime.now(TZ) + timedelta(days=1)
    with sqlite3.connect(db) as conn:
        for i in range(8):
            _minted(conn, future + timedelta(minutes=15 * i))

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["candidates"] == 0
    assert res["quarantined"] == 0
    assert res["future_minted_left_alone"] == 8
    assert _flags(db, future) is None, "a future minted row was labelled"


def test_the_live_edge_is_not_touched(tmp_path):
    """A slot the recorder may still be writing must be left alone."""
    db = _db(tmp_path)
    just_now = datetime.now(TZ) - timedelta(minutes=5)
    with sqlite3.connect(db) as conn:
        _minted(conn, just_now)

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["quarantined"] == 0
    assert _flags(db, just_now) is None


def test_real_observations_are_never_labelled(tmp_path):
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        # A genuine all-zero night slot: boring, but real -- it carries SoC and battery.
        _measured(conn, past, pv=0.0, load=0.0)
        _measured(conn, past + timedelta(minutes=15), pv=3.0, load=0.5)

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["candidates"] == 0
    assert res["quarantined"] == 0
    assert _flags(db, past) is None, "a genuine zero-production slot was quarantined"


def _repaired_zero_hour(conn, slot: datetime):
    """What repair.py writes when HA statistics confirm an hour was genuinely zero.

    All six energy columns zero, batt_* None, soc_end_percent never set -- i.e. it
    matches the price-mint fingerprint EXACTLY. Only its quality_flags tag distinguishes
    "statistics confirmed this hour was really zero" from "this row was fabricated".
    """
    conn.execute(
        "INSERT INTO slot_observations (slot_start, slot_end, import_price_sek_kwh, "
        "quality_flags) VALUES (?,?,?,?)",
        (
            slot.isoformat(),
            (slot + timedelta(minutes=15)).isoformat(),
            2.0,
            json.dumps({"repaired": "statistics_backfill"}),
        ),
    )


def test_repaired_rows_keep_their_provenance(tmp_path):
    """THE inversion this module must never cause.

    repair.py's reconstructed zero-hour rows match the fingerprint exactly. Relabelling
    one turns "HA statistics confirmed this hour was genuinely zero" into "this row was
    fabricated" -- the precise inversion the quarantine exists to prevent.

    Regression guard: the first version of this test used a row carrying SoC and
    battery, which can never match the fingerprint, so it passed no matter what the
    UPDATE did. It guarded nothing.
    """
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        _repaired_zero_hour(conn, past)

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["candidates"] == 0, "a repaired row was selected for quarantine"
    assert res["quarantined"] == 0
    assert json.loads(_flags(db, past)) == {"repaired": "statistics_backfill"}


def test_recorder_rescue_flags_keep_their_provenance(tmp_path):
    """Same rule for the recorder's own load_rescued/load_read_invalid tags."""
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO slot_observations (slot_start, slot_end, quality_flags) VALUES (?,?,?)",
            (
                past.isoformat(),
                (past + timedelta(minutes=15)).isoformat(),
                json.dumps({"load_rescued": True}),
            ),
        )

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["quarantined"] == 0
    assert json.loads(_flags(db, past)) == {"load_rescued": True}


def test_empty_flag_string_is_still_quarantinable(tmp_path):
    """ "{}" and "" mean 'no writer claimed this', same as NULL."""
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO slot_observations (slot_start, slot_end, import_price_sek_kwh, "
            "quality_flags) VALUES (?,?,?,?)",
            (past.isoformat(), (past + timedelta(minutes=15)).isoformat(), 2.0, "{}"),
        )

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["quarantined"] == 1


def test_stale_labels_are_cleared_when_a_row_gains_real_data(tmp_path):
    """bin/backfill_ha.py rewrites historical energy without touching quality_flags.

    A row we labelled that now holds real measurements would otherwise carry
    "price_mint" forever. Clearing it is the implicit un-quarantine.
    """
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        _minted(conn, past)

    assert quarantine_fabricated(db, TZ, dry_run=False)["quarantined"] == 1

    # Backfill fills in real energy, leaving the now-false label behind.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE slot_observations SET pv_kwh = 3.0, load_kwh = 0.5, "
            "batt_charge_kwh = 0.1, batt_discharge_kwh = 0.0 WHERE slot_start = ?",
            (past.isoformat(),),
        )

    res = quarantine_fabricated(db, TZ, dry_run=False)
    assert res["stale_labels_cleared"] == 1
    assert _flags(db, past) is None, "a row holding real data kept its price_mint label"


def test_live_edge_margin_exceeds_the_dst_shift():
    """slot_start is a LOCAL ISO string compared lexically.

    During the autumn fall-back the same wall clock occurs twice, so a row at
    02:15+01:00 (an hour in the FUTURE) sorts before a cutoff of 02:15+02:00. A margin
    wider than the 1-hour shift puts the cutoff's wall clock before the repeated hour,
    so nothing inside it can be selected at any offset.
    """
    from backend.learning.quarantine import _LIVE_EDGE_MINUTES

    assert _LIVE_EDGE_MINUTES > 60


def test_is_idempotent(tmp_path):
    db = _db(tmp_path)
    past = datetime.now(TZ) - timedelta(days=2)
    with sqlite3.connect(db) as conn:
        for i in range(5):
            _minted(conn, past + timedelta(minutes=15 * i))

    first = quarantine_fabricated(db, TZ, dry_run=False)
    assert first["quarantined"] == 5

    second = quarantine_fabricated(db, TZ, dry_run=False)
    assert second["candidates"] == 0
    assert second["quarantined"] == 0
    assert second["already_quarantined"] == 5


def test_missing_db_reports_error_and_creates_nothing(tmp_path):
    missing = str(tmp_path / "nope.db")
    res = quarantine_fabricated(missing, TZ, dry_run=True)
    assert res["error"]
    assert not Path(missing).exists(), "a wrong db_path silently created a new database"


@pytest.mark.parametrize("dry", [True, False])
def test_empty_db_is_a_noop(tmp_path, dry):
    res = quarantine_fabricated(_db(tmp_path), TZ, dry_run=dry)
    assert res["candidates"] == 0
    assert res["quarantined"] == 0
    assert res["error"] is None
