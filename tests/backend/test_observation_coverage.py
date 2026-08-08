"""Observation coverage + zero-fabrication tripwire (2026-08-08).

The distinction these tests defend: a slot that genuinely produced 0 kWh (night, snow,
inverter off) is REAL DATA and must count as covered; a slot nobody ever measured must
not. Before the fix those two were byte-identical on disk.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest
import pytz

from backend.learning.coverage import classify_coverage, observation_coverage

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
    export_price_sek_kwh REAL
)
"""


def _db(tmp_path) -> str:
    p = str(tmp_path / "cov.db")
    with sqlite3.connect(p) as conn:
        conn.execute(_DDL)
    return p


def _boundary() -> datetime:
    now = datetime.now(TZ)
    return now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)


def _measured(conn, slot: datetime, pv=1.0, load=0.4, price=2.0):
    """A row the recorder wrote: carries SoC and battery, so it is real."""
    conn.execute(
        "INSERT INTO slot_observations (slot_start, slot_end, pv_kwh, load_kwh, "
        "batt_charge_kwh, batt_discharge_kwh, soc_end_percent, import_price_sek_kwh) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            slot.isoformat(),
            (slot + timedelta(minutes=15)).isoformat(),
            pv,
            load,
            0.0,
            0.0,
            55.0,
            price,
        ),
    )


def _minted(conn, slot: datetime):
    """A row store_slot_prices fabricated: prices only, no measurement of any kind."""
    conn.execute(
        "INSERT INTO slot_observations (slot_start, slot_end, import_price_sek_kwh) VALUES (?,?,?)",
        (slot.isoformat(), (slot + timedelta(minutes=15)).isoformat(), 2.0),
    )


def test_full_coverage_is_clean(tmp_path):
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(96):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=1)
    assert cov["evaluable"] is True
    assert cov["expected_slots"] == 96
    assert cov["covered_slots"] == 96
    assert cov["coverage"] == 1.0
    assert classify_coverage(cov) == []


def test_genuine_zero_production_counts_as_covered(tmp_path):
    """Night / snow / inverter-off slots are REAL data, not gaps.

    This is the distinction the whole change turns on -- if these were treated as
    missing, the fix would throw away exactly the slots it was built to protect.
    """
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(96):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i), pv=0.0, load=0.0)

    cov = observation_coverage(db, TZ, days=1)
    assert cov["covered_slots"] == 96
    assert cov["coverage"] == 1.0
    assert cov["unmeasured_rows"] == 0
    assert classify_coverage(cov) == []


def test_minted_rows_do_not_count_as_coverage(tmp_path):
    """The original bug: a full grid of rows, none of them a measurement."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(96):
            _minted(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=1)
    assert cov["rows_present"] == 96
    assert cov["covered_slots"] == 0
    assert cov["unmeasured_rows"] == 96
    assert cov["coverage"] == 0.0

    codes = {c for _s, c, _m, _g in classify_coverage(cov)}
    assert "obs_coverage_low" in codes
    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_coverage_low"] == "critical"


def test_partial_coverage_warns_before_it_is_critical(tmp_path):
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        # 80 of 96 = 83.3% -> below the 0.88 warning band, above 0.75 critical.
        for i in range(80):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=1)
    assert 0.75 < cov["coverage"] < 0.88
    issues = classify_coverage(cov)
    assert [s for s, c, _m, _g in issues if c == "obs_coverage_low"] == ["warning"]


def test_healthy_live_coverage_does_not_trip(tmp_path):
    """94.3% is what a healthy box actually reports. It must not alarm."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(91):  # 91/96 = 94.8%
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=1)
    assert cov["coverage"] > 0.88
    assert classify_coverage(cov) == []


def test_future_minted_rows_are_the_tripwire(tmp_path):
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(96):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))
        for i in range(8):
            _minted(conn, end + timedelta(minutes=15 * (i + 1)))

    cov = observation_coverage(db, TZ, days=1)
    assert cov["future_minted_rows"] == 8

    # No fix marker yet -> legacy rows draining, warn only.
    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_future_minted_rows"] == "warning"

    # Marker still inside the grace window -> still a warning.
    cov["fix_applied_at"] = (datetime.now(TZ) - timedelta(hours=2)).isoformat()
    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_future_minted_rows"] == "warning"

    # Marker aged past the grace window -> the bug is back. Critical.
    cov["fix_applied_at"] = (datetime.now(TZ) - timedelta(hours=96)).isoformat()
    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_future_minted_rows"] == "critical"


def test_unpriced_real_rows_are_reported(tmp_path):
    """A recorded slot with no price silently drops out of savings -- make it visible.

    Before 2026-08-08 the price mint accidentally papered over this by pre-carrying the
    price into the row the recorder later wrote. Nothing does now.
    """
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(90):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))
        for i in range(90, 96):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i), price=None)

    cov = observation_coverage(db, TZ, days=1)
    assert cov["unpriced_rows"] == 6
    assert cov["coverage"] == 1.0, "unpriced rows are still REAL observations"

    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_unpriced_rows"] == "warning"


def test_a_single_unpriced_slot_does_not_alarm(tmp_path):
    """One failed price fetch is a blip, not a gap in the savings series."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(95):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))
        _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * 95), price=None)

    cov = observation_coverage(db, TZ, days=1)
    assert cov["unpriced_rows"] == 1
    assert classify_coverage(cov) == []


def test_minted_rows_are_not_counted_as_unpriced(tmp_path):
    """Fabricated rows carry a price and no measurement -- the opposite problem."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(96):
            _minted(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=1)
    assert cov["unpriced_rows"] == 0


def test_total_outage_is_critical_not_silent(tmp_path):
    """An EMPTY window with older history is a dead recorder -- the loudest case.

    Regression guard: the first cut returned [] whenever rows_present == 0, which meant
    the tripwire went completely silent during exactly the outage it exists to catch.
    """
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        # History exists, but all of it predates the window.
        for i in range(20):
            _measured(conn, end - timedelta(days=30) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=7)
    assert cov["rows_present"] == 0
    assert cov["has_history"] is True

    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_none_recorded"] == "critical"


def test_fresh_install_stays_quiet(tmp_path):
    """No history anywhere -> genuinely nothing to report."""
    cov = observation_coverage(_db(tmp_path), TZ, days=7)
    assert cov["has_history"] is False
    assert classify_coverage(cov) == []


def test_a_stalled_recorder_is_caught_by_recency_not_the_average(tmp_path):
    """A 7-day mean barely moves when the recorder dies; recency must catch it."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        # A full week of good data, then nothing for 8 hours.
        for i in range(7 * 96):
            slot = end - timedelta(days=7) + timedelta(minutes=15 * i)
            if slot < end - timedelta(hours=8):
                _measured(conn, slot)

    cov = observation_coverage(db, TZ, days=7)
    # The window average is still healthy -- that is the whole point.
    assert cov["coverage"] > 0.88
    assert cov["hours_since_last_observation"] >= 8.0

    sev = {c: s for s, c, _m, _g in classify_coverage(cov)}
    assert sev["obs_stale"] == "critical"


def test_minted_rows_do_not_make_a_dead_recorder_look_alive(tmp_path):
    """Recency must ignore fabricated rows or the tripwire defeats itself."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        for i in range(48):
            _measured(conn, end - timedelta(days=2) + timedelta(minutes=15 * i))
        # Fabricated rows right up to the boundary.
        for i in range(8):
            _minted(conn, end - timedelta(hours=1) + timedelta(minutes=7 * i))

    cov = observation_coverage(db, TZ, days=7)
    assert cov["hours_since_last_observation"] > 6.0, (
        "a minted row was mistaken for a real observation"
    )


def test_fresh_db_is_not_scored_against_slots_predating_it(tmp_path):
    """A new or restored DB must not report CRITICAL for its first six days."""
    db = _db(tmp_path)
    end = _boundary()
    with sqlite3.connect(db) as conn:
        # Perfect recording, but only for the last day of a 7-day window.
        for i in range(96):
            _measured(conn, end - timedelta(days=1) + timedelta(minutes=15 * i))

    cov = observation_coverage(db, TZ, days=7)
    assert cov["coverage"] == 1.0, "scored against slots that predate the database"
    assert [c for _s, c, _m, _g in classify_coverage(cov) if c == "obs_coverage_low"] == []


def test_unevaluable_is_critical_not_silent():
    """A check that cannot run must say so. Silence is the failure mode being abolished."""
    issues = classify_coverage({"evaluable": False, "error": "no such table", "rows_present": 0})
    assert [(s, c) for s, c, _m, _g in issues] == [("critical", "obs_coverage_unavailable")]


def test_missing_db_reports_unevaluable(tmp_path):
    cov = observation_coverage(str(tmp_path / "nope.db"), TZ, days=1)
    assert cov["evaluable"] is False
    assert cov["error"]


def test_empty_db_is_quiet(tmp_path):
    """A fresh install has nothing to say."""
    cov = observation_coverage(_db(tmp_path), TZ, days=1)
    assert cov["rows_present"] == 0
    assert classify_coverage(cov) == []


@pytest.mark.parametrize("days", [1, 7])
def test_expected_slots_uses_a_dst_safe_grid(tmp_path, days):
    cov = observation_coverage(_db(tmp_path), TZ, days=days)
    # Not hardcoded days*96: Europe/Stockholm has 92- and 100-slot days.
    assert cov["expected_slots"] in (days * 96, days * 96 - 4, days * 96 + 4)
