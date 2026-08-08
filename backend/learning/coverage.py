"""Observation coverage + zero-fabrication tripwire (2026-08-08).

Coverage is measured by ROW EXISTENCE on the slot grid, minus the price-mint
fingerprint. There is deliberately NO provenance column and NO soc/batt predicate:

- a provenance stamp written by ``store_slot_observations`` would be applied by
  ``BackfillEngine`` and ``repair.py`` too, certifying all-zero rows as measured;
- ``soc_end_percent IS NOT NULL AND batt_charge_kwh IS NOT NULL`` misclassifies every
  BackfillEngine row (its auto sensor map has no battery channel) and every repair.py
  row (repair.py never sets ``soc_end_percent``).

After the ``store_slot_prices`` fix the only writers that can CREATE a row are
``store_slot_observations``' callers -- all of which are real measurements or
reconstructions -- so row existence IS the coverage signal.

This module is REPORT-ONLY. Nothing here is ever used as a training or eval filter;
the training/eval paths are deliberately untouched by the 2026-08-08 change.

WHY THIS EXISTS: a recorder collapse used to be INVISIBLE. The price mint pre-created a
complete row for every future slot, so a month of missing observations read downstream
as a bad PV FORECAST rather than as missing data. Nothing alarmed.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from utils.time_utils import dst_safe_date_range

logger = logging.getLogger("darkstar.learning.coverage")

# A row that store_slot_prices minted: no evidence any writer ever touched it.
# This cannot misfire on a real recorder row because the recorder assigns
# batt_charge_kwh/batt_discharge_kwh UNCONDITIONALLY from
# calculate_energy_from_cumulative, which always returns a float (it falls back to
# power_kw * 0.25). Night / snow / inverter-off slots therefore still carry battery
# values and are correctly counted as covered -- exactly the real-zero-vs-missing
# distinction this whole change turns on.
# ⚠️ The `batt_* IS NULL` clauses are the load-bearing ones; do NOT drop them as
# "redundant" because of the soc_end_percent clause. The recorder's SoC skip is
# CONDITIONAL -- `if soc_percent is None and soc_entity` -- so on a site with no
# battery_soc sensor configured it happily writes rows with soc_end_percent NULL, and
# the battery clauses are then the only thing keeping real rows out of this predicate.
PRICE_MINT_PREDICATE = """
        soc_end_percent      IS NULL
    AND batt_charge_kwh      IS NULL
    AND batt_discharge_kwh   IS NULL
    AND COALESCE(pv_kwh, 0)          <= 0.0
    AND COALESCE(load_kwh, 0)        <= 0.0
    AND COALESCE(import_kwh, 0)      <= 0.0
    AND COALESCE(export_kwh, 0)      <= 0.0
    AND COALESCE(water_kwh, 0)       <= 0.0
    AND COALESCE(ev_charging_kwh, 0) <= 0.0
"""

# Calibrated against live data, not guessed: a healthy box currently reports ~94.3%
# (634 samples on a 672-slot 7-day grid). A 0.95 band would ship pre-tripped, which is
# the alarm-fatigue outcome that let the original bug hide for a month.
COVERAGE_WARNING = 0.88
COVERAGE_CRITICAL = 0.75

# Legacy minted rows age into the past within ~48h. Escalate only after that drains,
# so "legacy rows draining" is distinguishable from "the bug came back".
FIX_GRACE_HOURS = 72

# More than an hour of unpriced slots is a real gap in the savings series, not a blip
# around a single failed Nordpool fetch.
UNPRICED_WARNING = 4

# Recency thresholds. The recorder writes every 15 minutes, so 2h is already 8 missed
# slots; 6h cannot be explained by a restart.
STALE_WARNING_HOURS = 2.0
STALE_CRITICAL_HOURS = 6.0


def observation_coverage(
    db_path: str,
    tz: Any,
    days: int = 7,
    fix_applied_at: str | None = None,
) -> dict[str, Any]:
    """Coverage over the CLOSED window ending at the last completed 15-min boundary.

    Synchronous on purpose -- call via ``asyncio.to_thread``. Every query is bounded to
    the trailing window (or to ``slot_start >= now``) so the slot_start primary-key
    index applies and nothing does an unbounded scan.
    """
    now = datetime.now(tz)
    boundary = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    window_start = boundary - timedelta(days=days)

    result: dict[str, Any] = {
        "window_start": window_start.isoformat(),
        "window_end": boundary.isoformat(),
        "expected_slots": 0,
        "rows_present": 0,
        "covered_slots": 0,
        "unmeasured_rows": 0,
        "coverage": 0.0,
        "unpriced_rows": 0,
        "future_minted_rows": 0,
        "has_history": False,
        "first_observation_at": None,
        "last_observation_at": None,
        "hours_since_last_observation": None,
        "fix_applied_at": fix_applied_at,
        "evaluable": True,
        "error": None,
    }

    try:
        # DST-safe: Europe/Stockholm has 92- and 100-slot days. Hardcoding days*96
        # would report 0.958 on a perfect recorder every March.
        grid = dst_safe_date_range(window_start, boundary, "15min", tz, inclusive="left")
        result["expected_slots"] = len(grid)

        start_s = window_start.isoformat()
        end_s = boundary.isoformat()
        now_s = boundary.isoformat()

        # Read-only URI: a plain sqlite3.connect() CREATES the file when it is absent,
        # which would turn "wrong db_path" into a permanently empty-but-valid DB.
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0) as conn:
            cur = conn.execute(
                f"""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN {PRICE_MINT_PREDICATE} THEN 1 ELSE 0 END),
                    SUM(CASE WHEN import_price_sek_kwh IS NULL
                              AND NOT ({PRICE_MINT_PREDICATE}) THEN 1 ELSE 0 END)
                  FROM slot_observations
                 WHERE slot_start >= ? AND slot_start < ?
                """,
                (start_s, end_s),
            )
            rows_present, minted_in_window, unpriced = cur.fetchone()
            rows_present = int(rows_present or 0)
            minted_in_window = int(minted_in_window or 0)
            result["unpriced_rows"] = int(unpriced or 0)

            cur = conn.execute(
                f"""
                SELECT COUNT(*)
                  FROM slot_observations
                 WHERE slot_start >= ? AND ({PRICE_MINT_PREDICATE})
                """,
                (now_s,),
            )
            result["future_minted_rows"] = int((cur.fetchone() or [0])[0] or 0)

            # Distinguish a FRESH INSTALL (no history at all -> stay quiet) from a
            # TOTAL OUTAGE (history exists, window is empty -> the loudest case there
            # is). Without this, an empty window returns silence -- precisely the
            # failure mode this module exists to abolish.
            cur = conn.execute(
                "SELECT MIN(slot_start) FROM slot_observations WHERE slot_start < ?",
                (end_s,),
            )
            first_seen = (cur.fetchone() or [None])[0]
            result["has_history"] = first_seen is not None
            result["first_observation_at"] = first_seen

            # Recency: a window average is far too slow on its own. A recorder that
            # died an hour ago barely moves a 7-day mean, so a dead recorder would not
            # read critical for ~42h. MAX over real rows only -- minted rows must not
            # make a dead recorder look alive.
            cur = conn.execute(
                f"""
                SELECT MAX(slot_start)
                  FROM slot_observations
                 WHERE slot_start < ? AND NOT ({PRICE_MINT_PREDICATE})
                """,
                (end_s,),
            )
            last_seen = (cur.fetchone() or [None])[0]
            result["last_observation_at"] = last_seen
            if last_seen:
                try:
                    delta = boundary - datetime.fromisoformat(last_seen)
                    result["hours_since_last_observation"] = round(
                        delta.total_seconds() / 3600.0, 2
                    )
                except (TypeError, ValueError):
                    pass

        covered = max(0, rows_present - minted_in_window)
        result["rows_present"] = rows_present
        result["covered_slots"] = covered
        result["unmeasured_rows"] = minted_in_window

        # A DB that only started recording partway into the window (fresh install,
        # restored backup) must not be scored against slots that predate its own
        # existence -- that would report CRITICAL for ~6 days on every new install and
        # train the operator to ignore this check.
        expected = result["expected_slots"]
        if first_seen:
            try:
                first_dt = datetime.fromisoformat(first_seen)
                if first_dt > window_start:
                    expected = len(
                        dst_safe_date_range(first_dt, boundary, "15min", tz, inclusive="left")
                    )
                    result["expected_slots"] = expected
                    result["window_start"] = first_dt.isoformat()
            except (TypeError, ValueError):
                pass

        if expected > 0:
            result["coverage"] = round(min(1.0, covered / expected), 4)
    except Exception as e:
        logger.warning("observation_coverage failed: %s", e)
        result["evaluable"] = False
        result["error"] = str(e)

    return result


def classify_coverage(cov: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Map a coverage dict to (severity, code, message, guidance) tuples."""
    out: list[tuple[str, str, str, str]] = []

    # A check that cannot evaluate must NOT return silence. Silence is precisely the
    # failure mode this change exists to abolish.
    if not cov.get("evaluable", False):
        out.append(
            (
                "critical",
                "obs_coverage_unavailable",
                f"Observation coverage could not be evaluated: {cov.get('error')}",
                "The observation-integrity tripwire is blind. Check the learning DB "
                "path and permissions; do not trust forecast-accuracy metrics until "
                "this evaluates again.",
            )
        )
        return out

    # An EMPTY WINDOW is the loudest possible signal when the DB has history: the
    # recorder has written nothing at all for the whole window. Only a genuinely fresh
    # install (no history anywhere) is allowed to be quiet.
    if cov.get("rows_present", 0) == 0:
        if cov.get("has_history"):
            out.append(
                (
                    "critical",
                    "obs_none_recorded",
                    "No observations were recorded in the entire coverage window, but "
                    "the database has older history.",
                    "The recorder is completely dead, or the DB path changed under it. "
                    "Every forecast-accuracy metric is meaningless right now -- missing "
                    "slots are not zero production. Check the recorder service and the "
                    "HA connection before drawing any conclusion from /api/forecast/eval.",
                )
            )
        return out

    # Recency beats the window average for detecting a recorder that just died.
    stale_h = cov.get("hours_since_last_observation")
    if stale_h is not None:
        if stale_h >= STALE_CRITICAL_HOURS:
            out.append(
                (
                    "critical",
                    "obs_stale",
                    f"No observation recorded for {stale_h:.1f} hours "
                    f"(last: {cov.get('last_observation_at')}).",
                    "The recorder has stopped. Slots recorded from now on will be "
                    "missing, and anything scoring forecasts against them will read "
                    "as a forecast regression rather than as missing data.",
                )
            )
        elif stale_h >= STALE_WARNING_HOURS:
            out.append(
                (
                    "warning",
                    "obs_stale",
                    f"No observation recorded for {stale_h:.1f} hours "
                    f"(last: {cov.get('last_observation_at')}).",
                    "Expected cadence is one observation per 15 minutes. A short gap "
                    "around a restart is normal; a growing one is not.",
                )
            )

    future_minted = cov.get("future_minted_rows", 0)
    if future_minted > 0:
        fix_at = cov.get("fix_applied_at")
        aged_out = False
        if fix_at:
            try:
                applied = datetime.fromisoformat(fix_at)
                ref = datetime.now(applied.tzinfo) if applied.tzinfo else datetime.now()
                aged_out = (ref - applied) > timedelta(hours=FIX_GRACE_HOURS)
            except (TypeError, ValueError):
                aged_out = False

        out.append(
            (
                "critical" if aged_out else "warning",
                "obs_future_minted_rows",
                f"{future_minted} future-dated observation rows carry no measurement "
                f"(zero energy, no SoC, no battery).",
                "These are price-mint artefacts. Rows predating 2026-08-08 are "
                "expected and drain within ~48h. If this persists past the grace "
                "window, a writer is fabricating observation rows again -- check "
                "LearningStore.store_slot_prices and bin/backfill_vattenfall.py for a "
                "regression to INSERT/upsert.",
            )
        )

    # A real observation with no price silently drops out of savings and arbitrage
    # (savings.py skips rows with import_price None), which then feeds AuroraReflex's
    # battery_cycle_cost tuning. Before 2026-08-08 the price mint accidentally papered
    # over this by pre-carrying the price; nothing does now, so it must be visible.
    unpriced = cov.get("unpriced_rows", 0)
    if unpriced > UNPRICED_WARNING:
        out.append(
            (
                "warning",
                "obs_unpriced_rows",
                f"{unpriced} recorded slots have no import price.",
                "Those slots are excluded from savings and arbitrage totals, which "
                "feed Reflex's battery-cost tuning. The periodic backfill_missing_prices "
                "repair should clear them; if the count keeps growing, the price fetch "
                "is failing persistently.",
            )
        )

    coverage = cov.get("coverage", 0.0)
    expected = cov.get("expected_slots", 0)
    if expected > 0:
        if coverage < COVERAGE_CRITICAL:
            out.append(
                (
                    "critical",
                    "obs_coverage_low",
                    f"Only {coverage:.1%} of the last "
                    f"{cov.get('window_start', '')[:10]}..{cov.get('window_end', '')[:10]} "
                    f"slot grid has a real observation "
                    f"({cov.get('covered_slots')}/{expected}).",
                    "Forecast-accuracy metrics are NOT trustworthy at this coverage: "
                    "missing slots are scored against absent actuals. Check the "
                    "recorder logs for skipped observations and the HA sensors it "
                    "depends on before drawing any conclusion about model quality.",
                )
            )
        elif coverage < COVERAGE_WARNING:
            out.append(
                (
                    "warning",
                    "obs_coverage_low",
                    f"Observation coverage is {coverage:.1%} "
                    f"({cov.get('covered_slots')}/{expected} slots).",
                    "Some slots were never recorded. Occasional gaps around restarts "
                    "are normal; a sustained decline means the recorder is dropping "
                    "observations and forecast metrics will drift accordingly.",
                )
            )

    return out
