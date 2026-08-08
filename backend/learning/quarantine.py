"""Quarantine historical price-mint rows in slot_observations (2026-08-08).

Until 2026-08-08 store_slot_prices upserted price rows into slot_observations, and the
six energy columns carry default=0.0 + server_default, so every future slot was
pre-minted as a complete, provenance-free ZERO observation. Those rows are still on
disk. This module LABELS them so "never measured" stops being indistinguishable from
"measured 0.0 kWh".

It MARKS, it does not DELETE. Deleting would discard the one piece of real data those
rows carry -- the price -- which backend/core/price_outlook.py reads as a 14-day
export average and ml/price_features._get_price_lags reads as backward lag features.
Marking is reversible; deletion is not.

TWO SAFETY RULES, both load-bearing:

1. PAST ROWS ONLY. store_slot_observations preserves an existing quality_flags when the
   incoming record carries "{}" (see its on_conflict set_). So stamping a FUTURE minted
   row would leave the price_mint label on it permanently once the recorder writes the
   real observation for that slot -- mislabelling real data as fabricated forever.
   Future minted rows need no action: they either become real or age out.

2. THE FINGERPRINT, NEVER created_at. store_slot_observations never writes created_at,
   so it stays frozen at price-mint time for essentially EVERY row in the table; a
   `created_at < slot_start` discriminator flags the whole database. The only sound
   marker is the absence of any measurement: no SoC, no battery, and all six energy
   columns zero.

3. ONLY ROWS WITH NO PROVENANCE. The fingerprint alone is NOT sufficient. repair.py
   writes reconstructed rows with batt_charge_kwh=None, batt_discharge_kwh=None and no
   soc_end_percent, tagged quality_flags={"repaired": "statistics_backfill"} -- and when
   HA statistics confirm an hour was genuinely zero, such a row matches the fingerprint
   exactly. Overwriting its tag would invert "statistics confirmed this hour was really
   zero" into "this row was fabricated", the precise inversion this module exists to
   prevent. A real price-mint row always has quality_flags NULL (store_slot_prices never
   wrote flags), so requiring an empty flag field costs nothing and protects every
   writer that has claimed a row.
"""

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from typing import Any

from backend.learning.coverage import PRICE_MINT_PREDICATE

logger = logging.getLogger("darkstar.learning.quarantine")

QUARANTINE_FLAG = "price_mint"

# Never touch a row this close to now. 90 minutes, not 30, because slot_start is stored
# as a LOCAL ISO string with offset and compared lexically. During the autumn DST
# fall-back the same wall clock occurs twice, so a row written at 02:15+01:00 (an hour
# in the FUTURE) sorts before a cutoff of 02:15+02:00 and would be selected -- defeating
# rule 1 by exactly the mechanism it warns about. A margin wider than the 1-hour shift
# puts the cutoff's wall clock before the repeated hour entirely, so no row inside it
# can be selected at any offset.
_LIVE_EDGE_MINUTES = 90

# A price-mint row carries no provenance at all. Anything else has been claimed by a
# writer (recorder rescue flags, repair.py's "repaired" tag) and must not be relabelled.
_NO_PROVENANCE = "(quality_flags IS NULL OR TRIM(quality_flags) IN ('', '{}'))"


def _rw_conn(db_path: str) -> Any:
    # Explicitly read-write, but do NOT create: a wrong path must fail loudly rather
    # than silently quarantining nothing in a brand-new empty file. closing() because
    # sqlite3's own context manager commits but does NOT close the connection.
    return closing(sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=15.0))


def quarantine_fabricated(
    db_path: str,
    tz: Any,
    *,
    dry_run: bool = True,
    before: datetime | None = None,
) -> dict[str, Any]:
    """Label historical price-mint rows. Returns a summary; dry_run writes nothing."""
    now = datetime.now(tz)

    result: dict[str, Any] = {
        "dry_run": dry_run,
        "cutoff": None,
        "candidates": 0,
        "already_quarantined": 0,
        "quarantined": 0,
        "stale_labels_cleared": 0,
        "future_minted_left_alone": 0,
        "first_slot": None,
        "last_slot": None,
        "error": None,
    }

    # Rows matching the fingerprint that no writer has claimed.
    unlabelled = f"({PRICE_MINT_PREDICATE}) AND slot_start < ? AND {_NO_PROVENANCE}"

    # A row we labelled that has since acquired REAL data (e.g. bin/backfill_ha.py
    # rewrites historical energy without touching quality_flags). Its label is now a
    # lie; clearing it is the implicit un-quarantine.
    stale = f"NOT ({PRICE_MINT_PREDICATE}) AND quality_flags LIKE '%\"{QUARANTINE_FLAG}\"%'"

    try:
        cutoff = before or (now - timedelta(minutes=_LIVE_EDGE_MINUTES))
        if cutoff > now:
            cutoff = now
        cutoff_s = cutoff.isoformat()
        result["cutoff"] = cutoff_s

        with _rw_conn(db_path) as conn:
            cur = conn.execute(
                f"SELECT COUNT(*), MIN(slot_start), MAX(slot_start) "
                f"FROM slot_observations WHERE {unlabelled}",
                (cutoff_s,),
            )
            count, first_slot, last_slot = cur.fetchone()
            result["candidates"] = int(count or 0)
            result["first_slot"] = first_slot
            result["last_slot"] = last_slot

            cur = conn.execute(
                f"SELECT COUNT(*) FROM slot_observations "
                f"WHERE ({PRICE_MINT_PREDICATE}) AND slot_start < ? "
                f"AND quality_flags LIKE '%\"{QUARANTINE_FLAG}\"%'",
                (cutoff_s,),
            )
            result["already_quarantined"] = int((cur.fetchone() or [0])[0] or 0)

            cur = conn.execute(f"SELECT COUNT(*) FROM slot_observations WHERE {stale}")
            stale_count = int((cur.fetchone() or [0])[0] or 0)

            # Reported so the operator can see they were deliberately skipped.
            cur = conn.execute(
                f"SELECT COUNT(*) FROM slot_observations WHERE ({PRICE_MINT_PREDICATE}) AND slot_start >= ?",
                (cutoff_s,),
            )
            result["future_minted_left_alone"] = int((cur.fetchone() or [0])[0] or 0)

            if dry_run:
                result["stale_labels_cleared"] = stale_count
            elif result["candidates"] > 0 or stale_count > 0:
                flags = json.dumps(
                    {QUARANTINE_FLAG: True, "quarantined_at": now.isoformat()},
                    separators=(",", ":"),
                )
                cur = conn.execute(
                    f"UPDATE slot_observations SET quality_flags = ? WHERE {unlabelled}",
                    (flags, cutoff_s),
                )
                labelled = cur.rowcount or 0

                cur = conn.execute(
                    f"UPDATE slot_observations SET quality_flags = NULL WHERE {stale}"
                )
                cleared = cur.rowcount or 0

                # Only report counts that survived the commit.
                conn.commit()
                result["quarantined"] = labelled
                result["stale_labels_cleared"] = cleared
                logger.warning(
                    "Quarantined %d price-mint observation rows (%s .. %s); cleared %d "
                    "stale labels; %d future minted rows left alone by design.",
                    labelled,
                    first_slot,
                    last_slot,
                    cleared,
                    result["future_minted_left_alone"],
                )
    except Exception as e:
        logger.exception("quarantine_fabricated failed")
        result["error"] = str(e)

    return result
