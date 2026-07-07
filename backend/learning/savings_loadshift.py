"""Load-shift savings (v2) — credit streams for what Darkstar does with LOADS.

Complements backend/learning/savings.py (the battery-layer counterfactual): these
streams credit the value of MOVING load in time — hot-water heating scheduled to
cheap/PV hours, and deferrable appliance cycles shifted from the human's start
press to a cheaper window. Published as separate sensors
(``sensor.darkstar_loadshift_*``) so the battery floor is never conflated.

Valuation rule (the double-count guard): every shifted kWh is valued at the slot's
MARGINAL grid price in the realized world — the import price when the slot was
net-importing, the export price when net-exporting (tie -> import price: one more
kWh of load would have been imported). The battery counterfactual holds the
realized load profile fixed in both of its worlds, so no kWh is priced twice, and
PV-surplus heating is automatically credited (it only cost the forgone export).

Baselines:
- Water, primary (``four_cheapest_hours``): the same measured daily kWh heated at
  the day's 4 cheapest clock hours, capped at element power per slot, valued at
  import price — literally the pre-Darkstar automations for the main tank, and
  deliberately the hardest baseline to beat (comfort-driven mid-price heating
  goes NEGATIVE and is shown).
- Water, secondary (``daily_average``): the same kWh at the day's unweighted
  average import price — for tanks with no predecessor automation (villavagn).
- Appliances: run-at-arm-time — the measured cycle kWh laid over the slots from
  the arm press, at those slots' marginal prices, vs the actual run window at its
  marginal prices. Both sides use the marginal rule so a never-deferred cycle
  scores exactly 0 and a deadline-pressure run scores negative.

Honesty contract (matches savings.py):
- raw signs everywhere — negative prices and negative credits pass through;
- no mocked inputs — only measured kWh is credited (never seed/assumed energy);
  unpriceable slots/cycles are COUNTED and exposed as coverage, never silently
  folded toward zero;
- EV charging is deliberately NOT credited (the surplus controller has its own
  economics and no arm-time counterfactual) — spelled out in the sensor.

All computation is pure over already-fetched rows; the only I/O helpers are the
JSONL cycle-ledger loader and its enricher (which persists the run join so old
cycles stay priceable after the HA detection history ages out).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "WATER_BASELINE_DAILY_AVERAGE",
    "WATER_BASELINE_FOUR_CHEAPEST",
    "ApplianceShiftSummary",
    "CycleLedgerEntry",
    "LoadshiftSummary",
    "MeasuredRun",
    "WaterShiftSummary",
    "compute_appliance_shift",
    "compute_water_shift",
    "dedupe_ledger",
    "enrich_cycle_ledger",
    "load_cycle_ledger",
    "marginal_price",
    "water_unattributed_kwh",
]

logger = logging.getLogger("darkstar.savings_loadshift")

WATER_BASELINE_FOUR_CHEAPEST = "four_cheapest_hours"
WATER_BASELINE_DAILY_AVERAGE = "daily_average"

_SLOT_HOURS = 0.25


def marginal_price(row: Mapping[str, Any]) -> float | None:
    """The slot's marginal grid price: what one more/less kWh of load was worth.

    Import price when the slot was net-importing, export price when net-exporting
    (a shifted kWh there displaced export revenue, not import cost). Tie — including
    the islanded 0/0 case — resolves to import price: one additional kWh of load
    would have been imported. Returns None when the needed price is missing (the
    slot then cannot be valued; callers must COUNT it, not assume 0).
    """
    imp = float(row.get("import_kwh") or 0.0)
    exp = float(row.get("export_kwh") or 0.0)
    key = "export_price_sek_kwh" if exp > imp else "import_price_sek_kwh"
    price = row.get(key)
    return float(price) if price is not None else None


# ---------------------------------------------------------------------------
# Stream 1 — water heating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterShiftSummary:
    """One tank's load-shift credit over a window (positive = Darkstar saved money)."""

    tank_id: str
    baseline_name: str
    actual_cost_sek: float
    baseline_cost_sek: float
    credit_sek: float  # baseline - actual
    valued_kwh: float  # tank energy that carried the needed prices
    unvalued_kwh: float  # tank energy that could not be valued (missing prices)
    n_days: int

    @property
    def coverage(self) -> float:
        """Fraction of the tank's energy that could be valued (1.0 when idle)."""
        total = self.valued_kwh + self.unvalued_kwh
        return self.valued_kwh / total if total else 1.0


def _day_of(slot_start: str) -> str:
    """Local calendar day of an ISO slot_start (stored in local-offset format)."""
    return slot_start[:10]


def _four_cheapest_baseline_cost(
    day_rows: Sequence[Mapping[str, Any]],
    energy_kwh: float,
    element_power_kw: float,
    slot_hours: float,
) -> float:
    """Cost of heating ``energy_kwh`` the way the pre-Darkstar automations did:
    inside the day's 4 cheapest clock hours, cheapest slot first, capped at
    element power per slot, at import price.

    Feasibility spill: energy beyond the 4 hours' element capacity fills the
    remaining priced slots cheapest-first (still capped) — without the cap the
    baseline could "heat" 10 kWh in one cheap hour, a straw man in Darkstar's
    favor. Any residual beyond ALL priced slots (physically impossible unless the
    day's price record is thin) is valued at the day's average import price so
    energy is never silently dropped from the baseline.
    """
    priced = [r for r in day_rows if r.get("import_price_sek_kwh") is not None]
    if not priced or energy_kwh <= 0:
        return 0.0

    by_hour: dict[str, list[Mapping[str, Any]]] = {}
    for r in priced:
        by_hour.setdefault(str(r["slot_start"])[11:13], []).append(r)
    hour_price = {
        h: sum(float(r["import_price_sek_kwh"]) for r in rows) / len(rows)
        for h, rows in by_hour.items()
    }
    cheapest_hours = sorted(hour_price, key=lambda h: hour_price[h])[:4]

    in_hours = [r for h in cheapest_hours for r in by_hour[h]]
    rest = [r for r in priced if str(r["slot_start"])[11:13] not in cheapest_hours]
    ordered = sorted(in_hours, key=lambda r: float(r["import_price_sek_kwh"])) + sorted(
        rest, key=lambda r: float(r["import_price_sek_kwh"])
    )

    cap = element_power_kw * slot_hours
    remaining = energy_kwh
    cost = 0.0
    for r in ordered:
        if remaining <= 0:
            break
        take = min(remaining, cap)
        cost += take * float(r["import_price_sek_kwh"])
        remaining -= take
    if remaining > 0:
        avg = sum(float(r["import_price_sek_kwh"]) for r in priced) / len(priced)
        cost += remaining * avg
    return cost


def compute_water_shift(
    obs_rows: Sequence[Mapping[str, Any]],
    device_rows: Iterable[Mapping[str, Any]],
    *,
    tank_id: str,
    element_power_kw: float,
    baseline: str = WATER_BASELINE_FOUR_CHEAPEST,
    slot_hours: float = _SLOT_HOURS,
) -> WaterShiftSummary:
    """Water-heating load-shift credit for one tank over the rows' window.

    ``obs_rows`` are get_observation_rows_between dicts (grid flows + prices);
    ``device_rows`` are get_device_energy_rows_between dicts. Days are the local
    calendar days present in the device rows; the baseline is reconstructed from
    each day's RECORDED slot prices (never re-fetched — pitfall: retro prices are
    already in the DB and the old automations picked hours per local day).
    """
    if baseline not in (WATER_BASELINE_FOUR_CHEAPEST, WATER_BASELINE_DAILY_AVERAGE):
        raise ValueError(f"unknown water baseline: {baseline!r}")

    rows_by_slot = {str(r["slot_start"]): r for r in obs_rows}
    days_obs: dict[str, list[Mapping[str, Any]]] = {}
    for r in obs_rows:
        days_obs.setdefault(_day_of(str(r["slot_start"])), []).append(r)

    tank_days: dict[str, list[tuple[str, float]]] = {}
    for d in device_rows:
        if str(d["device_id"]) != tank_id:
            continue
        kwh = float(d["kwh"] or 0.0)
        if kwh <= 0:
            continue
        slot = str(d["slot_start"])
        tank_days.setdefault(_day_of(slot), []).append((slot, kwh))

    actual = 0.0
    baseline_cost = 0.0
    valued = 0.0
    unvalued = 0.0
    for day, slots in tank_days.items():
        day_rows = days_obs.get(day, [])
        priced = [r for r in day_rows if r.get("import_price_sek_kwh") is not None]
        if not priced:
            # No import prices recorded for the whole day: neither baseline can be
            # built — count the day's energy as unvalued rather than inventing one.
            unvalued += sum(kwh for _s, kwh in slots)
            continue

        day_valued = 0.0
        for slot, kwh in slots:
            row = rows_by_slot.get(slot)
            price = marginal_price(row) if row is not None else None
            if price is None:
                unvalued += kwh
                continue
            actual += kwh * price
            day_valued += kwh
        valued += day_valued

        if baseline == WATER_BASELINE_DAILY_AVERAGE:
            avg = sum(float(r["import_price_sek_kwh"]) for r in priced) / len(priced)
            baseline_cost += day_valued * avg
        else:
            baseline_cost += _four_cheapest_baseline_cost(
                day_rows, day_valued, element_power_kw, slot_hours
            )

    return WaterShiftSummary(
        tank_id=tank_id,
        baseline_name=baseline,
        actual_cost_sek=round(actual, 4),
        baseline_cost_sek=round(baseline_cost, 4),
        credit_sek=round(baseline_cost - actual, 4),
        valued_kwh=round(valued, 4),
        unvalued_kwh=round(unvalued, 4),
        n_days=len(tank_days),
    )


def water_unattributed_kwh(
    obs_rows: Sequence[Mapping[str, Any]],
    device_rows: Iterable[Mapping[str, Any]],
    tank_ids: Iterable[str],
) -> float:
    """Aggregate water energy not attributable to any tank's device rows.

    Per slot: ``max(0, water_kwh - sum(tank device rows))``, summed over the
    window. ``compute_water_shift`` only sees device rows with kwh>0, so a slot
    whose per-device value was never recorded (recorder snapshot fallback) or
    was wiped would otherwise vanish from the per-tank stream while coverage
    stayed 1.0 — the shortfall must be COUNTED as unvalued, never silently
    folded (module contract). The caller cannot attribute it to a specific
    tank, so it is exposed as a separate unattributed bucket.
    """
    ids = {str(t) for t in tank_ids}
    dev_by_slot: dict[str, float] = {}
    for d in device_rows:
        if str(d["device_id"]) not in ids:
            continue  # EV chargers share the table but not the water aggregate
        slot = str(d["slot_start"])
        dev_by_slot[slot] = dev_by_slot.get(slot, 0.0) + float(d["kwh"] or 0.0)
    total = 0.0
    for r in obs_rows:
        water = float(r.get("water_kwh") or 0.0)
        shortfall = water - dev_by_slot.get(str(r["slot_start"]), 0.0)
        # 1e-9: float-noise tolerance only, not a materiality threshold.
        if shortfall > 1e-9:
            total += shortfall
    return round(total, 4)


# ---------------------------------------------------------------------------
# Stream 2 — deferrable appliances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleLedgerEntry:
    """One arm->done record from data/deferrable_cycles.jsonl."""

    load_id: str
    armed_ts: float
    done_ts: float
    held_by_us_ever: bool = False
    deadline_ts: float | None = None
    measured_kwh: float | None = None
    # Persisted by enrich_cycle_ledger (the executor only sees instantaneous W):
    # the matched detected run's window, so the cycle stays priceable after the
    # ~14-day HA detection history (bounded further by recorder purge) ages out.
    run_start_ts: float | None = None
    run_end_ts: float | None = None


@dataclass(frozen=True)
class MeasuredRun:
    """A measured appliance run window (from cycle detection on the plug meter)."""

    start_ts: float
    end_ts: float
    energy_kwh: float


@dataclass(frozen=True)
class ApplianceShiftSummary:
    """One appliance's load-shift credit over a window."""

    load_id: str
    actual_cost_sek: float
    baseline_cost_sek: float
    credit_sek: float  # baseline - actual
    n_valued_cycles: int
    unvalued_cycles: int  # completed cycles with no measured kWh / no full prices

    @property
    def coverage(self) -> float:
        """Fraction of completed cycles that could be valued (1.0 when none ran)."""
        total = self.n_valued_cycles + self.unvalued_cycles
        return self.n_valued_cycles / total if total else 1.0


def _opt_float(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key)
    return float(value) if value is not None else None


def _entry_from_raw(raw: Mapping[str, Any]) -> CycleLedgerEntry:
    """Parse one ledger JSON object (raises ValueError/TypeError/KeyError on junk)."""
    return CycleLedgerEntry(
        load_id=str(raw["load_id"]),
        armed_ts=float(raw["armed_ts"]),
        done_ts=float(raw["done_ts"]),
        held_by_us_ever=bool(raw.get("held_by_us_ever", False)),
        deadline_ts=_opt_float(raw, "deadline_ts"),
        measured_kwh=_opt_float(raw, "measured_kwh"),
        run_start_ts=_opt_float(raw, "run_start_ts"),
        run_end_ts=_opt_float(raw, "run_end_ts"),
    )


def load_cycle_ledger(path: str | Path) -> list[CycleLedgerEntry]:
    """Read the append-only JSONL cycle ledger. Missing file -> []. Corrupt lines
    are skipped (a torn write must not take the savings sensor down)."""
    p = Path(path)
    if not p.exists():
        return []
    entries: list[CycleLedgerEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_entry_from_raw(json.loads(line)))
        except (ValueError, TypeError, KeyError):
            continue
    return entries


def enrich_cycle_ledger(
    path: str | Path,
    runs_by_load: Mapping[str, Sequence[MeasuredRun]],
) -> list[CycleLedgerEntry]:
    """Load the ledger, back-filling energy + run window from detected cycles,
    and atomically persist the enrichment. Returns the (enriched) entries.

    The executor writes rows with ``measured_kwh=None`` (it only sees
    instantaneous W); the matching detected cycle exists only inside the
    ~14-day HA history horizon (less with recorder purge), so without
    persisting the join every older cycle becomes permanently unvalued and the
    30d appliance credit decays. One tick of overlap between done-time and the
    horizon suffices (both the row and the cycle exist at the first publish
    after done). Idempotent: already-filled rows are skipped. The rewrite is in
    place (temp file + atomic replace) — an appended superseding row would be
    dropped by dedupe_ledger's strictly-greater done_ts rule. A failed rewrite
    is logged, never raised; the enriched entries are still returned so this
    tick's valuation works regardless.
    """
    p = Path(path)
    if not p.exists():
        return []
    entries: list[CycleLedgerEntry] = []
    out_lines: list[str] = []
    changed = False
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw: dict[str, Any] = json.loads(stripped)
            entry = _entry_from_raw(raw)
        except (ValueError, TypeError, KeyError):
            out_lines.append(line)  # keep corrupt lines verbatim (loader skips them)
            continue
        if entry.measured_kwh is None or entry.run_start_ts is None or entry.run_end_ts is None:
            run = _match_run(runs_by_load.get(entry.load_id, ()), entry)
            if run is not None and run.energy_kwh > 0:
                if raw.get("measured_kwh") is None:
                    raw["measured_kwh"] = run.energy_kwh
                if raw.get("run_start_ts") is None:
                    raw["run_start_ts"] = run.start_ts
                if raw.get("run_end_ts") is None:
                    raw["run_end_ts"] = run.end_ts
                entry = _entry_from_raw(raw)
                changed = True
        entries.append(entry)
        out_lines.append(json.dumps(raw))
    if changed:
        try:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            tmp.replace(p)  # atomic on POSIX
        except OSError as exc:
            logger.warning("Cycle-ledger enrichment rewrite failed: %s", exc)
    return entries


def dedupe_ledger(entries: Iterable[CycleLedgerEntry]) -> list[CycleLedgerEntry]:
    """Collapse rearm-cooldown continuations: rows sharing (load_id, armed_ts) are
    one physical programme; keep the one with the LAST done_ts."""
    best: dict[tuple[str, float], CycleLedgerEntry] = {}
    for e in entries:
        key = (e.load_id, e.armed_ts)
        cur = best.get(key)
        if cur is None or e.done_ts > cur.done_ts:
            best[key] = e
    return sorted(best.values(), key=lambda e: e.done_ts)


def _slot_ts_index(obs_rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, Mapping[str, Any]]]:
    out: list[tuple[float, Mapping[str, Any]]] = []
    for r in obs_rows:
        try:
            out.append((datetime.fromisoformat(str(r["slot_start"])).timestamp(), r))
        except (ValueError, TypeError, KeyError):
            continue
    out.sort(key=lambda t: t[0])
    return out


def _window_cost(
    slots: Sequence[tuple[float, Mapping[str, Any]]],
    start_ts: float,
    end_ts: float,
    energy_kwh: float,
    slot_hours: float,
) -> float | None:
    """Cost of ``energy_kwh`` spread flat over [start_ts, end_ts) at each
    overlapped slot's marginal price.

    Returns None when the window is not fully covered by priced observation
    slots — an unpriceable window must make the cycle UNVALUED, never silently
    shrink its cost toward zero.
    """
    duration = end_ts - start_ts
    if duration <= 0:
        return None
    slot_s = slot_hours * 3600.0
    covered = 0.0
    cost = 0.0
    for ts, row in slots:
        overlap = min(end_ts, ts + slot_s) - max(start_ts, ts)
        if overlap <= 0:
            continue
        price = marginal_price(row)
        if price is None:
            return None
        covered += overlap
        cost += energy_kwh * (overlap / duration) * price
    # 1 s tolerance for ISO-timestamp rounding at the window edges.
    if covered < duration - 1.0:
        return None
    return cost


def _match_run(
    runs: Sequence[MeasuredRun], entry: CycleLedgerEntry, tolerance_s: float = 900.0
) -> MeasuredRun | None:
    """The measured run belonging to a ledger entry: the run with the largest
    overlap with [armed_ts - tol, done_ts + tol]. None when nothing overlaps."""
    lo = entry.armed_ts - tolerance_s
    hi = entry.done_ts + tolerance_s
    best: MeasuredRun | None = None
    best_overlap = 0.0
    for run in runs:
        overlap = min(hi, run.end_ts) - max(lo, run.start_ts)
        if overlap > best_overlap:
            best_overlap = overlap
            best = run
    return best


def compute_appliance_shift(
    entries: Iterable[CycleLedgerEntry],
    runs: Sequence[MeasuredRun],
    obs_rows: Sequence[Mapping[str, Any]],
    *,
    load_id: str,
    window_start_ts: float | None = None,
    window_end_ts: float | None = None,
    slot_hours: float = _SLOT_HOURS,
) -> ApplianceShiftSummary:
    """Appliance load-shift credit: run-at-arm-time baseline vs the actual run.

    A cycle is attributed to the window its ``done_ts`` falls in (completion day
    — a cross-midnight cycle is credited once, on the day it finished). Only
    measured energy counts: a cycle with no matching measured run (live or
    persisted), or whose arm/run window is not fully priced by the observation
    rows, lands in ``unvalued_cycles``. Cycles Darkstar never held
    (``held_by_us_ever`` False) score exactly 0 by construction — their
    armed_ts is never priced; deadline-pressure runs that cost MORE than the
    arm press would have go negative — unclamped.
    """
    slots = _slot_ts_index(obs_rows)
    actual = 0.0
    baseline = 0.0
    n_valued = 0
    unvalued = 0
    for entry in dedupe_ledger(entries):
        if entry.load_id != load_id:
            continue
        if window_start_ts is not None and entry.done_ts < window_start_ts:
            continue
        if window_end_ts is not None and entry.done_ts >= window_end_ts:
            continue

        run = _match_run(runs, entry)
        if (
            run is None
            and entry.run_start_ts is not None
            and entry.run_end_ts is not None
            and entry.measured_kwh is not None
        ):
            # Durable fallback: enrich_cycle_ledger persisted the matched run at
            # done time, so cycles older than the detection horizon stay priced
            # (prices are durable too — 30d slot observations in the learning DB).
            run = MeasuredRun(entry.run_start_ts, entry.run_end_ts, entry.measured_kwh)
        energy = entry.measured_kwh if entry.measured_kwh is not None else None
        if run is not None and energy is None:
            energy = run.energy_kwh
        if run is None or energy is None or energy <= 0:
            unvalued += 1
            continue

        actual_cost = _window_cost(slots, run.start_ts, run.end_ts, energy, slot_hours)
        if not entry.held_by_us_ever:
            # Contract enforced by construction: a cycle Darkstar never held
            # scores exactly 0 — never priced against its armed_ts, which for a
            # chain-merged back-to-back reload can be a DIFFERENT programme's
            # arm press (fabricated credit either sign otherwise).
            if actual_cost is None:
                unvalued += 1
                continue
            baseline += actual_cost
            actual += actual_cost
            n_valued += 1
            continue

        run_duration = run.end_ts - run.start_ts
        baseline_cost = _window_cost(
            slots, entry.armed_ts, entry.armed_ts + run_duration, energy, slot_hours
        )
        if baseline_cost is None or actual_cost is None:
            unvalued += 1
            continue
        baseline += baseline_cost
        actual += actual_cost
        n_valued += 1

    return ApplianceShiftSummary(
        load_id=load_id,
        actual_cost_sek=round(actual, 4),
        baseline_cost_sek=round(baseline, 4),
        credit_sek=round(baseline - actual, 4),
        n_valued_cycles=n_valued,
        unvalued_cycles=unvalued,
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadshiftSummary:
    """All load-shift streams over one window (today / rolling 30d)."""

    water: tuple[WaterShiftSummary, ...] = ()
    appliances: tuple[ApplianceShiftSummary, ...] = ()

    @property
    def credit_sek(self) -> float:
        return round(
            sum(w.credit_sek for w in self.water) + sum(a.credit_sek for a in self.appliances),
            4,
        )
