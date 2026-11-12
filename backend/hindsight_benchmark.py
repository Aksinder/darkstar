"""Daily battery hindsight benchmark (pure logic, no I/O).

Ported in spirit from upstream ergetie/darkstar 8c4fa6d7 (battery hindsight benchmark)
and 67bd4717 (grid-only baseline), adapted to this fork's data model:

* Inputs are the recorder's slot_observations. ``load_kwh`` is BASE load (water, EV
  and cyclic loads subtracted), so the whole-house demand is load + water + ev +
  cyclic — same as backend/learning/savings.py. ``cyclic_kwh`` is NULL on rows written
  before the column existed; the service fills it via savings.attach_cyclic_kwh, and
  a still-missing value counts as 0 (never as an unverified slot).
* Actual battery moves are the meter-implied flows (import - export + pv - demand),
  so the "actual" cost reconstructs the real grid bill exactly and the three schedules
  below are priced with ONE model. On balance-rescued slots (the norm on sunny
  quarters here) load itself came from that identity, so the meter-implied flow
  equals the measured battery flow by construction.
* Slot provenance uses this fork's quality_flags: ``load_rescued`` is normal and
  accepted; ``grid_unmeasured``, ``soc_stale``, ``load_read_invalid``,
  ``balance_residual_kwh``, ``multi_slot_delta``, ``base_load_clamped`` and
  ``load_spike_zeroed`` count as UNVERIFIED (a few are tolerated with a warning, more
  exclude the day). The device-level ``water_unmeasured`` / ``pv_unmeasured`` /
  ``ev_unmeasured`` are DEGRADED only: the device's energy stays in ``load_kwh``, so
  whole-house demand is intact and the slot is still scored. Every record carries the
  per-flag slot counts of both groups, so an excluded or degraded day says why.
* ``soc_start_percent`` is never populated in this DB; the day's start stock comes
  from the previous slot's ``soc_end_percent``, or — when that is missing — is
  back-derived from the first slot of the day that has one.
* The battery-only optimum is solved with planner/solver/battery_lp.py in benchmark
  mode (hard physics, configured SoC window widened to include both end stocks, end
  at the same stock as reality, no discharge while an EV charged) — NOT Kepler, so
  the nightly job costs well under a second per day on the 2-vCPU box and never
  touches the live solver.

Three schedules on the same recorded demand/PV/prices:
  actual       what the battery did
  hindsight    best battery-only schedule with perfect foresight
  grid_only    no battery at all (PV serves load, surplus exports, deficit imports)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from planner.errors import PlannerError
from planner.solver.battery_lp import BenchmarkTerms, solve_battery_only
from planner.solver.types import KeplerInput, KeplerInputSlot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from planner.solver.types import KeplerConfig

# 2: demand adds cyclic_kwh back (pool pump/filter were missing from v1 demand, which
# understated grid-only and hindsight costs). The bump rescores the stored window.
# 3: base_load_clamped and load_spike_zeroed slots count as unverified (demand above
# the metered total / an energy column zeroed as a spike), and records carry per-flag
# reason counts (unverified_by_flag, degraded_by_flag, slot/price completeness).
BENCHMARK_VERSION = 3
STATE_PREFIX = "hindsight_benchmark:"
SLOT_MINUTES = 15
MAX_UNVERIFIED_SLOTS = 4
MIN_IDEAL_VALUE_SEK = 1.0
EV_CHARGING_MIN_KW = 0.02
ENERGY_FIELDS = ("import_kwh", "export_kwh", "pv_kwh", "load_kwh", "water_kwh", "ev_charging_kwh")
PRICE_FIELDS = ("import_price_sek_kwh", "export_price_sek_kwh")
UNVERIFIED_FLAGS = (
    "grid_unmeasured",
    "soc_stale",
    "load_read_invalid",
    "balance_residual_kwh",
    "multi_slot_delta",
    "base_load_clamped",
    "load_spike_zeroed",
)
# Counted per day, never excluding: one device was not measured from power history,
# but its energy is still inside the slot's whole-house demand (see module docstring).
DEGRADED_FLAGS = ("water_unmeasured", "pv_unmeasured", "ev_unmeasured")


def state_key(day: date) -> str:
    return f"{STATE_PREFIX}{day.isoformat()}"


def _finite(row: Mapping[str, Any], key: str) -> bool:
    v = row.get(key)
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _flags(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("quality_flags")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {"unparseable": True}
    return parsed if isinstance(parsed, dict) else {}


def unverified_codes(row: Mapping[str, Any]) -> list[str]:
    """Why one slot is unverified: its UNVERIFIED_FLAGS, plus ``unparseable`` (flags
    not JSON) and ``non_finite_energy`` (an energy column missing or not finite)."""
    flags = _flags(row)
    codes = ["unparseable"] if flags.get("unparseable") else []
    codes += [k for k in UNVERIFIED_FLAGS if flags.get(k)]
    if not all(_finite(row, k) for k in ENERGY_FIELDS):
        codes.append("non_finite_energy")
    return codes


def degraded_codes(row: Mapping[str, Any]) -> list[str]:
    flags = _flags(row)
    return [k for k in DEGRADED_FLAGS if flags.get(k)]


def demand_kwh(row: Mapping[str, Any]) -> float:
    return (
        float(row["load_kwh"])
        + float(row["water_kwh"])
        + float(row["ev_charging_kwh"])
        + float(row.get("cyclic_kwh") or 0.0)
    )


def slot_bill(net_grid_kwh: float, import_price: float, export_price: float) -> float:
    """Import paid at import price, export earned at export price."""
    return max(net_grid_kwh, 0.0) * import_price - max(-net_grid_kwh, 0.0) * export_price


def meter_battery_flow(row: Mapping[str, Any]) -> tuple[float, float]:
    """(charge, discharge) kWh implied by the meter balance of one slot."""
    net = (
        float(row["import_kwh"]) - float(row["export_kwh"]) + float(row["pv_kwh"]) - demand_kwh(row)
    )
    return max(net, 0.0), max(-net, 0.0)


def schedule_cost(
    rows: Sequence[Mapping[str, Any]], flows: Sequence[tuple[float, float]], wear: float
) -> float:
    """Cost of a battery schedule on recorded demand/PV/prices (Kepler's half-cycle wear)."""
    total = 0.0
    for row, (ch, dis) in zip(rows, flows, strict=True):
        net = demand_kwh(row) - float(row["pv_kwh"]) + ch - dis
        total += slot_bill(
            net, float(row["import_price_sek_kwh"]), float(row["export_price_sek_kwh"])
        )
        total += (ch + dis) * wear * 0.5
    return total


def local_day_bounds(day: date, tz: Any) -> tuple[datetime, datetime]:
    start = tz.localize(datetime.combine(day, datetime.min.time()))
    end = tz.localize(datetime.combine(day + timedelta(days=1), datetime.min.time()))
    return start, end


def rows_by_local_day(rows: Sequence[dict[str, Any]], tz: Any) -> dict[date, list[dict[str, Any]]]:
    """Group rows by local date, each day sorted by real time (DST-safe)."""
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: datetime.fromisoformat(r["slot_start"])):
        grouped.setdefault(
            datetime.fromisoformat(row["slot_start"]).astimezone(tz).date(), []
        ).append(row)
    return grouped


def base_record(day: date, config: KeplerConfig) -> dict[str, Any]:
    return {
        "version": BENCHMARK_VERSION,
        "date": day.isoformat(),
        "status": "excluded",
        "reason": None,
        "warnings": [],
        "capacity_kwh": config.capacity_kwh,
        "wear_cost_sek_per_kwh": config.wear_cost_sek_per_kwh,
        "day_slots": 0,
        "unverified_slots": 0,
        # Reason codes (v3): slots per flag. A slot with two flags counts under both,
        # so the values can sum to more than unverified_slots.
        "unverified_by_flag": {},
        "degraded_by_flag": {},
        "expected_slots": None,
        "recorded_slots": None,
        "unpriced_slots": None,
        "start_soc_kwh": None,
        "stored_change_kwh": None,
        "metered_cost_sek": None,
        "actual_cost_sek": None,
        "hindsight_cost_sek": None,
        "grid_only_cost_sek": None,
        "missed_sek": None,
        "value_actual_sek": None,
        "value_hindsight_sek": None,
        "score": None,
        "actual_cycles_kwh": None,
        "hindsight_cycles_kwh": None,
        "solver_status": None,
        "solve_seconds": None,
    }


def _check_day(
    day: date, rows: list[dict[str, Any]], tz: Any, now: datetime
) -> tuple[str | None, dict[str, Any]]:
    """First failing data reason (or None) and the record fields that explain it."""
    start, end = local_day_bounds(day, tz)
    expected = int((end - start).total_seconds() // (SLOT_MINUTES * 60))
    detail: dict[str, Any] = {
        "unverified_slots": 0,
        "unverified_by_flag": {},
        "degraded_by_flag": {},
        "expected_slots": expected,
        "recorded_slots": len(rows),
        "unpriced_slots": None,
    }
    if end > now:
        return "not_finished", detail
    if len(rows) != expected:
        return "incomplete_day", detail
    times = [
        (datetime.fromisoformat(r["slot_start"]), datetime.fromisoformat(r["slot_end"]))
        for r in rows
    ]
    if (
        times[0][0] != start
        or times[-1][1] != end
        or any(times[i - 1][1] != times[i][0] for i in range(1, len(times)))
    ):
        return "incomplete_day", detail
    unverified = 0
    for r in rows:
        codes = unverified_codes(r)
        unverified += bool(codes)
        for code in codes:
            detail["unverified_by_flag"][code] = detail["unverified_by_flag"].get(code, 0) + 1
        for code in degraded_codes(r):
            detail["degraded_by_flag"][code] = detail["degraded_by_flag"].get(code, 0) + 1
    detail["unverified_slots"] = unverified
    detail["unpriced_slots"] = sum(not all(_finite(r, k) for k in PRICE_FIELDS) for r in rows)
    if unverified > MAX_UNVERIFIED_SLOTS:
        return "not_clean_recording", detail
    if detail["unpriced_slots"]:
        return "missing_prices", detail
    return None, detail


def _filled(row: dict[str, Any]) -> dict[str, Any]:
    """Missing energy values as 0 kWh so actual and hindsight see identical inputs."""
    if all(_finite(row, k) for k in ENERGY_FIELDS):
        return row
    out = dict(row)
    for k in ENERGY_FIELDS:
        if not _finite(row, k):
            out[k] = 0.0
    return out


def _start_stock(
    rows: list[dict[str, Any]],
    previous_row: Mapping[str, Any] | None,
    flows: list[tuple[float, float]],
    config: KeplerConfig,
) -> float | None:
    """Stored energy at the start of the day (kWh), or None when undeterminable.

    Uses charge_efficiency for ALL recorded charge: the meter-implied flows carry no
    DC-PV / grid split (see benchmark_day for why pv_charge_efficiency is off here).
    """
    cap = config.capacity_kwh
    if previous_row is not None and _finite(previous_row, "soc_end_percent"):
        return float(previous_row["soc_end_percent"]) * cap / 100.0
    stock_delta = 0.0
    for row, (ch, dis) in zip(rows, flows, strict=True):
        stock_delta += ch * config.charge_efficiency - dis / config.discharge_efficiency
        if _finite(row, "soc_end_percent") and not _flags(row).get("soc_stale"):
            return float(row["soc_end_percent"]) * cap / 100.0 - stock_delta
    return None


def benchmark_day(
    day: date,
    rows_by_day: Mapping[date, list[dict[str, Any]]],
    config: KeplerConfig,
    tz: Any,
    now: datetime,
) -> dict[str, Any]:
    """Score one finished local day. ``rows_by_day`` must include the previous day."""
    # Single charge efficiency on BOTH sides of the comparison. The recorded flows are
    # meter-implied AC battery flows with no DC-PV / grid split, so the start/terminal
    # stock below (_start_stock, terminal) can only apply charge_efficiency. Letting
    # the hindsight solve use pv_charge_efficiency on top would let it hit the same
    # terminal stock with less charge than the stock accounting assumes — a phantom
    # hindsight saving. Any pv/grid split of recorded data (e.g. min(ch, hybrid pv))
    # would be a guess, so the benchmark stays on the legacy single-efficiency model.
    config = replace(config, pv_charge_efficiency=None)
    record = base_record(day, config)
    raw_rows = rows_by_day.get(day, [])
    reason, detail = _check_day(day, raw_rows, tz, now)
    record.update(detail)
    if reason is not None:
        record["reason"] = reason
        return record
    rows = [_filled(r) for r in raw_rows]
    record["day_slots"] = len(rows)
    if record["unverified_slots"]:
        record["warnings"].append("unverified_slots")

    flows = [meter_battery_flow(r) for r in rows]
    previous = rows_by_day.get(day - timedelta(days=1)) or []
    prev_row = (
        previous[-1]
        if previous
        and previous[-1].get("slot_end")
        and datetime.fromisoformat(str(previous[-1]["slot_end"]))
        == datetime.fromisoformat(raw_rows[0]["slot_start"])
        else None
    )
    initial = _start_stock(rows, prev_row, flows, config)
    if initial is None:
        record["reason"] = "missing_soc"
        return record
    if prev_row is None:
        record["warnings"].append("start_soc_back_derived")

    cap = config.capacity_kwh
    terminal = initial + sum(
        ch * config.charge_efficiency - dis / config.discharge_efficiency for ch, dis in flows
    )
    if not (0.0 <= initial <= cap and 0.0 <= terminal <= cap):
        # Flow drift vs SoC or capacity config; clamp so the solve stays feasible.
        initial = min(max(initial, 0.0), cap)
        terminal = min(max(terminal, 0.0), cap)
        record["warnings"].append("stock_clamped")
    record["start_soc_kwh"] = round(initial, 3)
    record["stored_change_kwh"] = round(terminal - initial, 3)

    slots = [
        KeplerInputSlot(
            start_time=datetime.fromisoformat(r["slot_start"]),
            end_time=datetime.fromisoformat(r["slot_end"]),
            load_kwh=demand_kwh(r),
            pv_kwh=float(r["pv_kwh"]),
            import_price_sek_kwh=float(r["import_price_sek_kwh"]),
            export_price_sek_kwh=float(r["export_price_sek_kwh"]),
        )
        for r in rows
    ]
    no_discharge = []
    for r, s in zip(rows, slots, strict=True):
        hours = (s.end_time - s.start_time).total_seconds() / 3600.0
        no_discharge.append(hours > 0 and float(r["ev_charging_kwh"]) / hours > EV_CHARGING_MIN_KW)

    started = time.monotonic()
    try:
        result = solve_battery_only(
            KeplerInput(slots=slots, initial_soc_kwh=initial),
            config,
            mode="benchmark",
            benchmark=BenchmarkTerms(terminal_soc_kwh=terminal, no_discharge_slots=no_discharge),
        )
    except PlannerError as exc:
        record["solve_seconds"] = round(time.monotonic() - started, 3)
        record["status"] = "failed"
        record["reason"] = "solver_not_optimal"
        record["solver_status"] = str(exc.details.get("solver_status") or exc.code.value)
        return record
    record["solve_seconds"] = round(time.monotonic() - started, 3)
    record["solver_status"] = result.status_msg

    wear = config.wear_cost_sek_per_kwh
    hindsight_flows = [(s.charge_kwh, s.discharge_kwh) for s in result.slots]
    actual = schedule_cost(rows, flows, wear)
    hindsight = schedule_cost(rows, hindsight_flows, wear)
    grid_only = schedule_cost(rows, [(0.0, 0.0)] * len(rows), wear)
    metered = sum(
        float(r["import_kwh"]) * float(r["import_price_sek_kwh"])
        - float(r["export_kwh"]) * float(r["export_price_sek_kwh"])
        for r in rows
    )
    value_hindsight = grid_only - hindsight
    record.update(
        status="scored",
        metered_cost_sek=round(metered, 4),
        actual_cost_sek=round(actual, 4),
        hindsight_cost_sek=round(hindsight, 4),
        grid_only_cost_sek=round(grid_only, 4),
        missed_sek=round(actual - hindsight, 4),
        value_actual_sek=round(grid_only - actual, 4),
        value_hindsight_sek=round(value_hindsight, 4),
        score=(
            round((grid_only - actual) / value_hindsight, 4)
            if value_hindsight >= MIN_IDEAL_VALUE_SEK
            else None
        ),
        actual_cycles_kwh=round(sum(d for _, d in flows), 3),
        hindsight_cycles_kwh=round(sum(d for _, d in hindsight_flows), 3),
    )
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals over scored days plus counts by status, reason and warning.

    ``unverified_by_flag`` / ``degraded_by_flag`` sum the per-day slot counts over all
    days; ``excluded_by_flag`` over the days excluded as ``not_clean_recording`` only,
    i.e. which flags cost whole days. Records stored before v3 lack the counts.
    """
    scored = [r for r in records if r.get("status") == "scored"]
    value_actual = sum(r["value_actual_sek"] for r in scored)
    value_hindsight = sum(r["value_hindsight_sek"] for r in scored)
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_warning: dict[str, int] = {}
    unverified_by_flag: dict[str, int] = {}
    degraded_by_flag: dict[str, int] = {}
    excluded_by_flag: dict[str, int] = {}
    for r in records:
        targets = [(unverified_by_flag, r.get("unverified_by_flag"))]
        targets.append((degraded_by_flag, r.get("degraded_by_flag")))
        if r.get("reason") == "not_clean_recording":
            targets.append((excluded_by_flag, r.get("unverified_by_flag")))
        for total, counts in targets:
            for flag, n in (counts or {}).items():
                total[flag] = total.get(flag, 0) + int(n)
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        if r.get("reason"):
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        for w in r.get("warnings") or []:
            by_warning[w] = by_warning.get(w, 0) + 1
    return {
        "days_total": len(records),
        "days_scored": len(scored),
        "actual_cost_sek": round(sum(r["actual_cost_sek"] for r in scored), 3),
        "hindsight_cost_sek": round(sum(r["hindsight_cost_sek"] for r in scored), 3),
        "grid_only_cost_sek": round(sum(r["grid_only_cost_sek"] for r in scored), 3),
        "missed_sek": round(sum(r["missed_sek"] for r in scored), 3),
        "value_actual_sek": round(value_actual, 3),
        "value_hindsight_sek": round(value_hindsight, 3),
        "score": (
            round(value_actual / value_hindsight, 4)
            if value_hindsight >= MIN_IDEAL_VALUE_SEK
            else None
        ),
        "by_status": by_status,
        "by_reason": by_reason,
        "by_warning": by_warning,
        "unverified_by_flag": unverified_by_flag,
        "degraded_by_flag": degraded_by_flag,
        "excluded_by_flag": excluded_by_flag,
    }
