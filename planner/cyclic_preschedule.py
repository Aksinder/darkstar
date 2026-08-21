"""
Greedy pre-scheduling of cyclic loads (pool pump, filter) OUTSIDE the MILP.

Why not the solver: a pump is 0.26-0.39 kW — noise against the battery economics —
but as a planned load it is ~30 hourly binaries per device over the horizon, plus
day floors and rolling max-gap windows, all disjunctive. Measured on the live
2026-08-20 instance: 6 s without the pumps, 11 s with them at zero spacing, and on
the 2-vCPU box the morning instance blew straight through the 240 s budget (262 s)
— the old plan was kept and the pumps never entered any plan at all. The solver
kept timing out over loads whose entire daily cost is a couple of kronor.

So the pumps are planned here, greedily, against the import price curve: per day
bucket, the cheapest whole hours until the daily need is met, then the cheapest
hour inside any gap that exceeds max_hours_between. Their consumption is then added
to the slot load the solver sees (fixed base load, zero new variables) and written
into the schedule's per-slot ``water_heaters`` so the executor drives the switches
exactly as before.

What is given up: the solver can no longer shift a pump to soak a forecast PV
surplus or dodge a battery-discharge hour. The executor's measured opportunistic
gates (surplus_run / presence) cover the surplus case live, which is the honest
signal anyway; the rest is rounding error at these powers.

Pure: no I/O, no config access — takes prices and returns slot indices.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class PriceSlot:
    start_time: datetime
    import_price_sek_kwh: float


@dataclass(frozen=True)
class CyclicSpec:
    id: str
    power_kw: float
    min_kwh_per_day: float
    max_hours_between: float | None = None
    heated_today_kwh: float = 0.0  # energy already delivered in bucket 0
    # When the load last ran (or now, if running). Anchors the max-gap rule: without
    # it the first planned hour could sit arbitrarily far from the last real run —
    # observed 2026-08-21: filter off at 16:11, next planned hour 01:00, a 9 h gap
    # against a 6 h rule, because the rule only looked BETWEEN planned hours.
    last_run_end: datetime | None = None
    enabled: bool = True


def _as_naive_like(dt: datetime | None, reference: datetime) -> datetime | None:
    """Bring ``dt`` into the same awareness as the slot timestamps so they compare.

    HA hands us an aware ISO timestamp; the planner's slots may be aware or naive
    depending on the caller. Comparing the two raises — and a raise here would take
    the whole plan down over a pump's gap anchor.
    """
    if dt is None:
        return None
    if reference.tzinfo is None and dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    if reference.tzinfo is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=reference.tzinfo)
    return dt


def _hour_key(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _bucket_date(dt: datetime, defer_hours: float):
    d = dt.date()
    if defer_hours > 0 and dt.hour < defer_hours:
        d = d - timedelta(days=1)
    return d


def preschedule_cyclic_loads(
    specs: list[CyclicSpec],
    slots: list[PriceSlot],
    *,
    defer_up_to_hours: float = 0.0,
) -> dict[str, dict[int, float]]:
    """Return ``{load_id: {slot_index: kw}}`` — the slots each load runs in.

    Whole hours only (every slot of a chosen hour), matching the tanks' hourly
    blocks so the executor never sees a 15-minute pump flicker. Day buckets follow
    the same ``defer_up_to_hours`` offset as the solver's water floors, so "a day"
    means the same thing on both sides of the fence.
    """
    out: dict[str, dict[int, float]] = {}
    if not slots:
        return out

    # hour -> slot indices, and per-bucket hour lists in chronological order.
    hour_slots: dict[datetime, list[int]] = defaultdict(list)
    for i, s in enumerate(slots):
        hour_slots[_hour_key(s.start_time)].append(i)
    hours = sorted(hour_slots)
    hour_price = {
        h: sum(slots[i].import_price_sek_kwh for i in idx) / len(idx)
        for h, idx in hour_slots.items()
    }
    buckets: dict[object, list[datetime]] = defaultdict(list)
    for h in hours:
        buckets[_bucket_date(h, defer_up_to_hours)].append(h)
    bucket_keys = sorted(buckets)

    for spec in specs:
        if not spec.enabled or spec.power_kw <= 0:
            continue
        chosen: set[datetime] = set()
        # The gap clock starts at the last REAL run, then carries across buckets.
        last_on: datetime | None = _as_naive_like(spec.last_run_end, slots[0].start_time)
        for bi, key in enumerate(bucket_keys):
            bhours = buckets[key]
            need_kwh = spec.min_kwh_per_day - (spec.heated_today_kwh if bi == 0 else 0.0)
            need_hours = max(0, math.ceil(max(0.0, need_kwh) / spec.power_kw - 1e-9))
            # A partial first bucket cannot deliver more hours than it has left.
            need_hours = min(need_hours, len(bhours))

            picked = sorted(sorted(bhours, key=lambda h: hour_price[h])[:need_hours])

            # Gap repair: inside any stretch longer than max_hours_between without a
            # run, add the cheapest hour of that stretch. Repeats until clean — each
            # pass strictly shrinks the worst gap, so it terminates.
            if spec.max_hours_between is not None and spec.max_hours_between > 0:
                gap = timedelta(hours=spec.max_hours_between)
                while True:
                    timeline = sorted(set(picked))
                    violated = False
                    prev = last_on
                    for h in [*timeline, None]:
                        # the open stretch from prev to h (or to the bucket's end)
                        end = h if h is not None else bhours[-1] + timedelta(hours=1)
                        if prev is not None and (end - prev) > gap:
                            window = [
                                x for x in bhours
                                if prev < x < end and x not in picked
                            ]
                            if window:
                                picked.append(min(window, key=lambda x: hour_price[x]))
                                violated = True
                                break
                        if h is not None:
                            prev = h
                    if not violated:
                        break
            chosen.update(picked)
            if picked:
                last_on = max(picked)

        out[spec.id] = {
            i: spec.power_kw for h in chosen for i in hour_slots[h]
        }
    return out
