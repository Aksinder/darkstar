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

``mode: window`` loads (room heaters) are not scheduled at all — they are on while a
window is active — and are only BOOKED here: window_slot_kw, and
window_plan_with_price_rules when the load has a price rule.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executor.window_price import PriceRules, RuleState


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


def window_slot_kw(
    slot_bounds: list[tuple[datetime, datetime]],
    windows: list[tuple[datetime, datetime]],
    power_kw: float,
) -> dict[int, float]:
    """Fixed load of a ``mode: window`` load (a room heater): ``{slot_index: kw}``.

    Such a load has no daily need to place — it is simply ON while one of its
    windows (calendar events) is active — so there is nothing to optimise: every
    slot overlapping a window carries ``power_kw``, pro-rated by the share of the
    slot the window covers (the slot's AVERAGE kW, so kW x slot hours is the energy
    booked). Slots outside every window are absent from the result; the pipeline
    writes them as an explicit 0.0.

    ``windows`` must be timezone-aware. Slots may be aware or naive, as in
    :func:`preschedule_cyclic_loads`.
    """
    out: dict[int, float] = {}
    if power_kw <= 0 or not slot_bounds or not windows:
        return out
    reference = slot_bounds[0][0]
    spans: list[tuple[datetime, datetime]] = []
    for w_start, w_end in windows:
        a = _as_naive_like(w_start, reference)
        b = _as_naive_like(w_end, reference)
        if a is not None and b is not None and b > a:
            spans.append((a, b))
    for i, (s_start, s_end) in enumerate(slot_bounds):
        length = (s_end - s_start).total_seconds()
        if length <= 0:
            continue
        covered = 0.0
        for a, b in spans:
            overlap = (min(b, s_end) - max(a, s_start)).total_seconds()
            if overlap > 0:
                covered += overlap
        if covered > 0:
            out[i] = power_kw * min(1.0, covered / length)
    return out


def window_plan_with_price_rules(
    slot_bounds: list[tuple[datetime, datetime]],
    prices: list[float],
    windows: list[tuple[datetime, datetime]],
    power_kw: float,
    rules: PriceRules,
    state: RuleState | None = None,
) -> dict[int, float]:
    """:func:`window_slot_kw` with the load's price rules applied (cheap early start,
    peak off — executor/window_price.py): ``{slot_index: kw}``.

    The executor decides these rules live, from the current price. So that the solver
    sees the load it will really get, the SAME decision function is walked forward
    over the horizon here with the forecast price of each slot, carrying the
    early-start latch and the peak-off max/min-on clocks from slot to slot:

    * a slot the heater is latched on early in is booked at ``power_kw`` although no
      window covers it;
    * a slot it is peak-off in is booked 0 (absent from the result) although a window
      covers it.

    ``windows`` already start ``preheat_minutes`` early, so "when the plug would next
    come on" is simply the next window start. ``state`` is where the plug stands now
    (from the executor's status sensor); without it, a heater inside a window at slot
    0 is taken to have just come on, which books load rather than a peak-off.
    """
    # Imported here, like the adapter's executor.config import: the executor package
    # pulls in the whole engine, and this module is otherwise dependency-free.
    from executor.window_price import PHASE_EARLY, RuleState, simulate

    cover = window_slot_kw(slot_bounds, windows, 1.0)
    if power_kw <= 0 or not slot_bounds:
        return {}
    if not rules.enabled:
        return {i: power_kw * share for i, share in cover.items()}

    reference = slot_bounds[0][0]
    win_starts = sorted(
        w for w in (_as_naive_like(start, reference) for start, _ in windows) if w is not None
    )
    starts = [s for s, _ in slot_bounds]
    in_window = [i in cover for i in range(len(starts))]
    next_on = [next((w for w in win_starts if w > s), None) for s in starts]
    if state is None:
        state = RuleState(heat_since=starts[0] if in_window[0] else None)
    else:
        state = RuleState(
            early_latched=state.early_latched,
            peak_off_since=_as_naive_like(state.peak_off_since, reference),
            heat_since=_as_naive_like(state.heat_since, reference),
        )
    verdicts = simulate(
        rules, state, starts=starts, prices=prices, in_window=in_window, next_on=next_on
    )
    out: dict[int, float] = {}
    latched_before = False
    for i, verdict in enumerate(verdicts):
        if verdict.heat:
            # A window that begins mid-slot covers only part of it — unless the
            # heater was already on, latched, when the slot began.
            whole = verdict.phase == PHASE_EARLY or latched_before
            out[i] = power_kw * (1.0 if whole else cover.get(i, 1.0))
        latched_before = verdict.phase == PHASE_EARLY and verdict.heat
    return out
