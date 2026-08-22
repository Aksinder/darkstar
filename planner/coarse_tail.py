"""
Coarse-tail horizon: full look-ahead, fewer decision variables.

The MILP's cost is superlinear in slot count. Measured on the live 2026-08-22
instance (three tanks, two EVs, two pumps, real prices):

    96 slots / 24 h ->  3.5 s      176 slots / 44 h -> 15.4 s
    128 slots / 32 h ->  6.9 s      192 slots / 48 h -> 25.6 s

Doubling the horizon costs seven times the time. The box is ~15x slower than the
dev machine, which put the 04:00 instance (176 slots — the horizon is LONGEST in
the small hours, since tomorrow's prices arrive at 13:00 the day before) straight
through the 240 s budget: 23 timeouts on 2026-08-22, all between 04:00 and 08:59,
each one throwing the plan away and keeping a staler one. Nothing else measured
as a driver — pumps, EVs, load_groups and the water need were all inside noise.

The fix keeps the whole look-ahead and drops RESOLUTION beyond the first day:
15-minute slots for `fine_hours`, hourly after that. Standard receding-horizon
practice — the far end is re-solved long before it is executed, so its resolution
buys nothing. Same instance, 44 h: 116 variables, 4.4 s. At 48 h: 120 variables,
1.0 s against 25.6.

The result is expanded back to the original 15-minute grid before it leaves here,
so every consumer downstream — the dataframe join, the executor, the schedule
file — sees exactly the slot list it always saw.

Pure: no I/O, no config.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

from planner.solver.types import KeplerInputSlot, KeplerResultSlot

if TYPE_CHECKING:
    from planner.solver.types import KeplerResult


def coarsen_slots(
    slots: list[KeplerInputSlot], fine_hours: float
) -> tuple[list[KeplerInputSlot], list[list[int]]]:
    """Merge slots beyond ``fine_hours`` into hourly ones.

    Returns ``(coarse_slots, groups)`` where ``groups[i]`` lists the ORIGINAL slot
    indices that coarse slot ``i`` covers — the map expand_result() reverses.

    Energies (load, pv) are SUMMED and prices AVERAGED: a merged slot must present
    the same total energy at the same average price, or the solver would optimise a
    different day than the one that is coming. Only whole clock hours merge; a
    partial hour at either edge stays as-is rather than becoming a slot whose
    duration lies about what it covers.

    ``fine_hours <= 0`` or a horizon shorter than it: everything stays fine, and the
    identity mapping makes the caller's expand step a no-op.
    """
    if not slots or fine_hours <= 0:
        return list(slots), [[i] for i in range(len(slots))]

    boundary = slots[0].start_time + timedelta(hours=fine_hours)
    coarse: list[KeplerInputSlot] = []
    groups: list[list[int]] = []

    i = 0
    n = len(slots)
    while i < n:
        s = slots[i]
        if s.start_time < boundary:
            coarse.append(s)
            groups.append([i])
            i += 1
            continue
        # Collect the rest of this clock hour.
        hour = s.start_time.replace(minute=0, second=0, microsecond=0)
        j = i
        members: list[int] = []
        while j < n and slots[j].start_time.replace(
            minute=0, second=0, microsecond=0
        ) == hour:
            members.append(j)
            j += 1
        if len(members) == 1:
            coarse.append(slots[members[0]])
            groups.append(members)
            i = j
            continue
        first, last = slots[members[0]], slots[members[-1]]
        coarse.append(
            KeplerInputSlot(
                start_time=first.start_time,
                end_time=last.end_time,
                load_kwh=sum(slots[k].load_kwh for k in members),
                pv_kwh=sum(slots[k].pv_kwh for k in members),
                import_price_sek_kwh=sum(
                    slots[k].import_price_sek_kwh for k in members
                ) / len(members),
                export_price_sek_kwh=sum(
                    slots[k].export_price_sek_kwh for k in members
                ) / len(members),
            )
        )
        groups.append(members)
        i = j
    return coarse, groups


def _split_slot(
    coarse: KeplerResultSlot, originals: list[KeplerInputSlot]
) -> list[KeplerResultSlot]:
    """One coarse result slot back into its original quarter-hours.

    ENERGIES are divided evenly and POWERS carried through unchanged — the two obey
    different arithmetic, and mixing them up is how an expanded plan starts lying.
    A 2 kWh hourly charge becomes four 0.5 kWh quarters (the same energy); a 3.4 kW
    heater stays 3.4 kW in each (the same power, for the same total duration).

    ``soc_kwh`` is the END-of-slot state, so it is interpolated across the members
    rather than repeated: a repeated value would draw a flat line through an hour
    the battery actually spent charging.
    """
    k = len(originals)
    if k <= 1:
        return [replace(coarse, start_time=originals[0].start_time,
                        end_time=originals[0].end_time)]
    share = 1.0 / k
    out: list[KeplerResultSlot] = []
    soc_start = coarse.soc_kwh - (coarse.charge_kwh - coarse.discharge_kwh)
    for n, orig in enumerate(originals, start=1):
        out.append(
            replace(
                coarse,
                start_time=orig.start_time,
                end_time=orig.end_time,
                charge_kwh=coarse.charge_kwh * share,
                discharge_kwh=coarse.discharge_kwh * share,
                grid_import_kwh=coarse.grid_import_kwh * share,
                grid_export_kwh=coarse.grid_export_kwh * share,
                cost_sek=coarse.cost_sek * share,
                soc_kwh=soc_start
                + (coarse.charge_kwh - coarse.discharge_kwh) * share * n,
            )
        )
    return out


def expand_result(
    result: KeplerResult,
    groups: list[list[int]],
    original_slots: list[KeplerInputSlot],
) -> KeplerResult:
    """Put a coarse-tail result back on the original 15-minute grid.

    Everything downstream — the dataframe join (which asserts equal length), the
    schedule file, the executor — is entitled to the grid it handed in. Coarsening
    is an internal solver optimisation and must not leak past this module.
    """
    if len(result.slots) != len(groups):
        # Solver returned a different shape than we asked for: hand back what it
        # said rather than inventing an alignment. Loud upstream, not silent here.
        return result
    expanded: list[KeplerResultSlot] = []
    for coarse_slot, members in zip(result.slots, groups, strict=True):
        expanded.extend(_split_slot(coarse_slot, [original_slots[m] for m in members]))
    return replace(result, slots=expanded)
