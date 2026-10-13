"""Owner EV-priority and one-off-departure awareness for the planner.

Two Home Assistant helpers existed and were honored only by the real-time servo,
and only weakly (live 2026-08-27, both set the evening before, Tesla still at
41 % at departure):

* ``input_select.darkstar_ev_priority`` ordered nothing but the daytime
  SURPLUS distribution — the planner's per-kWh incentive ladders
  (``penalty_levels``) never saw it, so the FMB kept outbidding the Tesla for
  every cheap night slot.
* ``input_datetime.darkstar_tesla_departure`` only drove the servo's low
  guarantee band (floor_soc 40), so "departure 07:25" meant "40 % by 07:25".

This module gives both helpers their intended planner-side meaning:

**Priority re-weighting** — when the select is not ``auto``, every
non-preferred charger's bucket values are clamped BELOW the preferred
charger's lowest band (minus an epsilon). The preferred car then wins every
slot both cars want; the demoted car still charges from surplus or genuinely
free energy. ``auto`` leaves the configured ladders untouched.

**One-off departure** — a user-set future departure inside the horizon lifts
the charger's top (urgent) band up to the departure SoC target, so the MILP
books cheap slots toward the trip target ahead of time instead of leaving the
work to the servo's dawn ramp. Best effort by construction: the band is an
incentive, and the servo's grid-backed floor remains the hard guarantee.

Pure functions; the pipeline reads the HA entities and passes plain values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planner.solver.types import EVChargerInput

logger = logging.getLogger("darkstar.ev_priority")

# The demoted car must sit strictly below the preferred car's ladder, but a
# zero value would make it invisible even to free surplus in the plan.
_DEMOTION_EPSILON_SEK = 0.05
_DEMOTION_FLOOR_SEK = 0.05


def apply_priority_order(
    chargers: list[EVChargerInput],
    order: list[str] | None,
) -> list[str]:
    """Clamp non-preferred chargers' bucket values below the preferred car's.

    ``order`` is the priority_orders mapping for the select's current state
    (e.g. ``["tesla", "easee_fmb"]``), or None for auto/unmapped — a no-op.
    Returns human-readable log lines describing what changed.
    """
    if not order:
        return []
    by_id = {c.id: c for c in chargers}
    preferred = next((by_id[i] for i in order if i in by_id), None)
    if preferred is None or not preferred.incentive_buckets:
        return []
    ceiling = round(
        max(
            _DEMOTION_FLOOR_SEK,
            min(b.value_sek for b in preferred.incentive_buckets) - _DEMOTION_EPSILON_SEK,
        ),
        4,
    )
    notes: list[str] = []
    for c in chargers:
        if c.id == preferred.id or not c.incentive_buckets:
            continue
        changed = False
        for b in c.incentive_buckets:
            if b.value_sek > ceiling:
                b.value_sek = ceiling
                changed = True
        if changed:
            notes.append(
                f"EV priority: {c.id} demoted below {preferred.id} "
                f"(bucket values capped at {ceiling:.2f} SEK/kWh)"
            )
    return notes


def apply_departure_target(
    chargers: list[EVChargerInput],
    charger_id: str,
    departure_in_hours: float | None,
    target_soc: float | None,
    horizon_hours: float,
) -> list[str]:
    """Lift the charger's top band to the departure target for a live one-off trip.

    Only a FUTURE departure inside the plan horizon counts — a stale entity
    (past date) is inert, exactly like the servo treats it.
    """
    if (
        departure_in_hours is None
        or departure_in_hours <= 0.0
        or departure_in_hours > horizon_hours
        or target_soc is None
        or target_soc <= 0.0
    ):
        return []
    ch = next((c for c in chargers if c.id == charger_id), None)
    if ch is None or not ch.incentive_buckets:
        return []
    top = min(ch.incentive_buckets, key=lambda b: b.threshold_soc)
    if target_soc <= top.threshold_soc:
        return []
    old = top.threshold_soc
    top.threshold_soc = float(target_soc)
    return [
        f"EV departure: {charger_id} leaves in {departure_in_hours:.1f} h — urgent band "
        f"raised {old:.0f}% -> {target_soc:.0f}% (value {top.value_sek:.2f} SEK/kWh)"
    ]


def live_floor_soc(configured: float | None, live: float | None) -> float | None:
    """The owner's live minimum-level helper, falling back to the config constant.

    An unreadable helper (None — HA restarting, entity renamed) or a value outside
    0-100 falls back to the configured ``floor_soc``, so a broken helper can never
    remove or inflate the guarantee. 0 is a legal choice: "no guaranteed level".
    """
    if live is None or not 0.0 <= live <= 100.0:
        return configured
    return float(live)


def track_floor_band(
    band_soc: float | None, configured_floor: float | None, live_floor: float | None
) -> float | None:
    """Where a guarantee band's upper SoC sits once the owner moves the floor.

    Shared by the planner (first incentive band) and the servo (plan-floor gate),
    which must agree or the plan books slots the servo then refuses to back.
    A band that IS the floor band (Tesla: band 40 == floor_soc 40) follows the
    helper both ways. A band above the floor (FMB: comfort band 86, floor 40) is a
    different number and only ever rises — lowering the minimum level must not
    shrink the comfort band it sits inside.
    """
    if band_soc is None or live_floor is None:
        return band_soc
    if configured_floor is not None and abs(band_soc - configured_floor) < 1e-6:
        return float(live_floor)
    return max(float(band_soc), float(live_floor))


def apply_floor_target(
    chargers: list[EVChargerInput],
    charger_id: str,
    configured_floor: float | None,
    live_floor: float | None,
) -> list[str]:
    """Move the charger's first band with the owner's minimum-level helper.

    Runs BEFORE the one-off departure lift, which may raise the band further.
    Later bands are lifted to at least the first so the ladder stays monotonic.
    """
    ch = next((c for c in chargers if c.id == charger_id), None)
    if ch is None or not ch.incentive_buckets or live_floor is None:
        return []
    top = min(ch.incentive_buckets, key=lambda b: b.threshold_soc)
    old = top.threshold_soc
    new = track_floor_band(old, configured_floor, live_floor)
    if new is None or abs(new - old) < 1e-6:
        return []
    top.threshold_soc = new
    for b in ch.incentive_buckets:
        if b.threshold_soc < new:
            b.threshold_soc = new
    return [
        f"EV minimum level: {charger_id} guarantee band {old:.0f}% -> {new:.0f}% "
        f"(value {top.value_sek:.2f} SEK/kWh)"
    ]
