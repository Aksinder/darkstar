"""
Water heaters as the fuse guard's third shed lever.

The S4 guard protects the HOUSE main: 25 A per phase, minus a margin. Nothing else is
at risk here — the cars sit on the house feed, not the villavagn's, so a car can only
ever trip the main and only by pushing one or more phases past 25 A (owner, 2026-08-18).

Until now the guard could clamp cars and cap battery charging, but never a water heater.
That ordering is backwards under scarcity: a tank is happy to wait an hour, a car may be
leaving in the morning. Shedding the car to protect a phase that a 3.4 kW element is
sitting on spends the expensive option to save the cheap one.

DELIBERATELY NOT fail-safe-to-shed. The EV layer already stops every car on blind
sensors — the larger load, and one that can catch up later. Cutting hot water on a
sensor hiccup is a worse trade than the marginal trip risk that remains once the cars
are already off, so unreadable phases here mean "leave the heater alone".
"""

from __future__ import annotations

from .ev_surplus import _export_credit_a


def should_shed_for_fuse(
    *,
    phase_currents_a: dict[str, float] | None,
    budget_a: float | None,
    heater_phases: tuple[str, ...] = (),
    grid_w: float = 0.0,
    voltage_v: float = 230.0,
) -> tuple[bool, str]:
    """
    Must this heater be forced OFF to keep the main fuse inside its budget?

    Args:
        phase_currents_a: measured per-phase magnitudes. None/empty = blind.
        budget_a: limit minus margin. None = the guard is off.
        heater_phases: which phases this heater draws on. EMPTY = unknown, which counts
            against EVERY phase — the same conservative convention the EV phase_map
            uses, because an unmapped load could be on the one that is overloaded.
        grid_w: signed grid power, used for the export credit.

    Returns:
        (shed, reason).
    """
    if budget_a is None:
        return False, "guard off"
    if not phase_currents_a:
        return False, "phase sensors unreadable"

    # The meter is direction-blind, so a big reading during EXPORT is not an overload:
    # added consumption removes it 1:1. Same credit the EV clamp and battery cap use.
    credit = _export_credit_a(grid_w, voltage_v)
    watched = (
        {p: i for p, i in phase_currents_a.items() if p in heater_phases}
        if heater_phases
        else phase_currents_a
    )
    if not watched:
        return False, "no watched phase"

    worst_p, worst_a = max(
        ((p, max(0.0, abs(i) - credit)) for p, i in watched.items()), key=lambda x: x[1]
    )
    if worst_a > budget_a:
        return True, f"phase {worst_p} at {worst_a:.1f} A over {budget_a:.1f} A budget"
    return False, f"worst phase {worst_a:.1f} A within {budget_a:.1f} A"
