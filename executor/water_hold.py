"""
Idle-hold for water heaters with their OWN thermostat (the spa).

A tank is a dumb resistive load: "plan says off" means "write temp_off" and the
element stops. The spa is not — it is a thermostatted appliance whose set-point
Darkstar merely nudges (input_number -> bridge -> climate heat/fan_only). Writing
fan_only there does not just stop a heat block, it throws away the standing warmth
the block bought, so the next bath starts from cold.

Owner rule (2026-08-15): "we don't need to TURN OFF the spa if it isn't drawing
anything — that helps it stay warm when we have surplus — but we DO turn off if
it's running without surplus, or the price is high."

So the off-write is suppressed only while it is genuinely free to leave alone:

    hold  <=>  price <= ceiling  AND  (load is idle  OR  we are exporting)

Everything else writes OFF exactly as before. Note what this does NOT do: it never
turns the spa ON. A cold spa stays cold until the planner books it a block — the
hold only preserves warmth that a block already paid for.

Fails CLOSED: an unreadable power or grid sensor writes OFF. Blind must not mean
"leave a 1.8 kW element running at the evening peak".
"""

from __future__ import annotations

# Below this the load counts as not heating (the spa's circulation pump alone).
DEFAULT_IDLE_POWER_W = 100.0
# Grid must be exporting at least this much to count as surplus. A margin, not 0:
# meter noise around zero is not surplus, and the spa's own 1.8 kW is already in
# the reading — a spa heating on true surplus still shows net export.
DEFAULT_SURPLUS_MARGIN_W = 200.0


def should_hold_off_write(
    *,
    power_w: float | None,
    grid_w: float | None,
    price_sek_kwh: float | None,
    idle_power_w: float = DEFAULT_IDLE_POWER_W,
    max_price_sek_kwh: float | None = None,
    surplus_margin_w: float = DEFAULT_SURPLUS_MARGIN_W,
) -> tuple[bool, str]:
    """
    Decide whether to SKIP the planned off-write for a self-thermostatted heater.

    Args:
        power_w: measured load, W. None = unreadable.
        grid_w: signed grid power, W, POSITIVE = import (the house convention).
            None = unreadable.
        price_sek_kwh: current import price. None = unknown (treated as unknown,
            not as free — see below).
        idle_power_w: at or below this the load counts as idle.
        max_price_sek_kwh: hold ceiling. None = no ceiling (price never forces off).
        surplus_margin_w: export must exceed this to count as surplus.

    Returns:
        (hold, reason). hold=True means "do not write; leave the appliance alone".
    """
    # Price ceiling wins over everything: at 2.5 kr/kWh a self-topping-up spa is
    # exactly what the owner asked us to stop, idle right now or not.
    if max_price_sek_kwh is not None:
        if price_sek_kwh is None:
            return False, "price unknown"
        if price_sek_kwh > max_price_sek_kwh:
            return False, f"price {price_sek_kwh:.2f} > {max_price_sek_kwh:.2f}"

    if power_w is None:
        return False, "power sensor unreadable"

    if power_w <= idle_power_w:
        return True, f"idle ({power_w:.0f}W)"

    # It IS drawing. Only surplus justifies letting it continue.
    if grid_w is None:
        return False, "grid sensor unreadable"
    if grid_w <= -surplus_margin_w:
        return True, f"heating on surplus (export {-grid_w:.0f}W)"

    return False, f"drawing {power_w:.0f}W without surplus (grid {grid_w:+.0f}W)"
