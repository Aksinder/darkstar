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

    hold  <=>  effective price <= ceiling  AND  (surplus  OR  idle)

Two definitions carry the whole rule:

**Surplus is not the same as export.** On a sunny morning here the grid meter reads
about zero while the battery soaks 8 kW of PV — physically there is plenty of spare
energy, it is just being stored rather than sold. Reading the meter alone would call
that "no surplus" and kill the spa on the sunniest hour of the day. So surplus means
exporting OR charging the battery with at least the heater's own draw to spare.

**The effective price follows the surplus.** Spare PV costs what we would have got
for it (the export price), not what we would have paid for grid (the import price).
Comparing a sunlit hour against the 1.20 import price would fail every summer noon.
Without surplus the marginal kWh really is bought, so the import price applies.

Everything else writes OFF exactly as before. Note what this does NOT do: it never
turns the spa ON. A cold spa stays cold until the planner books it a block — the
hold only preserves warmth that a block already paid for.

Fails CLOSED: an unreadable sensor, or an unknown price under a configured ceiling,
writes OFF. Blind must not mean "leave a 1.8 kW element running at the evening peak".
"""

from __future__ import annotations

# Below this the load counts as not heating (the spa's circulation pump alone).
DEFAULT_IDLE_POWER_W = 100.0
# Grid must be exporting at least this much to count as surplus. A margin, not 0:
# meter noise around zero is not surplus.
DEFAULT_SURPLUS_MARGIN_W = 200.0


def price_percentile(prices: list[float], percentile: float) -> float | None:
    """The ``percentile``-th value of a price window, linear-interpolated (numpy type 7).

    Deliberately a ROLLING window rather than a calendar day (owner 2026-08-18: "spatak
    borde vara dynamiskt mot period och inte per dygn"). A fixed ceiling tuned in a cheap
    week silently stops working when the market moves: 0.9 SEK/kWh permitted most of
    summer, then sat below EVERY hour once spot reached 1.43 and the spa simply went cold.
    A percentile keeps the same INTENT — "only the cheap part of what is coming" —
    whatever the absolute level happens to be.

    Mirrors dynamic_wtp_from_prices() in the planner adapter; kept separate because the
    executor deliberately imports nothing from the planner.
    """
    window = sorted(p for p in prices if p is not None)
    if not window:
        return None
    if len(window) == 1:
        return window[0]
    rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(window) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(window) - 1)
    frac = rank - lo
    return window[lo] * (1.0 - frac) + window[hi] * frac


def battery_charge_w(
    *, servo_signed_w: float | None, house_signed_w: float | None
) -> float | None:
    """
    Normalise battery power to POSITIVE = charging.

    Two conventions live side by side in this codebase and they disagree in SIGN:
    the servo's ``executor.ev_surplus.battery_power_entity`` is documented "+ charge"
    (sensor.battery_charging_power_signed = +4797 while charging), whereas
    ``input_sensors.battery_power`` is the inverter's own convention, NEGATIVE while
    charging (sensor.battery_power = -4797 at the same instant) — see recorder.py,
    which derives batt_charge_kw as ``abs(battery_kw) if battery_kw < 0``.

    Reading the house sensor with the servo's convention silently inverts the surplus
    test, so a battery soaking 8 kW of PV reads as -8000 and never counts as surplus —
    killing the hold on precisely the sunny hours it exists for. Prefer the servo's
    entity; fall back to the house sensor with its sign flipped.
    """
    if servo_signed_w is not None:
        return servo_signed_w
    if house_signed_w is None:
        return None
    return -house_signed_w


def _has_surplus(
    grid_w: float | None,
    battery_w: float | None,
    heater_w: float,
    surplus_margin_w: float,
    own_draw_w: float = 0.0,
) -> bool:
    """Exporting, or storing at least this heater's draw into the battery.

    ``own_draw_w`` is subtracted from the meter first, because a heater running ON the
    surplus has already consumed it: the meter reads ~0 precisely BECAUSE the element
    is on. Counting that against itself makes the surplus vanish the moment the load
    starts, which flaps the heater between two targets. The honest question is "would
    there be surplus if this heater were off", and that is grid minus its own draw.
    (Same failure the EV servo had: never let your own actuation read as scarcity.)
    """
    # A discharging battery settles it: the house is already drawing down storage, so
    # the marginal kWh this heater eats comes from the battery (or, under a forced
    # export, from energy we chose to sell) — never from spare production. Checked
    # BEFORE the own-draw credit, which would otherwise turn "battery covering the
    # spa" into "the spa is running on surplus".
    if battery_w is not None and battery_w <= -surplus_margin_w:
        return False
    if grid_w is not None and (grid_w - own_draw_w) <= -surplus_margin_w:
        return True
    return battery_w is not None and battery_w >= max(heater_w, surplus_margin_w)


def should_hold_off_write(
    *,
    power_w: float | None,
    grid_w: float | None,
    battery_w: float | None = None,
    import_price_sek_kwh: float | None,
    export_price_sek_kwh: float | None = None,
    heater_power_w: float = 0.0,
    idle_power_w: float = DEFAULT_IDLE_POWER_W,
    max_price_sek_kwh: float | None = None,
    surplus_margin_w: float = DEFAULT_SURPLUS_MARGIN_W,
) -> tuple[bool, str]:
    """
    Decide whether to SKIP the planned off-write for a self-thermostatted heater.

    Args:
        power_w: measured load, W. None = unreadable.
        grid_w: signed grid power, W, POSITIVE = import (the house convention).
        battery_w: signed battery power, W, POSITIVE = charging.
        import_price_sek_kwh: current import price; None = unknown.
        export_price_sek_kwh: current export price; None falls back to import.
        heater_power_w: this heater's rated draw, used to size the surplus test.
        idle_power_w: at or below this the load counts as idle.
        max_price_sek_kwh: hold ceiling. None = no ceiling (price never forces off).
        surplus_margin_w: export/charge must exceed this to count as surplus.

    Returns:
        (hold, reason). hold=True means "do not write; leave the appliance alone".
    """
    own = power_w or 0.0
    surplus = _has_surplus(grid_w, battery_w, heater_power_w, surplus_margin_w, own)

    if max_price_sek_kwh is not None:
        # Spare PV costs the export revenue foregone; bought energy costs import.
        price = export_price_sek_kwh if surplus else import_price_sek_kwh
        if price is None:
            price = import_price_sek_kwh
        if price is None:
            return False, "price unknown"
        if price > max_price_sek_kwh:
            return False, f"price {price:.2f} > {max_price_sek_kwh:.2f}"

    if surplus:
        return True, "surplus available"

    if power_w is None:
        return False, "power sensor unreadable"
    if power_w <= idle_power_w:
        return True, f"idle ({power_w:.0f}W)"

    return False, f"drawing {power_w:.0f}W without surplus"


def should_boost_on_surplus(
    *,
    power_w: float | None,
    grid_w: float | None,
    battery_w: float | None,
    import_price_sek_kwh: float | None,
    export_price_sek_kwh: float | None,
    heater_power_w: float,
    max_price_sek_kwh: float | None,
    heated_today_kwh: float | None = None,
    absorb_cap_kwh_per_day: float | None = None,
    surplus_margin_w: float = DEFAULT_SURPLUS_MARGIN_W,
) -> tuple[bool, str]:
    """
    Should this heater be pushed to its BOOST target right now?

    Owner rule (2026-08-15): "spa borde tryckas upp till 40 om det finns överskott och
    elpriset inte är dyrt." The planner cannot express this here — its boost path needs
    excess_pv_sink == "water_heater_boost", which is deliberately off, being the
    mechanism behind the 36 kWh/day phantom-water incident. So the decision is made
    from MEASURED surplus each tick instead of a forecast, which is both truer to the
    request and self-limiting: the boost lasts exactly as long as the surplus does.

    Unlike the idle-hold this DOES command heat, so it carries the daily energy bound
    the planner would otherwise have enforced (absorb_cap_kwh_per_day).

    Returns (boost, reason).
    """
    own = power_w or 0.0
    if not _has_surplus(grid_w, battery_w, heater_power_w, surplus_margin_w, own):
        return False, "no surplus"

    if max_price_sek_kwh is not None:
        price = export_price_sek_kwh
        if price is None:
            price = import_price_sek_kwh
        if price is None:
            return False, "price unknown"
        if price > max_price_sek_kwh:
            return False, f"price {price:.2f} > {max_price_sek_kwh:.2f}"

    if (
        absorb_cap_kwh_per_day is not None
        and heated_today_kwh is not None
        and heated_today_kwh >= absorb_cap_kwh_per_day
    ):
        return False, f"daily cap reached ({heated_today_kwh:.1f} kWh)"

    return True, "surplus + cheap"


# Appliance states that mean "the element may run". A climate bridge reports "heat";
# a relay-style appliance reports "on".
DEFAULT_HEATING_STATES = ("heat", "on", "heating", "true")


def detect_appliance_drift(
    *,
    intended_temp: float | None,
    off_temp: float | None,
    appliance_state: str | None,
    appliance_setpoint_c: float | None = None,
    power_w: float | None = None,
    idle_power_w: float = DEFAULT_IDLE_POWER_W,
    heating_states: tuple[str, ...] = DEFAULT_HEATING_STATES,
    setpoint_tolerance_c: float = 0.5,
) -> tuple[bool, str]:
    """
    Has the appliance drifted away from what we intend, so that we must re-assert?

    Darkstar drives a thermostatted heater through a helper: it writes
    ``input_number.<x>_target_temp`` and an HA bridge relays that to the climate
    entity. The write is change-gated against THE HELPER, so once the helper reads 20
    nothing more is sent — and if anything else moves the appliance (the tub's own app
    or panel, a device reboot), Darkstar never notices. Observed live 2026-08-15: the
    helper sat at 20 from 13:12 while the spa ran heat/38 from 13:18 with the element
    drawing 1.8 kW, and no tick corrected it. The switch path has always self-healed
    because it compares the RELAY; this gives the bridge path the same property.

    Two signals, because either alone can lie: the reported mode can lag the device,
    and power alone cannot distinguish "heating" from a circulation pump. Either one
    showing heat while we intend off is enough to re-assert — a false re-assert costs
    one idempotent service call, a missed one costs a 1.8 kW element at the evening peak.

    Returns (drifted, reason).
    """
    if intended_temp is None or off_temp is None:
        return False, "nothing intended"

    state = (appliance_state or "").strip().lower()
    known = bool(state) and state not in ("unknown", "unavailable")
    drawing = power_w is not None and power_w > idle_power_w
    heating = (known and state in heating_states) or drawing

    intended_off = float(intended_temp) <= float(off_temp)

    if intended_off:
        if heating:
            why = state if known and state in heating_states else f"{power_w:.0f}W"
            return True, f"appliance heating ({why}) while we intend off"
        return False, "off as intended"

    # We intend HEAT. A device sitting idle means our target never landed.
    if known and state not in heating_states:
        return True, f"appliance {state} while we intend {intended_temp:.0f}C"
    if (
        appliance_setpoint_c is not None
        and abs(float(appliance_setpoint_c) - float(intended_temp)) > setpoint_tolerance_c
    ):
        return True, (
            f"setpoint {appliance_setpoint_c:.0f}C != intended {intended_temp:.0f}C"
        )
    return False, "heating as intended"
