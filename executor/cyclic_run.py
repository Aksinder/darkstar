"""
Opportunistic run gates for cyclic loads (the pool pump and the bog filter).

The planner already guarantees each pump its daily floor and books it into the
cheapest hours it can find. This module answers a different question, one the
planner cannot: *may this pump run RIGHT NOW, on top of its plan?*

Owner rule for the bog filter (2026-08-19): "6 timmar, far ocksa kora vid
overskott om det inte ar extra dyra timmar samt nar vi ar hemma om utrymme
finns." Three clauses, three mechanisms:

* **6 timmar** is the planner's floor (min_kwh_per_day). Not this module's job.
* **vid overskott** is measured, not forecast -- the same surplus semantics the
  spa uses, so a battery soaking PV counts and the pump's own draw is credited
  back. Forecasting it would repeat the phantom-water mistake.
* **nar vi ar hemma** is a presence gate, deliberately held to a TIGHTER price
  ceiling than the surplus gate: surplus energy is nearly free, company is not.

"Om utrymme finns" is the load-bearing half of the request, and it means two
different rooms at once -- room in the price (this is a nice-to-have, so it
stands down when the hour is dear) and room in the day's budget
(max_extra_hours_per_day). Without the second, a week at home with sun would
simply run the pump around the clock, which is not what "om utrymme finns"
means in any reading.

These gates only ever turn a pump ON. They can neither cancel a planned block
nor override the fuse guard, the control pause, or a human's force_off -- the
caller applies them last, below everything that can say no.

Fails CLOSED in both directions that matter: an unknown price under a
configured ceiling, or an empty price window when a percentile ceiling is
configured, yields NO extra run. An opportunistic luxury is exactly the thing
to skip when we cannot see what it costs.
"""

from __future__ import annotations

from executor.water_hold import (
    DEFAULT_SURPLUS_MARGIN_W,
    has_surplus,
    price_percentile,
)

# Presence states that count as "someone is home".
HOME_STATES = ("home", "on", "true")


def anyone_home(states: list[str | None]) -> bool | None:
    """Is at least one tracked person home? None when nothing is readable.

    None is distinct from False on purpose: a device tracker that has gone
    unavailable must not read as "nobody home and also nobody away" -- the caller
    treats None as "cannot tell" and declines the presence gate, rather than
    inventing an answer from a dead sensor.
    """
    known = [s for s in states if s is not None and str(s).strip().lower() not in
             ("", "unknown", "unavailable")]
    if not known:
        return None
    return any(str(s).strip().lower() in HOME_STATES for s in known)


def _ceiling(
    window: list[float], percentile: float | None
) -> tuple[float | None, bool]:
    """(ceiling, configured). A configured-but-uncomputable ceiling blocks the run."""
    if percentile is None:
        return None, False
    return price_percentile(window, percentile), True


def should_run_opportunistically(
    *,
    plan_wants_on: bool,
    power_w: float | None,
    grid_w: float | None,
    battery_w: float | None,
    load_power_w: float,
    import_price_sek_kwh: float | None,
    export_price_sek_kwh: float | None,
    import_price_window: list[float],
    export_price_window: list[float] | None = None,
    surplus_run: bool = False,
    max_price_percentile: float | None = None,
    presence_home: bool | None = None,
    presence_max_price_percentile: float | None = None,
    extra_hours_today: float = 0.0,
    max_extra_hours_per_day: float | None = None,
    surplus_margin_w: float = DEFAULT_SURPLUS_MARGIN_W,
) -> tuple[bool, str]:
    """
    May this cyclic load run on top of its plan right now?

    Args:
        plan_wants_on: the planner already wants it on -- then this is not an
            EXTRA run at all and the whole question is moot.
        power_w: this load's measured draw, W. None = unreadable.
        grid_w: signed grid power, W, POSITIVE = import.
        battery_w: signed battery power, W, POSITIVE = charging.
        load_power_w: rated draw, used to size the surplus test.
        import_price_window: rolling import prices, for the no-surplus case.
        export_price_window: rolling export prices. A percentile is only meaningful
            against the series the compared price came FROM, and under surplus the
            effective price is the EXPORT price -- which runs about a krona below
            import here, so measuring it against an import percentile would clear
            almost any ceiling and the gate would quietly stop gating. Absent, with
            a percentile configured, the surplus gate DECLINES rather than borrowing
            the import series: substituting a series a krona too high is exactly the
            silent permissiveness this argument exists to prevent.
        surplus_run: enable the surplus gate.
        max_price_percentile: ceiling for the surplus gate ("inte extra dyra
            timmar" -- a HIGH percentile: block only the dearest tail).
        presence_home: True/False/None from anyone_home().
        presence_max_price_percentile: ceiling for the presence gate. Tighter
            than the surplus one, because company does not pay for the kWh.
        extra_hours_today: opportunistic run-time already spent this day.
        max_extra_hours_per_day: budget for it. None = no budget configured,
            which disables BOTH gates rather than uncapping them.

    Returns:
        (run, reason).
    """
    if plan_wants_on:
        return False, "plan already wants it on"

    if not surplus_run and presence_max_price_percentile is None:
        return False, "no opportunistic gate configured"

    # The budget is checked first: it is the cheapest test and the one that makes
    # "om utrymme finns" mean something. An unset budget is treated as zero room,
    # never as unlimited -- forgetting to set a cap must not silently buy a pump
    # that runs around the clock.
    if max_extra_hours_per_day is None:
        return False, "no extra-hours budget configured"
    if extra_hours_today >= max_extra_hours_per_day:
        return False, (
            f"extra budget spent ({extra_hours_today:.1f}/"
            f"{max_extra_hours_per_day:.1f} h)"
        )

    own = power_w or 0.0
    surplus = has_surplus(grid_w, battery_w, load_power_w, surplus_margin_w, own)
    # Spare PV costs the export revenue foregone; bought energy costs import. The
    # window follows the price, so like is always compared with like.
    if surplus and export_price_sek_kwh is not None:
        price = export_price_sek_kwh
        window = export_price_window or []
    else:
        price = import_price_sek_kwh
        window = import_price_window

    if surplus_run and surplus:
        cap, configured = _ceiling(window, max_price_percentile)
        if not configured:
            return True, "surplus"
        if cap is None:
            return False, "surplus but no price window"
        if price is None:
            return False, "surplus but price unknown"
        if price <= cap:
            return True, f"surplus, price {price:.2f} <= P{max_price_percentile:.0f} {cap:.2f}"
        return False, f"surplus but price {price:.2f} > {cap:.2f}"

    if presence_max_price_percentile is not None and presence_home:
        # Presence without surplus means bought energy, so the import series applies.
        cap, _ = _ceiling(window, presence_max_price_percentile)
        if cap is None:
            return False, "home but no price window"
        if price is None:
            return False, "home but price unknown"
        if price <= cap:
            return True, (
                f"home, price {price:.2f} <= P{presence_max_price_percentile:.0f} {cap:.2f}"
            )
        return False, f"home but price {price:.2f} > {cap:.2f}"

    if presence_home is None and presence_max_price_percentile is not None:
        return False, "presence unreadable"
    return False, "no surplus, nobody home"
