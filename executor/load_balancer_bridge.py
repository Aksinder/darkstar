"""
Impedance match between our runtime and upstream's load balancer.

``executor/load_balancer.py`` is ported from upstream and expects upstream's
data model: phase currents keyed by int, EV chargers described by
``EVChargerDeviceConfig``. Our fork replaced the EV layer with
``executor/ev_surplus*`` and never carried upstream's, so the decision core
transplanted cleanly but its wiring did not. This module is that wiring, kept
separate from the engine because it is pure (apart from the one HA read helper)
and therefore testable, and because mapping between two data models is exactly
where silent mistakes live.

Mismatches worth naming:

**Phase identity.** Our fuse guard keys phases by the Sungrow meter's names
("a"/"b"/"c", which ARE L1/L2/L3 in order on this site — mapped 2026-10-12) or
"l1".."l3"; upstream keys them 1/2/3. An unrecognised key is dropped rather
than guessed — a phase silently mapped to the wrong number would budget the
wrong conductor, which is worse than not guarding it.

**Staleness.** Upstream judges each phase by its own reading timestamp. We now
do too (``read_phase_samples``): each phase carries its HA ``last_reported``
(falling back to ``last_updated``/``last_changed``), so a steady sensor that
keeps reporting the same value is not mistaken for a dead one, and one dead
phase degrades only the entries on that phase. The servo's own all-or-nothing
reader is untouched — it serves the S4 fuse clamp, which stays the live guard.

**Who owns the EV current.** Upstream's balancer owns the charger setpoint and
ramps it toward the PLAN. Here the ev_surplus servo owns it (surplus charging
runs with no plan at all), so the balancer is a CAP on the servo: its target is
what the servo commanded this tick, its setpoint is what is really applied (the
servo's command, or lower while the balancer's own cap is in force). Feeding it
the plan would make it "stop" every unplanned surplus session.

**The EV-goal stack.** Upstream's 1-phase relief / re-fit inputs
(relief_available, goal_at_risk, surplus_slot, return_to_phases) are left at
their inert defaults: this fork has no phase switching to drive.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .load_balancer import (
    BATTERY_PHASES,
    BatteryBalancerInput,
    EVBalancerInput,
    LoadBalancerStatus,
    ShedLoadInput,
)

if TYPE_CHECKING:
    from .config import BalancedLoadConfig, GiveWayOrderEntry

logger = logging.getLogger(__name__)

# "a"/"l1"/"L1"/"1"/1 all mean phase 1 (Sungrow meter a/b/c = L1/L2/L3 on this
# site). Anything else is dropped, never guessed.
_PHASE_NAMES = {
    "l1": 1,
    "l2": 2,
    "l3": 3,
    "a": 1,
    "b": 2,
    "c": 3,
    "1": 1,
    "2": 2,
    "3": 3,
}


def phase_to_int(key: Any) -> int | None:
    """Map a phase key to upstream's 1/2/3, or None when it is not recognised."""
    return _PHASE_NAMES.get(str(key).strip().lower())


def phase_currents_to_int_keys(currents: dict[str, float] | None) -> dict[int, float]:
    """Our name-keyed phase currents in upstream's int-keyed form."""
    if not currents:
        return {}
    out: dict[int, float] = {}
    for key, amps in currents.items():
        phase = phase_to_int(key)
        if phase is not None:
            out[phase] = float(amps)
    return out


def phases_for_charger(phase_map: tuple[str, ...] | list[str]) -> list[int]:
    """Which phases a charger draws on.

    An explicit phase_map decides. Without one the answer is UNKNOWN — for a
    3-phase charger because it genuinely uses all three, and for a 1-phase
    charger because we do not know WHICH one. Both cases therefore count
    against every phase: the same conservative convention the EV clamp and the
    water fuse-shed already use, because an unmapped load could be sitting on
    the very phase that is overloaded.
    """
    mapped = [p for p in (phase_to_int(x) for x in phase_map or ()) if p is not None]
    return sorted(set(mapped)) if mapped else [1, 2, 3]


def _positive_int(value: Any) -> int | None:
    """A commanded current as an int, None when off/unknown (0 A = not charging)."""
    if value is None:
        return None
    try:
        amps = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amps) or amps <= 0:
        return None
    return int(amps)


def ev_entries(
    chargers: list[Any],
    *,
    setpoints_a: dict[str, float | None],
    targets_a: dict[str, float | None],
    measured_a: dict[str, float | None] | None = None,
) -> dict[str, EVBalancerInput]:
    """Build balancer inputs for the chargers we can actually throttle.

    setpoints_a: the amps really applied now (servo command, capped by any
    balancer cap in force). targets_a: the amps the servo wants this tick — the
    balancer's ceiling. None/0 means off.

    A charger without a settable current path (``controllable=False``) is an
    on/off device: it belongs in ``load_balancing.loads`` as a shed entry, not
    here, because the balancer would otherwise "throttle" it by writing a
    setpoint nothing reads.
    """
    measured = measured_a or {}
    out: dict[str, EVBalancerInput] = {}
    for c in chargers:
        if not getattr(c, "controllable", True):
            continue
        raw_measured = measured.get(c.id)
        out[c.id] = EVBalancerInput(
            charger_id=c.id,
            phases=phases_for_charger(getattr(c, "phase_map", ())),
            current_setpoint_a=_positive_int(setpoints_a.get(c.id)),
            planner_target_a=_positive_int(targets_a.get(c.id)),
            min_current_a=int(getattr(c, "min_current_a", 6) or 6),
            max_current_a=int(getattr(c, "max_current_a", 16) or 16),
            measured_a=None if raw_measured is None else float(raw_measured),
        )
    return out


def shed_entries(
    loads: list[BalancedLoadConfig],
    *,
    power_w: dict[str, float | None] | None = None,
) -> dict[str, ShedLoadInput]:
    """Build balancer inputs for the configured on/off shed-able loads.

    Restore power: the live power_sensor reading (``power_w``) when present,
    else the configured power_kw, else None (worst case — the core then waits
    for the whole phase to be free before restoring).
    """
    live = power_w or {}
    out: dict[str, ShedLoadInput] = {}
    for ld in loads:
        if not ld.device_id:
            continue
        sensor_w = live.get(ld.device_id)
        configured_kw = getattr(ld, "power_kw", None)
        if sensor_w is not None and sensor_w > 0:
            watts, source = float(sensor_w), "sensor"
        elif configured_kw:
            watts, source = float(configured_kw) * 1000.0, "configured"
        else:
            watts, source = None, "worst_case"
        out[ld.device_id] = ShedLoadInput(
            load_id=ld.device_id,
            device_type=ld.device_type.value,
            phases=sorted({p for p in (phase_to_int(x) for x in ld.phases) if p is not None}),
            power_w=watts,
            power_source=source,
            min_off_s=int(getattr(ld, "min_off_s", 0) or 0),
            min_on_s=int(getattr(ld, "min_on_s", 0) or 0),
        )
    return out


def battery_entry(
    decision: Any,
    controller: Any,
    *,
    grid_w: float | None,
    battery_w: float | None,
    control_unit: str,
) -> BatteryBalancerInput | None:
    """The house battery's planned grid charging as a give-way entry.

    Only a W-controlled inverter in "charge" intent is a planned grid charge the
    balancer can reason about (SH10RT: symmetric 3-phase, so the cut spreads
    over all three phases — it relieves an overloaded phase by only a third of
    its watts, which is exactly the "battery is not a per-phase lever" fact).
    reducible_w is the grid-sourced part only: min(planned, grid import).
    grid_w: + import / - export; battery_w: + charging.
    """
    if control_unit != "W" or decision is None:
        return None
    active = getattr(decision, "mode_intent", None) == "charge"
    try:
        planned = max(0.0, float(getattr(decision, "charge_value", 0.0) or 0.0))
    except (TypeError, ValueError):
        planned = 0.0
    import_w = max(0.0, float(grid_w)) if grid_w is not None else planned
    return BatteryBalancerInput(
        planned_w=planned,
        applied_cap_w=planned,
        reducible_w=min(planned, import_w),
        active=active and planned > 0,
        min_w=float(getattr(controller, "min_charge_w", 500.0) or 0.0),
        phases=list(BATTERY_PHASES),
        measured_w=None if battery_w is None else max(0.0, float(battery_w)),
        step_w=float(getattr(controller, "round_step_w", 100.0) or 0.0),
    )


def ordered_entries(
    give_way_order: list[GiveWayOrderEntry],
    ev_by_id: dict[str, EVBalancerInput],
    shed_by_id: dict[str, ShedLoadInput],
    battery: BatteryBalancerInput | None = None,
) -> list[EVBalancerInput | ShedLoadInput | BatteryBalancerInput]:
    """Resolve the configured give-way order into the list the balancer ticks.

    Order IS the policy here — position decides who gives way first — so an
    entry naming a device that does not exist is skipped rather than appended
    somewhere arbitrary. config.heal_give_way_order is what adds missing
    devices, at a considered position.
    """
    entries: list[EVBalancerInput | ShedLoadInput | BatteryBalancerInput] = []
    for entry in give_way_order:
        if entry.kind == "charger" and entry.id in ev_by_id:
            entries.append(ev_by_id[entry.id])
        elif entry.kind == "shed" and entry.id in shed_by_id:
            entries.append(shed_by_id[entry.id])
        elif entry.kind == "battery" and battery is not None:
            entries.append(battery)
    return entries


def uniform_updated_at(phase_current_a: dict[int, float], now: datetime) -> dict[int, datetime]:
    """One timestamp for every phase we have a reading for (legacy helper)."""
    return dict.fromkeys(phase_current_a, now)


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo is not None else None


def reading_timestamp(state: dict[str, Any]) -> datetime | None:
    """When HA last heard this sensor: last_reported -> last_updated -> last_changed.

    last_reported (HA >= 2024.3) moves on every report, even an unchanged value,
    so a steady-but-alive sensor is not stale. A naive timestamp is rejected
    rather than guessed into a timezone (the naive-datetime trap this site has
    hit before).
    """
    for key in ("last_reported", "last_updated", "last_changed"):
        ts = _parse_ts(state.get(key))
        if ts is not None:
            return ts
    return None


async def read_phase_samples(
    ha: Any, phase_entities: dict[str, str]
) -> tuple[dict[int, float], dict[int, datetime]]:
    """Per-phase |A| readings with their REAL reading timestamps.

    Each phase is read on its own: an unreadable, non-numeric or timestamp-less
    phase is simply absent, which the balancer treats as stale for that phase
    only. Magnitude, not sign: export current heats the fuse just the same.
    """
    currents: dict[int, float] = {}
    stamps: dict[int, datetime] = {}
    for name, entity in phase_entities.items():
        phase = phase_to_int(name)
        if phase is None or not entity:
            continue
        try:
            state = await ha.get_state(entity)
        except Exception as exc:
            logger.debug("Load balancer: phase read failed for %s: %s", entity, exc)
            continue
        if not isinstance(state, dict):
            continue
        try:
            value = float(state.get("state"))  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            continue
        ts = reading_timestamp(state)
        if not math.isfinite(value) or ts is None:
            continue
        currents[phase] = abs(value)
        stamps[phase] = ts
    return currents, stamps


def status_payload(
    status: LoadBalancerStatus | None,
    *,
    actuate: bool,
    applied: list[str] | None = None,
) -> dict[str, Any]:
    """JSON-safe summary for /api/executor/status["load_balancer"]."""
    if status is None:
        return {"enabled": False, "mode": "observe", "state": "not_run"}
    trip_margin = {
        f"L{p}": round(max(0.0, (status.fuse_safety_percent or 0.0) - heat), 1)
        for p, heat in status.phase_heat_percent.items()
    }
    battery = status.battery_output
    return {
        "enabled": status.enabled,
        "mode": "actuate" if actuate else "observe",
        "state": status.state,
        "reason": status.reason,
        "main_fuse_a": status.main_fuse_a,
        "phase_current_a": {f"L{p}": round(a, 2) for p, a in status.phase_current_a.items()},
        "phase_headroom_a": {f"L{p}": round(a, 2) for p, a in status.phase_headroom_a.items()},
        "fuse_heat_percent": {f"L{p}": round(h, 1) for p, h in status.phase_heat_percent.items()},
        "fuse_heat_state": {f"L{p}": s for p, s in status.phase_heat_state.items()},
        "fuse_safety_percent": status.fuse_safety_percent,
        # Percentage points of trip-time heat left before the model demands relief.
        "trip_margin_percent": trip_margin,
        "would": {
            "chargers": [
                {
                    "id": o.charger_id,
                    "target_a": o.target_a,
                    "state": o.state,
                    "reason": o.reason,
                }
                for o in status.ev_outputs
            ],
            "sheds": [
                {
                    "id": o.load_id,
                    "off": o.shed,
                    "actively_shedding": o.actively_shedding,
                    "reason": o.reason,
                }
                for o in status.shed_outputs
            ],
            "battery": None
            if battery is None
            else {
                "planned_w": battery.planned_w,
                "cap_w": battery.cap_w,
                "state": battery.state,
                "reason": battery.reason,
            },
        },
        "applied": list(applied or []),
    }


def would_summary(status: LoadBalancerStatus) -> str:
    """One-line 'what it would do' for the log."""
    parts = [
        f"{o.charger_id}->{o.target_a}A" if o.target_a is not None else f"{o.charger_id}:pause"
        for o in status.ev_outputs
        if o.state != "idle"
    ]
    parts += [f"{o.load_id}:shed" for o in status.shed_outputs if o.actively_shedding]
    bat = status.battery_output
    if bat is not None and bat.state != "idle":
        parts.append(f"battery cap {bat.cap_w:.0f}W ({bat.state})")
    return ", ".join(parts) or "nothing"
