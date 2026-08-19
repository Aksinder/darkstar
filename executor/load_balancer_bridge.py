"""
Impedance match between our runtime and upstream's load balancer.

``executor/load_balancer.py`` is ported verbatim from upstream and expects
upstream's data model: phase currents keyed by int, EV chargers described by
``EVChargerDeviceConfig``. Our fork replaced the EV layer with
``executor/ev_surplus*`` and never carried upstream's, so the decision core
transplanted cleanly but its wiring did not. This module is that wiring, kept
separate from the engine because it is pure and therefore testable, and because
mapping between two data models is exactly where silent mistakes live.

Two mismatches are worth naming:

**Phase identity.** We key phases by name ("l1"/"l2"/"l3", matching the fuse
guard's ``phase_entities`` config); upstream keys them 1/2/3. An unrecognised
key is dropped rather than guessed — a phase silently mapped to the wrong
number would budget the wrong conductor, which is worse than not guarding it.

**Staleness granularity.** Upstream tests each phase's own ``last_updated`` and
degrades that phase alone. Our reader is deliberately all-or-nothing: it
returns None if ANY phase is stale, because a partially blind guard budgets the
wrong phases. So we feed one timestamp for every phase and simply do not call
the balancer when the read came back None. Upstream's per-phase staleness path
is therefore dead code here — it is kept intact rather than stripped, so the
ported core stays diffable against upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .load_balancer import EVBalancerInput, ShedLoadInput

if TYPE_CHECKING:
    from datetime import datetime

    from .config import BalancedLoadConfig, GiveWayOrderEntry

# "l1"/"L1"/"1"/1 all mean phase 1. Anything else is dropped, never guessed.
_PHASE_NAMES = {"l1": 1, "l2": 2, "l3": 3, "1": 1, "2": 2, "3": 3}


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
    the very phase that is overloaded. (Deliberately not parameterised on phase
    count — an earlier draft branched on it and both branches returned the same
    answer, which reads like a decision where there is none.)
    """
    mapped = [p for p in (phase_to_int(x) for x in phase_map or ()) if p is not None]
    return sorted(set(mapped)) if mapped else [1, 2, 3]


def ev_entries(
    chargers: list[Any],
    *,
    setpoints_a: dict[str, float | None],
    targets_a: dict[str, float | None],
) -> dict[str, EVBalancerInput]:
    """Build balancer inputs for the chargers we can actually throttle.

    A charger without a settable current entity (``controllable=False``) is an
    on/off device: it belongs in ``load_balancing.loads`` as a shed entry, not
    here, because the balancer would otherwise "throttle" it by writing a
    setpoint nothing reads.
    """
    out: dict[str, EVBalancerInput] = {}
    for c in chargers:
        if not getattr(c, "controllable", True):
            continue
        target = targets_a.get(c.id)
        setpoint = setpoints_a.get(c.id)
        out[c.id] = EVBalancerInput(
            charger_id=c.id,
            phases=phases_for_charger(getattr(c, "phase_map", ())),
            current_setpoint_a=None if setpoint is None else int(setpoint),
            planner_target_a=None if target is None else int(target),
            min_current_a=int(getattr(c, "min_current_a", 6) or 6),
            max_current_a=int(getattr(c, "max_current_a", 16) or 16),
        )
    return out


def shed_entries(loads: list[BalancedLoadConfig]) -> dict[str, ShedLoadInput]:
    """Build balancer inputs for the configured on/off shed-able loads."""
    return {
        ld.device_id: ShedLoadInput(
            load_id=ld.device_id,
            device_type=ld.device_type.value,
            phases=sorted({p for p in (phase_to_int(x) for x in ld.phases) if p is not None}),
        )
        for ld in loads
        if ld.device_id
    }


def ordered_entries(
    give_way_order: list[GiveWayOrderEntry],
    ev_by_id: dict[str, EVBalancerInput],
    shed_by_id: dict[str, ShedLoadInput],
) -> list[EVBalancerInput | ShedLoadInput]:
    """Resolve the configured give-way order into the list the balancer ticks.

    Order IS the policy here — position decides who gives way first — so an
    entry naming a device that does not exist is skipped rather than appended
    somewhere arbitrary. config.heal_give_way_order is what adds missing
    devices, at a considered position.
    """
    entries: list[EVBalancerInput | ShedLoadInput] = []
    for entry in give_way_order:
        if entry.kind == "charger" and entry.id in ev_by_id:
            entries.append(ev_by_id[entry.id])
        elif entry.kind == "shed" and entry.id in shed_by_id:
            entries.append(shed_by_id[entry.id])
    return entries


def uniform_updated_at(
    phase_current_a: dict[int, float], now: datetime
) -> dict[int, datetime]:
    """One timestamp for every phase we have a reading for.

    Honest about what we know: our reader already rejected the whole snapshot if
    any phase was stale, so every phase present here is fresh as of `now`.
    """
    return dict.fromkeys(phase_current_a, now)
