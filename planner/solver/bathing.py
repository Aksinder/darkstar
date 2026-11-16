"""
Spa bathing times: the temperature-state model kepler adds for a bathing heater.

Pure helpers kept out of kepler.py so the solver only wires them in. Everything
here is continuous: the heater's EXISTING binaries (water_heat / water_boost) drive
an exact Newtonian recursion (backend/core/spa_thermal.py), and every comfort rule
is a soft, slack-priced row. No integer variables are added, and no row can make
the model infeasible:

    Tt[0]   == T0
    Tt[t+1] == a_t*Tt[t] + (1-a_t)*ambient + g_t*on[t]      a_t = exp(-k*h_t)
    over[t] >= Tt[t+1] - temp_max                            (soft cap)
    Tt[k_j] + sr_j >= ready_temp                             (ready by, per window)
    Tt[t+1] + sh_t >= ready_temp - hold_tolerance            (hold, in-window slots)
    Tt[s]   + sf_s >= eco_floor                              (s >= 1)

Window -> index mapping (t_now = the planning start, i.e. slot 0's end minus its
planned hours, so a partial slot 0 is respected):
  * end <= t_now: dropped (no target, no hold).
  * start <= t_now: in progress, hold only, no ready target.
  * ready_by <= t_now or ready_by < slots[0].end_time: ready target on state 1
    (soft, inevitably short when the heater cannot make it in one slot).
  * ready_by > slots[-1].end_time: beyond the horizon, no ready target.
  * otherwise k_j = max{k : slots[k-1].end_time <= ready_by}: the target binds at
    the last state boundary at or before ready_by (a coarse tail snaps it to the
    previous whole hour, which is conservative).
  * hold slots: slots fully inside [start, end]; slot 0 counts from t_now.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pulp  # type: ignore[import]

from backend.core.spa_thermal import decay_factor, heat_gain_c, max_reachable_temp_c

from .types import BathingReport, BathingSpec, KeplerInputSlot, WaterHeaterInput

logger = logging.getLogger("darkstar.kepler")

# Margin (degC) on the state variable bounds. The bounds only help numerics; they
# are chosen wide enough to never bind on any heating pattern (see build).
_BOUND_MARGIN_C = 15.0

# Tripwire: a time-boxed incumbent whose AVOIDABLE ready shortfall (planned
# shortfall minus the physically unavoidable part) exceeds this is rejected.
BATHING_TRIPWIRE_C = 1.0


def _aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def validate_bathing_spec(heater: WaterHeaterInput, slots: list[KeplerInputSlot]) -> str | None:
    """Why ``heater.bathing`` cannot be modelled, or None when it is valid.

    An invalid spec is dropped by the caller (logger.error) and the heater stays on
    the legacy daily-kWh path. Never raises: a mixed naive/aware pairing of window
    and slot datetimes is reported here instead of exploding as a TypeError later.
    """
    spec = heater.bathing
    if spec is None:
        return "no spec"
    if not slots:
        return "no slots"
    numbers: dict[str, float] = {
        "initial_temp_c": spec.initial_temp_c,
        "ambient_c": spec.ambient_c,
        "heat_capacity_kwh_per_c": spec.heat_capacity_kwh_per_c,
        "cooling_k_per_h": spec.cooling_k_per_h,
        "ready_temp_c": spec.ready_temp_c,
        "eco_floor_c": spec.eco_floor_c,
        "temp_max_c": spec.temp_max_c,
        "ready_shortfall_sek_per_c": spec.ready_shortfall_sek_per_c,
        "hold_shortfall_sek_per_c_h": spec.hold_shortfall_sek_per_c_h,
        "floor_shortfall_sek_per_c_h": spec.floor_shortfall_sek_per_c_h,
        "hold_tolerance_c": spec.hold_tolerance_c,
        "overshoot_penalty_sek_per_c": spec.overshoot_penalty_sek_per_c,
        "power_kw": heater.power_kw,
    }
    for name, val in numbers.items():
        try:
            if not math.isfinite(float(val)):
                return f"{name} is not finite ({val!r})"
        except (TypeError, ValueError):
            return f"{name} is not a number ({val!r})"
    if spec.heat_capacity_kwh_per_c <= 0:
        return "heat_capacity_kwh_per_c must be > 0"
    if spec.cooling_k_per_h <= 0:
        return "cooling_k_per_h must be > 0"
    if not (spec.eco_floor_c < spec.ready_temp_c <= spec.temp_max_c):
        return (
            f"need eco_floor_c < ready_temp_c <= temp_max_c "
            f"(got {spec.eco_floor_c}, {spec.ready_temp_c}, {spec.temp_max_c})"
        )
    if heater.power_kw <= 0:
        return "heater power_kw must be > 0"
    for name in (
        "ready_shortfall_sek_per_c",
        "hold_shortfall_sek_per_c_h",
        "floor_shortfall_sek_per_c_h",
        "hold_tolerance_c",
        "overshoot_penalty_sek_per_c",
    ):
        if numbers[name] < 0:
            return f"{name} must be >= 0"
    slots_aware = _aware(slots[0].start_time)
    if _aware(slots[-1].end_time) != slots_aware:
        return "slots mix naive and aware datetimes"
    for j, w in enumerate(spec.windows):
        for name in ("start", "end", "ready_by"):
            dt: Any = getattr(w, name)
            if not isinstance(dt, datetime):
                return f"window {j} {name} is not a datetime ({dt!r})"
            if _aware(dt) != slots_aware:
                return (
                    f"window {j} {name} is {'aware' if _aware(dt) else 'naive'} but the "
                    f"slots are {'aware' if slots_aware else 'naive'}"
                )
    return None


@dataclass
class BathingWindowMap:
    """Where each BathingSpec window binds in the solver's state/slot indices."""

    # Per window (BathingSpec.windows order): the state index the ready row binds
    # at (1..T), or None (dropped, in progress, beyond the horizon).
    ready_state: list[int | None] = field(default_factory=lambda: [])
    # Slot indices t whose END state t+1 is held at ready - tolerance (union of all
    # windows, sorted, unique).
    hold_slots: list[int] = field(default_factory=lambda: [])


def map_windows(
    spec: BathingSpec, slots: list[KeplerInputSlot], slot_hours: list[float]
) -> BathingWindowMap:
    """Map the spec's windows onto the solver grid (see the module docstring)."""
    T = len(slots)
    out = BathingWindowMap()
    if T == 0:
        out.ready_state = [None for _ in spec.windows]
        return out
    t_now: datetime = slots[0].end_time - timedelta(hours=slot_hours[0])
    hold: set[int] = set()
    for w in spec.windows:
        if w.end <= t_now:
            out.ready_state.append(None)
            continue
        if w.start > t_now:
            if w.ready_by <= t_now or w.ready_by < slots[0].end_time:
                out.ready_state.append(1)
            elif w.ready_by > slots[-1].end_time:
                out.ready_state.append(None)  # horizon rule: no target
            else:
                k_j = 0
                for k in range(1, T + 1):
                    if slots[k - 1].end_time <= w.ready_by:
                        k_j = k
                    else:
                        break
                out.ready_state.append(k_j if k_j >= 1 else 1)
        else:
            out.ready_state.append(None)  # in progress: hold only
        for t in range(T):
            eff_start = t_now if t == 0 else slots[t].start_time
            if eff_start >= w.start and slots[t].end_time <= w.end:
                hold.add(t)
    out.hold_slots = sorted(hold)
    return out


@dataclass
class BathingModel:
    """LP handles for one bathing heater (read back after the solve)."""

    heater_id: str
    spec: BathingSpec
    wmap: BathingWindowMap
    temp: dict[int, Any]  # Tt[s], s = 0..T
    over: dict[int, Any]  # over[t], t = 0..T-1
    ready_short: dict[int, Any]  # window j -> sr_j
    hold_short: dict[int, Any]  # slot t -> sh_t
    floor_short: dict[int, Any]  # state s -> sf_s
    unavoidable_c: list[float | None]


def unavoidable_shortfalls(
    spec: BathingSpec,
    power_kw: float,
    wmap: BathingWindowMap,
    slot_hours: list[float],
) -> list[float | None]:
    """Per window: max(0, ready - best physically reachable T at its ready state).

    "Best" = element continuously on from now to the state the ready row binds at
    (<= ready_by), so a physically late schedule never counts as a solver failure.
    """
    out: list[float | None] = []
    for k_j in wmap.ready_state:
        if k_j is None:
            out.append(None)
            continue
        hours = sum(slot_hours[:k_j])
        best = max_reachable_temp_c(
            spec.initial_temp_c,
            spec.ambient_c,
            power_kw,
            spec.heat_capacity_kwh_per_c,
            spec.cooling_k_per_h,
            hours,
        )
        out.append(max(0.0, spec.ready_temp_c - best))
    return out


def build_bathing_model(
    prob: Any,
    heater: WaterHeaterInput,
    spec: BathingSpec,
    on_terms: list[Any],
    slot_hours: list[float],
    wmap: BathingWindowMap,
    total_cost: list[Any],
) -> BathingModel:
    """Add the temperature recursion and its soft rows for one heater.

    ``on_terms[t]`` is the heater's element-on expression in slot t (heat plus
    boost, the same expression that prices its energy in the balance).
    """
    T = len(on_terms)
    d = heater.id
    safe_d = d.replace("-", "_").replace(".", "_")
    k = spec.cooling_k_per_h
    C = spec.heat_capacity_kwh_per_c
    P = heater.power_kw
    amb = spec.ambient_c
    T0 = spec.initial_temp_c

    total_h = sum(slot_hours)
    lo: float = min(T0, amb) - 1.0
    # All-off decays monotonically toward ambient, so lo never binds. All-on is the
    # pointwise-highest trajectory; keeping hi above it means even a long force_on
    # run cannot turn the (deliberately soft) cap into an infeasible bound.
    hi: float = max(
        max(T0, spec.temp_max_c) + _BOUND_MARGIN_C,
        max_reachable_temp_c(T0, amb, P, C, k, total_h) + 1.0,
    )
    temp: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
        f"bath_temp_{safe_d}", range(T + 1), lowBound=lo, upBound=hi
    )
    over: dict[int, Any] = pulp.LpVariable.dicts(  # type: ignore[reportUnknownMemberType]
        f"bath_over_{safe_d}", range(T), lowBound=0.0
    )
    prob += temp[0] == T0  # type: ignore[operator]
    for t in range(T):
        h = slot_hours[t]
        a = decay_factor(k, h)
        g = heat_gain_c(P, C, k, h)
        prob += (  # type: ignore[operator]
            temp[t + 1] == a * temp[t] + (1.0 - a) * amb + g * on_terms[t]
        )
        prob += over[t] >= temp[t + 1] - spec.temp_max_c  # type: ignore[operator]

    ready_short: dict[int, Any] = {}
    for j, k_j in enumerate(wmap.ready_state):
        if k_j is None:
            continue
        sr = pulp.LpVariable(
            f"bath_ready_short_{safe_d}_{j}",
            lowBound=0.0,
            upBound=max(0.0, spec.ready_temp_c - lo),
        )
        prob += temp[k_j] + sr >= spec.ready_temp_c  # type: ignore[operator]
        ready_short[j] = sr

    hold_target = spec.ready_temp_c - spec.hold_tolerance_c
    hold_short: dict[int, Any] = {}
    for t in wmap.hold_slots:
        sh = pulp.LpVariable(f"bath_hold_short_{safe_d}_{t}", lowBound=0.0)
        prob += temp[t + 1] + sh >= hold_target  # type: ignore[operator]
        hold_short[t] = sh

    floor_short: dict[int, Any] = {}
    for s in range(1, T + 1):
        sf = pulp.LpVariable(f"bath_floor_short_{safe_d}_{s}", lowBound=0.0)
        prob += temp[s] + sf >= spec.eco_floor_c  # type: ignore[operator]
        floor_short[s] = sf

    if ready_short and spec.ready_shortfall_sek_per_c > 0:
        total_cost.append(spec.ready_shortfall_sek_per_c * pulp.lpSum(ready_short.values()))
    if hold_short and spec.hold_shortfall_sek_per_c_h > 0:
        total_cost.append(
            spec.hold_shortfall_sek_per_c_h
            * pulp.lpSum(slot_hours[t] * sh for t, sh in hold_short.items())
        )
    if floor_short and spec.floor_shortfall_sek_per_c_h > 0:
        total_cost.append(
            spec.floor_shortfall_sek_per_c_h
            * pulp.lpSum(slot_hours[s - 1] * sf for s, sf in floor_short.items())
        )
    if spec.overshoot_penalty_sek_per_c > 0:
        total_cost.append(spec.overshoot_penalty_sek_per_c * pulp.lpSum(over.values()))

    return BathingModel(
        heater_id=d,
        spec=spec,
        wmap=wmap,
        temp=temp,
        over=over,
        ready_short=ready_short,
        hold_short=hold_short,
        floor_short=floor_short,
        unavoidable_c=unavoidable_shortfalls(spec, P, wmap, slot_hours),
    )


def _val(var: Any) -> float | None:
    v: Any = pulp.value(var)  # type: ignore[arg-type]
    return float(v) if v is not None else None


def solved_temps(model: BathingModel) -> list[float | None]:
    """Solved Tt[s] for s = 0..T (None where the solver left no value)."""
    return [_val(model.temp[s]) for s in range(len(model.temp))]


def build_report(model: BathingModel, slot_hours: list[float]) -> BathingReport:
    """Post-solve report computed from the solved TEMPERATURES (not the slacks), so
    a time-boxed incumbent that leaves a slack needlessly large is not misread."""
    spec = model.spec
    temps = solved_temps(model)
    ready_temps: list[float | None] = []
    shortfalls: list[float | None] = []
    for k_j in model.wmap.ready_state:
        tk = temps[k_j] if k_j is not None else None
        if tk is None:
            ready_temps.append(None)
            shortfalls.append(None)
        else:
            ready_temps.append(tk)
            shortfalls.append(max(0.0, spec.ready_temp_c - tk))
    hold_target = spec.ready_temp_c - spec.hold_tolerance_c
    hold_c_h = 0.0
    for t in model.wmap.hold_slots:
        tt = temps[t + 1]
        if tt is not None:
            hold_c_h += slot_hours[t] * max(0.0, hold_target - tt)
    floor_c_h = 0.0
    overshoot = 0.0
    for s in range(1, len(temps)):
        ts = temps[s]
        if ts is None:
            continue
        floor_c_h += slot_hours[s - 1] * max(0.0, spec.eco_floor_c - ts)
        overshoot = max(overshoot, ts - spec.temp_max_c)
    return BathingReport(
        heater_id=model.heater_id,
        window_ready_temp_c=ready_temps,
        window_shortfall_c=shortfalls,
        hold_shortfall_c_h=hold_c_h,
        floor_shortfall_c_h=floor_c_h,
        overshoot_c=max(0.0, overshoot),
        unavoidable_shortfall_c=list(model.unavoidable_c),
    )


def avoidable_shortfall_c(report: BathingReport) -> float:
    """Largest per-window ready shortfall beyond the physically unavoidable part."""
    worst = 0.0
    for j, short in enumerate(report.window_shortfall_c):
        if short is None:
            continue
        unavoidable = (
            report.unavoidable_shortfall_c[j] if j < len(report.unavoidable_shortfall_c) else None
        ) or 0.0
        worst = max(worst, short - unavoidable)
    return worst
