"""Spa thermal physics: a lumped Newtonian tank with a heating element.

ONE physics for the spa bathing feature, shared by the planner (kepler's
temperature recursion, the shadow-mode simulation, the post-solve report) and the
executor (catch-up rule). Pure stdlib (math): no I/O, no pulp, no planner import,
so the executor may import it (see the executor/water_hold.py header).

Model (exact discretisation of dT/dt = P*on/C - k*(T - ambient) over a slot of
h hours with the element on for a fraction ``on`` of the slot's power):

    a       = exp(-k * h)
    T[t+1]  = a * T[t] + (1 - a) * ambient + (1 - a) / (k * C) * P * on[t]

C is the heat capacity in kWh per degC, k the cooling constant per hour, P the
element power in kW. For k -> 0 the gain term tends to P * h / C. This is the
same Newtonian loss as planner/thermal.py WaterTankModel.temp_after_loss with
volume = C * 1000 / 1.1628 litres and UA = k * C * 1000 W/K, plus a heater term.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "decay_factor",
    "heat_gain_c",
    "heat_rate_c_per_h",
    "hours_to_reach_c",
    "max_reachable_temp_c",
    "net_heat_rate_c_per_h",
    "simulate_temps_c",
    "step_temp_c",
]

# Below this cooling constant the closed forms divide by ~0; use the k -> 0 limit.
_K_EPS = 1e-9


def decay_factor(k_per_h: float, hours: float) -> float:
    """Fraction of the excess over ambient left after ``hours``: exp(-k*h)."""
    return math.exp(-k_per_h * max(0.0, hours))


def heat_gain_c(power_kw: float, capacity_kwh_per_c: float, k_per_h: float, hours: float) -> float:
    """Temperature rise (degC) from the element fully on for ``hours``, net of the
    extra loss it causes: (1 - a) / (k * C) * P, or P * h / C when k ~ 0."""
    if capacity_kwh_per_c <= 0:
        return 0.0
    h = max(0.0, hours)
    if k_per_h < _K_EPS:
        return power_kw * h / capacity_kwh_per_c
    a = decay_factor(k_per_h, h)
    return (1.0 - a) / (k_per_h * capacity_kwh_per_c) * power_kw


def step_temp_c(
    temp_c: float,
    ambient_c: float,
    k_per_h: float,
    hours: float,
    *,
    power_kw: float = 0.0,
    capacity_kwh_per_c: float = 0.9,
    on_fraction: float = 0.0,
) -> float:
    """Temperature after one slot of ``hours`` (element on for ``on_fraction``)."""
    a = decay_factor(k_per_h, hours)
    gain = heat_gain_c(power_kw, capacity_kwh_per_c, k_per_h, hours)
    return a * temp_c + (1.0 - a) * ambient_c + gain * on_fraction


def simulate_temps_c(
    temp0_c: float,
    ambient_c: float,
    k_per_h: float,
    capacity_kwh_per_c: float,
    power_kw: float,
    hours: Sequence[float],
    on_fraction: Sequence[float],
) -> list[float]:
    """End-of-slot temperatures (len == len(hours)) for a heating pattern.

    ``on_fraction`` shorter than ``hours`` is padded with 0 (element off)."""
    temps: list[float] = []
    temp = temp0_c
    for i, h in enumerate(hours):
        on = on_fraction[i] if i < len(on_fraction) else 0.0
        temp = step_temp_c(
            temp,
            ambient_c,
            k_per_h,
            h,
            power_kw=power_kw,
            capacity_kwh_per_c=capacity_kwh_per_c,
            on_fraction=on,
        )
        temps.append(temp)
    return temps


def heat_rate_c_per_h(power_kw: float, capacity_kwh_per_c: float) -> float:
    """Gross heating rate P / C in degC per hour (0 when C <= 0)."""
    if capacity_kwh_per_c <= 0:
        return 0.0
    return power_kw / capacity_kwh_per_c


def net_heat_rate_c_per_h(
    temp_c: float,
    ambient_c: float,
    power_kw: float,
    capacity_kwh_per_c: float,
    k_per_h: float,
) -> float:
    """Instantaneous net rate with the element on: P / C - k * (T - ambient)."""
    return heat_rate_c_per_h(power_kw, capacity_kwh_per_c) - k_per_h * (temp_c - ambient_c)


def hours_to_reach_c(
    temp0_c: float,
    target_c: float,
    ambient_c: float,
    power_kw: float,
    capacity_kwh_per_c: float,
    k_per_h: float,
) -> float | None:
    """Hours with the element continuously on to go from temp0 to target.

    0.0 when already at/above target; None when the target is unreachable (at or
    above the asymptote ambient + P / (k * C), or no heating power)."""
    if temp0_c >= target_c:
        return 0.0
    if power_kw <= 0 or capacity_kwh_per_c <= 0:
        return None
    if k_per_h < _K_EPS:
        return (target_c - temp0_c) * capacity_kwh_per_c / power_kw
    asymptote = ambient_c + power_kw / (k_per_h * capacity_kwh_per_c)
    if target_c >= asymptote:
        return None
    ratio = (asymptote - target_c) / (asymptote - temp0_c)
    if ratio <= 0:
        return None
    return -math.log(ratio) / k_per_h


def max_reachable_temp_c(
    temp0_c: float,
    ambient_c: float,
    power_kw: float,
    capacity_kwh_per_c: float,
    k_per_h: float,
    hours: float,
) -> float:
    """Temperature after ``hours`` with the element continuously on (the physical
    best case; no temp_max cap applied)."""
    return step_temp_c(
        temp0_c,
        ambient_c,
        k_per_h,
        hours,
        power_kw=max(0.0, power_kw),
        capacity_kwh_per_c=capacity_kwh_per_c,
        on_fraction=1.0,
    )
