"""
Thermal model for a hot-water tank (step 5 / Phase 2 of the deferrable-loads
blueprint: model the water heater as a thermal battery).

Treats the tank as a lumped thermal mass with Newtonian standing losses toward
ambient. This lets the planner reason about the tank's *state of charge* (stored
useful heat) instead of a binary on/off element: charge when energy is cheap /
PV is in surplus, coast on losses, and meet a comfort floor or a "hot water by
HH:MM" deadline.

Pure stdlib (math) — no I/O, no pulp — so it is fully unit-testable and can be
used both by the planner (to build MILP coefficients) and by diagnostics.

Physics:
- Heat capacity C = volume_liters * c_p, with water c_p = 1.163 Wh/(litre*K)
  (4186 J/(kg*K) / 3600), assuming ~1 kg per litre.
- Heating: dT = energy_kwh * 1000 / C.
- Standing loss (Newton's law of cooling): T(t) = T_amb + (T0 - T_amb) * e^(-t/tau)
  with time constant tau = C / UA  [hours], UA in W/K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["WaterTankModel"]

# Specific heat of water expressed per litre in Wh: 4186 J/(kg*K) / 3600 s/h.
_C_P_WH_PER_LITRE_K = 4186.0 / 3600.0  # ~= 1.1628


@dataclass(frozen=True)
class WaterTankModel:
    """Lumped-capacitance hot-water tank.

    Args:
        volume_litres: tank volume (litres ~= kg of water).
        t_cold_c: cold-inlet / reference temperature (bottom of the useful range).
        t_max_c: maximum safe tank temperature (top of the useful range).
        ua_w_per_k: overall heat-loss coefficient (W per K above ambient). A
            typical well-insulated 200 L tank is ~1.5-2.5 W/K (~50-100 W standby
            at a 40 K delta).
    """

    volume_litres: float
    t_cold_c: float = 10.0
    t_max_c: float = 85.0
    ua_w_per_k: float = 2.0

    @property
    def heat_capacity_wh_per_k(self) -> float:
        """Thermal capacity of the tank contents, Wh per K."""
        return self.volume_litres * _C_P_WH_PER_LITRE_K

    def stored_kwh(self, temp_c: float) -> float:
        """Useful stored energy above the cold reference, kWh (>= 0)."""
        delta = max(0.0, temp_c - self.t_cold_c)
        return self.heat_capacity_wh_per_k * delta / 1000.0

    def capacity_kwh(self) -> float:
        """Maximum useful stored energy (cold -> max), kWh."""
        return self.stored_kwh(self.t_max_c)

    def soc_percent(self, temp_c: float) -> float:
        """State of charge (0-100%) over the usable [t_cold, t_max] range."""
        span = self.t_max_c - self.t_cold_c
        if span <= 0:
            return 0.0
        return max(0.0, min(100.0, (temp_c - self.t_cold_c) / span * 100.0))

    def energy_to_heat_kwh(self, t_from_c: float, t_to_c: float) -> float:
        """Energy to raise the tank from t_from to t_to, kWh (>= 0)."""
        delta = max(0.0, t_to_c - t_from_c)
        return self.heat_capacity_wh_per_k * delta / 1000.0

    def temp_after_heating(self, temp_c: float, energy_kwh: float) -> float:
        """Tank temperature after adding ``energy_kwh`` (capped at t_max)."""
        rise = energy_kwh * 1000.0 / self.heat_capacity_wh_per_k
        return min(self.t_max_c, temp_c + rise)

    def time_constant_hours(self) -> float:
        """Cooling time constant tau = C / UA, in hours (inf if no loss)."""
        if self.ua_w_per_k <= 0:
            return math.inf
        return self.heat_capacity_wh_per_k / self.ua_w_per_k

    def temp_after_loss(self, temp_c: float, hours: float, t_ambient_c: float = 20.0) -> float:
        """Tank temperature after ``hours`` of standing loss toward ambient."""
        if hours <= 0 or self.ua_w_per_k <= 0:
            return temp_c
        tau = self.time_constant_hours()
        return t_ambient_c + (temp_c - t_ambient_c) * math.exp(-hours / tau)

    def standby_loss_kwh(self, temp_c: float, hours: float, t_ambient_c: float = 20.0) -> float:
        """Useful energy lost to standing losses over ``hours`` (>= 0)."""
        t_after = self.temp_after_loss(temp_c, hours, t_ambient_c)
        return max(0.0, self.stored_kwh(temp_c) - self.stored_kwh(t_after))

    def avg_loss_kw(self, temp_c: float, t_ambient_c: float = 20.0) -> float:
        """Instantaneous standby loss at ``temp_c``, kW (UA * dT)."""
        return max(0.0, self.ua_w_per_k * (temp_c - t_ambient_c) / 1000.0)
