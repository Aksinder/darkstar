"""
Hot-water availability estimator (open-loop tank state observer).

The hard part for a domestic hot-water tank without a temperature probe or a
flow meter: heating energy *in* is measured (the element's power sensor), but
the hot water drawn *out* is not. A naive integrator therefore drifts.

This estimator keeps the drift bounded without any extra hardware:

1. Integrate measured heating energy in, minus modelled standing losses
   (``WaterTankModel``), to track stored useful energy.
2. **Auto-anchor to FULL** whenever the element has been drawing (near) full
   power for a sustained period and then switches off: the thermostat being
   satisfied after a real heat-up means the tank reached its setpoint = full.
   This resets accumulated error several times a day (what a manual "calibrate"
   button does, done automatically).

Between anchors the stored estimate is an upper bound (unaccounted draws lower
the true level), but every thermostat-satisfied event re-pins it to full.

It also answers "what did we draw?" in aggregate, which *is* observable even
without a flow meter: over any window that starts and ends at full,
``draw ~= heating_energy_in minus standing_losses`` (the element replaces exactly
what was tapped plus what leaked). See ``estimate_draw_kwh``.

Pure stdlib — no I/O — so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planner.thermal import WaterTankModel

__all__ = ["HotWaterEstimator", "estimate_draw_kwh"]


@dataclass
class HotWaterEstimator:
    """Open-loop hot-water tank state observer.

    Args:
        tank: the tank's physical model.
        t_ambient_c: ambient temperature around the tank (for standing losses).
        heating_on_w: power (W) above which the element counts as actively heating.
        full_anchor_after_min: minimum sustained heating before a switch-off is
            treated as "thermostat satisfied -> tank full".
        comfort_c: usable hot-water temperature (e.g. shower temperature).
    """

    tank: WaterTankModel
    t_ambient_c: float = 20.0
    heating_on_w: float = 200.0
    full_anchor_after_min: float = 8.0
    comfort_c: float = 40.0

    # Mutable state (defaults to a full tank at t_max).
    stored_kwh: float = field(default=-1.0)
    _heating_run_min: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.stored_kwh < 0:
            self.stored_kwh = self.tank.capacity_kwh()

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_temperature(
        cls, tank: WaterTankModel, temp_c: float, **kw: float
    ) -> HotWaterEstimator:
        """Seed the observer from a known (e.g. just-calibrated) temperature."""
        est = cls(tank=tank, **kw)  # type: ignore[arg-type]
        est.stored_kwh = tank.stored_kwh(temp_c)
        return est

    # -- state views --------------------------------------------------------

    def temperature_c(self) -> float:
        """Equivalent (well-mixed) tank temperature for the stored energy."""
        return self.tank.t_cold_c + self.stored_kwh * 1000.0 / self.tank.heat_capacity_wh_per_k

    def soc_percent(self) -> float:
        cap = self.tank.capacity_kwh()
        return max(0.0, min(100.0, (self.stored_kwh / cap * 100.0) if cap > 0 else 0.0))

    def liters_in_tank(self) -> float:
        """Litres of usable hot water physically in the tank (volume x SoC)."""
        return self.tank.volume_litres * self.soc_percent() / 100.0

    def mixed_liters_at(self, comfort_c: float | None = None) -> float:
        """Litres deliverable at ``comfort_c`` by mixing tank water with cold inlet.

        A tank at T can be diluted with cold (t_cold) to deliver more than its
        own volume at a lower comfort temperature. Returns 0 once the tank falls
        to/below comfort.
        """
        comfort = self.comfort_c if comfort_c is None else comfort_c
        t = self.temperature_c()
        denom = comfort - self.tank.t_cold_c
        if t <= comfort or denom <= 0:
            return 0.0
        return self.tank.volume_litres * (t - self.tank.t_cold_c) / denom

    # -- update -------------------------------------------------------------

    def update(
        self, dt_minutes: float, heating_kw: float, t_ambient_c: float | None = None
    ) -> None:
        """Advance the observer by one step.

        Args:
            dt_minutes: elapsed time since the previous update.
            heating_kw: measured element power this step (kW).
            t_ambient_c: optional ambient override for this step.
        """
        if dt_minutes <= 0:
            return
        dt_h = dt_minutes / 60.0
        ambient = self.t_ambient_c if t_ambient_c is None else t_ambient_c

        # Standing loss is always present, based on the current temperature.
        self.stored_kwh -= self.tank.standby_loss_kwh(self.temperature_c(), dt_h, ambient)

        heating_on = heating_kw * 1000.0 >= self.heating_on_w
        if heating_on:
            self.stored_kwh += heating_kw * dt_h
            self._heating_run_min += dt_minutes
        else:
            # Element just switched off after a real heat-up => tank is full.
            if self._heating_run_min >= self.full_anchor_after_min:
                self.stored_kwh = self.tank.capacity_kwh()
            self._heating_run_min = 0.0

        # Clamp to physical limits.
        self.stored_kwh = max(0.0, min(self.tank.capacity_kwh(), self.stored_kwh))

    def anchor_full(self) -> None:
        """Force the state to full (e.g. on a manual calibrate-to-100%)."""
        self.stored_kwh = self.tank.capacity_kwh()
        self._heating_run_min = 0.0


def estimate_draw_kwh(
    tank: WaterTankModel,
    heating_energy_kwh: float,
    avg_temp_c: float,
    hours: float,
    t_ambient_c: float = 20.0,
) -> float:
    """Estimate hot water drawn over a window that begins and ends at full.

    Observable without a flow meter: the element replaces exactly what was
    tapped plus standing losses, so over a full->full window
    ``draw = heating_in - standing_losses``. ``avg_temp_c`` is the window's mean
    tank temperature (drives the loss term). Clamped at 0.
    """
    losses = tank.avg_loss_kw(avg_temp_c, t_ambient_c) * hours
    return max(0.0, heating_energy_kwh - losses)
