"""
EV surplus-follow controller (real-time, variable current).

Darkstar's planner schedules EV charging as a deferrable energy block (grid/PV, never
the home battery). This module is the *executor-side* real-time counterpart: when the
cars charge live, it dials each charger's current up/down so charging tracks the chosen
energy budget instead of silently draining the home battery.

It is a small incremental feedback controller (not a planner): each cycle it measures the
live grid + battery + price and nudges the EV current toward a target. The energy budget
is a 3-tier source policy:

1. **Solar surplus** — always: any PV beyond house load goes to the cars.
2. **Cheap / negative grid** — when ``import_price <= cheap_grid_price``: allow up to
   ``cheap_grid_allowance_w`` of grid import for the cars.
3. **Home battery** — only when ``import_price <= battery_assist_max_price`` (e.g. ``0`` =
   negative price) AND a lot of solar is still forecast today
   (``remaining_solar_kwh >= battery_assist_min_remaining_solar_kwh``) AND the battery is
   above ``battery_assist_floor_soc``: then let the battery discharge up to
   ``battery_assist_allowance_w`` into the cars — it will refill from the abundant
   remaining solar, so that energy is effectively free.

The control signal is the grid + battery meter, so it self-corrects regardless of what the
house load happens to be. Pure: no HA / I/O; the engine feeds live values in and applies
the returned per-charger commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EVSurplusConfig:
    """Tunables for the EV surplus-follow controller (executor.ev_surplus)."""

    enabled: bool = False
    # Tier 2 — cheap/negative grid. None => never use grid for EV (solar surplus only).
    cheap_grid_price_sek: float | None = None
    cheap_grid_allowance_w: float = 3680.0  # how much grid import to allow for EV when cheap
    # Tier 3 — home battery, only under negative price + lots of solar still to come.
    battery_assist_enabled: bool = False
    battery_assist_max_price_sek: float = 0.0  # only assist when import_price <= this (<=0 = negative)
    battery_assist_min_remaining_solar_kwh: float = 8.0  # need this much PV still forecast today
    battery_assist_floor_soc: float = 30.0  # never discharge the home battery below this for EV
    battery_assist_allowance_w: float = 4000.0  # cap on battery discharge fed to the cars
    # Controller dynamics.
    gain: float = 0.5  # fraction of the headroom applied per cycle (stability vs speed)
    deadband_w: float = 250.0  # hold when |headroom| is within this (anti-jitter)
    min_charge_current_a: float = 6.0  # below this a charger can't charge => turn off
    default_voltage_v: float = 230.0


@dataclass
class ChargerState:
    """Live state + capabilities of one EV charger."""

    id: str
    plugged: bool
    at_home: bool
    enabled: bool
    current_power_w: float
    max_current_a: float
    min_current_a: float = 6.0
    phases: int = 3
    voltage_v: float = 230.0
    controllable: bool = True  # has a settable charge-current entity (vs on/off only)
    priority: int = 0  # lower = filled first


@dataclass
class EVSurplusInputs:
    """Live measurements for one control cycle."""

    pv_w: float
    grid_w: float  # signed: + import, - export
    battery_w: float  # signed: + charging, - discharging
    battery_soc_percent: float
    import_price_sek: float
    remaining_solar_kwh: float
    chargers: list[ChargerState] = field(default_factory=lambda: [])


@dataclass
class ChargerCommand:
    """Desired actuation for one charger this cycle."""

    id: str
    switch_on: bool
    set_current_a: float | None  # None => binary charger / leave current unchanged
    target_power_w: float
    reason: str


def _charger_max_w(c: ChargerState) -> float:
    return c.max_current_a * c.voltage_v * c.phases


def _charger_min_on_w(c: ChargerState, cfg: EVSurplusConfig) -> float:
    return max(c.min_current_a, cfg.min_charge_current_a) * c.voltage_v * c.phases


def compute_ev_surplus(
    inputs: EVSurplusInputs, cfg: EVSurplusConfig
) -> list[ChargerCommand]:
    """Return per-charger commands that steer EV draw toward the allowed energy budget.

    The math: ``headroom_w`` is how much MORE the cars may pull this cycle without
    violating policy::

        headroom = (grid_setpoint - grid) + battery + battery_assist_allowance

    - ``(grid_setpoint - grid)``: room on the grid. ``grid_setpoint`` is 0 (consume all
      solar, import nothing) unless cheap-grid is active, then it's the cheap allowance.
    - ``+ battery``: if the home battery is *charging* (absorbing surplus) that surplus is
      handed to the cars instead; if it's *discharging*, headroom goes negative and the
      cars back off — which is exactly "don't drain the home battery".
    - ``+ battery_assist_allowance``: extra slack permitting a bounded battery discharge,
      only under the negative-price + solar-headroom tier.

    A ``gain``/``deadband`` make it a stable incremental loop. The new total is then
    distributed across the plugged-in, home chargers by priority and clamped per charger.
    """
    chargers = inputs.chargers
    active = [c for c in chargers if c.enabled and c.plugged and c.at_home]
    if not cfg.enabled or not active:
        # Release control: leave plugged/home chargers as-is (no command). Disabled =>
        # the controller never touches anything.
        return []

    # --- Tier gating ---
    cheap_grid = (
        cfg.cheap_grid_price_sek is not None
        and inputs.import_price_sek <= cfg.cheap_grid_price_sek
    )
    grid_setpoint_w = cfg.cheap_grid_allowance_w if cheap_grid else 0.0

    battery_tier = (
        cfg.battery_assist_enabled
        and inputs.import_price_sek <= cfg.battery_assist_max_price_sek
        and inputs.remaining_solar_kwh >= cfg.battery_assist_min_remaining_solar_kwh
        and inputs.battery_soc_percent > cfg.battery_assist_floor_soc
    )
    battery_allow_w = cfg.battery_assist_allowance_w if battery_tier else 0.0

    current_total_w = sum(c.current_power_w for c in active)
    headroom_w = (grid_setpoint_w - inputs.grid_w) + inputs.battery_w + battery_allow_w

    if abs(headroom_w) < cfg.deadband_w:
        target_total_w = current_total_w  # hold
    elif current_total_w < 1.0 and headroom_w > 0.0:
        # Cold-start kick: when the fleet is idle, commit the FULL available headroom
        # (not the gain-damped step) so a high-minimum charger — e.g. a 3-phase car whose
        # 6 A floor is ~4.1 kW — can actually cross its start threshold instead of being
        # stranded below it by the damping. Subsequent cycles ramp/settle via the gain.
        target_total_w = headroom_w
    else:
        target_total_w = current_total_w + cfg.gain * headroom_w

    fleet_max_w = sum(_charger_max_w(c) for c in active)
    target_total_w = max(0.0, min(fleet_max_w, target_total_w))

    why = (
        f"grid={inputs.grid_w:.0f}W batt={inputs.battery_w:.0f}W "
        f"setpoint={grid_setpoint_w:.0f}W battery_tier={battery_tier} "
        f"headroom={headroom_w:.0f}W -> target={target_total_w:.0f}W"
    )

    # --- Distribute the budget across chargers, greedy by priority ---
    commands: list[ChargerCommand] = []
    remaining_w = target_total_w
    for c in sorted(active, key=lambda x: (x.priority, x.id)):
        min_on_w = _charger_min_on_w(c, cfg)
        max_w = _charger_max_w(c)
        if remaining_w < min_on_w:
            commands.append(
                ChargerCommand(c.id, switch_on=False, set_current_a=0.0 if c.controllable else None,
                               target_power_w=0.0, reason=f"off: budget {remaining_w:.0f}<min {min_on_w:.0f}; {why}")
            )
            continue
        give_w = min(remaining_w, max_w)
        remaining_w -= give_w
        if c.controllable:
            amps = give_w / (c.voltage_v * c.phases)
            amps = max(c.min_current_a, min(c.max_current_a, amps))
            commands.append(
                ChargerCommand(c.id, switch_on=True, set_current_a=round(amps, 1),
                               target_power_w=give_w, reason=f"on {amps:.1f}A; {why}")
            )
        else:
            # Binary charger: on at its fixed draw (can't throttle).
            commands.append(
                ChargerCommand(c.id, switch_on=True, set_current_a=None,
                               target_power_w=max_w, reason=f"on (binary); {why}")
            )
    return commands
