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
    # Re-enable hysteresis for the battery tier: once the tier has dropped out at the
    # floor, SoC must recover to floor + this before it re-engages. Without it, SoC
    # dithering right at the floor flips the +/-allowance in and out of the headroom
    # every tick, and the resulting start/stop commands bypass the write-guard.
    battery_assist_soc_hysteresis: float = 3.0
    # Controller dynamics.
    gain: float = 0.5  # fraction of the headroom applied per cycle (stability vs speed)
    deadband_w: float = 250.0  # hold when |headroom| is within this (anti-jitter)
    # Main-fuse guard: usable ampere budget PER PHASE (fuse rating minus margin, e.g.
    # 25 - 2 = 23). None => guard disabled (legacy). The clamp runs in the greedy order,
    # so urgent deadline floors keep their budget and topup cars are shed first; it
    # TRUMPS floors — a grid-backed guarantee must never blow the main fuse.
    fuse_budget_a: float | None = None
    # Quantized-control stability: the effective fleet deadband is widened to
    # K x (largest 1-step power quantum among commanded-ON controllable chargers),
    # so a charger whose smallest move exceeds the band can never limit-cycle
    # between two adjacent amp levels (the 2026-07 Tesla 14<->16 A wake-storm).
    # NOTE: at gain 0.5 the loop has an implicit dead zone of ~one quantum from
    # mid-cell, but round-to-nearest still flips at cell boundaries with
    # arbitrarily small headroom (~13% of states in simulation) — K <= 1.0 is NOT
    # inert, but K = 1.5 is what measurably kills the write churn. 0 disables.
    quantum_deadband_k: float = 1.5
    # Runtime Schmitt quantizer: suppress a +/-1-step rewrite until the RAW
    # (unsnapped) amp target has moved at least this fraction of a step away from
    # the currently written value. Kills midpoint dither and the config-voltage
    # vs real-voltage mismatch churn (e.g. 230 V configured, 222 V actual).
    schmitt_fraction: float = 0.7
    min_charge_current_a: float = 5.0  # global safety floor; per-charger min_current_a wins if higher
    default_voltage_v: float = 230.0
    # Amp grid for commanded currents. 1 A gives the finest tracking both cars
    # support (Tesla 5-16 A, Easee 6-16 A); anti-churn is handled by the quantum
    # deadband + Schmitt quantizer + per-charger write guards, NOT by a coarse step.
    current_step_a: float = 1.0
    # On/off hysteresis: a charger that's OFF needs this much MORE than its min before it
    # starts; once ON it keeps running down to its true min. Stops flapping at the threshold.
    start_hysteresis: float = 0.15  # fraction of min_on power (e.g. 15% headroom to start)


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
    # Manual override (read from an HA input_select per charger). "auto" = surplus control;
    # "force_on" = charge at max regardless of surplus; "force_off" = never charge.
    override: str = "auto"
    # --- Departure / target-SoC awareness (all optional; None => behave as before) ---
    # The car's current SoC. For the FMB this is the dead-reckoned input_number.fmb_soc; for
    # the Tesla it's the real sensor.white_betty_battery_level. None => no cap and no deadline
    # logic (pure opportunistic surplus follow, the legacy behaviour).
    soc_percent: float | None = None
    # Desired SoC by the deadline. soc >= target => stop (never overcharge). None => no target.
    target_soc_percent: float | None = None
    # The GUARANTEE band's upper SoC — what the grid-backed deadline floor charges toward.
    # Distinct from target_soc_percent (the comfort cap): the Tesla's cap is the car's own
    # charge_limit (90) but its weekday-morning guarantee is only ~40. None => the deadline
    # floor falls back to target_soc_percent (legacy: one number served both roles).
    floor_soc_percent: float | None = None
    # Comfort-demotion threshold: at/above this SoC the charger yields the SURPLUS class
    # to every non-demoted charger (owner: "FMB till 150 km, sedan Teslan, sedan FMB mer").
    # It keeps charging — from whatever surplus remains — up to target_soc_percent.
    # Floors are unaffected (need outranks comfort). None => never demoted.
    comfort_soc_percent: float | None = None
    # Which grid phases this charger's draw lands on, e.g. ("a",) for a 1-phase car or
    # ("a", "b", "c") for a 3-phase one. EMPTY = unknown => the fuse guard budgets the
    # charger's amps on EVERY phase (conservative — over-restrictive only under real
    # congestion, never unsafe). NOTE: this maps the CAR'S DRAW, not the charger box's
    # wiring — the 3-phase-wired Easee carries the FMB on one leg.
    phase_map: tuple[str, ...] = ()
    # This tick's PLANNER floor (W): kepler's price-placed charging for the current slot,
    # already soc-gated and continuity-held by the runtime. Merged with the deadline floor
    # via max() (never sum). 0 = no plan floor. In the ordering, plan floors rank BEHIND
    # deadline floors (need before price-optimality) but ahead of the surplus class.
    plan_floor_w: float = 0.0
    # Usable battery capacity, to convert an SoC gap into energy. 0 => no deadline floor.
    capacity_kwh: float = 0.0
    # Hours until the departure deadline. None / <=0 => no grid-backed deadline floor (the car
    # only ever charges from opportunistic surplus toward its target).
    deadline_hours: float | None = None
    # Plug->battery charge efficiency, for sizing the deadline floor from the SoC gap.
    charge_efficiency: float = 0.9
    # Runtime's last COMMANDED on/off state. None => infer from measured power. Using
    # the commanded state (not the measured watts) means a car that is slow to start
    # drawing still counts as ON for the quantum band / hysteresis / start kick.
    commanded_on: bool | None = None
    # Min-OFF dwell active: this charger was recently stopped and must not restart
    # yet (anti-flap). Deadline floors are exempt — grid-backed forcing punches through.
    start_inhibited: bool = False


@dataclass
class EVSurplusInputs:
    """Live measurements for one control cycle."""

    pv_w: float
    grid_w: float  # signed: + import, - export
    battery_w: float  # signed: + charging, - discharging
    battery_soc_percent: float
    import_price_sek: float
    remaining_solar_kwh: float
    # Previous cycle's battery-tier state (runtime-tracked) — drives the SoC
    # re-enable hysteresis in battery_tier_active().
    battery_tier_active_prev: bool = False
    # Grid phase current magnitudes in AMPERE, keyed by phase name (e.g. {"a": 12.3, ...}).
    # This is the MAIN-FUSE current (direction-blind |A| — export blows fuses too).
    # Empty dict = no fresh readings; the pure fuse clamp then allows NO increases
    # (hold-or-reduce). The runtime's stale fail-safe handles full sensor loss with
    # explicit stops BEFORE compute is ever reached.
    phase_currents_a: dict[str, float] = field(default_factory=dict)
    chargers: list[ChargerState] = field(default_factory=lambda: [])


@dataclass
class ChargerCommand:
    """Desired actuation for one charger this cycle."""

    id: str
    switch_on: bool
    set_current_a: float | None  # None => binary charger / leave current unchanged
    target_power_w: float
    reason: str
    # True when the 25 A/phase fuse guard reduced this command below its wanted level.
    # The write guard treats such REDUCTIONS like stops (bypass min_step/min_interval):
    # a fuse-relief write must never wait out a pacing interval.
    fuse_limited: bool = False
    # Unsnapped amp target (pre-step-grid), for the runtime Schmitt quantizer.
    raw_amps: float | None = None


@dataclass
class WriteGuardConfig:
    """Rate-limit charge-current writes.

    Easee's ``set_charger_dynamic_limit`` (dynamicChargerCurrent) is stored in VOLATILE
    memory and is flash-safe — it is the non-dynamic ``max``/circuit limits that wear the
    flash, and Darkstar never writes those. But Easee still warns that "some cars might get
    upset if the current is changed too frequently", and Tesla current writes hit the car's
    API / wake it. So we only actuate when the target moves by a real step AND a minimum
    interval has elapsed. Exception: a STOP (target 0 / off) is always allowed immediately
    so home-battery protection is never delayed.

    Ref: developer.easee.com/docs/current-limits-and-control (dynamic = volatile, flash-safe;
    non-dynamic limits wear flash; avoid over-frequent current changes for the car's sake).
    """

    min_step_a: float = 2.0  # only rewrite when the target differs by >= this many amps
    min_interval_s: float = 90.0  # ...and at least this long since the last write
    # Direction-aware overrides (None => fall back to min_interval_s). Increases can
    # be paced hard (a Tesla up-write wakes the car / costs an API call) while
    # decreases stay fast (backing off protects the home battery).
    min_interval_up_s: float | None = None
    min_interval_down_s: float | None = None


def should_write_current(
    last_a: float | None,
    last_write_ts: float | None,
    new_a: float,
    now_ts: float,
    cfg: WriteGuardConfig,
    fuse_relief: bool = False,
) -> bool:
    """True if the new charge-current target should be written this cycle.

    First write always proceeds. A stop (new_a <= 0) is always allowed (safety — never
    delay backing off the home battery). A fuse-guard REDUCTION (``fuse_relief`` with
    new < last) gets the same safety class: an overloaded phase must not wait out
    min_step_a or a pacing interval (an Easee sub-2A trim would otherwise be dropped
    and a Tesla shed deferred 90 s while the fuse cooks). Otherwise require both a
    step >= ``min_step_a`` and >= ``min_interval_s`` since the last write.
    """
    if last_a is None or last_write_ts is None:
        return True
    if fuse_relief and new_a < last_a:
        return True  # fuse-overload relief — act immediately, any step size
    if new_a <= 0.0 < last_a:
        return True  # stopping / dropping to zero — act immediately
    if last_a <= 0.0 < new_a:
        return True  # starting from a stop — act immediately (symmetric to the stop bypass) so a
        # deadline-forced or surplus restart isn't stranded at 0 for up to min_interval_s. Rapid
        # on/off flapping is already prevented upstream by the start hysteresis in compute_ev_surplus.
    if abs(new_a - last_a) < cfg.min_step_a:
        return False
    interval = cfg.min_interval_up_s if new_a > last_a else cfg.min_interval_down_s
    if interval is None:
        interval = cfg.min_interval_s
    return (now_ts - last_write_ts) >= interval


def _export_credit_a(grid_w: float, voltage_v: float = 230.0) -> float:
    """Per-phase ampere credit for net EXPORT (grid_w < 0).

    The |A| phase readings are direction-blind, but BOTH site inverters push
    symmetric 3-phase, so net export contributes |grid_w|/(3V) amps per phase that
    ADDED CONSUMPTION removes 1:1. Without this credit an export-loaded phase
    deadlocks EV starts (an OFF car never changes the reading, so there is no
    iteration path) and the battery cap ratchets itself to zero (reducing charge
    RAISES export). The credit never exceeds the reading (clamped at use sites).
    """
    return max(0.0, -grid_w) / (3.0 * voltage_v)


def fuse_battery_charge_cap_w(
    phase_currents_a: dict[str, float],
    battery_charge_w: float,
    budget_a: float,
    voltage_v: float = 230.0,
    grid_w: float = 0.0,
    ev_alloc_a: dict[str, float] | None = None,
) -> float:
    """Max battery charge SETPOINT (W) that keeps every phase within the fuse budget.

    The battery is the fuse guard's only non-EV shed lever: planner grid-charging
    (9.5 kW ≈ 13.8 A on all phases) co-scheduled with the 1-phase VVB block can sit at
    30+ A on the VVB phase with zero cars to clamp. The battery charges 3-phase
    SYMMETRIC, and its present grid draw is already IN the meter readings, so the cap
    is iterative: allow growth up to the measured min-phase headroom; when a phase
    goes over budget on the IMPORT side the setpoint is pulled below the present
    charge level. Export-side overloads are credited via _export_credit_a — pulling
    the charge DOWN during export would raise export 1:1 (a positive-feedback
    ratchet, review-caught). ``ev_alloc_a`` is the ampere increase the EV clamp
    granted THIS tick from the same meter snapshot — without subtracting it the two
    levers double-spend the same headroom. Empty readings => 0 (fail safe: no grid
    charging on blind sensors; PV then flows to export instead, which is loss-free).
    """
    if not phase_currents_a:
        return 0.0
    credit = _export_credit_a(grid_w, voltage_v)
    alloc = ev_alloc_a or {}
    headroom_a = min(
        budget_a - max(0.0, abs(i) - credit) - alloc.get(p, 0.0)
        for p, i in phase_currents_a.items()
    )
    return max(0.0, max(0.0, battery_charge_w) + headroom_a * 3.0 * voltage_v)


def _fuse_delta_allowed_a(
    c: ChargerState,
    inputs: EVSurplusInputs,
    cfg: EVSurplusConfig,
    alloc: dict[str, float],
) -> tuple[float | None, tuple[str, ...]]:
    """Allowed per-phase ampere INCREASE for this charger, or (None, ()) = guard off.

    Blind (no/partial readings) => 0.0: hold-or-reduce only. Export-aware via
    _export_credit_a. Shared by the force_on path and pass 2 — a manual comfort
    override does not outrank the main fuse when a deadline guarantee does not.
    """
    if cfg.fuse_budget_a is None:
        return None, ()
    phases = c.phase_map or tuple(sorted(inputs.phase_currents_a.keys()))
    if not inputs.phase_currents_a or any(
        p not in inputs.phase_currents_a for p in phases
    ):
        return 0.0, phases
    credit = _export_credit_a(inputs.grid_w, c.voltage_v)
    return min(
        cfg.fuse_budget_a
        - max(0.0, inputs.phase_currents_a[p] - credit)
        - alloc.get(p, 0.0)
        for p in phases
    ), phases


def _charger_max_w(c: ChargerState) -> float:
    return c.max_current_a * c.voltage_v * c.phases


def _charger_min_on_w(c: ChargerState, cfg: EVSurplusConfig) -> float:
    return max(c.min_current_a, cfg.min_charge_current_a) * c.voltage_v * c.phases


def _quantum_w(c: ChargerState, cfg: EVSurplusConfig) -> float:
    """Power of one amp-step for this charger (its smallest possible move)."""
    return max(0.0, cfg.current_step_a) * c.phases * c.voltage_v


def _is_on(c: ChargerState) -> bool:
    """Commanded state when known (start-lag safe), else infer from measured power."""
    return c.commanded_on if c.commanded_on is not None else c.current_power_w > 100.0


def battery_tier_active(inputs: EVSurplusInputs, cfg: EVSurplusConfig) -> bool:
    """Battery-assist tier gate, with SoC re-enable hysteresis.

    Once the tier has dropped out at the floor, SoC must recover to
    floor + battery_assist_soc_hysteresis before it re-engages — SoC dithering
    right at the floor must not flip the +/-allowance every tick (those
    start/stop commands bypass the write-guard).
    """
    floor = cfg.battery_assist_floor_soc + (
        0.0 if inputs.battery_tier_active_prev else cfg.battery_assist_soc_hysteresis
    )
    return (
        cfg.battery_assist_enabled
        and inputs.import_price_sek <= cfg.battery_assist_max_price_sek
        and inputs.remaining_solar_kwh >= cfg.battery_assist_min_remaining_solar_kwh
        and inputs.battery_soc_percent > floor
    )


def _soc_at_or_above_target(c: ChargerState) -> bool:
    """True when the car has a known SoC at/above its target — stop (never overcharge)."""
    return (
        c.soc_percent is not None
        and c.target_soc_percent is not None
        and c.soc_percent >= c.target_soc_percent
    )


def _deadline_required_w(c: ChargerState, cfg: EVSurplusConfig) -> float:
    """Grid-backed power floor needed to reach the FLOOR SoC by the deadline, else 0.

    This is the average plug power required from *now*::

        required = (floor - soc)/100 * capacity / efficiency / hours_remaining

    The floor target is ``floor_soc_percent`` (the guarantee band), falling back to
    ``target_soc_percent`` when unset. Using the comfort cap here was a real bug: the
    Tesla's cap is the car's own charge_limit (90), so a weekday deadline grid-forced
    price-blind charging toward 90 every night instead of stopping at the ~40 guarantee.

    Returns 0 (no forcing) when any input is missing OR when ``required`` is still below the
    charger's minimum on-power: a sub-minimum requirement means there is plenty of time, so
    we let opportunistic solar handle it and don't pay for grid yet. As the deadline nears
    with the car still under the floor, ``required`` climbs; once it crosses the minimum we
    begin forcing grid, and it ramps toward the charger max as time runs out. Capped at the
    charger max. The floor is honoured regardless of surplus, so a deadline car "overtakes"
    the others.
    """
    floor = c.floor_soc_percent if c.floor_soc_percent is not None else c.target_soc_percent
    # The floor may never exceed the comfort cap: the cap check stops charging at target,
    # so a mis-configured floor above it would size grid forcing toward an unreachable SoC.
    if floor is not None and c.target_soc_percent is not None:
        floor = min(floor, c.target_soc_percent)
    if (
        c.soc_percent is None
        or floor is None
        or c.capacity_kwh <= 0.0
        or c.deadline_hours is None
        or c.deadline_hours <= 0.0
    ):
        return 0.0
    soc_gap = max(0.0, floor - c.soc_percent) / 100.0
    if soc_gap <= 0.0:
        return 0.0
    energy_at_plug_kwh = soc_gap * c.capacity_kwh / max(0.05, c.charge_efficiency)
    required_w = energy_at_plug_kwh * 1000.0 / c.deadline_hours
    if required_w < _charger_min_on_w(c, cfg):
        return 0.0  # enough time left — wait for cheaper/solar energy instead of forcing grid
    return min(required_w, _charger_max_w(c))


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

    commands: list[ChargerCommand] = []

    # --- Manual overrides first (the user overrules the auto control) ---
    # force_off: never charge. force_on: charge at max regardless of surplus. Their draw
    # is reflected in the live grid/battery, so the remaining "auto" chargers see less
    # headroom and back off accordingly — no double counting.
    forced_off = [c for c in active if c.override == "force_off"]
    forced_on = [c for c in active if c.override == "force_on"]
    auto = [c for c in active if c.override not in ("force_off", "force_on")]

    for c in forced_off:
        commands.append(
            ChargerCommand(c.id, switch_on=False, set_current_a=0.0 if c.controllable else None,
                           target_power_w=0.0, reason="override: force_off")
        )
    # Fuse-guard allocation ledger for the WHOLE cycle: force_on grants are charged
    # here first, so the auto chargers in pass 2 see the remaining budget.
    fuse_alloc_a: dict[str, float] = {}
    for c in forced_on:
        # force_on wants max — but the main fuse outranks a manual comfort override
        # exactly as it outranks a deadline guarantee (review-caught: an unclamped
        # force_on at 16 A on the VVB phase was a ~35 A stack).
        amps = c.max_current_a
        fuse_capped = False
        delta_a, fuse_phases = _fuse_delta_allowed_a(c, inputs, cfg, fuse_alloc_a)
        if delta_a is not None:
            per_phase_now_a = c.current_power_w / (c.voltage_v * c.phases)
            cap_a = max(0.0, per_phase_now_a + delta_a)
            if cap_a < amps:
                amps = cap_a
                fuse_capped = True
            step_f = max(0.0, cfg.current_step_a)
            if fuse_capped and step_f > 0:
                amps = (amps // step_f) * step_f
            if amps < c.min_current_a or (not c.controllable and fuse_capped):
                commands.append(
                    ChargerCommand(c.id, switch_on=False,
                                   set_current_a=0.0 if c.controllable else None,
                                   target_power_w=0.0,
                                   reason="off: fuse (force_on denied)",
                                   fuse_limited=True)
                )
                continue
            for p in fuse_phases:
                fuse_alloc_a[p] = fuse_alloc_a.get(p, 0.0) + max(
                    0.0, amps - per_phase_now_a
                )
        commands.append(
            ChargerCommand(c.id, switch_on=True,
                           set_current_a=amps if c.controllable else None,
                           target_power_w=amps * c.voltage_v * c.phases
                           if c.controllable else _charger_max_w(c),
                           reason="override: force_on"
                           + (" (fuse-capped)" if fuse_capped else " (max)"),
                           fuse_limited=fuse_capped)
        )
    if not auto:
        return commands

    # --- Tier gating (only the auto-managed chargers share the surplus budget) ---
    cheap_grid = (
        cfg.cheap_grid_price_sek is not None
        and inputs.import_price_sek <= cfg.cheap_grid_price_sek
    )
    grid_setpoint_w = cfg.cheap_grid_allowance_w if cheap_grid else 0.0

    battery_tier = battery_tier_active(inputs, cfg)
    battery_allow_w = cfg.battery_assist_allowance_w if battery_tier else 0.0

    # --- SoC caps FIRST -----------------------------------------------------------
    # A car at/above its target is done (never overcharge). Detected BEFORE the fleet
    # total so a just-capped charger's measured watts are not redistributed to the
    # others while the meter still shows its (stopping) draw — that redistribution
    # produced an instant multi-amp jump on the remaining charger followed by a
    # write-per-interval walk-down (each one a Tesla wake). Excluding it makes the
    # handoff a gain-damped ramp-up instead: slower to reabsorb (the home battery
    # buffers meanwhile) but robust to actuation lag.
    capped = [c for c in auto if _soc_at_or_above_target(c)]
    capped_ids = {c.id for c in capped}
    for c in capped:
        commands.append(
            ChargerCommand(c.id, switch_on=False, set_current_a=0.0 if c.controllable else None,
                           target_power_w=0.0,
                           reason=f"cap: soc {c.soc_percent:.0f}>=target {c.target_soc_percent:.0f}")
        )
    chargeable = [c for c in auto if c.id not in capped_ids]
    if not chargeable:
        return commands

    current_total_w = sum(c.current_power_w for c in chargeable)
    headroom_w = (grid_setpoint_w - inputs.grid_w) + inputs.battery_w + battery_allow_w

    # Quantum-aware effective deadband: never demand a move smaller than the largest
    # single amp-step among the chargers actually running (see quantum_deadband_k).
    on_quanta = [_quantum_w(c, cfg) for c in chargeable if c.controllable and _is_on(c)]
    eff_deadband_w = max(cfg.deadband_w, cfg.quantum_deadband_k * max(on_quanta, default=0.0))

    if abs(headroom_w) < eff_deadband_w:
        target_total_w = current_total_w  # hold
    elif current_total_w < 1.0 and headroom_w > 0.0:
        # Cold-start kick: when the fleet is idle, commit the FULL available headroom
        # (not the gain-damped step) so a high-minimum charger — e.g. a 3-phase car whose
        # 6 A floor is ~4.1 kW — can actually cross its start threshold instead of being
        # stranded below it by the damping. Subsequent cycles ramp/settle via the gain.
        target_total_w = headroom_w
    else:
        target_total_w = current_total_w + cfg.gain * headroom_w

    fleet_max_w = sum(_charger_max_w(c) for c in chargeable)
    target_total_w = max(0.0, min(fleet_max_w, target_total_w))

    why = (
        f"grid={inputs.grid_w:.0f}W batt={inputs.battery_w:.0f}W "
        f"setpoint={grid_setpoint_w:.0f}W battery_tier={battery_tier} "
        f"headroom={headroom_w:.0f}W band={eff_deadband_w:.0f}W -> target={target_total_w:.0f}W"
    )

    # --- Deadline floors --------------------------------------------------------
    # A car with a target + deadline gets a grid-backed FLOOR (see
    # _deadline_required_w): the avg power it must pull now to make the deadline.
    # Floors are honoured regardless of surplus, so a deadline car "overtakes" the
    # others; free surplus is applied first so grid is only paid for the gap.
    # Two floor sources per charger: the grid-backed deadline guarantee and the
    # planner's price-placed slot (plan_floor_w, pre-gated by the runtime). Merged
    # with max() — NEVER sum (the same energy need must not be double-counted).
    deadline_w = {c.id: _deadline_required_w(c, cfg) for c in chargeable}
    floor_w = {c.id: max(deadline_w[c.id], c.plan_floor_w) for c in chargeable}

    # Deadline cars overtake (sort key 0), then by priority. Used for BOTH surplus
    # distribution (free energy goes to the deadline car first) and emission order.
    # Within the floor class, DEADLINE URGENCY (fewest hours left) outranks priority: a
    # commuter car whose 07:30 guarantee is at risk must not be starved by another car's
    # (future) plan floor or configured priority under fuse/surplus scarcity. Floors
    # without a deadline sort behind every deadline floor (urgency inf), then priority.
    # Within the SURPLUS class, comfort-demotion ranks before priority: a car at/above
    # its comfort_soc yields to every non-demoted car, then keeps charging on what's left.
    def _order_key(x: ChargerState) -> tuple[int, float, int, int, str]:
        has_floor = floor_w[x.id] > 0.0
        urgency = (
            x.deadline_hours
            if has_floor and x.deadline_hours is not None
            else float("inf")
        )
        demoted = int(
            not has_floor
            and x.comfort_soc_percent is not None
            and x.soc_percent is not None
            and x.soc_percent >= x.comfort_soc_percent
        )
        return (0 if has_floor else 1, urgency, demoted, x.priority, x.id)

    order = sorted(chargeable, key=_order_key)

    # --- Per-charger start kick (multi-charger deadlock fix) ---------------------
    # With one charger already at max, the gain-damped step can leave a second, OFF
    # charger permanently stranded below its start threshold even though the FULL
    # undamped headroom would clear it (verified: Easee@max + Tesla OFF stayed off
    # for any surplus < ~11.7 kW — ~3-4 kW wasted indefinitely). If the undamped
    # allocation would start an OFF, non-dwell-inhibited charger that the damped one
    # would not, commit the undamped total this cycle; later cycles settle via gain.
    undamped_total_w = max(0.0, min(fleet_max_w, current_total_w + headroom_w))
    if undamped_total_w > target_total_w:

        def _greedy_shares(total_w: float) -> dict[str, float]:
            shares: dict[str, float] = {}
            rem = total_w
            for cc in order:
                give = min(rem, _charger_max_w(cc))
                shares[cc.id] = give
                rem -= give
            return shares

        damped_shares = _greedy_shares(target_total_w)
        undamped_shares = _greedy_shares(undamped_total_w)
        for c in order:
            if _is_on(c) or c.start_inhibited:
                continue
            start_thr_w = _charger_min_on_w(c, cfg) * (1.0 + cfg.start_hysteresis)
            if undamped_shares[c.id] >= start_thr_w > damped_shares[c.id]:
                target_total_w = undamped_total_w
                why += f" kick={c.id}"
                break

    # Pass 1: hand out the opportunistic surplus greedily in that order.
    surplus_share = {c.id: 0.0 for c in chargeable}
    remaining_w = target_total_w
    for c in order:
        give = min(remaining_w, _charger_max_w(c))
        surplus_share[c.id] = give
        remaining_w -= give

    # Pass 2: each car charges to max(free surplus share, grid-backed deadline floor).
    step = max(0.0, cfg.current_step_a)
    # (fuse_alloc_a was seeded by the force_on grants above — one shared ledger:
    # ampere increases GRANTED this cycle, per phase. The meter already carries every
    # charger's PRESENT draw, so only deltas are tracked; a 3-phase (or unknown-phase,
    # conservative) charger's grant consumes budget on every phase in its map.)
    for c in order:
        max_w = _charger_max_w(c)
        min_on_w = _charger_min_on_w(c, cfg)
        forced = floor_w[c.id] > 0.0
        target_w = min(max_w, max(surplus_share[c.id], floor_w[c.id]))

        # --- 25 A/phase fuse clamp (runs BEFORE the on/off thresholds so a capped
        # target below min-on falls through to OFF — even for a deadline floor:
        # fuse safety trumps the punch-through). ---
        fuse_capped = False
        per_phase_now_a = c.current_power_w / (c.voltage_v * c.phases)
        delta_allowed_a, fuse_phases = _fuse_delta_allowed_a(c, inputs, cfg, fuse_alloc_a)
        if delta_allowed_a is not None:
            cap_w = max(0.0, per_phase_now_a + delta_allowed_a) * c.voltage_v * c.phases
            if cap_w < target_w:
                target_w = cap_w
                fuse_capped = True
        # A binary charger cannot throttle: unless its FULL draw fits, it must be off.
        if fuse_capped and not c.controllable and target_w < max_w:
            target_w = 0.0
        # Hysteresis: an OFF charger needs extra headroom to start; once ON it runs down to
        # its true min. A deadline floor is exempt — it must run even with no surplus.
        currently_on = _is_on(c)
        # Min-OFF dwell: a recently stopped charger may not restart yet (anti-flap).
        # Deadline floors punch through — grid-backed forcing must never be delayed.
        if not currently_on and c.start_inhibited and not forced:
            commands.append(
                ChargerCommand(c.id, switch_on=False, set_current_a=0.0 if c.controllable else None,
                               target_power_w=0.0,
                               reason=f"off: start dwell; {why}")
            )
            continue
        threshold_w = min_on_w if currently_on else min_on_w * (1.0 + cfg.start_hysteresis)
        if target_w < threshold_w and not (forced and target_w >= min_on_w):
            # NOTE: a fuse cap below min-on lands here even for a deadline floor —
            # the punch-through requires target_w >= min_on_w, which the cap denies.
            off_tag = "fuse" if fuse_capped else f"target {target_w:.0f}<thr {threshold_w:.0f}"
            commands.append(
                ChargerCommand(c.id, switch_on=False, set_current_a=0.0 if c.controllable else None,
                               target_power_w=0.0,
                               reason=f"off: {off_tag}; {why}",
                               fuse_limited=fuse_capped)
            )
            continue
        if forced and surplus_share[c.id] < floor_w[c.id]:
            tag = "deadline" if deadline_w[c.id] >= c.plan_floor_w else "plan"
        else:
            tag = "surplus"
        if fuse_capped:
            tag += "+fuse"
        if c.controllable:
            raw_amps = target_w / (c.voltage_v * c.phases)
            amps = raw_amps
            # Snap to the step grid, then clamp. Churn suppression is the quantum
            # band + the runtime Schmitt quantizer (raw_amps below), not a coarse step.
            # A fuse-capped target snaps DOWN — rounding up would exceed the budget.
            if step > 0:
                amps = (amps // step) * step if fuse_capped else round(amps / step) * step
            amps = max(c.min_current_a, min(c.max_current_a, amps))
            if cfg.fuse_budget_a is not None and fuse_phases:
                granted_delta_a = max(0.0, amps - per_phase_now_a)
                for p in fuse_phases:
                    fuse_alloc_a[p] = fuse_alloc_a.get(p, 0.0) + granted_delta_a
            commands.append(
                ChargerCommand(c.id, switch_on=True, set_current_a=round(amps, 1),
                               target_power_w=amps * c.voltage_v * c.phases,
                               reason=f"on {amps:.1f}A ({tag}); {why}",
                               fuse_limited=fuse_capped,
                               raw_amps=round(raw_amps, 2))
            )
        else:
            # Binary charger: on at its fixed draw (can't throttle).
            if cfg.fuse_budget_a is not None and fuse_phases:
                granted_delta_a = max(
                    0.0, max_w / (c.voltage_v * c.phases) - per_phase_now_a
                )
                for p in fuse_phases:
                    fuse_alloc_a[p] = fuse_alloc_a.get(p, 0.0) + granted_delta_a
            commands.append(
                ChargerCommand(c.id, switch_on=True, set_current_a=None,
                               target_power_w=max_w, reason=f"on (binary, {tag}); {why}")
            )
    return commands
