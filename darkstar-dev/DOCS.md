# [FORK] Darkstar Energy Manager

This is the **Aksinder fork** of [ergetie/darkstar](https://github.com/ergetie/darkstar).
It tracks upstream and adds device-level controllers and pricing features on top of the
core pipeline (Aurora ML forecasts → Kepler MILP planner → Home Assistant executor).

Everything below is **fork-added** and **defaults to OFF or observe-only** — installing
this add-on changes nothing until you enable a feature in the config. (One exception: a
small always-on recorder guard that skips glitched zero-load readings — a data-quality
fix, not a behavior change.) Configure via the
add-on Web UI (Settings) or the YAML config; keys shown are the exact schema keys from
`config.default.yaml`.

---

## Deferrable smart appliances (dishwasher / washing machine)

Turnkey: give Darkstar a **power sensor** and it auto-arms when a cycle starts (power
rises), computes the cheapest *forecast* window for the whole run before a deadline
(duration-aware, unlike "is it cheap right now?" triggers), publishes its
recommendation, notifies, and detects done. **Plug actuation is not implemented yet** —
Darkstar recommends; your existing automation (or you) stays the actuator, so it can
never fight your current setup.

```yaml
deferrable_loads:
  - id: washer
    name: "Washing machine"
    enabled: true
    power_sensor: sensor.washer_power        # REQUIRED: auto-arm/run/done from draw
    switch_entity: switch.washer_plug        # read-only today: a 0 W reading while the
                                             # plug is held OFF counts as deferred, not done
    on_threshold_w: 10                       # sustained draw >= this => cycle started
    off_threshold_w: 3                       # sustained draw < this => cycle done
    start_debounce_s: 3
    done_delay_s: 300
    override_entity: input_boolean.washer_override   # optional: force run now
    deadline_mode: cheapest_within_hours     # or: hard_deadline (+ hard_deadline: "07:00")
    window_hours: 14
    energy_sensor: sensor.washer_energy      # optional cumulative kWh => accurate per-cycle energy

executor:
  deferrable_appliances:
    enabled: true
    observe_only: true          # publish state + notify only — never touches the plug.
                                # (Actuation is not implemented yet; today this flag
                                # only changes the notification wording.)
    notify_service: ""          # e.g. notify.mobile_app_phone
    publish_prefix: "darkstar_" # publishes sensor.darkstar_<id>_state
    slot_minutes: 15
```

`sensor.darkstar_<id>_state` reports `idle / armed / waiting / running` with
`recommended_action` and `recommended_start` attributes.

## Real-time EV surplus charging (`executor.ev_surplus`)

Dials each charger's current every tick so EV charging tracks an energy budget instead
of draining the home battery. Three-tier source policy: **solar surplus → cheap/negative
grid → home battery** (battery only when the price is negative *and* plenty of solar is
still forecast). Per-charger priority, min/max amps, and a manual override
`input_select` (auto / force_on / force_off).

Per-charger extras: `soc_entity` + `target_soc_entity` (cap — never overcharge),
`departure_entity` (an `input_datetime`; if the car is behind the curve to reach target
by departure, a grid-backed floor ramps up and that car overtakes the others), and
`vacation_target_soc` (`executor.ev_surplus.vacation_entity` flips the target and clears
deadlines — vacation = never grid-forced; opportunistic surplus/cheap-grid only, toward
the low vacation target).

```yaml
executor:
  ev_surplus:
    enabled: true
    pv_power_entity: sensor.total_pv_power
    grid_power_entity: sensor.grid_power          # signed, + import
    battery_power_entity: sensor.battery_power    # signed, + charge
    battery_soc_entity: sensor.battery_soc
    price_entity: sensor.spot_price               # SPOT scale (SEK/kWh)
    remaining_solar_entity: ""                    # e.g. Solcast "remaining today" (kWh)
    vacation_entity: ""                           # e.g. input_boolean.vacation_mode
    policy:
      cheap_grid_price_sek: 0.30
      cheap_grid_allowance_w: 3680
      battery_assist:
        enabled: false
        max_price_sek: 0.0          # battery helps only when spot <= this (0 = negative)
        min_remaining_solar_kwh: 8.0
        floor_soc: 40.0
    chargers:
      - id: easee
        priority: 0                 # lower = filled first
        min_current_a: 6
        max_current_a: 16
        phases: 1
        switch_entity: switch.easee_charger_enabled
        easee_device_id: "<HA device id>"   # uses easee.set_charger_dynamic_limit
        power_entity: sensor.easee_power
        plug_entity: binary_sensor.easee_plugged_in
        override_entity: input_select.darkstar_ev_easee_mode
```

> **Easee note:** current is set via `set_charger_dynamic_limit` (volatile RAM —
> flash-safe). The fork never automates the non-dynamic max/circuit limits, which
> wear the charger's flash memory.

## EV SoC estimator for cars with no API (`executor.fmb_soc_estimator`)

Dead-reckons the SoC of an EV that reports none: integrates delivered energy **up**
while charging (charger lifetime-energy counter), drifts **down** at a learned daily
consumption rate while idle, and self-calibrates that rate from each full-to-full
cycle. State persists across restarts. One-shot manual reseed via `seed_soc`, ongoing
corrections via an optional `correction_entity`, and `writeback_entity` publishes the
estimate into an `input_number` your own dashboards/automations can use.

## Thermal water tanks — hot-water level without a temperature probe

Set a water heater to `type: thermal` and Darkstar models the tank as a thermal battery
from its heating-power meter alone: energy in **up**, standing loss + a **learned**
hot-water draw rate **down**, self-calibrated from each full-to-full reheat. Publishes
`sensor.<prefix><id>_hot_water_level` (%), `_liters_remaining` (L),
`_estimated_temperature` (°C), and `_draw_today` (kWh). State persists across restarts.

```yaml
water_heaters:
  - id: main_tank
    name: "Water heater"
    enabled: true
    power_kw: 3.0
    min_kwh_per_day: 6.0
    type: thermal
    power_sensor: sensor.wh_power    # heating-element power (W); `sensor:` also works
    volume_litres: 200
    t_cold_c: 10
    t_max_c: 75
    ua_w_per_k: 2.5                  # standing-loss coefficient
    prior_draw_kw: 0.15              # draw seed until self-calibrated
    sensor_prefix: "darkstar_"
    target_entity: input_number.wh_target_temp   # per-heater control entity
```

Vacation mode can now **partition** tanks: a tank with `exclude_from_vacation: true`
keeps normal comfort heating while the rest get anti-legionella-only (e.g. a rented-out
guest house while you're away).

## Single-source pricing — follow your contract from HA helpers

Both sides of the price can be driven by HA entities, so a contract change is a helper
edit — no config redeploy. Entities accept **SEK/kWh or öre/kWh** (auto-converted by
`unit_of_measurement`).

**Import** — `import = (spot + grid transfer + energy tax) × (1 + VAT)`:

```yaml
pricing:
  vat_percent: 25.0
  fees_include_vat: true             # your fee helpers are "as billed" (VAT-inclusive):
                                     # VAT is applied to the spot only and the fees are
                                     # added as-is — no double-VAT. false (default) =
                                     # legacy VAT-exclusive fees.
  grid_transfer_fee_entity: input_number.grid_transfer_fee   # öre or SEK per kWh
  energy_tax_entity: input_number.energy_tax
```

**Export** — `export = (spot if export_includes_spot) + premium + grid benefit − fee`:

```yaml
pricing:
  export_includes_spot: true
  export_premium_entity: input_number.export_premium          # elhandlarens påslag
  export_grid_benefit_entity: input_number.export_grid_benefit  # nätnytta
  export_fee_entity: ""
```

When the **effective** export price goes negative the planner curtails/stores instead
of exporting (paying to export). An optional real-time executor clamp exists too:

```yaml
executor:
  export_curtailment:
    enabled: false                 # forces the inverter export limit to 0 W below the
    threshold_sek_per_kwh: 0.0     # threshold; restores it above. Leave OFF if another
    restore_limit_w: 0.0           # integration owns your export-limit entity.
```

## Multi-inverter sites — correct AC-limit modeling

On an AC-coupled multi-inverter site (e.g. a hybrid + a PV-only inverter),
`system.inverter.max_ac_power_kw` used to subtract **all** PV from the battery
inverter's AC headroom — zeroing battery discharge on sunny days. Tag the arrays that
are physically on the hybrid/battery inverter and only their share competes for its AC
bus:

```yaml
system:
  inverter:
    max_ac_power_kw: 10.0           # the hybrid inverter's AC rating
  solar_arrays:
    - name: "PV-only inverter array"
      kwp: 11.4
      azimuth: 230
      tilt: 30                      # untagged: not on the battery inverter
    - name: "Hybrid string"
      kwp: 4.5
      azimuth: 230
      tilt: 30
      on_battery_inverter: true     # counts against max_ac_power_kw
```

Omit the tags on a single-inverter system — behaviour is unchanged.

## Load priority — willingness-to-pay (WTP) tiers

A unified "this load matters more than that one" layer. Each controllable load gets a
tier with a reservation price (SEK/kWh); the planner runs a load only while its WTP
meets the marginal energy price — a spa defers through price peaks and soaks surplus,
hot water keeps its guarantee.

```yaml
load_priority:
  enabled: true
  tiers:
    important: { base_wtp_sek_per_kwh: 3.0, urgency_wtp_sek_per_kwh: 5.0 }
    comfort:   { base_wtp_sek_per_kwh: 0.4, urgency_wtp_sek_per_kwh: 0.6 }
  loads:
    main_tank:
      tier: important
      wtp_percentile: 50   # DYNAMIC cap: recomputed each plan as this percentile of the
                           # next 24 h of import prices — never starves on an expensive
                           # day, always refuses the day's priciest hours.
```

Dynamic-percentile heaters keep the daily-minimum reliability floor (they can't
defer-forever); static-WTP loads may skip a day when energy costs more than their WTP.

## Phase-aware load modeling

- **`phase_observer`** (read-only): learns each metered device's electrical phase by
  correlating its power changes against the per-phase grid meter, reconstructs the
  per-phase house-load split, and publishes `sensor.darkstar_phase_a/b/c_load`,
  `sensor.darkstar_phase_imbalance`, and `sensor.darkstar_<device>_phase`. Learns an
  hour-of-day phase profile the planner's realism simulation uses.
- **`phase_aware`** (planner, opt-in): prices the hidden cost of a heavy phase (buy
  high on one phase, sell low on the others) and discharges the battery to cover it —
  **only when economic**. Inert until the observer has learned a phase split.

## Excess-PV custom sink — climate support + price gating

The excess-PV sink can drive a `climate.*` entity (e.g. soak surplus as air-con cooling)
and run **alongside** `water_heater_boost` instead of replacing it. Gate it by export
price so it only fires when exporting pays little or nothing:

```yaml
executor:
  excess_pv:
    sink: water_heater_boost          # primary sink stays
    custom_entity:
      enabled: true                   # run this sink alongside the primary
      entity: climate.guest_house
      price_ceiling_sek_per_kwh: 0.20 # only on low/negative export-price surplus hours
      climate_mode: "cool"
      target_temp: "22"
      comfort_min_temp: "20"          # anti-overcool floor
      power_kw: 1.0
```

## Smaller additions

- **`unknown_load.enabled`** — publishes the unmetered residual
  (total − metered controllable) as `sensor.darkstar_unknown_load`, a guide for which
  load is worth metering next. Excludes grid-fed EV charging and holds the last value
  when the total-load source glitches.
- **`battery_value.enabled`** — continuous terminal stored-energy value: credits energy
  left at horizon end at a conservative fraction of the cheapest upcoming import price,
  so the planner holds cheap energy instead of dumping it at low prices. Never makes it
  buy grid just to inflate SoC.
- **EV come-home prediction** (`ev_chargers[].come_home`) — pre-positions a soft, capped
  battery buffer when the car is likely to come home (learned arrival profile), plus a
  robust home-zone gate (`home_entity`, `home_radius_km`, `home_grace_minutes`) so a car
  charging elsewhere never leaks into the plan.
- **Recorder guards** — glitched/frozen total-load reads (e.g. Modbus dropouts) are
  skipped instead of poisoning the learning data.

---

## Deploy notes

- Dev builds auto-version (`dev-YYYYMMDD.HHMM`); the **Changelog** tab lists every build.
- If an update doesn't appear after a build: Add-on Store → ⋮ → **Check for updates**
  (the Supervisor's store cache can lag).
- Full design notes live in the repo under `docs/designs/`.
