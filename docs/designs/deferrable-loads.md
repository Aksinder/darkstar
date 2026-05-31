# Blueprint: Generic Deferrable Loads (dishwasher, washing machine, …)

**Objective:** Let Darkstar co-optimise household deferrable appliances (dishwasher,
washing machine, and similar known-draw loads) inside the existing Kepler MILP —
placing each run in the globally cheapest window (free PV surplus → battery → cheap
grid) instead of the current price-only HA automations, while respecting a deadline.

**Status:** Draft / for review. No code yet.

**Constraints:**
- 100% local. No new external dependencies.
- Reuse the existing deferrable-load machinery (`WaterHeaterInput` / `EVChargerInput`
  → adapter → Kepler). Do **not** build a second optimiser that competes with Darkstar
  for the shared battery/grid/PV resource.
- Device I/O stays in Home Assistant; Darkstar makes the *decision* and toggles an HA
  control entity, exactly like the water-heater bridge today.
- Opt-in: a global `deferrable_loads.enabled` toggle (default off) and per-load `enabled`.

---

## 1. Decisions (locked with the user)

| # | Question | Decision |
|---|----------|----------|
| 1 | Interruptible? | **No** — dishwasher & washing machine run as **one contiguous block** (non-preemptible). |
| 2 | Deadline model | **Both**, selectable per load: hard `deadline` (finish by HH:MM) *or* soft `cheapest_within_hours: X`. |
| 3 | How a run is requested | **Auto-detect** a queued/started cycle (the user already has HA detection code) — no manual "request" button required. |
| 4 | Is native `dishwasher` a separate machine? | **No** — same physical dishwasher. The Swedish `diskmaskin_*` Shelly stack is the *old* (disabled) generation; the English `dishwasher_*` stack (Home Connect-style: `binary_sensor.dishwasher_running`, `sensor.dishwasher_status`, cost tracking) is the *current* one. |

---

## 2. Current state in this home (from HA inventory)

**Dishwasher** — controlled today by a homegrown HA scheduler:
- Relay/metering: `switch.diskmaskin` (Shelly), `sensor.diskmaskin_energy` (≈9.8 kWh
  lifetime). **Note:** `sensor.diskmaskin_power` reads ≈0 W — real run state now comes
  from the native integration, not the plug.
- Live stack (active): `automation.dishwasher_start_detection`,
  `automation.dishwasher_start_when_cheap`, `automation.dishwasher_done`,
  `binary_sensor.dishwasher_running`, `sensor.dishwasher_status`,
  `input_boolean.dishwasher_active` / `_waiting`, `input_datetime.dishwasher_started`,
  plus `sensor.dishwasher_actual_cost` vs `…_reference_cost` (savings tracking).
- Old stack (disabled): `automation.diskmaskin_automatisera` ("billigaste timmarna"),
  `…_startdetektering`, `…_klar`, `binary_sensor.diskmaskin_kor`,
  `input_boolean.diskmaskin_vantar`.

**Washing machine** — `switch.tvattstuga_plugg` (Shelly) + `sensor.tvattstuga_plugg_power`
(idle ≈3.4 W, heating bursts ≈1.3 kW) + `sensor.tvattstuga_plugg_energy` (≈642 kWh lifetime).
No clean run-state binary sensor; the cycle must be inferred from power.

**Today's limitation:** the HA schedulers shift runs to the cheapest *Nordpool grid* hours
only. They ignore PV surplus and the battery plan — so a load can run at 03:00 on "cheap"
grid instead of during free midday solar, and it can collide with Darkstar's battery
schedule (e.g. start a 1.8 kW load during a slot Darkstar planned to grid-charge, breaching
the import limit). Co-optimising in one MILP removes both problems.

---

## 3. Cycle characterisation — why this needs a learning component

The user asked to look historically at *how long* and *how much energy* each cycle takes.
The 10-day HA history shows none of the raw signals are directly parseable:

- `sensor.diskmaskin_power` ≈ 0 W (38 816 sub-watt noise points / 10 days) — the dishwasher
  is no longer metered on that plug.
- `binary_sensor.dishwasher_running` **flaps** on/off every 5–40 s (732 transitions / 10 days)
  — derived from a power threshold without debounce.
- `sensor.tvattstuga_plugg_power` only retains big jumps under `significant_changes_only`,
  so wash/spin phases are lost; visible heating bursts are ≈1.3 kW.

**Conclusion:** robust cycle detection is non-trivial and belongs *inside* Darkstar as a
dedicated **Cycle Learning** module, not as ad-hoc HA templates. It must:
1. Pick the best per-machine signal (native run-state for the dishwasher, plug power for
   the washer).
2. **Debounce** (min-on-time, gap-merge) to collapse flapping into one cycle.
3. **Integrate energy** (trapezoidal on the power series) and record `duration`,
   `energy_kwh`, and a coarse `cycle_profile_kw` (per-15-min) per completed run.
4. Maintain a rolling per-load estimate (median duration/energy + typical profile) used by
   the planner. Seed from `config` defaults until enough real runs are observed.

Ballpark seed values (to refine from learned data): dishwasher ≈ 1–2 h, ≈1.0–1.5 kWh, peak
≈1.3–2.0 kW; washing machine ≈ 1–2 h, ≈0.5–1.5 kWh, peak ≈1.3 kW.

---

## 4. Architecture

```
HA (device I/O + raw signals)                Darkstar
┌───────────────────────────┐   sensors   ┌──────────────────────────────────┐
│ switch.<load>  (Shelly)    │────────────▶│ Cycle Learning (debounce, energy)│
│ binary_sensor.<load>_run   │             │   → learned duration/energy/profile│
│ sensor.<load>_power/energy │             ├──────────────────────────────────┤
│ (existing detection code)  │  "queued"   │ Planner / Kepler MILP            │
│                            │────────────▶│   deferrable_load constraints     │
│ switch.<load>  ◀───────────│   command   │   (one contiguous block, deadline)│
└───────────────────────────┘             │ Executor → toggles switch.<load>  │
                                          └──────────────────────────────────┘
```

### 4.1 Config (same shape as `water_heaters` / `ev_chargers`)

```yaml
deferrable_loads:
  - id: dishwasher
    name: "Diskmaskin"
    enabled: true
    interruptible: false            # decision #1 — run as one block
    # Cycle size: learned over time, with config seed/fallback:
    duration_min: 110               # seed; overwritten by learned median
    energy_kwh: 1.2                 # seed; overwritten by learned median
    cycle_profile_kw: null          # optional learned per-15-min profile
    # Deadline: decision #2 — pick ONE mode per load
    deadline_mode: cheapest_within_hours   # or: hard_deadline
    window_hours: 14                # for cheapest_within_hours
    # hard_deadline: "07:00"        # for hard_deadline
    phase: A                        # optional, for phase-balancing (§4.5)
    control:
      switch: switch.diskmaskin               # Darkstar gates power here
      running: binary_sensor.dishwasher_running
      power_sensor: null                      # ≈0 on this plug; use run-state
      energy_sensor: sensor.diskmaskin_energy
      queued_signal: input_boolean.dishwasher_active   # auto-detect (decision #3)
  - id: washing_machine
    name: "Tvättmaskin"
    enabled: true
    interruptible: false
    duration_min: 100
    energy_kwh: 0.9
    deadline_mode: hard_deadline
    hard_deadline: "16:00"
    phase: B
    control:
      switch: switch.tvattstuga_plugg
      power_sensor: sensor.tvattstuga_plugg_power
      energy_sensor: sensor.tvattstuga_plugg_energy
      queued_signal: null            # inferred from power > threshold
```

### 4.2 Auto-detect trigger (decision #3)

The user already detects a queued/started cycle in HA. Darkstar consumes that as a
**"pending run" flag** per load (`queued_signal`, or power crossing a start threshold).
Flow:
1. User loads the machine and starts it (or it latches "ready when powered").
2. HA detection fires → `queued_signal` ON, and Darkstar **cuts** `switch.<load>` (hold).
3. Planner schedules the optimal contiguous block within the deadline.
4. Executor **restores** `switch.<load>` at the chosen start; `running` confirms the cycle;
   Cycle Learning records duration/energy on completion; `queued_signal` clears.

> Safety note: gating power to a non-interruptible appliance is only safe if the machine
> resumes/starts cleanly when powered (Shelly `power_on_behavior` = on, or a latching
> start). This is already how the current automations operate. For machines that do **not**
> resume cleanly, prefer a "request-by" UX instead of cut-and-restore. Flagged as an
> open question per machine.

### 4.3 MILP integration (in `planner/solver/kepler.py`, beside water/EV)

For each load with a pending run, add to the demand side of the energy balance and:

- **Non-interruptible block:** binaries `start[d][t]` and `run[d][t]`, with
  - exactly one start within `[earliest, deadline]`: `Σ_t start[d][t] == 1`,
  - contiguity: `run` is the N-slot window following the start
    (`run[d][t] = Σ_{k=0..N-1} start[d][t-k]`), N = `ceil(duration / slot)`,
  - completion before the deadline (hard) or a soft tardiness penalty
    (`cheapest_within_hours`).
- **Energy:** map `energy_kwh` (or `cycle_profile_kw`) onto the N running slots and add to
  `s.load_kwh` so it participates in the balance like any load.
- **No reward term needed:** the run is served at each slot's effective cost (import price,
  or ~free during PV surplus), so the MILP naturally picks the cheapest window. With the
  battery self-consume fix it will draw PV/battery before grid.

This is a small superset of the existing water-heater block/spacing constraints (water is
*interruptible* blocks; appliances are a *single contiguous* block).

### 4.4 Adapter + types

Add `DeferrableLoadInput` to `planner/solver/types.py` and `build_deferrable_load_inputs`
to `planner/solver/adapter.py`, mirroring `build_water_heater_inputs` /
`build_ev_charger_inputs`. The pipeline passes pending-run state (from HA `initial_state`)
through, exactly like `water_heater_states` / `ev_charger_states`.

### 4.5 Phase awareness (optional, ties to the grid-import bug)

These are large **single-phase** loads. The live-data investigation showed midday grid
imports caused by single-phase loads the 3-phase PV feed can't reach. Tagging each load
with `phase: A/B/C` lets the planner avoid stacking dishwasher + washer + water heater on
the same phase simultaneously (a soft penalty on concurrent same-phase deferrable load).
Optional, but it attacks the root cause directly. Requires per-phase load context.

---

## 5. VVB as a thermal state-of-charge (Phase 2)

Today the water heater is modelled as on/off with `min_kwh_per_day`. Upgrade it to a
**thermal battery**:
- `stored_kwh = m · c_p · (T_tank − T_cold) / 3600`, SoC% from the tank temperature sensor.
- Standing losses per slot (≈ U·A·(T_tank − T_ambient)).
- Comfort floor (never below X °C) + optional "hot water needed by HH:MM" deadline.
- Charge when cheap / PV-surplus, coast on losses otherwise; dump excess PV up to `temp_max`.

This reuses the same deferrable/optimisation pattern (the tank *is* a battery) and is the
natural follow-on once the appliance model lands. Bigger change — separate OpenSpec.

---

## 6. Migration path

1. Land `deferrable_loads` (disabled by default) + Cycle Learning; observe learned stats
   in debug for a week to validate against reality.
2. Per machine, **disable** the HA price-only scheduler
   (`automation.dishwasher_start_when_cheap`, old `diskmaskin_automatisera`) and let
   Darkstar own scheduling + the `switch`.
3. Keep `binary_sensor.*_running` / `queued_signal` for detection and completion.
4. Phase 2: VVB thermal SoC.

---

## 7. Suggested implementation order (OpenSpec changes)

1. **Cycle Learning** — detection/debounce/energy + learned per-load stats (read-only; no
   control). Validates the data before we trust it.
2. **Deferrable-load MILP + adapter/types** — scheduling only (dry-run, no HA writes).
3. **Executor control + auto-detect bridge** — gate/restore `switch.<load>`.
4. **Phase awareness** (optional).
5. **VVB thermal SoC** (Phase 2).

---

## 8. Open questions

1. Per machine: does it **resume/start cleanly when powered**? (Decides cut-and-restore vs
   request-by-UX.) Dishwasher likely yes (current automations do it); confirm the washer.
2. `cheapest_within_hours` vs `hard_deadline` default **per load** — what does each machine
   need? (e.g. dishwasher overnight = soft 14 h; washer = hard "done by 16:00".)
3. Phase mapping: which phase is each appliance / the water heater on? (Enables §4.5.)
4. Should learned cycle stats be **per program** (eco vs intensive) or a single rolling
   estimate per machine to start?
