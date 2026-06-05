# Blueprint: Predbat-inspired improvements (realism simulation + stored-energy value)

**Status:** Both improvements **implemented** (with tests). Improvement A (realism
simulation) is always-on observability; Improvement B (continuous stored-energy value)
is opt-in via `battery_value.enabled` (default off). See also
`docs/designs/phase-aware-load-modeling.md`, which subsumes A's phase modeling with
learned per-phase data.

**Objective:** Adopt the two strongest ideas from Predbat's approach that address
weaknesses we found in Darkstar's pure-MILP planner:

1. A **post-MILP realism simulation** that models the *executor's actual inverter
   behaviour* (and phase reality), catching plans that look optimal to the LP but
   import grid in practice.
2. A **continuous stored-energy value** term so the planner values the energy left
   in the battery, instead of relying only on a hard terminal SoC floor (which is
   what caused the midday over-conservatism we already had to relax).

Both are validation/objective refinements — the Kepler MILP stays the optimiser.

---

## Background: where the two systems differ

- **Darkstar** solves an idealised MILP (`planner/solver/kepler.py`). It assumes the
  inverter can do exactly what the schedule says (discharge to cover any deficit,
  treat the house as a single net node).
- **Predbat** runs a predictive *simulation* of the real battery/inverter/home over
  the horizon (coarse→fine search) and values the energy stored in the battery as
  part of its cost metric.

The two live-data issues we found map directly onto Predbat's strengths:
- The **idle-freeze + phase-imbalance** grid imports (full battery, 11 kW sun, ~7 kW
  single-phase load → +3.9 kW net import) are invisible to the LP because it models a
  single net node and assumes discharge is always available. A behaviour simulation
  would have flagged it.
- The **over-conservative midday holding** came from a hard terminal SoC floor
  (`target_under_violation`, penalty `target_soc_penalty_sek`). Predbat instead
  *values* stored energy continuously, which is gentler and less prone to "buy grid
  just in case".

---

## Improvement A — Post-MILP realism simulation

### Current state
`planner/simulation.py::simulate_schedule` only integrates `charge_kw`/`discharge_kw`
into a projected SoC with a round-trip efficiency. It does **not** model:
- the executor freezing discharge during idle/Hold slots (`max_discharge ≈ 10 W`),
- single-phase loads a 3-phase inverter cannot fully cover (gross per-phase import),
- charge/discharge power-rate limits or efficiency curves.

So the schedule's *predicted* cost can diverge from reality with no warning.

### Proposed
Add a realistic executor-behaviour simulator (extend `planner/simulation.py`):

```
simulate_realistic(schedule_df, config, *, phase_loads=None) -> SimResult
```

Per slot, replay the schedule against an inverter-behaviour model:
- **Discharge availability:** in slots the executor would run *idle* (SoC ≤ target,
  no PV surplus), discharge is unavailable — so any sub-slot/extra load is served
  from grid even with a full battery. (Mirror `executor/controller.py` mode logic.)
- **Phase coverage (optional):** if per-phase load shares are configured, model the
  3-phase inverter spreading support evenly; compute gross per-phase import that the
  single net-node LP hid. Tag each load with `phase` (already in `deferrable_loads`).
- **Rate / efficiency:** clamp charge/discharge to the configured kW limits and apply
  the efficiency curve.

Output a `SimResult` with: `simulated_cost_sek`, `predicted_cost_sek` (from Kepler),
`realism_gap_sek`, and a per-slot list of "unexpected grid import" flags.

### Wiring
- Call `simulate_realistic` in `planner/pipeline.py` right after `solver.solve(...)`.
- Store `realism_gap_sek` + flagged slots in the schedule debug (`s_index_debug` /
  the saved `schedule.json`) and publish a `sensor.darkstar_plan_realism_gap` (via the
  existing HA publisher) so divergence is visible on the dashboard.
- **Validation only — never blocks the schedule.** A large gap is a signal to fix the
  executor mode logic (e.g. the self-consume-on-PV-surplus fix) or rebalance phases,
  not to reject the plan.

### Value
Turns the class of bug we found from "discovered by reading HA logs weeks later" into
"surfaced on the dashboard the same hour".

---

## Improvement B — Continuous stored-energy value

### Current state
`kepler.py` has **no terminal-value reward**. The K20 comment claims
"terminal_value and wear_cost are sufficient", but the only terminal mechanism is the
**hard floor**: `soc[T] >= target_soc_kwh - target_under_violation` penalised at
`target_soc_penalty_sek` (200 SEK/kWh). A hard floor over-reserves: it will buy grid
to reach the floor even when upcoming PV/cheap hours make that wasteful (the symptom
we relaxed with the PV-aware bridging reserve).

### Proposed
Add a genuine **terminal value credit** to the objective — reward energy left in the
battery at the end of the horizon at its expected future worth:

```
objective += - battery_value_sek_per_kwh * soc[T]
```

- `battery_value_sek_per_kwh` = expected marginal value of stored energy ≈ the energy
  it will avoid importing soon. A safe estimate: the **min of the forward average
  import price over the look-ahead window** (the bridging-reserve window we already
  compute), minus wear, clamped to `[0, max_import_price]`. Risk-appetite can scale it.
- This lets the planner hold cheaply-charged energy when it is genuinely worth more
  than exporting now — **without** a hard floor forcing grid top-ups.

### Why this avoids the K20 bug
K20 was removed because `stored_energy_cost` added a cost on *discharge* with no
offsetting credit on *charge*, making charging look unprofitable. The correct form is
a **terminal credit on `soc[T]` only** (value what remains at the end), which is
symmetric and cannot penalise mid-horizon cycling. Wear cost still discourages
pointless churn.

### Relationship to the safety floor
The terminal value **complements and softens** the floor:
- Keep the floor as a *hard safety* at a low reserve (don't strand the home overnight).
- Let the terminal value handle the *economic* "how much extra to keep" decision,
  continuously, instead of the floor's all-or-nothing penalty.
- Net effect: less "buy grid just in case", same overnight safety.

### Wiring
- `KeplerConfig`: add `battery_value_sek_per_kwh: float = 0.0` (0 = disabled).
- `kepler.py`: subtract `battery_value_sek_per_kwh * soc[T]` in the objective.
- `planner/pipeline.py`: derive the value from the forward price window (next to
  `calculate_safety_floor`) and lower `target_soc_penalty_sek` accordingly.
- Add tests: a scenario where cheap energy is held for an expensive evening *without*
  a hard floor; a scenario proving no mid-horizon charge/discharge distortion.

---

## What we deliberately keep
- **The MILP stays the optimiser.** Predbat's heuristic search is not better than a
  correct LP; we only borrow its *realism check* and *energy valuation*.
- **Darkstar's ML forecasting (Aurora) and deferrable-load MILP** remain — these are
  areas where Darkstar is ahead of Predbat.

## Suggested order
1. **Improvement A** (realism simulation) — pure, read-only, no risk; immediately
   surfaces executor/reality gaps. Highest value-to-risk.
2. **Improvement B** (stored-energy value) — objective change; needs tuning + tests,
   so land it after A is validating plans so we can measure its effect.

## Out of scope (noted from the comparison, not adopted now)
- Tariff comparison ("which contract is cheapest") — useful, separate feature.
- Charge/discharge rate-curve & efficiency calibration from observed data — a learning
  enhancement, separate spec.
