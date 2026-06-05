# Blueprint: Phase-aware load modeling (Observe -> Recommend -> Control)

**Status:** Phase 1 (Observe) + Phase 2 (Recommend) + Phase 3a (phase-aware scheduling)
**implemented**. Feasibility proven on live data (see below). Phase 3b (battery
phase-compensation) intentionally NOT built — it needs asymmetric per-phase inverter
output, which the Sungrow SH almost certainly does not support (verify before any work).

Phase 3a wiring: a deferrable load with no explicit `phase` in config now inherits its
**learned** phase from `phase_model.json` (`load_device_phases`), so the existing
`deferrable_phase_penalty` (same-phase concurrency avoidance, already in the MILP)
balances real single-phase appliances without manual wiring. Gated on `phase_observer`.

**Implemented modules:**
- `backend/learning/phase_learning.py` — device->phase learning, per-phase load model.
- `backend/learning/phase_recommend.py` — rebalancing recommendations.
- `backend/learning/cycle_publisher.py` / `cycle_publisher_service.py` — sensors +
  `PhaseObserverService` (learn -> publish `sensor.darkstar_phase_*` /
  `sensor.darkstar_phase_recommendation` -> persist `phase_model.json`).
- `planner/pipeline.py` — realism sim reads the measured fractions.
- Enable via the `phase_observer` block in config (off by default).

**Objective:** Give Darkstar awareness of *which electrical phase* each load sits on,
so it can (1) surface the real per-phase grid cost the single-net-node MILP hides,
(2) recommend moving single-phase loads to balance the phases, and eventually
(3) schedule deferrable loads onto the phase with headroom.

This is the **#1 real-money lever** found in the live installation: the recurring
"buy grid even though the battery is full and the sun is shining" symptom is a
**phase-imbalance** problem, not a planning-horizon problem.

---

## Why this matters (live evidence)

Darkstar's planner (`planner/solver/kepler.py`) models the house as a single net
node. A 3-phase hybrid inverter (Sungrow SH) delivers PV + battery support
**balanced** across the three phases. So when a heavy **single-phase** load runs, the
inverter cannot put extra power on just that phase: the loaded phase imports from the
grid while the other two export — even with a full battery and surplus PV. The net
node nets to zero, so the LP never sees the cost.

Observed on the live meter (`sensor.meter_phase_a/b/c_active_power`):
- A snapshot read: A = -143 W, B = -427 W, C = +473 W -> **exporting on two phases
  while importing on the third, simultaneously**. That +473 W is pure waste: energy
  sold cheap on A/B and bought expensive on C in the same instant.

The existing realism simulation (`planner/simulation.py::simulate_realistic`,
Improvement A) already models this cost — but only if it is fed **real per-phase load
fractions**. Today those fractions are a static config guess. This blueprint makes
them **measured and learned**.

---

## Feasibility proof (done on live data, 2026-06)

We tested whether each device's phase can be **learned automatically** (no manual
wiring audit) by correlating each metered device's power *changes* against the
per-phase meter *changes*.

**Method.** For a device power series `d(t)` and phase meters `A/B/C(t)`, regress the
per-step change `Delta(phase)` on `Delta(device)` (least squares through origin) over
steps where `|Delta(device)| > 200 W`. The phase whose net moves ~1:1 with the device
is the device's phase.

**Key refinement — cancel the balanced inverter.** The inverter injects PV+battery
support equally on all phases, which masks the load phase (especially when the battery
is actively covering the load). Subtract the three-phase mean:
`rel_X = phase_X - mean(A,B,C)`. The balanced injection cancels, leaving only the load
imbalance. A single-phase load `L` on phase X then produces the signature
`Delta(rel_X) ~ +2/3 * L`, `Delta(rel_Y) = Delta(rel_Z) ~ -1/3 * L`.

**Results (5-min statistics over 3 days, then native-resolution confirmation):**

| Device | Sensor | Learned phase | rel A / B / C signature | Verdict |
|--------|--------|---------------|--------------------------|---------|
| Easee EV charger | `sensor.easee_niska_it_power` | **Phase A** | **+0.70** / -0.36 / -0.34 | textbook single-phase (matches +2/3, -1/3, -1/3) |
| Washing machine | `sensor.tvattstuga_plugg_power` | **Phase B** | -0.31 / **+0.77** / -0.46 | textbook single-phase |
| House hot-water (VVB) | `sensor.house_vvb_real_power` | **3-phase / balanced** | -0.04 / 0.00 / +0.04 | correctly flagged as phase-neutral |

The house VVB result is important: a native-resolution replay of a clean turn-on
(02:30 UTC, +3050 W) showed **all three phases rising together** (absolute
+367 / +482 / +598 W) with a near-zero relative signature — i.e. it is a **3-phase
heating element** and contributes **nothing** to imbalance. The remaining ~1.6 kW of
the step was absorbed by the battery ramping its (balanced) discharge — exactly the
masking the relative-to-mean trick is designed to remove.

**Conclusions:**
1. Device->phase mapping is learnable from existing sensors. No wiring audit needed.
2. The method correctly **classifies load type**: single-phase (imbalance source,
   schedulable/movable) vs 3-phase (phase-neutral, leave alone).
3. Fast thermostat loads need **native-resolution** edge detection (5-min means smear
   the on/off switching). We already have this engine in
   `backend/learning/cycle_learning.py` (hysteresis state machine + trapezoidal
   integration over raw HA history).

---

## Phased plan

### Phase 1 — Observe (low risk, high value; build first)
- **Per-phase load model.** From `meter_phase_a/b/c_active_power` (+ battery + PV),
  reconstruct per-phase *load* each slot and publish the running phase split + an
  imbalance metric (e.g. max phase minus mean) as HA sensors.
- **Device->phase learner.** Reuse `cycle_learning.py`'s native-resolution edge
  detection; for each configured device, run the relative-to-mean correlation and
  store `{device -> phase, confidence, load_type: single|three_phase}`. Re-learn
  periodically (the existing publisher service loop).
- **Feed the realism sim.** Replace the static `phase_load_fractions` config with the
  *measured* per-phase fractions, so `simulate_realistic` reports a real
  `realism_gap_sek` and flags the imbalanced slots. Publish
  `sensor.darkstar_plan_realism_gap`.
- **Dashboard.** A per-phase load/imbalance card so the waste is visible the same hour.

**Deliverable:** the installation can *see* its imbalance and its cost. No control,
no risk.

### Phase 2 — Recommend (one-time physical fix, lasting benefit)
- Analyse historical imbalance + each single-phase device's load + its learned phase.
- Produce a ranked, plain-language suggestion list, e.g. *"Move the washing machine
  (B, ~1.3 kW, runs ~daytime) to phase C; estimated -X kWh/yr grid import."*
- This is an electrician one-off that permanently lowers grid import. Highest ROI,
  zero runtime tech-debt.

### Phase 3 — Control (only after 1 & 2 are validated)
- **Phase-aware deferrable scheduling.** The deferrable MILP already carries a `phase`
  field and a `deferrable_phase_penalty` (built earlier but currently unused data).
  Wire the learned phases in so big deferrable loads are scheduled onto the phase with
  the most headroom in each slot.
- **Battery phase-compensation:** only if the inverter supports **asymmetric** per-phase
  output. The Sungrow SH almost certainly delivers balanced 3-phase, so this is
  **likely not possible** — must be verified against the actual model before any work.
  If unsupported, the levers remain scheduling (Phase 3a) + physical rebalancing
  (Phase 2), which already attack the root cause.

---

## Honest caveats / risks
- **Balanced inverter masking** — handled by the relative-to-mean method (proven), but
  it means the battery can *never* fix imbalance directly; the fix is scheduling +
  rewiring, not dispatch.
- **Asymmetric battery output** — unknown for the Sungrow SH; assume **no** until
  verified. This bounds Phase 3 to scheduling.
- **Mapping drift** — a device replugged to another phase invalidates its learned
  mapping; periodic re-learning + a confidence threshold guard against acting on stale
  data. Low-confidence mappings are observed, not acted on.
- **Scope** — this is a subsystem. Ship Phase 1 alone first; it is pure observability.

---

## Wiring (where it touches existing code)
- `backend/learning/cycle_learning.py` — reuse native-resolution edge detection; add a
  `learn_device_phase(power_samples, phase_a/b/c_samples) -> PhaseMapping`.
- `backend/learning/cycle_publisher_service.py` — periodic re-learn; publish per-phase
  load + imbalance + per-device phase sensors.
- `planner/simulation.py` — source `phase_fractions` from the learned model instead of
  static config (the function already accepts them).
- `planner/pipeline.py` — already calls the realism check; now it gets real fractions.
- `config.default.yaml` — list which device power sensors to learn from (we already
  enumerate them for deferrable loads).

## Relationship to the other improvements
- This **subsumes the value** of Improvement A's phase modeling by making it data-driven.
- It is **higher priority than Improvement B** (continuous stored-energy value) for this
  installation, because phase imbalance is the actual recurring cost. B is a refinement
  on top; this attacks the root cause. Suggested order: Phase 1 (Observe) -> Phase 2
  (Recommend) -> Improvement B -> Phase 3 (Control).

## Suggested order
1. **Phase 1 (Observe)** — per-phase model + device->phase learner + realism sim wiring
   + dashboard. Pure read-only. Build first.
2. **Phase 2 (Recommend)** — rebalancing suggestions.
3. **Improvement B** — stored-energy value (separate blueprint).
4. **Phase 3 (Control)** — phase-aware scheduling; battery compensation only if the
   inverter supports it.
