# Fork strategy ("own + harden") + phase-aware MILP sketch

**Verdict from the live audit:** the *engine* (Kepler MILP, Aurora ML, profile-driven executor)
is a strong base worth building on; the *strategy/glue/config* layer and the *CI safety net*
are where rigor was missing (a `risk_appetite=5` mapping silently halved the load forecast and
`load_safety_margin_percent` was dead — both shipped to production). So: **own the fork, harden
the weak layer, replace single subsystems only when proven limiting.** Do NOT rewrite from
scratch — the code bent to ~12 substantive changes this session with green lint/types/tests.

---

## Part A — "Own + harden" plan

### A0. Ownership / governance
- Remotes today: `origin = ergetie/darkstar` (upstream beta, **never push**), `fork = Aksinder/darkstar`
  (your deploy target). That's the right shape — keep it.
- **Treat the fork as your product** (soft-hard fork): your `main` is the source of truth; pull
  from `origin` selectively, never auto-merge. Cadence: monthly `git log origin/main ^main` review,
  cherry-pick valuable fixes, port them through your CI.
- **Fix the version identity:** the in-code app version reports `2.6.2-beta` (stale) while the image
  is `dev-YYYYMMDD.HHMM`. Bump `backend/core/version.py` to a fork-owned scheme so `/api/status`
  and the UI tell the truth.
- Confirm the upstream licence permits your private fork + private GHCR (we already keep GHCR private).

### A1. Safety net — DO FIRST (this is the root cause of shipped bugs)
Current `ci.yml` gates are thin:
- `ruff` runs on **`backend/` only** — `planner/`, `ml/`, `executor/`, `strategy/` are unlinted in CI.
- pytest runs **only `tests/api/test_api_routes.py`** — the ~400 planner/solver/executor/ev/ml tests
  **never run in CI**. (This is how the load-halving bug shipped.)
- **No `pyright`** anywhere; **no frontend `tsc --noEmit`** (only `vite build`, which doesn't type-check).
- The image build (`build-addon.yml`) does **not** depend on `ci.yml` passing.

Harden:
1. Expand ruff to `backend/ planner/ ml/ executor/ strategy/ tests/`.
2. Add a `pyright` job (we keep new code at 0 new errors already — make it a gate).
3. Run the **full** suite: `pytest` (not one file). Fix or `xfail` the known env-failures (lightgbm/
   libomp import, the SQLAlchemy test) so "green" is meaningful.
4. Frontend: add `tsc --noEmit` to CI (eslint alone misses type errors → a TS bug can ship via `vite build`).
5. Gate the image build on CI green (`needs:` or fold the build into the CI workflow).

### A2. Debt backlog (prioritized — from the audit)
**Done this session:** risk/load floor (load_safety_margin now enforced), export-override centralized,
EV home-gate on schedule + live power flow, negative-price export C1–C4 (off by default).

**P0 — correctness/footguns still open**
- **SlotPlan reconstruction drops fields** (`executor/engine.py:1247,1307`): the EV `at_home` bug class.
  Replace ad-hoc `SlotPlan(...)` rebuilds with one `slot.copy(**overrides)` helper so fields can't be
  silently dropped.

**P1 — observability + trust**
- Publish HA sensors: `sensor.darkstar_pv_forecast_today_kwh`, `…_load_…`, `…_forecast_model_status`
  (trained/seeding, last_trained, sample_count). Today there is *zero* external visibility of model
  health — you must shell into the container or hit `/api/*`.
- Fix `/api/system/health` planner status (`last_run=null` while 7026 plans exist).
- Price-model status counter: report the **joined** trainable-sample count, not `COUNT(*)` of forecasts.
- Normalize the `slot_start` join (`ml/price_train.py:144`) to canonical UTC ISO + log the join hit-rate
  (a silent zero-match seeding stall is currently invisible).

**P2 — cleanup + tuning**
- Remove dead code (`_set_max_export_power` was unreferenced before C3; the `unavailable` mkaiser
  export automations).
- Strategy defaults: document `risk_appetite` semantics (1→p90 … 5→p25 load), consider shipping
  `battery_value.enabled: true` + `excess_pv.sink: water_heater_boost` as better defaults.

### A3. Operational hygiene (your HA side, one-time, owner action)
- **Single inverter owner:** set Predbat + original Darkstar + `[DEV]` Darkstar to `boot: manual`
  (only `[FORK]` `boot: auto`). They're idle now but a restart would make them fight `[FORK]` over
  `select.ems_mode` / `number.battery_forced_charge_discharge_power`.
- **Export-limit ownership:** the inverter's feed-in mode (`sensor.export_power_limit_mode_raw=170`)
  owns `number.export_power_limit`. Decide owner before enabling C3.

### A4. Working loop (already proven this session)
branch → CI green → build to fork → live-verify via `/api/*` + HA → changelog auto-appends. Keep it.

---

## Part B — Phase-aware MILP sketch

### B0. The limitation
`Kepler` optimizes a **single net node**: one scalar `grid_import[t]` / `grid_export[t]`, one battery,
one aggregate load. It is **phase-blind**. But the house is 3-phase, loads sit on specific phases
(Easee→A, Tvätt→B, VVB→3-phase), and the inverter balances across phases. So **per-phase imbalance
import** — you buy on a loaded phase while the inverter exports cheaply on another — is invisible to
the optimizer. We measured ~**8.4 kWh/week** of hidden per-phase import at SoC ≥ 95%.

### B1. Data foundation — already built ✅
- `backend/learning/phase_learning.py`: learns each metered device's phase by correlating its power
  against the per-phase meter (cancelling the balanced inverter), reconstructs per-phase load
  fractions, publishes `sensor.darkstar_phase_a/b/c_load`, `…_phase_imbalance`, `…_<device>_phase`,
  writes `phase_model.json`.
- `backend/learning/phase_recommend.py`: rebalancing recommendations.
- Live per-phase meter: `sensor.meter_phase_a/b/c_active_power`.
- Today this feeds only the **realism simulation** (post-solve), not the solver's decisions.

### B2. Model extension (per phase p ∈ {A,B,C}, per slot t)
```
# Inputs (from phase_model.json fractions × totals, or measured)
load[t,p]              # per-phase house load
pv[t,p]               # balanced 3-phase string inverter: pv[t]/3 (or measured)

# Decision vars
grid_import[t,p] >= 0
grid_export[t,p] >= 0
charge[t,p] >= 0, discharge[t,p] >= 0      # battery contribution on phase p

# Per-phase energy balance
load[t,p] + charge[t,p] + grid_export[t,p] + curtailment[t,p]
    == pv[t,p] + discharge[t,p] + grid_import[t,p]

# Inverter coupling (THE crux — depends on hardware, see B4):
#   balanced inverter -> charge[t,A]==charge[t,B]==charge[t,C] (=charge[t]/3), same for discharge
#   per-phase-capable -> bounded imbalance |charge[t,p] - charge[t]/3| <= delta
```
**Objective:** sum import cost / export revenue **per phase**. The imbalance forces import on the
heavy phase while exporting on the light phase → the spread loss the net model hides becomes visible
and minimizable. Reuse `deferrable_load_settings.phase_penalty_sek` — but now *physically grounded*
(it discourages piling controllable load on the already-heavy phase, against the real per-phase balance).

### B3. Decision levers it unlocks
- Schedule deferrables (Tvätt/VVB/EV) onto the **least-loaded** phase, or defer when the target phase
  is heavy.
- Honestly surface **rebalancing recommendations** (move the VVB element / redistribute circuits) when
  a balanced inverter *cannot* fix structural imbalance — instead of silently eating the import.
- Value per-phase export correctly (export-on-light + import-on-heavy nets to a loss).

### B4. The honest caveat — verify the inverter first
If the Sungrow SH can **only deliver balanced** power (P/3 per phase — typical for hybrid inverters),
the MILP **cannot eliminate** structural imbalance. Its win is then: (a) *seeing* the cost, (b)
*scheduling controllable loads* on the right phase, (c) *recommending physical rebalancing*. If it can
do bounded per-phase injection, the solver can actively offset more. **Verify the SH's per-phase
capability** (spec / meter behaviour under forced charge) before sizing the expected gain.

### B5. Incremental migration (no Kepler rewrite)
- **Phase 0 (done):** phase_observer learns + publishes. ✅
- **Phase 1 (low risk):** post-solve per-phase *evaluation* — quantify the per-phase import cost of the
  net-optimal plan as a metric (the realism sim already exposes `gap_sek` / `extra_import_kwh`). Ship as
  a visible number; no solver change.
- **Phase 2 (opt-in):** add per-phase vars/constraints to Kepler behind `kepler.phase_aware: false`.
  Validate on historical data: does it reduce the measured ~8.4 kWh/wk? Keep the net model default.
- **Phase 3:** if validated, flip the default. Most of the gain will come from smarter deferrable phase
  scheduling + rebalancing recs (given a balanced inverter), not battery magic — set expectations
  accordingly.

Each phase is flag-gated and test-covered → a *subsystem* replacement, never a from-scratch rebuild.
