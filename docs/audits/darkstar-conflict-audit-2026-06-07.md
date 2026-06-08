# Darkstar Conflict-Audit Report

**Date:** 2026-06-07 · **System:** Sungrow SH10RT + [FORK] Darkstar Energy Manager (`df798f97_darkstar-fork`) · **Scope:** every non-Darkstar writer of Darkstar's control entities (inverter registers, VVB/Spa/EV loads, pricing inputs)

---

## 1. TL;DR Verdict

**1 live, actively-firing conflict. 1 live latent conflict (no-op today). The rest is dormant scenes + dead registry ghosts.**

| Class | Count | Examples |
|---|---|---|
| **ACTIVE conflict (firing + actually overriding Darkstar today)** | **1** | `automation.vvb_solar_excess_start` — writes `input_number.vvb_darkstar_target_temp=70` on solar export; verified 30-second tug-of-war with Darkstar today 15:26→15:27 |
| **LIVE but currently no-op / dormant-branch conflict** | **3** | `automation.vvb_solar_excess_stop` (last acted 2026-05-03), `automation.spa_solar_excess_start`/`_stop` (write a helper that is frozen at 40.0; spa offline, bridge dormant) |
| **Latent — loaded but no caller / never fires** | **9** | 7 mkaiser control scenes (battery_forced_charge/discharge/bypass, self_consumption ×2, max_export, zero_export) + `automation.sungrow_max_export_scene_sets_rated_limit` + `scene` companions |
| **NOT a conflict (observer / reflection / disabled)** | several | `number.white_betty_charge_current` (tesla_fleet reflection), `binary_sensor.white_betty_scheduled_charging_pending` (read-only sensor), Easee smart_charging=off, ev_session_* cost-trackers, disabled energy_spa_* automations |
| **INERT GHOSTS (state=unavailable, restored, package removed)** | **~40+** | the 34 `automation.sungrow_inverter_*`, `automation.energy_ev_charging_control`, `car_charging_disable_*`, villavagn_* |
| **SPECIAL: external writer on export limit** | **1** | `number.export_power_limit` rewritten 40→8400 W every ~15 min by the **Sungrow Modbus integration itself** (not Darkstar, not an HA automation) |

**Bottom line:** Only **`vvb_solar_excess_start`** is actively fighting Darkstar right now. The export-limit "existing controller" is real but is the **Sungrow integration's own write-back**, and it only matters once Darkstar C3 is enabled (still correctly disabled). Everything else is either dormant (manual-activation footguns) or dead clutter.

---

## 2. Controller Map — everything touching a Darkstar entity

Legend: **Loaded** = entity/automation is live (not unavailable/restored). **Fires** = actually executes its action recently. Severity reflects *current* state.

### 2a. INVERTER SIDE (`select.ems_mode`, `select.battery_forced_charge_discharge`, `number.battery_*_power`, `number.export_power_limit`, `switch.export_power_limit`)

| Controller | What it writes | Darkstar's intent on that entity | Loaded? | Fires? | Severity |
|---|---|---|---|---|---|
| **Sungrow Modbus integration (export write-back)** | `number.export_power_limit` 40→8400 W, ~15-min quarter-hour cadence; last 2026-06-07 16:45 | Darkstar writes this only in **export** mode or via C3 curtailment (both off now); must be single-owner | yes | **yes (continuous)** | **MEDIUM** (latent — would thrash only if C3 enabled) |
| `automation.sungrow_max_export_scene_sets_rated_limit` | sets export limit to rated max on scene activation | Darkstar owns `export_power_limit` | yes (state=on, bodyless ghost) | no — last 2026-04-19, trigger scene never fires | LOW |
| `scene.max_export_power` | `switch.export_power_limit` (→on) | Darkstar keeps switch on anyway | yes | never activated (no caller) | LOW |
| `scene.zero_export_power` | `switch.export_power_limit` + `number.export_power_limit` (→0) | exact C3 curtailment pair | yes | never activated (no caller) | LOW |
| `scene.battery_forced_charge` | `select.ems_mode` + `select.battery_forced_charge_discharge` | Darkstar drives both every cycle | yes | never (0 scene.turn_on callers) | LOW |
| `scene.battery_forced_discharge` | same two selectors | same | yes | never (no caller) | LOW |
| `scene.battery_bypass_mode` | same two selectors | same | yes | never (no caller) | LOW |
| `scene.self_consumption_mode_max_battery_discharge` | same two selectors | same | yes | never (no caller) | LOW |
| `scene.self_consumption_mode_no_battery_discharge` | + `number.battery_max_discharge_power` | Darkstar sets discharge per mode | yes | never (no caller) | LOW |
| **34× `automation.sungrow_inverter_*`** | all 7 inverter registers (mkaiser sync) | Darkstar is sole writer | **no** (unavailable/restored) | never | **NONE** (ghost) |
| `scene.battery_forced_charge` companion automations | — | — | mixed | never | NONE |

**Core 5 registers** (`ems_mode`, `battery_forced_charge_discharge`, `forced_charge_discharge_power`, `battery_max_charge_power`, `battery_max_discharge_power`): **no non-Darkstar writer fires.** Live values match Darkstar's self-consumption writes (last_changed today 18:00). Clean.

### 2b. LOAD SIDE — VVB / Spa / EV

| Controller | What it writes | Darkstar's intent on that entity | Loaded? | Fires? | Severity |
|---|---|---|---|---|---|
| **`automation.vvb_solar_excess_start`** | `input_number.vvb_darkstar_target_temp = 70` on grid export (→ bridge turns `switch.vvb` ON) | Darkstar owns this helper: 60=heat / 40=idle per plan | yes (state=on) | **YES — fired today 15:26:31, verified trace** | **HIGH** |
| `automation.vvb_solar_excess_stop` | `input_number.vvb_darkstar_target_temp`→low (inferred from sibling); → `switch.vvb` OFF | same helper | yes (state=on) | no — last acted 2026-05-03; all runs today failed_conditions | LOW |
| `automation.spa_solar_excess_start` | `input_number.spa_darkstar_target_temp = 40` on export (verified trace 16:00:37) | Darkstar owns helper (spa = non-priority load) | yes | fires, but **no-op** (helper static at 40.0 for 9d; spa offline; bridge dormant) | LOW |
| `automation.spa_solar_excess_stop` | `input_number.spa_darkstar_target_temp`→off (mirror of start) | same helper | yes | no — last acted 2026-05-08; all runs today failed_conditions | LOW |
| `automation.energy_spa_solar_heating` | `switch.layzspa_..._heat_regulation` direct on/off | Darkstar controls spa heat via climate/bridge | yes but **state=off (disabled)** | no | LOW (re-enable risk) |
| `automation.energy_phase_2_priority_vvb_spa` | `climate.layzspa` hvac_mode (fan_only/heat) | same climate entity | yes but **state=off (disabled)** | no | LOW (re-enable risk) |
| `number.white_betty_charge_current` | (Tesla amps) | Darkstar commands only the on/off switch, not amps | yes | reflection only | **NONE** — tesla_fleet `..._charge_current_request`, null-user cloud push, no HA writer |
| `binary_sensor.white_betty_scheduled_charging_pending` | nothing (read-only sensor) | — | yes | observes only | **NONE** — sensor cannot write; timeline shows HA driving the switch, not the car |
| `switch.easee_niska_it_smart_charging` | Easee cloud price-charging (if on) | Darkstar gates via Tesla switch | yes but **=off** (7d) | no | NONE (keep off) |
| `automation.ev_session_tesla_*` / `ev_session_easee_*` | `input_number.ev_session_*` cost accumulators only | — | yes | observers (cost-tracking) | **NONE** |
| `input_select.darkstar_ev_tesla_override` (=auto) | lifts soft SoC reserve floor only | Darkstar reads it | yes | — | NONE |
| `automation.energy_spa_pump_auto_maintain` | spa circulation **pump** only | Darkstar controls heat, not pump | yes | never fired | NONE (complementary) |
| VVB observers: `house_vvb_auto_calibrate_full`, `vvb_log_*`, `vvb_count_runtime_*` | calibration / runtime input_numbers | — | yes | observe only | NONE |

### 2c. INERT GHOSTS (state=unavailable, restored=true, last_changed 2026-05-30 — package removed, cannot fire)

| Controller | Would have written | Severity |
|---|---|---|
| **34× `automation.sungrow_inverter_*`** (update_ems_mode, ..._forced_charge_discharge_*, ..._max_charge/discharge_power*, export_power_limit*, soc limits, load_adjustment, mpp_scan, backup_mode) | all 7 inverter registers | NONE |
| `automation.energy_ev_charging_control` (id energy_ev_charging) | `switch.white_betty_charge` / charge_current / Easee enable (legacy EV controller) | NONE (re-add risk) |
| `automation.car_charging_disable_home_battery_discharge_during_ev_charging_2` | battery discharge selectors | NONE |
| `automation.villavagn_control_villavagn_spabad_heater_on_power_phase_2` | spa heat + villavagn VVB (phase guard) | NONE |
| Swedish-named VVB ghosts (`vvb_kor_under_billigaste_5h`, `vvb_starta/stang_vid_soloverskott`, runtime ones) | VVB switch | NONE |
| generic `ev_session_start/end/accumulate_cost` (superseded by per-vehicle) | cost accumulators | NONE |

---

## 3. `number.export_power_limit` Ownership Conclusion

**Who writes it right now:** the **Sungrow Modbus integration's own template `number` entity** (`platform=template`, `unique_id=uid_export_power_limit`), re-pushing the register value 40→8400 W on a ~15-min, quarter-hour-aligned cadence (last write 2026-06-07 16:45 → 1400 W). Confirmed: `user_id=null`/`parent_id=null` (backend write, not a person), **0 logbook entries** despite 8+ value changes, and **no HA automation/script/scene** is the writer (the 5 `sungrow_inverter_export_power_limit*` config matches are all unavailable/restored ghosts).

**Who is NOT writing it:** Darkstar. With `ems_mode='Self-consumption'` (not export mode) and `export_curtailment.enabled=false` (C3 off, `actions.py:645` gates the write), Darkstar's export-limit write path is inactive today.

**Conclusion:**
- The MEMORY note (`export-limit-existing-controller.md`) is **correct that another owner exists**, but the owner is the **Sungrow integration register write-back**, not a rogue HA automation. The 1400 W value is **not static** — it is the latest of dozens of dynamic values.
- This is a **real but currently-dormant single-owner conflict**: it would thrash with Darkstar **only if C3 export-curtailment is enabled**. Today there is no double-write.
- **Decision: keep `executor.export_curtailment.enabled=false` (C3 OFF).** Before ever enabling C3, pick one owner of `number.export_power_limit` — either disable/neuter the Sungrow export-limit write path and let Darkstar own it (gated on spot<0), or leave Sungrow as sole owner and never let Darkstar write it.

---

## 4. What We're Trying To Do With Darkstar

Darkstar is the **single price/forecast-driven optimizer** for the house. Every ~60 s it picks a battery mode and pushes it to the inverter, and per planning slot it schedules the deferrable loads:

- **Inverter / battery:** choose **charge** (grid/PV into battery when cheap), **export** (force-discharge to grid when the export price beats holding — spot SE3 + ~10 öre premium + nätnytta), **self-consumption** (PV covers house, normal), or **idle** (battery parked at/under SoC target, or held still while the EV charges). It writes `select.ems_mode`, `select.battery_forced_charge_discharge`, the forced/charge/discharge power numbers, and (only in export mode or C3 curtailment) the export-limit number+switch. At **negative export price**, C3 is meant to clamp export to 0 W — but C3 is currently disabled pending the export-limit ownership fix.
- **VVB (hot water):** Darkstar sets `input_number.vvb_darkstar_target_temp` (40 idle / 60 heat / boost) and a bridge automation turns the physical `switch.vvb` on when target>50. It already soaks solar excess as part of its own plan.
- **Spa (Lay-Z-Spa, non-priority load):** Darkstar sets `input_number.spa_darkstar_target_temp`; a bridge drives `climate.layzspa`. Low priority, heated only when cheap/surplus.
- **EV (Tesla via Easee):** Darkstar turns `switch.white_betty_charge` on/off per slot (~10-min windows, 30-min safety stop), gated on the car being home; it does **not** set Tesla amps and relies on Easee built-in scheduling staying off.

A conflict is therefore **anything other than Darkstar that writes one of these helpers/entities or commands the underlying device** in a way that fights the plan. The danger cases are the solar-excess automations that write Darkstar's *own* target helpers, and any controller that grabs the export limit.

---

## 5. Prioritized Remediation

### P0 — ACTIVE conflict, fix now
1. **`automation.vvb_solar_excess_start` — DISABLE or rescope.**
   It hijacks Darkstar's exact control input (`input_number.vvb_darkstar_target_temp`→70) on every export spike, forcing the VVB ON via the bridge regardless of Darkstar's price plan. Verified live tug-of-war today (Darkstar overrode it back to 40 within 30 s). Darkstar already handles solar-excess VVB soaking. **Action:** disable it, **or** point it at a separate boost flag Darkstar reads — never let it write `input_number.vvb_darkstar_target_temp` directly.

### P1 — Live but dormant; disable as a pair to prevent future fights
2. **`automation.vvb_solar_excess_stop`** — disable together with `vvb_solar_excess_start` (it's the off-side mirror writing the same helper; last acted 2026-05-03 but will fire again when export drops).
3. **`automation.spa_solar_excess_start` + `automation.spa_solar_excess_stop`** — disable as a pair (both write `input_number.spa_darkstar_target_temp`). Currently no-op (spa offline, helper frozen at 40, bridge dormant) but will become an active second-writer the moment the spa comes back online. Confirm with the user first if they *want* solar to force spa heat regardless of price; otherwise let Darkstar own it.

### P2 — Disabled automations: keep disabled, guard against re-enable
4. **`automation.energy_spa_solar_heating`** and **`automation.energy_phase_2_priority_vvb_spa`** — leave **off**. Both directly command the spa heat switch / `climate.layzspa` hvac_mode and would fight Darkstar if re-enabled. Flag so they are not silently turned back on.
5. **`switch.easee_niska_it_smart_charging`** — keep **off** (required for Darkstar to own EV charging via the Tesla switch).

### P3 — Latent footguns: delete the orphaned mkaiser scenes
6. **Delete (or document as never-wire) the 7 mkaiser control scenes:** `scene.battery_forced_charge`, `scene.battery_forced_discharge`, `scene.battery_bypass_mode`, `scene.self_consumption_mode_max_battery_discharge`, `scene.self_consumption_mode_no_battery_discharge`, `scene.max_export_power`, `scene.zero_export_power`. None has a caller and none has fired in 7d, but a single manual tap would slam Darkstar's live selectors / export pair. The two export scenes additionally collide with the unresolved export-limit ownership.
7. **`automation.sungrow_max_export_scene_sets_rated_limit`** — delete (bodyless ghost, trigger scene never fires).

### P4 — Inert clutter: registry cleanup only (no functional effect)
8. **Purge the ~40 unavailable/restored ghosts** to stop them resurfacing in audits: the **34 `automation.sungrow_inverter_*`**, `automation.energy_ev_charging_control`, `car_charging_disable_home_battery_discharge_during_ev_charging_2`, `villavagn_*`, and the Swedish-named VVB ghosts. They cannot fire (package removed); cleanup is hygiene, not safety.

### KEEP — do not touch (legitimate / observers)
- `df798f97_darkstar-fork` add-on — **is** Darkstar, the legitimate owner of all 7 inverter entities + load helpers.
- VVB/spa/EV bridges (`vvb_darkstar_bridge_temp_to_switch`, `spa_darkstar_bridge_temp_to_switch`) — Darkstar's own translation layer.
- All `ev_session_*` cost-trackers, `vvb_log_*`/`house_vvb_auto_calibrate_full` calibration, `energy_spa_pump_auto_maintain` (pump only), `number.white_betty_charge_current` reflection, `binary_sensor.white_betty_scheduled_charging_pending`, `input_select.darkstar_ev_tesla_override` — pure observers/reflections, not conflicts.

### Decision gate
- **Keep `executor.export_curtailment.enabled=false` (Darkstar C3 OFF)** until `number.export_power_limit` ownership is resolved (Sungrow integration write-back vs. Darkstar). Then update MEMORY: the "managed by another automation" note should be corrected to **"managed by the Sungrow Modbus integration's own register write-back"**, and the 1400 W "static" assumption removed.