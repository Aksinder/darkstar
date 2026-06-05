# Deploy & verify: phase-aware energy + stored-energy value

Everything in `feat/phase-aware-energy` is **opt-in and read-only until you enable it**
— deploying the image changes nothing on its own. Turn features on one at a time and
verify each before the next. Recommended order: **Observe → verify → Recommend →
(optional) Phase 3a scheduling → (optional) Improvement B**.

Your config lives in `/config/darkstar/config.yaml` and is shared across STABLE / DEV /
FORK, so enabling a block there applies as soon as the FORK image runs it.

---

## Step 0 — Deploy the FORK image (keeps all settings)

The add-on reads `/config/darkstar/config.yaml`; the image only carries code. So a new
build never touches your settings.

1. In the fork, the `feat/phase-aware-energy` branch is pushed. Merge it into the
   branch your `[FORK]` add-on builds from (or point the add-on at it), then let the
   GitHub Action build the image to your **private** GHCR.
2. In Home Assistant: Settings → Add-ons → **[FORK] Darkstar** → Update/Rebuild to the
   new version tag.
3. Watch the add-on log for:
   - `✅ Phase observer loop scheduled`
   - `✅ Cycle publisher loop scheduled`

Nothing else changes — every new feature below is still off.

---

## Step 1 — Observe (Phase 1): learn phases + per-phase load

Add to `/config/darkstar/config.yaml`:

```yaml
phase_observer:
  enabled: true
  phase_a_sensor: sensor.meter_phase_a_active_power
  phase_b_sensor: sensor.meter_phase_b_active_power
  phase_c_sensor: sensor.meter_phase_c_active_power
  battery_power_sensor: sensor.battery_power      # discharge positive
  battery_power_scale: 1.0
  # pv_power_sensor: sensor.<your_pv_power>        # optional; only improves the load fractions
  import_price_sek_kwh: 2.0
  export_price_sek_kwh: 0.5
  devices:
    - id: easee
      name: "Easee laddare"
      power_sensor: sensor.easee_niska_it_power    # proven on live data -> phase A
    - id: tvatt
      name: "Tvättmaskin"
      power_sensor: sensor.tvattstuga_plugg_power  # proven on live data -> phase B
    # add more single-phase consumers here (VVB villavagn, dishwasher, ...)
```

Restart the add-on. Within a few minutes these sensors appear (POST to `/api/states`):

| Sensor | Meaning |
|--------|---------|
| `sensor.darkstar_phase_a/b/c_load` | per-phase house load (W) + share % |
| `sensor.darkstar_phase_imbalance` | hidden grid cost the MILP misses (W) |
| `sensor.darkstar_<id>_phase` | each device's learned phase (`A`/`B`/`C`/`3-fas`/`okänd`) |
| `sensor.darkstar_phase_recommendation` | top rebalancing suggestion (see Step 3) |

Open the **Fasbalans** dashboard (sidebar, `mdi:sine-wave`) — the cards populate as the
sensors arrive.

> Confidence grows with history. The first hour a device may read `okänd`; after a day
> or two of normal use it settles. Low-confidence mappings are shown but not acted on.

---

## Step 2 — Verify Observe

Let it run **2–3 days**, then check:

- [ ] `sensor.darkstar_easee_phase` = `A`, `sensor.darkstar_tvatt_phase` = `B` (matches
      the live-data proof). A 3-phase load (e.g. the house VVB element) should read
      `3-fas`.
- [ ] `sensor.darkstar_phase_a/b/c_load` sum looks like your real house load; the
      `share_percent` attribute is sane.
- [ ] `sensor.darkstar_phase_imbalance` is **non-zero during PV-surplus midday** (that's
      the money signal). If it spikes when the battery is full and the sun is up, that's
      exactly the bug this work targets.
- [ ] `/config/darkstar/phase_model.json` exists and contains `fractions`, `devices`,
      `recommendations`.

The planner now feeds these **measured** fractions into the realism simulation, so
`sensor.darkstar_plan_realism_gap` (if published) reflects real imbalance instead of a
guess.

---

## Step 3 — Read the recommendations (Phase 2)

`sensor.darkstar_phase_recommendation` state is the top suggestion, e.g.
`Flytta Tvätt B→C (~640 kr/år)`; its `recommendations` attribute holds the ranked list
(device, from→to, kWh/yr, SEK/yr, confidence). The Fasbalans dashboard renders them.

These are **one-time electrician moves** that permanently cut grid import. "balanserat"
means no single move helps (already split, or nothing stacked). Act on the top one or
two; re-check after a week of new history.

---

## Step 4 (optional) — Phase-aware scheduling (Phase 3a)

Once phases are learned and stable, let the deferrable scheduler avoid running two big
single-phase appliances on the same phase at once:

```yaml
deferrable_load_settings:
  phase_penalty_sek: 5.0   # >0 enables it; start small, raise if you want stricter spread
```

A deferrable load with no explicit `phase:` now inherits its **learned** phase
automatically, so this balances real appliances with no manual wiring. Explicit
`phase:` in a load's config still wins.

> Battery phase-compensation is **not** included: the Sungrow SH delivers balanced
> 3-phase and (almost certainly) cannot push asymmetric per-phase output. Verify against
> your model before expecting the battery to fix imbalance — the real levers are
> scheduling (this step) and physical rebalancing (Step 3).

---

## Step 5 (optional) — Continuous stored-energy value (Improvement B)

Softens the "buy grid just in case" behaviour by valuing energy left at the end of the
horizon:

```yaml
battery_value:
  enabled: true
  lookahead_hours: 12
  scale: 0.75     # 0.5 = more willing to trade energy now; 1.0 = hold more
```

It is bounded so it can **never** make the planner buy grid to inflate the terminal SoC
(value ≤ the cheapest forward import price). Watch the planner log line
`Battery value (Improvement B): X SEK/kWh` and compare a few days of grid import / SoC
curves before raising `scale`.

---

## Verification checklist (quick reference)

| # | Check | Where |
|---|-------|-------|
| 1 | Add-on log shows both loops scheduled | Add-on log |
| 2 | `sensor.darkstar_phase_*` exist | Developer Tools → States |
| 3 | Easee→A, Tvätt→B, house VVB→3-fas | Fasbalans dashboard |
| 4 | Imbalance non-zero at midday PV surplus | `sensor.darkstar_phase_imbalance` |
| 5 | `phase_model.json` written | `/config/darkstar/` |
| 6 | Recommendation reads sensibly | `sensor.darkstar_phase_recommendation` |
| 7 | (3a) deferrable loads spread across phases | planner schedule / logs |
| 8 | (B) battery-value log line, no spurious grid buys | planner log |

---

## Troubleshooting

- **Sensors never appear** → `phase_observer.enabled` is false, the phase sensor names
  are wrong, or HA url/token missing for the publisher. Check the add-on log for
  `Phase observer: not enabled/configured` or `HA url/token missing`.
- **Device stays `okänd`** → it didn't switch enough during the window (needs clear
  on/off steps >200 W), or it's genuinely multi-phase. Give it more days.
- **Device shows `3-fas` unexpectedly** → it really is a balanced 3-phase load (the
  house VVB element is), so it does not contribute to imbalance — correct.
- **Recommendation always "balanserat"** → loads are already split, or no two heavy
  single-phase loads stack on one phase. Nothing to fix.

## Hot-water sensors — one naming scheme

This install already publishes hot-water state via **HA-native template sensors**
(`sensor.house_vvb_*`, `sensor.villavagn_vvb_*`, with an `House VVB - Auto calibrate
full` automation). That is the **canonical scheme** and the Varmvatten dashboard uses
it directly — nothing to change.

Darkstar's own thermal hot-water publisher (`water_heaters: type: thermal`) now emits
the **same suffix scheme** (`_hot_water_level` / `_liters_remaining` /
`_estimated_temperature` / `_draw_today`) under a `sensor_prefix` that defaults to
`darkstar_`. So:

- **Keep `water_heaters` hot-water publishing OFF for tanks that already have template
  sensors** — a template sensor owns its `entity_id` and a REST push would conflict.
- Only use Darkstar's publisher for a tank **without** native sensors. Set
  `sensor_prefix: ""` if you want it to own the bare canonical ids
  (`<id>_hot_water_level` …); leave the default `darkstar_` to namespace it safely.

Net: one suffix convention everywhere; no `_soc`-vs-`_level` divergence.

## Safety

All of this is additive: sensor writes + one JSON file, plus opt-in objective/penalty
terms that default to off. Nothing controls hardware in the Observe/Recommend phases,
and `feat/phase-aware-energy` is on **your fork only** — never pushed to `ergetie`.
