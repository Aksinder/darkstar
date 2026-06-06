# Blueprint: EV home-zone gate (presence, robust both ways)

**Status:** Implemented (zone + distance + grace). Come-home prediction = backlog (below).

**Objective:** The planner must schedule EV charging only when the car is (about to be)
charging at *our* house — never for a car charging elsewhere (its API reports
plug/charging regardless of location), and never wrongly skip a car that *is* home.

---

## Why a plain home/not_home is not enough

Two opposite failure modes:

1. **False include** — the car charges away (work / Supercharger) but its
   `plug_sensor` reads "connected", so the net-node planner schedules an 11 kW phantom
   load and reserves battery/grid for it. *(This was a real, live bug: an away Tesla got
   a future 11 kW slot. Fixed — the gate now forces `plugged_in=False` when away, and the
   pipeline threads `at_home` through via `ev_state_for_solver`.)*
2. **False exclude** — the car sits in the driveway/garage charging at home, but a tight
   HA zone (default `zone.home` radius is 100 m) plus GPS drift reads `not_home`, so the
   car is wrongly excluded and misses cheap/solar charging.

## The gate (`backend/core/ev_presence.py::ev_is_home`)

`at_home` is true if **any** of:
- **zone** — the tracker state ∈ `home_states` (the normal case),
- **radius** — the car is within `home_radius_km` of `system.location` (haversine on the
  tracker's `latitude`/`longitude` attrs) — absorbs GPS drift / covers the property even
  with a tight zone,
- **grace** — the tracker flipped away < `home_grace_minutes` ago (debounce momentary
  drift).

`radius_km`/`grace_minutes` default to 0 → plain zone check (no behaviour change unless
configured). Pure + unit-tested; enforced in `get_initial_state` → `ev_state_for_solver`
→ `build_ev_charger_inputs` (`plugged_in` forced False when away).

Config (per charger):
```yaml
home_entity: device_tracker.white_betty_location
home_states: ["home"]
home_radius_km: 0.3        # ~property radius; absorbs GPS drift
home_grace_minutes: 10     # keep "home" briefly after a flip
```

Also recommended (HA side): enlarge `zone.home` radius from 100 m to ~200 m so the
driveway/garage counts as home.

### Definitive signal (further hardening, optional)
The most reliable "it's on our circuit" signal is the **charging appearing on our home
grid/phase meter** when the cable connects. OR-ing that in would catch any case the
location misses. Not built yet; lower priority than radius+grace.

## `is_historical` — kept separate (by design)
The gate governs only **future intent** (`get_initial_state` → the plan). Historical
schedule slots are *actuals* (what happened when the car was home) and are never
re-decided. Verification splits `future_ev_kw_sum` vs `historical_ev_kw_sum`; dashboards
must read EV intent on `is_historical=false` slots only.

---

## Come-home prediction (Step 1 — implemented)
Pre-positions a **soft, capped, low-weight** home-battery buffer when the car is likely to
come home, so its charging can lean on cheap/solar energy. It never charges the car or
forces grid purchase — the hard gate above still governs actual charging.

**Three distance zones** (`backend/core/ev_arrival.py`):
- **near** (`distance <= near_radius_km`, default 5 km): treat as certainly arriving →
  `p = 1.0`, full reserve.
- **extended** (`distance <= extended_radius_km`, default 30 km — the island/region):
  `p` from the learned **arrival profile** (weekday x hour fraction of time home, built from
  `device_tracker` history over ~8 weeks, rebuilt every 6 h by `ev_arrival_service`).
- **beyond**: `p = 0`.

**Reservation:** `reserve_kwh = clamp(p x buffer_kwh, 0, max_reserve_kwh)`. The pipeline
lifts the **soft** target-SoC floor by the summed reserve (same risk-scaled penalty, so
economics still override). Default off (`come_home.enabled`).

**Phantom-load safeguards (as required):** soft only (never a hard charge), capped
(`max_reserve_kwh`), zone/probability-weighted, and a manual **override** via an HA
`input_select` (`auto` / `force_reserve` / `force_off`) read at plan time — `force_off`
also blocks the hard gate. Every come-home reservation and override is logged
(`EV <id> come-home: zone=… p=… reserve=… kWh`).

### Backlog (Step 2+)
- Condition `p` on current distance/region buckets ("from D km at time T, P(home within H h)").
- Per-slot soft reserve at the *predicted arrival time* (instead of a terminal-SoC bump).
- A small model (logistic/GBM) only if the rolling stats prove insufficient.
