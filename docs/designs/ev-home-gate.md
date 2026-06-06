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

## Backlog: come-home prediction (NOT built)
"The car is on the island / heading home, so reserve charging capacity ahead of time."
Deliberately **out of the gate** because being *near* home ≠ charging at our outlet (a
car 5 km away at a friend's charges there). This is a separate, predictive feature with
real risk: it would reserve battery/grid for a car that *might* not come home, re-introducing
a softer version of the phantom-load problem. If pursued, model it as a low-confidence
"expected arrival" signal that nudges (not commits) the plan, and gate the actual charge
on real presence (the gate above) when the car arrives. Keep separate from the hard gate.
