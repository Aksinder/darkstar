# Blueprint: load priority / willingness-to-pay (WTP) layer

**Status:** Implemented. Flag-gated (`load_priority.enabled`, default off).

**Problem:** The planner minimises cost, but "cost" alone can't express *which* loads
matter. A water heater that must always have hot water, a spa that's nice-to-have, and an
EV that needs a target SoC by morning are not interchangeable. We want a single, unified way
to say "this load is worth running up to *X* SEK/kWh" and have the solver schedule
accordingly — defer or skip low-value loads under scarcity, keep high-value loads running,
and never heat the comfort loads at the day's most expensive hours.

## The price scale (read this first)

All WTP numbers are on Darkstar's **import-price scale**, not the raw spot:

```
import_price = (nordpool_spot + grid_transfer_fee + energy_tax) × (1 + vat)
```

with the components from `config["pricing"]` (e.g. transfer 0.36, tax 0.428, VAT 25 %). On a
normal Swedish summer day that puts the import price at **~2.0 (midday-solar floor) → ~3.9
(evening peak) SEK/kWh** — the fixed fees + VAT shift everything well above the raw spot
(~0.04–0.30). A WTP of `3.0` therefore means "worth running below the evening peak", not
"absurdly high". Verify the live scale from the add-on's `/api/schedule`
(`import_price_sek_kwh` per slot).

## Config

```yaml
load_priority:
  enabled: false
  rank_step_sek_per_kwh: 0.001       # intra-tier tiebreak granularity (lower rank wins)
  tiers:
    important: { base_wtp_sek_per_kwh: 3.0, urgency_wtp_sek_per_kwh: 5.0 }
    comfort:   { base_wtp_sek_per_kwh: 0.4, urgency_wtp_sek_per_kwh: 0.6 }
  loads:
    main_tank:                       # key = the load/heater/charger id
      tier: important
      rank: 0                        # lower rank = preferred within the tier
      # base_wtp_sek_per_kwh: 2.5    # optional per-load override of the tier value
      # wtp_percentile: 50           # DYNAMIC cap (see below) — preferred for the VVB
```

`build_load_priorities()` (adapter) resolves tier defaults + per-load overrides + the
rank tiebreak into a flat `LoadPriority` per load, so the solver consumes pure numbers.
A load referencing an unknown tier is skipped (logged), never raising — a bad YAML can
never harden into an infeasible solve.

## How the WTP enters the solver (Kepler)

WTP is a **reservation price**: a load is worth running in a slot when its WTP for that slot
meets or exceeds the slot's marginal energy price. Three load families consume it:

- **Deferrable loads** (dishwasher…): a credit `−wtp_t · energy` on the run, where `wtp_t`
  ramps from `base_wtp` at the earliest start to `base_wtp + urgency_wtp` at the deadline
  (linear urgency). Replaces the legacy tardiness penalty for priority loads.
- **Water heaters**: a *satiated* credit. A `served` variable is capped at the daily comfort
  need (`served ≤ min_kwh_per_day`) and at what was actually heated, with credit
  `−wtp · served`. The satiation cap means it never over-heats; the credit means it heats up
  to the need wherever the marginal price is below `wtp`. **No urgency ramp** for heaters —
  the cap/floor below do that job.
- **EV charging**: incentive-bucket values are sourced from the charger's `wtp_tier`, folding
  EV onto the same scale while keeping the SoC-bucket capacity structure.

When `load_priority.enabled` is false (or a load has no entry) behaviour is byte-identical
to before.

## Reliability: who is forced, who may skip

The legacy per-day reliability penalty (`water_heating.reliability_penalty_sek`, e.g. 1000)
forces a heater to meet `min_kwh_per_day`. The WTP layer changes who it applies to:

| Heater kind | Reliability floor | Behaviour |
|---|---|---|
| Non-priority | **ON** | Always meets the daily minimum (legacy). |
| Priority, **static** WTP | **suppressed** | May *skip* a day when every slot costs more than its WTP (correct for a spa / nice-to-have). |
| Priority, **dynamic** WTP | **ON** | Always meets the daily minimum, *in the cheap band the cap leaves* (correct for the VVB). |

> ⚠️ `max_hours_between_heating` is **not** enforced anywhere in the solver — it is read into
> config but has no effect. Do not rely on it as a "must heat every N h" backstop. The
> reliability floor above is the real backstop.

## Dynamic percentile cap (`wtp_percentile`)

A *static* SEK cap is brittle: on a uniformly-expensive day (winter, price spike) no slot
sits below it, so a priority heater with reliability suppressed would never heat — cold
water. The fix: set `wtp_percentile` (0–100) instead of a fixed `base_wtp`.

Each plan, the pipeline recomputes that load's `base_wtp` as the **Nth percentile of the
rolling 24 h import-price distribution** (`dynamic_wtp_from_prices`, linear-interpolated,
numpy "type 7"). So the cap *tracks the day*:

- It permits heating in the cheapest ~N % of **that day's** hours. The daily need is only a
  few slots — far inside that band — so there is **always** a cheap slot to use. It never
  starves, on any price level.
- It still refuses the day's most **expensive** hours (everything above the percentile).
- Combined with the reliability floor (kept on for dynamic loads), the result is
  **belt-and-suspenders**: the *cap* steers heating to the cheapest hours, the *floor*
  guarantees it actually heats. Perpetual deferral / "never heats" is impossible.

Lower percentile = stricter peak-avoidance; higher = more hot-water headroom. ~45–55 is a
good range for the VVB. The static `base_wtp_sek_per_kwh` remains as a fallback if the
percentile can't be computed (e.g. no price slots).

## Files

- `planner/solver/types.py` — `LoadPriority` (`base_wtp`, `urgency_wtp`, `rank_epsilon`,
  `dynamic_percentile`).
- `planner/solver/adapter.py` — `build_load_priorities()`, `dynamic_wtp_from_prices()`.
- `planner/solver/kepler.py` — deferrable/water/EV WTP terms; reliability-floor exclusion
  (suppress only for *static* priority loads).
- `planner/pipeline.py` — dynamic-cap recompute from the next-24 h slots before solve.
- `frontend/src/pages/settings/LoadPriorityTab.tsx` — drag-to-rank tier editor.
- Tests: `tests/planner/test_load_priority_solver.py`.
