# Blueprint: negative-price export curtailment (solver-integrated)

**Status:** Implemented (C1–C4). Off by default.

**Problem:** On sunny low-demand days the spot price collapses to ~0 or goes negative
mid-day. The system would still push surplus PV to the grid — at a price where *we pay to
export* — and would also dump near-zero-value PV instead of storing it for the expensive
evening (the user's root complaint: buying power in the evening with a full mid-day battery).

Measured on the live site: ~6 % of recent quarter-hours ≤ 0.05 SEK/kWh, genuine negatives
(down to −0.03 historically; −0.01 the next day), and the 16 kWh battery sits at 100 % for
5–10 h mid-day → **zero absorption headroom exactly when prices are lowest**.

## The four parts

### C1 — Correct export-price model (`backend/core/prices.py`)
`calculate_import_export_prices` now builds the export price from configurable components
instead of assuming raw spot:

```
export = (spot if export_includes_spot) + premium + grid_benefit - fee
```

Each component is a literal SEK/kWh in `config["pricing"]`, or an HA sensor via
`export_*_entity` (resolved live in `forecasts.get_all_input_data` →
`resolve_export_price_components`, threaded into `get_nordpool_data(pricing_overrides=...)`
with a component-aware cache key). So the effective export compensation follows the
contract (e.g. spot + 0.10 påslag + nätnytta) with no redeploy. Defaults are 0 → legacy
`export == spot`.

### C2 — Price-conditioned curtailment (`planner/solver/kepler.py`, config)
The MILP energy balance already lets `curtailment[t]` absorb surplus PV. The penalty
`curtailment_penalty_sek` was 0.1 — *higher* than shallow negative prices — so the optimizer
preferred to export at −0.05 (cost 0.05) over curtailing (cost 0.10). Lowered the default to
**0.001**: curtailing now beats negative-price export at any depth, while using/storing PV
(charge, water, load) still always wins (those carry real value ≫ 0.001).

### C3 — Executor real-time clamp (`executor/`)
The plan's intent is physically enforced: when the current slot's effective export price is
below `executor.export_curtailment.threshold_sek_per_kwh` (default 0 = when you'd pay), the
executor forces the inverter export-power limit to **0 W** (via the profile's
`export_power_limit` entity + switch); above it, the feed-in limit is restored. The restore
value is `restore_limit_w`, or auto-captured from the inverter's resting limit the moment
before the first clamp. Runs as an additive post-mode step (skipped in explicit grid-export
mode), honours `shadow_mode`, and the 100 W write threshold prevents EEPROM chatter.
`export_price_sek_kwh` is threaded plan → `SlotPlan` → `ControllerDecision`.

### C4 — Absorption (verified)
With correct prices the optimizer stores near-zero mid-day surplus for the expensive evening
rather than exporting it (`tests/planner/test_kepler_absorption.py`), directly attacking the
evening-import complaint. The `excess_pv_reward_sek_per_kwh` sink incentive is unchanged.

## Economics (why this order)
Absorption into a sink that displaces evening import is worth ~1 SEK/kWh but is
capacity-limited (battery full mid-day, EV often away, VVB saturates after one heat-up).
Curtailment is worth only the avoided negative but is near-free to run and is the backstop
for the residual surplus when all sinks are full. They are complementary: absorption is
offense (capture value), curtailment is defense (stop the bleed).

## Config (all off by default)
```yaml
pricing:
  export_premium_sek_kwh: 0.10          # or export_premium_entity: input_number.darkstar_export_premium
  export_grid_benefit_sek_kwh: 0.05     # or export_grid_benefit_entity: input_number.darkstar_export_grid_benefit
kepler:
  curtailment_penalty_sek: 0.001
executor:
  export_curtailment:
    enabled: true
    threshold_sek_per_kwh: 0.0          # curtail when effective export price < 0
    restore_limit_w: 3720               # feed-in limit to restore to (0 = auto-capture)
```

## Backlog
- Expose `curtailment_kwh` on `KeplerResultSlot` + a sensor so curtailed energy is visible.
- Per-slot hysteresis state if near-zero crossings ever cause limit chatter (contiguous
  negative blocks make this unlikely today).
