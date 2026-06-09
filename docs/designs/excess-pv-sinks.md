# Blueprint: excess-PV sinks — climate (AC) cooling + multi-sink coexistence

**Status:** Implemented. Default off (`executor.excess_pv.sink: disabled`).

**Problem:** On sunny days the 16 kWh battery fills by mid-afternoon and then surplus PV is
exported at the low/negative mid-day price. Two improvements over the original Excess PV
Dispatch (water-heater boost / plain custom entity):

1. soak that surplus into a **comfort cooling load** (the villavagn AC) instead of exporting
   it for next to nothing — but only on **low or negative** price hours, and without
   over-cooling; and
2. let that AC sink run **alongside** the water-heater boost, not instead of it.

This builds on the existing machinery (`docs/designs/` Excess PV Dispatch / RELEASE_NOTES):
the solver already models a per-slot `custom_entity_active` binary gated by a forecast
excess-PV flag and the SoC threshold, and the executor already toggles a custom entity.

## 1. Price-gated activation (`excess_pv.custom_entity.price_ceiling_sek_per_kwh`)

A new optional ceiling restricts the custom-entity sink to slots whose **export price ≤
ceiling** — i.e. only soak surplus locally when grid export pays little or nothing
(including negative prices); sell it otherwise. `None`/blank = legacy unrestricted.

- Solver: `KeplerConfig.excess_pv_price_ceiling_sek_per_kwh`; in the custom-entity branch a
  slot is forced off when `export_price > ceiling`.
- Adapter: `_excess_pv_price_ceiling()`.

## 2. Climate actuation + comfort floor (executor)

`set_custom_entity` detects a `climate.*` entity and delegates to `_set_climate_sink`:

- **ON** → `climate.set_hvac_mode(climate_mode)` (default `cool`) + `set_temperature(target_temp)`.
- **OFF** → `climate.set_hvac_mode("off")`.
- **Comfort floor** → if `current_temperature ≤ comfort_min_temp`, the ON action is skipped
  and the unit forced off, so surplus never over-cools the space.
- Idempotent (no-op when already in the desired mode) and shadow-mode safe.

Plain entities (switch/number/input_boolean) keep the legacy on_value/off_value write path —
behaviour is byte-identical when the entity is not a climate domain.

## 3. Multi-sink coexistence (`excess_pv.custom_entity.enabled`)

The single `sink` selector (`water_heater_boost | custom_entity | disabled`) could only pick
one sink, so enabling the AC used to disable the main-VVB boost. The solver already models
the two as independent constraint blocks — only the config string coupled them. A new
`custom_entity.enabled` flag activates the custom-entity sink **regardless** of `sink`, so it
runs alongside `water_heater_boost`. `sink: custom_entity` still works and implies enabled.

Touch points: `executor/config.py` (flag + parse), `executor/actions.py` (guard),
`executor/engine.py` (actuation gate), `planner/solver/kepler.py`
(`custom_entity_enabled = sink == "custom_entity" or excess_pv_custom_entity_enabled`),
`planner/solver/adapter.py` (wire), `planner/pipeline.py` (compute excess-PV flags when the
flag is set even if the primary sink is `water_heater_boost`).

## Example config — villavagn AC alongside the VVB boost

```yaml
executor:
  excess_pv:
    sink: water_heater_boost          # keep the main-VVB boost
    soc_threshold_percent: 95         # battery essentially full first
    custom_entity:
      enabled: true                   # run the AC sink ALONGSIDE the boost
      entity: climate.villavagn
      price_ceiling_sek_per_kwh: 0.20 # only on low/minus export-price hours
      climate_mode: cool
      target_temp: 22                 # cool to here when ON
      comfort_min_temp: 20            # never cool below this (anti-overcool)
      power_kw: 1.0                   # AC draw estimate (sizes the solver reward)
```

A slot cools only when **all** hold: forecast PV surplus **and** battery ≥ 95 % **and**
export price ≤ ceiling **and** current temperature > `comfort_min_temp`.

## Tests

`tests/planner/test_kepler_custom_entity.py` (price-gate; independent-enable),
`tests/executor/test_executor_actions.py` (climate cool+temp, comfort-floor block, off,
idempotent, shadow, runs-with-boost-sink).
