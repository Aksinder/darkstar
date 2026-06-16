## dev-20260616.0905 — 2026-06-16 09:07

- feat(ev): departure/target-SoC awareness + per-tank vacation (Fas 0-2)

## dev-20260614.1715 — 2026-06-14 17:17

- fix(recorder): skip observation on glitched/zero load_power (Sungrow modbus)

## dev-20260614.1437 — 2026-06-14 14:39

- fix(loads): unknown_load excludes EV charging (inverter load_power omits grid-fed EVs)

## dev-20260614.1153 — 2026-06-14 11:55

- fix(loads): unknown_load holds last value when total-load source is bad

## dev-20260614.0819 — 2026-06-14 08:20

- feat(loads): publish the unknown-load residual (observability, default OFF)

## dev-20260614.0642 — 2026-06-14 06:43

- feat(hot-water): FMB-style learned draw + persistence for the VVB tank estimator

## dev-20260613.2147 — 2026-06-13 21:49

- fix(executor): FMB SoC writeback review — shadow, call_service, false-full, self-adoption

## dev-20260613.1844 — 2026-06-13 18:46

- feat(executor): FMB SoC manual correction via user-editable input_number

## dev-20260613.1834 — 2026-06-13 18:36

- feat(executor): FMB SoC estimator one-shot manual reseed (seed_soc)

## dev-20260613.1757 — 2026-06-13 17:59

- feat(executor): FMB SoC estimator (self-calibrating dead-reckoning)

## dev-20260611.0117 — 2026-06-11 01:20

- fix(executor+api): EV controller slow-tick (concurrent reads) + planner-status reporting

## dev-20260610.1804 — 2026-06-10 18:06

- feat(executor): wire EV surplus controller into the engine (increment 2)

## dev-20260609.0505 — 2026-06-09 05:06

- fix(load-priority): keep the reliability floor for dynamic-percentile heaters

## dev-20260608.2133 — 2026-06-08 21:35

- feat(load-priority): dynamic percentile WTP cap (price cap tracks the day)

## dev-20260608.1939 — 2026-06-08 19:41

- feat(excess-pv): decouple custom_entity sink so it runs alongside water_heater_boost

## dev-20260608.1716 — 2026-06-08 17:18

- feat(excess-pv): villavagn AC as a price-gated, comfort-bounded cooling sink

## dev-20260608.1651 — 2026-06-08 16:53

- feat(planner): phase-aware imbalance cost — battery covers the heavy phase when economic

## dev-20260608.1630 — 2026-06-08 16:32

- feat(phase): per-slot phase forecasting (learn an hour-of-day phase profile)

## dev-20260608.1436 — 2026-06-08 14:38

- docs: add 2026-06-07 Darkstar conflict-audit report

## dev-20260608.1132 — 2026-06-08 11:34

- feat(planner): extend WTP priority layer to water heaters (increment 2)

## dev-20260607.2229 — 2026-06-07 22:30

- feat(planner): load priority / willingness-to-pay layer (increment 1, flag-gated off)

## dev-20260607.1811 — 2026-06-07 18:13

- fix(executor): don't force grid-charge a near-full battery on PV surplus

## dev-20260607.1730 — 2026-06-07 17:32

- fix(phase): publish realism sensor from the phase-observer loop (was wired to an inactive loop)

## dev-20260607.0914 — 2026-06-07 09:15

- feat(phase): publish forward per-phase imbalance cost (Phase 1 visibility)

## dev-20260607.0758 — 2026-06-07 08:00

- fix(ui): home-gate the live power-flow EV node (away car charging elsewhere not shown)

## dev-20260607.0739 — 2026-06-07 07:40

- fix(ui): hide planned EV-charging series when no charge is planned (away car)

## dev-20260607.0725 — 2026-06-07 07:26

- fix(planner): enforce load_safety_margin_percent floor + centralize export-price overrides

## dev-20260606.2214 — 2026-06-06 22:16

- docs(changelog): add HA add-on CHANGELOG + CI auto-append per dev build

# Changelog — [FORK] Darkstar Energy Manager

Newest first. Dev builds auto-prepend an entry (the triggering commit) via CI; the notes
below summarize the larger changes.

## dev-20260606.2154 — 2026-06-06
- **Negative-price export curtailment** (solver-integrated, off by default)
  - C1: export price = spot + premium + nätnytta − fee (literal **or** HA sensor, follows your contract live)
  - C2: planner curtails/stores instead of exporting when the effective export price < 0
  - C3: executor real-time export clamp — opt-in via `executor.export_curtailment`
  - See `docs/designs/export-curtailment.md`

## dev-20260606.1714 — 2026-06-06
- **EV come-home prediction (Step 1)** — soft, capped battery pre-positioning when the car is
  likely to come home; manual override via HA `input_select` (auto / force_reserve / force_off)
- Robust EV home gate — distance tolerance + grace window
- Fix: thread the home-zone gate through the pipeline (an away EV was being scheduled)

## Earlier
- Phase-aware load modeling (observe + rebalancing recommendations); per-phase sensors
- Hot-water sensor naming consolidation
- Continuous terminal stored-energy value (opt-in)
- See `docs/designs/` for the full design notes
