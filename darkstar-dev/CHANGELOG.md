## dev-20260803.2003 — 2026-08-03 20:05

- feat(executor): runtime SoC-floor guard on battery-export intent (arbitrage gate R1)

## dev-20260803.1915 — 2026-08-03 19:17

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260803.1906 — 2026-08-03 19:09

- fix(recorder): balance-rescue load instead of skipping + statistics repair of corrupted history

## dev-20260726.1316 — 2026-07-26 13:17

- fix(planner): UTC-aware retry timestamps — naive-local stalled replanning for hours

## dev-20260722.1658 — 2026-07-22 17:01

- fix(solver): CBC fallback on slow no-incumbent timeout + failing-instance dump + 240s budget

## dev-20260715.0846 — 2026-07-15 08:48

- feat(hot-water): switch-aware fill estimator — count draw down only when the switch is OFF

## dev-20260714.2144 — 2026-07-14 21:46

- Merge pull request #3 from Aksinder/fix/pv-training-clean-floor

## dev-20260712.2113 — 2026-07-12 21:15

- Merge fork/main (build #18 version bump) into deploy-main

## dev-20260712.2052 — 2026-07-12 20:53

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260710.0545 — 2026-07-10 05:47

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260709.1754 — 2026-07-09 17:55

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260709.1031 — 2026-07-09 10:33

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260708.1942 — 2026-07-08 19:44

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260708.1636 — 2026-07-08 16:38

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260707.2019 — 2026-07-07 20:21

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260707.2001 — 2026-07-07 20:03

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260707.1410 — 2026-07-07 14:12

- Merge remote-tracking branch 'fork/main' into deploy-main

## dev-20260705.2024 — 2026-07-05 20:26

- perf+guard(solver): proven-optimal CBC in seconds — hourly water blocks, threads, honest time-box handling

## dev-20260705.1112 — 2026-07-05 11:14

- fix(solver): CBC-first with a real time budget — GLPK was shipping garbage as "Optimal"

## dev-20260705.1100 — 2026-07-05 11:02

- fix(executor+planner): humans outrank plans — manual-ON respect, real boost, vacation truth

## dev-20260704.2307 — 2026-07-04 23:09

- fix(config): stop the migration from deleting the smart-appliance config

## dev-20260704.1906 — 2026-07-04 19:08

- feat(executor+planner): deferrable plug actuation (Fas 3) + p10-weighted PV

## dev-20260704.1833 — 2026-07-04 18:35

- feat(honesty): full-objective reporting + ML baseline A/B rows

## dev-20260704.1817 — 2026-07-04 18:19

- feat(executor): drive switch/input_boolean water-heater targets directly

## dev-20260704.1802 — 2026-07-04 18:04

- Merge branch 'upstream-fixes' into deploy-main

## dev-20260704.1203 — 2026-07-04 12:05

- feat(planner): effekttariff peak-demand charge + wire the soft import cap

## dev-20260630.2236 — 2026-06-30 22:38

- feat(pricing): add fees_include_vat for VAT-inclusive grid fee helpers

## dev-20260629.1727 — 2026-06-29 17:29

- fix(planner): model hybrid-inverter PV share for max_ac_power_kw (Option A)

## dev-20260629.1017 — 2026-06-29 10:19

- feat(pricing): single-source import price — grid_transfer_fee/energy_tax entity overrides

## dev-20260617.0626 — 2026-06-17 06:28

- feat(deferrable): turnkey smart-appliance controller (power sensor in, observe-first)

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
