## dev-20260825.1946 — 2026-08-25 19:48

- Recorder: meter the finished slot, integrate time-weighted, price by slot key

## dev-20260824.1649 — 2026-08-24 16:51

- Water: anti-short-cycle hold for the surplus boost

## dev-20260824.1307 — 2026-08-24 13:09

- Spa: drive the climate mode; EV cap: gate on commanded draw, not intent

## dev-20260824.0823 — 2026-08-24 08:25

- Price integrity: internal spot as servo source, fail-open C3, negative fetch memo

## dev-20260823.1547 — 2026-08-23 15:48

- ev_surplus: a car that starts itself is ON — reality over memory, and a failure ledger that forgets

## dev-20260823.1145 — 2026-08-23 11:47

- ev_surplus: the return channel — cap the battery's charge power for the cars

## dev-20260822.2341 — 2026-08-22 23:43

- deferrable: price the waiting — stop chasing öre with hours

## dev-20260822.1235 — 2026-08-22 12:37

- planner: coarse-tail horizon — full look-ahead, a quarter of the variables

## dev-20260821.1418 — 2026-08-21 14:20

- cyclic loads: remember what ran (heated_today) and when (gap anchored at last run)

## dev-20260821.0708 — 2026-08-21 07:10

- cyclic loads: plan pumps greedily outside the MILP; absent from plan is not off

## dev-20260820.1827 — 2026-08-20 18:28

- engine: per-load isolation for the water chain and cyclic loop — no more shared fate

## dev-20260820.1801 — 2026-08-20 18:03

- actions: set_cyclic_load called call_service on the wrong object — live outage

## dev-20260820.1658 — 2026-08-20 17:00

- cyclic loads: spacing defaults to zero — the tanks' 5 h default timed out the box

## dev-20260820.1017 — 2026-08-20 10:18

- deferrable: ask before restoring a manually cut plug

## dev-20260820.0939 — 2026-08-20 09:41

- deferrable: read WHO switched the plug, and restore the resting state

## dev-20260820.0926 — 2026-08-20 09:28

- config: sweep phase_aware.weight, dead since the fuse-relief rewrite

## dev-20260820.0801 — 2026-08-20 08:03

- phase_aware: repurpose from phantom billing cost to fuse relief

## dev-20260820.0532 — 2026-08-20 05:35

- engine: rebuild the deferrable controller when its config changes on reload

## dev-20260820.0515 — 2026-08-20 05:17

- ev_surplus: auto-expiry on the manual override (mirrors the heaters)

## dev-20260819.1826 — 2026-08-19 18:28

- controller: cap export_with_load_w at the discharge limit we command

## dev-20260819.1800 — 2026-08-19 18:02

- load_balancer: port upstream's per-phase fuse guard (observe-only)

## dev-20260819.1749 — 2026-08-19 17:51

- cyclic_run: take the surplus percentile from the export series

## dev-20260819.1658 — 2026-08-19 17:00

- cyclic_loads: opportunistic surplus and presence run gates

## dev-20260819.1529 — 2026-08-19 15:31

- cyclic_loads: pool pump and filter as first-class planned loads

## dev-20260819.0335 — 2026-08-19 03:37

- Tell a human when a write never reaches the appliance

## dev-20260818.1828 — 2026-08-18 18:30

- may_skip_day, and a manual override for water heaters

## dev-20260818.1744 — 2026-08-18 17:46

- Per-load window for the dynamic WTP percentile

## dev-20260818.1451 — 2026-08-18 14:54

- Make the percentile window configurable, and backfill it from passed hours

## dev-20260818.0458 — 2026-08-18 05:00

- load_groups: cap the loads that share a sub-panel

## dev-20260815.1313 — 2026-08-15 13:15

- Compare the real appliance against our own intent, not just our helper

## dev-20260815.1136 — 2026-08-15 11:37

- Push a self-thermostatted heater to its boost target on measured surplus

## dev-20260815.1021 — 2026-08-15 10:23

- The servo was blind to its own actuation and switched off the load it created

## dev-20260815.1002 — 2026-08-15 10:03

- Battery-yield gate becomes a fill DEADLINE, not an SoC threshold

## dev-20260815.0949 — 2026-08-15 09:50

- Idle-hold read the battery with the wrong sign convention

## dev-20260815.0900 — 2026-08-15 09:02

- Idle-hold: a non-numeric water target is not our case, act normally

## dev-20260815.0846 — 2026-08-15 08:48

- Wake a sleeping car on actuation failure; idle-hold for the spa's own thermostat

## dev-20260814.0658 — 2026-08-14 07:00

- feat(planner): absorb_cap_kwh_per_day config override for water heaters

## dev-20260814.0500 — 2026-08-14 05:02

- feat(executor): per-device water-heater temperature overrides

## dev-20260813.1915 — 2026-08-13 19:17

- fix(executor): battery-yield gate conditioned on PLANNED battery charging

## dev-20260813.1843 — 2026-08-13 18:45

- feat(executor): battery-yield gate — the home battery fills first on spike days

## dev-20260813.1810 — 2026-08-13 18:12

- fix(executor): R1 SoC floor applies to force_export quick actions too

## dev-20260813.1720 — 2026-08-13 17:22

- fix(executor): actuation-failure backoff — a sleeping Tesla must not be hammered

## dev-20260813.0647 — 2026-08-13 06:49

- fix(executor): S3 review fixes — urgency source-awareness, bounded hold, vacation gate

## dev-20260812.1847 — 2026-08-12 18:49

- fix(executor): hotfix — engine attribute is inverter_profile, not .profile

## dev-20260812.1649 — 2026-08-12 16:51

- fix(executor): S4 review fixes — battery shed reaches charge mode, export-aware clamp, hardened fail-safes

## dev-20260812.1553 — 2026-08-12 15:55

- fix(config): resolve symlinks before atomic write — saves were silently lost

## dev-20260812.1339 — 2026-08-12 13:41

- feat(ev): comfort demotion — "FMB till 150 km, sedan Teslan, sedan FMB mer"

## dev-20260812.0638 — 2026-08-12 06:40

- fix(config): stop comment-token duplication in template merge (config bloat)

## dev-20260811.1731 — 2026-08-11 17:33

- feat(ev): config-constant comfort cap (target_soc) per charger

## dev-20260811.1532 — 2026-08-11 15:33

- fix(executor): propagate reloaded config to the dispatcher

## dev-20260810.0659 — 2026-08-10 07:01

- fix(solver): close the three accepted review MINORs on the absorption cap

## dev-20260809.1939 — 2026-08-09 19:41

- feat(observations): quarantine historical price-mint rows (label, never delete)

## dev-20260808.1656 — 2026-08-08 16:59

- style(types): rowcount lives on CursorResult, not Result[Any]

## dev-20260804.0742 — 2026-08-04 07:44

- feat(ev-surplus): anti-hunt hardening — quantum deadband, Schmitt, per-charger pacing, deadlock fix

## dev-20260804.0516 — 2026-08-04 05:18

- feat(executor): C3 switch-method export curtailment + export-mode limit fix

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
