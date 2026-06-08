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
