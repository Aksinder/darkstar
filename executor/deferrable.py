"""
Deferrable-load executor control (step 3 of docs/designs/deferrable-loads.md).

Pure decision logic that translates the planner's schedule + live HA state into
a desired power-switch state for each deferrable appliance (dishwasher, washing
machine, ...). No I/O here: the engine takes these decisions and writes the
Shelly switch. Keeping it pure makes the control behaviour fully unit-testable.

Control model (auto-detect / cut-and-restore):
- A run becomes *pending* when HA detection fires (queued/started cycle).
- While pending but outside the planned window, Darkstar holds the switch OFF so
  the appliance waits. When the planned slot arrives, the switch goes ON and the
  appliance starts/resumes (Shelly power_on_behavior = on).
- When the run is no longer pending (completed), Darkstar releases control and
  leaves the switch ON so the appliance is powered normally.
- If a load is not under Darkstar control (not pending, or feature disabled),
  its switch is never touched — the user keeps manual control.

Safety: a non-interruptible appliance must resume cleanly when re-powered. A
per-run ``max_hold_minutes`` cap prevents Darkstar from holding a queued cycle
indefinitely if the plan/state desyncs (fail-open to ON).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DeferrableLoadDecision",
    "DeferrableLoadState",
    "decide_deferrable_loads",
]


@dataclass
class DeferrableLoadState:
    """Live HA state for one deferrable load."""

    id: str
    pending: bool = False  # a cycle is queued/started (from HA detection)
    running: bool = False  # appliance currently drawing power
    switch_on: bool = True  # current power-switch state
    # Minutes this run has already been held OFF by Darkstar (for the safety cap).
    held_minutes: float = 0.0


@dataclass
class DeferrableLoadDecision:
    """Desired power-switch state for one deferrable load."""

    id: str
    switch_on: bool
    write: bool  # whether the engine should actually write the switch
    reason: str


def decide_deferrable_loads(
    loads_config: list[dict[str, Any]],
    states: list[DeferrableLoadState],
    run_now: dict[str, bool],
    *,
    enabled: bool,
    max_hold_minutes: float = 720.0,
) -> list[DeferrableLoadDecision]:
    """Decide the switch state for each deferrable load.

    Args:
        loads_config: ``deferrable_loads`` config list (id, enabled, switch_entity).
        states: live per-load HA state.
        run_now: per-load flag — is the load planned to run in the *current* slot?
        enabled: master toggle. When False, Darkstar touches no switches.
        max_hold_minutes: fail-open cap; if a pending run has been held longer
            than this, power it on regardless of the plan (avoids stranding a
            queued cycle on a stale plan).

    Returns:
        One decision per configured, enabled load that has a control switch.
    """
    cfg_by_id = {str(c.get("id", "")): c for c in loads_config if c.get("id")}
    state_by_id = {s.id: s for s in states}
    decisions: list[DeferrableLoadDecision] = []

    for lid, cfg in cfg_by_id.items():
        if not cfg.get("enabled", True) or not cfg.get("switch_entity"):
            continue
        state = state_by_id.get(lid, DeferrableLoadState(id=lid))

        if not enabled:
            decisions.append(
                DeferrableLoadDecision(lid, state.switch_on, write=False, reason="feature disabled")
            )
            continue

        if not state.pending:
            # No queued cycle: leave power on, do not interfere with manual use.
            decisions.append(
                DeferrableLoadDecision(lid, switch_on=True, write=False, reason="no pending run")
            )
            continue

        # Pending run: gate power to the planned window.
        if state.held_minutes >= max_hold_minutes:
            decisions.append(
                DeferrableLoadDecision(
                    lid, switch_on=True, write=not state.switch_on, reason="hold cap reached"
                )
            )
            continue

        should_run = bool(run_now.get(lid, False))
        decisions.append(
            DeferrableLoadDecision(
                lid,
                switch_on=should_run,
                write=(should_run != state.switch_on),
                reason="planned run window" if should_run else "holding for cheaper window",
            )
        )

    return decisions
