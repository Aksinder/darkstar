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
    "AppliancePowerConfig",
    "AppliancePowerState",
    "DeferrableLoadDecision",
    "DeferrableLoadState",
    "WindowSlot",
    "cheapest_window_start",
    "decide_deferrable_loads",
    "recommend_appliance_action",
    "update_appliance_power_state",
]


@dataclass
class WindowSlot:
    """One forward price slot (from the planner schedule) for window scheduling."""

    start_ts: float  # epoch seconds at the slot start
    import_price_sek_kwh: float


def cheapest_window_start(
    slots: list[WindowSlot],
    now_ts: float,
    duration_slots: int,
    deadline_ts: float | None,
    *,
    energy_kwh: float = 0.0,
    wait_cost_sek_per_hour: float = 0.0,
) -> float | None:
    """Start ts of the best contiguous ``duration_slots`` block.

    Considers only blocks that start at/after the current slot and finish at/before
    ``deadline_ts``. Returns the winning block's start ts, or ``None`` if the run can't
    fit before the deadline (caller should then run immediately — deadline pressure).
    This is the forecast-aware core: it costs the WHOLE cycle against the forward price
    curve, so it never starts a long run right before a peak the way a "cheap right now?"
    trigger does.

    **Waiting is not free.** With ``wait_cost_sek_per_hour`` > 0 (and a known
    ``energy_kwh``) each candidate is scored on its ELECTRICITY cost plus the price of
    the delay before it starts, so the cheapest block only wins if it wins by more than
    the wait is worth. Observed 2026-08-22: a dishwasher armed at 13:27 was deferred to
    01:30 the next morning over a 23:00 start that cost SEVEN ÖRE more — formally
    optimal, and not what anyone means by "run it when power is cheap". The default is
    0.0, which is exactly the old absolute-cheapest behaviour.

    The wait price is in SEK per hour, so it is a real preference the owner can reason
    about ("I'll pay 5 öre to have it done an hour sooner") rather than a percentage
    that means different things at 0.9 and 3.6 SEK/kWh. Without ``energy_kwh`` there is
    no way to convert a price curve into kronor, so the penalty stays off rather than
    being applied to a dimensionless number.
    """
    n = len(slots)
    if duration_slots <= 0 or n < duration_slots:
        return None
    slot_len = (slots[1].start_ts - slots[0].start_ts) if n >= 2 else 900.0
    # Allow the in-progress current slot to count as a valid start.
    earliest_start = now_ts - slot_len
    # Energy per slot: the cycle's kWh spread evenly over its duration. This is what
    # turns "sum of prices" into kronor, and it is the same assumption the caller's
    # duration_slots already encodes.
    kwh_per_slot = (energy_kwh / duration_slots) if duration_slots > 0 else 0.0
    penalise = wait_cost_sek_per_hour > 0.0 and energy_kwh > 0.0
    best_start: float | None = None
    best_score: float | None = None
    for i in range(n - duration_slots + 1):
        block = slots[i : i + duration_slots]
        start_ts = block[0].start_ts
        end_ts = block[-1].start_ts + slot_len
        if start_ts < earliest_start:
            continue
        if deadline_ts is not None and end_ts > deadline_ts:
            continue
        price_sum = sum(s.import_price_sek_kwh for s in block)
        if penalise:
            # Kronor, comparable across candidates: electricity + the cost of waiting.
            # Delay is clamped at zero so a block already in progress is not credited
            # for starting in the past.
            delay_h = max(0.0, (start_ts - now_ts)) / 3600.0
            score = price_sum * kwh_per_slot + wait_cost_sek_per_hour * delay_h
        else:
            score = price_sum
        if best_score is None or score < best_score:
            best_score = score
            best_start = start_ts
    return best_start


def recommend_appliance_action(
    slots: list[WindowSlot],
    now_ts: float,
    duration_slots: int,
    deadline_ts: float | None,
    *,
    energy_kwh: float = 0.0,
    wait_cost_sek_per_hour: float = 0.0,
) -> tuple[str, float | None]:
    """Forecast-aware recommendation for an armed cycle.

    Returns ``("run", None)`` when the chosen window has started (or the run can't
    fit before the deadline => run now), else ``("defer", window_start_ts)``.
    See cheapest_window_start for what ``wait_cost_sek_per_hour`` buys.
    """
    start = cheapest_window_start(
        slots, now_ts, duration_slots, deadline_ts,
        energy_kwh=energy_kwh, wait_cost_sek_per_hour=wait_cost_sek_per_hour,
    )
    if start is None:
        return ("run", None)  # cannot fit before deadline -> run now
    if start <= now_ts:
        return ("run", start)
    return ("defer", start)


@dataclass
class AppliancePowerConfig:
    """Power-based auto-arm / done detection tunables for one appliance.

    Defaults mirror the proven HA washing-machine automations (arm >10 W for 3 s,
    done <3 W for 5 min) so a turnkey setup needs only a ``power_sensor`` + the plug.
    """

    on_threshold_w: float = 10.0  # sustained draw >= this => a cycle has started (arm)
    off_threshold_w: float = 3.0  # sustained draw < this => the cycle has finished (done)
    start_debounce_s: float = 3.0  # power must stay above on_threshold this long to arm
    done_delay_s: float = 300.0  # power must stay below off_threshold this long to call it done
    power_scale: float = 1.0  # multiply metered power (e.g. 2 if only one phase is metered)
    # An arm this soon after a done is a CONTINUATION (soak/anti-crease pause that
    # outlasted done_delay_s, sensor hiccup, resume after re-power), not a fresh cycle:
    # it re-arms silently (no "armed" event), so the actuation layer's pause gate —
    # which only opens on a genuine start — can never cut a mid-programme machine.
    # Worst case of a too-long cooldown is a missed defer (runs at current price),
    # never a cut. Genuine back-to-back loads within 15 min just run immediately.
    rearm_cooldown_s: float = 900.0
    # Hand a MANUALLY cut cycle back to Darkstar after this long (0 = never, the
    # historical behaviour). Darkstar never re-energizes a human-cut plug on its own,
    # which is right in the moment — the human may have opened the machine, or stopped
    # it to fix something — but wrong forever: an interrupted programme then sits dead
    # until someone remembers it. After this delay Darkstar reclaims OWNERSHIP only;
    # the ordinary window logic still decides when to actually re-energize, so a
    # reclaimed cycle waits for its cheap window rather than starting on the spot.
    manual_cut_return_s: float = 0.0
    # When the hand-back timer expires, ASK rather than act: send an actionable
    # notification offering "turn it back on" / "leave it off". Silently energizing
    # something a person switched off is the one part of this feature that touches the
    # physical world without being asked, so the default is to put the choice back to
    # them. Ignoring the notification simply keeps the plug off — the safe outcome
    # needs no tap — and Darkstar asks again one interval later.
    manual_cut_ask: bool = False


@dataclass
class AppliancePowerState:
    """Persisted per-appliance state for the power-only arm/run/done state machine."""

    pending: bool = False  # a cycle is armed (started, not yet finished)
    running: bool = False  # currently drawing power
    start_ts: float | None = None  # epoch seconds when this cycle armed
    above_since: float | None = None  # first ts power has been continuously >= on_threshold
    below_since: float | None = None  # first ts power has been continuously < off_threshold
    switch_was_on: bool = True  # switch state at the previous tick (re-power grace)
    last_done_ts: float | None = None  # when the last cycle completed (re-arm cooldown)
    held_by_us: bool = False  # True while DARKSTAR holds the plug OFF (vs a manual cut)
    # When a HUMAN cut the plug mid-cycle (off, pending, not held_by_us). Feeds the
    # hand-back timer; None whenever the plug is on or the cut is ours.
    manual_off_since: float | None = None
    # When we last asked the owner whether to restore this plug. Stops the question
    # repeating every tick, and re-arms the ask one interval later.
    restore_asked_at: float | None = None


# HA stamps every state with the context that produced it. A service call carries the
# calling user's id; a device reporting its own state carries None. So an OFF plug whose
# context names a user OTHER than Darkstar's own token was switched off by a person —
# observed fact, not inference from our own memory.
CUT_BY_US = "ours"
CUT_BY_HUMAN = "human"
CUT_BY_UNKNOWN = "unknown"


def classify_plug_cut(
    *,
    context_user_id: str | None,
    darkstar_user_id: str | None,
    held_by_us: bool,
) -> str:
    """Who is responsible for this plug being off?

    A named user that is not ours beats our own bookkeeping: if a person switched it
    off, it does not matter what Darkstar believes it was holding. That ordering is the
    point of reading the context at all — ``held_by_us`` is memory, and memory survives
    neither a lost state file nor a command we only think landed.

    ``None`` context means the device reported its own state rather than anyone
    commanding it (a reboot, an integration refresh), so it decides nothing; we fall
    back to what we remember, and to UNKNOWN when we remember nothing. UNKNOWN is
    treated as a human cut by callers — never re-energize what you cannot prove you
    switched off yourself.
    """
    if (
        context_user_id is not None
        and darkstar_user_id is not None
        and context_user_id != darkstar_user_id
    ):
        return CUT_BY_HUMAN
    if held_by_us:
        return CUT_BY_US
    if context_user_id is not None and darkstar_user_id is None:
        # Someone commanded it and we cannot yet prove it was not us. Conservative.
        return CUT_BY_UNKNOWN
    return CUT_BY_UNKNOWN


def restore_decision(
    *,
    reclaim_due: bool,
    ask: bool,
    can_notify: bool,
    restore_asked_at: float | None,
    now_ts: float,
    manual_cut_return_s: float,
) -> str:
    """What to do when the hand-back timer expires: "restore" | "ask" | "wait".

    Asking is the honest default where a notification can actually reach someone:
    restoring is the only part of this feature that energizes something a person
    deliberately switched off, so the choice goes back to them rather than being
    taken on their behalf.

    An unanswered question is not a failure — silence keeps the plug off, which is the
    conservative outcome, and the ask re-arms one interval later so a notification
    missed at 3am is not the end of it. With no notify service configured there is
    nobody to ask, so we fall back to restoring: a promise of "hand it back after N
    minutes" that silently never fires would be worse than acting.
    """
    if not reclaim_due:
        return "wait"
    if not (ask and can_notify):
        return "restore"
    if restore_asked_at is None:
        return "ask"
    return "ask" if (now_ts - restore_asked_at) >= manual_cut_return_s else "wait"


def should_reclaim_after_manual_cut(
    *,
    switch_on: bool,
    held_by_us: bool,
    manual_off_since: float | None,
    now_ts: float,
    manual_cut_return_s: float,
) -> bool:
    """Has a human-cut plug waited long enough to come back under Darkstar's control?

    Reclaims OWNERSHIP, not power. What the caller does with it depends on the state:
      * a cycle still ``pending`` — set held_by_us and let the ordinary defer/run window
        logic decide, so the interrupted programme resumes at its cheap window rather
        than lurching on the moment the timer expires;
      * an IDLE appliance — restore the plug to ON, its resting state.

    That second case was excluded in the first draft, on the reasoning that reclaiming
    an idle appliance would let Darkstar switch on a machine nobody started. The owner
    corrected the premise (2026-08-20): the resting state here IS plug-on, because start
    detection watches for the appliance's own power draw. A plug left off is not a
    machine kept safe, it is a detector switched off — the next person loads the
    dishwasher, presses start, and nothing happens. Energizing an idle plug draws
    nothing on its own; it only makes the appliance available again.

    Still opt-in (``manual_cut_return_s`` > 0) and still deliberately slow, because
    re-energizing something a person switched off is a physical act, not a preference.
    """
    if manual_cut_return_s <= 0 or switch_on or held_by_us:
        return False
    if manual_off_since is None:
        return False
    return (now_ts - manual_off_since) >= manual_cut_return_s


def update_appliance_power_state(
    prev: AppliancePowerState,
    power_w: float,
    switch_on: bool,
    now_ts: float,
    cfg: AppliancePowerConfig,
) -> tuple[AppliancePowerState, str | None]:
    """Advance the power-only arm/run/done state machine by one tick.

    Mirrors the proven HA automations so a user supplies only a power sensor:
    - **arm** when draw is sustained >= ``on_threshold_w`` for ``start_debounce_s``;
    - **done** when draw stays < ``off_threshold_w`` for ``done_delay_s`` *while the
      plug is ON* — a zero reading because Darkstar cut power to defer is NOT
      completion (the ``switch_on`` guard), exactly like the automation's
      ``waiting == off`` condition on its Done rule.

    Returns the new state plus an event (``"armed"`` | ``"done"`` | ``None``) the
    caller can use to trigger a replan / notification.
    """
    p = max(0.0, power_w) * cfg.power_scale
    above = p >= cfg.on_threshold_w
    below = p < cfg.off_threshold_w

    above_since = prev.above_since if above else None
    if above and above_since is None:
        above_since = now_ts
    below_since = prev.below_since if below else None
    if below and below_since is None:
        below_since = now_ts
    # Re-power grace: the moment the plug transitions OFF -> ON, the low-draw clock
    # restarts. Without this, below_since carries hours of held-OFF time into the
    # first powered tick and a single lagging 0 W read fires an instant false "done"
    # (stranding the resumed cycle). The done_delay must be measured from re-power.
    if switch_on and not prev.switch_was_on:
        below_since = now_ts if below else None

    pending = prev.pending
    start_ts = prev.start_ts
    last_done_ts = prev.last_done_ts
    running = above
    event: str | None = None

    if not pending:
        # Arm on a sustained start draw (only possible while the plug is powered).
        if above_since is not None and (now_ts - above_since) >= cfg.start_debounce_s:
            pending = True
            start_ts = now_ts
            recently_done = (
                prev.last_done_ts is not None
                and (now_ts - prev.last_done_ts) < cfg.rearm_cooldown_s
            )
            # A re-arm shortly after a done is a continuation of the same physical
            # programme (see rearm_cooldown_s): keep the cycle pending but emit NO
            # "armed" event, so the pause gate stays closed and notifications don't
            # double-fire.
            event = None if recently_done else "armed"
    elif switch_on and below_since is not None and (now_ts - below_since) >= cfg.done_delay_s:
        # Done only when powered AND draw has stayed low long enough. A 0 W reading
        # while Darkstar holds the plug OFF to defer must never look like completion.
        pending = False
        running = False
        start_ts = None
        last_done_ts = now_ts
        event = "done"

    # Stamp / clear the manual-cut clock. Set on the first tick the plug is seen OFF
    # while a cycle is pending; cleared whenever it is powered again. Ownership is not
    # known here (the runtime owns held_by_us), so this is only the "off since" fact —
    # should_reclaim_after_manual_cut() applies the ownership test.
    manual_off_since = prev.manual_off_since
    if switch_on:
        manual_off_since = None
    elif manual_off_since is None:
        manual_off_since = now_ts

    return (
        AppliancePowerState(
            pending=pending,
            running=running,
            start_ts=start_ts,
            above_since=above_since,
            below_since=below_since,
            switch_was_on=switch_on,
            manual_off_since=manual_off_since,
            last_done_ts=last_done_ts,
            held_by_us=prev.held_by_us,
        ),
        event,
    )


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
