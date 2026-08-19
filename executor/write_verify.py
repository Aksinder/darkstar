"""
Escalation for writes that never reach the appliance.

Darkstar's existing verification reads back the entity it WROTE — for a bridged
heater that is its own helper, so "input_number is 20" is confirmed while the tub
merrily heats on. Two live failures on 2026-08-19 came from exactly that gap:

  * The spa ran heat/1800 W for 3.5 h with the helper reading 20 the whole time.
  * A shed command landed on an entity that was `unavailable`; the service call
    reported success and nothing happened.

Drift detection (see water_hold.detect_appliance_drift) now catches both and
re-asserts. What it cannot do is tell anyone when the re-assertion ITSELF keeps
failing — and a correction loop that silently loses is worse than no loop, because
the logs look busy while the appliance does as it pleases.

This module is the escalation half: count consecutive failures, alert ONCE when the
count says the correction is not working, and say so again when it recovers. It is
pure — no I/O, no clock — so the thresholds are testable without a live system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class VerifySignal(Enum):
    """What the caller should tell the human, if anything."""

    NONE = "none"
    ALERT = "alert"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class VerifyState:
    """Per-target memory. Frozen so a caller cannot mutate it by accident."""

    streak: int = 0
    alerted: bool = False


def note_attempt(
    state: VerifyState, *, ok: bool, threshold: int = 3
) -> tuple[VerifyState, VerifySignal]:
    """
    Fold one verification result into the state.

    Args:
        state: previous state for this target.
        ok: did the appliance match our intent this tick?
        threshold: consecutive failures before alerting. Not 1: one tick of
            mismatch is normal — the write has been sent and the device has not
            answered yet. Two can be a slow integration. Three consecutive means
            the re-assertion is not landing, which is the case worth waking
            someone for.

    Returns:
        (new state, signal). ALERT fires exactly once per failure episode, so a
        fault that persists for hours does not become an hourly alarm; RECOVERED
        fires only if we had alerted, so a blip that never reached the threshold
        stays silent in both directions.
    """
    if ok:
        if state.alerted:
            return VerifyState(), VerifySignal.RECOVERED
        return VerifyState(), VerifySignal.NONE

    streak = state.streak + 1
    if streak >= max(1, threshold) and not state.alerted:
        return VerifyState(streak=streak, alerted=True), VerifySignal.ALERT
    return replace(state, streak=streak), VerifySignal.NONE
