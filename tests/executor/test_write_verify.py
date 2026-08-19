"""Escalation when a write never reaches the appliance.

Darkstar's older verification read back the entity it WROTE — for a bridged heater
that is its own helper, so "input_number is 20" gets confirmed while the tub heats on.
Both live failures on 2026-08-19 came from that gap: the spa ran heat/1800 W for
3.5 h with the helper reading 20, and a shed command landed on an `unavailable` entity
and reported success.

Drift detection now corrects both. This is the other half: telling a human when the
CORRECTION keeps losing — a correction loop that silently fails is worse than none,
because the log looks busy while the appliance does as it pleases.
"""

from __future__ import annotations

from executor.write_verify import VerifySignal, VerifyState, note_attempt


def _run(results, threshold=3):
    """Fold a sequence of outcomes, returning the signal emitted at each step."""
    state, signals = VerifyState(), []
    for ok in results:
        state, sig = note_attempt(state, ok=ok, threshold=threshold)
        signals.append(sig)
    return state, signals


class TestItAlertsOnlyWhenTheCorrectionIsLosing:
    def test_one_failure_is_silent(self):
        """The write has been sent and the device has not answered yet."""
        _, sig = _run([False])
        assert sig == [VerifySignal.NONE]

    def test_two_failures_are_still_silent(self):
        """A slow integration is not a fault."""
        _, sigs = _run([False, False])
        assert sigs == [VerifySignal.NONE, VerifySignal.NONE]

    def test_the_third_consecutive_failure_alerts(self):
        _, sigs = _run([False, False, False])
        assert sigs[-1] == VerifySignal.ALERT

    def test_a_success_between_failures_resets_the_count(self):
        """Two blips an hour apart must not add up to an alarm."""
        _, sigs = _run([False, False, True, False, False])
        assert VerifySignal.ALERT not in sigs


class TestItDoesNotBecomeAnAlarmClock:
    def test_a_persistent_fault_alerts_exactly_once(self):
        _, sigs = _run([False] * 20)
        assert sigs.count(VerifySignal.ALERT) == 1

    def test_recovery_is_reported_once(self):
        _, sigs = _run([False, False, False, True, True, True])
        assert sigs.count(VerifySignal.RECOVERED) == 1
        assert sigs[3] == VerifySignal.RECOVERED

    def test_recovery_is_silent_if_we_never_alerted(self):
        """A blip that never crossed the threshold stays quiet in both directions."""
        _, sigs = _run([False, True])
        assert set(sigs) == {VerifySignal.NONE}

    def test_a_second_episode_alerts_again(self):
        """Once resolved, the next real fault must not be swallowed."""
        _, sigs = _run([False] * 3 + [True] + [False] * 3)
        assert sigs.count(VerifySignal.ALERT) == 2


class TestThreshold:
    def test_a_threshold_of_one_alerts_immediately(self):
        _, sigs = _run([False], threshold=1)
        assert sigs == [VerifySignal.ALERT]

    def test_zero_is_clamped_to_one_rather_than_never_firing(self):
        """A nonsense threshold must not silently disable the alarm."""
        _, sigs = _run([False], threshold=0)
        assert sigs == [VerifySignal.ALERT]

    def test_a_high_threshold_waits(self):
        _, sigs = _run([False] * 9, threshold=10)
        assert VerifySignal.ALERT not in sigs


class TestStateIsImmutable:
    def test_the_caller_cannot_mutate_the_previous_state(self):
        before = VerifyState(streak=2)
        after, _ = note_attempt(before, ok=False, threshold=3)
        assert before.streak == 2, "note_attempt must not mutate its input"
        assert after.streak == 3
