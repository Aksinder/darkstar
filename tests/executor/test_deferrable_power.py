"""Tests for the power-only auto-arm/run/done state machine (turnkey appliances)."""

from executor.deferrable import (
    AppliancePowerConfig,
    AppliancePowerState,
    WindowSlot,
    cheapest_window_start,
    recommend_appliance_action,
    update_appliance_power_state,
)


def _slots(prices, t0=0.0, slot_len=900.0):
    return [WindowSlot(t0 + i * slot_len, p) for i, p in enumerate(prices)]


class TestForecastWindow:
    def test_run_when_now_is_cheapest(self):
        slots = _slots([0.5, 0.5, 2.0, 2.0, 2.0])
        action, _ = recommend_appliance_action(slots, now_ts=0.0, duration_slots=2, deadline_ts=None)
        assert action == "run"

    def test_defer_to_cheaper_future_block(self):
        slots = _slots([2.0, 2.0, 0.3, 0.3, 0.3])
        action, start = recommend_appliance_action(slots, 0.0, duration_slots=2, deadline_ts=None)
        assert action == "defer" and start == 2 * 900.0

    def test_cannot_fit_before_deadline_runs_now(self):
        slots = _slots([2.0, 0.3, 0.3])
        action, start = recommend_appliance_action(slots, 0.0, duration_slots=2, deadline_ts=900.0)
        assert action == "run" and start is None

    def test_bug_scenario_defers_pre_peak_start(self):
        # Armed at the tail of cheap (0.8) with the day's peak (2.0) imminent; a genuinely
        # cheaper full-cycle window exists after the peak. Must DEFER, not "good price -> run".
        slots = _slots([0.8, 2.0, 2.0, 0.9, 0.9])  # 2-slot cycle
        action, start = recommend_appliance_action(slots, 0.0, duration_slots=2, deadline_ts=None)
        assert action == "defer" and start == 3 * 900.0  # [0.9,0.9]=1.8 beats [0.8,2.0]=2.8

    def test_does_not_pick_a_past_window(self):
        # now is in slot index 2; earlier (cheaper) slots are in the past and ineligible.
        slots = _slots([0.1, 0.1, 0.9, 0.9, 0.2, 0.2])
        start = cheapest_window_start(slots, now_ts=2 * 900.0, duration_slots=2, deadline_ts=None)
        assert start == 4 * 900.0  # the future [0.2,0.2], not the past [0.1,0.1]

CFG = AppliancePowerConfig(
    on_threshold_w=10.0, off_threshold_w=3.0, start_debounce_s=3.0, done_delay_s=300.0
)


def _step(state, power_w, switch_on, now_ts, cfg=CFG):
    return update_appliance_power_state(state, power_w, switch_on, now_ts, cfg)


class TestArm:
    def test_arms_after_debounce(self):
        s = AppliancePowerState()
        s, ev = _step(s, 2000.0, True, 0.0)  # power up, debounce not yet met
        assert s.pending is False and ev is None
        s, ev = _step(s, 2000.0, True, 3.0)  # 3 s sustained => arm
        assert s.pending is True and ev == "armed" and s.start_ts == 3.0

    def test_no_false_arm_before_debounce(self):
        s = AppliancePowerState()
        s, ev = _step(s, 2000.0, True, 0.0)
        s, ev = _step(s, 2000.0, True, 2.0)  # only 2 s
        assert s.pending is False and ev is None

    def test_brief_blip_does_not_arm(self):
        s = AppliancePowerState()
        s, _ = _step(s, 2000.0, True, 0.0)
        s, ev = _step(s, 1.0, True, 1.0)  # drops back before debounce => reset
        assert s.above_since is None and s.pending is False and ev is None

    def test_power_scale_applied(self):
        cfg = AppliancePowerConfig(on_threshold_w=10.0, start_debounce_s=0.0, power_scale=2.0)
        s = AppliancePowerState()
        s, ev = _step(s, 6.0, True, 0.0, cfg)  # 6 * 2 = 12 >= 10
        assert s.pending is True and ev == "armed"


class TestDone:
    def _armed(self):
        s = AppliancePowerState()
        s, _ = _step(s, 2000.0, True, 0.0)
        s, _ = _step(s, 2000.0, True, 3.0)
        assert s.pending is True
        return s

    def test_done_after_delay_while_powered(self):
        s = self._armed()
        s, ev = _step(s, 1.0, True, 100.0)  # drop below off_threshold, plug ON
        assert s.pending is True and ev is None  # delay not yet met
        s, ev = _step(s, 1.0, True, 100.0 + 300.0)  # 300 s low => done
        assert s.pending is False and ev == "done" and s.running is False

    def test_low_power_while_held_off_is_not_done(self):
        # Darkstar cut the plug to defer: 0 W for >5 min must NOT count as completion.
        s = self._armed()
        s, ev = _step(s, 0.0, False, 100.0)  # plug OFF (deferring)
        s, ev = _step(s, 0.0, False, 100.0 + 600.0)  # 10 min held off
        assert s.pending is True and ev is None  # still armed, not "done"

    def test_resume_then_finish(self):
        s = self._armed()
        s, _ = _step(s, 0.0, False, 50.0)  # deferring (plug off)
        s, _ = _step(s, 0.0, False, 200.0)
        s, _ = _step(s, 2000.0, True, 300.0)  # window opens, plug on, resumes
        assert s.pending is True and s.running is True
        s, _ = _step(s, 1.0, True, 400.0)  # cycle winds down
        s, ev = _step(s, 1.0, True, 400.0 + 300.0)
        assert s.pending is False and ev == "done"


class TestRunningFlag:
    def test_running_tracks_draw(self):
        s = AppliancePowerState()
        s, _ = _step(s, 2000.0, True, 0.0)
        assert s.running is True
        s, _ = _step(s, 2000.0, True, 3.0)
        assert s.running is True
        s, _ = _step(s, 1.0, True, 4.0)  # idle draw
        assert s.running is False


class TestReArm:
    def test_rearm_within_cooldown_is_a_silent_continuation(self):
        """A start soon after a done is a soak-pause / resume continuation of the same
        physical programme: pending re-establishes but NO 'armed' event fires, so the
        actuation pause-gate stays closed and notifications don't double-fire."""
        s = AppliancePowerState()
        s, _ = _step(s, 2000.0, True, 0.0)
        s, _ = _step(s, 2000.0, True, 3.0)  # armed
        s, _ = _step(s, 1.0, True, 10.0)
        s, ev = _step(s, 1.0, True, 10.0 + 300.0)  # done
        assert s.pending is False and ev == "done"
        # Heater kicks back in 690 s after the done (inside the 900 s cooldown).
        s, _ = _step(s, 2000.0, True, 1000.0)
        s, ev = _step(s, 2000.0, True, 1003.0)
        assert s.pending is True and ev is None and s.start_ts == 1003.0

    def test_rearms_with_event_after_cooldown(self):
        """A start well after the previous done is a genuine new load: full arm event."""
        s = AppliancePowerState()
        s, _ = _step(s, 2000.0, True, 0.0)
        s, _ = _step(s, 2000.0, True, 3.0)  # armed
        s, _ = _step(s, 1.0, True, 10.0)
        s, ev = _step(s, 1.0, True, 310.0)  # done at t=310
        assert ev == "done"
        # Next load starts 2 h later (past the 900 s cooldown).
        s, _ = _step(s, 2000.0, True, 7510.0)
        s, ev = _step(s, 2000.0, True, 7513.0)
        assert s.pending is True and ev == "armed" and s.start_ts == 7513.0
