"""load-balancer-graceful-degradation: target margin, averaged ramp-up, pause
debounce, severe overload, 1-phase relief, and a replay of the prod
2026-09-25 afternoon phase history."""

from datetime import datetime, timedelta
from typing import ClassVar

import pytest

from executor.config import LoadBalancingConfig
from executor.load_balancer import EVBalancerInput, LoadBalancer, ShedLoadInput
from tests.executor.fake_relief import InstantRelief

BASE = datetime(2026, 9, 25, 16, 0, 0)
TICK_S = 5


def make_lb(**overrides) -> LoadBalancer:
    """16 A fuse with the shipped graceful-degradation defaults."""
    cfg = LoadBalancingConfig(enabled=True, main_fuse_a=16)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return LoadBalancer(cfg, relief_trigger=InstantRelief(cfg))


def fresh(at: datetime) -> dict[int, datetime]:
    return {1: at, 2: at, 3: at}


def ev(
    setpoint,
    target=10,
    phases=(1, 2, 3),
    relief=False,
    line=1,
    draw=None,
    at_risk=True,
    surplus=False,
) -> EVBalancerInput:
    return EVBalancerInput(
        "goe",
        list(phases),
        setpoint,
        target,
        min_current_a=6,
        max_current_a=16,
        effective_draw_a=draw,
        phase_1_line=line,
        relief_available=relief,
        goal_at_risk=at_risk,
        surplus_slot=surplus,
    )


def warm_up(lb: LoadBalancer, grid: dict[int, float], entry=None, seconds=60) -> datetime:
    """Feed identical fresh readings for a full averaging window."""
    now = BASE
    for i in range(seconds // TICK_S + 1):
        now = BASE + timedelta(seconds=i * TICK_S)
        lb.tick(now, grid, fresh(now), [entry] if entry else [])
    return now


# --- 2.3 Averaged sensing ---------------------------------------------------


class TestAveragedSensing:
    def test_spike_moves_average_only_by_its_share(self):
        lb = make_lb()
        now = warm_up(lb, {1: 9.0, 2: 9.0, 3: 9.0})
        now += timedelta(seconds=TICK_S)
        lb.tick(now, {1: 9.0, 2: 15.0, 3: 9.0}, fresh(now), [])
        avg = lb.averaged_current_a(2, now)
        assert avg is not None
        # 13 samples in the 60 s window, one of them the 15 A spike
        assert avg == pytest.approx(9.0 + 6.0 / 13, abs=0.01)

    def test_increases_blocked_after_startup_until_window_fills(self):
        lb = make_lb()
        grid = {1: 2.0, 2: 2.0, 3: 2.0}
        setpoint = 6
        for i in range(12):  # 0..55 s: window not yet full
            now = BASE + timedelta(seconds=i * TICK_S)
            out = lb.tick(now, grid, fresh(now), [ev(setpoint)]).ev_outputs[0]
            assert out.target_a == 6
        now = BASE + timedelta(seconds=60)
        out = lb.tick(now, grid, fresh(now), [ev(setpoint)]).ev_outputs[0]
        assert out.target_a == 7

    def test_stale_episode_resets_the_window(self):
        lb = make_lb()
        grid = {1: 2.0, 2: 2.0, 3: 2.0}
        now = warm_up(lb, grid)
        assert lb.averaged_current_a(1, now) is not None
        # L1 stops reporting for longer than sensor_stale_after_s
        stale_at = now
        now += timedelta(seconds=35)
        lb.tick(now, grid, {1: stale_at, 2: now, 3: now}, [])
        assert lb.averaged_current_a(1, now) is None
        # Fresh again: increases stay blocked until a full window has passed
        recovered = now + timedelta(seconds=TICK_S)
        out = lb.tick(recovered, grid, fresh(recovered), [ev(6)]).ev_outputs[0]
        assert out.target_a == 6
        for i in range(1, 13):
            t = recovered + timedelta(seconds=i * TICK_S)
            out = lb.tick(t, grid, fresh(t), [ev(6)]).ev_outputs[0]
        assert out.target_a == 7

    def test_own_ramp_is_not_hidden_by_averaging_lag(self):
        """The chargers count at their latest draw, so an averaged ramp cannot
        overshoot the target by the window's lag."""
        lb = make_lb()
        house = 1.0
        setpoint = 6
        now = BASE
        for i in range(200):
            now = BASE + timedelta(seconds=i * TICK_S)
            grid = {1: house + setpoint, 2: house + setpoint, 3: house + setpoint}
            out = lb.tick(now, grid, fresh(now), [ev(setpoint, target=16)]).ev_outputs[0]
            setpoint = out.target_a
            assert house + setpoint <= 16 * 0.85 + 1e-9
        assert setpoint == 12  # 1 A house + 12 A = 13 A <= 13.6 A; 14 A would exceed


# --- 3.3 Margin, debounce and severe pause ----------------------------------


class TestTargetMarginAndPause:
    def test_quiet_night_ramps_to_full_planned_current(self):
        lb = make_lb()
        house = 2.0
        setpoint = 6
        for i in range(120):
            now = BASE + timedelta(seconds=i * TICK_S)
            grid = dict.fromkeys((1, 2, 3), house + setpoint)
            out = lb.tick(now, grid, fresh(now), [ev(setpoint, target=10)]).ev_outputs[0]
            setpoint = out.target_a
        assert setpoint == 10  # projected 12 A <= 13.6 A

    def test_ramp_stops_at_target_without_reducing(self):
        lb = make_lb()
        grid = {1: 13.0, 2: 8.0, 3: 8.0}  # L1 13 A with the charger at 8 A
        now = warm_up(lb, grid, ev(8))
        now += timedelta(seconds=TICK_S)
        out = lb.tick(now, grid, fresh(now), [ev(8)]).ev_outputs[0]
        assert out.target_a == 8  # 9 A would project 14 A > 13.6 A; no reduction below fuse

    def test_load_hovering_at_95_percent_blocks_increase(self):
        lb = make_lb()
        grid = {1: 15.2, 2: 8.0, 3: 8.0}
        now = warm_up(lb, grid, ev(8))
        now += timedelta(seconds=TICK_S)
        out = lb.tick(now, grid, fresh(now), [ev(8)]).ev_outputs[0]
        assert out.target_a == 8

    def test_one_second_spike_does_not_pause(self):
        lb = LoadBalancer(LoadBalancingConfig(enabled=True, main_fuse_a=16), interval_seconds=15)
        now = warm_up(lb, {1: 10.0, 2: 10.0, 3: 10.0}, ev(6))
        now += timedelta(seconds=1)
        out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 19.0}, fresh(now), [ev(6)]).ev_outputs[0]
        assert out.target_a is not None and out.state != "paused"
        now += timedelta(seconds=1)
        out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 11.0}, fresh(now), [ev(6)]).ev_outputs[0]
        assert out.target_a is not None and out.target_a >= 6
        assert out.state != "paused"

    def test_floor_pauses_immediately_when_trigger_requests_relief(self):
        lb = make_lb()
        now = BASE
        out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 19.0}, fresh(now), [ev(6)]).ev_outputs[0]
        assert out.state == "paused"

    def test_severe_overload_pauses_in_the_same_tick(self):
        lb = make_lb()
        now = BASE
        # 21 A on a 16 A fuse = 131 % > 125 %
        out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 21.0}, fresh(now), [ev(6, relief=True)])
        assert out.ev_outputs[0].state == "paused"
        assert not out.ev_outputs[0].relief_1p_requested

    def test_decrease_uses_the_momentary_reading(self):
        lb = make_lb()
        now = warm_up(lb, {1: 12.0, 2: 8.0, 3: 8.0}, ev(10))
        now += timedelta(seconds=TICK_S)
        out = lb.tick(now, {1: 19.0, 2: 8.0, 3: 8.0}, fresh(now), [ev(10)]).ev_outputs[0]
        assert out.target_a is not None and out.target_a <= 7

    def test_reduction_above_floor_is_immediate_not_debounced(self):
        lb = make_lb()
        now = BASE
        out = lb.tick(now, {1: 20.0, 2: 8.0, 3: 8.0}, fresh(now), [ev(12)]).ev_outputs[0]
        assert out.target_a == 8

    def test_resume_blocked_while_momentary_reading_exceeds_target(self):
        lb = make_lb()
        now = BASE
        lb.tick(now, {1: 10.0, 2: 10.0, 3: 22.0}, fresh(now), [ev(6)])
        # L3 reads 9 A: 9 + 6 = 15 A > 13.6 A, although headroom to the fuse is 7 A
        t = now
        for _ in range(60):
            t += timedelta(seconds=TICK_S)
            out = lb.tick(t, {1: 1.0, 2: 1.0, 3: 9.0}, fresh(t), [ev(None)]).ev_outputs[0]
            assert out.target_a is None

    def test_shed_restore_waits_for_averaged_margin(self):
        lb = make_lb(resume_delay_s=10)
        wh = ShedLoadInput("wh", "water_heater", [2], power_w=500.0, power_source="configured")
        now = BASE
        lb.tick(now, {1: 5.0, 2: 17.0, 3: 5.0}, fresh(now), [wh])
        t = now
        for _ in range(30):
            t += timedelta(seconds=TICK_S)
            status = lb.tick(t, {1: 5.0, 2: 14.0, 3: 5.0}, fresh(t), [wh])
            assert status.shed_outputs[0].shed  # 14 A > 13.6 A target
        for _ in range(14):
            t += timedelta(seconds=TICK_S)
            status = lb.tick(t, {1: 5.0, 2: 6.0, 3: 5.0}, fresh(t), [wh])
        assert not status.shed_outputs[0].shed


class TestSafetyInvariants:
    """Fuse-protection invariants of the graceful-degradation changes."""

    def test_increase_capped_by_momentary_headroom(self):
        """A low average must never let a raise push this tick's reading past
        the fuse: with 0.5 A momentary headroom no 1 A step is taken."""
        lb = make_lb()
        now = warm_up(lb, {1: 6.0, 2: 6.0, 3: 6.0}, ev(6))
        now += timedelta(seconds=TICK_S)
        out = lb.tick(now, {1: 6.0, 2: 15.5, 3: 6.0}, fresh(now), [ev(6)]).ev_outputs[0]
        assert out.target_a == 6

    def test_increase_step_limited_to_whole_momentary_headroom(self):
        lb = make_lb(increase_step_a=4)
        # One tick short of a full window: nothing ramped (and so no granted
        # step is pending) before the tick under test.
        now = warm_up(lb, {1: 6.0, 2: 6.0, 3: 6.0}, ev(6), seconds=55)
        now += timedelta(seconds=TICK_S)
        # Average leaves room for +4 A, momentary L2 headroom only 2.5 A.
        out = lb.tick(now, {1: 6.0, 2: 13.5, 3: 6.0}, fresh(now), [ev(6)]).ev_outputs[0]
        assert out.target_a == 8

    def test_debounce_hold_never_above_prior_setpoint(self):
        """A setpoint below the floor cannot be 'held' at the floor during an
        overload (that would be a raise) — it pauses instead."""
        lb = make_lb()
        now = BASE
        entry = EVBalancerInput("goe", [1, 2, 3], 5, 10, min_current_a=6, max_current_a=16)
        out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 18.0}, fresh(now), [entry]).ev_outputs[0]
        assert out.target_a is None and out.state == "paused"

    def test_shed_relief_hold_never_below_floor(self):
        """Holding for unmeasured shed relief uses the measured draw, but never
        commands a sub-floor setpoint."""
        lb = make_lb()
        wh = ShedLoadInput("wh", "water_heater", [3])
        now = BASE
        status = lb.tick(now, {1: 8.0, 2: 8.0, 3: 18.0}, fresh(now), [wh, ev(10, draw=4)])
        assert status.shed_outputs[0].shed
        out = status.ev_outputs[0]
        assert out.target_a is not None and 6 <= out.target_a <= 10

    def test_stale_sensor_forces_floor_then_pauses_after_resume_delay(self):
        lb = make_lb()
        t = BASE
        stale_ts = {1: t, 2: t, 3: t - timedelta(seconds=3600)}
        out = lb.tick(t, {1: 10.0, 2: 10.0, 3: 10.0}, stale_ts, [ev(10)]).ev_outputs[0]
        assert out.target_a == 6 and out.state == "stale_fallback"
        t += timedelta(seconds=lb.config.resume_delay_s)
        stale_ts = {1: t, 2: t, 3: t - timedelta(seconds=3600)}
        out = lb.tick(t, {1: 10.0, 2: 10.0, 3: 10.0}, stale_ts, [ev(6)]).ev_outputs[0]
        assert out.target_a is None and out.state == "paused"


# --- 4.x 1-phase relief -----------------------------------------------------


class TestReliefLadder:
    def _sustain(self, lb, grid, entry, ticks=1):
        now = BASE
        out = None
        for i in range(ticks):
            now = BASE + timedelta(seconds=i * TICK_S)
            out = lb.tick(now, grid, fresh(now), [entry]).ev_outputs[0]
        return out

    def test_l3_overload_requests_1_phase_on_l1(self):
        lb = make_lb()
        # L3 headroom -4 A at the floor, L1 has room
        out = self._sustain(lb, {1: 8.0, 2: 8.0, 3: 20.0}, ev(6, relief=True))
        assert out.relief_1p_requested
        assert out.target_a == 6 and out.state == "throttling"
        assert "L1" in out.reason and "L3" in out.reason

    def test_overload_on_the_1_phase_line_pauses(self):
        lb = make_lb()
        out = self._sustain(lb, {1: 20.0, 2: 8.0, 3: 8.0}, ev(6, relief=True))
        assert not out.relief_1p_requested
        assert out.state == "paused"

    def test_dwell_not_elapsed_pauses(self):
        lb = make_lb()
        out = self._sustain(lb, {1: 8.0, 2: 8.0, 3: 20.0}, ev(6, relief=False))
        assert not out.relief_1p_requested
        assert out.state == "paused"

    def test_charger_wired_to_l2_is_not_relief_for_l2(self):
        lb = make_lb()
        out = self._sustain(lb, {1: 8.0, 2: 20.0, 3: 8.0}, ev(6, relief=True, line=2))
        assert not out.relief_1p_requested and out.state == "paused"

    def test_charger_without_phase_switching_is_throttle_then_pause(self):
        lb = make_lb()
        first = self._sustain(lb, {1: 8.0, 2: 8.0, 3: 24.0 - 5}, ev(12), ticks=1)
        assert first.target_a == 9  # immediate throttle by the deficit
        lb2 = make_lb()
        out = self._sustain(lb2, {1: 8.0, 2: 8.0, 3: 20.0}, ev(6))
        assert out.state == "paused" and not out.relief_1p_requested

    def test_1_phase_charger_is_never_asked_for_relief(self):
        lb = make_lb()
        out = self._sustain(lb, {1: 20.0, 2: 8.0, 3: 8.0}, ev(6, phases=(1,), relief=True))
        assert out.state == "paused" and not out.relief_1p_requested

    def test_phase_set_expansion_restarts_at_floor(self):
        lb = make_lb()
        now = warm_up(lb, {1: 13.0, 2: 1.0, 3: 1.0}, ev(12, target=16, phases=(1,)))
        now += timedelta(seconds=TICK_S)
        out = lb.tick(now, {1: 13.0, 2: 1.0, 3: 1.0}, fresh(now), [ev(12, target=8)])
        assert out.ev_outputs[0].target_a == 6


class TestReliefOnlyWhenGoalAtRisk:
    """ev-charging-losses 6.3: lossy 1-phase relief only when the goal is at risk."""

    def _sustain(self, lb, grid, entry, ticks=1):
        return TestReliefLadder._sustain(self, lb, grid, entry, ticks)  # type: ignore[arg-type]

    def test_goal_on_track_pauses_instead_of_1_phase(self):
        out = self._sustain(make_lb(), {1: 8.0, 2: 8.0, 3: 20.0}, ev(6, relief=True, at_risk=False))
        assert not out.relief_1p_requested
        assert out.state == "paused"

    def test_goal_at_risk_requests_1_phase(self):
        out = self._sustain(make_lb(), {1: 8.0, 2: 8.0, 3: 20.0}, ev(6, relief=True, at_risk=True))
        assert out.relief_1p_requested

    def test_resume_stays_in_current_mode_when_on_track(self):
        lb = make_lb()
        _pause(lb, BASE, relief=True)
        grid = {1: 2.0, 2: 2.0, 3: 12.0}  # only the 1-phase alternative fits
        outs = _paused_ticks(lb, BASE, grid, ev(None, relief=True, at_risk=False), 120)
        assert all(o.target_a is None and o.state == "paused" for _, o in outs)

    def test_surplus_slot_on_track_requests_1_phase(self):
        entry = ev(6, relief=True, at_risk=False, surplus=True)
        out = self._sustain(make_lb(), {1: 8.0, 2: 8.0, 3: 20.0}, entry)
        assert out.relief_1p_requested and out.state == "throttling"

    def test_non_surplus_on_track_pauses(self):
        entry = ev(6, relief=True, at_risk=False, surplus=False)
        out = self._sustain(make_lb(), {1: 8.0, 2: 8.0, 3: 20.0}, entry)
        assert not out.relief_1p_requested and out.state == "paused"

    def test_resume_in_surplus_slot_may_use_1_phase(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE, relief=True)
        grid = {1: 2.0, 2: 2.0, 3: 12.0}  # only the 1-phase alternative fits
        entry = ev(None, relief=True, at_risk=False, surplus=True)
        out = _paused_ticks(lb, BASE, grid, entry, 15)[-1][1]
        assert out.relief_1p_requested and out.refit_from_pause and out.target_a == 10


# --- 8.11 Quick re-fit while paused (D17) -------------------------------------


def _pause(lb: LoadBalancer, now: datetime, relief: bool = False) -> None:
    """Severe L3 overload pauses the charger immediately."""
    out = lb.tick(now, {1: 10.0, 2: 10.0, 3: 22.0}, fresh(now), [ev(6, relief=relief)])
    assert out.ev_outputs[0].state == "paused"


def _paused_ticks(lb, start, grid, entry, seconds):
    """Tick every TICK_S for `seconds` after `start`; return (time, output) pairs."""
    outs = []
    for i in range(1, seconds // TICK_S + 1):
        t = start + timedelta(seconds=i * TICK_S)
        outs.append((t, lb.tick(t, grid, fresh(t), [entry]).ev_outputs[0]))
    return outs


class TestQuickRefit:
    QUIET: ClassVar[dict[int, float]] = {1: 2.0, 2: 2.0, 3: 2.0}

    def test_refits_after_confirm_window_not_resume_delay(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        outs = _paused_ticks(lb, BASE, self.QUIET, ev(None), 15)
        assert [o.target_a is None for _, o in outs] == [True, True, False]
        t, out = outs[-1]
        # Fitting since BASE+5 s; resumed at BASE+15 s, far below resume_delay_s.
        assert (t - BASE).total_seconds() == 15 < lb.config.resume_delay_s
        assert out.state == "throttling" and not out.relief_1p_requested

    def test_resumes_at_largest_fitting_amps_capped_by_plan(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        # 13.6 A target - 2 A house = 11.6 A -> 11 A, under a 16 A plan
        out = _paused_ticks(lb, BASE, self.QUIET, ev(None, target=16), 15)[-1][1]
        assert out.target_a == 11
        lb2 = make_lb(ramp_up_window_s=0)
        _pause(lb2, BASE)
        out = _paused_ticks(lb2, BASE, self.QUIET, ev(None, target=8), 15)[-1][1]
        assert out.target_a == 8

    def test_resume_amps_fit_the_tightest_phase(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        grid = {1: 2.0, 2: 6.0, 3: 2.0}  # L2: 13.6 - 6 = 7.6 A -> 7 A
        out = _paused_ticks(lb, BASE, grid, ev(None, target=16), 15)[-1][1]
        assert out.target_a == 7
        assert 6.0 + out.target_a <= 16 * 0.85

    def test_picks_1_phase_when_only_the_1_phase_line_fits(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE, relief=True)
        grid = {1: 2.0, 2: 2.0, 3: 12.0}  # L3: 13.6 - 12 = 1.6 A < 6 A
        outs = _paused_ticks(lb, BASE, grid, ev(None, target=10, relief=True), 15)
        assert all(o.target_a is None for _, o in outs[:-1])
        out = outs[-1][1]
        assert out.relief_1p_requested and out.refit_from_pause
        assert out.target_a == 10  # L1 fits 11 A, capped by the 10 A plan
        assert "1-phase on L1" in out.reason

    def test_prefers_current_mode_when_it_fits(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE, relief=True)
        out = _paused_ticks(lb, BASE, self.QUIET, ev(None, relief=True), 15)[-1][1]
        assert out.target_a is not None and not out.relief_1p_requested

    def test_stays_paused_when_dwell_not_elapsed_and_current_mode_does_not_fit(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        grid = {1: 2.0, 2: 2.0, 3: 12.0}
        # relief_available=False: dwell since the last switch not elapsed
        outs = _paused_ticks(lb, BASE, grid, ev(None, relief=False), 300)
        assert all(o.target_a is None and o.state == "paused" for _, o in outs)
        # Once the dwell elapses (relief available), the 1-phase re-fit follows.
        start = outs[-1][0]
        outs = _paused_ticks(lb, start, grid, ev(None, relief=True), 15)
        assert outs[-1][1].relief_1p_requested and outs[-1][1].target_a is not None

    def test_1_phase_line_overloaded_stays_paused(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE, relief=True)
        grid = {1: 12.0, 2: 2.0, 3: 2.0}
        outs = _paused_ticks(lb, BASE, grid, ev(None, relief=True), 120)
        assert all(o.target_a is None for _, o in outs)

    def test_stale_reading_restarts_the_confirm_window(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        t = BASE + timedelta(seconds=5)
        lb.tick(t, self.QUIET, fresh(t), [ev(None)])  # fits
        t += timedelta(seconds=TICK_S)
        stale = {1: t, 2: t, 3: t - timedelta(seconds=60)}
        # Stale-sensor behaviour is unchanged (floor fallback); it breaks the
        # unbroken run of fresh fitting readings the re-fit needs.
        assert lb.tick(t, self.QUIET, stale, [ev(None)]).ev_outputs[0].state == "stale_fallback"
        outs = _paused_ticks(lb, t, self.QUIET, ev(None), 10)
        assert [o.target_a is None for _, o in outs] == [True, True]
        outs = _paused_ticks(lb, outs[-1][0], self.QUIET, ev(None), 5)
        assert outs[-1][1].target_a is not None

    def test_refit_never_exceeds_target_margin_or_fuse(self):
        """Fuse invariant: whatever the resume amps, projected phase current
        stays within the target margin (and hence below main_fuse_a)."""
        for house in (0.0, 1.5, 3.3, 5.0, 7.6):
            lb = make_lb(ramp_up_window_s=0)
            _pause(lb, BASE)
            grid = {1: house, 2: house, 3: house}
            out = _paused_ticks(lb, BASE, grid, ev(None, target=16), 15)[-1][1]
            assert out.target_a is not None
            assert house + out.target_a <= 16 * 0.85 + 1e-9

    def _refit_then_pause(self, lb, start, confirm_s):
        """Wait out the confirm window on quiet readings, re-fit, then pause
        again with a severe spike shortly after. Returns the pause time."""
        outs = _paused_ticks(lb, start, self.QUIET, ev(None), confirm_s + TICK_S)
        resumed = [t for t, o in outs if o.target_a is not None]
        assert resumed, f"no re-fit within {confirm_s + TICK_S}s"
        # The first fitting tick is start+5 s, so the re-fit lands confirm_s later.
        assert (resumed[0] - start).total_seconds() == confirm_s + TICK_S
        pause_at = resumed[0] + timedelta(seconds=60)
        _pause(lb, pause_at)
        return pause_at

    def test_backoff_grows_on_repeated_pauses_and_resets(self):
        lb = make_lb(ramp_up_window_s=0)
        _pause(lb, BASE)
        assert lb.refit_confirm_s("goe") == 10
        t = self._refit_then_pause(lb, BASE, 10)
        assert lb.refit_confirm_s("goe") == 30
        t = self._refit_then_pause(lb, t, 30)
        assert lb.refit_confirm_s("goe") == 120
        t = self._refit_then_pause(lb, t, 120)
        assert lb.refit_confirm_s("goe") == 120  # capped at the last step
        # Re-fit, then charge stably for longer than the flap window.
        outs = _paused_ticks(lb, t, self.QUIET, ev(None), 125)
        resumed_at = next(tt for tt, o in outs if o.target_a is not None)
        pause_at = resumed_at + timedelta(seconds=601)
        _pause(lb, pause_at)
        assert lb.refit_confirm_s("goe") == 10
        # The 10 s window is effective again.
        outs = _paused_ticks(lb, pause_at, self.QUIET, ev(None), 15)
        assert outs[-1][1].target_a is not None

    def test_two_paused_chargers_refit_in_reverse_give_way_order(self):
        """The lower-listed charger re-fits first; the higher-listed one
        waits (timers kept) and then re-fits into what is left, never
        spending the room the first one took."""

        def charger(cid, setpoint, target):
            e = ev(setpoint, target=target)
            e.charger_id = cid
            # The sensors report what the car draws, so a resume is measured
            # (and leaves the pending-increase ledger) on the next tick.
            e.measured_a = float(setpoint) if setpoint is not None else None
            return e

        lb = make_lb(ramp_up_window_s=0)
        severe = {1: 10.0, 2: 10.0, 3: 30.0}
        out = lb.tick(BASE, severe, fresh(BASE), [charger("a", 6, 6), charger("b", 6, 6)])
        assert [o.state for o in out.ev_outputs] == ["paused", "throttling"]
        # Seed the second pause after its own fresh overload sample; one
        # actuator may cut per phase sample.
        later = BASE + timedelta(seconds=15)
        out = lb.tick(later, severe, fresh(later), [charger("a", None, 6), charger("b", 6, 6)])
        assert [o.state for o in out.ev_outputs] == ["paused", "paused"]

        a_set = b_set = None
        house = 0.0
        resumed = {}
        for i in range(1, 7):
            t = BASE + timedelta(seconds=15 + i * TICK_S)
            own = (a_set or 0) + (b_set or 0)
            grid = dict.fromkeys((1, 2, 3), house + own)
            out = lb.tick(t, grid, fresh(t), [charger("a", a_set, 6), charger("b", b_set, 6)])
            a_out, b_out = out.ev_outputs
            if a_set is None and a_out.target_a is not None:
                resumed["a"] = i
            if b_set is None and b_out.target_a is not None:
                resumed["b"] = i
            a_set, b_set = a_out.target_a, b_out.target_a
            # Fuse invariant across both chargers.
            assert house + (a_set or 0) + (b_set or 0) <= 16 * 0.85 + 1e-9
        assert resumed["b"] <= resumed["a"]  # both may resume if the shared budget fits
        assert a_set == 6 and b_set == 6

    def test_backoff_never_shorter_than_configured_confirm(self):
        lb = make_lb(resume_confirm_s=60, ramp_up_window_s=0)
        _pause(lb, BASE)
        assert lb.refit_confirm_s("goe") == 60
        self._refit_then_pause(lb, BASE, 60)
        assert lb.refit_confirm_s("goe") == 60  # max(60, 30)
        lb._ev_refit_level["goe"] = 2
        assert lb.refit_confirm_s("goe") == 120
