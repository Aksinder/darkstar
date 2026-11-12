"""Price rules of window-mode loads: cheap early start and peak off
(executor/window_price.py).

One pure decision function serves the executor (live price) and the planner
(forecast prices), so everything about the rules is pinned down here, without any
engine: thresholds, the early-start latch, the peak-off max/min-on bounds, the
unknown-price fail direction, and both rules together.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytz

from executor.window_price import (
    PHASE_EARLY,
    PHASE_IDLE,
    PHASE_PEAK_OFF,
    PHASE_WINDOW,
    PriceRules,
    RuleState,
    rolling_price_window,
    rule_state_from_attributes,
    simulate,
    step,
    thresholds,
)

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
WINDOW_START = T0 + timedelta(hours=24)

OWNER = PriceRules(
    early_start_hours=14,
    early_start_price_percentile=30,
    peak_off_price_percentile=85,
    peak_off_max_minutes=180,
    peak_off_min_on_minutes=120,
)
EARLY_ONLY = PriceRules(early_start_hours=14)
PEAK_ONLY = PriceRules(peak_off_price_percentile=85)
EARLY_THR, PEAK_THR = 0.80, 2.50


def _h(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def _step(rules, state, hours, *, price, in_window=False, next_on=WINDOW_START, **kw):
    return step(
        rules, state, now=_h(hours), in_window=in_window, next_on=next_on, price=price,
        early_threshold=EARLY_THR if rules.early_enabled else None,
        peak_threshold=PEAK_THR if rules.peak_enabled else None,
        **kw,
    )


class TestConfig:
    def test_everything_is_off_by_default(self):
        rules = PriceRules.from_mapping({"id": "kim_heater", "mode": "window"})
        assert rules == PriceRules()
        assert not rules.enabled and not rules.early_enabled and not rules.peak_enabled
        assert (rules.early_start_hours, rules.peak_off_price_percentile) == (0.0, None)
        assert (rules.early_start_price_percentile, rules.price_window_hours) == (30.0, 24.0)
        assert (rules.peak_off_max_minutes, rules.peak_off_min_on_minutes) == (180.0, 120.0)
        assert rules.peak_off_temp_margin_c == 1.0

    def test_the_owners_values(self):
        rules = PriceRules.from_mapping({
            "early_start_hours": 14, "early_start_price_percentile": 30,
            "peak_off_price_percentile": 85, "peak_off_max_minutes": 180,
            "peak_off_min_on_minutes": 120,
        })
        assert rules == OWNER and rules.enabled

    def test_blank_and_bad_values_fall_back_instead_of_raising(self):
        rules = PriceRules.from_mapping({
            "early_start_hours": "", "early_start_price_percentile": "cheap",
            "peak_off_price_percentile": "", "peak_off_max_minutes": -5,
            "price_window_hours": 0,
        })
        assert rules.early_start_hours == 0.0 and rules.early_start_price_percentile == 30.0
        assert rules.peak_off_price_percentile is None
        assert rules.peak_off_max_minutes == 0.0 and rules.price_window_hours == 24.0

    def test_a_zero_percentile_is_a_setting_not_unset(self):
        assert PriceRules.from_mapping({"peak_off_price_percentile": 0}).peak_enabled


class TestThresholds:
    PRICES = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)

    def test_percentiles_of_the_window(self):
        early, peak = thresholds(OWNER, self.PRICES)
        assert early == pytest.approx(0.8)  # 30th percentile
        assert peak == pytest.approx(1.9)  # 85th

    def test_a_rule_that_is_off_has_no_threshold(self):
        assert thresholds(PriceRules(), self.PRICES) == (None, None)
        assert thresholds(EARLY_ONLY, self.PRICES)[1] is None
        assert thresholds(PEAK_ONLY, self.PRICES)[0] is None

    def test_no_prices_no_thresholds(self):
        assert thresholds(OWNER, []) == (None, None)
        assert thresholds(OWNER, [None, None]) == (None, None)


class TestEarlyStart:
    def test_starts_on_a_cheap_price_inside_the_span_only(self):
        # 15 h before: too early, however cheap.
        state, v = _step(EARLY_ONLY, RuleState(), 9, price=0.10)
        assert (v.heat, v.phase, state.early_latched) == (False, PHASE_IDLE, False)
        # 14 h before, dear: no.
        state, v = _step(EARLY_ONLY, state, 10, price=0.81)
        assert not v.heat
        # 13 h before, at the threshold: yes.
        state, v = _step(EARLY_ONLY, state, 11, price=0.80)
        assert (v.heat, v.phase, state.early_latched) == (True, PHASE_EARLY, True)

    def test_latched_it_stays_on_when_the_price_rises_again(self):
        state, _ = _step(EARLY_ONLY, RuleState(), 11, price=0.50)
        for hours, price in ((12, 0.9), (16, 3.5), (23.9, 9.9)):
            state, v = _step(EARLY_ONLY, state, hours, price=price)
            assert (v.heat, v.phase) == (True, PHASE_EARLY)

    def test_the_latch_hands_over_to_the_window_without_a_gap(self):
        state, _ = _step(EARLY_ONLY, RuleState(), 11, price=0.50)
        since = state.heat_since
        state, v = _step(EARLY_ONLY, state, 24, price=3.0, in_window=True, next_on=None)
        assert (v.heat, v.phase, state.early_latched) == (True, PHASE_WINDOW, False)
        assert state.heat_since == since  # one uninterrupted heat period

    def test_the_latch_is_dropped_when_the_window_goes_away_or_moves_out_of_reach(self):
        state, _ = _step(EARLY_ONLY, RuleState(), 11, price=0.50)
        _, v = _step(EARLY_ONLY, state, 12, price=3.0, next_on=None)  # event deleted
        assert not v.heat
        _, v = _step(EARLY_ONLY, state, 12, price=3.0, next_on=_h(40))  # moved a day
        assert not v.heat

    def test_an_unknown_price_never_starts_early(self):
        state, v = _step(EARLY_ONLY, RuleState(), 11, price=None)
        assert not v.heat and not state.early_latched

    def test_an_unknown_price_does_not_drop_a_latch(self):
        state, _ = _step(EARLY_ONLY, RuleState(), 11, price=0.50)
        state, v = _step(EARLY_ONLY, state, 12, price=None)
        assert v.heat and state.early_latched

    def test_no_threshold_no_early_start(self):
        _, v = step(EARLY_ONLY, RuleState(), now=_h(11), in_window=False,
                    next_on=WINDOW_START, price=0.1, early_threshold=None,
                    peak_threshold=None)
        assert not v.heat

    def test_without_a_known_start_time_it_is_a_no_op(self):
        """Non-calendar gates have no future: nothing to start early for."""
        _, v = _step(EARLY_ONLY, RuleState(), 11, price=0.01, next_on=None)
        assert not v.heat

    def test_off_means_off(self):
        _, v = _step(PriceRules(), RuleState(), 23, price=0.0)
        assert not v.heat


def _run(rules, steps, state=None):
    """[(hours, price)] inside a window -> [(heat, phase)], carrying the state."""
    state = state or RuleState()
    out = []
    for hours, price in steps:
        state, v = _step(rules, state, hours, price=price, in_window=True, next_on=None)
        out.append((v.heat, v.phase))
    return out, state


ON = (True, PHASE_WINDOW)
OFF = (False, PHASE_PEAK_OFF)


class TestPeakOff:
    def test_off_at_or_above_the_threshold_on_below_it(self):
        out, _ = _run(PEAK_ONLY, [(0, 2.49), (2, 2.50), (3, 9.0), (4, 2.49)])
        assert out == [ON, OFF, OFF, ON]

    def test_one_stretch_lasts_at_most_the_maximum(self):
        """Without a temperature sensor, time is the only guard against a cold room."""
        out, state = _run(PEAK_ONLY, [(0, 1.0), (2, 9.0), (4.98, 9.0), (5, 9.0), (5.5, 9.0)])
        assert out == [ON, OFF, OFF, ON, ON]
        assert state.heat_since == _h(5) and state.peak_off_since is None

    def test_after_a_stretch_it_must_run_the_minimum_before_the_next(self):
        steps = [(0, 1.0), (2, 9.0), (5, 9.0), (6.98, 9.0), (7, 9.0), (9.99, 9.0), (10, 9.0)]
        out, _ = _run(PEAK_ONLY, steps)
        assert out == [ON, OFF, ON, ON, OFF, OFF, ON]

    def test_min_on_also_applies_after_a_short_stretch(self):
        out, _ = _run(PEAK_ONLY, [(0, 1.0), (2, 9.0), (2.5, 1.0), (3, 9.0), (4.5, 9.0)])
        assert out == [ON, OFF, ON, ON, OFF]

    def test_a_heater_that_just_came_on_is_not_switched_off_at_once(self):
        out, _ = _run(PEAK_ONLY, [(0, 1.0), (0.5, 9.0), (1.99, 9.0), (2, 9.0)])
        assert out == [ON, ON, ON, OFF]

    def test_a_window_that_begins_at_a_peak_waits_it_out(self):
        out, _ = _run(PEAK_ONLY, [(0, 9.0), (2.9, 9.0), (3, 9.0)])
        assert out == [OFF, OFF, ON]

    def test_an_unknown_price_never_switches_off_and_ends_a_running_peak_off(self):
        out, _ = _run(PEAK_ONLY, [(0, 1.0), (3, None), (4, 9.0), (5, None), (6, 9.0)])
        assert out == [ON, ON, OFF, ON, ON]

    def test_no_threshold_no_peak_off(self):
        _, v = step(PEAK_ONLY, RuleState(), now=_h(0), in_window=True, next_on=None,
                    price=99.0, early_threshold=None, peak_threshold=None)
        assert v.heat

    def test_the_thermostat_guard_forbids_and_ends_a_peak_off(self):
        state, v = _step(PEAK_ONLY, RuleState(), 0, price=9.0, in_window=True,
                         next_on=None, keep_warm=True)
        assert v.heat and state.peak_off_since is None
        state, v = _step(PEAK_ONLY, RuleState(), 0, price=9.0, in_window=True, next_on=None)
        assert not v.heat
        state, v = _step(PEAK_ONLY, state, 1, price=9.0, in_window=True, next_on=None,
                         keep_warm=True)
        assert v.heat and "too cold" in v.reason and state.heat_since == _h(1)

    def test_zero_max_minutes_disables_it(self):
        rules = PriceRules(peak_off_price_percentile=85, peak_off_max_minutes=0)
        out, _ = _run(rules, [(0, 9.0), (5, 9.0)])
        assert out == [ON, ON]

    def test_outside_any_heat_period_there_is_nothing_to_switch_off(self):
        state, v = _step(PEAK_ONLY, RuleState(), 0, price=9.0, in_window=False)
        assert (v.heat, v.phase, state) == (False, PHASE_IDLE, RuleState())

    def test_leaving_the_window_forgets_the_clocks(self):
        _, state = _run(PEAK_ONLY, [(0, 1.0), (2, 9.0)])
        state, _ = _step(PEAK_ONLY, state, 3, price=9.0, in_window=False, next_on=None)
        assert state == RuleState()


class TestBothRules:
    def test_a_peak_interrupts_an_early_start_and_the_latch_survives_it(self):
        state, v = _step(OWNER, RuleState(), 11, price=0.50)
        assert v.phase == PHASE_EARLY
        state, v = _step(OWNER, state, 12, price=3.0)  # on < 120 min: stays on
        assert (v.heat, v.phase) == (True, PHASE_EARLY)
        state, v = _step(OWNER, state, 13, price=3.0)  # 120 min on: peak-off
        assert (v.heat, v.phase) == (False, PHASE_PEAK_OFF) and state.early_latched
        state, v = _step(OWNER, state, 14, price=1.5)  # peak over, dearer than "cheap"
        assert (v.heat, v.phase) == (True, PHASE_EARLY)  # latched: back on anyway

    def test_a_price_plateau_is_neither_cheap_nor_a_peak(self):
        """When most of the window costs the same, that price is the 30th AND the
        85th percentile: the heater would start early and be peak-off 3 h in every
        5, for no saving at all. Cheap must be below the median, a peak above it."""
        assert thresholds(OWNER, [1.0] * 24) == (None, None)
        # One dearer hour: it alone is a peak; nothing is cheap.
        assert thresholds(OWNER, [1.0] * 23 + [1.4]) == (None, 1.4)
        # One cheaper hour: it alone is cheap; nothing is a peak.
        assert thresholds(OWNER, [0.2] + [1.0] * 23) == (0.2, None)

    def test_a_peak_off_runs_on_from_early_start_into_the_window(self):
        state, _ = _step(OWNER, RuleState(), 20, price=0.5)
        state, v = _step(OWNER, state, 23, price=3.0)
        assert v.phase == PHASE_PEAK_OFF
        state, v = _step(OWNER, state, 24.5, price=3.0, in_window=True, next_on=None)
        assert v.phase == PHASE_PEAK_OFF  # 90 min in
        state, v = _step(OWNER, state, 26, price=3.0, in_window=True, next_on=None)
        assert (v.heat, v.phase) == (True, PHASE_WINDOW)  # 180 min reached


class TestDst:
    def test_clocks_count_real_time_across_the_autumn_change(self):
        """25 October 2026, Stockholm: 02:00-03:00 happens twice. A peak-off begun at
        01:30 CEST has lasted 3 real hours at 03:30 CET, not 2."""
        tz = pytz.timezone("Europe/Stockholm")
        began = tz.localize(datetime(2026, 10, 25, 1, 30), is_dst=True)
        kw = {"in_window": True, "next_on": None, "price": 9.0, "early_threshold": None,
                  "peak_threshold": PEAK_THR}
        state, v = step(PEAK_ONLY, RuleState(), now=began, **kw)
        assert v.phase == PHASE_PEAK_OFF
        second_230 = tz.localize(datetime(2026, 10, 25, 2, 30), is_dst=False)
        state, v = step(PEAK_ONLY, state, now=second_230, **kw)
        assert v.phase == PHASE_PEAK_OFF  # 2 real hours
        state, v = step(PEAK_ONLY, state, now=tz.localize(datetime(2026, 10, 25, 3, 30)), **kw)
        assert v.heat  # 3 real hours: the maximum

    def test_the_early_span_is_real_hours_before_the_window(self):
        tz = pytz.timezone("Europe/Stockholm")
        start = tz.localize(datetime(2026, 10, 25, 15, 0))  # CET, after the change
        kw = {"in_window": False, "next_on": start, "price": 0.1, "early_threshold": EARLY_THR,
                  "peak_threshold": None}
        # 14 real hours before 15:00 CET is 02:00 CEST (the FIRST 02:00), not 01:00.
        too_early = tz.localize(datetime(2026, 10, 25, 1, 59), is_dst=True)
        assert not step(EARLY_ONLY, RuleState(), now=too_early, **kw)[1].heat
        in_span = tz.localize(datetime(2026, 10, 25, 2, 0), is_dst=True)
        assert start - in_span == timedelta(hours=14)
        assert step(EARLY_ONLY, RuleState(), now=in_span, **kw)[1].heat


def _series(prices, start=T0, minutes=60):
    return [start + timedelta(minutes=minutes * i) for i in range(len(prices))]


class TestRollingWindow:
    def test_forward_first(self):
        prices = [float(i) for i in range(48)]
        assert rolling_price_window(_series(prices), prices, 10, 24) == prices[10:34]

    def test_backfilled_from_the_past_when_the_series_runs_out(self):
        """Like the executor before tomorrow's prices exist: what the future cannot
        supply comes from the hours already passed."""
        prices = [float(i) for i in range(24)]
        got = rolling_price_window(_series(prices), prices, 20, 24)
        assert got == prices[-1 - 23:-4] + prices[20:]  # 20 past hours + 4 ahead
        assert len(got) == 24

    def test_slot_zero_has_no_past(self):
        prices = [1.0, 2.0, 3.0]
        assert rolling_price_window(_series(prices), prices, 0, 24) == prices

    def test_unknown_prices_are_left_out(self):
        prices = [1.0, None, 3.0]
        assert rolling_price_window(_series(prices), prices, 0, 24) == [1.0, 3.0]


class TestSimulate:
    def test_carries_latch_and_clocks_from_slot_to_slot(self):
        # Hourly slots from T0; window from hour 12. Valley at 2-4, peak at 6-8 and 13-17.
        prices = [2.0, 2.0, 0.3, 0.3, 0.3, 1.5, 3.5, 3.5, 3.5, 1.5, 1.5, 1.5,
                  1.5, 3.5, 3.5, 3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        starts = _series(prices)
        in_window = [i >= 12 for i in range(24)]
        next_on = [starts[12] if i < 12 else None for i in range(24)]
        verdicts = simulate(OWNER, RuleState(), starts=starts, prices=prices,
                            in_window=in_window, next_on=next_on)
        phases = [v.phase for v in verdicts]
        assert phases[:2] == [PHASE_IDLE] * 2  # dear: not yet
        assert phases[2:6] == [PHASE_EARLY] * 4  # valley latches; stays on at 1.5
        assert phases[6:9] == [PHASE_PEAK_OFF] * 3  # 4 h on, then the morning peak
        assert phases[9:12] == [PHASE_EARLY] * 3  # latch survived the peak
        assert phases[12] == PHASE_WINDOW
        assert phases[13:16] == [PHASE_PEAK_OFF] * 3  # 180 min at most...
        assert phases[16:18] == [PHASE_WINDOW] * 2  # ...then on, although still dear
        assert phases[18:] == [PHASE_WINDOW] * 6
        assert [v.heat for v in verdicts] == [p in (PHASE_EARLY, PHASE_WINDOW) for p in phases]

    def test_without_rules_it_is_the_window(self):
        prices = [9.0, 0.1, 9.0, 0.1]
        verdicts = simulate(PriceRules(), RuleState(), starts=_series(prices), prices=prices,
                            in_window=[False, True, True, False],
                            next_on=[T0 + timedelta(hours=1), None, None, None])
        assert [v.heat for v in verdicts] == [False, True, True, False]


class TestPublishedState:
    def test_round_trip(self):
        attrs = {"early_start_latched": True, "peak_off_since": "2026-11-13T07:00+01:00",
                 "heat_since": None}
        assert rule_state_from_attributes(attrs) == RuleState(
            early_latched=True,
            peak_off_since=datetime(2026, 11, 13, 6, 0, tzinfo=UTC),
        )

    @pytest.mark.parametrize("attrs", [None, {}, {"reason": "x"}])
    def test_nothing_published_is_no_state(self, attrs):
        assert rule_state_from_attributes(attrs) is None

    def test_damaged_or_naive_timestamps_are_dropped(self):
        attrs = {"early_start_latched": "yes", "peak_off_since": "garbage",
                 "heat_since": "2026-11-13T07:00:00"}
        assert rule_state_from_attributes(attrs) == RuleState()
