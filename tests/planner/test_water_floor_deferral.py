"""The water-floor cliff, and the opt-in deferral that removes it.

What is being fixed. The daily floor is a kWh total per day-bucket, priced at 15 SEK
per missing kWh, and water is decided per wall-clock hour. Near the end of a bucket
there is nowhere cheap left to put a small residual, and the hour in progress is the
one place where LESS than an hour can be bought — so the rest of this hour is the
cheapest total on offer whatever it costs per kWh.

``floor_deferral`` lets that residual roll to the next bucket, but only for a tank
MEASURED full recently that still holds a shower reserve at the boundary. It defaults
to off, and the tests below pin that too.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from planner.hot_water import HotWaterEstimator
from planner.pipeline import _apply_water_shortfall_gate
from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot, WaterHeaterInput
from planner.thermal import WaterTankModel
from planner.water_floor_deferral import (
    DeferralConfig,
    bucket_boundary,
    decide_floor_deferral,
)

TANK = WaterTankModel(volume_litres=195.0, t_cold_c=10.0, t_max_c=75.0, ua_w_per_k=2.5)
CAPACITY = TANK.capacity_kwh()  # 14.74 kWh
POWER = 3.4


def _slots(start: datetime, hours: float, price) -> list[KeplerInputSlot]:
    """15-minute slots from ``start``; ``price`` maps a datetime to SEK/kWh."""
    n = round(hours * 4)
    return [
        KeplerInputSlot(
            start_time=(t := start + timedelta(minutes=15 * i)),
            end_time=t + timedelta(minutes=15),
            load_kwh=0.1,
            pv_kwh=0.0,
            import_price_sek_kwh=price(t),
            export_price_sek_kwh=max(0.0, price(t) - 1.0),
        )
        for i in range(n)
    ]


def _autumn(t: datetime) -> float:
    """Night trough, morning and evening peaks, cheap early afternoon.

    The shoulder between the morning peak and the cheap band matters: with the cheap
    band starting AT the bucket boundary a residual is bought as one extra hour glued
    onto the next bucket's block (one start penalty instead of two), which hides the
    stub. Real curves have the shoulder (2026-10-01: 2.19 at 09, 1.86 at 10, 1.76 at 13).
    """
    h = t.hour
    if h < 6:
        return 1.1
    if h < 10:
        return 2.5
    if h < 12:
        return 1.6
    if h < 16:
        return 1.0
    if h < 21:
        return 2.8
    return 1.4


# =====================================================================================
# 1. The decision, as a pure function
# =====================================================================================

NOW = datetime(2026, 10, 1, 6, 15)
ON = DeferralConfig(enabled=True)


def _loss(stored: float, hours: float) -> float:
    temp = TANK.t_cold_c + stored * 1000.0 / TANK.heat_capacity_wh_per_k
    return TANK.standby_loss_kwh(temp, hours)


def _decide(**over):
    args = {
        "cfg": ON,
        "slots": _slots(NOW, 17.75, _autumn),
        "now_local": NOW,
        "defer_hours": 10.0,
        "heater_kw": POWER,
        "residual_kwh": 0.6,
        "stored_kwh": CAPACITY - 0.6,
        "capacity_kwh": CAPACITY,
        "volume_litres": 195.0,
        "learned_draw_kw": 0.25,
        "min_since_saturated": 120.0,
        "state_age_min": 3.0,
        "standby_loss_kwh_fn": _loss,
    }
    args.update(over)
    return decide_floor_deferral(**args)


class TestTheDecision:
    def test_the_case_it_exists_for(self):
        """06:15, bucket ends 10:00, only the morning peak left, tank measured full two
        hours ago and 0.6 kWh short of full."""
        d = _decide()
        assert d.defer, d.reason

    def test_off_by_default(self):
        assert not DeferralConfig().enabled
        assert not DeferralConfig.from_heater({"id": "main_tank"}).enabled
        assert not _decide(cfg=DeferralConfig()).defer

    def test_nothing_owed_is_not_a_deferral(self):
        assert not _decide(residual_kwh=0.0).defer

    # -- evidence ---------------------------------------------------------------------

    def test_a_tank_never_measured_full_is_not_deferred(self):
        """stored_kwh says full after ANY heating run, including one the plan cut short.
        Without the thermostat's own word the model's is not enough."""
        assert not _decide(min_since_saturated=None).defer
        assert not _decide(min_since_saturated=-1.0).defer

    def test_an_old_measurement_is_not_evidence(self):
        assert _decide(min_since_saturated=8 * 60.0).defer
        assert not _decide(min_since_saturated=8 * 60.0 + 1).defer

    def test_a_dead_publisher_is_not_a_full_tank(self):
        """The state file has no timestamp; a frozen 'full 2 h ago' must not last."""
        assert not _decide(state_age_min=None).defer
        assert not _decide(state_age_min=45.0).defer

    @pytest.mark.parametrize(
        "missing",
        [
            {"stored_kwh": None},
            {"capacity_kwh": None},
            {"capacity_kwh": 0.0},
            {"volume_litres": None},
            {"learned_draw_kw": None},
            {"learned_draw_kw": 0.0},
            {"slots": []},
        ],
    )
    def test_any_missing_input_keeps_the_floor(self, missing):
        assert not _decide(**missing).defer

    def test_a_failing_loss_model_keeps_the_floor(self):
        def boom(_s, _h):
            raise RuntimeError("no tank model")

        assert not _decide(standby_loss_kwh_fn=boom).defer

    # -- comfort ----------------------------------------------------------------------

    def test_a_tank_without_the_shower_reserve_is_heated_at_any_price(self):
        """Two showers above 40 C must remain at the boundary. 195 L at 40 C holds 6.80
        kWh, two showers are 4.19: below ~11 kWh projected, the floor stands."""
        d = _decide(stored_kwh=11.5)  # 3.75 h of draw + loss takes it under
        assert not d.defer
        assert "reserve" in d.reason

    def test_the_reserve_is_the_owners_dial(self):
        stored = 11.5
        assert not _decide(stored_kwh=stored).defer
        relaxed = DeferralConfig(enabled=True, reserve_showers=1.0)
        assert _decide(stored_kwh=stored, cfg=relaxed).defer

    def test_a_high_learned_draw_shrinks_what_is_left(self):
        assert _decide(stored_kwh=12.5).defer
        assert not _decide(stored_kwh=12.5, learned_draw_kw=0.6).defer

    # -- window and price -------------------------------------------------------------

    def test_only_the_tail_of_a_bucket_qualifies(self):
        """At 20:00 the bucket has 14 hours left and the night trough is in it."""
        evening = datetime(2026, 10, 1, 20, 0)
        d = _decide(now_local=evening, slots=_slots(evening, 28, _autumn))
        assert not d.defer
        assert "bucket ends in" in d.reason

    def test_a_cheap_slot_before_the_boundary_is_used_not_skipped(self):
        """05:00 with the night trough still running: the solver should heat there."""
        early = datetime(2026, 10, 1, 5, 0)
        d = _decide(now_local=early, slots=_slots(early, 19, _autumn))
        assert not d.defer
        assert "cheap enough" in d.reason

    def test_flat_prices_defer_nothing(self):
        assert not _decide(slots=_slots(NOW, 17.75, lambda _t: 1.5)).defer

    def test_no_view_of_the_next_bucket_defers_nothing(self):
        """A horizon that stops at the boundary has nothing to roll the residual to."""
        assert not _decide(slots=_slots(NOW, 3.75, _autumn)).defer

    def test_forecast_pv_before_the_boundary_counts_as_cheap(self):
        slots = _slots(NOW, 17.75, _autumn)
        for s in slots:
            if s.start_time.hour == 9:
                s.pv_kwh = 2.0  # 8 kW of surplus: heating there costs the export price
                s.export_price_sek_kwh = 0.3
        d = _decide(slots=slots)
        assert not d.defer
        assert "cheap enough" in d.reason

    def test_naive_and_aware_times_are_refused_not_guessed(self):
        import pytz

        aware = pytz.timezone("Europe/Stockholm").localize(NOW)
        assert not _decide(now_local=aware).defer


class TestBucketBoundary:
    """Must be kepler's bucketing exactly: hour < defer_hours belongs to yesterday."""

    def test_morning_belongs_to_the_bucket_ending_today(self):
        assert bucket_boundary(datetime(2026, 10, 1, 6, 15), 10) == datetime(2026, 10, 1, 10)

    def test_from_the_boundary_on_it_is_tomorrows(self):
        assert bucket_boundary(datetime(2026, 10, 1, 10, 0), 10) == datetime(2026, 10, 2, 10)
        assert bucket_boundary(datetime(2026, 10, 1, 23, 45), 10) == datetime(2026, 10, 2, 10)

    def test_no_deferral_hours_means_midnight(self):
        assert bucket_boundary(datetime(2026, 10, 1, 6, 15), 0) == datetime(2026, 10, 2, 0)

    def test_it_agrees_with_the_solvers_buckets(self):
        """Every slot before the boundary shares slot 0's bucket; the first one at or
        after it does not."""
        for hh in (0, 5, 9, 10, 13, 23):
            now = datetime(2026, 10, 1, hh, 15)
            b = bucket_boundary(now, 10)

            def bucket(t: datetime):
                return t.date() - timedelta(days=1) if t.hour < 10 else t.date()

            t = now
            while t < b:
                assert bucket(t) == bucket(now)
                t += timedelta(minutes=15)
            assert bucket(b) != bucket(now)


# =====================================================================================
# 2. Wired into the pipeline's floor gate
# =====================================================================================

TANK_CFG = {
    "id": "main_tank",
    "volume_litres": 195,
    "t_cold_c": 10,
    "t_max_c": 75,
    "ua_w_per_k": 2.5,
    "shortfall_gate": {"enabled": False},
    "floor_deferral": {"enabled": True},
}
HEATED = 3.24  # the live log line's own figure


def _gate(state, cfg=None, heated=HEATED, age=3.0, slots=None, now=NOW):
    h = WaterHeaterInput(
        id="main_tank",
        power_kw=POWER,
        min_kwh_per_day=6.0,
        max_hours_between_heating=6.0,
        min_spacing_hours=4.0,
        heated_today_kwh=heated,
    )
    with (
        patch("planner.pipeline._load_hot_water_states", return_value={"main_tank": state}),
        patch("planner.pipeline._hot_water_state_age_min", return_value=age),
    ):
        _apply_water_shortfall_gate(
            [h],
            [cfg or TANK_CFG],
            slots if slots is not None else _slots(now, 17.75, _autumn),
            now,
            {"main_tank": heated},
            defer_hours=10.0,
        )
    return h


def _state(headroom=0.33, sat_age=120.0, **over):
    s = {
        "stored_kwh": CAPACITY - headroom,
        "saturated_min": 0.0,
        "learned_draw_kw": 0.25,
        "min_since_saturated": sat_age,
    }
    s.update(over)
    return s


class TestInThePipeline:
    def test_the_live_log_line(self):
        """'CAPPED 6.00 -> 3.57 kWh gross (0.33 headroom + 3.24 already heated today)':
        bucket 0 now owes nothing, and nothing else moved."""
        h = _gate(_state())
        assert h.min_kwh_per_day == pytest.approx(3.57, abs=0.01)  # the cap, untouched
        assert h.first_bucket_min_kwh == pytest.approx(HEATED)
        assert max(0.0, h.first_bucket_min_kwh - HEATED) == 0.0  # what kepler will book

    def test_off_by_default_leaves_the_field_alone(self):
        cfg = {k: v for k, v in TANK_CFG.items() if k != "floor_deferral"}
        h = _gate(_state(), cfg=cfg)
        assert h.first_bucket_min_kwh is None
        assert h.min_kwh_per_day == pytest.approx(3.57, abs=0.01)

    def test_a_state_file_from_before_this_change_defers_nothing(self):
        """Old publishers do not write min_since_saturated. Missing means unknown."""
        s = _state()
        del s["min_since_saturated"]
        assert _gate(s).first_bucket_min_kwh is None

    def test_a_stale_state_file_defers_nothing(self):
        assert _gate(_state(), age=None).first_bucket_min_kwh is None
        assert _gate(_state(), age=90.0).first_bucket_min_kwh is None

    def test_a_drained_tank_keeps_its_floor(self):
        """9 kWh of headroom is a real need, however recently it was full."""
        h = _gate(_state(headroom=9.0))
        assert h.first_bucket_min_kwh is None
        assert h.min_kwh_per_day == 6.0

    def test_it_never_touches_the_daily_floor_itself(self):
        """Later buckets read min_kwh_per_day. Deferring must not lower them."""
        with_d = _gate(_state())
        cfg = {k: v for k, v in TANK_CFG.items() if k != "floor_deferral"}
        without = _gate(_state(), cfg=cfg)
        assert with_d.min_kwh_per_day == without.min_kwh_per_day

    def test_a_struck_floor_is_left_to_the_strike(self):
        h = _gate(_state(saturated_min=15.0))
        assert h.min_kwh_per_day == 0.0
        assert h.first_bucket_min_kwh is None


# =====================================================================================
# 3. The solver
# =====================================================================================


def _kcfg(heater: WaterHeaterInput, **over) -> KeplerConfig:
    base = {
        "capacity_kwh": 10.0,
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        # Dear enough that the battery sits still: these tests are about water.
        "wear_cost_sek_per_kwh": 5.0,
        "water_heaters": [heater],
        "water_reliability_penalty_sek": 15.0,
        "water_block_start_penalty_sek": 3.0,
        "defer_up_to_hours": 10.0,
        "water_hourly_blocks": True,
    }
    base.update(over)
    return KeplerConfig(**base)


def _heater(min_kwh=6.0, heated=0.0, **over) -> WaterHeaterInput:
    return WaterHeaterInput(
        id="main_tank",
        power_kw=POWER,
        min_kwh_per_day=min_kwh,
        max_hours_between_heating=6.0,
        min_spacing_hours=0.0,
        heated_today_kwh=heated,
        **over,
    )


def _solve(slots, cfg):
    return KeplerSolver().solve(KeplerInput(slots=slots, initial_soc_kwh=5.0), cfg)


def _on(result) -> list[datetime]:
    return [s.start_time for s in result.slots if s.water_heater_results.get("main_tank", 0) > 0]


def _kwh(result, before: datetime | None = None, since: datetime | None = None) -> float:
    return sum(
        POWER * 0.25
        for t in _on(result)
        if (before is None or t < before) and (since is None or t >= since)
    )


BOUNDARY = datetime(2026, 10, 1, 10, 0)
MORNING = _slots(NOW, 17.75, _autumn)  # 06:15 -> 24:00, day-ahead prices not in yet


class TestTheCliffItself:
    def test_reproduced_a_small_residual_buys_the_rest_of_the_peak_hour(self):
        """0.5 kWh owed books nothing; 0.7 kWh books 2.55 kWh at 06:15 — in the morning
        peak, because three quarters of an hour is the smallest thing on sale."""
        small = _solve(MORNING, _kcfg(_heater(heated=5.5)))
        assert _kwh(small, before=BOUNDARY) == 0.0

        big = _solve(MORNING, _kcfg(_heater(heated=5.3)))
        on = [t for t in _on(big) if t < BOUNDARY]
        assert on[0] == NOW
        assert _kwh(big, before=BOUNDARY) == pytest.approx(2.55)


class TestFirstBucketFloor:
    def test_a_deferred_bucket_books_nothing_before_the_boundary(self):
        h = _heater(heated=5.3, first_bucket_min_kwh=5.3)
        assert _kwh(_solve(MORNING, _kcfg(h)), before=BOUNDARY) == 0.0

    def test_the_next_bucket_still_gets_its_whole_floor(self):
        """The residual is rolled, not forgiven: the bucket after the boundary is
        untouched and heats in its cheap hours."""
        h = _heater(heated=5.3, first_bucket_min_kwh=5.3)
        r = _solve(MORNING, _kcfg(h))
        after = [t for t in _on(r) if t >= BOUNDARY]
        assert _kwh(r, since=BOUNDARY) >= 6.0
        assert all(12 <= t.hour < 16 for t in after), "the cheap afternoon band"

    def test_it_can_only_lower(self):
        """A value above the daily floor must not raise bucket 0."""
        base = _solve(MORNING, _kcfg(_heater(heated=5.3)))
        raised = _solve(MORNING, _kcfg(_heater(heated=5.3, first_bucket_min_kwh=50.0)))
        assert _kwh(raised, before=BOUNDARY) == _kwh(base, before=BOUNDARY)

    def test_a_negative_value_is_zero_not_a_credit(self):
        r = _solve(MORNING, _kcfg(_heater(heated=0.0, first_bucket_min_kwh=-5.0)))
        assert _kwh(r, before=BOUNDARY) == 0.0

    def test_none_is_byte_for_byte_the_old_model(self):
        a = _solve(MORNING, _kcfg(_heater(heated=5.3)))
        b = _solve(MORNING, _kcfg(_heater(heated=5.3, first_bucket_min_kwh=None)))
        assert _on(a) == _on(b)
        assert a.total_cost_sek == pytest.approx(b.total_cost_sek)

    def test_deferring_never_makes_heating_pay(self):
        """The water-boost phantom came from a reward. This adds none: with the floor
        deferred AND every later floor removed there is nothing to gain from heating,
        so the solver heats nothing at all."""
        h = _heater(min_kwh=0.0, heated=0.0, first_bucket_min_kwh=0.0)
        r = _solve(MORNING, _kcfg(h))
        assert _on(r) == []


# =====================================================================================
# 4. The measurement the deferral rests on
# =====================================================================================


class TestMeasuredSaturation:
    def _est(self) -> HotWaterEstimator:
        return HotWaterEstimator.from_temperature(TANK, 60.0)

    def test_unknown_until_observed(self):
        est = self._est()
        assert est.minutes_since_saturated is None
        est.update(5.0, 0.0, switch_on=False)
        assert est.minutes_since_saturated is None

    def test_relay_on_and_idle_long_enough_is_the_measurement(self):
        est = self._est()
        est.update(5.0, 0.0, switch_on=True)
        assert est.minutes_since_saturated is None  # 5 min is not yet saturation
        est.update(5.0, 0.0, switch_on=True)
        assert est.minutes_since_saturated == 0.0

    def test_it_ages_once_the_relay_opens(self):
        est = self._est()
        for _ in range(3):
            est.update(5.0, 0.0, switch_on=True)
        for _ in range(6):
            est.update(5.0, 0.0, switch_on=False)
        assert est.minutes_since_saturated == pytest.approx(30.0)

    def test_a_block_the_plan_cut_short_is_not_a_measurement(self):
        """THE distinction. The element draws full power for 45 minutes and the relay
        opens under it: stored_kwh re-pins to full, the thermostat never spoke."""
        est = HotWaterEstimator.from_temperature(TANK, 30.0)
        for _ in range(9):
            est.update(5.0, POWER, switch_on=True)
        est.update(5.0, 0.0, switch_on=False)
        assert est.stored_kwh == pytest.approx(CAPACITY)  # the model says full
        assert est.minutes_since_saturated is None  # the measurement does not

    def test_an_unknown_relay_never_counts(self):
        est = self._est()
        for _ in range(6):
            est.update(5.0, 0.0, switch_on=None)
        assert est.minutes_since_saturated is None

    def test_it_survives_a_restart(self):
        est = self._est()
        for _ in range(3):
            est.update(5.0, 0.0, switch_on=True)
        est.update(15.0, 0.0, switch_on=False)
        other = self._est()
        other.apply_state(est.state_dict())
        assert other.minutes_since_saturated == pytest.approx(15.0)

    def test_old_state_files_load_as_unknown(self):
        est = self._est()
        est.apply_state({"stored_kwh": 10.0, "saturated_min": 0.0})
        assert est.minutes_since_saturated is None
        assert est.state_dict()["min_since_saturated"] == -1.0


class TestGateIgnoresUnknownAge:
    """state_dict writes -1 for 'never'; the gate must read that as unknown."""

    def test_minus_one_is_not_zero_minutes_ago(self):
        h = _gate(_state(sat_age=-1.0))
        assert h.first_bucket_min_kwh is None


def test_simple_namespace_heaters_still_work():
    """The gate is also driven with duck-typed heaters in the existing tests."""
    h = SimpleNamespace(id="main_tank", min_kwh_per_day=6.0, power_kw=POWER)
    with (
        patch(
            "planner.pipeline._load_hot_water_states",
            return_value={"main_tank": _state()},
        ),
        patch("planner.pipeline._hot_water_state_age_min", return_value=3.0),
    ):
        _apply_water_shortfall_gate(
            [h], [TANK_CFG], _slots(NOW, 17.75, _autumn), NOW, {"main_tank": HEATED},
            defer_hours=10.0,
        )
    assert h.first_bucket_min_kwh == pytest.approx(HEATED)
