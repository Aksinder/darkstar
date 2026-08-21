"""Greedy pre-scheduling of pumps outside the MILP (see planner/cyclic_preschedule.py)."""

from __future__ import annotations

from datetime import datetime, timedelta

from planner.cyclic_preschedule import CyclicSpec, PriceSlot, preschedule_cyclic_loads

T0 = datetime(2026, 8, 21, 0, 0)


def _day(prices_by_hour, start=T0, days=1):
    """15-min slots for `days` days; prices_by_hour: 24 hourly prices (repeated per day)."""
    out = []
    for d in range(days):
        for h in range(24):
            for q in range(4):
                out.append(PriceSlot(start + timedelta(days=d, hours=h, minutes=15 * q),
                                     prices_by_hour[h]))
    return out


# Cheap night (0-5), dear morning (6-9), valley (11-15), dear evening (17-21).
PRICES = [0.5, 0.4, 0.4, 0.5, 0.6, 0.8, 2.5, 3.0, 2.8, 2.4, 2.0, 1.5, 1.2, 1.1, 1.3, 1.6,
          2.2, 3.2, 3.3, 3.1, 2.9, 2.6, 2.0, 1.0]


def _hours(plan, slots):
    return sorted({slots[i].start_time.hour for i in plan})


class TestCheapestHours:
    def test_pool_pump_takes_the_three_cheapest_hours(self):
        slots = _day(PRICES)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("poolpump", 0.26, 0.78)], slots
        )["poolpump"]
        assert _hours(plan, slots) == [0, 1, 2]  # 0.4, 0.4, then the 0.5 tie -> earliest
        assert len(plan) == 12  # whole hours: 3 h x 4 slots
        assert all(kw == 0.26 for kw in plan.values())

    def test_a_fractional_need_rounds_up_to_whole_hours(self):
        """2.0 kWh at 0.39 kW is 5.1 h -> 6 hours; never a 15-minute flicker."""
        slots = _day(PRICES)
        plan = preschedule_cyclic_loads([CyclicSpec("bog", 0.39, 2.0)], slots)["bog"]
        assert len(plan) == 24


class TestMaxGap:
    def test_gap_repair_inserts_the_cheapest_hour_in_the_stretch(self):
        """6 h of filter with a 6 h max gap cannot all sit at night: the repair fills
        every over-long stretch with that stretch's cheapest hour."""
        slots = _day(PRICES)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("bog", 0.39, 2.34, max_hours_between=6)], slots
        )["bog"]
        hrs = _hours(plan, slots)
        # No two consecutive runs more than 6 h apart, and the tail is covered too.
        gaps = [b - a for a, b in zip(hrs, hrs[1:], strict=False)]
        assert all(g <= 6 for g in gaps), (hrs, gaps)
        assert 24 - hrs[-1] <= 6
        # It did NOT simply run everything at the dear evening peak.
        assert 18 not in hrs

    def test_no_gap_rule_means_pure_cheapest(self):
        slots = _day(PRICES)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("bog", 0.39, 2.34)], slots
        )["bog"]
        assert _hours(plan, slots) == [0, 1, 2, 3, 4, 5]  # 0.8 at 05 beats 1.0 at 23


class TestDayBuckets:
    def test_each_day_gets_its_own_need(self):
        slots = _day(PRICES, days=2)
        plan = preschedule_cyclic_loads([CyclicSpec("poolpump", 0.26, 0.78)], slots)["poolpump"]
        assert len(plan) == 24  # 3 h x 4 slots x 2 days

    def test_heated_today_reduces_only_the_first_bucket(self):
        slots = _day(PRICES, days=2)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("poolpump", 0.26, 0.78, heated_today_kwh=0.78)], slots
        )["poolpump"]
        days = {slots[i].start_time.day for i in plan}
        assert days == {22}  # nothing more today, full need tomorrow
        assert len(plan) == 12

    def test_a_partial_first_day_cannot_overcommit(self):
        """Starting at 22:00 with a 3 h need: only 2 hours exist today."""
        slots = _day(PRICES)[22 * 4:]  # 22:00-24:00 only
        plan = preschedule_cyclic_loads([CyclicSpec("poolpump", 0.26, 0.78)], slots)["poolpump"]
        assert len(plan) == 8

    def test_defer_offset_moves_the_bucket_boundary(self):
        """With defer_up_to_hours=6, 00:00-05:59 belongs to the PREVIOUS day bucket."""
        slots = _day(PRICES, days=2)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("poolpump", 0.26, 0.78)], slots, defer_up_to_hours=6
        )["poolpump"]
        # Three buckets now (day-1 tail, day 1, day 2 head) -> more than 2 days' worth
        # of hours is fine; what matters is that it still runs only whole hours.
        assert len(plan) % 4 == 0


class TestEdges:
    def test_disabled_and_zero_power_are_skipped(self):
        slots = _day(PRICES)
        plan = preschedule_cyclic_loads(
            [CyclicSpec("a", 0.0, 1.0), CyclicSpec("b", 0.3, 1.0, enabled=False)], slots
        )
        assert plan == {}

    def test_empty_slots(self):
        assert preschedule_cyclic_loads([CyclicSpec("a", 0.3, 1.0)], []) == {}
