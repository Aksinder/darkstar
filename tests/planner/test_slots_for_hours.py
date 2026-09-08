"""A window configured in HOURS must not quietly become a window in SLOTS.

Two price windows in the pipeline made that mistake in different ways. The battery-value
window hardcoded `int(hours * 4)`; the dynamic-WTP window counted slots off the list that
had just been reassigned to the COARSENED grid, where a slot is no longer 15 minutes.
Both are correct today only because coarse_tail_fine_hours: 24 happens to cover the whole
24 h window — at fine_hours 12 the same code turns a 24 h rolling percentile into a
mixed-resolution one weighted 4:1 toward the near term, with nothing in the log to say so.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from planner.pipeline import slots_for_hours

T0 = datetime(2026, 9, 8, 0, 0)


def _series(n, minutes):
    return [
        SimpleNamespace(
            start_time=T0 + timedelta(minutes=minutes * i),
            end_time=T0 + timedelta(minutes=minutes * (i + 1)),
        )
        for i in range(n)
    ]


class TestItCountsHoursNotSlots:
    def test_quarter_hour_grid(self):
        assert slots_for_hours(_series(200, 15), 24) == 96
        assert slots_for_hours(_series(200, 15), 12) == 48

    def test_hourly_grid_gives_a_quarter_of_the_slots(self):
        """The old int(hours * 4) returned 96 here — four days, not one."""
        assert slots_for_hours(_series(200, 60), 24) == 24

    def test_five_minute_grid(self):
        assert slots_for_hours(_series(500, 5), 12) == 144


class TestItRefusesToReadPastTheEnd:
    def test_a_48h_ask_on_a_36h_horizon_is_clipped(self):
        assert slots_for_hours(_series(144, 15), 48) == 144

    def test_never_returns_zero(self):
        assert slots_for_hours(_series(10, 15), 0.01) == 1

    def test_an_empty_series_is_one(self):
        assert slots_for_hours([], 24) == 1


class TestDegenerateGrids:
    def test_a_zero_length_slot_falls_back_rather_than_dividing_by_zero(self):
        bad = [SimpleNamespace(start_time=T0, end_time=T0) for _ in range(200)]
        assert slots_for_hours(bad, 24) == 96

    def test_the_fallback_is_configurable(self):
        bad = [SimpleNamespace(start_time=T0, end_time=T0) for _ in range(200)]
        assert slots_for_hours(bad, 24, fallback_slot_min=60.0) == 24
