"""departure_days: the planner's weekday filter for recurring EV deadlines.

Shares the weekday vocabulary with the servo's recurring_deadline_days so plan floors
and servo floors agree on which mornings exist — a Friday-evening plan must not fill
the commuter car for a Saturday 07:30 that nobody drives.
"""

from datetime import datetime

import pytz

from planner.pipeline import calculate_ev_deadline
from planner.solver.adapter import build_ev_charger_inputs

TZ = pytz.timezone("Europe/Stockholm")
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]


def _at(y, mo, d, h, mi):
    return TZ.localize(datetime(y, mo, d, h, mi))


class TestDepartureDays:
    def test_friday_evening_rolls_to_monday(self):
        # 2026-08-14 is a Friday; 07:30 has passed -> next allowed day is Monday.
        dl = calculate_ev_deadline("07:30", _at(2026, 8, 14, 9, 0), "Europe/Stockholm", WEEKDAYS)
        assert dl is not None
        assert (dl.year, dl.month, dl.day, dl.hour, dl.minute) == (2026, 8, 17, 7, 30)

    def test_weekday_early_morning_same_day(self):
        dl = calculate_ev_deadline("07:30", _at(2026, 8, 12, 5, 0), "Europe/Stockholm", WEEKDAYS)
        assert dl is not None
        assert (dl.day, dl.hour, dl.minute) == (12, 7, 30)

    def test_saturday_rolls_to_monday(self):
        dl = calculate_ev_deadline("07:30", _at(2026, 8, 15, 5, 0), "Europe/Stockholm", WEEKDAYS)
        assert dl is not None
        assert (dl.day, dl.hour) == (17, 7)  # Monday

    def test_none_days_is_daily_legacy(self):
        dl = calculate_ev_deadline("07:30", _at(2026, 8, 14, 9, 0), "Europe/Stockholm", None)
        assert dl is not None
        assert (dl.day, dl.hour) == (15, 7)  # Saturday — daily behaviour unchanged

    def test_invalid_day_names_disable_deadline(self):
        """Wholly-invalid day list => None — SAME degradation as the servo, so a config
        typo can never make the planner fill weekend mornings the servo would not floor."""
        dl = calculate_ev_deadline(
            "07:30", _at(2026, 8, 14, 9, 0), "Europe/Stockholm", ["blursday"]
        )
        assert dl is None

    def test_case_and_long_names_accepted(self):
        dl = calculate_ev_deadline(
            "07:30", _at(2026, 8, 14, 9, 0), "Europe/Stockholm", ["Monday", "TUE"]
        )
        assert dl is not None
        assert (dl.day,) == (17,)  # Monday


class TestDSTCrossings:
    """The deadline is a WALL-CLOCK guarantee: 07:30 local must stay 07:30 local across
    DST transitions. Arithmetic on a pytz-aware `now` keeps the offset valid at plan time,
    which put the Monday-after-spring-forward deadline at 08:30 true local (1 h late) and
    the autumn one at 06:30 (1 h early). Assert on the UTC instant, not naive fields."""

    def test_spring_forward_friday_to_monday(self):
        # EU spring transition 2026-03-29 (Sunday). Friday 2026-03-27 20:00 CET (+01:00).
        now = _at(2026, 3, 27, 20, 0)
        dl = calculate_ev_deadline("07:30", now, "Europe/Stockholm", WEEKDAYS)
        assert dl is not None
        # Monday 2026-03-30 07:30 CEST (+02:00) == 05:30 UTC.
        assert dl.astimezone(pytz.UTC) == pytz.UTC.localize(datetime(2026, 3, 30, 5, 30))

    def test_fall_back_friday_to_monday(self):
        # EU autumn transition 2026-10-25 (Sunday). Friday 2026-10-23 20:00 CEST (+02:00).
        now = _at(2026, 10, 23, 20, 0)
        dl = calculate_ev_deadline("07:30", now, "Europe/Stockholm", WEEKDAYS)
        assert dl is not None
        # Monday 2026-10-26 07:30 CET (+01:00) == 06:30 UTC.
        assert dl.astimezone(pytz.UTC) == pytz.UTC.localize(datetime(2026, 10, 26, 6, 30))

    def test_servo_twin_agrees_across_spring_forward(self):
        """Plan floors and servo floors must reference the SAME instant."""
        from executor.ev_surplus_runtime import next_recurring_deadline_ts

        now = _at(2026, 3, 27, 20, 0)
        planner_dl = calculate_ev_deadline("07:30", now, "Europe/Stockholm", WEEKDAYS)
        servo_ts = next_recurring_deadline_ts(
            ("mon", "tue", "wed", "thu", "fri"), "07:30", now.timestamp(), "Europe/Stockholm"
        )
        assert planner_dl is not None and servo_ts is not None
        assert planner_dl.timestamp() == servo_ts


class TestVacationIncentiveStrip:
    """The vacation strip must yield zero incentives WITHOUT crashing the adapter.

    Regression: stripping with penalty_levels=None blew up build_ev_charger_inputs
    (dict.get's default only applies when the key is ABSENT) — a TypeError on every
    planner run for the whole vacation, taking water/battery replanning down with it.
    """

    @staticmethod
    def _cfg():
        return [
            {
                "id": "tesla", "enabled": True, "max_power_kw": 11.0,
                "battery_capacity_kwh": 60.0,
                "penalty_levels": [
                    {"max_soc": 40, "penalty_sek": 2.5},
                    {"max_soc": 90, "penalty_sek": 0.40},
                ],
            },
        ]

    @staticmethod
    def _state():
        return [{"id": "tesla", "soc_percent": 30.0, "plugged_in": True,
                 "at_home": True, "deadline": None}]

    def test_stripped_config_builds_zero_buckets(self):
        stripped = [{**e, "penalty_levels": []} for e in self._cfg()]
        inputs = build_ev_charger_inputs(stripped, self._state())
        assert len(inputs) == 1
        assert inputs[0].incentive_buckets == []

    def test_explicit_none_is_survivable(self):
        """User YAML `penalty_levels:` (empty => None) must not crash either."""
        noneed = [{**e, "penalty_levels": None} for e in self._cfg()]
        inputs = build_ev_charger_inputs(noneed, self._state())
        assert inputs[0].incentive_buckets == []

    def test_unstripped_config_keeps_buckets(self):
        inputs = build_ev_charger_inputs(self._cfg(), self._state())
        assert [b.threshold_soc for b in inputs[0].incentive_buckets] == [40.0, 90.0]
