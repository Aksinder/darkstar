"""Cyclic loads (pool pump, filter) ride the water-heater solver primitive.

A pump and a tank pose the SAME problem to the solver: a recurring daily energy need,
splittable across slots, with a minimum spacing between starts and a maximum gap
between them. The MILP block contains no temperature term at all — verified by grep,
and by these tests — so a second copy of it for pumps would be duplicated constraints
to keep in sync forever, for no modelling gain. Only naming and actuation differ, and
both live outside the solver.
"""

from __future__ import annotations

from planner.solver.adapter import (
    build_water_heater_inputs,
    cyclic_loads_as_heater_specs,
)

POOL = {
    "id": "poolpump",
    "switch_entity": "switch.poolpump",
    "power_kw": 0.26,
    "min_kwh_per_day": 0.78,
    "max_hours_between": 24,
    "min_spacing_hours": 0,
}


def _inputs(loads):
    return build_water_heater_inputs(cyclic_loads_as_heater_specs(loads), {}, [])


class TestMapping:
    def test_a_pump_becomes_a_recurring_daily_need(self):
        (pump,) = _inputs([POOL])
        assert pump.id == "poolpump"
        assert pump.power_kw == 0.26
        assert pump.min_kwh_per_day == 0.78

    def test_honest_names_are_translated(self):
        """The config says max_hours_between; a pump is not spelled as a heater."""
        (pump,) = _inputs([POOL])
        assert pump.max_hours_between_heating == 24.0

    def test_the_gap_defaults_to_a_day(self):
        (pump,) = _inputs([{**POOL, "max_hours_between": None}])
        assert pump.max_hours_between_heating == 24.0

    def test_non_dicts_are_skipped_not_fatal(self):
        assert _inputs(["nonsense", None, POOL]) != []

    def test_an_empty_list_is_empty(self):
        assert _inputs([]) == []


class TestSpacingZeroIsNotAbsent:
    """`or 5.0` collapsed an explicit "no spacing" into five hours of forced
    separation. Found 2026-08-19: the spa has been configured 0 since setup and
    silently ran with 5 the whole time, and a pump wanting continuous blocks would
    have hit the same wall."""

    def test_explicit_zero_is_honoured(self):
        (pump,) = _inputs([POOL])
        assert pump.min_spacing_hours == 0.0

    def test_absent_still_defaults_to_five(self):
        spec = {k: v for k, v in POOL.items() if k != "min_spacing_hours"}
        (pump,) = _inputs([spec])
        assert pump.min_spacing_hours == 5.0

    def test_an_explicit_value_passes_through(self):
        (pump,) = _inputs([{**POOL, "min_spacing_hours": 4}])
        assert pump.min_spacing_hours == 4.0

    def test_a_water_heater_gets_the_same_fix(self):
        """The bug was in the shared builder, so tanks were affected too."""
        (tank,) = build_water_heater_inputs(
            [{"id": "spa", "power_kw": 1.8, "min_kwh_per_day": 4,
              "water_min_spacing_hours": 0}], {}, [],
        )
        assert tank.min_spacing_hours == 0.0
