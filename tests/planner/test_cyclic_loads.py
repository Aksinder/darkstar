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

    def test_absent_defaults_to_zero_for_cyclic_loads(self):
        """Changed 2026-08-20: inheriting the tanks' 5 h default was both wrong
        (a pump has no reheat dynamics) and expensive (36 s vs 11 s on the live
        instance — past the box's 240 s budget, so the pumps never entered a
        plan). Tanks keep their 5 h default; see the water-heater test below."""
        spec = {k: v for k, v in POOL.items() if k != "min_spacing_hours"}
        (pump,) = _inputs([spec])
        assert pump.min_spacing_hours == 0.0

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


class TestSpacingDefaultsToZero:
    """The tanks' 5 h spacing default must NOT leak onto cyclic loads.

    Semantics: spacing keeps tank heating from fragmenting; a pump has no reheat
    dynamics and wants fragmentation freedom — max_hours_between is its real
    requirement. Solver cost: the spacing constraint is disjunctive, and the
    inherited default measured 36 s vs 11 s on the live 2026-08-20 instance —
    past the 240 s box budget, so dump-and-keep-plan fired and the pumps
    silently never entered a plan. The gate that silently does not gate, again.
    """

    def test_absent_spacing_maps_to_zero(self):
        from planner.solver.adapter import cyclic_loads_as_heater_specs

        specs = cyclic_loads_as_heater_specs(
            [{"id": "poolpump", "power_kw": 0.26, "min_kwh_per_day": 0.78}]
        )
        assert specs[0]["water_min_spacing_hours"] == 0.0

    def test_explicit_spacing_still_wins(self):
        from planner.solver.adapter import cyclic_loads_as_heater_specs

        specs = cyclic_loads_as_heater_specs(
            [{"id": "poolpump", "power_kw": 0.26, "min_spacing_hours": 2.0}]
        )
        assert specs[0]["water_min_spacing_hours"] == 2.0
