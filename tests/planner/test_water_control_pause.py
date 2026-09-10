"""A control-paused tank must not be planned.

The executor has always honoured these helpers by skipping actuation. The planner did
not know about them — and planned water heat goes straight into kepler's node balance
(kepler.py ~576). So a device that cannot draw was reserving PV or grid import in the
plan's arithmetic: on a sunny day the solver could decline to charge the battery because
it expected the tank to soak the surplus, and export it instead.

2026-09-09: the villavagn tank sat control-paused for three days, at 0% and 11.7 C, with
its 3 kWh reliability floor booked on every replan, 14 slots anchored, and not one kWh
ever actuated. Nothing in the plan said why.

FAIL-SAFE DIRECTION. An unreadable pause helper reads as NOT paused, in the planner
exactly as in the executor. Inverting it here would let one HA hiccup silently drop a
tank out of the plan, which is worse than the phantom load this removes.
"""

from __future__ import annotations

from planner.solver.adapter import build_water_heater_inputs

CFG = [
    {"id": "main_tank", "enabled": True, "power_kw": 3.4, "min_kwh_per_day": 6},
    {"id": "villavagn_tank", "enabled": True, "power_kw": 1.6, "min_kwh_per_day": 3},
]


def _ids(states):
    return [h.id for h in build_water_heater_inputs(CFG, {}, states)]


class TestThePausedDeviceLeavesThePlan:
    def test_a_paused_tank_is_excluded(self):
        ids = _ids([{"id": "villavagn_tank", "control_paused": True}])
        assert ids == ["main_tank"]

    def test_the_others_are_untouched(self):
        ids = _ids([{"id": "villavagn_tank", "control_paused": True}])
        assert "main_tank" in ids

    def test_both_paused_leaves_nothing(self):
        ids = _ids(
            [
                {"id": "main_tank", "control_paused": True},
                {"id": "villavagn_tank", "control_paused": True},
            ]
        )
        assert ids == []

    def test_no_pause_plans_everything(self):
        assert sorted(_ids([])) == ["main_tank", "villavagn_tank"]

    def test_a_paused_tank_carries_no_floor_into_the_plan(self):
        """The point of the exclusion: its daily floor must not be booked."""
        heaters = build_water_heater_inputs(
            CFG, {}, [{"id": "villavagn_tank", "control_paused": True}]
        )
        assert all(h.id != "villavagn_tank" for h in heaters)


class TestFailSafeDirection:
    """Every ambiguous form must read as NOT paused — the same direction the executor
    uses, for the same reason."""

    def test_flag_absent_is_not_paused(self):
        assert "villavagn_tank" in _ids([{"id": "villavagn_tank"}])

    def test_flag_false_is_not_paused(self):
        assert "villavagn_tank" in _ids([{"id": "villavagn_tank", "control_paused": False}])

    def test_flag_none_is_not_paused(self):
        assert "villavagn_tank" in _ids([{"id": "villavagn_tank", "control_paused": None}])

    def test_no_states_at_all_is_not_paused(self):
        assert sorted(_ids(None)) == ["main_tank", "villavagn_tank"]

    def test_a_state_entry_without_an_id_is_ignored(self):
        assert sorted(_ids([{"control_paused": True}])) == ["main_tank", "villavagn_tank"]

    def test_a_pause_for_an_unknown_id_changes_nothing(self):
        assert sorted(_ids([{"id": "spa", "control_paused": True}])) == [
            "main_tank",
            "villavagn_tank",
        ]


class TestItComposesWithTheExistingFilter:
    def test_disabled_still_wins_on_its_own(self):
        cfg = [dict(CFG[0], enabled=False), CFG[1]]
        assert build_water_heater_inputs(cfg, {}, [])[0].id == "villavagn_tank"

    def test_zero_power_still_excluded(self):
        cfg = [dict(CFG[0], power_kw=0.0), CFG[1]]
        assert [h.id for h in build_water_heater_inputs(cfg, {}, [])] == ["villavagn_tank"]
