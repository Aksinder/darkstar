"""Water heaters as the fuse guard's third shed lever.

The S4 guard protects the HOUSE main — 25 A per phase, less a margin. That is the only
thing at risk: the cars sit on the house feed, not the villavagn's, so a car can trip
the main and nothing else, and only by pushing a phase past 25 A (owner, 2026-08-18).

Before this the guard could clamp cars and cap battery charging but never a tank, which
is backwards under scarcity: a tank waits an hour happily, a car may be leaving in the
morning. Shedding the car to protect a phase a 3.4 kW element sits on spends the
expensive option to save the cheap one.
"""

from __future__ import annotations

from executor.fuse_shed import should_shed_for_fuse

BUDGET = 23.0  # 25 A limit less a 2 A margin


def _shed(**kw):
    base = {
        "phase_currents_a": {"a": 8.0, "b": 26.0, "c": 7.0},
        "budget_a": BUDGET,
        "heater_phases": (),
        "grid_w": 2000.0,
    }
    base.update(kw)
    return should_shed_for_fuse(**base)


class TestTheOverloadCase:
    def test_a_phase_over_budget_sheds(self):
        shed, why = _shed()
        assert shed is True
        assert "phase b" in why and "26.0" in why

    def test_everything_within_budget_does_not_shed(self):
        assert _shed(phase_currents_a={"a": 8.0, "b": 12.0, "c": 7.0})[0] is False

    def test_the_budget_is_exclusive_at_the_boundary(self):
        assert _shed(phase_currents_a={"a": 0.0, "b": BUDGET, "c": 0.0})[0] is False


class TestPhaseMapping:
    def test_a_mapped_heater_ignores_someone_elses_overload(self):
        """The tank on phase A must not be shed for a car overloading phase B."""
        assert _shed(heater_phases=("a",))[0] is False

    def test_a_mapped_heater_sheds_for_its_own_phase(self):
        assert _shed(heater_phases=("b",))[0] is True

    def test_an_unmapped_heater_counts_against_every_phase(self):
        """Unknown wiring is conservative — it could be on the overloaded one."""
        assert _shed(heater_phases=())[0] is True

    def test_a_multi_phase_heater_watches_all_of_its_phases(self):
        assert _shed(heater_phases=("a", "b"))[0] is True
        assert _shed(heater_phases=("a", "c"))[0] is False


class TestExportCredit:
    """The meter is direction-blind: a big reading while EXPORTING is not an overload,
    because added consumption removes it one-for-one."""

    def test_export_credits_the_reading_down(self):
        # 6 kW export ~ 8.7 A per phase credited: 26 - 8.7 = 17.3, inside budget.
        assert _shed(grid_w=-6000.0)[0] is False

    def test_a_genuine_overload_survives_a_small_export(self):
        assert _shed(grid_w=-500.0)[0] is True


class TestDeliberatelyNotFailSafeToShed:
    """The EV layer already stops every car on blind sensors — the larger load, and one
    that can catch up. Cutting hot water on a sensor hiccup is the worse trade."""

    def test_blind_sensors_leave_the_heater_alone(self):
        shed, why = _shed(phase_currents_a=None)
        assert shed is False
        assert "unreadable" in why

    def test_empty_readings_leave_the_heater_alone(self):
        assert _shed(phase_currents_a={})[0] is False

    def test_the_guard_being_off_never_sheds(self):
        shed, why = _shed(budget_a=None)
        assert shed is False
        assert "guard off" in why

    def test_a_phase_map_naming_nothing_measured_does_not_shed(self):
        """A typo must not silently arm or disarm — no watched phase, no verdict."""
        shed, why = _shed(heater_phases=("z",))
        assert shed is False
        assert "no watched phase" in why
