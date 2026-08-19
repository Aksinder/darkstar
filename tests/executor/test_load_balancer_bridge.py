"""Mapping between our runtime and upstream's ported load balancer.

The balancer core arrived verbatim from upstream with its own 40 tests. What is
NOT covered by those is the impedance match: our phases are named, upstream's
are numbered, and our EV layer is a different one entirely. Mapping code is
where silent mistakes live, so it gets its own tests.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from executor.config import BalancedLoadConfig, BalancedLoadType, GiveWayOrderEntry
from executor.load_balancer_bridge import (
    ev_entries,
    ordered_entries,
    phase_currents_to_int_keys,
    phase_to_int,
    phases_for_charger,
    shed_entries,
    uniform_updated_at,
)


class TestPhaseIdentity:
    def test_our_names_map_to_upstream_numbers(self):
        assert (phase_to_int("l1"), phase_to_int("L2"), phase_to_int("l3")) == (1, 2, 3)

    def test_bare_numbers_also_work(self):
        assert phase_to_int("2") == 2
        assert phase_to_int(3) == 3

    def test_an_unknown_key_is_dropped_not_guessed(self):
        """A phase silently mapped to the wrong number budgets the wrong conductor."""
        assert phase_to_int("phase_a") is None
        assert phase_to_int("") is None

    def test_currents_are_remapped(self):
        assert phase_currents_to_int_keys({"l1": 12.0, "l2": 24.0, "l3": 0.5}) == {
            1: 12.0, 2: 24.0, 3: 0.5,
        }

    def test_an_unknown_phase_does_not_poison_the_others(self):
        assert phase_currents_to_int_keys({"l1": 12.0, "spare": 99.0}) == {1: 12.0}

    def test_no_reading_is_an_empty_map(self):
        assert phase_currents_to_int_keys(None) == {}


class TestChargerPhases:
    def test_an_explicit_map_decides(self):
        assert phases_for_charger(("l2",)) == [2]

    def test_duplicates_and_order_are_normalised(self):
        assert phases_for_charger(("l3", "l1", "l3")) == [1, 3]

    def test_no_map_counts_against_every_phase(self):
        """Unknown must be conservative: the load could be on the overloaded one."""
        assert phases_for_charger(()) == [1, 2, 3]

    def test_an_unparseable_map_is_also_unknown(self):
        assert phases_for_charger(("nonsense",)) == [1, 2, 3]


def _charger(cid, **over):
    base = {
        "id": cid, "phase_map": (), "phases": 3,
        "min_current_a": 6.0, "max_current_a": 16.0, "controllable": True,
    }
    base.update(over)
    return SimpleNamespace(**base)


class TestEVEntries:
    def test_a_throttleable_charger_becomes_an_entry(self):
        out = ev_entries(
            [_charger("easee_fmb", phase_map=("l2",))],
            setpoints_a={"easee_fmb": 10.0},
            targets_a={"easee_fmb": 16.0},
        )
        ev = out["easee_fmb"]
        assert (ev.phases, ev.current_setpoint_a, ev.planner_target_a) == ([2], 10, 16)
        assert (ev.min_current_a, ev.max_current_a) == (6, 16)

    def test_an_onoff_charger_is_not_throttleable(self):
        """Writing a setpoint nothing reads is worse than shedding it as a load."""
        assert ev_entries(
            [_charger("tesla", controllable=False)], setpoints_a={}, targets_a={}
        ) == {}

    def test_not_charging_and_not_planned_are_both_none(self):
        ev = ev_entries([_charger("x")], setpoints_a={}, targets_a={})["x"]
        assert ev.current_setpoint_a is None
        assert ev.planner_target_a is None


class TestShedEntries:
    def test_a_configured_load_becomes_an_entry(self):
        out = shed_entries([
            BalancedLoadConfig(
                device_type=BalancedLoadType.WATER_HEATER, device_id="spa", phases=["l2"]
            )
        ])
        assert out["spa"].phases == [2]
        assert out["spa"].device_type == "water_heater"

    def test_an_unmapped_load_gets_no_phases(self):
        """Empty is upstream's own 'every phase' convention inside the balancer."""
        out = shed_entries([BalancedLoadConfig(device_id="villavagn", phases=[])])
        assert out["villavagn"].phases == []

    def test_an_id_less_entry_is_skipped(self):
        assert shed_entries([BalancedLoadConfig(device_id="")]) == {}


class TestOrderIsThePolicy:
    def test_entries_come_back_in_configured_order(self):
        evs = ev_entries([_charger("fmb")], setpoints_a={}, targets_a={})
        sheds = shed_entries([
            BalancedLoadConfig(device_id="spa", phases=["l2"]),
            BalancedLoadConfig(device_id="villavagn", phases=["l2"]),
        ])
        order = [
            GiveWayOrderEntry(kind="charger", id="fmb"),
            GiveWayOrderEntry(kind="shed", id="spa"),
            GiveWayOrderEntry(kind="shed", id="villavagn"),
        ]
        got = ordered_entries(order, evs, sheds)
        assert [getattr(e, "charger_id", None) or e.load_id for e in got] == [
            "fmb", "spa", "villavagn",
        ]

    def test_a_missing_device_is_skipped_not_appended(self):
        """Position IS the policy; appending somewhere arbitrary invents a priority."""
        got = ordered_entries(
            [GiveWayOrderEntry(kind="shed", id="ghost")], {}, {}
        )
        assert got == []

    def test_kind_must_match_the_registry(self):
        sheds = shed_entries([BalancedLoadConfig(device_id="spa", phases=["l2"])])
        assert ordered_entries(
            [GiveWayOrderEntry(kind="charger", id="spa")], {}, sheds
        ) == []


class TestStaleness:
    def test_every_present_phase_is_stamped_now(self):
        """Our reader already rejected the snapshot if ANY phase was stale."""
        now = datetime(2026, 8, 19, 18, 0, 0)
        assert uniform_updated_at({1: 12.0, 2: 24.0}, now) == {1: now, 2: now}

    def test_no_reading_stamps_nothing(self):
        """Which makes every phase read as stale to the balancer — fail-safe."""
        assert uniform_updated_at({}, datetime(2026, 8, 19)) == {}
