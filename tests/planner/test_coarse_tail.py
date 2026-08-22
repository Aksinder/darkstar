"""Coarse-tail horizon: full look-ahead, fewer variables (planner/coarse_tail.py)."""

from __future__ import annotations

from datetime import datetime, timedelta

from planner.coarse_tail import coarsen_slots, expand_result
from planner.solver.types import KeplerInputSlot, KeplerResult, KeplerResultSlot

T0 = datetime(2026, 8, 22, 4, 0)


def _slots(n, start=T0):
    return [
        KeplerInputSlot(
            start_time=start + timedelta(minutes=15 * i),
            end_time=start + timedelta(minutes=15 * (i + 1)),
            load_kwh=0.25,
            pv_kwh=0.1,
            import_price_sek_kwh=1.0 + i * 0.01,
            export_price_sek_kwh=0.5,
        )
        for i in range(n)
    ]


class TestCoarsening:
    def test_the_fine_window_is_untouched(self):
        slots = _slots(96 + 16)  # 24 h fine + 4 h tail
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        assert coarse[:96] == slots[:96]
        assert groups[:96] == [[i] for i in range(96)]

    def test_the_tail_becomes_hourly(self):
        coarse, groups = coarsen_slots(_slots(96 + 16), fine_hours=24)
        assert len(coarse) == 96 + 4
        assert groups[96] == [96, 97, 98, 99]
        assert coarse[96].end_time - coarse[96].start_time == timedelta(hours=1)

    def test_energy_is_summed_and_price_averaged(self):
        """A merged slot must present the same ENERGY at the same average price —
        anything else optimises a different day than the one that is coming."""
        slots = _slots(96 + 4)
        coarse, _ = coarsen_slots(slots, fine_hours=24)
        merged = coarse[-1]
        assert merged.load_kwh == sum(s.load_kwh for s in slots[96:])
        assert merged.pv_kwh == sum(s.pv_kwh for s in slots[96:])
        assert merged.import_price_sek_kwh == sum(
            s.import_price_sek_kwh for s in slots[96:]
        ) / 4

    def test_total_energy_is_conserved(self):
        slots = _slots(192)
        coarse, _ = coarsen_slots(slots, fine_hours=24)
        assert sum(s.load_kwh for s in coarse) == sum(s.load_kwh for s in slots)
        assert sum(s.pv_kwh for s in coarse) == sum(s.pv_kwh for s in slots)

    def test_the_variable_count_actually_drops(self):
        """The whole point: 48 h of look-ahead at 120 variables instead of 192."""
        coarse, _ = coarsen_slots(_slots(192), fine_hours=24)
        assert len(coarse) == 96 + 24

    def test_a_partial_hour_is_left_alone(self):
        """A merged slot whose duration lies about what it covers is worse than a
        fine one — the tail's first hour here has only two quarters."""
        slots = _slots(96 + 2, start=T0 + timedelta(minutes=30))
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        assert sum(len(g) for g in groups) == len(slots)

    def test_disabled_is_the_identity(self):
        slots = _slots(20)
        for fine in (0, -1):
            coarse, groups = coarsen_slots(slots, fine_hours=fine)
            assert coarse == slots
            assert groups == [[i] for i in range(20)]

    def test_a_horizon_shorter_than_the_window_stays_fine(self):
        slots = _slots(40)  # 10 h < 24 h
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        assert coarse == slots
        assert groups == [[i] for i in range(40)]

    def test_empty(self):
        assert coarsen_slots([], 24) == ([], [])


def _result(coarse, per_slot):
    return KeplerResult(
        slots=[
            KeplerResultSlot(
                start_time=c.start_time, end_time=c.end_time,
                soc_kwh=10.0 + i, cost_sek=1.0, **per_slot,
            )
            for i, c in enumerate(coarse)
        ],
        total_cost_sek=1.0 * len(coarse), is_optimal=True, status_msg="Optimal",
    )


class TestExpansion:
    """Coarsening is an internal solver optimisation; every consumer downstream is
    entitled to the 15-minute grid it handed in."""

    def test_the_grid_is_restored(self):
        slots = _slots(96 + 8)
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        out = expand_result(
            _result(coarse, {"charge_kwh": 2.0, "discharge_kwh": 0.0,
                             "grid_import_kwh": 2.0, "grid_export_kwh": 0.0}),
            groups, slots,
        )
        assert len(out.slots) == len(slots)
        assert [s.start_time for s in out.slots] == [s.start_time for s in slots]

    def test_energies_are_divided_not_repeated(self):
        """A 2 kWh hourly charge is four 0.5 kWh quarters — repeating it would
        quadruple the energy in the plan."""
        slots = _slots(96 + 4)
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        out = expand_result(
            _result(coarse, {"charge_kwh": 2.0, "discharge_kwh": 0.0,
                             "grid_import_kwh": 2.0, "grid_export_kwh": 0.0}),
            groups, slots,
        )
        tail = out.slots[96:]
        assert len(tail) == 4
        assert all(abs(s.charge_kwh - 0.5) < 1e-9 for s in tail)
        assert abs(sum(s.charge_kwh for s in tail) - 2.0) < 1e-9

    def test_powers_are_carried_through_unchanged(self):
        """Energies divide, POWERS do not: a 3.4 kW heater runs at 3.4 kW in each
        quarter of the hour it was given."""
        slots = _slots(96 + 4)
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        out = expand_result(
            _result(coarse, {"charge_kwh": 0.0, "discharge_kwh": 0.0,
                             "grid_import_kwh": 0.0, "grid_export_kwh": 0.0,
                             "water_heat_kw": 3.4,
                             "water_heater_results": {"main_tank": 3.4}}),
            groups, slots,
        )
        assert all(s.water_heat_kw == 3.4 for s in out.slots[96:])
        assert all(s.water_heater_results["main_tank"] == 3.4 for s in out.slots[96:])

    def test_soc_is_interpolated_not_repeated(self):
        """soc_kwh is the END of a slot; repeating it draws a flat line through an
        hour the battery actually spent charging."""
        slots = _slots(96 + 4)
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        out = expand_result(
            _result(coarse, {"charge_kwh": 2.0, "discharge_kwh": 0.0,
                             "grid_import_kwh": 2.0, "grid_export_kwh": 0.0}),
            groups, slots,
        )
        tail_soc = [s.soc_kwh for s in out.slots[96:]]
        assert tail_soc == sorted(tail_soc)
        assert abs(tail_soc[-1] - out.slots[96 + 3].soc_kwh) < 1e-9
        # Ends where the coarse slot said it would.
        assert abs(tail_soc[-1] - (10.0 + 96)) < 1e-9

    def test_cost_is_conserved(self):
        slots = _slots(96 + 8)
        coarse, groups = coarsen_slots(slots, fine_hours=24)
        res = _result(coarse, {"charge_kwh": 1.0, "discharge_kwh": 0.0,
                               "grid_import_kwh": 1.0, "grid_export_kwh": 0.0})
        out = expand_result(res, groups, slots)
        assert abs(sum(s.cost_sek for s in out.slots)
                   - sum(s.cost_sek for s in res.slots)) < 1e-9

    def test_a_shape_mismatch_is_handed_back_untouched(self):
        """Never invent an alignment: if the solver returned a different shape,
        that is an upstream problem and must stay visible."""
        slots = _slots(20)
        _, groups = coarsen_slots(slots, fine_hours=24)
        res = _result(_slots(5), {"charge_kwh": 0.0, "discharge_kwh": 0.0,
                                  "grid_import_kwh": 0.0, "grid_export_kwh": 0.0})
        assert expand_result(res, groups, slots) is res

    def test_round_trip_with_coarsening_disabled(self):
        slots = _slots(12)
        coarse, groups = coarsen_slots(slots, fine_hours=0)
        out = expand_result(
            _result(coarse, {"charge_kwh": 1.0, "discharge_kwh": 0.0,
                             "grid_import_kwh": 1.0, "grid_export_kwh": 0.0}),
            groups, slots,
        )
        assert len(out.slots) == 12
        assert all(s.charge_kwh == 1.0 for s in out.slots)
