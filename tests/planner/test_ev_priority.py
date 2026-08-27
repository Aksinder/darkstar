"""Owner EV levers: priority re-weighting + one-off departure band extension.

Live driver 2026-08-27: input_select.darkstar_ev_priority=tesla_first (set
19:12) and departure 07:25 (set 22:53) were both ignored by the planner —
penalty_levels are static config — so the FMB bought every cheap night slot up
to its 86 % band while the Tesla stood at 41 % at departure.
"""

from __future__ import annotations

from planner.ev_priority import apply_departure_target, apply_priority_order
from planner.solver.types import EVChargerInput, IncentiveBucket


def _tesla() -> EVChargerInput:
    return EVChargerInput(
        id="tesla",
        max_power_kw=11.0,
        battery_capacity_kwh=60.0,
        current_soc_percent=41.0,
        plugged_in=True,
        deadline=None,
        incentive_buckets=[
            IncentiveBucket(threshold_soc=40.0, value_sek=2.5),
            IncentiveBucket(threshold_soc=90.0, value_sek=0.4),
        ],
    )


def _fmb() -> EVChargerInput:
    return EVChargerInput(
        id="easee_fmb",
        max_power_kw=3.7,
        battery_capacity_kwh=28.0,
        current_soc_percent=71.0,
        plugged_in=True,
        deadline=None,
        incentive_buckets=[
            IncentiveBucket(threshold_soc=86.0, value_sek=2.0),
            IncentiveBucket(threshold_soc=100.0, value_sek=0.35),
        ],
    )


class TestPriorityOrder:
    def test_tesla_first_demotes_fmb_below_teslas_lowest_band(self):
        tesla, fmb = _tesla(), _fmb()
        notes = apply_priority_order([tesla, fmb], ["tesla", "easee_fmb"])
        assert notes, "the demotion must be logged"
        # Tesla untouched
        assert [b.value_sek for b in tesla.incentive_buckets] == [2.5, 0.4]
        # FMB capped strictly below Tesla's lowest band (0.4 - 0.05)
        assert all(b.value_sek <= 0.35 for b in fmb.incentive_buckets)
        assert fmb.incentive_buckets[0].value_sek == 0.35

    def test_fmb_first_demotes_tesla(self):
        tesla, fmb = _tesla(), _fmb()
        apply_priority_order([tesla, fmb], ["easee_fmb", "tesla"])
        assert [b.value_sek for b in fmb.incentive_buckets] == [2.0, 0.35]
        # ceiling = 0.35 - 0.05 = 0.30
        assert all(b.value_sek <= 0.30 for b in tesla.incentive_buckets)

    def test_auto_is_a_noop(self):
        tesla, fmb = _tesla(), _fmb()
        assert apply_priority_order([tesla, fmb], None) == []
        assert [b.value_sek for b in tesla.incentive_buckets] == [2.5, 0.4]
        assert [b.value_sek for b in fmb.incentive_buckets] == [2.0, 0.35]

    def test_unknown_preferred_id_is_a_noop(self):
        tesla, fmb = _tesla(), _fmb()
        assert apply_priority_order([tesla, fmb], ["nonexistent"]) == []
        assert [b.value_sek for b in fmb.incentive_buckets] == [2.0, 0.35]

    def test_ceiling_never_goes_below_the_floor(self):
        # A preferred car with a 0.05 lowest band must not zero the other out.
        tesla, fmb = _tesla(), _fmb()
        tesla.incentive_buckets[1].value_sek = 0.05
        apply_priority_order([tesla, fmb], ["tesla", "easee_fmb"])
        assert all(b.value_sek >= 0.05 for b in fmb.incentive_buckets)


class TestDepartureTarget:
    def test_future_departure_lifts_the_urgent_band(self):
        tesla = _tesla()
        notes = apply_departure_target([tesla], "tesla", 9.0, 80.0, 43.0)
        assert notes
        top = min(tesla.incentive_buckets, key=lambda b: b.threshold_soc)
        assert top.threshold_soc == 80.0
        assert top.value_sek == 2.5, "the urgent VALUE must not change"

    def test_departure_beyond_horizon_is_inert(self):
        tesla = _tesla()
        assert apply_departure_target([tesla], "tesla", 50.0, 80.0, 43.0) == []
        assert tesla.incentive_buckets[0].threshold_soc == 40.0

    def test_past_departure_is_inert(self):
        tesla = _tesla()
        assert apply_departure_target([tesla], "tesla", -2.0, 80.0, 43.0) == []
        assert apply_departure_target([tesla], "tesla", None, 80.0, 43.0) == []

    def test_target_below_existing_band_is_inert(self):
        tesla = _tesla()
        assert apply_departure_target([tesla], "tesla", 9.0, 35.0, 43.0) == []
        assert tesla.incentive_buckets[0].threshold_soc == 40.0

    def test_missing_target_is_inert(self):
        tesla = _tesla()
        assert apply_departure_target([tesla], "tesla", 9.0, None, 43.0) == []

    def test_unknown_charger_is_inert(self):
        tesla = _tesla()
        assert apply_departure_target([tesla], "polestar", 9.0, 80.0, 43.0) == []
