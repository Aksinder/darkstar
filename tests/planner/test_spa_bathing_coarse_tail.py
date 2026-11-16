"""Coarse tail x spa bathing: the per-slot temperature is an END-of-slot state.

expand_result() puts an hourly tail back on the 15-minute grid. Like soc_kwh, the
predicted spa temperature must be interpolated from the previous coarse slot's
end value, not repeated (a flat hour would hide the ramp the executor rides).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from planner.coarse_tail import coarsen_slots, expand_result
from planner.solver.types import KeplerInputSlot, KeplerResult, KeplerResultSlot

BASE = datetime(2026, 1, 15, 6, 0)


def _quarters(n: int) -> list[KeplerInputSlot]:
    return [
        KeplerInputSlot(
            start_time=BASE + timedelta(minutes=15 * i),
            end_time=BASE + timedelta(minutes=15 * (i + 1)),
            load_kwh=0.1,
            pv_kwh=0.0,
            import_price_sek_kwh=1.0,
            export_price_sek_kwh=0.4,
        )
        for i in range(n)
    ]


def _result_slot(s: KeplerInputSlot, temps: dict[str, float]) -> KeplerResultSlot:
    return KeplerResultSlot(
        start_time=s.start_time,
        end_time=s.end_time,
        charge_kwh=0.0,
        discharge_kwh=0.0,
        grid_import_kwh=0.0,
        grid_export_kwh=0.0,
        soc_kwh=4.0,
        cost_sek=0.0,
        water_heater_temp_c=temps,
    )


def test_expand_interpolates_from_the_previous_end_value():
    originals = _quarters(12)  # 06:00-09:00
    coarse, groups = coarsen_slots(originals, fine_hours=1.0)  # 4 fine + 2 hourly
    assert [len(g) for g in groups] == [1, 1, 1, 1, 4, 4]
    temps = [30.0, 30.5, 31.0, 31.5, 33.5, 32.5]
    result = KeplerResult(
        slots=[_result_slot(c, {"spa": t}) for c, t in zip(coarse, temps, strict=True)],
        total_cost_sek=0.0,
        is_optimal=True,
        status_msg="Optimal",
    )
    out = expand_result(result, groups, originals)
    got = [s.water_heater_temp_c["spa"] for s in out.slots]
    assert got[:4] == temps[:4]  # fine head: unchanged
    assert got[4:8] == pytest.approx([32.0, 32.5, 33.0, 33.5])  # 31.5 -> 33.5
    assert got[8:12] == pytest.approx([33.25, 33.0, 32.75, 32.5])  # 33.5 -> 32.5
    # Each member owns its dict (no aliasing across the split).
    out.slots[4].water_heater_temp_c["spa"] = 0.0
    assert out.slots[5].water_heater_temp_c["spa"] == pytest.approx(32.5)


def test_first_coarse_slot_copies_and_absent_heaters_stay_absent():
    originals = _quarters(8)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]  # an all-hourly grid
    hour0 = KeplerInputSlot(BASE, BASE + timedelta(hours=1), 0.4, 0.0, 1.0, 0.4)
    hour1 = KeplerInputSlot(hour0.end_time, hour0.end_time + timedelta(hours=1), 0.4, 0.0, 1.0, 0.4)
    result = KeplerResult(
        slots=[_result_slot(hour0, {"spa": 31.0}), _result_slot(hour1, {})],
        total_cost_sek=0.0,
        is_optimal=True,
        status_msg="Optimal",
    )
    out = expand_result(result, groups, originals)
    # No previous slot: the end value is copied.
    assert [s.water_heater_temp_c["spa"] for s in out.slots[:4]] == [31.0] * 4
    assert all(s.water_heater_temp_c == {} for s in out.slots[4:])


def test_no_bathing_keeps_empty_dicts():
    originals = _quarters(8)
    coarse, groups = coarsen_slots(originals, fine_hours=1.0)
    result = KeplerResult(
        slots=[_result_slot(c, {}) for c in coarse],
        total_cost_sek=0.0,
        is_optimal=True,
        status_msg="Optimal",
    )
    out = expand_result(result, groups, originals)
    assert len(out.slots) == 8
    assert all(s.water_heater_temp_c == {} for s in out.slots)
