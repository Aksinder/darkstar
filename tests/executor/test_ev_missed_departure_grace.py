"""A missed ONE-OFF departure stays urgent for a bounded grace window.

The car is still plugged in and short of its trip target, so the trip has not
happened yet (charger/cloud outage, late plug-in). Upstream 5440027b idea. The
recurring weekday deadline is unchanged.
"""

from __future__ import annotations

import pytest

from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import FakeHA, _cfg_dict, _states, _ts

DEP = "input_datetime.tesla_departure"
# Saturday: the weekday recurrence (Mon 07:30) is ~50 h away and lazy.
SAT_05 = _ts(2026, 8, 15, 5, 0)


def _ctl(trip_target: bool = False) -> tuple[EVSurplusController, object]:
    raw = _cfg_dict()
    if trip_target:
        tesla = next(c for c in raw["ev_surplus"]["chargers"] if c["id"] == "tesla")
        tesla["departure_target_entity"] = "input_number.trip"
    cfg = parse_ev_surplus_config(raw)
    assert cfg is not None
    ctl = EVSurplusController(cfg)
    return ctl, next(c for c in cfg.chargers if c.id == "tesla")


async def _read(ctl, ccfg, now, dep_ts, **state_over):
    ha = FakeHA(_states(**state_over), {DEP: {"timestamp": dep_ts}})
    return await ctl._read_charger(ha, ccfg, now, False)


@pytest.mark.asyncio
async def test_missed_departure_below_floor_is_urgent():
    ctl, ccfg = _ctl()
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 3600)  # departed 04:00, soc 30 < 40
    assert st.deadline_hours == pytest.approx(0.25)
    assert st.floor_soc_percent == 40.0


@pytest.mark.asyncio
async def test_missed_departure_uses_trip_target():
    ctl, ccfg = _ctl(trip_target=True)
    st = await _read(
        ctl, ccfg, SAT_05, SAT_05 - 3600,
        **{"sensor.tesla_soc": "60", "input_number.trip": "80"},
    )
    assert st.deadline_hours == pytest.approx(0.25)
    assert st.floor_soc_percent == 80.0


@pytest.mark.asyncio
async def test_grace_expires_after_four_hours():
    ctl, ccfg = _ctl()
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 4 * 3600 - 60)
    assert st.deadline_hours == pytest.approx(50.5, abs=0.01)  # recurrence only


@pytest.mark.asyncio
async def test_target_reached_ends_grace():
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    st = await _read(ctl, ccfg, SAT_05, dep, **{"sensor.tesla_soc": "41"})
    assert st.deadline_hours == pytest.approx(50.5, abs=0.01)
    # ...for good: a later dip below (e.g. a noisy reading) does not re-arm it.
    st = await _read(ctl, ccfg, SAT_05 + 600, dep, **{"sensor.tesla_soc": "39"})
    assert st.deadline_hours == pytest.approx(50.5 - 600 / 3600, abs=0.01)


@pytest.mark.asyncio
async def test_unplug_after_departure_ends_grace_for_good():
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    assert (await _read(ctl, ccfg, SAT_05, dep)).deadline_hours == pytest.approx(0.25)
    st = await _read(ctl, ccfg, SAT_05 + 60, dep, **{"binary_sensor.tesla_plug": "off"})
    assert st.deadline_hours != pytest.approx(0.25)
    # Car back and plugged inside the window: the trip happened, the goal is gone.
    st = await _read(ctl, ccfg, SAT_05 + 1800, dep)
    assert st.deadline_hours == pytest.approx(50.5 - 0.5, abs=0.01)


@pytest.mark.asyncio
async def test_plug_unavailable_holds_and_keeps_grace():
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    await _read(ctl, ccfg, SAT_05, dep)
    st = await _read(ctl, ccfg, SAT_05 + 60, dep, **{"binary_sensor.tesla_plug": "unavailable"})
    assert st.deadline_hours == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_new_departure_starts_fresh():
    ctl, ccfg = _ctl()
    await _read(ctl, ccfg, SAT_05, SAT_05 - 3600, **{"binary_sensor.tesla_plug": "off"})
    st = await _read(ctl, ccfg, SAT_05 + 60, SAT_05 + 1800)  # new future departure
    assert st.deadline_hours == pytest.approx((1800 - 60) / 3600, abs=0.01)


@pytest.mark.asyncio
async def test_vacation_still_clears_the_grace_deadline():
    ctl, ccfg = _ctl()
    ha = FakeHA(_states(), {DEP: {"timestamp": SAT_05 - 3600}})
    st = await ctl._read_charger(ha, ccfg, SAT_05, True)
    assert st.deadline_hours is None
