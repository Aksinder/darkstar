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


async def _ticks(ctl, ccfg, start, end, dep_ts, step=300.0, **state_over):
    """Tick every ``step`` s over [start, end] (the servo observing the car)."""
    t = start
    st = None
    while t <= end:
        st = await _read(ctl, ccfg, t, dep_ts, **state_over)
        t += step
    return st


def _seen_since(ctl, since, last_seen):
    """The car was observed plugged in at home continuously over [since, last_seen]."""
    ctl._present_run["tesla"] = (since, last_seen)


@pytest.mark.asyncio
async def test_missed_departure_below_floor_is_urgent():
    ctl, ccfg = _ctl()
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 3600)  # departed 04:00, soc 30 < 40
    assert st.deadline_hours == pytest.approx(0.25)
    assert st.floor_soc_percent == 40.0


@pytest.mark.asyncio
async def test_missed_departure_uses_trip_target():
    ctl, ccfg = _ctl(trip_target=True)
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
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
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    st = await _read(ctl, ccfg, SAT_05, dep, **{"sensor.tesla_soc": "41"})
    assert st.deadline_hours == pytest.approx(50.5, abs=0.01)
    # ...for good: a later dip below (e.g. a noisy reading) does not re-arm it.
    st = await _read(ctl, ccfg, SAT_05 + 600, dep, **{"sensor.tesla_soc": "39"})
    assert st.deadline_hours == pytest.approx(50.5 - 600 / 3600, abs=0.01)


@pytest.mark.asyncio
async def test_unplug_after_departure_ends_grace_for_good():
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    assert (await _read(ctl, ccfg, SAT_05, dep)).deadline_hours == pytest.approx(0.25)
    st = await _read(ctl, ccfg, SAT_05 + 60, dep, **{"binary_sensor.tesla_plug": "off"})
    assert st.deadline_hours != pytest.approx(0.25)
    # Car back and plugged inside the window: the trip happened, the goal is gone.
    st = await _read(ctl, ccfg, SAT_05 + 1800, dep)
    assert st.deadline_hours == pytest.approx(50.5 - 0.5, abs=0.01)


@pytest.mark.asyncio
async def test_short_plug_dropout_holds_the_grace_then_pauses_then_resumes():
    # A held "plugged" is not an observation, but one unreadable tick must not
    # stop the charger: the active grace rides through 180 s, pauses beyond that
    # (no full power at a car nobody can see) and is never ended by it.
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    assert (await _read(ctl, ccfg, SAT_05, dep)).deadline_hours == pytest.approx(0.25)
    dark = {"binary_sensor.tesla_plug": "unavailable"}
    for dt in (60, 120, 180):
        st = await _read(ctl, ccfg, SAT_05 + dt, dep, **dark)
        assert st.deadline_hours == pytest.approx(0.25), dt
    st = await _read(ctl, ccfg, SAT_05 + 240, dep, **dark)
    assert st.deadline_hours != pytest.approx(0.25)
    assert st.plugged is True  # the hold itself is unchanged: the car stays in the fleet
    assert "tesla" not in ctl._missed_dep_done
    st = await _read(ctl, ccfg, SAT_05 + 300, dep)
    assert st.deadline_hours == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_flapping_plug_sensor_does_not_churn_and_warns_once(caplog):
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    with caplog.at_level("INFO", logger="darkstar.ev_surplus"):
        for i in range(20):
            over = {"binary_sensor.tesla_plug": "unavailable"} if i % 2 else {}
            st = await _read(ctl, ccfg, SAT_05 + 60 * i, dep, **over)
            assert st.deadline_hours == pytest.approx(0.25), i
    armed = [r for r in caplog.records if "missed its one-off departure" in r.getMessage()]
    assert len(armed) == 1


@pytest.mark.asyncio
async def test_pause_and_resume_is_one_episode_one_warning(caplog):
    ctl, ccfg = _ctl()
    dep = SAT_05 - 3600
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    with caplog.at_level("INFO", logger="darkstar.ev_surplus"):
        await _read(ctl, ccfg, SAT_05, dep)
        await _ticks(ctl, ccfg, SAT_05 + 60, SAT_05 + 600, dep, step=60,
                     **{"binary_sensor.tesla_plug": "unavailable"})
        st = await _read(ctl, ccfg, SAT_05 + 660, dep)
    assert st.deadline_hours == pytest.approx(0.25)
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("missed its one-off departure" in m for m in msgs) == 1
    assert sum("grace paused" in m for m in msgs) == 1
    assert sum("grace resumed" in m for m in msgs) == 1


@pytest.mark.asyncio
async def test_home_unavailable_counts_as_unobserved_too():
    raw = _cfg_dict()
    tesla = next(c for c in raw["ev_surplus"]["chargers"] if c["id"] == "tesla")
    tesla["home_entity"] = "device_tracker.tesla"
    cfg = parse_ev_surplus_config(raw)
    assert cfg is not None
    ctl = EVSurplusController(cfg)
    ccfg = next(c for c in cfg.chargers if c.id == "tesla")
    dep = SAT_05 - 3600
    home = {"device_tracker.tesla": "home"}
    await _ticks(ctl, ccfg, dep - 1200, SAT_05, dep, **home)
    assert ctl._missed_dep_active == {"tesla"}
    st = await _ticks(ctl, ccfg, SAT_05 + 60, SAT_05 + 300, dep, step=60,
                      **{"device_tracker.tesla": "unknown"})
    assert st.deadline_hours != pytest.approx(0.25)
    st = await _read(ctl, ccfg, SAT_05 + 360, dep, **home)
    assert st.deadline_hours == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_held_presence_does_not_extend_the_observed_run():
    ctl, ccfg = _ctl()
    dep = SAT_05 + 3600
    await _ticks(ctl, ccfg, SAT_05 - 600, SAT_05, dep)
    assert ctl._present_run["tesla"] == (SAT_05 - 600, SAT_05)
    await _ticks(ctl, ccfg, SAT_05 + 300, SAT_05 + 1800, dep,
                 **{"binary_sensor.tesla_plug": "unavailable"})
    assert ctl._present_run["tesla"] == (SAT_05 - 600, SAT_05)


@pytest.mark.asyncio
async def test_car_left_during_a_blackout_across_the_departure_never_arms():
    # Plug entity goes dark at 06:30 holding "plugged"; the car leaves unseen and
    # the 07:30 departure passes. The held reading must not arm the grace, and the
    # first fresh reading (unplugged) ends it.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    dark = {"binary_sensor.tesla_plug": "unavailable"}
    await _ticks(ctl, ccfg, SAT_05, t0730 - 3600, t0730)  # seen until 06:30
    st = await _ticks(ctl, ccfg, t0730 - 3300, t0730 + 3600, t0730, **dark)
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_active == set()
    st = await _read(ctl, ccfg, t0730 + 3900, t0730, **{"binary_sensor.tesla_plug": "off"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == t0730


DARK = {"binary_sensor.tesla_plug": "unavailable"}


@pytest.mark.asyncio
async def test_outage_straddling_the_departure_still_gets_the_grace():
    # Reviewer scenario A: plug entity dark from 20 min before to 20 min after the
    # departure, the car plugged in the whole time. Charger/cloud outages around
    # the departure are what the grace is for.
    ctl, ccfg = _ctl()
    dep = SAT_05 + 3600
    await _ticks(ctl, ccfg, SAT_05, dep - 1200, dep, step=60)
    st = await _ticks(ctl, ccfg, dep - 1140, dep + 1200, dep, step=60, **DARK)
    assert st.deadline_hours != pytest.approx(0.25)  # not at a car nobody can see
    assert ctl._missed_dep_done == {}
    st = await _read(ctl, ccfg, dep + 1260, dep)
    assert st.deadline_hours == pytest.approx(0.25)
    assert ctl._missed_dep_done == {}


@pytest.mark.asyncio
async def test_outage_during_an_active_grace_does_not_end_it():
    # Reviewer scenario B: 16-min outage in the middle of an active grace.
    ctl, ccfg = _ctl()
    dep = SAT_05 + 3600
    await _ticks(ctl, ccfg, SAT_05, dep + 600, dep, step=60)
    assert ctl._missed_dep_active == {"tesla"}
    st = await _ticks(ctl, ccfg, dep + 660, dep + 1560, dep, step=60, **DARK)
    assert st.deadline_hours != pytest.approx(0.25)  # paused past the 180 s hold
    assert ctl._missed_dep_done == {}
    st = await _read(ctl, ccfg, dep + 1620, dep)
    assert st.deadline_hours == pytest.approx(0.25)
    assert ctl._missed_dep_done == {}


@pytest.mark.asyncio
async def test_long_blackout_is_bridged_when_the_soc_did_not_drop():
    # Dark 07:40-09:00, then plugged + home with the SoC where it was (or higher:
    # the car may have charged meanwhile).
    for soc_after in ("30", "29", "35"):
        ctl, ccfg = _ctl()
        t0730 = SAT_05 + 2.5 * 3600
        await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
        await _ticks(ctl, ccfg, t0730 + 900, t0730 + 5100, t0730, **DARK)
        st = await _read(ctl, ccfg, t0730 + 5400, t0730, **{"sensor.tesla_soc": soc_after})
        assert st.deadline_hours == pytest.approx(0.25), soc_after
        assert ctl._present_run["tesla"][0] == SAT_05


@pytest.mark.asyncio
async def test_left_and_came_back_with_lower_soc_gets_no_grace():
    # Same blackout, but the car comes back 5 points lower: it drove. No grace,
    # and for good.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 5100, t0730, **DARK)
    st = await _read(ctl, ccfg, t0730 + 5400, t0730, **{"sensor.tesla_soc": "25"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == t0730
    st = await _read(ctl, ccfg, t0730 + 5700, t0730, **{"sensor.tesla_soc": "25"})
    assert st.deadline_hours != pytest.approx(0.25)


@pytest.mark.asyncio
async def test_left_before_the_departure_during_a_blackout_and_returned_lower():
    ctl, ccfg = _ctl()
    dep = SAT_05 + 3600
    await _ticks(ctl, ccfg, SAT_05, dep - 1200, dep, step=60)
    await _ticks(ctl, ccfg, dep - 1140, dep + 1200, dep, step=60, **DARK)
    st = await _read(ctl, ccfg, dep + 1260, dep, **{"sensor.tesla_soc": "22"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == dep


@pytest.mark.asyncio
async def test_no_bridge_without_a_fresh_soc_but_not_terminal_either():
    # Plug readable again while the car's SoC is still a held value: there is
    # nothing to judge the blackout by yet. Stay paused; decide on a fresh SoC.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 5100, t0730, **DARK)
    st = await _ticks(ctl, ccfg, t0730 + 5400, t0730 + 6600, t0730,
                      **{"sensor.tesla_soc": "unavailable"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done == {}
    st = await _read(ctl, ccfg, t0730 + 6900, t0730)
    assert st.deadline_hours == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_no_bridge_without_a_fresh_soc_then_lower_soc_ends_it():
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 5100, t0730, **DARK)
    await _ticks(ctl, ccfg, t0730 + 5400, t0730 + 6600, t0730,
                 **{"sensor.tesla_soc": "unavailable"})
    st = await _read(ctl, ccfg, t0730 + 6900, t0730, **{"sensor.tesla_soc": "20"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == t0730


@pytest.mark.asyncio
async def test_unplugged_reading_inside_a_blackout_still_ends_the_grace():
    # Dark, one fresh "unplugged", dark again, then plugged with the same SoC.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 1800, t0730, **DARK)
    await _read(ctl, ccfg, t0730 + 2100, t0730, **{"binary_sensor.tesla_plug": "off"})
    await _ticks(ctl, ccfg, t0730 + 2400, t0730 + 3600, t0730, **DARK)
    st = await _read(ctl, ccfg, t0730 + 3900, t0730)
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == t0730


@pytest.mark.asyncio
async def test_blackout_does_not_extend_the_four_hour_window():
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 + 600, t0730)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 4 * 3600, t0730, **DARK)
    st = await _read(ctl, ccfg, t0730 + 4 * 3600 + 300, t0730)
    assert st.deadline_hours != pytest.approx(0.25)


# --- near-target tolerance (post-departure grace only) ---------------------


def _ctl_tol(tolerance, trip_target: bool = False):
    raw = _cfg_dict()
    if tolerance is not None:
        raw["ev_surplus"]["missed_departure_tolerance_percent"] = tolerance
    if trip_target:
        tesla = next(c for c in raw["ev_surplus"]["chargers"] if c["id"] == "tesla")
        tesla["departure_target_entity"] = "input_number.trip"
    cfg = parse_ev_surplus_config(raw)
    assert cfg is not None
    return EVSurplusController(cfg), next(c for c in cfg.chargers if c.id == "tesla")


@pytest.mark.parametrize(
    ("tolerance", "soc", "grace"),
    [
        (None, "37", True),     # default 2: 3 points short -> catch up
        (None, "37.9", True),
        (None, "38", False),    # exactly at target - tolerance -> no grace
        (None, "39", False),    # the 1-2 points-under car
        (None, "40", False),
        (0, "39.9", True),      # 0 = legacy: only at/above the target
        (0, "40", False),
        (5, "34.9", True),
        (5, "35", False),
    ],
)
@pytest.mark.asyncio
async def test_near_target_tolerance_boundary(tolerance, soc, grace):
    ctl, ccfg = _ctl_tol(tolerance)
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 3600, **{"sensor.tesla_soc": soc})
    assert (st.deadline_hours == pytest.approx(0.25)) is grace


@pytest.mark.asyncio
async def test_tolerance_applies_to_the_trip_target():
    ctl, ccfg = _ctl_tol(None, trip_target=True)
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    over = {"sensor.tesla_soc": "78", "input_number.trip": "80"}
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 3600, **over)
    assert st.deadline_hours != pytest.approx(0.25)


@pytest.mark.asyncio
async def test_within_tolerance_ends_the_grace_for_good():
    ctl, ccfg = _ctl_tol(None)
    dep = SAT_05 - 3600
    _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
    assert (await _read(ctl, ccfg, SAT_05, dep)).deadline_hours == pytest.approx(0.25)
    st = await _read(ctl, ccfg, SAT_05 + 60, dep, **{"sensor.tesla_soc": "38"})
    assert st.deadline_hours != pytest.approx(0.25)
    assert ctl._missed_dep_done.get("tesla") == dep
    st = await _read(ctl, ccfg, SAT_05 + 120, dep, **{"sensor.tesla_soc": "36"})
    assert st.deadline_hours != pytest.approx(0.25)


@pytest.mark.parametrize("tolerance", [None, 0, 5, 50])
@pytest.mark.asyncio
async def test_tolerance_never_touches_a_future_departure(tolerance):
    # Departure always wins BEFORE the deadline: one point short is still pursued
    # all the way to the target.
    ctl, ccfg = _ctl_tol(tolerance)
    st = await _read(ctl, ccfg, SAT_05, SAT_05 + 1800, **{"sensor.tesla_soc": "39"})
    assert st.deadline_hours == pytest.approx(0.5)
    assert st.floor_soc_percent == 40.0


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("absent", 2.0), (None, 2.0), ("", 2.0), ("abc", 2.0), (-1, 2.0),
        (float("nan"), 2.0), (float("inf"), 2.0),
        (0, 0.0), ("3", 3.0), (5, 5.0), (1.5, 1.5),
    ],
)
def test_tolerance_config_parsing(raw_value, expected):
    raw = _cfg_dict()
    if raw_value != "absent":
        raw["ev_surplus"]["missed_departure_tolerance_percent"] = raw_value
    cfg = parse_ev_surplus_config(raw)
    assert cfg is not None
    assert cfg.missed_departure_tolerance_percent == expected


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


@pytest.mark.asyncio
async def test_not_seen_at_departure_never_arms():
    # Cold start (or a long gap) after the departure: the car may have made its trip.
    ctl, ccfg = _ctl()
    st = await _read(ctl, ccfg, SAT_05, SAT_05 - 3600)
    assert st.deadline_hours == pytest.approx(50.5, abs=0.01)


@pytest.mark.asyncio
async def test_left_after_departure_and_returned_is_not_missed():
    # Departure 07:30; the car leaves 07:45 and is plugged back in at 09:00.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    await _ticks(ctl, ccfg, SAT_05, t0730 - 60, t0730)  # plugged at home before it
    st = await _read(ctl, ccfg, t0730 + 300, t0730)  # still here after: grace runs
    assert st.deadline_hours == pytest.approx(0.25)
    await _ticks(ctl, ccfg, t0730 + 900, t0730 + 3600, t0730,
                 **{"binary_sensor.tesla_plug": "off"})
    st = await _read(ctl, ccfg, t0730 + 5400, t0730)  # 09:00, back in the cable
    assert st.deadline_hours != pytest.approx(0.25)


@pytest.mark.asyncio
async def test_left_unseen_during_restart_is_not_missed():
    # Seen plugged until 07:40, then nothing until 09:00 (restart / stalled loop):
    # the car may have left and returned, so the grace must not arm.
    ctl, ccfg = _ctl()
    t0730 = SAT_05 + 2.5 * 3600
    _seen_since(ctl, SAT_05, t0730 + 600)
    st = await _read(ctl, ccfg, t0730 + 5400, t0730)
    assert st.deadline_hours != pytest.approx(0.25)


@pytest.mark.asyncio
async def test_grace_end_survives_restart(tmp_path):
    from backend.core import ev_hold_store

    ev_hold_store.enable(tmp_path / "holds.json")
    try:
        ctl, ccfg = _ctl()
        dep = SAT_05 - 3600
        _seen_since(ctl, SAT_05 - 7200, SAT_05 - 60)
        assert (await _read(ctl, ccfg, SAT_05, dep)).deadline_hours == pytest.approx(0.25)
        await _read(ctl, ccfg, SAT_05 + 60, dep, **{"binary_sensor.tesla_plug": "off"})
        # Restart; the car is back in its cable within the grace window and the
        # restored presence run would otherwise be continuous.
        ctl2, ccfg2 = _ctl()
        assert ctl2._missed_dep_done.get("tesla") == dep
        _seen_since(ctl2, SAT_05 - 7200, SAT_05 + 120)
        st = await _read(ctl2, ccfg2, SAT_05 + 180, dep)
        assert st.deadline_hours != pytest.approx(0.25)
    finally:
        ev_hold_store.disable()


@pytest.mark.asyncio
async def test_grace_continues_across_a_short_restart(tmp_path):
    from backend.core import ev_hold_store

    ev_hold_store.enable(tmp_path / "holds.json")
    try:
        ctl, ccfg = _ctl()
        dep = SAT_05 - 3600
        await _ticks(ctl, ccfg, dep - 1200, SAT_05, dep)
        assert ctl._missed_dep_active == {"tesla"}
        ctl2, ccfg2 = _ctl()  # restarted ~2 min later, the run is restored
        assert ctl2._present_run["tesla"][0] == dep - 1200
        st = await _read(ctl2, ccfg2, SAT_05 + 120, dep)
        assert st.deadline_hours == pytest.approx(0.25)
    finally:
        ev_hold_store.disable()
