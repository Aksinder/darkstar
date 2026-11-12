"""``mode: window`` cyclic loads (room heaters) in the planner, and control pause
for every cyclic load.

A window load is not optimised: it is fixed load inside its windows. What must hold
is the bookkeeping — only in-window slots carry it, partial slots pro-rated — and
the schedule contract the executor relies on: a planned load has a value in EVERY
slot (non-zero inside, explicit 0.0 outside), an unplanned one is absent.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
import pytz

from backend.core.load_windows import windows_to_iso
from planner.cyclic_preschedule import (
    CyclicSpec,
    PriceSlot,
    preschedule_cyclic_loads,
    window_slot_kw,
)
from planner.pipeline import (
    PlannerPipeline,
    book_cyclic_fixed_load,
    merge_cyclic_into_schedule,
)
from planner.solver.types import KeplerInputSlot

TZ = pytz.timezone("Europe/Stockholm")
T0 = TZ.localize(datetime(2026, 11, 13, 12, 0))
Q = timedelta(minutes=15)


def _slots(n: int = 96, start: datetime = T0, price: float = 1.0) -> list[KeplerInputSlot]:
    return [
        KeplerInputSlot(start + i * Q, start + (i + 1) * Q, 0.25, 0.0, price + (i % 7) * 0.1, 0.5)
        for i in range(n)
    ]


def _bounds(slots: list[KeplerInputSlot]) -> list[tuple[datetime, datetime]]:
    return [(s.start_time, s.end_time) for s in slots]


def _preschedule(config: dict[str, Any], slots, states=None) -> dict[str, dict[int, float]]:
    # The method reads nothing from the instance.
    return PlannerPipeline._preschedule_cyclic_loads(None, config, slots, states)  # type: ignore[arg-type]


KIM = {
    "id": "kim_heater",
    "mode": "window",
    "switch_entity": "switch.kims_uttag",
    "power_kw": 1.0,
    "planned_power_factor": 0.5,
    "window_entities": ["calendar.kim"],
    "phase_map": ["b"],
}
POOL = {
    "id": "poolpump",
    "switch_entity": "switch.poolpump",
    "power_kw": 0.26,
    "min_kwh_per_day": 0.78,
    "max_hours_between": 24,
}
BOG = {
    "id": "bogfilter",
    "switch_entity": "switch.bogfilter",
    "power_kw": 0.39,
    "min_kwh_per_day": 2.34,
    "max_hours_between": 6,
    "phase_map": ["c"],
}


def _windows(*pairs: tuple[datetime, datetime]) -> list[list[str]]:
    return windows_to_iso(list(pairs))


class TestWindowSlotKw:
    def test_only_slots_inside_the_window_carry_load(self):
        slots = _slots(16)
        plan = window_slot_kw(_bounds(slots), [(T0 + 4 * Q, T0 + 8 * Q)], 0.5)
        assert plan == {4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5}

    def test_partially_covered_slots_are_pro_rated(self):
        """Window 13:05-13:50: 10 of 15 min in the first slot, 5 of 15 in the last."""
        slots = _slots(16)
        start = T0 + timedelta(minutes=65)
        plan = window_slot_kw(_bounds(slots), [(start, start + timedelta(minutes=45))], 0.6)
        assert plan == pytest.approx({4: 0.4, 5: 0.6, 6: 0.6, 7: 0.2})

    def test_a_window_that_began_before_the_horizon_covers_slot_0_fully(self):
        slots = _slots(8)
        plan = window_slot_kw(_bounds(slots), [(T0 - timedelta(days=3), T0 + 2 * Q)], 0.5)
        assert plan == {0: 0.5, 1: 0.5}

    def test_a_window_past_the_horizon_edge_stops_at_the_last_slot(self):
        slots = _slots(8)
        plan = window_slot_kw(_bounds(slots), [(T0 + 6 * Q, T0 + timedelta(days=9))], 0.5)
        assert plan == {6: 0.5, 7: 0.5}

    def test_two_windows_in_one_slot_add_up_but_never_exceed_the_slot(self):
        slots = _slots(2)
        plan = window_slot_kw(
            _bounds(slots),
            [(T0, T0 + timedelta(minutes=5)), (T0 + timedelta(minutes=10), T0 + Q)],
            0.9,
        )
        assert plan == pytest.approx({0: 0.6})

    def test_windows_in_utc_meet_slots_in_local_time(self):
        """The same instants in another zone: aware datetimes compare as instants."""
        slots = _slots(8)
        as_utc = [((T0 + 2 * Q).astimezone(UTC), (T0 + 4 * Q).astimezone(UTC))]
        assert window_slot_kw(_bounds(slots), as_utc, 0.5) == {2: 0.5, 3: 0.5}

    def test_dst_night_slots_are_matched_by_instant_not_wall_clock(self):
        """25 October 2026: 02:00-03:00 happens twice. A window ending at the SECOND
        02:30 (CET) must cover the first 02:xx hour (CEST) entirely."""
        start = TZ.localize(datetime(2026, 10, 25, 1, 0), is_dst=True).astimezone(UTC)
        slots = _slots(16, start=start)  # 01:00 CEST .. 04:00 CET, in UTC
        end = TZ.localize(datetime(2026, 10, 25, 2, 30), is_dst=False)
        plan = window_slot_kw(_bounds(slots), [(start - Q, end)], 0.5)
        assert sorted(plan) == list(range(10))  # 2.5 real hours
        assert sum(plan.values()) * 0.25 == pytest.approx(0.5 * 2.5)

    def test_naive_slots_are_supported_like_the_pump_scheduler(self):
        naive_start = datetime(2026, 11, 13, 12, 0)
        bounds = [(naive_start + i * Q, naive_start + (i + 1) * Q) for i in range(8)]
        local = naive_start.astimezone()  # the same wall clock, system zone
        plan = window_slot_kw(bounds, [(local + 2 * Q, local + 4 * Q)], 0.5)
        assert plan == {2: 0.5, 3: 0.5}

    @pytest.mark.parametrize("kw", [0.0, -1.0])
    def test_no_power_no_plan(self, kw):
        assert window_slot_kw(_bounds(_slots(4)), [(T0, T0 + Q)], kw) == {}


class TestWindowLoadInThePlan:
    def _plan(self, load=KIM, windows=None, slots=None, **state):
        slots = slots or _slots()
        st = {"id": load["id"], "windows": windows, **state}
        return _preschedule({"cyclic_loads": [load]}, slots, [st]), slots

    def test_books_nameplate_times_the_planned_factor_inside_the_window(self):
        plan, slots = self._plan(windows=_windows((T0 + 8 * Q, T0 + 16 * Q)))
        assert plan == {"kim_heater": dict.fromkeys(range(8, 16), 0.5)}
        book_cyclic_fixed_load(slots, plan, {"kim_heater": ["B"]})
        for i, s in enumerate(slots):
            inside = 8 <= i < 16
            assert s.load_kwh == pytest.approx(0.25 + (0.125 if inside else 0.0))
            assert s.phased_load_kwh == ({"B": pytest.approx(0.125)} if inside else None)

    def test_schedule_has_a_value_in_every_slot_nonzero_only_inside(self):
        plan, _ = self._plan(windows=_windows((T0 + 2 * Q, T0 + 4 * Q)), slots=_slots(6))
        rows = merge_cyclic_into_schedule([{"main_tank": 3.0}] + [{}] * 4 + [None], plan)
        assert [r["kim_heater"] for r in rows] == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        assert rows[0] == {"main_tank": 3.0, "kim_heater": 0.0}  # the tanks are kept
        assert all(isinstance(r["kim_heater"], float) for r in rows)

    def test_no_window_in_the_horizon_is_an_explicit_off_everywhere(self):
        plan, slots = self._plan(windows=[], slots=_slots(4))
        assert plan == {"kim_heater": {}}
        rows = merge_cyclic_into_schedule([{}] * 4, plan)
        assert rows == [{"kim_heater": 0.0}] * 4
        book_cyclic_fixed_load(slots, plan, {})
        assert all(s.load_kwh == 0.25 for s in slots)

    def test_unknown_windows_book_nothing_and_stay_out_of_the_schedule(self):
        """Calendar unreadable with no recent answer: absent, not an explicit 0 —
        a 0 would tell the executor's fallback "off on purpose"."""
        plan, _ = self._plan(windows=None, slots=_slots(4))
        assert plan == {}
        assert merge_cyclic_into_schedule([{}] * 4, plan) == [{}] * 4

    def test_a_state_entry_without_windows_at_all_is_unknown(self):
        assert _preschedule({"cyclic_loads": [KIM]}, _slots(4), []) == {}
        assert _preschedule({"cyclic_loads": [KIM]}, _slots(4), None) == {}

    def test_planned_power_factor_defaults_to_half_and_is_clamped(self):
        win = _windows((T0, T0 + Q))
        bare = {k: v for k, v in KIM.items() if k != "planned_power_factor"}
        assert self._plan(load=bare, windows=win)[0]["kim_heater"] == {0: 0.5}
        assert self._plan(load={**KIM, "planned_power_factor": 7}, windows=win)[0][
            "kim_heater"
        ] == {0: 1.0}
        assert self._plan(load={**KIM, "planned_power_factor": "x"}, windows=win)[0][
            "kim_heater"
        ] == {0: 0.5}

    @pytest.mark.parametrize("over", [{"power_kw": 0}, {"planned_power_factor": 0}])
    def test_zero_booked_power_leaves_the_load_out_rather_than_planning_it_off(self, over):
        """An in-window value of 0.0 would read as "off" to the executor's fallback."""
        plan, _ = self._plan(load={**KIM, **over}, windows=_windows((T0, T0 + Q)))
        assert plan == {}

    def test_disabled_and_mistyped_modes_are_left_out(self):
        win = _windows((T0, T0 + Q))
        assert self._plan(load={**KIM, "enabled": False}, windows=win)[0] == {}
        assert self._plan(load={**KIM, "mode": "windows"}, windows=win)[0] == {}

    def test_the_daily_need_keys_are_ignored_in_window_mode(self):
        load = {**KIM, "min_kwh_per_day": 9.0, "max_hours_between": 1}
        plan, _ = self._plan(load=load, windows=_windows((T0, T0 + 2 * Q)))
        assert plan == {"kim_heater": {0: 0.5, 1: 0.5}}

    def test_window_and_daily_loads_are_planned_side_by_side(self):
        slots = _slots()
        plan = _preschedule(
            {"cyclic_loads": [POOL, KIM]},
            slots,
            [{"id": "kim_heater", "windows": _windows((T0, T0 + 4 * Q))}],
        )
        assert plan["kim_heater"] == dict.fromkeys(range(4), 0.5)
        assert plan["poolpump"] == _preschedule({"cyclic_loads": [POOL]}, slots)["poolpump"]


class TestControlPauseReachesThePlanner:
    def test_a_paused_window_load_is_booked_zero_and_left_out(self):
        plan = _preschedule(
            {"cyclic_loads": [KIM]},
            _slots(8),
            [
                {
                    "id": "kim_heater",
                    "windows": _windows((T0, T0 + 8 * Q)),
                    "control_paused": True,
                    "control_paused_via": "input_boolean.kim_heater_paused",
                }
            ],
        )
        assert plan == {}

    def test_a_paused_classic_load_is_booked_zero_and_left_out(self):
        """Used to be planned anyway: control_pause_entities was read for the tanks
        only, so a paused pump reserved PV/import in the solver's arithmetic."""
        slots = _slots()
        free = _preschedule({"cyclic_loads": [POOL, BOG]}, slots)
        assert set(free) == {"poolpump", "bogfilter"}
        paused = _preschedule(
            {"cyclic_loads": [POOL, BOG]},
            slots,
            [{"id": "poolpump", "control_paused": True, "control_paused_via": "input_boolean.x"}],
        )
        assert paused == {"bogfilter": free["bogfilter"]}
        book_cyclic_fixed_load(slots, paused, {})
        rows = merge_cyclic_into_schedule([{}] * len(slots), paused)
        assert all("poolpump" not in r for r in rows)


def _reference_cyclic_plan(config, slots, states=None):
    """The pre-change pipeline code, verbatim (ef1d5943 planner/pipeline.py)."""
    by_id = {str(st.get("id")): st for st in (states or []) if st.get("id")}
    specs = []
    for c in config.get("cyclic_loads", []) or []:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        gap = c.get("max_hours_between")
        st = by_id.get(str(c["id"]), {})
        last_run_end = None
        raw_end = st.get("last_run_end")
        if raw_end:
            with contextlib.suppress(ValueError, TypeError):
                last_run_end = datetime.fromisoformat(str(raw_end).replace("Z", "+00:00"))
        specs.append(
            CyclicSpec(
                id=str(c["id"]),
                power_kw=float(c.get("power_kw", 0.0) or 0.0),
                min_kwh_per_day=float(c.get("min_kwh_per_day", 0.0) or 0.0),
                max_hours_between=float(gap) if gap is not None else None,
                heated_today_kwh=float(st.get("heated_today_kwh", 0.0) or 0.0),
                last_run_end=last_run_end,
                enabled=bool(c.get("enabled", True)),
            )
        )
    if not specs:
        return {}
    wh_cfg = config.get("water_heating", {}) or {}
    return preschedule_cyclic_loads(
        specs,
        [PriceSlot(s.start_time, float(s.import_price_sek_kwh)) for s in slots],
        defer_up_to_hours=float(wh_cfg.get("defer_up_to_hours", 0.0) or 0.0),
    )


def _reference_outputs(slots, plan, phases):
    """The two inline pipeline loops of ef1d5943, verbatim: (slots booked, schedule)."""
    for lid, by_slot in plan.items():
        phs = phases.get(lid) or []
        for t, kw in by_slot.items():
            if 0 <= t < len(slots):
                sl = slots[t]
                e_kwh = kw * ((sl.end_time - sl.start_time).total_seconds() / 3600.0)
                sl.load_kwh += e_kwh
                if phs:
                    known = dict(sl.phased_load_kwh or {})
                    for ph in phs:
                        known[ph] = known.get(ph, 0.0) + e_kwh / len(phs)
                    sl.phased_load_kwh = known
    merged = []
    for pos in range(len(slots)):
        entry = {"main_tank": 3.0} if pos % 5 == 0 else {}
        for load_id, by_slot in plan.items():
            entry[load_id] = float(by_slot.get(pos, 0.0))
        merged.append(entry)
    return slots, merged


EXISTING_CONFIG: dict[str, Any] = {
    "water_heating": {"defer_up_to_hours": 3},
    "cyclic_loads": [
        POOL,
        BOG,
        {**POOL, "id": "disabled_pump", "enabled": False},
        {"id": "", "power_kw": 1.0},
        "nonsense",
    ],
}
EXISTING_STATES = [
    {"id": "poolpump", "heated_today_kwh": 0.26, "last_run_end": "2026-11-13T09:11:00+00:00"},
    {"id": "bogfilter", "heated_today_kwh": 0.0, "last_run_end": None},
]


class TestExistingConfigsAreUntouched:
    """No ``mode: window`` load configured => plan, fixed load and schedule output
    are identical to before the feature existed."""

    @pytest.mark.parametrize("states", [None, [], EXISTING_STATES])
    def test_plan_fixed_load_and_schedule_are_identical(self, states):
        phases = {"bogfilter": ["C"]}
        want_plan = _reference_cyclic_plan(EXISTING_CONFIG, _slots(192), states)
        want_slots, want_schedule = _reference_outputs(_slots(192), want_plan, phases)

        got_plan = _preschedule(EXISTING_CONFIG, _slots(192), states)
        got_slots = _slots(192)
        book_cyclic_fixed_load(got_slots, got_plan, phases)
        got_schedule = merge_cyclic_into_schedule(
            [{"main_tank": 3.0} if i % 5 == 0 else {} for i in range(192)], got_plan
        )

        assert got_plan == want_plan
        assert set(got_plan) == {"poolpump", "bogfilter"}  # and something was planned
        assert sum(len(v) for v in got_plan.values()) > 0
        assert got_slots == want_slots
        assert got_schedule == want_schedule

    def test_no_cyclic_loads_is_still_an_empty_plan(self):
        assert _preschedule({}, _slots(8)) == {}
        assert _preschedule({"cyclic_loads": []}, _slots(8)) == {}


# --------------------------------------------------------------------------- HA side


def _initial_state_config(tmp_path, cyclic_loads):
    import yaml

    cfg = {
        "timezone": "Europe/Stockholm",
        "system": {"battery": {"capacity_kwh": 10.0}, "has_ev_charger": False},
        "input_sensors": {"battery_soc": "sensor.batt_soc"},
        "learning": {"sqlite_path": str(tmp_path / "none.db")},
        "cyclic_loads": cyclic_loads,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


async def _cyclic_states(tmp_path, cyclic_loads, states, windows=None):
    from backend.core.ha_client import get_initial_state

    async def entity_state(entity_id):
        if entity_id == "sensor.batt_soc":
            return {"state": "75.0"}
        value = states.get(entity_id)
        if isinstance(value, Exception):
            raise value
        return None if value is None else {"state": value, "last_changed": "2026-11-13T09:00:00+00:00"}

    planned: list[str] = []

    async def plan_windows(raw, now, tz, *, get_state):
        planned.append(str(raw["id"]))
        assert now.tzinfo is not None
        return windows

    with (
        patch("backend.core.ha_client.get_ha_entity_state", side_effect=entity_state),
        patch("backend.core.ha_client.plan_windows_for_load", side_effect=plan_windows),
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
    ):
        res = await get_initial_state(_initial_state_config(tmp_path, cyclic_loads))
    return {s["id"]: s for s in res["cyclic_load_states"]}, planned


PAUSE = "input_boolean.kim_heater_paused"


class TestInitialState:
    @pytest.mark.asyncio
    async def test_window_load_carries_its_windows_and_skips_the_gap_anchor(self, tmp_path):
        wins = [(T0, T0 + 4 * Q)]
        states, planned = await _cyclic_states(
            tmp_path, [KIM, POOL], {"switch.poolpump": "on", "switch.kims_uttag": "on"}, wins
        )
        assert planned == ["kim_heater"]
        assert states["kim_heater"]["windows"] == windows_to_iso(wins)
        assert states["kim_heater"]["last_run_end"] is None
        assert "windows" not in states["poolpump"]
        assert states["poolpump"]["last_run_end"] is not None

    @pytest.mark.asyncio
    async def test_unknown_windows_are_passed_on_as_unknown(self, tmp_path):
        states, _ = await _cyclic_states(tmp_path, [KIM], {}, None)
        assert states["kim_heater"]["windows"] is None

    @pytest.mark.asyncio
    async def test_only_a_definite_on_pauses(self, tmp_path):
        kim = {**KIM, "control_pause_entities": [PAUSE]}
        pool = {**POOL, "control_pause_entities": ["input_boolean.pool_paused"]}
        states, planned = await _cyclic_states(
            tmp_path, [kim, pool], {PAUSE: "on", "input_boolean.pool_paused": "on"}, []
        )
        assert states["kim_heater"]["control_paused"] is True
        assert states["kim_heater"]["control_paused_via"] == PAUSE
        assert states["poolpump"]["control_paused"] is True
        assert planned == []  # a paused heater's calendar is not even read

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["off", "unavailable", "unknown", None, RuntimeError("x")])
    async def test_a_read_error_is_not_a_pause(self, tmp_path, state):
        kim = {**KIM, "control_pause_entities": [PAUSE]}
        pool = {**POOL, "control_pause_entities": [PAUSE]}
        states, planned = await _cyclic_states(tmp_path, [kim, pool], {PAUSE: state}, [])
        assert "control_paused" not in states["kim_heater"]
        assert "control_paused" not in states["poolpump"]
        assert planned == ["kim_heater"]

    @pytest.mark.asyncio
    async def test_window_loads_are_not_credited_a_daily_need(self, tmp_path, caplog):
        """heated_today is clamped to min_kwh_per_day; a load without one would only
        add a "credited=0.000 (min_kwh=0.00)" line to every plan."""
        from backend.core.ha_client import get_water_heated_today_by_tank

        cfg = {"cyclic_loads": [KIM, POOL], "learning": {"sqlite_path": str(tmp_path / "x.db")}}
        with caplog.at_level("INFO"):
            credited = await get_water_heated_today_by_tank(cfg)
        assert "kim_heater" not in credited
        assert "poolpump" in credited
        assert "kim_heater" not in caplog.text


# --------------------------------------------------------------------- price rules

OWNER_RULES = {
    "early_start_hours": 14,
    "early_start_price_percentile": 30,
    "peak_off_price_percentile": 85,
    "peak_off_max_minutes": 180,
    "peak_off_min_on_minutes": 120,
}
THU_NOON = TZ.localize(datetime(2026, 11, 12, 12, 0))
FRI_15 = TZ.localize(datetime(2026, 11, 13, 15, 0))


def _two_day_price(hours_from_thu_noon: float) -> float:
    """Thursday 12:00 -> Saturday 12:00. A night valley, morning and evening peaks."""
    clock = (12 + hours_from_thu_noon) % 24
    day = int((12 + hours_from_thu_noon) // 24)  # 0 Thu, 1 Fri, 2 Sat
    if day == 1 and clock < 2:
        return 2.0  # Friday's first two hours: not cheap yet
    if 2 <= clock < 5:
        return 0.30 if day == 1 else 0.35  # the night valley
    if 6 <= clock < 9:
        return 3.5 if day == 1 else 3.3  # morning peak
    if 17 <= clock < 20:
        return 3.4 if day == 0 else 3.6  # evening peak
    return 1.5 + 0.01 * (int(clock) % 5)


def _two_day_slots() -> list[KeplerInputSlot]:
    return [
        KeplerInputSlot(
            THU_NOON + i * Q, THU_NOON + (i + 1) * Q, 0.25, 0.0, _two_day_price(i / 4), 0.5
        )
        for i in range(192)
    ]


def _slot_at(day: int, hour: float) -> int:
    """Slot index of a wall-clock hour; day 0 = Thursday."""
    return int((day * 24 + hour - 12) * 4)


def _ranges(plan: dict[int, float]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in sorted(plan):
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


class TestPriceRulesInThePlan:
    """Cheap early start and peak off, simulated forward on the forecast prices with
    the executor's own decision function, so the solver sees the load it will get."""

    def _plan(self, rules, *, state=None, slots=None, windows=None):
        slots = slots or _two_day_slots()
        st: dict[str, Any] = {
            "id": "kim_heater",
            "windows": windows or _windows((FRI_15, FRI_15 + timedelta(days=7))),
        }
        if state is not None:
            st["price_rule_state"] = state
        plan = _preschedule({"cyclic_loads": [{**KIM, **rules}]}, slots, [st])
        return plan["kim_heater"], slots

    def test_a_load_without_the_new_keys_is_booked_exactly_as_before(self):
        plan, slots = self._plan({})
        assert plan == window_slot_kw(
            _bounds(slots), [(FRI_15, FRI_15 + timedelta(days=7))], 0.5
        )
        assert _ranges(plan) == [(_slot_at(1, 15), 192)]
        # ...also with partial slots and with the rules' state published.
        odd = _windows((FRI_15 + timedelta(minutes=5), FRI_15 + timedelta(minutes=50)))
        plan, slots = self._plan({}, windows=odd, state={"early_start_latched": True})
        assert plan == pytest.approx(
            {_slot_at(1, 15): 0.5 * 10 / 15, _slot_at(1, 15) + 1: 0.5,
             _slot_at(1, 15) + 2: 0.5, _slot_at(1, 15) + 3: 0.5 * 5 / 15}
        )

    def test_the_owners_rules_over_two_days(self):
        plan, _ = self._plan(OWNER_RULES)
        assert _ranges(plan) == [
            # Friday 01:00 opens the 14 h span, but 2.0 is not cheap. The valley at
            # 02:00 latches it on — and it stays on at 1.5 after the valley.
            (_slot_at(1, 2), _slot_at(1, 6)),
            # 06:00-09:00 morning peak: off. The latch survives it.
            (_slot_at(1, 9), _slot_at(1, 17)),  # early start, then the window at 15:00
            # 17:00-20:00 evening peak, inside the window: off.
            (_slot_at(1, 20), _slot_at(2, 6)),
            # Saturday 06:00-09:00 morning peak: off.
            (_slot_at(2, 9), 192),
        ]
        assert set(plan.values()) == {0.5}
        # Thursday's evening peak and night are before the span: nothing booked.
        assert min(plan) == _slot_at(1, 2)

    def test_the_schedule_keeps_explicit_zeros_and_the_booking_follows(self):
        plan, slots = self._plan(OWNER_RULES)
        rows = merge_cyclic_into_schedule([{}] * 192, {"kim_heater": plan})
        assert all("kim_heater" in r for r in rows)
        assert rows[_slot_at(1, 3)]["kim_heater"] == 0.5  # early start: non-zero
        assert rows[_slot_at(1, 7)]["kim_heater"] == 0.0  # peak off before the window
        assert rows[_slot_at(1, 18)]["kim_heater"] == 0.0  # peak off INSIDE the window
        assert rows[_slot_at(1, 16)]["kim_heater"] == 0.5
        book_cyclic_fixed_load(slots, {"kim_heater": plan}, {})
        assert slots[_slot_at(1, 3)].load_kwh == pytest.approx(0.25 + 0.125)
        assert slots[_slot_at(1, 18)].load_kwh == pytest.approx(0.25)

    def test_early_start_only(self):
        plan, _ = self._plan({"early_start_hours": 14})
        assert _ranges(plan) == [(_slot_at(1, 2), 192)]  # latched, no peak rule

    def test_peak_off_only(self):
        plan, _ = self._plan({"peak_off_price_percentile": 85})
        assert _ranges(plan) == [
            (_slot_at(1, 15), _slot_at(1, 17)),
            (_slot_at(1, 20), _slot_at(2, 6)),
            (_slot_at(2, 9), 192),
        ]

    def test_a_long_peak_is_cut_at_the_maximum_and_min_on_is_honoured(self):
        """Peak 17:00-20:00 with max 60 / min-on 60: off, on, off — not 3 h cold."""
        plan, _ = self._plan(
            {"peak_off_price_percentile": 85, "peak_off_max_minutes": 60,
             "peak_off_min_on_minutes": 60}
        )
        assert _ranges(plan)[:3] == [
            (_slot_at(1, 15), _slot_at(1, 17)),
            (_slot_at(1, 18), _slot_at(1, 19)),
            (_slot_at(1, 20), _slot_at(2, 6)),
        ]

    def test_a_shorter_span_starts_later(self):
        """12 h before 15:00 is 03:00: the valley is still on, so it latches there."""
        plan, _ = self._plan({"early_start_hours": 12})
        assert min(plan) == _slot_at(1, 3)
        # 9 h: 06:00. The valley is over; "cheap" is relative to the window ahead,
        # so after the morning peak the cheapest of the ordinary hours (10:00)
        # latches it. A percentile always finds a cheapest third.
        plan, _ = self._plan({"early_start_hours": 9})
        assert min(plan) == _slot_at(1, 10)

    def test_the_plugs_published_state_seeds_the_simulation(self):
        """Planning at Friday 06:00, in the morning peak: the valley is over, but the
        executor's status sensor says the heater is latched on — so it is booked
        from slot 0, not from the next cheap hour."""
        slots = _two_day_slots()[_slot_at(1, 6):]
        cold, _ = self._plan({"early_start_hours": 14}, slots=slots)
        assert min(cold) == (10 - 6) * 4  # no memory: waits for a cheap hour
        seeded, _ = self._plan(
            {"early_start_hours": 14}, slots=slots,
            state={"early_start_latched": True, "peak_off_since": None,
                   "heat_since": "2026-11-13T02:00+01:00"},
        )
        assert min(seeded) == 0 and _ranges(seeded) == [(0, len(slots))]

    def test_a_running_peak_off_is_continued_not_restarted(self):
        """Planning at Friday 18:00, peak-off since 17:00: two more hours off, not
        a fresh three."""
        slots = _two_day_slots()[_slot_at(1, 18):]
        plan, _ = self._plan(
            {"peak_off_price_percentile": 85}, slots=slots,
            state={"early_start_latched": False,
                   "peak_off_since": "2026-11-13T17:00+01:00", "heat_since": None},
        )
        assert min(plan) == 2 * 4
        long_ago, _ = self._plan(
            {"peak_off_price_percentile": 85}, slots=slots,
            state={"early_start_latched": False,
                   "peak_off_since": "2026-11-13T15:00+01:00", "heat_since": None},
        )
        assert min(long_ago) == 0  # its 180 min are up

    def test_without_published_state_a_heater_in_a_window_is_taken_to_be_on(self):
        """No status sensor (first plan, executor down): book load at a peak rather
        than assume an off-stretch nobody commanded."""
        slots = _two_day_slots()[_slot_at(1, 17):]
        plan, _ = self._plan({"peak_off_price_percentile": 85}, slots=slots)
        assert _ranges(plan)[0] == (0, 2 * 4)  # on for the min-on time, then off

    def test_a_window_starting_mid_slot_is_a_whole_slot_when_already_latched(self):
        odd = _windows((FRI_15 + timedelta(minutes=5), FRI_15 + timedelta(days=7)))
        latched, _ = self._plan({"early_start_hours": 14}, windows=odd)
        assert latched[_slot_at(1, 15)] == 0.5
        plain, _ = self._plan({}, windows=odd)
        assert plain[_slot_at(1, 15)] == pytest.approx(0.5 * 10 / 15)

    def test_naive_slots_are_supported(self):
        naive = [
            KeplerInputSlot(
                s.start_time.astimezone().replace(tzinfo=None),
                s.end_time.astimezone().replace(tzinfo=None),
                s.load_kwh, s.pv_kwh, s.import_price_sek_kwh, s.export_price_sek_kwh,
            )
            for s in _two_day_slots()
        ]
        aware, _ = self._plan(OWNER_RULES)
        plan, _ = self._plan(
            OWNER_RULES, slots=naive,
            state={"early_start_latched": False, "peak_off_since": None,
                   "heat_since": "2026-11-12T11:00+01:00"},
        )
        assert plan == aware

    @pytest.mark.asyncio
    async def test_initial_state_reads_the_status_sensor_only_for_loads_with_a_rule(
        self, tmp_path
    ):
        attrs = {"early_start_latched": True, "peak_off_since": None, "heat_since": None}
        ruled = {**KIM, "id": "alex_heater", **OWNER_RULES}
        read: list[str] = []

        from backend.core import ha_client as hac

        async def entity_state(entity_id):
            read.append(entity_id)
            if entity_id == "sensor.batt_soc":
                return {"state": "75.0"}
            if entity_id == "sensor.darkstar_alex_heater_state":
                return {"state": "early_start", "attributes": attrs}
            return None

        async def plan_windows(raw, now, tz, *, get_state):
            return []

        with (
            patch.object(hac, "get_ha_entity_state", side_effect=entity_state),
            patch.object(hac, "plan_windows_for_load", side_effect=plan_windows),
            patch("backend.core.secrets.load_home_assistant_config", return_value={}),
        ):
            res = await hac.get_initial_state(_initial_state_config(tmp_path, [KIM, ruled]))
        states = {s["id"]: s for s in res["cyclic_load_states"]}
        assert states["alex_heater"]["price_rule_state"] == attrs
        assert "price_rule_state" not in states["kim_heater"]
        assert "sensor.darkstar_kim_heater_state" not in read
