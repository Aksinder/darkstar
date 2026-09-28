"""The last-good EV readings must survive a restart.

2026-09-27 19:47 Home Assistant restarted with the Tesla asleep: every
white_betty_* entity went ``unknown``. The 24 h holds carried the car — plugged,
home, 19 % — until 03:30, when the nightly install restarted the add-on. The
holds lived in memory, so after it the servo read

    tesla binary_sensor.white_betty_charge_cable unreadable ('unknown')
    and nothing to hold (never read) -> False

and a car sitting in its cable was "not plugged". No start was commanded, so the
wake button (pressed only after a FAILED start) was never pressed. The car left
at 07:38 with 19 %.

Rule: both hold owners write their last-good readings to disk and read them back
at startup. Persistence is OFF unless the application enables it, so nothing
here may leak a file or a reading into another test.
"""

from __future__ import annotations

import json

import pytest

import backend.core.ha_client as hac
from backend.core import ev_hold_store
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config

PLUG = "binary_sensor.tesla_plug"
ON_STATES = ("on", "true", "plugged", "connected")
T0 = 1_790_000_000.0


@pytest.fixture(autouse=True)
def _isolated():
    dicts = (hac._LAST_GOOD_EV_PLUG, hac._LAST_GOOD_EV_SOC, hac._LAST_GOOD_EV_HOME)
    ev_hold_store.disable()
    for d in dicts:
        d.clear()
    yield
    ev_hold_store.disable()
    for d in dicts:
        d.clear()


class FakeHA:
    def __init__(self, states):
        self.states = states

    async def get_state_value(self, entity):
        return self.states.get(entity)


def _controller():
    return EVSurplusController(
        parse_ev_surplus_config(
            {
                "ev_surplus": {
                    "enabled": True,
                    "pv_power_entity": "sensor.pv",
                    "grid_power_entity": "sensor.grid",
                    "battery_power_entity": "sensor.batt",
                    "battery_soc_entity": "sensor.soc",
                    "chargers": [
                        {
                            "id": "tesla",
                            "priority": 0,
                            "min_current_a": 5,
                            "max_current_a": 16,
                            "phases": 3,
                            "switch_entity": "switch.tesla",
                            "plug_entity": PLUG,
                        }
                    ],
                }
            }
        )
    )


async def _plugged(ctrl, state, now_ts):
    return await ctrl._read_on_held(
        FakeHA({PLUG: state}),
        PLUG,
        ON_STATES,
        True,
        key="tesla",
        memory=ctrl._last_good_plug,
        now_ts=now_ts,
        what="plug",
    )


class TestStore:
    def test_off_by_default_writes_and_reads_nothing(self, tmp_path):
        assert ev_hold_store.is_enabled() is False
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0) is False
        assert ev_hold_store.load("servo") == {}
        assert list(tmp_path.iterdir()) == []

    def test_round_trip(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        holds = {"plug": {"tesla": (True, T0)}, "home": {"tesla": (False, T0 - 60)}}
        assert ev_hold_store.save("servo", holds, T0) is True
        assert ev_hold_store.load("servo") == holds

    def test_namespaces_do_not_overwrite_each_other(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0)
        ev_hold_store.save("planner", {"soc": {"tesla": (19.0, T0)}}, T0)
        assert ev_hold_store.load("servo") == {"plug": {"tesla": (True, T0)}}
        assert ev_hold_store.load("planner") == {"soc": {"tesla": (19.0, T0)}}

    def test_a_newer_timestamp_alone_is_throttled(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0) is True
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, T0 + 60)}}, T0 + 60) is False
        assert ev_hold_store.load("servo")["plug"]["tesla"][1] == T0
        late = T0 + ev_hold_store.SAVE_MIN_INTERVAL_S
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, late)}}, late) is True
        assert ev_hold_store.load("servo")["plug"]["tesla"][1] == late

    def test_a_changed_value_is_written_at_once(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0)
        assert ev_hold_store.save("servo", {"plug": {"tesla": (False, T0 + 1)}}, T0 + 1) is True
        assert ev_hold_store.load("servo")["plug"]["tesla"] == (False, T0 + 1)

    @pytest.mark.parametrize("content", ["", "{not json", "[1, 2]", '{"servo": 7}'])
    def test_an_unreadable_file_is_nothing_held_not_a_crash(self, tmp_path, content):
        path = tmp_path / "holds.json"
        path.write_text(content)
        ev_hold_store.enable(path)
        assert ev_hold_store.load("servo") == {}
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0) is True
        assert ev_hold_store.load("servo") == {"plug": {"tesla": (True, T0)}}

    def test_malformed_entries_are_dropped_one_by_one(self, tmp_path):
        path = tmp_path / "holds.json"
        path.write_text(
            json.dumps(
                {
                    "servo": {
                        "plug": {
                            "tesla": [True, T0],
                            "short": [True],
                            "text_ts": [True, "yesterday"],
                            "bool_ts": [True, True],
                            "scalar": 5,
                        }
                    }
                }
            )
        )
        ev_hold_store.enable(path)
        assert ev_hold_store.load("servo") == {"plug": {"tesla": (True, T0)}}

    def test_a_failed_write_is_swallowed(self, tmp_path):
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        ev_hold_store.enable(blocker / "holds.json")
        assert ev_hold_store.save("servo", {"plug": {"tesla": (True, T0)}}, T0) is False


class TestServoSurvivesARestart:
    @pytest.mark.asyncio
    async def test_the_incident(self, tmp_path):
        """Plugged in, entities go blank, add-on restarts: still plugged."""
        ev_hold_store.enable(tmp_path / "holds.json")
        before = _controller()
        assert await _plugged(before, "on", T0) is True

        after = _controller()  # the 03:30 restart
        assert await _plugged(after, "unknown", T0 + 8 * 3600) is True

    @pytest.mark.asyncio
    async def test_without_the_store_a_restart_forgets(self):
        """The control: this is what happened on 2026-09-28."""
        before = _controller()
        assert await _plugged(before, "on", T0) is True
        after = _controller()
        assert await _plugged(after, "unknown", T0 + 8 * 3600) is False

    @pytest.mark.asyncio
    async def test_a_car_last_seen_unplugged_is_not_invented(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        assert await _plugged(_controller(), "off", T0) is False
        assert await _plugged(_controller(), "unknown", T0 + 3600) is False

    @pytest.mark.asyncio
    async def test_a_persisted_hold_still_expires(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        assert await _plugged(_controller(), "on", T0) is True
        assert await _plugged(_controller(), "unknown", T0 + 25 * 3600) is False

    @pytest.mark.asyncio
    async def test_a_fresh_reading_replaces_the_persisted_one(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        assert await _plugged(_controller(), "on", T0) is True
        ctrl = _controller()
        assert await _plugged(ctrl, "off", T0 + 60) is False  # cable pulled
        assert await _plugged(_controller(), "unknown", T0 + 120) is False


class TestPlannerSideSurvivesARestart:
    def test_restore_fills_the_three_holds(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        ev_hold_store.save(
            "planner",
            {
                "soc": {"tesla": (19.0, T0)},
                "home": {"tesla": (True, T0)},
                "plug": {"tesla": (True, T0)},
            },
            T0,
        )
        hac.restore_ev_holds()
        assert hac._LAST_GOOD_EV_SOC == {"tesla": (19.0, T0)}
        assert hac._LAST_GOOD_EV_HOME == {"tesla": (True, T0)}
        assert hac._LAST_GOOD_EV_PLUG == {"tesla": (True, T0)}

    def test_a_newer_reading_in_memory_wins(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        ev_hold_store.save("planner", {"soc": {"tesla": (19.0, T0)}}, T0)
        hac._LAST_GOOD_EV_SOC["tesla"] = (42.0, T0 + 600)
        hac.restore_ev_holds()
        assert hac._LAST_GOOD_EV_SOC["tesla"] == (42.0, T0 + 600)

    def test_wrongly_typed_values_are_not_restored(self, tmp_path):
        path = tmp_path / "holds.json"
        path.write_text(
            json.dumps(
                {
                    "planner": {
                        "soc": {"tesla": [True, T0], "fmb": ["full", T0]},
                        "plug": {"tesla": ["on", T0]},
                    }
                }
            )
        )
        ev_hold_store.enable(path)
        hac.restore_ev_holds()
        assert hac._LAST_GOOD_EV_SOC == {} and hac._LAST_GOOD_EV_PLUG == {}

    def test_persist_writes_the_current_holds(self, tmp_path):
        ev_hold_store.enable(tmp_path / "holds.json")
        hac._LAST_GOOD_EV_SOC["tesla"] = (13.0, T0)
        hac._LAST_GOOD_EV_PLUG["tesla"] = (True, T0)
        hac._persist_ev_holds()
        assert ev_hold_store.load("planner") == {
            "soc": {"tesla": (13.0, T0)},
            "plug": {"tesla": (True, T0)},
        }

    def test_persist_is_a_no_op_while_the_store_is_off(self, tmp_path):
        hac._LAST_GOOD_EV_SOC["tesla"] = (13.0, T0)
        hac._persist_ev_holds()
        assert list(tmp_path.iterdir()) == []
