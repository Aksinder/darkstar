"""The websocket plug branch: unavailable/unknown hold the last state; replan only
on a real plugged<->unplugged transition (cloud dropouts used to drop the EV plan)."""

from unittest.mock import patch

import pytest

import backend.ha_socket as ha_socket

ENT = "binary_sensor.easee_ev_plugged_in"


def _client() -> ha_socket.HAWebSocketClient:
    client = object.__new__(ha_socket.HAWebSocketClient)
    client.monitored_entities = {ENT: "ev_plug_0"}
    client.latest_values = {}
    client.ev_charger_configs = [{"id": "easee_fmb", "name": "FMB"}]
    client.water_heater_configs = []
    client.inversion_flags = {}
    return client


@pytest.fixture
def env():
    import backend.core.ha_client as hc

    client = _client()
    with (
        patch.dict(hc._LAST_GOOD_EV_PLUG, {}, clear=True),
        patch.object(client, "_trigger_ev_replan") as replan,
        patch("backend.events.emit_ha_entity_change"),
        patch("backend.events.emit_live_metrics"),
    ):
        yield client, replan


def _feed(client, state):
    client._handle_state_change(ENT, {"entity_id": ENT, "state": state})


def test_unavailable_holds_plugged_and_does_not_replan(env):
    client, replan = env
    _feed(client, "on")
    replan.assert_called_once_with(charger_id="easee_fmb", plugged_in=True)
    replan.reset_mock()
    for blip in ("unavailable", "unknown", "unavailable"):
        _feed(client, blip)
        ev = client.latest_values["ev_chargers"][0]
        assert ev["plugged_in"] is True
        assert ev["plug_held"] is True
    replan.assert_not_called()
    # Back from the outage, still plugged: no transition -> no replan.
    _feed(client, "on")
    replan.assert_not_called()
    assert client.latest_values["ev_chargers"][0]["plug_held"] is False


def test_unavailable_holds_unplugged_too(env):
    client, replan = env
    _feed(client, "on")
    _feed(client, "off")
    replan.reset_mock()
    _feed(client, "unavailable")
    assert client.latest_values["ev_chargers"][0]["plugged_in"] is False
    replan.assert_not_called()


def test_repeated_plugged_events_do_not_replan(env):
    client, replan = env
    for _ in range(4):
        _feed(client, "on")
    assert replan.call_count == 1


def test_real_transitions_replan(env):
    client, replan = env
    _feed(client, "on")
    _feed(client, "unavailable")
    _feed(client, "off")  # a REAL reading after the outage: genuine unplug
    _feed(client, "on")
    assert [c.kwargs["plugged_in"] for c in replan.call_args_list] == [True, False, True]


def test_first_reading_unavailable_uses_planner_hold(env):
    client, replan = env
    with patch.object(ha_socket.HAWebSocketClient, "_planner_held_plug", return_value=True):
        _feed(client, "unavailable")
    assert client.latest_values["ev_chargers"][0]["plugged_in"] is True
    replan.assert_not_called()


def test_planner_held_plug_respects_hold_window():
    import time

    import backend.core.ha_client as hc

    with patch.dict(hc._LAST_GOOD_EV_PLUG, {}, clear=True):
        assert ha_socket.HAWebSocketClient._planner_held_plug("easee_fmb") is False
        hc._LAST_GOOD_EV_PLUG["easee_fmb"] = (True, time.time() - 60)
        assert ha_socket.HAWebSocketClient._planner_held_plug("easee_fmb") is True
        hc._LAST_GOOD_EV_PLUG["easee_fmb"] = (True, time.time() - hc._EV_PLUG_HOLD_S - 60)
        assert ha_socket.HAWebSocketClient._planner_held_plug("easee_fmb") is False


def test_seeding_still_suppresses_replan(env):
    client, replan = env
    client._seeding = True
    _feed(client, "on")
    client._seeding = False
    replan.assert_not_called()
    _feed(client, "off")
    replan.assert_called_once_with(charger_id="easee_fmb", plugged_in=False)


def test_first_real_unplug_vs_planner_held_plugged_replans(env):
    # Restart with the planner still holding "plugged": the first REAL reading says
    # unplugged — a change against the planner's belief, so it must replan.
    client, replan = env
    with patch.object(ha_socket.HAWebSocketClient, "_planner_held_plug", return_value=True):
        _feed(client, "off")
    replan.assert_called_once_with(charger_id="easee_fmb", plugged_in=False)
    assert client.latest_values["ev_chargers"][0]["plugged_in"] is False


def test_first_real_plugged_vs_planner_held_plugged_does_not_replan(env):
    client, replan = env
    with patch.object(ha_socket.HAWebSocketClient, "_planner_held_plug", return_value=True):
        _feed(client, "on")
    replan.assert_not_called()


def test_first_real_unplug_without_planner_hold_does_not_replan(env):
    client, replan = env
    _feed(client, "off")
    replan.assert_not_called()


def test_held_then_real_unplug_after_restart_replans(env):
    # previous=None, first reading unreadable (held plugged), then the real unplug.
    client, replan = env
    with patch.object(ha_socket.HAWebSocketClient, "_planner_held_plug", return_value=True):
        _feed(client, "unavailable")
        _feed(client, "off")
    replan.assert_called_once_with(charger_id="easee_fmb", plugged_in=False)
