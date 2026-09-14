"""A plug sensor that stopped answering has not said the cable is out.

2026-09-10 20:42 the Tesla Fleet integration went dark. binary_sensor.white_betty_
charge_cable read "unknown", bool() made that "unplugged", and a car sitting in its
cable at 33 % got nothing for eleven hours. Same disease as the SoC and presence
defaults next to it, same medicine: hold the last READABLE answer, and let only a
sensor that positively says "off" unplug the car.
"""

from __future__ import annotations

import pytest
import yaml

import backend.core.ha_client as hac


@pytest.fixture(autouse=True)
def _clean():
    for d in (hac._LAST_GOOD_EV_PLUG, hac._LAST_GOOD_EV_SOC, hac._LAST_GOOD_EV_HOME):
        d.clear()
    yield
    for d in (hac._LAST_GOOD_EV_PLUG, hac._LAST_GOOD_EV_SOC, hac._LAST_GOOD_EV_HOME):
        d.clear()


def _config(tmp_path):
    cfg = {
        "system": {
            "battery": {"capacity_kwh": 10.0},
            "has_ev_charger": True,
            "location": {"latitude": 57.6, "longitude": 18.5},
        },
        "ev_chargers": [
            {"id": "tesla", "enabled": True, "soc_sensor": "sensor.ev_soc",
             "plug_sensor": "binary_sensor.cable"}
        ],
        "input_sensors": {"battery_soc": "sensor.batt_soc"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


async def _run(tmp_path, *, plug_state):
    from unittest.mock import AsyncMock, patch

    from backend.core.ha_client import get_initial_state

    floats = {"sensor.batt_soc": 75.0, "sensor.ev_soc": 33.0}
    cable = {"state": plug_state, "attributes": {}, "last_changed": None}

    async def fake_entity_state(entity_id):
        return cable if entity_id == "binary_sensor.cable" else None

    with (
        patch("backend.core.ha_client.get_ha_sensor_float") as mock_float,
        patch("backend.core.ha_client.get_ha_entity_state", side_effect=fake_entity_state),
        patch("backend.core.ha_client.get_ha_bool", new_callable=AsyncMock, return_value=False),
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
    ):
        mock_float.side_effect = lambda e: floats.get(e)
        res = await get_initial_state(_config(tmp_path))
        return res["ev_charger_states"][0]


class TestTriStateRead:
    @pytest.mark.asyncio
    async def test_unknown_is_none_not_false(self):
        from unittest.mock import patch

        async def st(_e):
            return {"state": "unknown"}
        with patch("backend.core.ha_client.get_ha_entity_state", side_effect=st):
            assert await hac.get_ha_bool_or_none("binary_sensor.x") is None

    @pytest.mark.asyncio
    async def test_off_is_false(self):
        from unittest.mock import patch

        async def st(_e):
            return {"state": "off"}
        with patch("backend.core.ha_client.get_ha_entity_state", side_effect=st):
            assert await hac.get_ha_bool_or_none("binary_sensor.x") is False


class TestSilenceIsNotUnplugging:
    @pytest.mark.asyncio
    async def test_on_is_plugged(self, tmp_path):
        assert (await _run(tmp_path, plug_state="on"))["plugged_in"] is True

    @pytest.mark.asyncio
    async def test_off_still_unplugs(self, tmp_path):
        assert (await _run(tmp_path, plug_state="off"))["plugged_in"] is False

    @pytest.mark.asyncio
    async def test_unknown_holds_the_last_plugged(self, tmp_path):
        """THE case: the integration went dark with the cable in."""
        assert (await _run(tmp_path, plug_state="on"))["plugged_in"] is True
        assert (await _run(tmp_path, plug_state="unknown"))["plugged_in"] is True
        assert (await _run(tmp_path, plug_state="unavailable"))["plugged_in"] is True

    @pytest.mark.asyncio
    async def test_a_car_last_seen_unplugged_stays_out(self, tmp_path):
        assert (await _run(tmp_path, plug_state="off"))["plugged_in"] is False
        assert (await _run(tmp_path, plug_state="unknown"))["plugged_in"] is False

    @pytest.mark.asyncio
    async def test_nothing_to_hold_is_unplugged(self, tmp_path):
        assert (await _run(tmp_path, plug_state="unknown"))["plugged_in"] is False

    @pytest.mark.asyncio
    async def test_an_expired_hold_is_unplugged(self, tmp_path):
        import time
        await _run(tmp_path, plug_state="on")
        hac._LAST_GOOD_EV_PLUG["tesla"] = (True, time.time() - hac._EV_PLUG_HOLD_S - 1)
        assert (await _run(tmp_path, plug_state="unknown"))["plugged_in"] is False
