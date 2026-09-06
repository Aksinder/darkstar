"""An unreadable EV SoC must not read as an empty battery.

0% is not a neutral default. The penalty ladder treats a low SoC as the state that is
allowed to outbid price — that is its entire purpose — so a sensor that returns nothing
and is recorded as 0% asks the planner to buy at any cost.

2026-09-06 21:00: the Tesla integration dropped every white_betty_* entity to unknown
while spot stood at 1.27 SEK/kWh. Darkstar logged "defaulting to 0%". Nothing was
force-charged, but only because the controller happened to fall out of the surplus list
before anything acted on the number. That is luck of ordering, not a guard.

100% would be no better, only wrong in the other direction — it would sleep through a
night that a genuinely empty car needed. So this invents neither number.
"""

from __future__ import annotations

import time

import pytest

import backend.core.ha_client as hac


@pytest.fixture(autouse=True)
def _clean_hold():
    hac._LAST_GOOD_EV_SOC.clear()
    yield
    hac._LAST_GOOD_EV_SOC.clear()


class TestTheHoldWindow:
    def test_a_good_reading_is_remembered(self):
        hac._LAST_GOOD_EV_SOC["tesla"] = (64.0, time.time())
        soc, ts = hac._LAST_GOOD_EV_SOC["tesla"]
        assert soc == 64.0
        assert time.time() - ts < 5

    def test_the_hold_is_shorter_than_a_night(self):
        """A held value may carry the car across an integration hiccup. It must never
        quietly plan a whole night on a stale number."""
        assert 0 < hac._EV_SOC_HOLD_S <= 8 * 3600

    def test_a_fresh_hold_is_usable(self):
        hac._LAST_GOOD_EV_SOC["tesla"] = (27.0, time.time() - 600)
        soc, ts = hac._LAST_GOOD_EV_SOC["tesla"]
        assert (time.time() - ts) <= hac._EV_SOC_HOLD_S
        assert soc == 27.0

    def test_a_stale_hold_is_not(self):
        hac._LAST_GOOD_EV_SOC["tesla"] = (27.0, time.time() - hac._EV_SOC_HOLD_S - 1)
        _soc, ts = hac._LAST_GOOD_EV_SOC["tesla"]
        assert (time.time() - ts) > hac._EV_SOC_HOLD_S

    def test_the_hold_is_per_charger(self):
        """Two cars fail independently; one going dark must not speak for the other."""
        hac._LAST_GOOD_EV_SOC["tesla"] = (27.0, time.time())
        assert "easee_fmb" not in hac._LAST_GOOD_EV_SOC


def _config(tmp_path):
    import yaml
    cfg = {
        "system": {
            "battery": {"capacity_kwh": 10.0},
            "has_water_heater": False,
            "has_ev_charger": True,
        },
        "ev_chargers": [
            {"id": "tesla", "enabled": True, "soc_sensor": "sensor.ev_soc"}
        ],
        "input_sensors": {"battery_soc": "sensor.batt_soc"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


async def _run(tmp_path, soc_value):
    from unittest.mock import AsyncMock, patch

    from backend.core.ha_client import get_initial_state

    values = {"sensor.batt_soc": 75.0, "sensor.ev_soc": soc_value}
    with (
        patch("backend.core.ha_client.get_ha_sensor_float") as mock_sensor,
        patch("backend.core.ha_client.get_ha_bool", new_callable=AsyncMock, return_value=False),
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
    ):
        mock_sensor.side_effect = lambda e: values.get(e)
        return await get_initial_state(_config(tmp_path))


class TestWhatTheCarLooksLikeToThePlanner:
    @pytest.mark.asyncio
    async def test_a_readable_soc_is_reported_and_the_car_is_schedulable(self, tmp_path):
        res = await _run(tmp_path, 64.0)
        ev = res["ev_charger_states"][0]
        assert ev["soc_percent"] == 64.0
        assert ev["soc_known"] is True
        assert ev["plugged_in"] is True

    @pytest.mark.asyncio
    async def test_an_unknown_soc_withholds_the_car_instead_of_calling_it_empty(
        self, tmp_path
    ):
        """THE case. The number may still read 0, but the car is no longer offered to
        the planner as a plugged-in empty battery — which is what would have asked it to
        buy at 1.27 SEK/kWh."""
        res = await _run(tmp_path, None)
        ev = res["ev_charger_states"][0]
        assert ev["soc_known"] is False
        assert ev["plugged_in"] is False

    @pytest.mark.asyncio
    async def test_a_hiccup_is_carried_by_the_last_good_reading(self, tmp_path):
        """One good read, then the integration drops out: the car keeps its SoC and
        stays schedulable rather than vanishing on a single missed sample."""
        first = await _run(tmp_path, 64.0)
        assert first["ev_charger_states"][0]["soc_percent"] == 64.0
        second = await _run(tmp_path, None)
        ev = second["ev_charger_states"][0]
        assert ev["soc_percent"] == 64.0
        assert ev["soc_known"] is True
        assert ev["plugged_in"] is True

    @pytest.mark.asyncio
    async def test_a_stale_hold_expires_rather_than_planning_a_night_on_it(self, tmp_path):
        await _run(tmp_path, 64.0)
        soc, _ts = hac._LAST_GOOD_EV_SOC["tesla"]
        hac._LAST_GOOD_EV_SOC["tesla"] = (soc, time.time() - hac._EV_SOC_HOLD_S - 1)
        res = await _run(tmp_path, None)
        ev = res["ev_charger_states"][0]
        assert ev["soc_known"] is False
        assert ev["plugged_in"] is False
