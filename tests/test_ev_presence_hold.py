"""A tracker that stopped answering has not told us the car left.

2026-09-06 20:49 the Tesla Fleet integration dropped all 87 white_betty_* entities to
unknown. device_tracker.white_betty_location went with them, the zone check read "" and
returned "away", and Darkstar logged, once a cycle:

    EV tesla not home (away); excluding from plan

The car was on the drive, plugged in, at 11% SoC, with Departure=07:30 the next morning.
Nothing was scheduled for it. The grace window could not help: it is measured from
last_changed, which the flip to unknown had just reset.

Same disease as the SoC default it sits next to — an unreadable input resolving to a
definite answer instead of "I don't know" — and the same medicine: hold the last
readable value, and let only a tracker that positively places the car elsewhere exclude
it.
"""

from __future__ import annotations

import time

import pytest
import yaml

import backend.core.ha_client as hac


@pytest.fixture(autouse=True)
def _clean():
    hac._LAST_GOOD_EV_HOME.clear()
    hac._LAST_GOOD_EV_SOC.clear()
    yield
    hac._LAST_GOOD_EV_HOME.clear()
    hac._LAST_GOOD_EV_SOC.clear()


def _config(tmp_path):
    cfg = {
        "system": {
            "battery": {"capacity_kwh": 10.0},
            "has_ev_charger": True,
            "location": {"latitude": 57.6, "longitude": 18.5},
        },
        "ev_chargers": [
            {
                "id": "tesla",
                "enabled": True,
                "soc_sensor": "sensor.ev_soc",
                "home_entity": "device_tracker.car",
            }
        ],
        "input_sensors": {"battery_soc": "sensor.batt_soc"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


async def _run(tmp_path, *, zone_state, soc=64.0):
    from unittest.mock import AsyncMock, patch

    from backend.core.ha_client import get_initial_state

    floats = {"sensor.batt_soc": 75.0, "sensor.ev_soc": soc}
    tracker = {"state": zone_state, "attributes": {}, "last_changed": None}

    async def fake_entity_state(entity_id):
        return tracker if entity_id == "device_tracker.car" else None

    with (
        patch("backend.core.ha_client.get_ha_sensor_float") as mock_float,
        patch("backend.core.ha_client.get_ha_entity_state", side_effect=fake_entity_state),
        patch("backend.core.ha_client.get_ha_bool", new_callable=AsyncMock, return_value=False),
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
    ):
        mock_float.side_effect = lambda e: floats.get(e)
        res = await get_initial_state(_config(tmp_path))
        return res["ev_charger_states"][0]


class TestSilenceIsNotDeparture:
    @pytest.mark.asyncio
    async def test_a_home_tracker_is_home(self, tmp_path):
        assert (await _run(tmp_path, zone_state="home"))["at_home"] is True

    @pytest.mark.asyncio
    async def test_a_tracker_that_says_away_still_excludes(self, tmp_path):
        """The gate must keep working. A car charging somewhere else is a phantom load,
        which is the whole reason the home check exists."""
        assert (await _run(tmp_path, zone_state="not_home"))["at_home"] is False

    @pytest.mark.asyncio
    async def test_an_unknown_tracker_holds_the_last_known_home(self, tmp_path):
        """THE case."""
        assert (await _run(tmp_path, zone_state="home"))["at_home"] is True
        assert (await _run(tmp_path, zone_state="unknown"))["at_home"] is True

    @pytest.mark.asyncio
    async def test_unavailable_and_empty_are_treated_the_same(self, tmp_path):
        await _run(tmp_path, zone_state="home")
        assert (await _run(tmp_path, zone_state="unavailable"))["at_home"] is True
        assert (await _run(tmp_path, zone_state=""))["at_home"] is True

    @pytest.mark.asyncio
    async def test_a_car_last_seen_AWAY_is_not_dragged_home(self, tmp_path):
        """The hold carries the last known answer, whichever it was. It must not invent
        presence for a car that genuinely left before the tracker died."""
        assert (await _run(tmp_path, zone_state="not_home"))["at_home"] is False
        assert (await _run(tmp_path, zone_state="unknown"))["at_home"] is False

    @pytest.mark.asyncio
    async def test_nothing_ever_read_stays_away(self, tmp_path):
        """With no history at all there is nothing to hold, and the conservative answer
        is the old one."""
        assert (await _run(tmp_path, zone_state="unknown"))["at_home"] is False

    @pytest.mark.asyncio
    async def test_a_stale_hold_expires(self, tmp_path):
        await _run(tmp_path, zone_state="home")
        home, _ts = hac._LAST_GOOD_EV_HOME["tesla"]
        hac._LAST_GOOD_EV_HOME["tesla"] = (home, time.time() - hac._EV_HOME_HOLD_S - 1)
        assert (await _run(tmp_path, zone_state="unknown"))["at_home"] is False

    @pytest.mark.asyncio
    async def test_the_hold_outlasts_a_night(self, tmp_path):
        """An overnight outage is exactly the case this exists for."""
        assert hac._EV_HOME_HOLD_S >= 12 * 3600
