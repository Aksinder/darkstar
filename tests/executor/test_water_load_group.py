"""The real-time surplus boost must respect the load-group cap.

The MILP enforces load_groups at PLANNING time (planner/solver/kepler.py). The boost
deliberately OVERRIDES the plan — it sees real-time surplus the planner cannot — but it
decided per device, and the string "load_group" appeared nowhere under executor/. So the
cap the owner configured was bypassed the moment the sun came out.

Live on this site, 2026-09-03: the villavagn's spa (1.8 kW) and hot-water tank (1.6 kW)
share a 10 A sub-fuse and a 2.3 kW group; 3.4 kW does not fit. The planner honoured that
— it booked the spa zero times that day — but the boost re-commanded the spa to 40 C on
every 60 s tick while the tank sat out a 15-minute dwell. 22 collisions, an HA phase
guard shedding both each time, and the tank's water never got hot. The spa won every
round on that asymmetry alone.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from executor.actions import ActionDispatcher, HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    LoadGroupConfig,
    WaterHeaterDeviceConfig,
)
from executor.engine import ExecutorEngine

SPA_ENT = "input_number.spa_darkstar_target_temp"
TANK_ENT = "switch.villavagn_vvb"
HOUSE_ENT = "switch.vvb"


def _engine(*, grouped: bool = True):
    """An engine with the live shape: spa + villavagn tank grouped, house tank outside."""
    cfg = ExecutorConfig(
        inverter=InverterConfig(),
        controller=ControllerConfig(),
        water_heater_devices=[
            WaterHeaterDeviceConfig(
                id="spa", target_entity=SPA_ENT, power_kw=1.8, surplus_boost=True
            ),
            WaterHeaterDeviceConfig(
                id="villavagn_tank", target_entity=TANK_ENT, power_kw=1.6
            ),
            WaterHeaterDeviceConfig(id="main_tank", target_entity=HOUSE_ENT, power_kw=3.4),
        ],
        load_groups=(
            [LoadGroupConfig(id="villavagn", max_power_kw=2.3, members=["spa", "villavagn_tank"])]
            if grouped
            else []
        ),
    )
    ha = MagicMock(spec=HAClient)
    ha.get_state_value = AsyncMock(return_value="0")
    dispatcher = ActionDispatcher(
        ha_client=ha, config=cfg, profile=None, shadow_mode=True
    )
    engine = object.__new__(ExecutorEngine)
    engine.config = cfg
    engine.dispatcher = dispatcher
    return engine, dispatcher, {d.id: d for d in cfg.water_heater_devices}


class TestTheGate:
    def test_nothing_commanded_lets_the_boost_through(self):
        eng, _d, by = _engine()
        assert eng._group_sibling_claiming_supply(by["spa"]) is None

    def test_a_commanded_sibling_blocks_the_boost(self):
        eng, d, by = _engine()
        d._last_water_cmd[TANK_ENT] = "on"
        why = eng._group_sibling_claiming_supply(by["spa"])
        assert why is not None
        assert "villavagn_tank" in why and "2.3" in why

    def test_a_shed_sibling_still_blocks_it(self):
        """THE case, and why the check is on the COMMAND, not the current.

        2026-09-02: tank committed ON at 13:14:03, the phase guard cut it at 13:14:39,
        Darkstar boosted the spa at 13:15:09. The tank measured 0 W at that instant —
        shed, not idle. A check against measured power waves the boost through and
        re-creates the collision; its CLAIM on the shared supply had not been withdrawn,
        only its current had.
        """
        eng, d, by = _engine()
        d._last_water_cmd[TANK_ENT] = "on"  # commanded on...
        # ...and the relay is now off, cut by something that is not us. The gate must
        # not care: nothing here reads the relay or the power.
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    def test_a_live_commit_alone_blocks_it(self):
        """A committed block claims the supply even before the next command lands."""
        eng, d, by = _engine()
        d._water_commit_until[TANK_ENT] = time.time() + 600
        assert eng._group_sibling_claiming_supply(by["spa"]) is not None

    def test_an_expired_commit_does_not(self):
        eng, d, by = _engine()
        d._water_commit_until[TANK_ENT] = time.time() - 1
        assert eng._group_sibling_claiming_supply(by["spa"]) is None

    def test_a_sibling_commanded_off_lets_it_through(self):
        eng, d, by = _engine()
        d._last_water_cmd[TANK_ENT] = "off"
        assert eng._group_sibling_claiming_supply(by["spa"]) is None

    def test_a_device_outside_every_group_is_untouched(self):
        """main_tank shares no supply with these two and must never be gated."""
        eng, d, by = _engine()
        d._last_water_cmd[TANK_ENT] = "on"
        d._last_water_cmd[SPA_ENT] = "on"
        assert eng._group_sibling_claiming_supply(by["main_tank"]) is None

    def test_no_groups_configured_is_todays_behaviour(self):
        """The whole feature must be inert for a site that configures no groups."""
        eng, d, by = _engine(grouped=False)
        d._last_water_cmd[TANK_ENT] = "on"
        assert eng._group_sibling_claiming_supply(by["spa"]) is None

    def test_a_pair_that_fits_the_cap_is_not_blocked(self):
        """The gate is about the CAP, not about company. Two small loads may share."""
        eng, d, by = _engine()
        by["spa"].power_kw = 0.5
        by["villavagn_tank"].power_kw = 1.0  # 1.5 kW <= 2.3
        d._last_water_cmd[TANK_ENT] = "on"
        assert eng._group_sibling_claiming_supply(by["spa"]) is None


class TestClaimsSupply:
    """ActionDispatcher.water_claims_supply — 'told to draw', not 'drawing'."""

    def test_unknown_entity_claims_nothing(self):
        _e, d, _b = _engine()
        assert d.water_claims_supply(TANK_ENT) is False
        assert d.water_claims_supply(None) is False

    def test_commanded_on_claims(self):
        _e, d, _b = _engine()
        d._last_water_cmd[TANK_ENT] = "on"
        assert d.water_claims_supply(TANK_ENT) is True

    def test_commanded_off_does_not(self):
        _e, d, _b = _engine()
        d._last_water_cmd[TANK_ENT] = "off"
        assert d.water_claims_supply(TANK_ENT) is False


class TestParsing:
    def test_the_live_group_shape_parses(self):
        from executor.config import _parse_load_groups

        groups = _parse_load_groups(
            {"load_groups": [{"id": "villavagn", "max_power_kw": 2.3,
                              "members": ["spa", "villavagn_tank"]}]}
        )
        assert len(groups) == 1
        assert groups[0].max_power_kw == pytest.approx(2.3)
        assert groups[0].members == ["spa", "villavagn_tank"]

    @pytest.mark.parametrize(
        "raw",
        [
            {"load_groups": []},
            {"load_groups": "not a list"},
            {"load_groups": [{"id": "x", "max_power_kw": 0, "members": ["a", "b"]}]},
            {"load_groups": [{"id": "", "max_power_kw": 2.3, "members": ["a", "b"]}]},
            {"load_groups": [{"id": "x", "max_power_kw": 2.3, "members": ["a"]}]},
            {},
        ],
    )
    def test_a_group_that_constrains_nothing_is_dropped(self, raw):
        """A ceiling that cannot bind reads like a real one when someone is debugging
        why two loads collided. Drop it rather than keep a no-op."""
        from executor.config import _parse_load_groups

        assert _parse_load_groups(raw) == []
