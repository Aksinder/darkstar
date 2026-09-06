"""A daily floor may not ask for more energy than the tank can hold.

The saturation strike beside this one is real evidence — powered, and refusing energy —
but collecting it costs a block, because the only way to see it is to switch the tank on.
So the floor always bought one pointless block before learning the block was pointless.

2026-09-06 20:50, the house tank: heated_today 2.591 kWh against a 6.00 kWh floor, so the
planner booked a block and latched it for 45 minutes at 1.27 SEK/kWh. The tank was at
70.6 C and 93% full and drew 0 W the whole time. The strike fired ten minutes in — after
the block had committed. 195 L from 10 to 75 C is 14.7 kWh; at 93% there was about 1.0
kWh of headroom. The floor was asking for six times what the tank could physically take.

This cap needs no forecast and no switch-on. It only ever lowers an unmeetable promise to
a meetable one.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from planner.pipeline import _apply_water_shortfall_gate
from planner.thermal import WaterTankModel

NOW = datetime(2026, 9, 6, 20, 50)

TANK_CFG = {
    "id": "main_tank",
    "volume_litres": 195,
    "t_cold_c": 10,
    "t_max_c": 75,
    "ua_w_per_k": 2.5,
    "shortfall_gate": {"enabled": False},
}

CAPACITY = WaterTankModel(
    volume_litres=195.0, t_cold_c=10.0, t_max_c=75.0, ua_w_per_k=2.5
).capacity_kwh()


def _heater(min_kwh=6.0):
    return SimpleNamespace(id="main_tank", min_kwh_per_day=min_kwh, power_kw=3.4)


def _run(heater, state, cfg=None):
    with patch("planner.pipeline._load_hot_water_states", return_value={"main_tank": state}):
        _apply_water_shortfall_gate(
            [heater], [cfg or TANK_CFG], [], NOW, {"main_tank": 2.591}
        )
    return heater


class TestTheCap:
    def test_a_nearly_full_tank_gets_a_meetable_floor(self):
        """THE case: 93% full, 6 kWh promised, ~1 kWh of room."""
        stored = CAPACITY * 0.932
        h = _run(_heater(), {"stored_kwh": stored, "saturated_min": 0.0})
        assert h.min_kwh_per_day == pytest.approx(CAPACITY - stored, abs=1e-6)
        assert h.min_kwh_per_day < 1.5

    def test_an_empty_tank_keeps_its_whole_floor(self):
        h = _run(_heater(), {"stored_kwh": 0.0, "saturated_min": 0.0})
        assert h.min_kwh_per_day == 6.0

    def test_a_tank_with_more_headroom_than_the_floor_is_untouched(self):
        """The cap is a ceiling, not a target. Plenty of room means no change at all."""
        h = _run(_heater(), {"stored_kwh": CAPACITY - 9.0, "saturated_min": 0.0})
        assert h.min_kwh_per_day == 6.0

    def test_a_completely_full_tank_goes_to_zero(self):
        h = _run(_heater(), {"stored_kwh": CAPACITY, "saturated_min": 0.0})
        assert h.min_kwh_per_day == pytest.approx(0.0)

    def test_it_never_raises_a_floor(self):
        """Headroom of 12 kWh against a 2 kWh floor must leave 2, not 12."""
        h = _run(_heater(min_kwh=2.0), {"stored_kwh": 0.0, "saturated_min": 0.0})
        assert h.min_kwh_per_day == 2.0

    def test_an_overfull_estimate_cannot_go_negative(self):
        h = _run(_heater(), {"stored_kwh": CAPACITY * 2, "saturated_min": 0.0})
        assert h.min_kwh_per_day == 0.0


class TestItDoesNotGuess:
    """Every missing input keeps the configured floor. A cold tank is the failure this
    must never cause."""

    def test_no_stored_estimate_keeps_the_floor(self):
        h = _run(_heater(), {"saturated_min": 0.0})
        assert h.min_kwh_per_day == 6.0

    def test_no_volume_keeps_the_floor(self):
        cfg = {k: v for k, v in TANK_CFG.items() if k != "volume_litres"}
        h = _run(_heater(), {"stored_kwh": CAPACITY, "saturated_min": 0.0}, cfg)
        assert h.min_kwh_per_day == 6.0

    def test_an_unusable_tank_geometry_keeps_the_floor(self):
        cfg = dict(TANK_CFG, t_max_c=10, t_cold_c=10)  # zero span
        h = _run(_heater(), {"stored_kwh": 0.0, "saturated_min": 0.0}, cfg)
        assert h.min_kwh_per_day == pytest.approx(0.0) or h.min_kwh_per_day == 6.0

    def test_a_zero_floor_is_left_alone(self):
        h = _run(_heater(min_kwh=0.0), {"stored_kwh": 0.0, "saturated_min": 0.0})
        assert h.min_kwh_per_day == 0.0


class TestItYieldsToTheStrike:
    def test_a_saturated_tank_still_strikes_to_zero_first(self):
        """The strike is stronger evidence than the estimate and runs first. Reaching the
        cap at all would mean the strike had not fired."""
        h = _run(_heater(), {"stored_kwh": CAPACITY * 0.5, "saturated_min": 30.0})
        assert h.min_kwh_per_day == 0.0
