"""The forecast-PV self-consumption branches stop at the battery floor.

At or below the slot's SoC target the controller normally holds (idle). During a
slot whose FORECAST says PV > load it self-consumes instead, so transients the PV
feed cannot reach are served from the pack and "the surplus refills it". Nothing
in that branch checked whether the surplus arrived. Live 2026-10-05 10:41-12:00
(decision journal): six Hold slots at target 10 %, forecast PV 2.5-3.1 kW over a
1.2-1.6 kW load, SoC 7.7 -> 2.2 -> 0.0 %, and every mid-slot replan answered with
a 1-3 minute grid-charge burst (up to 6.96 kW) that the next slot discharged again.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from executor.config import ControllerConfig, InverterConfig
from executor.controller import Controller
from executor.engine import ExecutorEngine
from executor.override import SlotPlan, SystemState


@pytest.fixture
def controller() -> Controller:
    return Controller(ControllerConfig(), InverterConfig())


def _hold(soc_target: int = 10, pv_kw: float = 2.52, load_kw: float = 1.2, **kw) -> SlotPlan:
    return SlotPlan(soc_target=soc_target, pv_kw=pv_kw, load_kw=load_kw, **kw)


class TestFloor:
    @pytest.mark.parametrize("soc", [0.0, 2.2, 7.7, 8.8, 10.0, 10.4])
    def test_at_or_below_min_soc_holds(self, controller, soc):
        """The 2026-10-05 slots: Hold at 10 %, forecast surplus, pack below the floor."""
        d = controller.decide(_hold(), SystemState(current_soc_percent=soc))
        assert d.mode_intent == "idle"

    def test_one_sensor_step_above_the_floor_is_released(self, controller):
        """11.1 % is the Sungrow sensor's next step above 10.0 %. Above the slot
        target the ordinary rule applies — the guard adds no dead band of its own."""
        d = controller.decide(_hold(), SystemState(current_soc_percent=11.1))
        assert d.mode_intent == "self_consumption"

    def test_exception_unchanged_above_the_floor(self, controller):
        """The case the exception was written for: a full pack, at its target, with
        sun. Still self-consumption."""
        d = controller.decide(
            _hold(soc_target=100, pv_kw=11.0, load_kw=1.5),
            SystemState(current_soc_percent=100.0),
        )
        assert d.mode_intent == "self_consumption"

    def test_exception_unchanged_mid_range_below_target(self, controller):
        """Below a mid-range Hold target the exception still applies — only the
        floor is new."""
        d = controller.decide(
            _hold(soc_target=50), SystemState(current_soc_percent=44.4)
        )
        assert d.mode_intent == "self_consumption"

    def test_floor_follows_the_configured_min_soc(self, controller):
        """A site with min_soc 20: 18.8 % holds, 21.1 % does not (target 30)."""
        slot = _hold(soc_target=30)
        low = SystemState(current_soc_percent=18.8, min_soc_percent=20.0)
        high = SystemState(current_soc_percent=21.1, min_soc_percent=20.0)
        assert controller.decide(slot, low).mode_intent == "idle"
        assert controller.decide(slot, high).mode_intent == "self_consumption"

    def test_unknown_soc_keeps_the_old_behaviour(self, controller):
        """soc_known=False leaves current_soc_percent at its 50 % default: the guard
        must not read meaning into a number nobody measured."""
        state = SystemState(soc_known=False, min_soc_percent=60.0)
        d = controller.decide(_hold(soc_target=80), state)
        assert d.mode_intent == "self_consumption"


class TestPvDrivenChargeFloor:
    """charge_kw > 0 with forecast PV covering charge + load -> self_consumption, with
    no look at SoC at all. Live 2026-10-04 10:00: slot charge 0.38 kW, forecast PV
    2.08 kW, load 1.16 kW, SoC 8.8 % -> self_consumption, battery serving the house."""

    def test_at_floor_holds(self, controller):
        slot = _hold(soc_target=80, charge_kw=0.38, pv_kw=2.08, load_kw=1.16)
        d = controller.decide(slot, SystemState(current_soc_percent=8.8))
        assert d.mode_intent == "idle"

    def test_idle_keeps_the_charge_path_open(self, controller):
        """idle on the Sungrow profile renders max_charge_power from {{max_charge}};
        the decision must still carry the full charge ceiling."""
        slot = _hold(soc_target=80, charge_kw=0.38, pv_kw=2.08, load_kw=1.16)
        d = controller.decide(slot, SystemState(current_soc_percent=8.8))
        assert d.max_charge == ControllerConfig().max_charge_a
        assert d.max_charge > 0

    def test_above_floor_unchanged(self, controller):
        slot = _hold(soc_target=80, charge_kw=0.38, pv_kw=2.08, load_kw=1.16)
        d = controller.decide(slot, SystemState(current_soc_percent=11.1))
        assert d.mode_intent == "self_consumption"

    def test_pv_surplus_with_export_headroom_at_floor_holds(self, controller):
        """Case (a) of the branch: charge + PV export, no battery discharge."""
        slot = _hold(soc_target=60, charge_kw=2.0, export_kw=1.0, pv_kw=5.0, load_kw=1.0)
        d = controller.decide(slot, SystemState(current_soc_percent=10.0))
        assert d.mode_intent == "idle"

    def test_unknown_soc_unchanged(self, controller):
        slot = _hold(soc_target=80, charge_kw=0.38, pv_kw=2.08, load_kw=1.16)
        d = controller.decide(slot, SystemState(soc_known=False, min_soc_percent=60.0))
        assert d.mode_intent == "self_consumption"


class TestOtherBranchesUntouched:
    def test_no_forecast_surplus_was_already_idle(self, controller):
        d = controller.decide(
            _hold(pv_kw=0.0), SystemState(current_soc_percent=8.8)
        )
        assert d.mode_intent == "idle"

    def test_grid_charge_slot_above_floor_unchanged(self, controller):
        slot = _hold(charge_kw=3.0, pv_kw=0.5, load_kw=1.0)
        d = controller.decide(slot, SystemState(current_soc_percent=44.4))
        assert d.mode_intent == "charge"

    def test_grid_charge_slot_below_floor_still_charges(self, controller):
        slot = _hold(charge_kw=3.0, pv_kw=0.5, load_kw=1.0)
        d = controller.decide(slot, SystemState(current_soc_percent=5.5))
        assert d.mode_intent == "charge"

    def test_ev_slot_stays_idle(self, controller):
        d = controller.decide(
            _hold(soc_target=50, ev_charging_kw=7.0),
            SystemState(current_soc_percent=44.4),
        )
        assert d.mode_intent == "idle"

    def test_discharge_slot_at_floor_was_already_idle(self, controller):
        """Discharge slots carry target = min_soc; at the floor they idle with or
        without forecast surplus."""
        slot = _hold(discharge_kw=0.4)
        d = controller.decide(slot, SystemState(current_soc_percent=10.0))
        assert d.mode_intent == "idle"


class TestEngineFillsMinSoc:
    """SystemState.min_soc_percent was never populated from config; the guard would
    have used the dataclass default on every site."""

    def _gather(self, full_config):
        stub = SimpleNamespace(_full_config=full_config, ha_client=None)
        return asyncio.run(ExecutorEngine._gather_system_state(stub))  # type: ignore[arg-type]

    def test_reads_battery_min_soc(self):
        assert self._gather({"battery": {"min_soc_percent": 15}}).min_soc_percent == 15.0

    def test_missing_key_keeps_default(self):
        assert self._gather({}).min_soc_percent == 10.0
        assert self._gather({"battery": None}).min_soc_percent == 10.0

    def test_malformed_value_keeps_default(self):
        assert self._gather({"battery": {"min_soc_percent": "ten"}}).min_soc_percent == 10.0
