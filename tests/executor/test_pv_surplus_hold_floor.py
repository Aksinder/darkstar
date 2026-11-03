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

    @pytest.mark.parametrize("soc", [11.1, 44.4, 100.0])
    def test_hold_slot_idles_above_the_floor_too(self, controller, soc):
        """The PV>load exception is gone (Hold slots always idle): forecast surplus
        no longer opens the pack for discharge. 2026-10-25..11-01 it drained 34.7
        kWh in Hold slots under over-forecast PV and ratcheted the targets down."""
        d = controller.decide(_hold(soc_target=50), SystemState(current_soc_percent=soc))
        assert d.mode_intent == "idle"

    def test_full_pack_with_sun_idles_and_keeps_charge_path(self, controller):
        """The case d9dc45a9 was written for (full pack, at target, sun): idle now.
        Idle caps discharge only — the charge ceiling stays at the maximum, so real
        surplus still reaches the pack."""
        d = controller.decide(
            _hold(soc_target=100, pv_kw=11.0, load_kw=1.5),
            SystemState(current_soc_percent=100.0),
        )
        assert d.mode_intent == "idle"
        assert d.max_charge == ControllerConfig().max_charge_a

    def test_unknown_soc_hold_idles(self, controller):
        state = SystemState(soc_known=False, min_soc_percent=60.0)
        d = controller.decide(_hold(soc_target=80), state)
        assert d.mode_intent == "idle"


class TestHoldRatchet:
    """2026-11-01 08:15-13:00: Hold slots, forecast PV 2.4-3.2 kW over a ~1.5 kW
    load, actual PV 0.6-1.6 kW. Each replan set the Hold target to the SoC it
    entered with (planner/output/soc_target.py), the slot self-consumed, and the
    next replan held lower: 70 -> 68 -> 63 -> ... -> 10 %."""

    def test_replans_cannot_step_the_target_down(self, controller):
        soc = 70.0
        for target in (70, 68, 63, 60, 58, 54, 52, 51, 50, 48, 44, 39, 36, 33, 27, 19, 13):
            for live in (soc + 0.6, soc, soc - 0.6):
                slot = _hold(soc_target=target, pv_kw=3.0, load_kw=1.5)
                d = controller.decide(slot, SystemState(current_soc_percent=live))
                assert d.mode_intent == "idle", (target, live)
            soc = float(target)


class TestPvDrivenChargeFloor:
    """charge_kw > 0 with forecast PV covering charge + load used to self-consume with
    no look at SoC at all. Live 2026-10-04 10:00: slot charge 0.38 kW, forecast PV
    2.08 kW, load 1.16 kW, SoC 8.8 % -> self_consumption, battery serving the house.
    Since the PV-drain fix it idles at any SoC (pv_charge_allow_discharge opts back)."""

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
        assert d.mode_intent == "idle"
        opt_in = Controller(ControllerConfig(pv_charge_allow_discharge=True), InverterConfig())
        assert opt_in.decide(slot, SystemState(current_soc_percent=11.1)).mode_intent == (
            "self_consumption"
        )
        # The floor still wins over the opt-in.
        assert opt_in.decide(slot, SystemState(current_soc_percent=8.8)).mode_intent == "idle"

    def test_pv_surplus_with_export_headroom_at_floor_holds(self, controller):
        """Case (a) of the branch: charge + PV export, no battery discharge."""
        slot = _hold(soc_target=60, charge_kw=2.0, export_kw=1.0, pv_kw=5.0, load_kw=1.0)
        d = controller.decide(slot, SystemState(current_soc_percent=10.0))
        assert d.mode_intent == "idle"

    def test_unknown_soc_unchanged(self, controller):
        slot = _hold(soc_target=80, charge_kw=0.38, pv_kw=2.08, load_kw=1.16)
        d = controller.decide(slot, SystemState(soc_known=False, min_soc_percent=60.0))
        assert d.mode_intent == "idle"
        opt_in = Controller(ControllerConfig(pv_charge_allow_discharge=True), InverterConfig())
        d = opt_in.decide(slot, SystemState(soc_known=False, min_soc_percent=60.0))
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
