"""PV-driven charge slots never discharge the battery.

A slot with planned charge from FORECAST PV and discharge_kw == 0 used to run
self_consumption, which commands max_discharge. When real PV came in under the
forecast (overcast 2026-11-01), the pack covered the house instead of charging:
~8.1 kWh over 8 days, energy the plan meant the grid to supply. The branch now
idles (Sungrow idle = self-consumption EMS with a 10 W discharge cap and the full
charge ceiling), like Hold slots since 2a90a35c. Safety exceptions unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from executor.config import ControllerConfig, InverterConfig, load_executor_config
from executor.controller import Controller, apply_hold_fuse_relief
from executor.override import SlotPlan, SystemState


@pytest.fixture
def controller() -> Controller:
    return Controller(ControllerConfig(), InverterConfig())


def _pv_charge(**kw) -> SlotPlan:
    # Forecast: PV 4.0 covers charge 2.0 + load 1.0 -> PV-driven charge branch.
    base = {"charge_kw": 2.0, "discharge_kw": 0.0, "export_kw": 0.0, "pv_kw": 4.0,
            "load_kw": 1.0, "soc_target": 60}
    base.update(kw)
    return SlotPlan(**base)


class TestNoDischarge:
    def test_over_forecast_pv_below_load_idles(self, controller):
        """The plan's PV never arrives: the battery must rest, not feed the house.
        The controller only sees the plan, so whatever real PV is, the mode is idle."""
        d = controller.decide(_pv_charge(), SystemState(current_soc_percent=45))
        assert d.mode_intent == "idle"
        assert "Idle" in d.reason

    def test_pv_surplus_still_charges(self, controller):
        """Idle keeps the full charge ceiling, so real surplus charges the pack."""
        d = controller.decide(
            _pv_charge(export_kw=1.0, pv_kw=5.0), SystemState(current_soc_percent=45)
        )
        assert d.mode_intent == "idle"
        assert d.max_charge == ControllerConfig().max_charge_a > 0
        assert d.export_discharge_w == 0.0

    def test_near_full_topup_idles(self, controller):
        d = controller.decide(
            _pv_charge(charge_kw=0.1, pv_kw=1.5, load_kw=0.9), SystemState(current_soc_percent=99)
        )
        assert d.mode_intent == "idle"

    def test_opt_in_restores_self_consumption(self):
        c = Controller(ControllerConfig(pv_charge_allow_discharge=True), InverterConfig())
        d = c.decide(_pv_charge(), SystemState(current_soc_percent=45))
        assert d.mode_intent == "self_consumption"
        assert d.hold_fuse_relief_ok is False


class TestSafetyExceptionsIntact:
    def test_min_soc_latch_idles_and_is_not_relief_eligible(self, controller):
        state = SystemState(current_soc_percent=12, min_soc_percent=10, min_soc_latched=True)
        d = controller.decide(_pv_charge(), state)
        assert d.mode_intent == "idle"
        assert d.hold_fuse_relief_ok is False

    def test_fuse_relief_may_release_the_idle(self, controller):
        d = controller.decide(_pv_charge(), SystemState(current_soc_percent=45))
        assert d.hold_fuse_relief_ok is True
        apply_hold_fuse_relief(d)
        assert d.mode_intent == "self_consumption"
        assert "fuse relief" in d.reason

    def test_ev_isolation_not_relief_eligible(self, controller):
        d = controller.decide(_pv_charge(ev_isolation=True), SystemState(current_soc_percent=45))
        assert d.mode_intent == "idle"
        assert d.hold_fuse_relief_ok is False

    def test_ev_slot_not_relief_eligible(self, controller):
        d = controller.decide(
            _pv_charge(ev_charging_kw=7.0), SystemState(current_soc_percent=45)
        )
        assert d.mode_intent == "idle"
        assert d.hold_fuse_relief_ok is False

    def test_grid_charge_branch_untouched(self, controller):
        d = controller.decide(_pv_charge(pv_kw=0.5), SystemState(current_soc_percent=45))
        assert d.mode_intent == "charge"

    def test_hold_slot_still_idle(self, controller):
        d = controller.decide(
            SlotPlan(soc_target=50, pv_kw=4.0, load_kw=1.0), SystemState(current_soc_percent=70)
        )
        assert d.mode_intent == "idle"
        assert d.hold_fuse_relief_ok is True

    def test_planned_discharge_still_self_consumes(self, controller):
        d = controller.decide(
            SlotPlan(discharge_kw=1.5, soc_target=20), SystemState(current_soc_percent=60)
        )
        assert d.mode_intent == "self_consumption"


class TestConfig:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("false", False), ('"false"', False), ('"true"', True), ("null", False)],
    )
    def test_parsing(self, tmp_path, raw, expected):
        p = tmp_path / "config.yaml"
        p.write_text(
            f"executor:\n  controller:\n    pv_charge_allow_discharge: {raw}\n", encoding="utf-8"
        )
        assert load_executor_config(str(p)).controller.pv_charge_allow_discharge is expected

    def test_default_off(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("executor:\n  enabled: true\n", encoding="utf-8")
        assert load_executor_config(str(p)).controller.pv_charge_allow_discharge is False
        default = Path(__file__).parents[2] / "config.default.yaml"
        assert load_executor_config(str(default)).controller.pv_charge_allow_discharge is False
