"""Hold slots idle (ratchet fix) — but may relieve the fuse, and dust is a Hold.

Review findings on 2a90a35c:
* MEDIUM: a Hold slot always idles, so the battery never relieved a phase near the
  25 A main fuse. The controller now marks a plain Hold idle as relief-eligible
  (not at the min-SoC floor, no EV isolation / EV slot) and the engine flips it to
  self_consumption while the fuse guard sees an importing phase at/over budget.
* LOW: the schedule formatter rounds to 2 decimals, so 0.005-0.01 kW of solver dust
  reads 0.01 and, with export > 0, used to hit the "both flows" branch ->
  self_consumption -> the ratchet reopened in sunny export slots.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executor.config import ControllerConfig, InverterConfig
from executor.controller import Controller, ControllerDecision, apply_hold_fuse_relief
from executor.engine import HOLD_FUSE_RELIEF_HOLD_S, ExecutorEngine
from executor.override import SlotPlan, SystemState


@pytest.fixture
def controller() -> Controller:
    return Controller(ControllerConfig(), InverterConfig())


# --- LOW: rounding dust in a sunny export slot is a Hold --------------------------


def test_dust_discharge_with_export_is_a_hold(controller):
    slot = SlotPlan(export_kw=2.5, discharge_kw=0.01, charge_kw=0.0, soc_target=60)
    d = controller.decide(slot, SystemState(current_soc_percent=80))
    assert d.mode_intent == "idle"
    assert d.hold_fuse_relief_ok is True


def test_real_discharge_with_export_still_self_consumes(controller):
    slot = SlotPlan(export_kw=2.5, discharge_kw=0.3, charge_kw=0.0, soc_target=60)
    d = controller.decide(slot, SystemState(current_soc_percent=80))
    assert d.mode_intent == "self_consumption"
    assert d.hold_fuse_relief_ok is False


def test_dust_overlap_never_exports_even_with_zero_export_min():
    c = Controller(ControllerConfig(export_min_battery_kw=0.0), InverterConfig())
    slot = SlotPlan(export_kw=2.5, discharge_kw=0.01, soc_target=60)
    assert c.decide(slot, SystemState(current_soc_percent=80)).mode_intent == "idle"


# --- MEDIUM: relief eligibility ------------------------------------------------------


def test_hold_idle_is_relief_eligible(controller):
    d = controller.decide(SlotPlan(soc_target=50), SystemState(current_soc_percent=70))
    assert d.mode_intent == "idle"
    assert d.hold_fuse_relief_ok is True


def test_hold_at_min_soc_floor_is_not_eligible(controller):
    state = SystemState(current_soc_percent=10, min_soc_percent=10)
    d = controller.decide(SlotPlan(soc_target=10), state)
    assert d.mode_intent == "idle"
    assert d.hold_fuse_relief_ok is False


def test_hold_with_ev_slot_is_not_eligible(controller):
    d = controller.decide(
        SlotPlan(soc_target=50, ev_charging_kw=7.0), SystemState(current_soc_percent=70)
    )
    assert d.mode_intent == "idle"
    assert d.hold_fuse_relief_ok is False


def test_discharge_slot_at_target_idle_is_not_eligible(controller):
    d = controller.decide(
        SlotPlan(discharge_kw=2.0, soc_target=50), SystemState(current_soc_percent=50)
    )
    assert d.mode_intent == "idle"
    assert d.hold_fuse_relief_ok is False


def test_apply_hold_fuse_relief_flips_to_self_consumption():
    d = ControllerDecision(mode_intent="idle", max_charge=9000.0, hold_fuse_relief_ok=True)
    apply_hold_fuse_relief(d)
    assert d.mode_intent == "self_consumption"
    assert d.charge_value == 9000.0
    assert d.write_charge_current is True


def test_apply_hold_fuse_relief_ignores_ineligible():
    d = ControllerDecision(mode_intent="idle", max_charge=9000.0)
    apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"


# --- engine side: fuse guard readings decide ----------------------------------------


def _engine(phases, grid_w, *, fresh=True, enabled=True):
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._hold_fuse_relief_until = 0.0
    eng._ev_surplus = SimpleNamespace(
        cfg=SimpleNamespace(fuse_guard_enabled=enabled, fuse_stale_after_s=180.0),
        fuse_budget_a=23.0,
        last_phase_ok_ts=time.time() - (10 if fresh else 1000),
        last_phase_currents_a=phases,
        last_grid_w=grid_w,
    )
    return eng


def _hold():
    return ControllerDecision(mode_intent="idle", max_charge=9000.0, hold_fuse_relief_ok=True)


def test_engine_relieves_when_importing_phase_over_budget():
    eng = _engine({"a": 24.5, "b": 9.0, "c": 8.0}, 6000.0)
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "self_consumption"
    assert eng._hold_fuse_relief_until > time.time() + HOLD_FUSE_RELIEF_HOLD_S - 5


def test_engine_holds_relief_briefly_then_releases():
    eng = _engine({"a": 24.5, "b": 9.0, "c": 8.0}, 6000.0)
    eng._apply_hold_fuse_relief(_hold())
    eng._ev_surplus.last_phase_currents_a = {"a": 12.0, "b": 9.0, "c": 8.0}
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "self_consumption"  # anti-flap hold
    eng._hold_fuse_relief_until = time.time() - 1
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"


@pytest.mark.parametrize(
    ("phases", "grid_w", "fresh", "enabled"),
    [
        ({"a": 20.0, "b": 9.0, "c": 8.0}, 6000.0, True, True),  # under budget
        ({"a": 24.5, "b": 9.0, "c": 8.0}, -3000.0, True, True),  # exporting
        ({"a": 24.5, "b": 9.0, "c": 8.0}, 6000.0, False, True),  # stale readings
        ({"a": 24.5, "b": 9.0, "c": 8.0}, 6000.0, True, False),  # guard off
    ],
)
def test_engine_keeps_idle_without_a_fresh_import_overload(phases, grid_w, fresh, enabled):
    eng = _engine(phases, grid_w, fresh=fresh, enabled=enabled)
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"


def test_engine_never_flips_an_ineligible_idle():
    eng = _engine({"a": 30.0, "b": 9.0, "c": 8.0}, 6000.0)
    d = ControllerDecision(mode_intent="idle", max_charge=9000.0)  # e.g. min-SoC floor
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"


def test_engine_without_servo_is_a_noop():
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._hold_fuse_relief_until = 0.0
    eng._ev_surplus = None
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"


def test_engine_mock_servo_is_never_an_overload():
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._hold_fuse_relief_until = 0.0
    eng._ev_surplus = MagicMock()
    d = _hold()
    eng._apply_hold_fuse_relief(d)
    assert d.mode_intent == "idle"
