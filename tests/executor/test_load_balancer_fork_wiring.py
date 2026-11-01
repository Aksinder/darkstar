"""Fork wiring of the ported per-phase balancer: config keys, bridge, the
observe-only default and the narrow actuation paths.

Upstream's own tests (test_load_balancer*.py, test_fuse_model.py) cover the
decision core; these cover what is ours: how this site's config maps onto it
and that nothing is written unless load_balancing.actuate is true — and then
only through the Easee dynamic limit and existing HA switches.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from executor.config import (
    BalancedLoadConfig,
    LoadBalancingConfig,
    load_executor_config,
)
from executor.engine import ExecutorEngine
from executor.load_balancer import (
    EVBalancerOutput,
    LoadBalancer,
    LoadBalancerStatus,
    ShedLoadOutput,
)
from executor.load_balancer_actuator import LoadBalancerActuator
from executor.load_balancer_bridge import (
    battery_entry,
    ev_entries,
    phase_to_int,
    read_phase_samples,
    reading_timestamp,
    shed_entries,
    status_payload,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- config -------------------------------------------------------------------

LIVE_LIKE = """
timezone: Europe/Stockholm
system:
  has_battery: true
ev_chargers:
  - {id: easee_fmb, enabled: true, type: binary}
  - {id: tesla, enabled: true, type: binary}
executor:
  enabled: true
  ev_surplus:
    chargers:
      - {id: easee_fmb, easee_device_id: dev123, switch_entity: switch.easee_enabled}
      - {id: tesla, current_entity: number.tesla_current, switch_entity: switch.tesla}
      - {id: dumb, switch_entity: switch.dumb}
load_balancing:
  enabled: true
  protected_fuse_a: 25
  fuse_preset: diazed_gg
  loads:
    - {device_type: water_heater, device_id: spa, phases: [b], power_kw: 1.8}
  give_way_order:
    - {kind: shed, id: spa}
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return load_executor_config(str(path)).load_balancing


class TestConfig:
    def test_defaults_are_observe_only(self):
        cfg = LoadBalancingConfig()
        assert cfg.enabled is False
        assert cfg.actuate is False
        assert cfg.fuse_preset == "typ_c"
        assert cfg.target_margin_percent == 85.0
        assert cfg.nominal_voltage_v == 230.0

    def test_live_like_config(self, tmp_path):
        lb = _load(tmp_path, LIVE_LIKE)
        assert lb.enabled is True
        assert lb.actuate is False
        assert lb.main_fuse_a == 25
        assert lb.fuse_preset == "diazed_gg"
        assert lb.loads[0].phases == [2]  # meter name "b" = L2
        assert lb.loads[0].power_kw == 1.8

    def test_servo_chargers_survive_heal(self, tmp_path):
        """Our cars are top-level type: binary; without the servo lookup the heal
        dropped every charger and the balancer guarded no car."""
        lb = _load(tmp_path, LIVE_LIKE)
        order = [(e.kind, e.id) for e in lb.give_way_order]
        assert order == [
            ("charger", "easee_fmb"),
            ("charger", "tesla"),
            ("shed", "spa"),
            ("battery", "battery"),
        ]

    def test_actuate_needs_a_literal_true(self, tmp_path):
        lb = _load(
            tmp_path,
            LIVE_LIKE.replace(
                "enabled: true\n  protected", "enabled: true\n  actuate: yes-please\n  protected"
            ),
        )
        assert lb.actuate is False
        lb = _load(
            tmp_path,
            LIVE_LIKE.replace(
                "enabled: true\n  protected", "enabled: true\n  actuate: true\n  protected"
            ),
        )
        assert lb.actuate is True

    def test_invalid_fuse_preset_falls_back(self, tmp_path):
        lb = _load(tmp_path, LIVE_LIKE.replace("diazed_gg", "gL"))
        assert lb.fuse_preset == "typ_c"

    def test_custom_curve(self, tmp_path):
        text = LIVE_LIKE.replace(
            "fuse_preset: diazed_gg",
            "fuse_preset: custom\n  fuse_custom_points:\n"
            "    - {overload_percent: 300, time_s: 4}\n"
            "    - {overload_percent: 130, time_s: 1200}",
        )
        lb = _load(tmp_path, text)
        assert lb.fuse_preset == "custom"
        assert [(p.overload_percent, p.time_s) for p in lb.fuse_custom_points] == [
            (130.0, 1200.0),
            (300.0, 4.0),
        ]

    def test_invalid_protected_fuse_unconfigures(self, tmp_path):
        assert (
            _load(
                tmp_path, LIVE_LIKE.replace("protected_fuse_a: 25", "protected_fuse_a: 25.5")
            ).main_fuse_a
            is None
        )

    @pytest.mark.parametrize(("legacy", "expected"), [(90, 85.0), (80, 80.0)])
    def test_legacy_resume_margin(self, tmp_path, legacy, expected):
        text = LIVE_LIKE.replace(
            "enabled: true\n  protected",
            f"enabled: true\n  resume_margin_percent: {legacy}\n  protected",
        )
        assert _load(tmp_path, text).target_margin_percent == expected

    def test_unknown_load_phase_is_dropped(self, tmp_path):
        assert _load(tmp_path, LIVE_LIKE.replace("phases: [b]", "phases: [b, x]")).loads[
            0
        ].phases == [2]


# --- bridge -------------------------------------------------------------------


class FakeHA:
    def __init__(self, states: dict[str, Any] | None = None):
        self.states = states or {}
        self.calls: list[tuple[str, str, str | None, dict[str, Any] | None]] = []

    async def get_state(self, entity: str):
        value = self.states.get(entity)
        if isinstance(value, Exception):
            raise value
        return value

    async def call_service(self, domain, service, entity_id=None, data=None, **_):
        self.calls.append((domain, service, entity_id, data))
        return True


NOW = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)


class TestBridge:
    def test_meter_names_map_to_phases(self):
        assert [phase_to_int(x) for x in ("a", "B", "c")] == [1, 2, 3]

    def test_reading_timestamp_prefers_last_reported(self):
        st = {
            "last_reported": "2026-11-01T18:00:00+00:00",
            "last_updated": "2026-11-01T17:00:00+00:00",
        }
        assert reading_timestamp(st) == NOW
        assert reading_timestamp({"last_updated": "2026-11-01T18:00:00+00:00"}) == NOW
        # Naive timestamps are not guessed into a timezone.
        assert reading_timestamp({"last_updated": "2026-11-01T18:00:00"}) is None

    def test_per_phase_samples_carry_real_timestamps(self):
        ha = FakeHA(
            {
                "s.a": {"state": "-12.5", "last_reported": "2026-11-01T18:00:00+00:00"},
                "s.b": {"state": "unavailable", "last_reported": "2026-11-01T18:00:00+00:00"},
                "s.c": RuntimeError("404"),
            }
        )
        cur, ts = asyncio.run(read_phase_samples(ha, {"a": "s.a", "b": "s.b", "c": "s.c"}))
        assert cur == {1: 12.5}  # magnitude: export heats the fuse too
        assert ts == {1: NOW}

    def test_ev_setpoint_zero_is_off(self):
        c = SimpleNamespace(id="x", phase_map=("a",), min_current_a=6, max_current_a=16)
        ev = ev_entries([c], setpoints_a={"x": 0.0}, targets_a={"x": 10.0})["x"]
        assert ev.current_setpoint_a is None
        assert ev.planner_target_a == 10
        assert ev.phases == [1]

    def test_shed_power_source(self):
        loads = [
            BalancedLoadConfig(device_id="spa", phases=[2], power_kw=1.8),
            BalancedLoadConfig(device_id="vvb", phases=[2]),
        ]
        out = shed_entries(loads, power_w={"vvb": 1500.0})
        assert (out["spa"].power_w, out["spa"].power_source) == (1800.0, "configured")
        assert (out["vvb"].power_w, out["vvb"].power_source) == (1500.0, "sensor")
        assert shed_entries(loads)["vvb"].power_source == "worst_case"

    def test_battery_entry(self):
        ctrl = SimpleNamespace(min_charge_w=500.0, round_step_w=100.0)
        charge = SimpleNamespace(mode_intent="charge", charge_value=9500.0)
        b = battery_entry(charge, ctrl, grid_w=6000.0, battery_w=9400.0, control_unit="W")
        assert b is not None and b.active
        assert (b.planned_w, b.reducible_w, b.measured_w) == (9500.0, 6000.0, 9400.0)
        idle = SimpleNamespace(mode_intent="self_consumption", charge_value=0.0)
        assert not battery_entry(idle, ctrl, grid_w=0, battery_w=0, control_unit="W").active
        assert battery_entry(charge, ctrl, grid_w=0, battery_w=0, control_unit="A") is None

    def test_status_payload_exposes_trip_margin(self):
        status = LoadBalancerStatus(
            enabled=True,
            state="idle",
            reason="ok",
            main_fuse_a=25,
            phase_current_a={1: 20.0},
            phase_headroom_a={1: 5.0},
            phase_heat_percent={1: 30.0, 2: 0.0, 3: 0.0},
            fuse_safety_percent=65.0,
        )
        p = status_payload(status, actuate=False)
        assert p["mode"] == "observe"
        assert p["trip_margin_percent"]["L1"] == 35.0
        assert p["phase_headroom_a"] == {"L1": 5.0}
        assert status_payload(None, actuate=False)["state"] == "not_run"


# --- actuator -----------------------------------------------------------------


def _status(evs=(), sheds=()) -> LoadBalancerStatus:
    return LoadBalancerStatus(
        enabled=True,
        state="throttling",
        reason="",
        main_fuse_a=25,
        phase_current_a={},
        phase_headroom_a={},
        ev_outputs=list(evs),
        shed_outputs=list(sheds),
    )


EASEE = SimpleNamespace(
    id="easee_fmb", easee_device_id="dev123", switch_entity="switch.easee_enabled"
)
TESLA = SimpleNamespace(id="tesla", easee_device_id=None, switch_entity="switch.tesla")


def _apply(act, ha, status, servo, loads=None, shadow=False):
    return asyncio.run(
        act.apply(
            ha,
            status,
            chargers={"easee_fmb": EASEE, "tesla": TESLA},
            servo_a=servo,
            loads=loads or {},
            shadow=shadow,
        )
    )


class TestActuator:
    def test_easee_cap_uses_only_the_dynamic_limit(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(
            act, ha, _status([EVBalancerOutput("easee_fmb", 10, "throttling")]), {"easee_fmb": 16.0}
        )
        assert ha.calls == [
            (
                "easee",
                "set_charger_dynamic_limit",
                None,
                {"device_id": "dev123", "current": 10, "time_to_live": 0},
            ),
        ]
        assert act.applied_setpoint_a("easee_fmb", 16.0) == 10

    def test_easee_pause_is_dynamic_zero_never_the_enabled_switch(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(
            act, ha, _status([EVBalancerOutput("easee_fmb", None, "paused")]), {"easee_fmb": 16.0}
        )
        assert ha.calls == [
            (
                "easee",
                "set_charger_dynamic_limit",
                None,
                {"device_id": "dev123", "current": 0, "time_to_live": 0},
            ),
        ]
        assert all(c[0] == "easee" and c[1] == "set_charger_dynamic_limit" for c in ha.calls)

    def test_release_hands_back_the_servo_command(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(
            act, ha, _status([EVBalancerOutput("easee_fmb", 10, "throttling")]), {"easee_fmb": 16.0}
        )
        _apply(act, ha, _status([EVBalancerOutput("easee_fmb", 16, "idle")]), {"easee_fmb": 16.0})
        assert ha.calls[-1][3] == {"device_id": "dev123", "current": 16, "time_to_live": 0}
        assert act.applied_setpoint_a("easee_fmb", 16.0) == 16.0

    def test_same_cap_is_not_rewritten(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        for _ in range(3):
            _apply(
                act,
                ha,
                _status([EVBalancerOutput("easee_fmb", 10, "throttling")]),
                {"easee_fmb": 16.0},
            )
        assert len(ha.calls) == 1

    def test_a_servo_rewrite_voids_the_cap(self):
        """The servo wrote the register itself; our cap is no longer in force."""
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(
            act, ha, _status([EVBalancerOutput("easee_fmb", 10, "throttling")]), {"easee_fmb": 16.0}
        )
        assert act.applied_setpoint_a("easee_fmb", 14.0) == 14.0
        _apply(
            act, ha, _status([EVBalancerOutput("easee_fmb", 10, "throttling")]), {"easee_fmb": 14.0}
        )
        assert len(ha.calls) == 2

    def test_tesla_amps_are_never_written(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(act, ha, _status([EVBalancerOutput("tesla", 8, "throttling")]), {"tesla": 16.0})
        assert ha.calls == []

    def test_tesla_pause_and_release_use_its_switch(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(act, ha, _status([EVBalancerOutput("tesla", None, "paused")]), {"tesla": 16.0})
        _apply(act, ha, _status([EVBalancerOutput("tesla", 16, "idle")]), {"tesla": 16.0})
        assert ha.calls == [
            ("switch", "turn_off", "switch.tesla", None),
            ("switch", "turn_on", "switch.tesla", None),
        ]

    def test_servo_off_needs_no_release(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        _apply(act, ha, _status([EVBalancerOutput("tesla", None, "paused")]), {"tesla": 16.0})
        _apply(act, ha, _status([EVBalancerOutput("tesla", None, "idle")]), {"tesla": None})
        _apply(act, ha, _status([EVBalancerOutput("tesla", 16, "idle")]), {"tesla": 16.0})
        assert ha.calls == [("switch", "turn_off", "switch.tesla", None)]

    def test_shadow_sends_nothing(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        done = _apply(
            act,
            ha,
            _status([EVBalancerOutput("easee_fmb", 10, "throttling")]),
            {"easee_fmb": 16.0},
            shadow=True,
        )
        assert ha.calls == []
        assert "shadow" in done[0]

    def test_shed_only_with_an_explicit_switch_entity(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        loads = {
            "spa": BalancedLoadConfig(device_id="spa", phases=[2], entity="switch.spa"),
            "vvb": BalancedLoadConfig(device_id="vvb", phases=[2]),
            "num": BalancedLoadConfig(device_id="num", phases=[2], entity="number.x"),
        }
        shedding = [ShedLoadOutput(i, "water_heater", True, actively_shedding=True) for i in loads]
        _apply(act, ha, _status(sheds=shedding), {}, loads=loads)
        assert ha.calls == [("switch", "turn_off", "switch.spa", None)]
        _apply(
            act, ha, _status(sheds=[ShedLoadOutput("spa", "water_heater", False)]), {}, loads=loads
        )
        assert ha.calls[-1] == ("switch", "turn_on", "switch.spa", None)

    def test_planned_off_is_not_ours_to_restore(self):
        ha, act = FakeHA(), LoadBalancerActuator()
        loads = {"spa": BalancedLoadConfig(device_id="spa", phases=[2], entity="switch.spa")}
        _apply(
            act, ha, _status(sheds=[ShedLoadOutput("spa", "water_heater", True)]), {}, loads=loads
        )
        _apply(
            act, ha, _status(sheds=[ShedLoadOutput("spa", "water_heater", False)]), {}, loads=loads
        )
        assert ha.calls == []


# --- engine wiring: observe-only by default ------------------------------------


def _engine_stub(actuate: bool, ha: FakeHA, shadow: bool = False):
    cfg = LoadBalancingConfig(enabled=True, main_fuse_a=25, actuate=actuate)
    from executor.config import GiveWayOrderEntry

    cfg.give_way_order = [GiveWayOrderEntry(kind="charger", id="easee_fmb")]
    srv = SimpleNamespace(
        cfg=SimpleNamespace(
            fuse_phase_entities={"a": "s.a", "b": "s.b", "c": "s.c"},
            chargers=[
                SimpleNamespace(
                    id="easee_fmb",
                    phase_map=("a",),
                    min_current_a=6,
                    max_current_a=16,
                    controllable=True,
                    easee_device_id="dev123",
                    switch_entity="switch.easee_enabled",
                )
            ],
        ),
        commanded_current_a=lambda cid: 16.0,
        last_grid_w=8000.0,
        last_battery_w=0.0,
    )
    stub = SimpleNamespace(
        _ev_surplus=srv,
        config=SimpleNamespace(
            load_balancing=cfg,
            interval_seconds=60,
            timezone="Europe/Stockholm",
            shadow_mode=shadow,
            controller=SimpleNamespace(min_charge_w=500.0, round_step_w=100.0),
            inverter=SimpleNamespace(control_unit="W"),
        ),
        _load_balancer=LoadBalancer(cfg, interval_seconds=60),
        _balancer_actuator=LoadBalancerActuator(),
        _balancer_applied=[],
        _last_balancer_status=None,
        _last_logged_balancer_state=None,
        ha_client=ha,
        inverter_profile=None,
    )
    return stub


def _set_phases(ha: FakeHA, amps: dict[str, float]):
    stamp = datetime.now(UTC).isoformat()
    for name, value in amps.items():
        ha.states[f"s.{name}"] = {"state": str(value), "last_reported": stamp}


class TestEngineObserve:
    def test_observe_only_never_writes(self, caplog):
        ha = FakeHA()
        stub = _engine_stub(actuate=False, ha=ha)
        _set_phases(ha, {"a": 60.0, "b": 5.0, "c": 5.0})  # 2.4x: trip within one tick = urgent
        decision = SimpleNamespace(mode_intent="self_consumption", charge_value=0.0)
        with caplog.at_level(logging.INFO):
            asyncio.run(ExecutorEngine._observe_load_balancer(stub, decision))
        assert ha.calls == []
        status = stub._last_balancer_status
        assert status is not None and status.ev_outputs[0].target_a is None
        assert "OBSERVE, not acting" in caplog.text

    def test_actuate_applies_through_the_easee_dynamic_limit(self):
        ha = FakeHA()
        stub = _engine_stub(actuate=True, ha=ha)
        _set_phases(ha, {"a": 60.0, "b": 5.0, "c": 5.0})
        decision = SimpleNamespace(mode_intent="self_consumption", charge_value=0.0)
        asyncio.run(ExecutorEngine._observe_load_balancer(stub, decision))
        assert ha.calls == [
            (
                "easee",
                "set_charger_dynamic_limit",
                None,
                {"device_id": "dev123", "current": 0, "time_to_live": 0},
            ),
        ]

    def test_stale_phase_is_judged_by_its_own_timestamp(self):
        ha = FakeHA()
        stub = _engine_stub(actuate=False, ha=ha)
        _set_phases(ha, {"a": 10.0, "b": 5.0, "c": 5.0})
        old = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        ha.states["s.a"]["last_reported"] = old
        decision = SimpleNamespace(mode_intent="self_consumption", charge_value=0.0)
        asyncio.run(ExecutorEngine._observe_load_balancer(stub, decision))
        out = stub._last_balancer_status.ev_outputs[0]
        assert out.state == "stale_fallback"
        assert "L1" in out.reason
