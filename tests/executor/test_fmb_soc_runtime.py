"""Tests for the FMB SoC estimator runtime wiring (config parse + a run cycle)."""

from __future__ import annotations

from typing import Any

import pytest

from executor.actions import HACallError
from executor.fmb_soc_runtime import FmbSocEstimator, parse_fmb_soc_config


def test_parse_absent_returns_none():
    assert parse_fmb_soc_config({}) is None
    assert parse_fmb_soc_config({"fmb_soc_estimator": "nope"}) is None


def test_parse_reads_fields():
    cfg = parse_fmb_soc_config(
        {
            "fmb_soc_estimator": {
                "enabled": True,
                "capacity_kwh": 28,
                "prior_consumption_kwh_per_day": 5.6,
                "lifetime_energy_entity": "sensor.easee_niska_it_lifetime_energy",
                "power_entity": "sensor.easee_niska_it_power",
                "plug_entity": "binary_sensor.easee_ev_plugged_in",
                "enabled_switch_entity": "switch.easee_niska_it_charger_enabled",
                "dynamic_limit_entity": "sensor.easee_niska_it_dynamic_charger_limit",
                "status_entity": "sensor.easee_niska_it_status",
            }
        }
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.pure.capacity_kwh == 28
    assert cfg.pure.prior_consumption_kwh_per_day == 5.6
    assert cfg.lifetime_energy_entity == "sensor.easee_niska_it_lifetime_energy"
    assert cfg.publish_entity_id == "sensor.darkstar_fmb_soc_estimate"


class _FakeHA:
    """Minimal stand-in: serves states and records published sensors."""

    def __init__(self, states: dict[str, Any]):
        self.states = states
        self.published: dict[str, tuple[str, dict[str, Any]]] = {}
        self.service_calls: list[tuple[str, str, str | None, dict[str, Any]]] = []

    async def get_state_value(self, entity_id: str) -> str | None:
        v = self.states.get(entity_id)
        return None if v is None else str(v)

    async def set_state(self, entity_id: str, state: str, attributes: dict[str, Any]) -> bool:
        self.published[entity_id] = (state, attributes)
        return True

    async def call_service(
        self, domain: str, service: str, entity_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        self.service_calls.append((domain, service, entity_id, data or {}))
        return True


@pytest.mark.asyncio
async def test_run_publishes_and_advances(tmp_path):
    cfg = parse_fmb_soc_config(
        {
            "fmb_soc_estimator": {
                "enabled": True,
                "capacity_kwh": 28,
                "charge_efficiency": 1.0,
                "initial_soc": 50,
                "lifetime_energy_entity": "sensor.lt",
                "power_entity": "sensor.pw",
                "plug_entity": "binary_sensor.plug",
                "enabled_switch_entity": "switch.en",
                "dynamic_limit_entity": "sensor.lim",
                "status_entity": "sensor.st",
                "save_interval_s": 0,
                "state_path": str(tmp_path / "fmb_state.json"),
            }
        }
    )
    assert cfg is not None
    est = FmbSocEstimator(cfg)
    ha = _FakeHA(
        {
            "sensor.lt": 100.0,
            "sensor.pw": 0.0,
            "binary_sensor.plug": "on",
            "switch.en": "off",
            "sensor.lim": 0.0,
            "sensor.st": "awaiting_start",
        }
    )

    # Tick 1: baseline only.
    r1 = await est.run(ha, now_ts=1000.0)
    assert r1["enabled"] is True
    assert "sensor.darkstar_fmb_soc_estimate" in ha.published
    assert ha.published["sensor.darkstar_fmb_soc_estimate"][0] == "50.0"

    # Tick 2: +2.8 kWh delivered while charging → +10 % → 60 %.
    ha.states["sensor.lt"] = 102.8
    ha.states["sensor.pw"] = 3000.0
    r2 = await est.run(ha, now_ts=1060.0)
    assert abs(r2["soc"] - 60.0) < 1e-6
    assert ha.published["sensor.darkstar_fmb_soc_estimate"][0] == "60.0"

    # State persisted to disk.
    est2 = FmbSocEstimator(cfg)
    loaded = est2._load_state()
    assert abs(loaded.soc_pct - 60.0) < 1e-6


@pytest.mark.asyncio
async def test_writeback_to_input_number_and_suppressed_sensor(tmp_path):
    cfg = parse_fmb_soc_config(
        {
            "fmb_soc_estimator": {
                "enabled": True,
                "capacity_kwh": 28,
                "charge_efficiency": 1.0,
                "initial_soc": 50,
                "seed_soc": 27,
                "publish_entity_id": "",  # suppress the duplicate Darkstar SoC sensor
                "writeback_entity": "input_number.fmb_soc",
                "lifetime_energy_entity": "sensor.lt",
                "power_entity": "sensor.pw",
                "plug_entity": "binary_sensor.plug",
                "enabled_switch_entity": "switch.en",
                "status_entity": "sensor.st",
                "save_interval_s": 0,
                "state_path": str(tmp_path / "fmb_state.json"),
            }
        }
    )
    assert cfg is not None
    assert cfg.publish_entity_id == ""
    assert cfg.writeback_entity == "input_number.fmb_soc"
    est = FmbSocEstimator(cfg)
    ha = _FakeHA(
        {"sensor.lt": 100.0, "sensor.pw": 0.0, "binary_sensor.plug": "on",
         "switch.en": "off", "sensor.st": "idle"}
    )
    await est.run(ha, now_ts=1000.0)
    # SoC sensor suppressed; rate sensor still published.
    assert "sensor.darkstar_fmb_soc_estimate" not in ha.published
    assert "sensor.darkstar_fmb_consumption_rate" in ha.published
    # Seed 27 written into the input_number via input_number.set_value.
    sets = [c for c in ha.service_calls if c[:2] == ("input_number", "set_value")]
    assert sets, "expected an input_number.set_value writeback"
    assert sets[-1][2] == "input_number.fmb_soc"
    assert sets[-1][3]["value"] == 27

    # Second tick, unchanged SoC → no duplicate write (integer-change throttle).
    ha.service_calls.clear()
    await est.run(ha, now_ts=1010.0)
    assert not [c for c in ha.service_calls if c[:2] == ("input_number", "set_value")]


def _wb_cfg(tmp_path, **extra):
    base = {
        "enabled": True, "capacity_kwh": 28, "charge_efficiency": 1.0, "initial_soc": 50,
        "writeback_entity": "input_number.fmb_soc", "publish_entity_id": "",
        "lifetime_energy_entity": "sensor.lt", "power_entity": "sensor.pw",
        "plug_entity": "binary_sensor.plug", "enabled_switch_entity": "switch.en",
        "status_entity": "sensor.st", "save_interval_s": 0,
        "state_path": str(tmp_path / "s.json"),
    }
    base.update(extra)
    return parse_fmb_soc_config({"fmb_soc_estimator": base})


@pytest.mark.asyncio
async def test_writeback_failure_does_not_crash(tmp_path):
    cfg = _wb_cfg(tmp_path)
    est = FmbSocEstimator(cfg)
    ha = _FakeHA({"sensor.lt": 100.0, "sensor.pw": 0.0, "binary_sensor.plug": "on",
                  "switch.en": "off", "sensor.st": "idle"})

    async def _boom(*_a, **_k):
        raise HACallError(message="nope")

    ha.call_service = _boom  # type: ignore[assignment]
    r = await est.run(ha, now_ts=1000.0)  # must NOT raise
    assert r["enabled"] is True
    assert "sensor.darkstar_fmb_consumption_rate" in ha.published  # rate still published


@pytest.mark.asyncio
async def test_shadow_suppresses_writeback(tmp_path):
    cfg = _wb_cfg(tmp_path)
    est = FmbSocEstimator(cfg)
    ha = _FakeHA({"sensor.lt": 100.0, "sensor.pw": 0.0, "binary_sensor.plug": "on",
                  "switch.en": "off", "sensor.st": "idle"})
    await est.run(ha, now_ts=1000.0, shadow=True)
    assert not [c for c in ha.service_calls if c[:2] == ("input_number", "set_value")]


def test_correction_equal_to_writeback_is_disabled(tmp_path):
    cfg = _wb_cfg(tmp_path, correction_entity="input_number.fmb_soc")  # SAME as writeback
    est = FmbSocEstimator(cfg)
    assert est.cfg.correction_entity is None  # auto-disabled to avoid self-adoption


@pytest.mark.asyncio
async def test_disabled_is_noop():
    cfg = parse_fmb_soc_config({"fmb_soc_estimator": {"enabled": False}})
    assert cfg is not None
    est = FmbSocEstimator(cfg)
    ha = _FakeHA({})
    r = await est.run(ha, now_ts=1.0)
    assert r == {"enabled": False}
    assert ha.published == {}
