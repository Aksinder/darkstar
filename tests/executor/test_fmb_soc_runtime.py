"""Tests for the FMB SoC estimator runtime wiring (config parse + a run cycle)."""

from __future__ import annotations

from typing import Any

import pytest

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

    async def get_state_value(self, entity_id: str) -> str | None:
        v = self.states.get(entity_id)
        return None if v is None else str(v)

    async def set_state(self, entity_id: str, state: str, attributes: dict[str, Any]) -> bool:
        self.published[entity_id] = (state, attributes)
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
async def test_disabled_is_noop():
    cfg = parse_fmb_soc_config({"fmb_soc_estimator": {"enabled": False}})
    assert cfg is not None
    est = FmbSocEstimator(cfg)
    ha = _FakeHA({})
    r = await est.run(ha, now_ts=1.0)
    assert r == {"enabled": False}
    assert ha.published == {}
