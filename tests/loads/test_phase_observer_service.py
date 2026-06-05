"""Tests for the phase-observer orchestration service."""

from datetime import datetime, timedelta

import pytest

from backend.learning.cycle_publisher_service import (
    PhaseObserverService,
    PhaseSources,
    TrackedPhaseDevice,
    build_phase_observer_from_config,
)

NOW = datetime(2026, 6, 2, 2, 0, 0)

# Single-phase device on A toggling off/on, with a varying balanced inverter S.
_TOGGLE = [0, 0, 3000, 3000, 0, 0, 3000, 3000, 0, 0, 3000, 3000, 0, 0, 3000, 3000]
_INV = [0, 1500, 3000, 4500, 6000, 4500, 3000, 1500, 0, 2000, 4000, 6000, 4000, 2000, 0, 2500]


def _rows(values):
    """HA history rows (minimal_response): {state, last_updated} at 1-min spacing."""
    return [
        {"state": str(float(v)), "last_updated": (NOW + timedelta(minutes=i)).isoformat()}
        for i, v in enumerate(values)
    ]


def _grids():
    """grid_phase = load_phase - S/3; device load is all on phase A."""
    ga, gb, gc = [], [], []
    for la, s in zip(_TOGGLE, _INV, strict=True):
        share = s / 3.0
        ga.append(la - share)
        gb.append(-share)
        gc.append(-share)
    return ga, gb, gc


class TestBuildFromConfig:
    def test_disabled_returns_none(self):
        assert build_phase_observer_from_config({"phase_observer": {"enabled": False}}) is None

    def test_missing_phase_sensors_returns_none(self):
        assert build_phase_observer_from_config({"phase_observer": {"enabled": True}}) is None

    def test_includes_explicit_and_deferrable_devices(self):
        config = {
            "phase_observer": {
                "enabled": True,
                "phase_a_sensor": "sensor.a",
                "phase_b_sensor": "sensor.b",
                "phase_c_sensor": "sensor.c",
                "battery_power_sensor": "sensor.bat",
                "battery_power_scale": -1.0,
                "devices": [
                    {"id": "easee", "name": "Easee", "power_sensor": "sensor.easee"},
                ],
            },
            "deferrable_loads": [
                {"id": "tvatt", "name": "Tvätt", "power_sensor": "sensor.tvatt"},
                {"id": "no_power", "name": "X"},  # skipped: no power sensor
            ],
        }
        built = build_phase_observer_from_config(config)
        assert built is not None
        devices, sources = built
        ids = {d.id for d in devices}
        assert ids == {"easee", "tvatt"}
        assert sources.phase_a_entity == "sensor.a"
        assert sources.inverter_entities == (("sensor.bat", -1.0),)

    def test_explicit_device_not_duplicated_by_deferrable(self):
        config = {
            "phase_observer": {
                "enabled": True,
                "phase_a_sensor": "sensor.a",
                "phase_b_sensor": "sensor.b",
                "phase_c_sensor": "sensor.c",
                "devices": [{"id": "easee", "name": "Easee", "power_sensor": "sensor.easee"}],
            },
            "deferrable_loads": [
                {"id": "easee", "name": "dup", "power_sensor": "sensor.other"},
            ],
        }
        built = build_phase_observer_from_config(config)
        assert built is not None
        devices, _ = built
        assert [d.id for d in devices] == ["easee"]
        assert devices[0].power_entity == "sensor.easee"  # explicit wins


class _Capture:
    def __init__(self):
        self.sensors = []
        self.model = None

    async def publish(self, sensors):
        self.sensors = sensors
        return len(sensors)

    async def persist(self, model):
        self.model = model


@pytest.mark.asyncio
async def test_run_once_learns_phase_and_publishes_and_persists():
    ga, gb, gc = _grids()
    history = {
        "sensor.a": _rows(ga),
        "sensor.b": _rows(gb),
        "sensor.c": _rows(gc),
        "sensor.bat": _rows(_INV),
        "sensor.easee": _rows(_TOGGLE),
    }

    async def fetch_history(entity_id, hours):
        return history[entity_id]

    async def fetch_float(entity_id):
        return history[entity_id][-1]["state"] and float(history[entity_id][-1]["state"])

    cap = _Capture()
    svc = PhaseObserverService(
        devices=[TrackedPhaseDevice(id="easee", name="Easee", power_entity="sensor.easee")],
        sources=PhaseSources("sensor.a", "sensor.b", "sensor.c", (("sensor.bat", 1.0),)),
        fetch_history=fetch_history,
        fetch_float=fetch_float,
        publish=cap.publish,
        persist=cap.persist,
        now_fn=lambda: NOW,
    )
    published = await svc.run_once()

    assert published > 0
    by_id = {s.object_id: s for s in cap.sensors}
    # Per-phase load + imbalance sensors exist.
    assert "darkstar_phase_a_load" in by_id
    assert "darkstar_phase_imbalance" in by_id
    # The device was mapped to phase A.
    dev = by_id["darkstar_easee_phase"]
    assert dev.state == "A"
    assert dev.attributes["load_type"] == "single"
    # The recommendation sensor is always published (a single load can't be
    # rebalanced by moving it, so here it reports "balanserat").
    assert "darkstar_phase_recommendation" in by_id
    # Fractions + recommendations were persisted for the planner/dashboard.
    assert cap.model is not None
    assert set(cap.model["fractions"]) == {"A", "B", "C"}
    assert cap.model["devices"][0]["phase"] == "A"
    assert "recommendations" in cap.model
