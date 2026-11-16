"""Spa bathing HA entities from schedule.json ``meta.spa_bathing``.

``PublishedSensor`` gained a ``domain`` (default ``sensor``) so the on-track flag can
be a ``binary_sensor``; every existing sensor keeps its entity id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.learning.cycle_publisher import PublishedSensor, build_spa_bathing_sensors

META: dict[str, Any] = {
    "heater_id": "spa",
    "mode": "shadow",
    "reason": None,
    "planned_at": "2026-11-16T12:00:00+01:00",
    "measured_temp_c": 30.5,
    "ambient_c": 4.0,
    "ready_temp_c": 38.0,
    "eco_floor_c": 28.0,
    "params": {"heat_capacity_kwh_per_c": 0.9, "cooling_k_per_h": 0.021, "source": "config"},
    "windows": [["2026-11-16T19:00:00+01:00", "2026-11-16T21:00:00+01:00"]],
    "next_bath": {
        "start": "2026-11-16T19:00:00+01:00",
        "end": "2026-11-16T21:00:00+01:00",
        "ready_by": "2026-11-16T18:30:00+01:00",
        "predicted_temp_c": 38.24,
        "shortfall_c": 0.0,
    },
    "on_track": True,
}


def _by_id(sensors: list[PublishedSensor]) -> dict[str, PublishedSensor]:
    return {s.entity_id: s for s in sensors}


def test_domain_defaults_to_sensor():
    s = PublishedSensor(object_id="darkstar_x", state="1")
    assert s.domain == "sensor" and s.entity_id == "sensor.darkstar_x"
    assert PublishedSensor("y", "on", domain="binary_sensor").entity_id == "binary_sensor.y"


def test_entity_ids_and_states():
    sensors = _by_id(build_spa_bathing_sensors(META))
    assert set(sensors) == {
        "sensor.darkstar_spa_next_bath",
        "sensor.darkstar_spa_ready_forecast_c",
        "binary_sensor.darkstar_spa_on_track",
    }
    nb = sensors["sensor.darkstar_spa_next_bath"]
    assert nb.state == "2026-11-16T19:00:00+01:00"
    assert nb.device_class == "timestamp"
    assert nb.attributes["end"] == "2026-11-16T21:00:00+01:00"
    assert nb.attributes["ready_by"] == "2026-11-16T18:30:00+01:00"
    assert nb.attributes["mode"] == "shadow"
    assert nb.attributes["ready_temp_c"] == 38.0

    fc = sensors["sensor.darkstar_spa_ready_forecast_c"]
    assert fc.state == "38.2" and fc.unit == "°C"
    assert fc.attributes["shortfall_c"] == 0.0
    assert fc.attributes["measured_temp_c"] == 30.5

    ot = sensors["binary_sensor.darkstar_spa_on_track"]
    assert ot.domain == "binary_sensor" and ot.state == "on"
    assert ot.to_payload()["state"] == "on"
    assert all(s.friendly_name for s in sensors.values())


def test_off_track():
    meta = {**META, "on_track": False}
    assert (
        _by_id(build_spa_bathing_sensors(meta))["binary_sensor.darkstar_spa_on_track"].state
        == "off"
    )


@pytest.mark.parametrize("mode", ["disabled", "fallback"])
def test_not_in_bathing_mode_is_unavailable(mode):
    meta = {**META, "mode": mode, "on_track": None, "reason": "calendar_unknown"}
    sensors = _by_id(build_spa_bathing_sensors(meta))
    assert sensors["binary_sensor.darkstar_spa_on_track"].state == "unavailable"
    assert sensors["sensor.darkstar_spa_next_bath"].attributes["reason"] == "calendar_unknown"


@pytest.mark.parametrize("meta", [None, {}, {**META, "next_bath": None}])
def test_missing_data_is_unknown(meta):
    sensors = _by_id(build_spa_bathing_sensors(meta))
    assert sensors["sensor.darkstar_spa_next_bath"].state == "unknown"
    assert sensors["sensor.darkstar_spa_ready_forecast_c"].state == "unknown"


def test_no_prediction_is_unknown_not_zero():
    meta = {**META, "next_bath": {**META["next_bath"], "predicted_temp_c": None}}
    fc = _by_id(build_spa_bathing_sensors(meta))["sensor.darkstar_spa_ready_forecast_c"]
    assert fc.state == "unknown"


def test_shadow_on_track_null_is_unknown():
    meta = {**META, "on_track": None}
    ot = _by_id(build_spa_bathing_sensors(meta))["binary_sensor.darkstar_spa_on_track"]
    assert ot.state == "unknown"


def _run_loop_once(config: dict[str, Any], tmp_path, monkeypatch, schedule: dict | None):
    """Run run_publisher_loop for one tick; return the published entity ids."""
    from backend.learning import cycle_publisher_service as svc

    monkeypatch.chdir(tmp_path)
    if schedule is not None:
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "schedule.json").write_text(json.dumps(schedule))
    published: list[str] = []

    async def fake_publish(sensors, base_url, token):
        published.extend(s.entity_id for s in sensors)
        return len(sensors)

    class _Stop(Exception):
        pass

    async def stop_sleep(_s):
        raise _Stop

    with (
        patch(
            "backend.core.secrets.load_home_assistant_config",
            return_value={"url": "http://ha", "token": "t"},
        ),
        patch("backend.learning.cycle_publisher.publish_sensors", side_effect=fake_publish),
        patch("asyncio.sleep", AsyncMock(side_effect=stop_sleep)),
        contextlib.suppress(_Stop),
    ):
        asyncio.run(svc.run_publisher_loop(config))
    return published


def test_loop_does_not_start_without_bathing(tmp_path, monkeypatch):
    out = _run_loop_once(
        {"water_heaters": [{"id": "spa", "bathing": {"enabled": False}}]},
        tmp_path,
        monkeypatch,
        {"meta": {"spa_bathing": META}},
    )
    assert out == []


def test_loop_starts_for_bathing_and_publishes(tmp_path, monkeypatch):
    out = _run_loop_once(
        {"water_heaters": [{"id": "spa", "bathing": {"enabled": True}}]},
        tmp_path,
        monkeypatch,
        {"meta": {"spa_bathing": META}},
    )
    assert "binary_sensor.darkstar_spa_on_track" in out
    assert "sensor.darkstar_spa_next_bath" in out
