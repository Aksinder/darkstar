"""Tests for publishing cycle stats / hot-water state to HA sensors."""

from datetime import datetime, timedelta

from backend.learning.cycle_learning import CycleStats, DetectedCycle
from backend.learning.cycle_publisher import (
    PublishedSensor,
    build_hot_water_sensors,
    build_load_sensors,
    build_unknown_load_sensor,
)
from planner.hot_water import HotWaterEstimator
from planner.thermal import WaterTankModel

T0 = datetime(2026, 6, 1, 7, 0, 0)


def _by_id(sensors, suffix):
    return next(s for s in sensors if s.object_id.endswith(suffix))


def test_published_sensor_payload_merges_attributes():
    s = PublishedSensor(
        object_id="darkstar_x_draw_today",
        state="1.2",
        unit="kWh",
        device_class="energy",
        state_class="total_increasing",
        friendly_name="X",
        attributes={"foo": "bar"},
    )
    assert s.entity_id == "sensor.darkstar_x_draw_today"
    p = s.to_payload()
    assert p["state"] == "1.2"
    assert p["attributes"]["unit_of_measurement"] == "kWh"
    assert p["attributes"]["device_class"] == "energy"
    assert p["attributes"]["state_class"] == "total_increasing"
    assert p["attributes"]["friendly_name"] == "X"
    assert p["attributes"]["foo"] == "bar"


def test_build_load_sensors_basic():
    stats = CycleStats(
        n_cycles=8,
        learned=True,
        duration_min=114.0,
        duration_min_p90=121.0,
        energy_kwh=1.2,
        energy_kwh_p90=1.5,
        typical_profile_kw=[2.0, 0.2],
    )
    last = DetectedCycle(
        start=T0,
        end=T0 + timedelta(minutes=110),
        duration_min=110.0,
        energy_kwh=1.18,
        phase_minutes={"Filling": 5.0, "Washing": 80.0, "Drying": 25.0},
    )
    sensors = build_load_sensors("dishwasher", "Diskmaskin", stats, last, draw_today_kwh=2.4)

    ids = {s.object_id for s in sensors}
    assert ids == {
        "darkstar_dishwasher_last_cycle_energy",
        "darkstar_dishwasher_last_cycle_minutes",
        "darkstar_dishwasher_typical_minutes",
        "darkstar_dishwasher_draw_today",
    }
    energy = _by_id(sensors, "last_cycle_energy")
    assert energy.state == "1.18"
    assert energy.attributes["cycles_observed"] == 8
    assert energy.attributes["learned"] is True
    assert energy.attributes["last_cycle_phases"]["Washing"] == 80.0
    assert _by_id(sensors, "last_cycle_minutes").state == "110.0"
    assert _by_id(sensors, "typical_minutes").state == "114.0"
    assert _by_id(sensors, "draw_today").state == "2.4"


def test_build_load_sensors_no_cycle_yet():
    stats = CycleStats(0, False, 120.0, 120.0, 1.2, 1.2, [])
    sensors = build_load_sensors("washer", "Tvättmaskin", stats, None, draw_today_kwh=0.0)
    assert _by_id(sensors, "last_cycle_energy").state == "0.0"
    assert _by_id(sensors, "last_cycle_minutes").state == "0.0"
    # Seed/typical values still surface so the planner has a fallback.
    assert _by_id(sensors, "typical_minutes").state == "120.0"


def test_slug_sanitises_id():
    stats = CycleStats(0, False, 1.0, 1.0, 0.1, 0.1, [])
    sensors = build_load_sensors("VVB Huset.1", "VVB", stats, None, 0.0)
    assert _by_id(sensors, "draw_today").object_id == "darkstar_vvb_huset_1_draw_today"


def test_build_hot_water_sensors():
    tank = WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=80, ua_w_per_k=2.0)
    est = HotWaterEstimator.from_temperature(tank, 65.0)
    # Default prefix keeps a darkstar_ namespace (never collides with template sensors),
    # but the suffix scheme matches the canonical HA-native template sensors.
    sensors = build_hot_water_sensors("house_vvb", "VVB Huset", est, draw_today_kwh=4.5)

    ids = {s.object_id for s in sensors}
    assert ids == {
        "darkstar_house_vvb_hot_water_level",
        "darkstar_house_vvb_liters_remaining",
        "darkstar_house_vvb_estimated_temperature",
        "darkstar_house_vvb_draw_today",
    }
    temp = _by_id(sensors, "_estimated_temperature")
    assert temp.state == "65.0"
    assert temp.device_class == "temperature"
    soc = _by_id(sensors, "hot_water_level")
    # 65 over [10,80] -> ~78.6%
    assert abs(float(soc.state) - 78.6) < 0.5
    assert _by_id(sensors, "draw_today").state == "4.5"


def test_hot_water_sensor_prefix_can_be_emptied_to_own_canonical_ids():
    # With sensor_prefix="" the publisher emits the bare canonical ids (only safe when
    # no template sensor of the same id exists).
    tank = WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=80, ua_w_per_k=2.0)
    est = HotWaterEstimator.from_temperature(tank, 65.0)
    sensors = build_hot_water_sensors("house_vvb", "VVB", est, 0.0, object_id_prefix="")
    ids = {s.object_id for s in sensors}
    assert "house_vvb_hot_water_level" in ids
    assert "house_vvb_liters_remaining" in ids
    assert "house_vvb_estimated_temperature" in ids


def test_build_unknown_load_sensor():
    sensors = build_unknown_load_sensor(
        unknown_kw=1.234, total_kw=5.0, controllable_kw=3.766, drift_rate=0.07, ev_excluded_kw=11.0
    )
    assert len(sensors) == 1
    s = sensors[0]
    assert s.object_id == "darkstar_unknown_load"
    assert s.state == "1.234"
    assert s.unit == "kW"
    assert s.device_class == "power"
    assert s.attributes["total_load_kw"] == 5.0
    assert s.attributes["metered_controllable_kw"] == 3.766
    assert s.attributes["ev_excluded_kw"] == 11.0
    assert s.attributes["drift_rate"] == 0.07
