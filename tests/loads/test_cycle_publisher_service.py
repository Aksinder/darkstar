"""Tests for the deferrable-load publisher orchestration service."""

from datetime import datetime, timedelta

import pytest

from backend.learning.cycle_publisher_service import (
    DeferrablePublisherService,
    TrackedAppliance,
    TrackedTank,
    build_tracked_from_config,
)

NOW = datetime(2026, 6, 1, 12, 0, 0)


def _status_rows(start: datetime, phases, power_w=2000):
    """Build HA status history rows: Idle -> phases... -> Idle."""
    rows = [
        {
            "state": "Idle",
            "last_changed": (start - timedelta(minutes=5)).isoformat(),
            "attributes": {"power": 0},
        }
    ]
    t = start
    for phase, minutes in phases:
        rows.append(
            {"state": phase, "last_changed": t.isoformat(), "attributes": {"power": power_w}}
        )
        t = t + timedelta(minutes=minutes)
    rows.append({"state": "Idle", "last_changed": t.isoformat(), "attributes": {"power": 0}})
    # A second idle row well past the merge gap. A cycle is only knowably COMPLETE once
    # no future sample could still merge into it (see _is_complete), and real history
    # always runs on to "now" — a series that stops the instant the machine does is not
    # a finished cycle, it is one we caught at the boundary.
    rows.append(
        {
            "state": "Idle",
            "last_changed": (t + timedelta(minutes=30)).isoformat(),
            "attributes": {"power": 0},
        }
    )
    return rows


class _Capture:
    def __init__(self):
        self.sensors = []

    async def publish(self, sensors):
        self.sensors = sensors
        return len(sensors)


@pytest.mark.asyncio
async def test_appliance_cycle_published():
    # A dishwasher cycle earlier today: Filling 10m, Washing 80m, Rinsing 20m.
    start = NOW - timedelta(hours=3)
    rows = _status_rows(start, [("Filling", 10), ("Washing", 80), ("Rinsing", 20)])

    async def fetch_history(entity_id, hours):
        assert entity_id == "sensor.dishwasher_status"
        return rows

    async def fetch_float(entity_id):
        return 0.0

    cap = _Capture()
    svc = DeferrablePublisherService(
        appliances=[
            TrackedAppliance(
                id="dishwasher", name="Diskmaskin", signal_entity="sensor.dishwasher_status"
            )
        ],
        tanks=[],
        fetch_history=fetch_history,
        fetch_float=fetch_float,
        publish=cap.publish,
        now_fn=lambda: NOW,
    )
    n = await svc.run_once()
    assert n == 4  # 4 sensors per appliance
    # The detected cycles are retained for the load-shift savings join
    # (no second history pull needed).
    assert len(svc.last_cycles["dishwasher"]) == 1
    ids = {s.object_id for s in cap.sensors}
    assert "sensor.darkstar_dishwasher_last_cycle_minutes".removeprefix("sensor.") in ids
    minutes = next(s for s in cap.sensors if s.object_id.endswith("last_cycle_minutes"))
    assert float(minutes.state) >= 100  # ~110 min cycle detected
    draw = next(s for s in cap.sensors if s.object_id.endswith("draw_today"))
    assert float(draw.state) > 0  # today's draw accumulated


@pytest.mark.asyncio
async def test_energy_sensor_gives_accurate_cycle_energy():
    # Status sensor's power attr is tiny/unreliable; a cumulative kWh meter
    # provides the accurate per-cycle energy and today's draw.
    start = NOW - timedelta(hours=2)
    status_rows = _status_rows(start, [("Washing", 60)], power_w=50)  # bogus low power
    end = start + timedelta(minutes=60)
    energy_rows = [
        {"state": "100.0", "last_changed": (start - timedelta(minutes=30)).isoformat()},
        {"state": "100.0", "last_changed": start.isoformat()},
        {"state": "101.4", "last_changed": end.isoformat()},  # +1.4 kWh over the cycle
        {"state": "101.4", "last_changed": NOW.isoformat()},
    ]

    async def fetch_history(entity_id, hours):
        return energy_rows if entity_id == "sensor.washer_energy" else status_rows

    async def fetch_float(entity_id):
        return 0.0

    cap = _Capture()
    svc = DeferrablePublisherService(
        appliances=[
            TrackedAppliance(
                id="washer",
                name="Tvattmaskin",
                signal_entity="sensor.washing_machine_status",
                energy_sensor="sensor.washer_energy",
            )
        ],
        tanks=[],
        fetch_history=fetch_history,
        fetch_float=fetch_float,
        publish=cap.publish,
        now_fn=lambda: NOW,
    )
    await svc.run_once()
    energy = next(s for s in cap.sensors if s.object_id.endswith("last_cycle_energy"))
    draw = next(s for s in cap.sensors if s.object_id.endswith("draw_today"))
    assert float(energy.state) == pytest.approx(1.4, abs=0.01)  # from the meter, not 0.0x
    assert float(draw.state) == pytest.approx(1.4, abs=0.01)


@pytest.mark.asyncio
async def test_one_failing_load_does_not_block_others():
    async def fetch_history(entity_id, hours):
        if entity_id == "sensor.bad":
            raise RuntimeError("boom")
        return _status_rows(NOW - timedelta(hours=2), [("Washing", 60)])

    async def fetch_float(entity_id):
        return 0.0

    cap = _Capture()
    svc = DeferrablePublisherService(
        appliances=[
            TrackedAppliance(id="bad", name="Bad", signal_entity="sensor.bad"),
            TrackedAppliance(id="good", name="Good", signal_entity="sensor.good"),
        ],
        tanks=[],
        fetch_history=fetch_history,
        fetch_float=fetch_float,
        publish=cap.publish,
        now_fn=lambda: NOW,
    )
    n = await svc.run_once()
    assert n == 4  # only the good appliance's sensors
    assert all("good" in s.object_id for s in cap.sensors)


@pytest.mark.asyncio
async def test_tank_estimator_advances_and_publishes():
    powers = iter([3500.0, 3500.0, 0.0])  # heating, heating, off

    async def fetch_history(entity_id, hours):
        return []

    async def fetch_float(entity_id):
        return next(powers)

    cap = _Capture()
    tank = TrackedTank(
        id="house_vvb",
        name="VVB Huset",
        power_entity="sensor.house_vvb_real_power",
        volume_litres=200,
        t_max_c=80,
        ua_w_per_k=2.0,
    )
    times = iter([NOW, NOW + timedelta(minutes=15), NOW + timedelta(minutes=30)])
    svc = DeferrablePublisherService(
        appliances=[],
        tanks=[tank],
        fetch_history=fetch_history,
        fetch_float=fetch_float,
        publish=cap.publish,
        now_fn=lambda: next(times),
    )
    await svc.run_once()  # first tick: no dt, starts full
    await svc.run_once()  # dt=15m heating
    n = await svc.run_once()  # dt=15m heating
    assert n == 4  # 4 hot-water sensors
    ids = {s.object_id for s in cap.sensors}
    assert ids == {
        "darkstar_house_vvb_hot_water_level",
        "darkstar_house_vvb_liters_remaining",
        "darkstar_house_vvb_estimated_temperature",
        "darkstar_house_vvb_draw_today",
    }
    soc = next(s for s in cap.sensors if s.object_id.endswith("hot_water_level"))
    assert 0 <= float(soc.state) <= 100


def test_build_tracked_from_config():
    config = {
        "deferrable_loads": [
            {
                "id": "dishwasher",
                "name": "Diskmaskin",
                "enabled": True,
                "running_sensor": "sensor.dishwasher_status",
                "duration_min": 114,
                "energy_kwh": 1.2,
            },
            {"id": "off", "enabled": False, "running_sensor": "sensor.x"},
            {"id": "nosignal", "enabled": True},  # skipped: no signal
        ],
        "water_heaters": [
            {
                "id": "house_vvb",
                "name": "VVB Huset",
                "enabled": True,
                "type": "thermal",
                "power_sensor": "sensor.house_vvb_real_power",
                "volume_litres": 200,
                "t_max_c": 80,
                "ua_w_per_k": 2.5,
            },
            {"id": "main_tank", "enabled": True, "type": "binary"},  # skipped: not thermal
        ],
    }
    apps, tanks = build_tracked_from_config(config)
    assert [a.id for a in apps] == ["dishwasher"]
    assert apps[0].signal_kind == "status"
    assert apps[0].seed_duration_min == 114
    assert [t.id for t in tanks] == ["house_vvb"]
    assert tanks[0].volume_litres == 200
    assert tanks[0].ua_w_per_k == 2.5
