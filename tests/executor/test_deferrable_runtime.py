"""Tests for the deferrable smart-appliance runtime (parse + observe-first run)."""

import json
from datetime import datetime

import pytest
import pytz

from executor.deferrable_runtime import (
    DeferrableApplianceController,
    load_forward_slots,
    parse_deferrable_runtime_config,
)

TZ = pytz.timezone("Europe/Stockholm")


class FakeHA:
    def __init__(self, states):
        self.states = states
        self.published: dict[str, tuple] = {}
        self.notifications: list[tuple] = []
        self.calls: list[tuple] = []

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def set_state(self, entity_id, state, attributes=None):
        self.published[entity_id] = (state, attributes or {})
        return True

    async def send_notification(self, service, title, message, data=None):
        self.notifications.append((service, title, message))
        return True

    async def call_service(self, domain, service, entity_id=None, data=None):
        self.calls.append((domain, service, entity_id, data))
        return True


def _full_cfg(**over):
    base = {
        "timezone": "Europe/Stockholm",
        "deferrable_loads": [
            {"id": "washer", "name": "Tvättmaskin", "enabled": True,
             "power_sensor": "sensor.tvattmaskin_power", "switch_entity": "switch.tvattmaskin",
             "override_entity": "input_boolean.washing_machine_override",
             "duration_min": 120, "window_hours": 14},
            {"id": "no_power", "name": "X", "enabled": True},  # no power_sensor => skipped
        ],
        "executor": {
            "schedule_path": "schedule.json",
            "deferrable_appliances": {
                "enabled": True, "observe_only": True,
                "notify_service": "notify.notify_robert_emilia", "slot_minutes": 15,
            },
        },
    }
    base["executor"]["deferrable_appliances"].update(over)
    return base


def _write_schedule(tmp_path, prices, t0):
    """Write a schedule.json with 15-min slots starting at t0 (aware datetime)."""
    slots = []
    for i, p in enumerate(prices):
        start = t0.replace(microsecond=0) + __import__("datetime").timedelta(minutes=15 * i)
        slots.append({"start_time": start.isoformat(), "import_price_sek_kwh": p})
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"schedule": slots}), encoding="utf-8")
    return str(path)


class TestParse:
    def test_absent_returns_none(self):
        assert parse_deferrable_runtime_config({}) is None

    def test_parses_only_power_sensor_loads(self):
        cfg = parse_deferrable_runtime_config(_full_cfg())
        assert cfg is not None and cfg.enabled and cfg.observe_only
        assert [a.id for a in cfg.appliances] == ["washer"]  # no_power dropped
        w = cfg.appliances[0]
        assert w.switch_entity == "switch.tvattmaskin"
        assert w.power.on_threshold_w == 10.0


class TestLoadSlots:
    def test_reads_forward_slots(self, tmp_path):
        now = datetime.now(TZ)
        path = _write_schedule(tmp_path, [0.5, 0.6, 0.7], now)
        slots = load_forward_slots(path, now.timestamp(), "Europe/Stockholm")
        assert len(slots) == 3 and slots[0].import_price_sek_kwh == 0.5

    def test_missing_file_returns_empty(self):
        assert load_forward_slots("/nope/schedule.json", 0.0, "Europe/Stockholm") == []


class TestObserveRun:
    @pytest.mark.asyncio
    async def test_arm_publishes_state_and_notifies_without_actuating(self, tmp_path):
        cfg = parse_deferrable_runtime_config(_full_cfg())
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, [0.8] * 60, now)
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({
            "sensor.tvattmaskin_power": "2000", "switch.tvattmaskin": "on",
            "input_boolean.washing_machine_override": "off",
        })
        # Tick 1: power up (not yet debounced) ; Tick 2: 3 s later => arm.
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        st, attrs = ha.published["sensor.darkstar_washer_state"]
        assert st in ("armed", "running")
        assert attrs["load_id"] == "washer" and attrs["observe_only"] is True
        # Observe-only: never touches the plug.
        assert not any(c[1] in ("turn_on", "turn_off") for c in ha.calls)
        # Armed notification fired.
        assert any("Started" in m for _s, _t, m in ha.notifications)

    @pytest.mark.asyncio
    async def test_recommends_defer_before_peak(self, tmp_path):
        # now cheap-ish but a peak is imminent and a cheaper window exists later.
        cfg = parse_deferrable_runtime_config(_full_cfg(slot_minutes=15))
        now = datetime.now(TZ)
        # 120-min cycle = 8 slots. Make the first 8 slots straddle a peak, later 8 slots cheap.
        prices = [0.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] + [0.3] * 16
        cfg.schedule_path = _write_schedule(tmp_path, prices, now)
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({
            "sensor.tvattmaskin_power": "2000", "switch.tvattmaskin": "on",
            "input_boolean.washing_machine_override": "off",
        })
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        out = await ctrl.run(ha, t + 3.0, now, shadow=False)
        _st, attrs = ha.published["sensor.darkstar_washer_state"]
        assert attrs["recommended_action"] == "defer"
        assert attrs["would_defer"] is True
        assert out["appliances"][0]["action"] == "defer"

    @pytest.mark.asyncio
    async def test_disabled_noop(self, tmp_path):
        cfg = parse_deferrable_runtime_config(_full_cfg(enabled=False))
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({"sensor.tvattmaskin_power": "2000"})
        out = await ctrl.run(ha, 1000.0, datetime.now(TZ), shadow=False)
        assert out == {"enabled": False}
        assert ha.published == {}
