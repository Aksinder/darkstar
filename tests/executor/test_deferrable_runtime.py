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

    async def set_switch(self, entity_id, state):
        self.calls.append(("switch", "turn_on" if state else "turn_off", entity_id, None))
        self.states[entity_id] = "on" if state else "off"
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


def _defer_prices():
    """First 8 slots straddle a peak; a much cheaper 120-min window opens later."""
    return [0.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] + [0.3] * 16


class TestActuation:
    """Fas 3: observe_only=False gates the plug (armed+defer holds OFF, window resumes)."""

    def _armed_setup(self, tmp_path, prices, switch_state="on", override="off"):
        cfg = parse_deferrable_runtime_config(_full_cfg(observe_only=False))
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, prices, now)
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({
            "sensor.tvattmaskin_power": "2000", "switch.tvattmaskin": switch_state,
            "input_boolean.washing_machine_override": override,
        })
        return ctrl, ha, now

    @pytest.mark.asyncio
    async def test_armed_defer_holds_plug_off(self, tmp_path):
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        out = await ctrl.run(ha, t + 3.0, now, shadow=False)
        assert out["appliances"][0]["action"] == "defer"
        assert ("switch", "turn_off", "switch.tvattmaskin", None) in ha.calls
        _st, attrs = ha.published["sensor.darkstar_washer_state"]
        assert attrs["plug_commanded"] == "off"

    @pytest.mark.asyncio
    async def test_deferred_cycle_resumes_when_window_is_now(self, tmp_path):
        """Once the recommendation flips to run, the held-off plug is re-enabled."""
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed -> deferred, plug off
        assert ha.states["switch.tvattmaskin"] == "off"
        # Cheap-now schedule: recommendation becomes "run" while still pending.
        cfg_cheap = _write_schedule(tmp_path, [0.1] * 24, datetime.now(TZ))
        ctrl.cfg.schedule_path = cfg_cheap
        await ctrl.run(ha, t + 60.0, now, shadow=False)
        assert ("switch", "turn_on", "switch.tvattmaskin", None) in ha.calls
        assert ha.states["switch.tvattmaskin"] == "on"

    @pytest.mark.asyncio
    async def test_mid_cycle_run_is_never_paused_when_prices_turn(self, tmp_path):
        """A cycle that armed into a good window keeps running untouched even if the
        forecast later flips to defer — pausing is only allowed at start detection."""
        ctrl, ha, now = self._armed_setup(tmp_path, [0.1] * 24)  # cheap: arms as "run"
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        out = await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed event, action=run
        assert out["appliances"][0]["action"] == "run"
        assert not any(c[1] == "turn_off" for c in ha.calls)
        # Prices turn ugly mid-cycle: recommendation flips to defer, but no arm event
        # fires and the plug is on => hands off.
        ctrl.cfg.schedule_path = _write_schedule(tmp_path, _defer_prices(), datetime.now(TZ))
        out2 = await ctrl.run(ha, t + 600.0, now, shadow=False)
        assert out2["appliances"][0]["action"] == "defer"
        assert not any(c[1] == "turn_off" for c in ha.calls)
        assert ha.states["switch.tvattmaskin"] == "on"

    @pytest.mark.asyncio
    async def test_override_forces_plug_on(self, tmp_path):
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices(), switch_state="off",
                                          override="on")
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        assert ("switch", "turn_on", "switch.tvattmaskin", None) in ha.calls

    @pytest.mark.asyncio
    async def test_unreadable_switch_is_never_actuated(self, tmp_path):
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices(),
                                          switch_state="unavailable")
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        assert not any(c[1] in ("turn_on", "turn_off") for c in ha.calls)

    @pytest.mark.asyncio
    async def test_shadow_mode_blocks_actuation_even_when_armed(self, tmp_path):
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=True)
        await ctrl.run(ha, t + 3.0, now, shadow=True)
        assert not any(c[1] in ("turn_on", "turn_off") for c in ha.calls)

    @pytest.mark.asyncio
    async def test_unavailable_device_mid_hold_freezes_instead_of_stranding(self, tmp_path):
        """Review-reproduced critical: a one-tick Wi-Fi drop mid-hold used to fire a
        false 'done' (power unavailable->0 + hours-old below_since + switch defaulting
        ON), drop pending, and strand the plug OFF forever. Now the tick freezes."""
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed -> held OFF
        assert ha.states["switch.tvattmaskin"] == "off"
        n_notif = len(ha.notifications)

        # Device drops off the network: both readings unavailable for one tick.
        ha.states["sensor.tvattmaskin_power"] = "unavailable"
        ha.states["switch.tvattmaskin"] = "unavailable"
        out = await ctrl.run(ha, t + 1800.0, now, shadow=False)
        assert out["appliances"][0]["action"] == "frozen"
        assert ctrl._state["washer"].pending is True  # no false done
        assert len(ha.notifications) == n_notif  # no 'Done' lie

        # Device returns (Shelly reboots ON) — hold is re-asserted, cycle intact.
        ha.states["sensor.tvattmaskin_power"] = "0"
        ha.states["switch.tvattmaskin"] = "on"
        await ctrl.run(ha, t + 1860.0, now, shadow=False)
        assert ctrl._state["washer"].pending is True
        # Cheap window arrives -> resume through OUR hold path.
        ctrl.cfg.schedule_path = _write_schedule(tmp_path, [0.1] * 24, datetime.now(TZ))
        await ctrl.run(ha, t + 1920.0, now, shadow=False)
        assert ha.states["switch.tvattmaskin"] == "on"

    @pytest.mark.asyncio
    async def test_soak_pause_rearm_cannot_cut_mid_programme(self, tmp_path):
        """Review-reproduced critical: a >done_delay soak pause fired done->re-arm and
        the fresh 'armed' event re-opened the pause gate mid-programme. The re-arm
        cooldown now suppresses the event, so the plug stays ON."""
        ctrl, ha, now = self._armed_setup(tmp_path, [0.1] * 24)  # arms as run
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        # Soak pause: 2 W for 6+ minutes with the plug ON => false done.
        ha.states["sensor.tvattmaskin_power"] = "2"
        await ctrl.run(ha, t + 2400.0, now, shadow=False)
        await ctrl.run(ha, t + 2761.0, now, shadow=False)  # done fires
        assert ctrl._state["washer"].pending is False
        # Heater kicks back in; prices have turned ugly meanwhile.
        ctrl.cfg.schedule_path = _write_schedule(tmp_path, _defer_prices(), datetime.now(TZ))
        ha.states["sensor.tvattmaskin_power"] = "2000"
        await ctrl.run(ha, t + 2770.0, now, shadow=False)
        out = await ctrl.run(ha, t + 2774.0, now, shadow=False)  # re-arms silently
        assert ctrl._state["washer"].pending is True
        assert not any(c[1] == "turn_off" for c in ha.calls)  # never cut
        assert out["appliances"][0]["event"] is None  # silent continuation

    @pytest.mark.asyncio
    async def test_resume_grants_repower_grace_no_instant_done(self, tmp_path):
        """Review-reproduced: at window resume, below_since was hours old, so one
        lagging 0 W tick with the plug now ON fired an instant done. The OFF->ON
        transition now resets the low-draw clock."""
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # held OFF
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 7200.0, now, shadow=False)  # 2 h held, still pending
        # Window arrives; plug re-enabled; first powered tick still reads 0 W (lag).
        ctrl.cfg.schedule_path = _write_schedule(tmp_path, [0.1] * 24, datetime.now(TZ))
        await ctrl.run(ha, t + 7260.0, now, shadow=False)  # commands ON
        assert ha.states["switch.tvattmaskin"] == "on"
        out = await ctrl.run(ha, t + 7320.0, now, shadow=False)  # lagging 0 W tick
        assert ctrl._state["washer"].pending is True  # NOT instant-done
        assert out["appliances"][0]["event"] is None

    @pytest.mark.asyncio
    async def test_manual_off_wins_even_while_pending(self, tmp_path):
        """Review warning: a plug the USER cut must never be re-energized — only holds
        Darkstar owns (held_by_us) are resumed."""
        ctrl, ha, now = self._armed_setup(tmp_path, [0.1] * 24)  # run window: plug stays on
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        assert ctrl._state["washer"].pending is True
        # User cuts the plug mid-cycle (leaking hose). Power dies with it.
        ha.states["switch.tvattmaskin"] = "off"
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 60.0, now, shadow=False)
        await ctrl.run(ha, t + 120.0, now, shadow=False)
        assert not any(c[1] == "turn_on" for c in ha.calls)
        assert ha.states["switch.tvattmaskin"] == "off"

    @pytest.mark.asyncio
    async def test_boot_recovery_reenables_orphaned_plug(self, tmp_path):
        """Review warning: an add-on update wipes the state file mid-hold; the plug
        would stay OFF forever with no state entry to resume it. Boot recovery powers
        it back on once, with a notification."""
        cfg = parse_deferrable_runtime_config(_full_cfg(observe_only=False))
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, [0.5] * 8, now)
        # Fresh controller, EMPTY state file (simulates the wipe), plug found OFF.
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({
            "sensor.tvattmaskin_power": "0", "switch.tvattmaskin": "off",
            "input_boolean.washing_machine_override": "off",
        })
        await ctrl.run(ha, now.timestamp(), now, shadow=False)
        assert ("switch", "turn_on", "switch.tvattmaskin", None) in ha.calls
        assert any("re-enabled" in m.lower() or "restart" in m.lower()
                   for _s, _t, m in ha.notifications)
        # Second tick: no repeat.
        n = len([c for c in ha.calls if c[1] == "turn_on"])
        await ctrl.run(ha, now.timestamp() + 60.0, now, shadow=False)
        assert len([c for c in ha.calls if c[1] == "turn_on"]) == n

    @pytest.mark.asyncio
    async def test_deadline_anchored_to_arming_fails_open(self, tmp_path):
        """Review warning: the rolling now+window_hours deadline never closed. Anchored
        to start_ts, a cycle armed 15 h ago (window 14 h) must fail open to run."""
        ctrl, ha, now = self._armed_setup(tmp_path, _defer_prices())
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # held OFF
        # Simulate 15 h of holding by backdating the persisted start_ts.
        ctrl._state["washer"].start_ts = t - 15 * 3600.0
        ha.states["sensor.tvattmaskin_power"] = "0"
        out = await ctrl.run(ha, t + 60.0, now, shadow=False)
        assert out["appliances"][0]["action"] == "run"  # deadline passed => fail open
        assert ha.states["switch.tvattmaskin"] == "on"

    @pytest.mark.asyncio
    async def test_idle_plug_left_alone(self, tmp_path):
        """No pending cycle: the plug is not written at all (already on, stays on)."""
        cfg = parse_deferrable_runtime_config(_full_cfg(observe_only=False))
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, [0.5] * 8, now)
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "state.json"))
        ha = FakeHA({
            "sensor.tvattmaskin_power": "0", "switch.tvattmaskin": "on",
            "input_boolean.washing_machine_override": "off",
        })
        await ctrl.run(ha, now.timestamp(), now, shadow=False)
        assert not any(c[1] in ("turn_on", "turn_off") for c in ha.calls)
