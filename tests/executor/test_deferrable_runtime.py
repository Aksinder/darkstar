"""Tests for the deferrable smart-appliance runtime (parse + observe-first run)."""

import json
from datetime import datetime

import pytest
import pytz

from executor.deferrable import (
    AppliancePowerConfig,
    AppliancePowerState,
    update_appliance_power_state,
)
from executor.deferrable_runtime import (
    DeferrableApplianceController,
    load_forward_slots,
    parse_deferrable_runtime_config,
)

TZ = pytz.timezone("Europe/Stockholm")


class FakeHA:
    def __init__(self, states, context_user_ids=None, attributes=None):
        self.states = states
        self.published: dict[str, tuple] = {}
        self.notifications: list[tuple] = []
        self.calls: list[tuple] = []
        # entity -> context.user_id, so tests can say WHOSE hand touched a switch.
        # Absent means the device reported its own state (HA sends user_id None).
        self.context_user_ids = context_user_ids or {}
        # entity -> attribute dict. The learned cycle statistics ride here, not in the
        # state — sensor.darkstar_<id>_last_cycle_energy's STATE is the previous run
        # while typical_energy_kwh is the median over complete cycles.
        self.attributes = attributes or {}

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        """Full HA state object, including the context that produced it."""
        if entity not in self.states and entity not in self.context_user_ids:
            return None
        return {
            "state": self.states.get(entity),
            "attributes": self.attributes.get(entity, {}),
            "context": {
                "id": "test",
                "parent_id": None,
                "user_id": self.context_user_ids.get(entity),
            },
        }

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


class TestCycleLedger:
    """The append-only arm->done ledger (savings v2 input)."""

    def _ctrl(self, tmp_path, prices, observe_only=True, **states):
        cfg = parse_deferrable_runtime_config(_full_cfg(observe_only=observe_only))
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, prices, now)
        ctrl = DeferrableApplianceController(
            cfg,
            state_file=str(tmp_path / "state.json"),
            ledger_file=str(tmp_path / "cycles.jsonl"),
        )
        ha = FakeHA({
            "sensor.tvattmaskin_power": "2000", "switch.tvattmaskin": "on",
            "input_boolean.washing_machine_override": "off",
            **states,
        })
        return ctrl, ha, now, tmp_path / "cycles.jsonl"

    def _rows(self, path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @pytest.mark.asyncio
    async def test_done_writes_row_with_armed_ts_before_state_clear(self, tmp_path):
        ctrl, ha, now, ledger = self._ctrl(tmp_path, [0.5] * 60)
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed
        # Cycle finishes: low power sustained past done_delay_s (300 s).
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 400.0, now, shadow=False)  # below_since starts
        await ctrl.run(ha, t + 701.0, now, shadow=False)  # done fires
        rows = self._rows(ledger)
        assert len(rows) == 1
        row = rows[0]
        assert row["load_id"] == "washer"
        assert row["armed_ts"] == pytest.approx(t + 3.0)  # captured before the clear
        assert row["done_ts"] == pytest.approx(t + 701.0)
        assert row["held_by_us_ever"] is False
        assert row["measured_kwh"] is None
        # Soft deadline anchored to arming: window_hours (14 h) from armed_ts.
        assert row["deadline_ts"] == pytest.approx(t + 3.0 + 14 * 3600.0)

    @pytest.mark.asyncio
    async def test_continuation_keeps_first_armed_ts(self, tmp_path):
        """A re-arm within rearm_cooldown_s is the same physical programme: its
        done row must carry the FIRST armed_ts so the reader can collapse both
        rows into one cycle."""
        ctrl, ha, now, ledger = self._ctrl(tmp_path, [0.5] * 60)
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # genuine arm
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 400.0, now, shadow=False)
        await ctrl.run(ha, t + 701.0, now, shadow=False)  # done #1 (soak pause)
        # Programme resumes within the 900 s cooldown => silent continuation arm.
        ha.states["sensor.tvattmaskin_power"] = "2000"
        await ctrl.run(ha, t + 720.0, now, shadow=False)
        await ctrl.run(ha, t + 724.0, now, shadow=False)  # re-armed, no event
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 800.0, now, shadow=False)
        await ctrl.run(ha, t + 1101.0, now, shadow=False)  # done #2
        rows = self._rows(ledger)
        assert len(rows) == 2
        assert rows[0]["armed_ts"] == rows[1]["armed_ts"] == pytest.approx(t + 3.0)
        assert rows[1]["done_ts"] > rows[0]["done_ts"]

    @pytest.mark.asyncio
    async def test_new_cycle_after_cooldown_gets_fresh_armed_ts(self, tmp_path):
        ctrl, ha, now, ledger = self._ctrl(tmp_path, [0.5] * 200)
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 400.0, now, shadow=False)
        await ctrl.run(ha, t + 701.0, now, shadow=False)  # done #1
        # Next load starts well past rearm_cooldown_s (900 s): a genuine new cycle.
        t2 = t + 701.0 + 1000.0
        ha.states["sensor.tvattmaskin_power"] = "2000"
        await ctrl.run(ha, t2, now, shadow=False)
        await ctrl.run(ha, t2 + 3.0, now, shadow=False)  # armed (new chain)
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t2 + 100.0, now, shadow=False)
        await ctrl.run(ha, t2 + 401.0, now, shadow=False)  # done #2
        rows = self._rows(ledger)
        assert len(rows) == 2
        assert rows[0]["armed_ts"] == pytest.approx(t + 3.0)
        assert rows[1]["armed_ts"] == pytest.approx(t2 + 3.0)

    @pytest.mark.asyncio
    async def test_silent_rearm_past_merge_gap_starts_fresh_chain(self, tmp_path):
        """Savings-review regression: with done_delay_s > 300 a silent re-arm can
        land >20 min after the PHYSICAL stop — the cycle detectors then split the
        runs, so the chain must split too. Inheriting the old armed_ts would price
        a never-deferred reload against the previous programme's arm anchor
        (fabricated credit) and erase the previous row via (load_id, armed_ts)
        dedupe."""
        full = _full_cfg()
        full["deferrable_loads"][0]["done_delay_s"] = 600
        cfg = parse_deferrable_runtime_config(full)
        now = datetime.now(TZ)
        cfg.schedule_path = _write_schedule(tmp_path, [0.5] * 96, now)
        ctrl = DeferrableApplianceController(
            cfg,
            state_file=str(tmp_path / "state.json"),
            ledger_file=str(tmp_path / "cycles.jsonl"),
        )
        ha = FakeHA(
            {
                "sensor.tvattmaskin_power": "2000",
                "switch.tvattmaskin": "on",
                "input_boolean.washing_machine_override": "off",
            }
        )
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed #1
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 400.0, now, shadow=False)  # physical stop ~ t+400
        await ctrl.run(ha, t + 1001.0, now, shadow=False)  # done #1 (600 s delay)
        # Reload 700 s after done: silent (within the 900 s cooldown) but ~22 min
        # after the physical stop => detectors WILL split the runs.
        ha.states["sensor.tvattmaskin_power"] = "2000"
        await ctrl.run(ha, t + 1701.0, now, shadow=False)
        await ctrl.run(ha, t + 1704.5, now, shadow=False)  # silent re-arm
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 1800.0, now, shadow=False)
        await ctrl.run(ha, t + 2401.0, now, shadow=False)  # done #2
        rows = self._rows(tmp_path / "cycles.jsonl")
        assert len(rows) == 2
        assert rows[0]["armed_ts"] == pytest.approx(t + 3.0)
        # Fresh anchor: NOT inherited, so dedupe keeps both physical programmes.
        assert rows[1]["armed_ts"] == pytest.approx(t + 1704.5)

    @pytest.mark.asyncio
    async def test_held_by_us_ever_true_when_plug_was_held(self, tmp_path):
        """Actuating mode: armed+defer holds the plug OFF; when the cycle later
        completes, the ledger row must record that Darkstar shifted it."""
        prices = [0.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] + [0.3] * 16
        ctrl, ha, now, ledger = self._ctrl(tmp_path, prices, observe_only=False)
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed -> deferred, plug off
        assert ha.states["switch.tvattmaskin"] == "off"
        # Window arrives: cheap-now schedule resumes the plug.
        ctrl.cfg.schedule_path = _write_schedule(tmp_path, [0.1] * 24, datetime.now(TZ))
        await ctrl.run(ha, t + 60.0, now, shadow=False)
        assert ha.states["switch.tvattmaskin"] == "on"
        # Machine runs and finishes.
        ha.states["sensor.tvattmaskin_power"] = "2000"
        await ctrl.run(ha, t + 120.0, now, shadow=False)
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl.run(ha, t + 200.0, now, shadow=False)
        await ctrl.run(ha, t + 501.0, now, shadow=False)  # done
        rows = self._rows(ledger)
        assert len(rows) == 1
        assert rows[0]["held_by_us_ever"] is True

    @pytest.mark.asyncio
    async def test_chain_survives_state_file_roundtrip(self, tmp_path):
        """A restart between arm and done must not lose the chain's armed_ts."""
        ctrl, ha, now, ledger = self._ctrl(tmp_path, [0.5] * 60)
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)  # armed; state+chain saved
        # New controller instance (simulated restart) loads the persisted chain.
        ctrl2 = DeferrableApplianceController(
            ctrl.cfg,
            state_file=str(tmp_path / "state.json"),
            ledger_file=str(tmp_path / "cycles.jsonl"),
        )
        ha.states["sensor.tvattmaskin_power"] = "0"
        await ctrl2.run(ha, t + 400.0, now, shadow=False)
        await ctrl2.run(ha, t + 701.0, now, shadow=False)  # done
        rows = self._rows(ledger)
        assert len(rows) == 1
        assert rows[0]["armed_ts"] == pytest.approx(t + 3.0)


class TestScheduleTwoParsersOneFile:
    """ExecutorConfig and DeferrableRuntimeConfig both parse executor.schedule_path,
    each with its own fallback literal. They disagreed ("schedule.json" vs
    "data/schedule.json") from the day the runtime was written until 2026-08-20:
    with the key absent from the live config, the deferrable gate read a path
    that does not exist, got [], and failed open to "run" every tick — it armed
    the dishwasher on a peak morning and "recommended" running it. A silent
    permanent fail-open reads as a decision; this pins the defaults together."""

    def test_the_defaults_agree(self):
        from executor.config import ExecutorConfig
        from executor.deferrable_runtime import DeferrableRuntimeConfig

        assert DeferrableRuntimeConfig.schedule_path == ExecutorConfig.schedule_path

    def test_an_absent_key_parses_to_the_same_path_in_both(self):
        from executor.config import ExecutorConfig
        from executor.deferrable_runtime import parse_deferrable_runtime_config

        cfg = parse_deferrable_runtime_config(
            {"executor": {"deferrable_appliances": {"enabled": True}}}
        )
        assert cfg is not None
        assert cfg.schedule_path == ExecutorConfig.schedule_path

    def test_a_missing_schedule_file_is_loud_but_only_once(self, tmp_path, caplog):
        import logging

        from executor.deferrable_runtime import (
            _missing_schedule_warned,
            load_forward_slots,
        )

        missing = str(tmp_path / "nope" / "schedule.json")
        _missing_schedule_warned.discard(missing)
        with caplog.at_level(logging.WARNING, logger="darkstar.deferrable"):
            assert load_forward_slots(missing, 1000.0, "Europe/Stockholm") == []
            assert load_forward_slots(missing, 2000.0, "Europe/Stockholm") == []
        hits = [r for r in caplog.records if "fail open" in r.message]
        assert len(hits) == 1, "once per path, not per tick"

    def test_the_warning_rearms_when_the_file_appears(self, tmp_path, caplog):
        import json as _json
        import logging

        from executor.deferrable_runtime import (
            _missing_schedule_warned,
            load_forward_slots,
        )

        path = tmp_path / "schedule.json"
        missing = str(path)
        _missing_schedule_warned.discard(missing)
        load_forward_slots(missing, 1000.0, "Europe/Stockholm")
        path.write_text(_json.dumps({"schedule": []}))
        load_forward_slots(missing, 2000.0, "Europe/Stockholm")  # file exists: clears
        path.unlink()
        with caplog.at_level(logging.WARNING, logger="darkstar.deferrable"):
            load_forward_slots(missing, 3000.0, "Europe/Stockholm")
        assert any("fail open" in r.message for r in caplog.records)


class TestManualCutHandback:
    """End-to-end: a human cuts the plug mid-cycle, Darkstar reclaims it after the
    configured delay and resumes it at the cheap window — not the instant the timer
    expires. Owner request 2026-08-20."""

    def _cfg(self, return_minutes):
        from executor.deferrable_runtime import parse_deferrable_runtime_config

        return parse_deferrable_runtime_config(
            {
                "executor": {
                    "deferrable_appliances": {"enabled": True, "observe_only": False},
                    "schedule_path": "data/schedule.json",
                },
                "deferrable_loads": [
                    {
                        "id": "dishwasher",
                        "name": "Diskmaskin",
                        "power_sensor": "sensor.dw_power",
                        "switch_entity": "switch.dw",
                        "manual_cut_return_minutes": return_minutes,
                    }
                ],
            }
        )

    def test_the_knob_parses_into_seconds(self):
        cfg = self._cfg(60)
        assert cfg is not None
        assert cfg.appliances[0].power.manual_cut_return_s == 3600.0

    def test_absent_means_off(self):
        cfg = self._cfg(0)
        assert cfg is not None
        assert cfg.appliances[0].power.manual_cut_return_s == 0.0

    def test_a_reclaimed_cycle_waits_for_its_window_not_the_timer(self):
        """The timer hands back OWNERSHIP; price still decides when it runs."""
        from executor.deferrable import (
            AppliancePowerState,
            recommend_appliance_action,
            should_reclaim_after_manual_cut,
        )

        cut_at, now = 1000.0, 1000.0 + 3600.0
        assert should_reclaim_after_manual_cut(
            switch_on=False, held_by_us=False,
            manual_off_since=cut_at, now_ts=now, manual_cut_return_s=3600.0,
        )
        # Reclaimed => held_by_us. The dear hour still defers it.
        state = AppliancePowerState(pending=True, held_by_us=True)
        slots = _dear_then_cheap(now)
        action, window_start = recommend_appliance_action(slots, now, 2, None)
        assert action == "defer"
        assert window_start is not None and window_start > now
        # And the actuation branch keeps a reclaimed hold OFF while deferring.
        assert state.pending and state.held_by_us and action == "defer"


def _dear_then_cheap(now_ts):
    """Two dear slots, then two cheap ones."""
    from executor.deferrable import WindowSlot

    return [
        WindowSlot(start_ts=now_ts + i * 900.0, import_price_sek_kwh=p)
        for i, p in enumerate([3.0, 3.0, 1.0, 1.0])
    ]


class TestPlugOwnershipEndToEnd:
    """The runtime side of context-based attribution and the resting-state restore."""

    DARKSTAR = "97f1bc39f6184ab2a90da66a62f8b234"
    ROBERT = "613be4dd2bd54547bbe603f15be64363"

    def _cfg(self, return_minutes=60):
        from executor.deferrable_runtime import parse_deferrable_runtime_config

        return parse_deferrable_runtime_config(
            {
                "executor": {
                    "deferrable_appliances": {"enabled": True, "observe_only": False},
                    "schedule_path": "data/schedule.json",
                },
                "deferrable_loads": [
                    {
                        "id": "dishwasher",
                        "name": "Diskmaskin",
                        "power_sensor": "sensor.dw_power",
                        "switch_entity": "switch.dw",
                        "manual_cut_return_minutes": return_minutes,
                    }
                ],
            }
        )

    def _controller(self, cfg, tmp_path):
        from executor.deferrable_runtime import DeferrableApplianceController

        return DeferrableApplianceController(
            cfg, state_file=str(tmp_path / "state.json")
        )

    def _ha(self, switch="off", cutter=None):
        return FakeHA(
            {"sensor.dw_power": "0", "switch.dw": switch},
            context_user_ids={"switch.dw": cutter},
        )

    @staticmethod
    def _dt(ts):
        from datetime import datetime

        return datetime.fromtimestamp(ts)

    @pytest.mark.asyncio
    async def test_it_discovers_its_own_user_id_from_its_own_sensor(self, tmp_path):
        """Nothing but Darkstar writes sensor.darkstar_*, so reading one back names us."""
        cfg = self._cfg()
        ctrl = self._controller(cfg, tmp_path)
        ha = self._ha()
        ha.context_user_ids["sensor.darkstar_dishwasher_state"] = self.DARKSTAR
        ha.states["sensor.darkstar_dishwasher_state"] = "idle"
        assert await ctrl._darkstar_user_id(ha) == self.DARKSTAR

    @pytest.mark.asyncio
    async def test_discovery_failure_leaves_it_unknown(self, tmp_path):
        """And unknown must degrade to the conservative reading, never to 'that was us'."""
        ctrl = self._controller(self._cfg(), tmp_path)
        assert await ctrl._darkstar_user_id(self._ha()) is None

    @pytest.mark.asyncio
    async def test_a_human_cut_releases_a_hold_darkstar_thought_it_owned(self, tmp_path):
        from executor.deferrable import AppliancePowerState

        ctrl = self._controller(self._cfg(), tmp_path)
        ctrl._own_user_id = self.DARKSTAR
        ctrl._state["dishwasher"] = AppliancePowerState(
            pending=True, held_by_us=True, switch_was_on=False
        )
        await ctrl.run(self._ha(cutter=self.ROBERT), 1000.0, self._dt(1000.0), shadow=False)
        assert ctrl._state["dishwasher"].held_by_us is False

    @pytest.mark.asyncio
    async def test_our_own_cut_resumes_but_a_human_cut_does_not(self, tmp_path):
        """The behavioural difference attribution buys. Same state on both runs — a
        pending cycle, plug off, Darkstar believing it holds it, and a plan that says
        run. Ours resumes; a person's hand does not get overridden."""
        from executor.deferrable import AppliancePowerState

        def _run_with(cutter, tmp):
            ctrl = self._controller(self._cfg(), tmp)
            ctrl._own_user_id = self.DARKSTAR
            ctrl._state["dishwasher"] = AppliancePowerState(
                pending=True, held_by_us=True, switch_was_on=False
            )
            return ctrl, self._ha(cutter=cutter)

        ctrl_ours, ha_ours = _run_with(self.DARKSTAR, tmp_path)
        await ctrl_ours.run(ha_ours, 1000.0, self._dt(1000.0), shadow=False)
        assert ("switch", "turn_on", "switch.dw", None) in ha_ours.calls

        ctrl_theirs, ha_theirs = _run_with(self.ROBERT, tmp_path)
        await ctrl_theirs.run(ha_theirs, 1000.0, self._dt(1000.0), shadow=False)
        assert not [c for c in ha_theirs.calls if c[1] == "turn_on"]
        assert ctrl_theirs._state["dishwasher"].held_by_us is False

    @pytest.mark.asyncio
    async def test_an_idle_plug_is_restored_to_its_resting_state(self, tmp_path):
        """A plug left off is a start detector switched off — the next person loads
        the machine, presses start, and nothing happens."""
        from executor.deferrable import AppliancePowerState

        ctrl = self._controller(self._cfg(return_minutes=60), tmp_path)
        ctrl._own_user_id = self.DARKSTAR
        ctrl._state["dishwasher"] = AppliancePowerState(
            pending=False, held_by_us=False, manual_off_since=1000.0
        )
        ha = self._ha(cutter=self.ROBERT)
        await ctrl.run(ha, 1000.0 + 3601.0, self._dt(1000.0 + 3601.0), shadow=False)
        assert ("switch", "turn_on", "switch.dw", None) in ha.calls

    @pytest.mark.asyncio
    async def test_it_waits_out_the_timer_first(self, tmp_path):
        from executor.deferrable import AppliancePowerState

        ctrl = self._controller(self._cfg(return_minutes=60), tmp_path)
        ctrl._own_user_id = self.DARKSTAR
        ctrl._state["dishwasher"] = AppliancePowerState(
            pending=False, held_by_us=False, manual_off_since=1000.0
        )
        ha = self._ha(cutter=self.ROBERT)
        await ctrl.run(ha, 1000.0 + 60.0, self._dt(1000.0 + 60.0), shadow=False)
        assert not [c for c in ha.calls if c[1] == "turn_on"]

    @pytest.mark.asyncio
    async def test_zero_never_restores(self, tmp_path):
        from executor.deferrable import AppliancePowerState

        ctrl = self._controller(self._cfg(return_minutes=0), tmp_path)
        ctrl._own_user_id = self.DARKSTAR
        ctrl._state["dishwasher"] = AppliancePowerState(
            pending=False, held_by_us=False, manual_off_since=1000.0
        )
        ha = self._ha(cutter=self.ROBERT)
        await ctrl.run(ha, 1000.0 + 999999.0, self._dt(1000.0 + 999999.0), shadow=False)
        assert not [c for c in ha.calls if c[1] == "turn_on"]


class TestArmClockSurvivesBursts:
    """A fill-phase dip must not restart the arm debounce.

    Real trace, sensor.diskmaskin_power 2026-09-02: the cycle began at 06:49:22 and its
    first three minutes oscillated between 3.7 W and 33 W, crossing the 10 W on-threshold
    in both directions six times. Clearing above_since on every low sample restarted the
    3 s debounce each time and the arm landed at 06:53:27 — 4m05s in, and 80 seconds
    AFTER the heater had come on, so the hold cut part-heated water.

    Every dip was above off_threshold_w (3 W), the threshold that already means
    "genuinely idle". The arm clock now uses it as the exit condition.
    """

    TRACE = [
        (0, 0.63), (1, 3.26), (6, 6.17), (12, 7.43), (27, 7.53), (30, 14.62),
        (36, 16.75), (43, 12.4), (48, 3.93), (54, 16.73), (60, 8.48), (66, 16.72),
        (71, 3.73), (77, 11.54), (92, 11.51), (97, 8.81), (102, 3.93), (108, 6.59),
        (114, 15.58), (126, 16.87), (131, 18.62), (136, 24.95), (142, 32.69),
    ]

    def _arm_at(self, samples):
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        for t, p in samples:
            st, ev = update_appliance_power_state(
                st, power_w=p, switch_on=True, now_ts=float(t), cfg=cfg
            )
            if ev == "armed":
                return t
        return None

    def test_a_bursty_fill_arms_on_the_first_sustained_draw(self):
        """Sampling every reading, the arm lands at the first burst plus the debounce —
        t+36s, comfortably before the heater at t+166s."""
        assert self._arm_at(self.TRACE) == 36

    def test_a_dip_below_the_on_threshold_does_not_restart_the_clock(self):
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        st, _ = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=0.0, cfg=cfg
        )
        assert st.above_since == 0.0
        # 8 W: under the 10 W arm threshold but well over the 3 W idle threshold.
        st, _ = update_appliance_power_state(
            st, power_w=8.0, switch_on=True, now_ts=1.0, cfg=cfg
        )
        assert st.above_since == 0.0, "a fill-phase dip must not restart the debounce"

    def test_a_genuine_idle_reading_still_clears_the_clock(self):
        """The safety half: below off_threshold_w the machine really has stopped, and
        the clock must reset or a blip an hour ago could arm a cycle that never began."""
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        st, _ = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=0.0, cfg=cfg
        )
        st, ev = update_appliance_power_state(
            st, power_w=0.5, switch_on=True, now_ts=1.0, cfg=cfg
        )
        assert st.above_since is None
        assert ev is None


class TestArmClockUsesTheReadingsOwnAge:
    """The executor ticks every 60 s while start_debounce_s is 3 s.

    Starting the arm clock at now_ts therefore turns a 3-second debounce into a
    60-second one: the first tick only sets the clock, and a second tick a minute later
    is needed to satisfy it. Backdating to the reading's own last_changed lets an
    already-sustained draw arm on the FIRST tick.

    Bounded to exactly the debounce, deliberately: a power sensor can hold one value for
    hours, and an unbounded backdate would let a freshly-started process arm instantly on
    a machine already mid-cycle and open the pause gate on it.
    """

    def test_a_sustained_draw_arms_on_the_first_tick(self):
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        # One tick, seeing 16 W that has read 16 W for the last 45 seconds.
        st, ev = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=1000.0, cfg=cfg,
            power_since=955.0,
        )
        assert ev == "armed"

    def test_without_the_timestamp_it_still_takes_two_ticks(self):
        """The fallback path — a client that supplies no timestamp behaves as before."""
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        st, ev = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=1000.0, cfg=cfg
        )
        assert ev is None
        st, ev = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=1060.0, cfg=cfg
        )
        assert ev == "armed"

    def test_a_reading_that_just_changed_still_waits(self):
        """The debounce must still reject a blip: a value 1 s old is not sustained."""
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        st, ev = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=1000.0, cfg=cfg,
            power_since=999.0,
        )
        assert ev is None

    def test_the_backdate_can_never_exceed_the_debounce(self):
        """A sensor holding one value for hours must not backdate the clock by hours —
        that would arm instantly on a machine already mid-cycle after a restart."""
        cfg = AppliancePowerConfig()
        st = AppliancePowerState()
        st, _ev = update_appliance_power_state(
            st, power_w=16.0, switch_on=True, now_ts=10_000.0, cfg=cfg,
            power_since=1.0,  # ~3 hours old
        )
        assert st.above_since == 10_000.0 - cfg.start_debounce_s


class TestCycleEnergyInput:
    """2026-09-02: the washer ran through the day's three most expensive hours.

    The scorer was fed sensor.darkstar_washer_last_cycle_energy's STATE — literally the
    previous run — where the code intends the LEARNED median that rides on the same
    entity as the typical_energy_kwh attribute. The previous run had been correctly
    deferred, released at 12:07, never resumed by hand, and measured 0.057 kWh of
    tumbling with no heat phase. That fragment became the next decision's entire cost
    basis.

    It is an off switch, not a bias: the electricity term scales with kWh while the wait
    penalty is absolute SEK/hour, so below break-even "run now" wins for EVERY price
    curve. Nothing warned.
    """

    ENTITY = "sensor.darkstar_washer_last_cycle_energy"

    def _cfg(self, tmp_path, prices, now, *, wait_cost=0.05):
        raw = _full_cfg(observe_only=True)
        raw["deferrable_loads"][0]["wait_cost_sek_per_hour"] = wait_cost
        cfg = parse_deferrable_runtime_config(raw)
        cfg.schedule_path = _write_schedule(tmp_path, prices, now)
        return cfg

    async def _arm(self, tmp_path, cfg, *, state, attrs):
        """Drive an arm and return the washer's recommended action."""
        ctrl = DeferrableApplianceController(cfg, state_file=str(tmp_path / "s.json"))
        now = datetime.now(TZ)
        ha = FakeHA(
            {
                "sensor.tvattmaskin_power": "2000", "switch.tvattmaskin": "on",
                "input_boolean.washing_machine_override": "off",
                self.ENTITY: state,
            },
            attributes={self.ENTITY: attrs},
        )
        t = now.timestamp()
        await ctrl.run(ha, t, now, shadow=False)
        await ctrl.run(ha, t + 3.0, now, shadow=False)
        _st, published = ha.published["sensor.darkstar_washer_state"]
        return published["recommended_action"], ha

    # A cheap block sits well after the expensive present — deferring is worth ~1 SEK
    # at a real cycle energy, and worth nothing at a fragment.
    PRICES = [2.5] * 8 + [1.5] * 24

    @pytest.mark.asyncio
    async def test_the_incident_a_fragment_state_no_longer_decides(self, tmp_path):
        """The regression. State 0.1 (the fragment), attribute 0.955 (the median)."""
        cfg = self._cfg(tmp_path, self.PRICES, datetime.now(TZ))
        action, _ha = await self._arm(
            tmp_path, cfg,
            state="0.1",
            attrs={"typical_energy_kwh": 0.955, "learned": True, "cycles_observed": 5},
        )
        assert action == "defer"

    @pytest.mark.asyncio
    async def test_the_old_behaviour_would_have_run(self, tmp_path):
        """Proof the test above discriminates: with only the fragment available, the
        wait penalty dominates and the machine runs at the peak — exactly what happened.
        Here the floor catches it and drops the penalty, so it defers anyway; without
        BOTH fixes this is the failing case."""
        cfg = self._cfg(tmp_path, self.PRICES, datetime.now(TZ))
        action, ha = await self._arm(tmp_path, cfg, state="0.1", attrs={})
        assert action == "defer"  # saved by the plausibility floor, not by the median

    @pytest.mark.asyncio
    async def test_an_unlearned_median_is_not_trusted(self, tmp_path):
        """A median over too few cycles carries the same weakness as one raw sample."""
        cfg = self._cfg(tmp_path, self.PRICES, datetime.now(TZ))
        action, _ha = await self._arm(
            tmp_path, cfg,
            state="1.2",
            attrs={"typical_energy_kwh": 0.05, "learned": False},
        )
        # Falls back to the state's 1.2, which is a real cycle => ordinary deferral.
        assert action == "defer"

    @pytest.mark.asyncio
    async def test_no_attributes_falls_back_to_the_state(self, tmp_path):
        """An appliance whose publisher predates the attribute behaves as before."""
        cfg = self._cfg(tmp_path, self.PRICES, datetime.now(TZ))
        action, _ha = await self._arm(tmp_path, cfg, state="1.2", attrs={})
        assert action == "defer"

    @pytest.mark.asyncio
    async def test_a_real_cycle_still_respects_the_wait_penalty(self, tmp_path):
        """The penalty must still work — it exists because a formally-optimal defer to
        01:30 over a 23:00 start that cost seven öre more is not what anyone means by
        'run it when power is cheap'. A trivially cheaper block far away must not win."""
        now = datetime.now(TZ)
        # Present is 1.00; a block 6 h out is 0.99. One öre/kWh is not worth six hours.
        prices = [1.0] * 24 + [0.99] * 24
        cfg = self._cfg(tmp_path, prices, now, wait_cost=0.05)
        action, _ha = await self._arm(
            tmp_path, cfg,
            state="1.2",
            attrs={"typical_energy_kwh": 1.2, "learned": True, "cycles_observed": 9},
        )
        assert action == "run"
