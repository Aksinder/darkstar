"""Anti-hunt hardening of the EV surplus servo (2026-08-04 design review).

Pins the adversarially-verified redesign: quantum-aware fleet deadband (K=1.5),
1 A command grid + runtime Schmitt quantizer, per-charger write pacing with
direction-aware intervals, min-OFF dwell, the multi-charger start-kick deadlock
fix, the F3 capped-charger exclusion, battery-tier SoC hysteresis, the
phantom-car fix, and the Easee 6 A hard floor (owner-confirmed: the FMB stops
charging below 6 A).
"""

import asyncio

import pytest

from executor.ev_surplus import (
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    WriteGuardConfig,
    battery_tier_active,
    compute_ev_surplus,
    should_write_current,
)
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config


def _tesla(**over) -> ChargerState:
    kw: dict = {
        "id": "tesla", "plugged": True, "at_home": True, "enabled": True,
        "current_power_w": 0.0, "max_current_a": 16.0, "min_current_a": 5.0,
        "phases": 3, "voltage_v": 230.0, "controllable": True, "priority": 1,
    }
    kw.update(over)
    return ChargerState(**kw)


def _easee(**over) -> ChargerState:
    kw: dict = {
        "id": "easee", "plugged": True, "at_home": True, "enabled": True,
        "current_power_w": 0.0, "max_current_a": 16.0, "min_current_a": 6.0,
        "phases": 1, "voltage_v": 230.0, "controllable": True, "priority": 0,
    }
    kw.update(over)
    return ChargerState(**kw)


def _inputs(chargers, *, grid_w=0.0, battery_w=0.0, **over) -> EVSurplusInputs:
    kw: dict = {
        "pv_w": 0.0, "grid_w": grid_w, "battery_w": battery_w,
        "battery_soc_percent": 90.0, "import_price_sek": 1.0,
        "remaining_solar_kwh": 0.0, "chargers": chargers,
    }
    kw.update(over)
    return EVSurplusInputs(**kw)


def _cfg(**over) -> EVSurplusConfig:
    kw: dict = {"enabled": True}
    kw.update(over)
    return EVSurplusConfig(**kw)


def _cmd(commands, cid):
    return next(c for c in commands if c.id == cid)


# --- quantum-aware deadband -------------------------------------------------


class TestQuantumDeadband:
    def test_tesla_on_widens_band_to_hold(self):
        """Tesla commanded-ON: band = 1.5 x 690 = 1035 W -> 900 W headroom holds."""
        tesla = _tesla(commanded_on=True, current_power_w=9660.0)  # 14 A
        cmds = compute_ev_surplus(_inputs([tesla], grid_w=-900.0), _cfg())
        assert "band=1035" in _cmd(cmds, "tesla").reason
        # Hold: target stays at the measured total -> same 14 A.
        assert _cmd(cmds, "tesla").set_current_a == pytest.approx(14.0)

    def test_tesla_on_acts_above_band(self):
        tesla = _tesla(commanded_on=True, current_power_w=9660.0)
        cmds = compute_ev_surplus(_inputs([tesla], grid_w=-1500.0), _cfg())
        assert _cmd(cmds, "tesla").set_current_a > 14.0

    def test_easee_only_keeps_narrow_band(self):
        """Easee-only: 1.5 x 230 = 345 W band — 400 W headroom still acts."""
        easee = _easee(commanded_on=True, current_power_w=1840.0)  # 8 A
        cmds = compute_ev_surplus(_inputs([easee], grid_w=-400.0), _cfg())
        assert "band=345" in _cmd(cmds, "easee").reason

    def test_no_on_chargers_uses_config_deadband(self):
        tesla = _tesla(commanded_on=False)
        cmds = compute_ev_surplus(_inputs([tesla], grid_w=-100.0), _cfg())
        # 100 W < 250 W config band -> hold at 0 => off command.
        assert not _cmd(cmds, "tesla").switch_on


# --- start-kick deadlock fix ------------------------------------------------


class TestStartKick:
    def test_deadlock_regression_second_charger_starts(self):
        """Easee@max + Tesla OFF + 6 kW surplus: the damped step gives the Tesla only
        ~3 kW — below its ~3.97 kW start threshold — so without the kick it stays
        stranded (the verified deadlock: any surplus < ~7.9 kW). The undamped
        allocation clears the threshold, so the kick must start it."""
        easee = _easee(commanded_on=True, current_power_w=3680.0)  # 16 A @ max
        tesla = _tesla(commanded_on=False)
        cmds = compute_ev_surplus(_inputs([easee, tesla], grid_w=-6000.0), _cfg())
        assert _cmd(cmds, "tesla").switch_on
        assert "kick=tesla" in _cmd(cmds, "tesla").reason

    def test_kick_gated_by_dwell(self):
        easee = _easee(commanded_on=True, current_power_w=3680.0)
        tesla = _tesla(commanded_on=False, start_inhibited=True)
        cmds = compute_ev_surplus(_inputs([easee, tesla], grid_w=-8000.0), _cfg())
        assert not _cmd(cmds, "tesla").switch_on
        assert "dwell" in _cmd(cmds, "tesla").reason

    def test_deadline_floor_punches_through_dwell(self):
        tesla = _tesla(
            commanded_on=False, start_inhibited=True,
            soc_percent=30.0, target_soc_percent=80.0, capacity_kwh=60.0,
            deadline_hours=4.0,
        )
        cmds = compute_ev_surplus(_inputs([tesla], grid_w=0.0), _cfg())
        assert _cmd(cmds, "tesla").switch_on
        assert "deadline" in _cmd(cmds, "tesla").reason


# --- F3: capped charger excluded from the fleet total ------------------------


class TestCappedExclusion:
    def test_capped_easee_watts_not_redistributed(self):
        """The just-capped Easee still MEASURES 3.68 kW — excluding it from the
        total prevents an instant multi-amp Tesla jump before the meter settles."""
        easee = _easee(
            commanded_on=True, current_power_w=3680.0,
            soc_percent=100.0, target_soc_percent=100.0,
        )
        tesla = _tesla(commanded_on=True, current_power_w=6900.0)  # 10 A
        cmds = compute_ev_surplus(_inputs([easee, tesla], grid_w=0.0, battery_w=0.0), _cfg())
        assert not _cmd(cmds, "easee").switch_on  # capped
        # Fleet total counts ONLY the Tesla (6.9 kW), headroom 0 -> band-hold at 10 A;
        # the old bug added Easee's 3.68 kW to the total and jumped the Tesla.
        assert _cmd(cmds, "tesla").set_current_a == pytest.approx(10.0)


# --- battery-tier SoC hysteresis ---------------------------------------------


class TestBatteryTierHysteresis:
    def _cfg(self):
        return _cfg(
            battery_assist_enabled=True, battery_assist_max_price_sek=0.0,
            battery_assist_min_remaining_solar_kwh=0.0, battery_assist_floor_soc=40.0,
        )

    def test_prev_off_needs_floor_plus_hysteresis(self):
        inp = _inputs([], battery_soc_percent=41.0, import_price_sek=-0.1,
                      battery_tier_active_prev=False)
        assert not battery_tier_active(inp, self._cfg())

    def test_prev_on_keeps_running_above_floor(self):
        inp = _inputs([], battery_soc_percent=41.0, import_price_sek=-0.1,
                      battery_tier_active_prev=True)
        assert battery_tier_active(inp, self._cfg())

    def test_below_floor_always_off(self):
        inp = _inputs([], battery_soc_percent=39.5, import_price_sek=-0.1,
                      battery_tier_active_prev=True)
        assert not battery_tier_active(inp, self._cfg())


# --- direction-aware write guard --------------------------------------------


class TestDirectionAwareGuard:
    def test_up_paced_hard_down_fast(self):
        g = WriteGuardConfig(min_step_a=1.0, min_interval_s=90.0,
                             min_interval_up_s=600.0, min_interval_down_s=90.0)
        assert not should_write_current(10.0, 1000.0, 11.0, 1000.0 + 300.0, g)  # up @300s: no
        assert should_write_current(10.0, 1000.0, 11.0, 1000.0 + 600.0, g)  # up @600s: yes
        assert should_write_current(10.0, 1000.0, 9.0, 1000.0 + 90.0, g)  # down @90s: yes

    def test_none_falls_back_to_min_interval(self):
        g = WriteGuardConfig(min_step_a=1.0, min_interval_s=90.0)
        assert should_write_current(10.0, 1000.0, 11.0, 1000.0 + 90.0, g)


# --- raw_amps emitted --------------------------------------------------------


def test_raw_amps_is_presnap_target():
    tesla = _tesla(commanded_on=True, current_power_w=6900.0)  # 10 A
    cmds = compute_ev_surplus(_inputs([tesla], grid_w=-2000.0), _cfg())
    cmd = _cmd(cmds, "tesla")
    # target = 6900 + 0.5*2000 = 7900 W -> raw 11.45 A, snapped 11 A.
    assert cmd.raw_amps == pytest.approx(7900.0 / 690.0, abs=0.01)
    assert cmd.set_current_a == pytest.approx(11.0)


# --- Easee floor property fuzz ----------------------------------------------


def test_easee_floor_never_violated_fuzz():
    """No random state may ever command the Easee in (0,6) A or Tesla in (0,5) A."""
    import random

    rng = random.Random(42)
    for _ in range(2000):
        easee = _easee(
            commanded_on=rng.choice([True, False, None]),
            current_power_w=rng.uniform(0, 3700),
            soc_percent=rng.choice([None, rng.uniform(0, 100)]),
            target_soc_percent=rng.choice([None, rng.uniform(0, 100)]),
            capacity_kwh=rng.choice([0.0, 28.0]),
            deadline_hours=rng.choice([None, rng.uniform(0.1, 24)]),
            start_inhibited=rng.random() < 0.2,
        )
        tesla = _tesla(
            commanded_on=rng.choice([True, False, None]),
            current_power_w=rng.uniform(0, 11000),
            start_inhibited=rng.random() < 0.2,
        )
        inp = _inputs(
            [easee, tesla],
            grid_w=rng.uniform(-15000, 5000),
            battery_w=rng.uniform(-5000, 5000),
            battery_soc_percent=rng.uniform(0, 100),
            import_price_sek=rng.uniform(-1, 3),
        )
        for cmd in compute_ev_surplus(inp, _cfg()):
            if cmd.set_current_a is None:
                continue
            a = round(cmd.set_current_a)
            if cmd.id == "easee":
                assert not (0 < a < 6), f"Easee {a} A: {cmd.reason}"
            else:
                assert not (0 < a < 5), f"Tesla {a} A: {cmd.reason}"


# --- runtime: parse, phantom, Schmitt, R1, isolation, core-skip ---------------


class FakeHA:
    def __init__(self, states):
        self.states = states
        self.calls: list[tuple] = []

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        if entity not in self.states:
            return None
        return {"state": self.states.get(entity), "attributes": {}}

    async def call_service(self, domain, service, entity_id=None, data=None):
        self.calls.append((domain, service, entity_id, data))
        return True


def _runtime_cfg(**charger_over):
    tesla = {
        "id": "tesla", "priority": 1, "min_current_a": 5, "max_current_a": 16, "phases": 3,
        "switch_entity": "switch.tesla", "current_entity": "number.tesla_amps",
        "power_entity": "sensor.tesla_power", "plug_entity": "binary_sensor.tesla_plug",
    }
    tesla.update(charger_over)
    return parse_ev_surplus_config(
        {
            "ev_surplus": {
                "enabled": True,
                "grid_power_entity": "sensor.grid",
                "battery_power_entity": "sensor.batt",
                "battery_soc_entity": "sensor.soc",
                "price_entity": "sensor.price",
                "policy": {"current_step_a": 1.0},
                "write_guard": {"min_step_a": 1.0, "min_interval_s": 90},
                "chargers": [tesla],
            }
        }
    )


class TestRuntimeHardening:
    def test_parse_clamps_easee_min_below_six(self):
        cfg = parse_ev_surplus_config(
            {
                "ev_surplus": {
                    "enabled": True,
                    "chargers": [
                        {"id": "easee", "easee_device_id": "dev", "min_current_a": 5}
                    ],
                }
            }
        )
        assert cfg is not None
        assert cfg.chargers[0].min_current_a == 6.0

    def test_parse_per_charger_guard_and_shadow(self):
        cfg = _runtime_cfg(
            write_guard={"min_step_a": 1.0, "min_interval_up_s": 600, "min_interval_down_s": 90},
            min_off_s=900,
            shadow=True,
        )
        assert cfg is not None
        c = cfg.chargers[0]
        assert c.write_guard is not None
        assert c.write_guard.min_interval_up_s == 600.0
        assert c.min_off_s == 900.0
        assert c.shadow is True

    def test_phantom_configured_plug_unreadable_means_unplugged(self):
        cfg = _runtime_cfg()
        assert cfg is not None
        ctrl = EVSurplusController(cfg)
        ha = FakeHA({"sensor.grid": "-8000", "sensor.batt": "0", "sensor.soc": "90",
                     "sensor.price": "1.0"})  # tesla entities all MISSING
        result = asyncio.run(ctrl.run(ha, now_ts=1000.0))
        # Not plugged -> no active chargers -> no commands, no service calls.
        assert result.get("applied") == []
        assert ha.calls == []

    def test_runtime_hard_refuses_sub_six_easee_write(self):
        cfg = parse_ev_surplus_config(
            {
                "ev_surplus": {
                    "enabled": True,
                    "chargers": [{"id": "easee", "easee_device_id": "dev"}],
                }
            }
        )
        assert cfg is not None
        ctrl = EVSurplusController(cfg)
        ha = FakeHA({})

        class Cmd:
            id = "easee"
            switch_on = True
            set_current_a = 5.0
            raw_amps = 5.0
            reason = "forged"

        asyncio.run(ctrl._actuate(ha, cfg.chargers[0], Cmd(), 1000.0, False))
        assert ha.calls == []  # refused

    def test_core_sensor_unreadable_skips_tick(self):
        cfg = _runtime_cfg()
        assert cfg is not None
        ctrl = EVSurplusController(cfg)
        ha = FakeHA({"sensor.batt": "0", "sensor.soc": "90", "sensor.price": "1.0",
                     "binary_sensor.tesla_plug": "on", "sensor.tesla_power": "0"})
        result = asyncio.run(ctrl.run(ha, now_ts=1000.0))  # grid missing
        assert result.get("skipped") == "core sensors unreadable"
        assert ha.calls == []

    def test_schmitt_suppresses_midpoint_dither(self):
        cfg = _runtime_cfg()
        assert cfg is not None
        ctrl = EVSurplusController(cfg)
        ctrl._last_a["tesla"] = 14.0
        ctrl._last_ts["tesla"] = 0.0
        ctrl._last_switch["tesla"] = True
        ha = FakeHA({})

        class Cmd:
            id = "tesla"
            switch_on = True
            set_current_a = 15.0
            raw_amps = 14.55  # only 0.55 A from written 14 -> below 0.7 fraction
            reason = "dither"

        asyncio.run(ctrl._actuate(ha, cfg.chargers[0], Cmd(), 10_000.0, False))
        assert ha.calls == []  # suppressed

        Cmd.raw_amps = 14.75  # cleared the 0.7 fraction
        asyncio.run(ctrl._actuate(ha, cfg.chargers[0], Cmd(), 10_000.0, False))
        assert ("number", "set_value", "number.tesla_amps", {"value": 15.0}) in ha.calls

    def test_r1_switch_stop_zeroes_commanded_amps(self):
        """R1: after a switch-path stop, commanded_on must read False next cycle —
        else the start kick, min-OFF dwell and start hysteresis are all inert."""
        cfg = _runtime_cfg(min_off_s=900)
        assert cfg is not None
        ctrl = EVSurplusController(cfg)
        ctrl._last_a["tesla"] = 14.0
        ctrl._last_switch["tesla"] = True
        ha = FakeHA({"binary_sensor.tesla_plug": "on", "sensor.tesla_power": "9000"})

        class Stop:
            id = "tesla"
            switch_on = False
            set_current_a = 0.0
            raw_amps = None
            reason = "off"

        asyncio.run(ctrl._actuate(ha, cfg.chargers[0], Stop(), 1000.0, False))
        assert ctrl._last_a["tesla"] == 0.0
        assert ctrl._last_stop_ts["tesla"] == 1000.0

        state = asyncio.run(ctrl._read_charger(ha, cfg.chargers[0], 1500.0, False))
        assert state.commanded_on is False
        assert state.start_inhibited is True  # within min_off_s
        state2 = asyncio.run(ctrl._read_charger(ha, cfg.chargers[0], 2000.0, False))
        assert state2.start_inhibited is False  # dwell expired

    def test_actuation_isolation_second_charger_survives(self):
        cfg = parse_ev_surplus_config(
            {
                "ev_surplus": {
                    "enabled": True,
                    "grid_power_entity": "sensor.grid",
                    "battery_power_entity": "sensor.batt",
                    "battery_soc_entity": "sensor.soc",
                    "price_entity": "sensor.price",
                    "policy": {"current_step_a": 1.0},
                    "chargers": [
                        {"id": "a", "priority": 0, "min_current_a": 6, "phases": 1,
                         "switch_entity": "switch.a", "current_entity": "number.a",
                         "plug_entity": "binary_sensor.a_plug"},
                        {"id": "b", "priority": 1, "min_current_a": 6, "phases": 1,
                         "switch_entity": "switch.b", "current_entity": "number.b",
                         "plug_entity": "binary_sensor.b_plug"},
                    ],
                }
            }
        )
        assert cfg is not None
        ctrl = EVSurplusController(cfg)

        class ExplodingHA(FakeHA):
            async def call_service(self, domain, service, entity_id=None, data=None):
                if entity_id in ("switch.a", "number.a"):
                    raise RuntimeError("dead entity")
                return await super().call_service(domain, service, entity_id, data)

        ha = ExplodingHA({"sensor.grid": "-8000", "sensor.batt": "0", "sensor.soc": "90",
                          "sensor.price": "1.0",
                          "binary_sensor.a_plug": "on", "binary_sensor.b_plug": "on"})
        result = asyncio.run(ctrl.run(ha, now_ts=1000.0))
        # Charger a exploded, but b still got actuated.
        assert any(c[2] in ("switch.b", "number.b") for c in ha.calls)
        assert any(a["id"] == "b" for a in result.get("applied", []))
