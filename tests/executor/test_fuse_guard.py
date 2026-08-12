"""S4: the 25 A/phase main-fuse guard.

Review-mandated cases: the 32 A two-car stack, VVB-on-the-same-phase, REDUCTION of an
already-running charger (the formula bug the first design shipped), sub-min => OFF even
for deadline floors (fuse trumps punch-through), conservative unknown-phase budgeting,
guard-exempt relief writes, the battery shed lever, and the stale-sensor fail-safe.
"""

from datetime import UTC, datetime, timedelta

import pytest

from executor.ev_surplus import (
    EVSurplusConfig,
    EVSurplusInputs,
    WriteGuardConfig,
    compute_ev_surplus,
    fuse_battery_charge_cap_w,
    should_write_current,
)
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import _fmb, _tesla

BUDGET = EVSurplusConfig(enabled=True, fuse_budget_a=23.0)


def _inputs(chargers, phases, surplus_w=20000.0):
    """Surplus delivered via battery absorption (grid ~0) so the |A| readings stay
    physically coherent with the signed grid — the export credit is exercised only
    by tests that set grid_w negative explicitly WITH matching readings."""
    return EVSurplusInputs(
        pv_w=surplus_w, grid_w=0.0, battery_w=surplus_w, battery_soc_percent=95.0,
        import_price_sek=1.0, remaining_solar_kwh=0.0,
        phase_currents_a=phases, chargers=chargers,
    )


class TestPhaseClamp:
    def test_two_cars_share_the_phase_budget(self):
        """32 A case: FMB (unknown phase => all) takes 16 A, the Tesla's leftover
        5 A sits below its cold-start threshold => OFF. Never 32 A on one phase."""
        fmb = _fmb(soc_percent=50.0, deadline_hours=None)
        tesla = _tesla(soc_percent=60.0, deadline_hours=None)
        cmds = {c.id: c for c in compute_ev_surplus(
            _inputs([fmb, tesla], {"a": 2.0, "b": 2.0, "c": 2.0}), BUDGET
        )}
        assert cmds["easee_fmb"].switch_on
        assert cmds["easee_fmb"].set_current_a == 16.0
        assert not cmds["tesla"].switch_on
        assert cmds["tesla"].fuse_limited

    def test_mapped_phases_budget_independently(self):
        """Two 1-phase cars on DIFFERENT mapped phases don't rob each other."""
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, phase_map=("a",))
        other = _fmb(
            id="other", soc_percent=50.0, deadline_hours=None,
            phase_map=("b",), priority=1,
        )
        cmds = {c.id: c for c in compute_ev_surplus(
            _inputs([fmb, other], {"a": 2.0, "b": 2.0, "c": 2.0}), BUDGET
        )}
        assert cmds["easee_fmb"].set_current_a == 16.0
        assert cmds["other"].set_current_a == 16.0

    def test_vvb_on_the_mapped_phase_squeezes_the_car(self):
        """VVB (14.8 A) + house on phase a leaves ~3 A: below the FMB's 6 A floor => OFF."""
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, phase_map=("a",))
        cmds = compute_ev_surplus(
            _inputs([fmb], {"a": 20.0, "b": 3.0, "c": 3.0}), BUDGET
        )
        assert not cmds[0].switch_on and cmds[0].fuse_limited

    def test_running_charger_is_REDUCED_when_phase_overloads(self):
        """The critical formula case: Tesla ON at 16 A, phases at 24.5 A (over budget).
        delta = -1.5 => command ~14 A (floored snap), flagged fuse_limited."""
        tesla = _tesla(
            soc_percent=60.0, deadline_hours=None,
            current_power_w=16.0 * 3 * 230.0, commanded_on=True,
        )
        cmds = compute_ev_surplus(
            _inputs([tesla], {"a": 24.5, "b": 24.5, "c": 24.5}), BUDGET
        )
        cmd = cmds[0]
        assert cmd.switch_on and cmd.fuse_limited
        assert cmd.set_current_a == 14.0  # 16 - 1.5 = 14.5, floored to the 1 A grid

    def test_fuse_cap_below_min_kills_even_a_deadline_floor(self):
        """A grid-backed guarantee must never blow the main fuse."""
        tesla = _tesla(soc_percent=20.0, deadline_hours=1.0)  # urgent floor ~11 kW
        cmds = compute_ev_surplus(
            _inputs([tesla], {"a": 21.0, "b": 21.0, "c": 21.0}, surplus_w=0.0), BUDGET
        )
        assert not cmds[0].switch_on
        assert cmds[0].fuse_limited

    def test_deadline_floor_runs_capped_when_partially_squeezed(self):
        tesla = _tesla(soc_percent=20.0, deadline_hours=1.0)
        cmds = compute_ev_surplus(
            _inputs([tesla], {"a": 15.0, "b": 15.0, "c": 15.0}, surplus_w=0.0), BUDGET
        )
        cmd = cmds[0]
        assert cmd.switch_on and cmd.fuse_limited
        assert cmd.set_current_a == 8.0  # 23 - 15 headroom
        assert "deadline+fuse" in cmd.reason

    def test_guard_on_but_no_readings_allows_no_increase(self):
        fmb = _fmb(soc_percent=50.0, deadline_hours=None)
        cmds = compute_ev_surplus(_inputs([fmb], {}), BUDGET)
        assert not cmds[0].switch_on  # cold car, no increase allowed => stays off
        assert cmds[0].fuse_limited

    def test_guard_disabled_is_legacy(self):
        fmb = _fmb(soc_percent=50.0, deadline_hours=None)
        cmds = compute_ev_surplus(
            _inputs([fmb], {"a": 24.0, "b": 24.0, "c": 24.0}),
            EVSurplusConfig(enabled=True, fuse_budget_a=None),
        )
        assert cmds[0].switch_on and not cmds[0].fuse_limited


class TestBatteryShed:
    def test_headroom_grows_the_cap(self):
        cap = fuse_battery_charge_cap_w({"a": 20.0, "b": 15.0, "c": 10.0}, 3000.0, 23.0)
        assert cap == pytest.approx(3000.0 + 3.0 * 3 * 230.0)

    def test_overload_pulls_below_present_charge(self):
        cap = fuse_battery_charge_cap_w({"a": 25.0, "b": 15.0, "c": 10.0}, 3000.0, 23.0)
        assert cap == pytest.approx(3000.0 - 2.0 * 3 * 230.0)

    def test_blind_sensors_mean_zero(self):
        assert fuse_battery_charge_cap_w({}, 5000.0, 23.0) == 0.0

    def test_import_overload_still_sheds(self):
        """Signed grid says IMPORT: an overloaded phase pulls the setpoint down."""
        cap = fuse_battery_charge_cap_w(
            {"a": 25.0, "b": 15.0, "c": 10.0}, 3000.0, 23.0, grid_w=5000.0
        )
        assert cap == pytest.approx(3000.0 - 2.0 * 3 * 230.0)

    def test_export_overload_never_ratchets(self):
        """Review-caught positive feedback: pulling charge DOWN during export raises
        export 1:1. With the export credit the cap stays at/above present charge."""
        cap = fuse_battery_charge_cap_w(
            {"a": 24.0, "b": 24.0, "c": 24.0}, 2000.0, 23.0, grid_w=-16560.0
        )
        assert cap >= 2000.0

    def test_ev_alloc_is_subtracted(self):
        """The EV clamp and the battery cap must not double-spend one snapshot."""
        base = fuse_battery_charge_cap_w({"a": 10.0, "b": 10.0, "c": 10.0}, 0.0, 23.0)
        with_ev = fuse_battery_charge_cap_w(
            {"a": 10.0, "b": 10.0, "c": 10.0}, 0.0, 23.0,
            ev_alloc_a={"a": 8.0, "b": 8.0, "c": 8.0},
        )
        assert with_ev == pytest.approx(base - 8.0 * 3 * 230.0)


class TestReliefWrites:
    CFG = WriteGuardConfig(min_step_a=2.0, min_interval_s=90.0)

    def test_fuse_relief_bypasses_step_and_interval(self):
        assert should_write_current(16.0, 1000.0, 15.0, 1001.0, self.CFG, fuse_relief=True)

    def test_fuse_increase_still_paced(self):
        assert not should_write_current(10.0, 1000.0, 11.0, 1001.0, self.CFG, fuse_relief=True)

    def test_plain_reduction_still_paced(self):
        assert not should_write_current(16.0, 1000.0, 15.0, 1001.0, self.CFG)


def _fuse_cfg_dict():
    return {
        "ev_surplus": {
            "enabled": True,
            "pv_power_entity": "sensor.pv",
            "grid_power_entity": "sensor.grid",
            "battery_power_entity": "sensor.batt",
            "battery_soc_entity": "sensor.soc",
            "price_entity": "sensor.price",
            "fuse_guard": {
                "enabled": True,
                "limit_a": 25.0,
                "margin_a": 2.0,
                "stale_after_s": 180,
                "phase_entities": {
                    "a": "sensor.ph_a", "b": "sensor.ph_b", "c": "sensor.ph_c",
                },
            },
            "chargers": [
                {"id": "easee_fmb", "priority": 0, "min_current_a": 6,
                 "max_current_a": 16, "phases": 1, "easee_device_id": "dev",
                 "power_entity": "sensor.easee_power",
                 "plug_entity": "binary_sensor.easee_plug"},
                {"id": "tesla", "priority": 1, "min_current_a": 5, "max_current_a": 16,
                 "phases": 3, "switch_entity": "switch.tesla", "shadow": True,
                 "current_entity": "number.tesla_amps",
                 "power_entity": "sensor.tesla_power",
                 "plug_entity": "binary_sensor.tesla_plug",
                 "phase_map": ["A", "b", "c"]},
            ],
        },
        "timezone": "Europe/Stockholm",
    }


class FuseFakeHA:
    """FakeHA with last_updated support for the staleness checks."""

    def __init__(self, states, ages=None):
        self.states = states
        self.ages = ages or {}
        self.calls: list[tuple] = []
        self.now = datetime.now(UTC)

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        if entity not in self.states:
            return None
        lu = self.now - timedelta(seconds=self.ages.get(entity, 0))
        return {
            "state": self.states.get(entity),
            "attributes": {},
            "last_updated": lu.isoformat(),
        }

    async def call_service(self, domain, service, entity_id=None, data=None):
        self.calls.append((domain, service, entity_id, data))
        return True


class TestForceOnClamp:
    def test_force_on_is_fuse_capped(self):
        """A manual comfort override does not outrank the main fuse."""
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, override="force_on")
        cmds = compute_ev_surplus(
            _inputs([fmb], {"a": 24.5, "b": 24.5, "c": 24.5}, surplus_w=0.0), BUDGET
        )
        assert not cmds[0].switch_on and cmds[0].fuse_limited

    def test_force_on_partially_capped_runs_reduced(self):
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, override="force_on")
        cmds = compute_ev_surplus(
            _inputs([fmb], {"a": 13.0, "b": 3.0, "c": 3.0}, surplus_w=0.0), BUDGET
        )
        cmd = cmds[0]
        assert cmd.switch_on and cmd.fuse_limited
        assert cmd.set_current_a == 10.0  # 23 - 13 headroom (unknown phase => min)

    def test_force_on_grant_consumes_budget_for_auto_cars(self):
        forced = _fmb(soc_percent=50.0, deadline_hours=None, override="force_on")
        auto = _tesla(soc_percent=60.0, deadline_hours=None)
        cmds = {c.id: c for c in compute_ev_surplus(
            _inputs([forced, auto], {"a": 2.0, "b": 2.0, "c": 2.0}), BUDGET
        )}
        assert cmds["easee_fmb"].set_current_a == 16.0
        # 23 - 2 - 16 = 5 A left: below the Tesla's cold-start threshold => OFF.
        assert not cmds["tesla"].switch_on


class TestExportCredit:
    def test_ev_start_allowed_during_heavy_export(self):
        """Direction-blind |A| deadlocked EV starts during export (review-caught):
        an OFF car never changes the reading. The symmetric-export credit opens
        the budget — consuming more genuinely reduces those currents."""
        fmb = _fmb(soc_percent=50.0, deadline_hours=None)
        inputs = _inputs([fmb], {"a": 24.0, "b": 24.0, "c": 24.0})
        inputs.grid_w = -16560.0  # 24 A symmetric export
        cmds = compute_ev_surplus(inputs, BUDGET)
        assert cmds[0].switch_on
        assert cmds[0].set_current_a == 16.0


class TestRuntimeFuse:
    def test_parse_arms_the_pure_clamp(self):
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        assert cfg is not None and cfg.fuse_guard_enabled
        assert cfg.policy.fuse_budget_a == 23.0
        assert cfg.fuse_phase_entities == {
            "a": "sensor.ph_a", "b": "sensor.ph_b", "c": "sensor.ph_c",
        }
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.phase_map == ("a", "b", "c")
        fmb = next(c for c in cfg.chargers if c.id == "easee_fmb")
        assert fmb.phase_map == ()  # unknown => conservative

    @pytest.mark.asyncio
    async def test_stale_phase_sensors_stop_the_cars(self):
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        ctl = EVSurplusController(cfg)
        ctl._last_a["easee_fmb"] = 14.0  # charging per our own memory
        ha = FuseFakeHA(
            {"sensor.pv": "0", "sensor.grid": "0", "sensor.batt": "0",
             "sensor.soc": "50", "sensor.price": "1.0",
             "sensor.easee_power": "3200", "binary_sensor.easee_plug": "on",
             "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "on",
             "sensor.ph_a": "10", "sensor.ph_b": "10", "sensor.ph_c": "10"},
            ages={"sensor.ph_b": 999},  # one frozen phase = all-or-nothing stale
        )
        summary = await ctl.run(ha, now_ts=ha.now.timestamp())
        assert summary.get("fuse_failsafe") == "phase sensors stale"
        # The Easee got a real dynamic-limit-0 stop; the shadow Tesla got NO write.
        easee_stops = [c for c in ha.calls if c[0] == "easee" and c[3]["current"] == 0]
        assert easee_stops
        assert not [c for c in ha.calls if c[2] == "number.tesla_amps"]

    @pytest.mark.asyncio
    async def test_fresh_readings_feed_engine_cap(self):
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FuseFakeHA(
            {"sensor.pv": "0", "sensor.grid": "500", "sensor.batt": "2000",
             "sensor.soc": "50", "sensor.price": "1.0",
             "sensor.easee_power": "0", "binary_sensor.easee_plug": "off",
             "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "off",
             "sensor.ph_a": "20", "sensor.ph_b": "15", "sensor.ph_c": "10"},
        )
        now = ha.now.timestamp()
        await ctl.run(ha, now_ts=now)
        cap = ctl.fuse_battery_cap_w(now)
        assert cap == pytest.approx(2000.0 + 3.0 * 3 * 230.0)

    def test_engine_cap_semantics(self):
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        ctl = EVSurplusController(cfg)
        assert ctl.fuse_battery_cap_w(1000.0) == 0.0  # never read => fail safe
        cfg2 = parse_ev_surplus_config(
            {"ev_surplus": {"enabled": True, "chargers": [{"id": "x"}]}}
        )
        assert EVSurplusController(cfg2).fuse_battery_cap_w(1000.0) is None  # guard off


class TestReviewFixes:
    """Regression guards for the verified S4 review findings."""

    @pytest.mark.asyncio
    async def test_failsafe_respects_global_shadow(self):
        """Observe-only executor must not actuate live devices even to fail safe."""
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        ctl = EVSurplusController(cfg)
        ctl._last_a["easee_fmb"] = 14.0
        ha = FuseFakeHA(
            {"sensor.pv": "0", "sensor.grid": "0", "sensor.batt": "0",
             "sensor.soc": "50", "sensor.price": "1.0",
             "sensor.easee_power": "3200", "binary_sensor.easee_plug": "on",
             "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "on",
             "sensor.ph_a": "10", "sensor.ph_b": "10", "sensor.ph_c": "10"},
            ages={"sensor.ph_b": 999},
        )
        summary = await ctl.run(ha, now_ts=ha.now.timestamp(), shadow=True)
        assert summary.get("fuse_failsafe")
        assert ha.calls == []  # zero service calls in global shadow

    @pytest.mark.asyncio
    async def test_phase_read_exception_is_blindness_not_crash(self):
        """404 (renamed entity — this site's history) must reach the fail-safe."""
        cfg = parse_ev_surplus_config(_fuse_cfg_dict())
        ctl = EVSurplusController(cfg)
        ctl._last_a["easee_fmb"] = 14.0

        class RaisingHA(FuseFakeHA):
            async def get_state(self, entity):
                if entity == "sensor.ph_b":
                    raise RuntimeError("404 Not Found")
                return await super().get_state(entity)

        ha = RaisingHA(
            {"sensor.pv": "0", "sensor.grid": "0", "sensor.batt": "0",
             "sensor.soc": "50", "sensor.price": "1.0",
             "sensor.easee_power": "3200", "binary_sensor.easee_plug": "on",
             "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "on",
             "sensor.ph_a": "10", "sensor.ph_b": "10", "sensor.ph_c": "10"},
        )
        summary = await ctl.run(ha, now_ts=ha.now.timestamp())
        assert summary.get("fuse_failsafe") == "phase sensors stale"

    def test_parse_rejects_orphan_fuse_guard(self):
        """fuse_guard without servo/chargers/phase_entities would permanently zero
        the battery cap (never-read => 0) — reject loudly at parse."""
        raw = _fuse_cfg_dict()
        raw["ev_surplus"]["fuse_guard"]["phase_entities"] = {}
        cfg = parse_ev_surplus_config(raw)
        assert cfg is not None
        assert not cfg.fuse_guard_enabled
        assert cfg.policy.fuse_budget_a is None

    def test_parse_drops_typo_phase_map_to_conservative(self):
        raw = _fuse_cfg_dict()
        raw["ev_surplus"]["chargers"][1]["phase_map"] = ["l1", "l2", "l3"]
        cfg = parse_ev_surplus_config(raw)
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.phase_map == ()  # conservative, not silently missing lookups

    def test_engine_cap_hits_charge_value_too(self):
        """The critical finding: sungrow charge mode writes {{charge_value}} into
        BOTH battery registers — capping max_charge alone was a no-op."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from executor.engine import ExecutorEngine

        eng = MagicMock(spec=ExecutorEngine)
        eng.profile = None
        eng.config = SimpleNamespace(
            inverter=SimpleNamespace(control_unit="W")
        )
        eng._ev_surplus = SimpleNamespace(fuse_battery_cap_w=lambda _ts: 2350.0)
        decision = SimpleNamespace(charge_value=9500.0, max_charge=9500.0)
        ExecutorEngine._apply_fuse_battery_cap(eng, decision)
        assert decision.charge_value == 2400.0  # capped + rounded to 100 W
        assert decision.max_charge == 2400.0
