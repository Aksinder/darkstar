"""S3: the plan->servo bridge — kepler's price-placed slots become servo floors.

The bridge is what makes night charging price-OPTIMAL: without it the only
grid-backed mechanism is the deadline backstop, which smears the required energy
evenly across the remaining hours regardless of price.
"""

import pytest

from executor.ev_surplus import EVSurplusConfig, EVSurplusInputs, compute_ev_surplus
from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import (
    FakeHA,
    _cfg_dict as _prio_cfg_dict,
    _fmb,
    _states,
    _tesla,
)

CFG = EVSurplusConfig(enabled=True)


def _no_surplus(chargers):
    return EVSurplusInputs(
        pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
        import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=chargers,
    )


class TestPureFloorMerge:
    def test_plan_floor_charges_with_zero_surplus(self):
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, plan_floor_w=3680.0)
        cmds = compute_ev_surplus(_no_surplus([fmb]), CFG)
        assert cmds[0].switch_on
        assert "plan" in cmds[0].reason

    def test_max_not_sum_with_deadline(self):
        """Deadline needs 4.4 kW, plan says 3.68: floor = max = 4.4 (never 8.1)."""
        tesla = _tesla(soc_percent=30.0, deadline_hours=1.5, plan_floor_w=3680.0)
        cmds = compute_ev_surplus(_no_surplus([tesla]), CFG)
        cmd = cmds[0]
        assert cmd.switch_on and "deadline" in cmd.reason
        # 4444 W -> ~6.4 A raw, snapped: never the 8.1 kW sum (11.7 A)
        assert cmd.set_current_a is not None and cmd.set_current_a <= 7.0

    def test_deadline_car_outranks_plan_car_under_scarcity(self):
        """Floor-class ordering: deadline urgency first, plan floors behind."""
        tesla = _tesla(soc_percent=20.0, deadline_hours=1.0, priority=1)
        fmb = _fmb(soc_percent=50.0, deadline_hours=None,
                   plan_floor_w=3680.0, priority=0)
        cmds = compute_ev_surplus(_no_surplus([tesla, fmb]), CFG)
        assert cmds[0].id == "tesla"  # emitted first despite higher priority number

    def test_surplus_can_exceed_the_plan_floor(self):
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, plan_floor_w=1500.0)
        inputs = EVSurplusInputs(
            pv_w=6000.0, grid_w=0.0, battery_w=4000.0, battery_soc_percent=95.0,
            import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=[fmb],
        )
        cmds = compute_ev_surplus(inputs, CFG)
        assert cmds[0].switch_on
        assert cmds[0].set_current_a > 1500.0 / 230.0  # above the floor amps


def _bridge_cfg(plan_floor=True):
    raw = _prio_cfg_dict()
    for c in raw["ev_surplus"]["chargers"]:
        c["plan_floor"] = plan_floor
    planner = [
        {"id": "easee_fmb", "penalty_levels": [
            {"max_soc": 86, "penalty_sek": 2.0}, {"max_soc": 100, "penalty_sek": 0.35},
        ]},
        {"id": "tesla", "penalty_levels": [
            {"max_soc": 40, "penalty_sek": 2.5}, {"max_soc": 90, "penalty_sek": 0.4},
        ]},
    ]
    return parse_ev_surplus_config(raw, planner_ev_chargers=planner)


class TestRuntimeBridge:
    def test_gate_is_single_sourced_from_planner_band_one(self):
        cfg = _bridge_cfg()
        fmb = next(c for c in cfg.chargers if c.id == "easee_fmb")
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert fmb.plan_gate_soc == 86.0   # NOT floor_soc 40 — the S3 note
        assert tesla.plan_gate_soc == 40.0

    @pytest.mark.asyncio
    async def test_plan_floor_grid_charges_below_gate(self):
        """FMB at 50 % (< 86 gate), plan 3.7 kW, zero surplus: grid-backed charge."""
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"input_number.fmb_soc": "50", "sensor.tesla_soc": "60"}))
        summary = await ctl.run(ha, now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is True
        assert "plan" in cmds["easee_fmb"]["why"]

    @pytest.mark.asyncio
    async def test_gate_blocks_topup_band_plans(self):
        """FMB at 90 % (>= 86): the planned slot is a hint, never a grid floor (F11)."""
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"input_number.fmb_soc": "90", "sensor.tesla_soc": "60"}))
        summary = await ctl.run(ha, now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is False

    @pytest.mark.asyncio
    async def test_continuity_hold_survives_replan_flip(self):
        """Plan present tick 1, gone tick 2 (checkerboard replan): floor held."""
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        states = _states(**{"input_number.fmb_soc": "50", "sensor.tesla_soc": "60"})
        await ctl.run(FakeHA(states), now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        summary = await ctl.run(FakeHA(states), now_ts=1120.0, plan_kw={})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is True  # held despite empty plan

    @pytest.mark.asyncio
    async def test_hold_expires_after_thirty_minutes(self):
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        states = _states(**{"input_number.fmb_soc": "50", "sensor.tesla_soc": "60"})
        await ctl.run(FakeHA(states), now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        summary = await ctl.run(FakeHA(states), now_ts=1000.0 + 1900.0, plan_kw={})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is False  # hold expired, no plan, no surplus

    @pytest.mark.asyncio
    async def test_plan_floor_false_ignores_plans(self):
        cfg = _bridge_cfg(plan_floor=False)
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"input_number.fmb_soc": "50", "sensor.tesla_soc": "60"}))
        summary = await ctl.run(ha, now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is False

    def test_no_planner_entry_gates_everything(self):
        raw = _prio_cfg_dict()
        for c in raw["ev_surplus"]["chargers"]:
            c["plan_floor"] = True
        cfg = parse_ev_surplus_config(raw, planner_ev_chargers=[])
        assert all(c.plan_gate_soc is None for c in cfg.chargers)


class TestS3ReviewFixes:
    """Regression guards for the verified S3 review findings."""

    def test_plan_only_floor_never_borrows_deadline_urgency(self):
        """CRITICAL: both cars carry the 07:30 recurring deadline_hours at night.
        The FMB above its floor_soc has a PLAN-only floor — it must sort behind the
        Tesla's genuine deadline floor, not tie on borrowed urgency and win on
        priority (which starved the Tesla under fuse scarcity)."""
        tesla = _tesla(soc_percent=25.0, deadline_hours=1.5, priority=1)  # 6.7 kW need
        fmb = _fmb(
            soc_percent=50.0, floor_soc_percent=40.0,  # floor met => deadline_w 0
            deadline_hours=1.5,  # recurring deadline present, NOT binding
            plan_floor_w=3680.0, priority=0,
        )
        inputs = EVSurplusInputs(
            pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
            import_price_sek=1.0, remaining_solar_kwh=0.0,
            phase_currents_a={"a": 5.0, "b": 5.0, "c": 5.0},
            chargers=[tesla, fmb],
        )
        cmds = compute_ev_surplus(
            inputs, EVSurplusConfig(enabled=True, fuse_budget_a=23.0)
        )
        assert cmds[0].id == "tesla" and cmds[0].switch_on  # deadline served FIRST
        by_id = {c.id: c for c in cmds}
        # The Tesla's genuine need ((40-25)% * 60 kWh / 0.9 / 1.5 h = 6.67 kW ≈ 9.7 A)
        # is fully met; the FMB's comfort plan gets only what remains of the budget.
        assert by_id["tesla"].set_current_a >= 9.0

    @pytest.mark.asyncio
    async def test_vacation_gates_plan_floors_and_clears_hold(self):
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        states = _states(**{
            "input_number.fmb_soc": "50", "sensor.tesla_soc": "60",
            "input_boolean.vacation": "on",
        })
        summary = await ctl.run(FakeHA(states), now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is False
        assert ctl._plan_hold == {}
        assert ctl.last_plan_note.get("easee_fmb") == "vacation"

    @pytest.mark.asyncio
    async def test_hold_is_one_replan_period_not_thirty_minutes(self):
        """The rolling 30-min hold fired at every genuine block end (nightly tail
        buying not-worth-it energy). Now: bridges a 15-min flip, dead by ~16 min."""
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        states = _states(**{"input_number.fmb_soc": "50", "sensor.tesla_soc": "60"})
        await ctl.run(FakeHA(states), now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        mid = await ctl.run(FakeHA(states), now_ts=1000.0 + 900.0, plan_kw={})
        cmds_mid = {c["id"]: c for c in mid.get("applied", [])}
        assert cmds_mid["easee_fmb"]["on"] is True  # 15-min gap bridged
        late = await ctl.run(FakeHA(states), now_ts=1000.0 + 1000.0, plan_kw={})
        cmds_late = {c["id"]: c for c in late.get("applied", [])}
        assert cmds_late["easee_fmb"]["on"] is False  # tail bounded to ~16 min

    def test_f4_conflict_hard_disables_plan_floor(self):
        raw = _prio_cfg_dict()
        for c in raw["ev_surplus"]["chargers"]:
            c["plan_floor"] = True
        planner = [
            {"id": "easee_fmb", "switch_entity": "switch.easee",
             "penalty_levels": [{"max_soc": 86, "penalty_sek": 2.0}]},
            {"id": "tesla", "switch_entity": "",
             "penalty_levels": [{"max_soc": 40, "penalty_sek": 2.5}]},
        ]
        cfg = parse_ev_surplus_config(raw, planner_ev_chargers=planner)
        fmb = next(c for c in cfg.chargers if c.id == "easee_fmb")
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert fmb.plan_floor is False  # hard-disabled, not just warned
        assert tesla.plan_floor is True

    @pytest.mark.asyncio
    async def test_gated_plan_reports_note_for_notifier(self):
        cfg = _bridge_cfg()
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"input_number.fmb_soc": "90", "sensor.tesla_soc": "60"}))
        await ctl.run(ha, now_ts=1000.0, plan_kw={"easee_fmb": 3.7})
        assert ctl.last_plan_note.get("easee_fmb") == "gated"
