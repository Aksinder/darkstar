"""S2 priority package: floor_soc, deadline urgency ordering, recurring weekday
deadlines, the manual priority selector, and unconditional vacation deadline clear.

The floor/cap split is the critical regression guard: the Tesla's comfort cap is the
car's own charge_limit (90) but its weekday-morning guarantee is only ~40 — sizing the
grid-backed deadline floor from the cap grid-forced price-blind charging toward 90
every night.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from executor.ev_surplus import (
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    _deadline_required_w,
    compute_ev_surplus,
)
from executor.ev_surplus_runtime import (
    EVSurplusController,
    next_recurring_deadline_ts,
    parse_ev_surplus_config,
)

TZ = ZoneInfo("Europe/Stockholm")


def _ts(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


def _tesla(**over):
    base = {
        "id": "tesla", "plugged": True, "at_home": True, "enabled": True,
        "current_power_w": 0.0, "max_current_a": 16.0, "min_current_a": 5.0,
        "phases": 3, "voltage_v": 230.0, "controllable": True, "priority": 1,
        "soc_percent": 30.0, "target_soc_percent": 90.0, "floor_soc_percent": 40.0,
        "capacity_kwh": 60.0, "charge_efficiency": 0.9,
    }
    base.update(over)
    return ChargerState(**base)


def _fmb(**over):
    base = {
        "id": "easee_fmb", "plugged": True, "at_home": True, "enabled": True,
        "current_power_w": 0.0, "max_current_a": 16.0, "min_current_a": 6.0,
        "phases": 1, "voltage_v": 230.0, "controllable": True, "priority": 0,
        "soc_percent": 50.0, "target_soc_percent": 86.0, "floor_soc_percent": 86.0,
        "capacity_kwh": 28.0, "charge_efficiency": 0.9,
    }
    base.update(over)
    return ChargerState(**base)


class TestFloorSocDeadline:
    def test_floor_not_cap_sizes_the_deadline_floor(self):
        """soc 45 >= floor 40 => NO grid forcing, even though the 90-cap is far away."""
        c = _tesla(soc_percent=45.0, deadline_hours=1.5)
        assert _deadline_required_w(c, EVSurplusConfig()) == 0.0

    def test_below_floor_forces_toward_floor_energy_only(self):
        """soc 30 -> floor 40 over 1.5h: (0.10*60/0.9)/1.5 = 4.44 kW (not the 40 kWh to 90)."""
        c = _tesla(soc_percent=30.0, deadline_hours=1.5)
        req = _deadline_required_w(c, EVSurplusConfig())
        assert req == pytest.approx(4444.4, rel=0.01)

    def test_fallback_to_cap_when_floor_unset(self):
        """Legacy: no floor_soc => the cap still sizes the floor (backward compatible)."""
        c = _tesla(floor_soc_percent=None, soc_percent=85.0, deadline_hours=1.0)
        req = _deadline_required_w(c, EVSurplusConfig())
        # (0.05*60/0.9)/1h = 3.33 kW < min_on 3.45 kW => 0 (sub-min); at 80% it forces.
        assert req == 0.0
        c2 = _tesla(floor_soc_percent=None, soc_percent=80.0, deadline_hours=1.0)
        assert _deadline_required_w(c2, EVSurplusConfig()) > 0.0

    def test_sub_minimum_requirement_waits_for_solar(self):
        c = _tesla(soc_percent=30.0, deadline_hours=12.0)  # 555 W < 3450 W min-on
        assert _deadline_required_w(c, EVSurplusConfig()) == 0.0

    def test_floor_above_cap_is_clamped_to_cap(self):
        """A floor above the comfort cap must not size forcing toward unreachable SoC."""
        c = _tesla(
            floor_soc_percent=95.0, target_soc_percent=90.0,
            soc_percent=80.0, deadline_hours=1.0,
        )
        req = _deadline_required_w(c, EVSurplusConfig())
        # Sized from the 90-cap: (0.10*60/0.9)/1h = 6.67 kW — not from 95.
        assert req == pytest.approx(6666.7, rel=0.01)


class TestDeadlineUrgencyOrdering:
    def test_more_urgent_deadline_emitted_first_regardless_of_priority(self):
        """FMB prio 0 with a lazy deadline vs Tesla prio 1 with an urgent one: Tesla first."""
        tesla = _tesla(soc_percent=20.0, deadline_hours=1.0, priority=1)
        fmb = _fmb(soc_percent=20.0, deadline_hours=6.0, priority=0)
        inputs = EVSurplusInputs(
            pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
            import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=[tesla, fmb],
        )
        cmds = compute_ev_surplus(inputs, EVSurplusConfig(enabled=True))
        on = [c.id for c in cmds if c.switch_on]
        assert on and on[0] == "tesla"

    def test_surplus_class_still_ordered_by_priority(self):
        """No floors: plain surplus follows priority (fmb prio 0 fills before tesla)."""
        tesla = _tesla(soc_percent=60.0, deadline_hours=None, priority=1)
        fmb = _fmb(soc_percent=50.0, deadline_hours=None, priority=0)
        inputs = EVSurplusInputs(
            pv_w=6000.0, grid_w=-4000.0, battery_w=0.0, battery_soc_percent=95.0,
            import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=[tesla, fmb],
        )
        cmds = compute_ev_surplus(inputs, EVSurplusConfig(enabled=True))
        by_id = {c.id: c for c in cmds}
        # 4 kW surplus: FMB (max 3.68 kW, prio 0) takes it; tesla's leftover is sub-min.
        assert by_id["easee_fmb"].switch_on
        assert not by_id["tesla"].switch_on


class TestRecurringDeadline:
    def test_friday_evening_rolls_to_monday(self):
        # 2026-08-14 is a Friday.
        now = _ts(2026, 8, 14, 20, 0)
        ts = next_recurring_deadline_ts(
            ("mon", "tue", "wed", "thu", "fri"), "07:30", now, "Europe/Stockholm"
        )
        assert ts == _ts(2026, 8, 17, 7, 30)  # Monday

    def test_weekday_early_morning_is_same_day(self):
        now = _ts(2026, 8, 12, 5, 0)  # Wednesday 05:00
        ts = next_recurring_deadline_ts(("wed",), "07:30", now, "Europe/Stockholm")
        assert ts == _ts(2026, 8, 12, 7, 30)

    def test_past_todays_time_rolls_forward(self):
        now = _ts(2026, 8, 12, 8, 0)  # Wednesday 08:00, past 07:30
        ts = next_recurring_deadline_ts(("wed",), "07:30", now, "Europe/Stockholm")
        assert ts == _ts(2026, 8, 19, 7, 30)  # next Wednesday

    def test_invalid_inputs_disable_the_feature(self):
        now = _ts(2026, 8, 12, 5, 0)
        assert next_recurring_deadline_ts((), "07:30", now, "Europe/Stockholm") is None
        assert next_recurring_deadline_ts(("mon",), None, now, "Europe/Stockholm") is None
        assert next_recurring_deadline_ts(("mon",), "25:99", now, "Europe/Stockholm") is None
        assert next_recurring_deadline_ts(("blursday",), "07:30", now, "Europe/Stockholm") is None


class FakeHA:
    def __init__(self, states, attrs=None):
        self.states = states
        self.attrs = attrs or {}
        self.calls: list[tuple] = []

    async def get_state_value(self, entity):
        return self.states.get(entity)

    async def get_state(self, entity):
        if entity not in self.states and entity not in self.attrs:
            return None
        return {"state": self.states.get(entity), "attributes": self.attrs.get(entity, {})}

    async def call_service(self, domain, service, entity_id=None, data=None):
        self.calls.append((domain, service, entity_id, data))
        return True


def _cfg_dict(**over):
    base = {
        "enabled": True,
        "pv_power_entity": "sensor.pv",
        "grid_power_entity": "sensor.grid",
        "battery_power_entity": "sensor.batt",
        "battery_soc_entity": "sensor.soc",
        "price_entity": "sensor.price",
        "vacation_entity": "input_boolean.vacation",
        "priority_entity": "input_select.prio",
        "priority_orders": {
            "fmb_first": ["easee_fmb", "tesla"],
            "tesla_first": ["tesla", "easee_fmb"],
        },
        "policy": {"current_step_a": 1.0},
        "write_guard": {"min_step_a": 1.0, "min_interval_s": 90.0},
        "chargers": [
            {"id": "easee_fmb", "priority": 0, "min_current_a": 6, "max_current_a": 16,
             "phases": 1, "switch_entity": "switch.easee", "easee_device_id": "dev_easee",
             "power_entity": "sensor.easee_power", "plug_entity": "binary_sensor.easee_plug",
             "soc_entity": "input_number.fmb_soc", "floor_soc": 86, "capacity_kwh": 28},
            {"id": "tesla", "priority": 1, "min_current_a": 5, "max_current_a": 16,
             "phases": 3, "switch_entity": "switch.tesla",
             "current_entity": "number.tesla_amps",
             "power_entity": "sensor.tesla_power", "plug_entity": "binary_sensor.tesla_plug",
             "soc_entity": "sensor.tesla_soc", "floor_soc": 40, "capacity_kwh": 60,
             "recurring_deadline_days": ["Mon", "tue", "wed", "thu", "fri"],
             "recurring_deadline_time": "07:30",
             "departure_entity": "input_datetime.tesla_departure"},
        ],
    }
    base.update(over)
    return {"ev_surplus": base, "timezone": "Europe/Stockholm"}


def _states(**over):
    base = {
        "sensor.pv": "0", "sensor.grid": "0", "sensor.batt": "0",
        "sensor.soc": "50", "sensor.price": "1.0",
        "input_boolean.vacation": "off", "input_select.prio": "auto",
        "sensor.easee_power": "0", "binary_sensor.easee_plug": "on",
        "input_number.fmb_soc": "50",
        "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "on",
        "sensor.tesla_soc": "30",
    }
    base.update(over)
    return base


class TestParseNewFields:
    def test_parses_floor_recurring_selector_timezone(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        assert cfg is not None
        assert cfg.timezone == "Europe/Stockholm"
        assert cfg.priority_entity == "input_select.prio"
        assert cfg.priority_orders["tesla_first"] == ["tesla", "easee_fmb"]
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.floor_soc == 40.0
        assert tesla.recurring_deadline_days == ("mon", "tue", "wed", "thu", "fri")
        assert tesla.recurring_deadline_time == "07:30"
        fmb = next(c for c in cfg.chargers if c.id == "easee_fmb")
        assert fmb.floor_soc == 86.0 and fmb.recurring_deadline_days == ()

    def test_defaults_absent(self):
        cfg = parse_ev_surplus_config(
            {"ev_surplus": {"enabled": True, "chargers": [{"id": "x"}]}}
        )
        assert cfg is not None
        assert cfg.priority_entity is None and cfg.priority_orders == {}
        assert cfg.chargers[0].floor_soc is None
        assert cfg.chargers[0].recurring_deadline_days == ()

    def test_timezone_param_round_trips(self):
        """The caller passes the RESOLVED root timezone — executor.timezone does not
        exist in the schema, so without the param every site pinned to the fallback."""
        cfg = parse_ev_surplus_config(
            {"ev_surplus": {"enabled": True, "chargers": [{"id": "x"}]}},
            timezone="Europe/Helsinki",
        )
        assert cfg is not None and cfg.timezone == "Europe/Helsinki"

    def test_sexagesimal_time_coerced(self):
        """Unquoted YAML `recurring_deadline_time: 7:30` arrives as int 450."""
        cfg = parse_ev_surplus_config(
            {"ev_surplus": {"enabled": True, "chargers": [
                {"id": "x", "recurring_deadline_time": 450,
                 "recurring_deadline_days": ["mon"]},
            ]}},
        )
        assert cfg is not None
        assert cfg.chargers[0].recurring_deadline_time == "07:30"


@pytest.mark.asyncio
class TestRuntimeIntegration:
    async def _read_tesla(self, states, attrs=None, now_ts=None):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(states, attrs)
        tesla_cfg = next(c for c in cfg.chargers if c.id == "tesla")
        vacation = states.get("input_boolean.vacation") == "on"
        return await ctl._read_charger(
            ha, tesla_cfg, now_ts or _ts(2026, 8, 12, 5, 0), vacation
        )

    async def test_recurring_deadline_feeds_deadline_hours(self):
        st = await self._read_tesla(_states())  # Wednesday 05:00
        assert st.deadline_hours == pytest.approx(2.5, abs=0.01)  # 07:30 same day
        assert st.floor_soc_percent == 40.0

    async def test_one_off_departure_wins_when_earlier(self):
        now = _ts(2026, 8, 12, 5, 0)
        attrs = {"input_datetime.tesla_departure": {"timestamp": now + 3600}}  # 06:00
        st = await self._read_tesla(_states(), attrs)
        assert st.deadline_hours == pytest.approx(1.0, abs=0.01)

    async def test_past_departure_entity_is_inert(self):
        now = _ts(2026, 8, 12, 5, 0)
        attrs = {"input_datetime.tesla_departure": {"timestamp": now - 86400}}
        st = await self._read_tesla(_states(), attrs)
        assert st.deadline_hours == pytest.approx(2.5, abs=0.01)  # recurring still active

    async def test_vacation_clears_deadline_without_vacation_target(self):
        """The tesla block has NO vacation_target_soc — the clear must fire anyway."""
        st = await self._read_tesla(_states(**{"input_boolean.vacation": "on"}))
        assert st.deadline_hours is None

    async def test_weekend_morning_has_no_deadline(self):
        st = await self._read_tesla(_states(), now_ts=_ts(2026, 8, 15, 5, 0))  # Saturday
        # Next weekday occurrence = Monday 07:30, 50.5 h away — present but lazy,
        # far below any forcing threshold (and correctly NOT Saturday 07:30).
        assert st.deadline_hours == pytest.approx(50.5, abs=0.01)

    async def test_selector_reorders_surplus_class(self):
        """tesla_first: with surplus for only one car, the Tesla (min-on 3.45 kW) gets it."""
        states = _states(**{
            "input_select.prio": "tesla_first",
            "sensor.pv": "5000", "sensor.grid": "-4500", "sensor.soc": "95",
            "sensor.tesla_soc": "60",  # above floor 40 -> no deadline forcing, pure surplus
            "input_number.fmb_soc": "50",
        })
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(states)
        summary = await ctl.run(ha, now_ts=_ts(2026, 8, 12, 12, 0), shadow=True)
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds, f"no commands in summary: {summary}"
        assert cmds["tesla"]["on"] is True
        assert cmds["easee_fmb"]["on"] is False

    async def test_unknown_selector_option_degrades_to_configured(self):
        states = _states(**{
            "input_select.prio": "banana",
            "sensor.pv": "5000", "sensor.grid": "-4500", "sensor.soc": "95",
            "sensor.tesla_soc": "60",
            "input_number.fmb_soc": "50",
        })
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(states)
        summary = await ctl.run(ha, now_ts=_ts(2026, 8, 12, 12, 0), shadow=True)
        cmds = {c["id"]: c for c in summary.get("applied", [])}
        assert cmds["easee_fmb"]["on"] is True  # configured prio 0 wins again


class TestConfigConstantCap:
    """target_soc: the FMB's 150 km comfort cap as a plain config value (owner decision —
    no extra helper). An entity, when wired and readable, wins; config is the fallback."""

    @pytest.mark.asyncio
    async def test_config_cap_used_without_entity(self):
        raw = _cfg_dict()
        for c in raw["ev_surplus"]["chargers"]:
            if c["id"] == "easee_fmb":
                c["target_soc"] = 86
        cfg = parse_ev_surplus_config(raw)
        ctl = EVSurplusController(cfg)
        fmb_cfg = next(c for c in cfg.chargers if c.id == "easee_fmb")
        st = await ctl._read_charger(FakeHA(_states()), fmb_cfg, _ts(2026, 8, 12, 5, 0), False)
        assert st.target_soc_percent == 86.0

    @pytest.mark.asyncio
    async def test_entity_wins_over_config(self):
        raw = _cfg_dict()
        for c in raw["ev_surplus"]["chargers"]:
            if c["id"] == "easee_fmb":
                c["target_soc"] = 86
                c["target_soc_entity"] = "input_number.cap"
        cfg = parse_ev_surplus_config(raw)
        ctl = EVSurplusController(cfg)
        fmb_cfg = next(c for c in cfg.chargers if c.id == "easee_fmb")
        st = await ctl._read_charger(
            FakeHA(_states(**{"input_number.cap": "70"})), fmb_cfg, _ts(2026, 8, 12, 5, 0), False
        )
        assert st.target_soc_percent == 70.0
