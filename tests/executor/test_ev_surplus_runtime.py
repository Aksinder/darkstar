"""Tests for the EV surplus runtime (config parse + live read + actuation)."""

import pytest

from executor.ev_surplus_runtime import (
    EVSurplusController,
    parse_ev_surplus_config,
)


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
        "policy": {"cheap_grid_price_sek": 0.3, "current_step_a": 2.0},
        "write_guard": {"min_step_a": 2.0, "min_interval_s": 90.0},
        "chargers": [
            {"id": "easee", "priority": 0, "min_current_a": 6, "max_current_a": 16, "phases": 1,
             "switch_entity": "switch.easee", "easee_device_id": "dev_easee",
             "power_entity": "sensor.easee_power", "plug_entity": "binary_sensor.easee_plug"},
            {"id": "tesla", "priority": 1, "min_current_a": 5, "max_current_a": 16, "phases": 3,
             "switch_entity": "switch.tesla", "current_entity": "number.tesla_amps",
             "power_entity": "sensor.tesla_power", "plug_entity": "binary_sensor.tesla_plug",
             "override_entity": "input_select.tesla_mode"},
        ],
    }
    base.update(over)
    return {"ev_surplus": base}


def _states(**over):
    base = {
        "sensor.pv": "12000", "sensor.grid": "-10000", "sensor.batt": "0",
        "sensor.soc": "90", "sensor.price": "1.0",
        "sensor.easee_power": "0", "binary_sensor.easee_plug": "on",
        "sensor.tesla_power": "0", "binary_sensor.tesla_plug": "on",
        "input_select.tesla_mode": "auto",
    }
    base.update(over)
    return base


class TestParse:
    def test_absent_returns_none(self):
        assert parse_ev_surplus_config({}) is None

    def test_parses_chargers_and_policy(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        assert cfg is not None and cfg.enabled
        assert [c.id for c in cfg.chargers] == ["easee", "tesla"]
        easee = next(c for c in cfg.chargers if c.id == "easee")
        assert easee.controllable and easee.easee_device_id == "dev_easee" and easee.phases == 1
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.current_entity == "number.tesla_amps" and tesla.min_current_a == 5.0
        assert cfg.policy.cheap_grid_price_sek == 0.3

    def test_parses_departure_and_vacation_fields(self):
        cfg = parse_ev_surplus_config(
            _cfg_dict(
                vacation_entity="input_boolean.vacation_mode",
                chargers=[
                    {"id": "easee", "easee_device_id": "dev_easee",
                     "soc_entity": "input_number.fmb_soc", "capacity_kwh": 28,
                     "vacation_target_soc": 15},
                    {"id": "tesla", "current_entity": "number.tesla_amps",
                     "soc_entity": "sensor.tesla_soc",
                     "target_soc_entity": "input_number.tesla_target",
                     "departure_entity": "input_datetime.tesla_departure",
                     "capacity_kwh": 60, "charge_efficiency": 0.92},
                ],
            )
        )
        assert cfg is not None and cfg.vacation_entity == "input_boolean.vacation_mode"
        easee = next(c for c in cfg.chargers if c.id == "easee")
        assert easee.soc_entity == "input_number.fmb_soc" and easee.capacity_kwh == 28.0
        assert easee.vacation_target_soc == 15.0
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.departure_entity == "input_datetime.tesla_departure"
        assert tesla.target_soc_entity == "input_number.tesla_target"
        assert tesla.capacity_kwh == 60.0 and tesla.charge_efficiency == 0.92


class TestActuation:
    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self):
        cfg = parse_ev_surplus_config(_cfg_dict(enabled=False))
        ha = FakeHA(_states())
        out = await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        assert out == {"enabled": False}
        assert ha.calls == []

    @pytest.mark.asyncio
    async def test_surplus_actuates_both_paths(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ha = FakeHA(_states())
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        svc = {(d, s) for d, s, _e, _data in ha.calls}
        # Tesla via number.set_value, Easee via its dynamic-limit service, switches on.
        assert ("number", "set_value") in svc
        assert ("easee", "set_charger_dynamic_limit") in svc
        assert ("switch", "turn_on") in svc
        # NEVER the flash-wearing limits.
        assert ("easee", "set_charger_max_limit") not in svc
        assert ("easee", "set_circuit_max_limit") not in svc

    @pytest.mark.asyncio
    async def test_write_guard_suppresses_rapid_rewrite(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctrl = EVSurplusController(cfg)
        ha = FakeHA(_states())
        await ctrl.run(ha, now_ts=1000.0)
        n_current_writes = sum(1 for d, s, *_ in ha.calls if s in ("set_value", "set_charger_dynamic_limit"))
        assert n_current_writes >= 1
        # Second cycle 10 s later, near-identical conditions => no new current write (interval).
        ha.calls.clear()
        await ctrl.run(ha, now_ts=1010.0)
        n2 = sum(1 for d, s, *_ in ha.calls if s in ("set_value", "set_charger_dynamic_limit"))
        assert n2 == 0

    @pytest.mark.asyncio
    async def test_force_off_override_stops_tesla(self):
        ha = FakeHA(_states(**{"input_select.tesla_mode": "force_off"}))
        cfg = parse_ev_surplus_config(_cfg_dict())
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        # Tesla switch should be turned off; no current write to the Tesla number.
        assert ("switch", "turn_off", "switch.tesla", None) in ha.calls
        assert not any(s == "set_value" for _d, s, *_ in ha.calls)

    @pytest.mark.asyncio
    async def test_shadow_mode_no_calls(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ha = FakeHA(_states())
        await EVSurplusController(cfg).run(ha, now_ts=1000.0, shadow=True)
        assert ha.calls == []


class TestDepartureVacation:
    @pytest.mark.asyncio
    async def test_vacation_caps_fmb_off(self):
        # Vacation on + FMB above its 15% vacation target => FMB stops (switch off), even with
        # 10 kW of export available.
        cfg = parse_ev_surplus_config(
            _cfg_dict(
                vacation_entity="input_boolean.vacation_mode",
                chargers=[
                    {"id": "easee", "priority": 0, "phases": 1, "switch_entity": "switch.easee",
                     "easee_device_id": "dev_easee", "power_entity": "sensor.easee_power",
                     "plug_entity": "binary_sensor.easee_plug",
                     "soc_entity": "input_number.fmb_soc", "capacity_kwh": 28,
                     "vacation_target_soc": 15},
                ],
            )
        )
        ha = FakeHA(_states(**{
            "input_boolean.vacation_mode": "on", "input_number.fmb_soc": "50",
        }))
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        assert ("switch", "turn_off", "switch.easee", None) in ha.calls
        assert not any(s == "set_charger_dynamic_limit" for _d, s, *_ in ha.calls)

    @pytest.mark.asyncio
    async def test_switchless_easee_pauses_via_dynamic_limit_zero(self):
        # An Easee with NO switch_entity, capped off => dynamic limit 0 IS the stop.
        cfg = parse_ev_surplus_config(
            _cfg_dict(
                chargers=[
                    {"id": "easee", "priority": 0, "phases": 1, "easee_device_id": "dev_easee",
                     "power_entity": "sensor.easee_power", "plug_entity": "binary_sensor.easee_plug",
                     "soc_entity": "input_number.fmb_soc", "target_soc_entity": "input_number.fmb_target"},
                ],
            )
        )
        ha = FakeHA(_states(**{"input_number.fmb_soc": "90", "input_number.fmb_target": "15"}))
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        stops = [c for c in ha.calls if c[1] == "set_charger_dynamic_limit"]
        assert stops and stops[0][3]["current"] == 0

    @pytest.mark.asyncio
    async def test_vacation_clears_deadline_so_fmb_is_solar_only(self):
        # FMB has BOTH a vacation target (15%) and a future departure entity. In vacation the
        # departure deadline must be ignored (solar-only): with no surplus the FMB does NOT
        # grid-force toward 15% even though SoC (12%) is below target.
        cfg = parse_ev_surplus_config(
            _cfg_dict(
                vacation_entity="input_boolean.vacation_mode",
                chargers=[
                    {"id": "easee", "priority": 0, "phases": 1, "easee_device_id": "dev_easee",
                     "power_entity": "sensor.easee_power", "plug_entity": "binary_sensor.easee_plug",
                     "soc_entity": "input_number.fmb_soc", "capacity_kwh": 28,
                     "vacation_target_soc": 15,
                     "departure_entity": "input_datetime.fmb_departure"},
                ],
            )
        )
        ha = FakeHA(
            _states(**{
                "sensor.grid": "3000", "sensor.pv": "0",
                "input_boolean.vacation_mode": "on", "input_number.fmb_soc": "12",
            }),
            attrs={"input_datetime.fmb_departure": {"timestamp": 1000.0 + 3600.0}},
        )
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        # No grid-forced dynamic-limit write (would only happen if the deadline floor fired).
        assert not any(
            c[1] == "set_charger_dynamic_limit" and (c[3] or {}).get("current", 0) > 0
            for c in ha.calls
        )

    @pytest.mark.asyncio
    async def test_departure_behind_forces_tesla_grid(self):
        # Tesla 10%->80% of 60 kWh, departure in 2 h, no surplus (importing) => grid-forced
        # toward max via number.set_value.
        cfg = parse_ev_surplus_config(
            _cfg_dict(
                chargers=[
                    {"id": "tesla", "priority": 0, "phases": 3, "min_current_a": 5,
                     "max_current_a": 16, "switch_entity": "switch.tesla",
                     "current_entity": "number.tesla_amps", "power_entity": "sensor.tesla_power",
                     "plug_entity": "binary_sensor.tesla_plug",
                     "soc_entity": "sensor.tesla_soc",
                     "target_soc_entity": "input_number.tesla_target",
                     "departure_entity": "input_datetime.tesla_departure", "capacity_kwh": 60},
                ],
            )
        )
        ha = FakeHA(
            _states(**{
                "sensor.grid": "5000", "sensor.pv": "0",
                "sensor.tesla_soc": "10", "input_number.tesla_target": "80",
            }),
            attrs={"input_datetime.tesla_departure": {"timestamp": 1000.0 + 7200.0}},
        )
        await EVSurplusController(cfg).run(ha, now_ts=1000.0)
        sets = [c for c in ha.calls if c[1] == "set_value"]
        assert sets and sets[0][3]["value"] >= 10.0  # forced well above the 5 A min


class TestOverrideTimeout:
    """A forgotten force must expire. The heaters have had this from the start; the
    EVs did not, and the failure mode is worse here: a forgotten Tesla force_on
    grid-charges at 16 A three-phase to the car's own limit, at any price, forever.
    """

    def _cfg(self, timeout_min):
        raw = _cfg_dict()
        for c in raw["ev_surplus"]["chargers"]:
            if c["id"] == "tesla":
                c["override_timeout_minutes"] = timeout_min
        return parse_ev_surplus_config(raw)

    def _no_surplus_forced_on(self):
        """No surplus at a dear price: only a live force_on starts the Tesla."""
        return FakeHA(_states(**{
            "sensor.pv": "0", "sensor.grid": "2000", "sensor.price": "2.5",
            "input_select.tesla_mode": "force_on",
        }))

    @staticmethod
    def _tesla_started(ha):
        return ("switch", "turn_on", "switch.tesla", None) in ha.calls

    @pytest.mark.asyncio
    async def test_parses_the_timeout(self):
        cfg = self._cfg(90.0)
        tesla = next(c for c in cfg.chargers if c.id == "tesla")
        assert tesla.override_timeout_minutes == 90.0

    @pytest.mark.asyncio
    async def test_force_expires_back_to_auto(self):
        ctrl = EVSurplusController(self._cfg(30.0))
        ha = self._no_surplus_forced_on()
        await ctrl.run(ha, now_ts=1000.0)
        assert self._tesla_started(ha)
        # 31 minutes later the same helper still says force_on — but the clock ran out.
        ha2 = self._no_surplus_forced_on()
        await ctrl.run(ha2, now_ts=1000.0 + 31 * 60.0)
        assert not self._tesla_started(ha2)

    @pytest.mark.asyncio
    async def test_inside_the_window_the_force_holds(self):
        """The write guard remembers the car is already ON, so 'holds' shows as
        the ABSENCE of a stop — exactly what a live force feels like."""
        ctrl = EVSurplusController(self._cfg(30.0))
        await ctrl.run(self._no_surplus_forced_on(), now_ts=1000.0)
        ha2 = self._no_surplus_forced_on()
        await ctrl.run(ha2, now_ts=1000.0 + 29 * 60.0)
        assert ("switch", "turn_off", "switch.tesla", None) not in ha2.calls

    @pytest.mark.asyncio
    async def test_timeout_zero_never_expires(self):
        """0 is the explicit 'never' — the pre-existing behaviour, not a trap."""
        ctrl = EVSurplusController(self._cfg(0.0))
        await ctrl.run(self._no_surplus_forced_on(), now_ts=1000.0)
        ha2 = self._no_surplus_forced_on()
        await ctrl.run(ha2, now_ts=1000.0 + 48 * 3600.0)
        assert ("switch", "turn_off", "switch.tesla", None) not in ha2.calls

    @pytest.mark.asyncio
    async def test_flipping_back_to_auto_resets_the_clock(self):
        """auto -> force_on again starts a FRESH window, not the old one's tail."""
        ctrl = EVSurplusController(self._cfg(30.0))
        await ctrl.run(self._no_surplus_forced_on(), now_ts=1000.0)
        # Human returns it to auto...
        ha_auto = FakeHA(_states(**{
            "sensor.pv": "0", "sensor.grid": "2000", "sensor.price": "2.5",
            "input_select.tesla_mode": "auto",
        }))
        await ctrl.run(ha_auto, now_ts=1000.0 + 20 * 60.0)
        # ...and forces again 25 min after the FIRST force: fresh 30-min window.
        ha3 = self._no_surplus_forced_on()
        await ctrl.run(ha3, now_ts=1000.0 + 25 * 60.0)
        assert self._tesla_started(ha3)

    @pytest.mark.asyncio
    async def test_switching_force_direction_resets_the_clock(self):
        """force_off -> force_on is a NEW decision; it gets its own window."""
        ctrl = EVSurplusController(self._cfg(30.0))
        ha_off = FakeHA(_states(**{"input_select.tesla_mode": "force_off"}))
        await ctrl.run(ha_off, now_ts=1000.0)
        ha_on = self._no_surplus_forced_on()
        await ctrl.run(ha_on, now_ts=1000.0 + 25 * 60.0)
        assert self._tesla_started(ha_on)
