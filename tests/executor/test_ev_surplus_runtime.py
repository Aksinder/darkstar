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


class TestEvPriorityCapRuntime:
    """The runtime side: parse, compute each tick, hysteresis, and None on every
    tick that did not compute — a stale cap must never outlive its tick."""

    def _raw(self, enabled=True, hysteresis=200.0):
        raw = _cfg_dict()
        raw["ev_surplus"]["policy"]["ev_priority_battery_cap"] = {
            "enabled": enabled, "hysteresis_w": hysteresis,
        }
        # A reserve that is clearly inactive: lots of sun, battery well below yield.
        raw["ev_surplus"]["policy"].update({
            "battery_yield_soc": 95.0, "battery_capacity_kwh": 16.0,
            "battery_fill_margin_kwh": 3.0,
        })
        raw["ev_surplus"]["remaining_solar_entity"] = "sensor.remaining"
        return raw

    @pytest.mark.asyncio
    async def test_parses_the_flag(self):
        cfg = parse_ev_surplus_config(self._raw())
        assert cfg.policy.ev_priority_battery_cap_enabled is True
        assert cfg.policy.ev_priority_cap_hysteresis_w == 200.0
        assert parse_ev_surplus_config(_cfg_dict()).policy.ev_priority_battery_cap_enabled is False

    @pytest.mark.asyncio
    async def test_a_computed_tick_exposes_a_cap(self):
        ctrl = EVSurplusController(parse_ev_surplus_config(self._raw()))
        # Battery soaking 7 kW, Tesla at 3 kW, grid flat: the plan-scenario numbers.
        ha = FakeHA(_states(**{
            "sensor.grid": "0", "sensor.batt": "7000", "sensor.soc": "80",
            "sensor.remaining": "30", "sensor.tesla_power": "3000",
            "binary_sensor.easee_plug": "off",
        }))
        await ctrl.run(ha, now_ts=1000.0)
        cap = ctrl.ev_priority_battery_cap_w(1000.0)
        assert cap is not None and 0.0 < cap < 7000.0

    @pytest.mark.asyncio
    async def test_disabled_exposes_nothing(self):
        ctrl = EVSurplusController(parse_ev_surplus_config(self._raw(enabled=False)))
        ha = FakeHA(_states(**{"sensor.batt": "7000", "sensor.soc": "80",
                               "sensor.remaining": "30", "sensor.tesla_power": "3000"}))
        await ctrl.run(ha, now_ts=1000.0)
        assert ctrl.ev_priority_battery_cap_w(1000.0) is None

    @pytest.mark.asyncio
    async def test_a_skipped_tick_clears_the_cap(self):
        """Core sensors dark => the tick is skipped => no cap survives from before."""
        ctrl = EVSurplusController(parse_ev_surplus_config(self._raw()))
        ha = FakeHA(_states(**{"sensor.batt": "7000", "sensor.soc": "80",
                               "sensor.remaining": "30", "sensor.tesla_power": "3000"}))
        await ctrl.run(ha, now_ts=1000.0)
        assert ctrl.ev_priority_battery_cap_w(1000.0) is not None
        dark = FakeHA({k: v for k, v in ha.states.items() if k != "sensor.grid"})
        await ctrl.run(dark, now_ts=1060.0)
        assert ctrl.ev_priority_battery_cap_w(1060.0) is None

    @pytest.mark.asyncio
    async def test_small_drift_does_not_move_the_cap(self):
        """No write threshold exists on this register downstream; the hysteresis
        here is what keeps a drifting cap from writing the inverter every tick."""
        ctrl = EVSurplusController(parse_ev_surplus_config(self._raw(hysteresis=200.0)))
        ctrl._update_ev_priority_cap(3500.0)
        ctrl._update_ev_priority_cap(3420.0)   # 80 W drift: held
        assert ctrl.last_ev_priority_cap_w == 3500.0
        ctrl._update_ev_priority_cap(3200.0)   # 300 W: moves
        assert ctrl.last_ev_priority_cap_w == 3200.0
        ctrl._update_ev_priority_cap(None)     # release is never delayed
        assert ctrl.last_ev_priority_cap_w is None
        ctrl._update_ev_priority_cap(50.0)     # appearance is never delayed
        assert ctrl.last_ev_priority_cap_w == 50.0


class TestPriceSourceResolution:
    """Internal spot series first; price_entity only as fallback; 999 when both dark."""

    @pytest.mark.asyncio
    async def test_internal_price_wins_over_sensor(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"sensor.price": "0"}))  # dead-sensor signature
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=1.23)
        assert out["price_sek"] == 1.23
        assert out["price_source"] == "internal"

    @pytest.mark.asyncio
    async def test_sensor_fallback_when_internal_unavailable(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"sensor.price": "0.42"}))
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
        assert out["price_sek"] == 0.42
        assert out["price_source"] == "entity"

    @pytest.mark.asyncio
    async def test_fail_expensive_when_both_dark(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"sensor.price": "unavailable"}))
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
        assert out["price_sek"] == 999.0
        assert out["price_source"] == "default"

    @pytest.mark.asyncio
    async def test_fallback_warning_rate_limited(self, caplog):
        import logging
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states())
        with caplog.at_level(logging.WARNING, logger="darkstar.ev_surplus"):
            await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
            await ctl.run(ha, 1060.0, shadow=True, internal_spot_price_sek=None)
        warns = [r for r in caplog.records if "falling back to" in r.message]
        assert len(warns) == 1
        # Internal recovers => warn state resets; next outage warns again.
        await ctl.run(ha, 1120.0, shadow=True, internal_spot_price_sek=0.5)
        with caplog.at_level(logging.WARNING, logger="darkstar.ev_surplus"):
            await ctl.run(ha, 1180.0, shadow=True, internal_spot_price_sek=None)
        warns = [r for r in caplog.records if "falling back to" in r.message]
        assert len(warns) == 2

    @pytest.mark.asyncio
    async def test_sensor_zero_is_not_trusted_in_fallback(self):
        # The dead-template signature: float(0) on 'unavailable'. Indistinguishable
        # from genuine zero spot, so fallback mode refuses it (999, tiers closed).
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"sensor.price": "0"}))
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
        assert out["price_sek"] == 999.0
        assert out["price_source"] == "default"

    @pytest.mark.asyncio
    async def test_sensor_negative_is_not_trusted_in_fallback(self):
        # Only the internal series may authorize the negative-price tiers.
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states(**{"sensor.price": "-0.10"}))
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
        assert out["price_sek"] == 999.0
        assert out["price_source"] == "default"

    @pytest.mark.asyncio
    async def test_internal_negative_is_trusted(self):
        cfg = parse_ev_surplus_config(_cfg_dict())
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states())
        out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=-0.10)
        assert out["price_sek"] == -0.10
        assert out["price_source"] == "internal"

    @pytest.mark.asyncio
    async def test_warns_even_without_fallback_entity(self, caplog):
        import logging
        cfg = parse_ev_surplus_config(_cfg_dict(price_entity=""))
        ctl = EVSurplusController(cfg)
        ha = FakeHA(_states())
        with caplog.at_level(logging.WARNING, logger="darkstar.ev_surplus"):
            out = await ctl.run(ha, 1000.0, shadow=True, internal_spot_price_sek=None)
        assert out["price_source"] == "default"
        assert any("paid tiers closed" in r.message for r in caplog.records)


class TestDepartureTargetFloor:
    """A ONE-OFF departure raises the guarantee floor to the departure target.

    Live 2026-08-27: departure 07:25 set at 22:53, car stopped at 41 % because
    the deadline floor targeted floor_soc 40 — correct for the weekday
    recurrence, useless for an explicit trip. The recurrence keeps the low
    band (that distinction fixed the old charge-to-90-every-night bug).
    """

    def _charger(self, **over):
        base = {
            "id": "tesla", "priority": 1, "phases": 3,
            "switch_entity": "switch.tesla", "current_entity": "number.tesla_amps",
            "power_entity": "sensor.tesla_power", "plug_entity": "binary_sensor.tesla_plug",
            "soc_entity": "sensor.tesla_soc", "capacity_kwh": 60,
            "floor_soc": 40,
            "departure_entity": "input_datetime.tesla_departure",
            "departure_target_entity": "input_number.tesla_departure_soc",
            "min_current_a": 5, "max_current_a": 16,
        }
        base.update(over)
        return base

    async def _run(self, ha, cfg_over=None):
        cfg = parse_ev_surplus_config(_cfg_dict(chargers=[self._charger()], **(cfg_over or {})))
        ctrl = EVSurplusController(cfg)
        # No surplus: the only reason to charge is a deadline floor.
        await ctrl.run(ha, now_ts=1_000_000.0)
        return ha

    @pytest.mark.asyncio
    async def test_one_off_departure_forces_toward_target(self):
        # SoC 41, floor 40 (satisfied) BUT departure in 30 min with target 80:
        # the floor must retarget to 80 and grid-force despite zero surplus.
        ha = FakeHA(
            _states(**{
                "sensor.pv": "0", "sensor.grid": "0",
                "sensor.tesla_soc": "41",
                "input_number.tesla_departure_soc": "80",
            }),
            attrs={"input_datetime.tesla_departure": {"timestamp": 1_000_000.0 + 1800}},
        )
        await self._run(ha)
        on_calls = [c for c in ha.calls if c[:2] == ("switch", "turn_on")]
        assert on_calls, f"expected grid forcing toward the 80% departure target, calls={ha.calls}"

    @pytest.mark.asyncio
    async def test_satisfied_floor_without_departure_stays_off(self):
        # Same car, no departure set: floor 40 already met at SoC 41 => no forcing.
        ha = FakeHA(
            _states(**{
                "sensor.pv": "0", "sensor.grid": "0",
                "sensor.tesla_soc": "41",
                "input_number.tesla_departure_soc": "80",
            }),
        )
        await self._run(ha)
        on_calls = [c for c in ha.calls if c[:2] == ("switch", "turn_on")]
        assert not on_calls, f"no departure => the 40 floor is met, calls={ha.calls}"

    @pytest.mark.asyncio
    async def test_recurring_deadline_keeps_the_low_floor(self):
        # Weekday recurrence active (deadline in ~30 min via recurrence), SoC 41:
        # floor stays 40 (met) — the recurrence must NOT charge to the trip target.
        ha = FakeHA(
            _states(**{
                "sensor.pv": "0", "sensor.grid": "0",
                "sensor.tesla_soc": "41",
                "input_number.tesla_departure_soc": "80",
            }),
        )
        cfg = parse_ev_surplus_config(_cfg_dict(chargers=[self._charger(
            recurring_deadline_days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            recurring_deadline_time="23:59",
        )]))
        await EVSurplusController(cfg).run(ha, now_ts=1_000_000.0)
        on_calls = [c for c in ha.calls if c[:2] == ("switch", "turn_on")]
        assert not on_calls, f"recurrence keeps floor 40 (met at 41), calls={ha.calls}"

    @pytest.mark.asyncio
    async def test_stale_departure_is_inert(self):
        # A PAST departure timestamp must not raise the floor.
        ha = FakeHA(
            _states(**{
                "sensor.pv": "0", "sensor.grid": "0",
                "sensor.tesla_soc": "41",
                "input_number.tesla_departure_soc": "80",
            }),
            attrs={"input_datetime.tesla_departure": {"timestamp": 1_000_000.0 - 3600}},
        )
        await self._run(ha)
        on_calls = [c for c in ha.calls if c[:2] == ("switch", "turn_on")]
        assert not on_calls, f"stale departure must be inert, calls={ha.calls}"

    @pytest.mark.asyncio
    async def test_departure_later_than_recurrence_still_forces_toward_target(self):
        # Review round 1: a departure five minutes AFTER the weekday recurrence
        # lost the min-race and got no target guarantee — silently reproducing
        # the 2026-08-27 incident. The binding pair is now the one demanding
        # the higher charge rate, not the earlier one.
        ha = FakeHA(
            _states(**{
                "sensor.pv": "0", "sensor.grid": "0",
                "sensor.tesla_soc": "41",
                "input_number.tesla_departure_soc": "80",
            }),
            # Departure in 6h (39% gap => ~4.4 kW required > 3.45 kW charger min,
            # so the floor must force); recurrence fires ~5h from now (earlier!).
            attrs={"input_datetime.tesla_departure": {"timestamp": 1_000_000.0 + 6 * 3600}},
        )
        import datetime as _dt

        import pytz as _pytz
        tz = _pytz.timezone("Europe/Stockholm")
        rec_dt = _dt.datetime.fromtimestamp(1_000_000.0 + 5 * 3600, tz)
        cfg = parse_ev_surplus_config(_cfg_dict(chargers=[self._charger(
            recurring_deadline_days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            recurring_deadline_time=f"{rec_dt.hour:02d}:{rec_dt.minute:02d}",
        )]))
        await EVSurplusController(cfg).run(ha, now_ts=1_000_000.0)
        on_calls = [c for c in ha.calls if c[:2] == ("switch", "turn_on")]
        # Rate to 80 in 6h (6.5 %/h) beats rate to 40 in 5h (0 — already met):
        # the departure pair is binding and must force despite losing the min-race.
        assert on_calls, f"later-than-recurrence departure must still force, calls={ha.calls}"
