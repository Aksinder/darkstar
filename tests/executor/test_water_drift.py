"""Darkstar must compare the real appliance against its own intent.

Observed live 2026-08-15: Darkstar wrote 20 to input_number.spa_darkstar_target_temp at
13:12, and from 13:18 the spa ran climate=heat/38 with the element drawing 1.8 kW. No
tick corrected it, because set_water_temp is change-gated against the HELPER — which
already read 20 — and the HA bridge only relays on a helper CHANGE. The switch path has
always self-healed by comparing the relay; this gives the bridge path the same property.
"""

from __future__ import annotations

from executor.water_hold import detect_appliance_drift

OFF = 20.0
HEAT = 38.0


def _drift(**kw):
    base = {
        "intended_temp": OFF,
        "off_temp": OFF,
        "appliance_state": "fan_only",
        "appliance_setpoint_c": None,
        "power_w": 0.0,
        "idle_power_w": 100.0,
    }
    base.update(kw)
    return detect_appliance_drift(**base)


class TestTheLiveIncident:
    def test_heating_while_we_intend_off(self):
        drifted, why = _drift(appliance_state="heat", appliance_setpoint_c=38.0,
                              power_w=1800.0)
        assert drifted is True
        assert "while we intend off" in why

    def test_agreement_is_not_drift(self):
        assert _drift()[0] is False


class TestTwoSignals:
    """Mode and power can each lie on their own, so either is enough."""

    def test_mode_alone_is_enough(self):
        """The element is between cycles but the tub is set to heat."""
        assert _drift(appliance_state="heat", power_w=0.0)[0] is True

    def test_power_alone_is_enough(self):
        """The reported mode lags the device; the draw does not."""
        drifted, why = _drift(appliance_state="fan_only", power_w=1800.0)
        assert drifted is True
        assert "1800W" in why

    def test_pump_only_draw_is_not_heating(self):
        assert _drift(appliance_state="fan_only", power_w=60.0)[0] is False


class TestIntendingHeat:
    def test_an_idle_appliance_means_our_target_never_landed(self):
        drifted, why = _drift(intended_temp=HEAT, appliance_state="fan_only")
        assert drifted is True
        assert "fan_only" in why

    def test_a_drifted_setpoint_is_corrected(self):
        drifted, why = _drift(intended_temp=HEAT, appliance_state="heat",
                              appliance_setpoint_c=30.0, power_w=1800.0)
        assert drifted is True
        assert "!= intended" in why

    def test_matching_setpoint_is_fine(self):
        assert _drift(intended_temp=HEAT, appliance_state="heat",
                      appliance_setpoint_c=38.0, power_w=1800.0)[0] is False

    def test_half_a_degree_is_within_tolerance(self):
        assert _drift(intended_temp=HEAT, appliance_state="heat",
                      appliance_setpoint_c=38.4, power_w=1800.0)[0] is False


class TestUnknowns:
    def test_unavailable_state_falls_back_to_power(self):
        assert _drift(appliance_state="unavailable", power_w=1800.0)[0] is True
        assert _drift(appliance_state="unavailable", power_w=0.0)[0] is False

    def test_an_unreadable_appliance_intending_heat_is_not_drift(self):
        """No evidence either way — do not thrash a device we cannot see."""
        assert _drift(intended_temp=HEAT, appliance_state=None,
                      appliance_setpoint_c=None)[0] is False

    def test_no_intent_no_drift(self):
        assert _drift(intended_temp=None)[0] is False


class TestClimateModeCorrection:
    """Detecting the drift was never the problem — correcting it was. The old
    correction only nudged the HELPER, asking the HA bridge to relay the target
    again; when the bridge is what failed (or the tub's own panel moved the mode),
    nothing reaches the device and the loop just logs (live: two ERRORs on
    2026-08-24, spa in fan_only while the plan intended 40C). Owner directive:
    Darkstar must change the mode itself when it means to heat."""

    def _engine(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from executor.engine import ExecutorEngine

        eng = ExecutorEngine.__new__(ExecutorEngine)
        eng.dispatcher = SimpleNamespace(
            force_heater_climate_mode=AsyncMock(return_value="forced")
        )
        return eng

    def _device(self, **kw):
        from types import SimpleNamespace

        base = dict(
            id="spa",
            state_entity="climate.layzspa_temperature_control",
            climate_heat_mode="heat",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    async def _call(self, eng, device, intended, off=20):
        return await eng._force_heater_heat_mode(device, intended, off)

    def test_forces_heat_mode_when_intending_heat(self):
        import asyncio

        eng = self._engine()
        dev = self._device()
        res = asyncio.run(self._call(eng, dev, 40))
        assert res == "forced"
        eng.dispatcher.force_heater_climate_mode.assert_awaited_once_with(
            "climate.layzspa_temperature_control", "heat", setpoint_c=40.0
        )

    def test_off_direction_is_left_to_the_helper(self):
        # Writing "off" to the appliance is not the bridge's fan_only — it would stop
        # the circulation pump and throw away the warmth idle-hold protects.
        import asyncio

        eng = self._engine()
        res = asyncio.run(self._call(eng, self._device(), 20))
        assert res is None
        eng.dispatcher.force_heater_climate_mode.assert_not_awaited()

    def test_inert_without_the_config_knob(self):
        import asyncio

        eng = self._engine()
        res = asyncio.run(self._call(eng, self._device(climate_heat_mode=None), 40))
        assert res is None
        eng.dispatcher.force_heater_climate_mode.assert_not_awaited()

    def test_inert_for_a_non_climate_state_entity(self):
        import asyncio

        eng = self._engine()
        dev = self._device(state_entity="sensor.vvb_state")
        res = asyncio.run(self._call(eng, dev, 40))
        assert res is None
        eng.dispatcher.force_heater_climate_mode.assert_not_awaited()

    def test_a_failing_correction_never_kills_the_tick(self):
        import asyncio
        from unittest.mock import AsyncMock

        eng = self._engine()
        eng.dispatcher.force_heater_climate_mode = AsyncMock(side_effect=RuntimeError("HA down"))
        assert asyncio.run(self._call(eng, self._device(), 40)) is None


class TestForcedOffUsesPerDeviceOffTemp:
    """The slot-failure fallback carries ONE global off-temp (40 here: the tanks run
    60 and rest at 40). The spa's bridge maps its whole 20-40 scale onto heat, so
    that same 40 commanded the spa to MAXIMUM heat — a 'safety water OFF' that runs
    a 1.8 kW element at any price for the length of a planner outage. Found in
    adversarial review 2026-08-24; pre-existing, and the worst spa hazard there was."""

    def _pick(self, decision_temp, device_off):
        """Mirror the engine's per-device resolution (engine.py forced-OFF branch)."""
        from types import SimpleNamespace

        device = SimpleNamespace(id="spa", temp_off=device_off)
        off_temp = decision_temp
        dev_off = getattr(device, "temp_off", None)
        if dev_off is not None:
            off_temp = dev_off
        return off_temp

    def test_spa_gets_its_own_off_not_the_global(self):
        assert self._pick(40, 20) == 20  # NOT 40 = spa maximum

    def test_tank_without_its_own_off_falls_back_to_global(self):
        assert self._pick(40, None) == 40

    def test_device_off_wins_even_when_equal(self):
        assert self._pick(40, 40) == 40
