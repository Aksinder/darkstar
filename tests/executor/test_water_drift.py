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
