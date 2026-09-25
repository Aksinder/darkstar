"""EV source isolation must reach the inverter when the car starts on its own.

2026-09-25 05:02 the Tesla began charging outside the plan. The engine saw it
("EV charging detected: 5.00 kW (not in schedule) - Source isolation active")
and zeroed discharge_kw — which changes nothing on a W-profile: the commanded
discharge limit is always the pack maximum, and idle was only chosen when the
PLAN had an EV slot or SoC sat at target. The Sungrow stayed in self-consumption
at 9.5 kW and fed the car from the battery for 13 minutes (25 -> 7.7 %) until
the SoC floor forced idle. The night before, the same.

The isolated slot also used to be a hand-copied constructor that dropped every
field it did not list — water_heater_plans among them — so while a car charged,
every tank read as "planned off".

Rule: the engine marks the slot ev_isolation=True (via replace(), keeping every
other field) and the controller forces idle for any discharge-capable mode.
"""

from __future__ import annotations

from executor.config import (
    ControllerConfig,
    InverterConfig,
    WaterHeaterDeviceConfig,
    WaterHeaterGlobalConfig,
)
from executor.controller import make_decision
from executor.engine import isolate_slot_for_ev
from executor.override import SlotPlan, SystemState

ABOVE_TARGET = SystemState(current_soc_percent=65.0, min_soc_percent=10.0)


def _decide(slot, state=ABOVE_TARGET, **kw):
    return make_decision(slot, state, None, ControllerConfig(), InverterConfig(), **kw)


class TestControllerForcesIdleUnderIsolation:
    def test_unplanned_car_with_soc_above_target_goes_idle(self):
        slot = SlotPlan(discharge_kw=0.0, ev_charging_kw=0.0, soc_target=60, ev_isolation=True)
        d = _decide(slot)
        assert d.mode_intent == "idle"
        assert "EV isolation" in d.reason

    def test_same_slot_without_isolation_is_self_consumption(self):
        """The control: nothing else in the slot asks for idle."""
        slot = SlotPlan(discharge_kw=0.0, ev_charging_kw=0.0, soc_target=60)
        assert _decide(slot).mode_intent == "self_consumption"

    def test_pv_surplus_exception_does_not_reopen_the_battery(self):
        """At/below target with PV > load the controller prefers self_consumption
        so transients are covered — not while a car is on the line."""
        slot = SlotPlan(discharge_kw=0.0, pv_kw=5.0, load_kw=1.0, soc_target=70, ev_isolation=True)
        assert _decide(slot).mode_intent == "idle"

    def test_export_intent_is_downgraded_to_idle(self):
        """Belt and braces: the engine zeroes discharge_kw so export never
        triggers, but a slot that somehow still says export must not discharge."""
        slot = SlotPlan(discharge_kw=3.0, export_kw=3.0, soc_target=20, ev_isolation=True)
        assert _decide(slot).mode_intent == "idle"

    def test_grid_charge_is_left_alone(self):
        """Forced charge cannot discharge; the car and the pack both draw from the grid."""
        slot = SlotPlan(charge_kw=5.0, pv_kw=0.0, load_kw=1.0, soc_target=90, ev_isolation=True)
        assert _decide(slot).mode_intent == "charge"

    def test_planned_ev_slot_still_goes_idle(self):
        """The pre-existing path (plan says the car charges) is unchanged."""
        slot = SlotPlan(discharge_kw=0.0, ev_charging_kw=4.0, soc_target=60)
        assert _decide(slot).mode_intent == "idle"


class TestIsolatedSlotKeepsThePlan:
    def test_replace_keeps_every_field_and_marks_isolation(self):
        original = SlotPlan(
            charge_kw=0.0,
            discharge_kw=3.0,
            export_kw=0.0,
            load_kw=1.2,
            pv_kw=0.0,
            water_kw=3.4,
            ev_charging_kw=0.0,
            soc_target=60,
            soc_projected=58,
            ev_charger_plans={"tesla": 0.0, "easee_fmb": 0.0},
            water_heater_plans={"main_tank": 3.4, "spa": 1.8},
            water_heating_boost={"main_tank": True},
            custom_entity_active=True,
            sinks={"villavagn_ac": True},
            export_price_sek_kwh=1.42,
        )
        iso = isolate_slot_for_ev(original)
        assert iso.discharge_kw == 0.0 and iso.ev_isolation is True
        assert iso.water_heater_plans == {"main_tank": 3.4, "spa": 1.8}
        assert iso.water_heating_boost == {"main_tank": True}
        assert iso.ev_charger_plans == {"tesla": 0.0, "easee_fmb": 0.0}
        assert iso.sinks == {"villavagn_ac": True}
        assert iso.custom_entity_active is True
        assert iso.export_price_sek_kwh == 1.42
        assert (iso.water_kw, iso.load_kw, iso.soc_target, iso.soc_projected) == (3.4, 1.2, 60, 58)
        # and the original is untouched
        assert original.discharge_kw == 3.0 and original.ev_isolation is False

    def test_water_heaters_keep_their_planned_temperature_while_a_car_charges(self):
        devices = [
            WaterHeaterDeviceConfig(id="main_tank", target_entity="switch.vvb"),
            WaterHeaterDeviceConfig(
                id="spa", target_entity="input_number.spa", temp_off=20, temp_normal=38
            ),
        ]
        slot = isolate_slot_for_ev(
            SlotPlan(
                discharge_kw=2.0,
                water_kw=5.2,
                soc_target=60,
                water_heater_plans={"main_tank": 3.4, "spa": 1.8},
            )
        )
        d = _decide(
            slot, water_heater_config=WaterHeaterGlobalConfig(), water_heater_devices=devices
        )
        assert d.mode_intent == "idle"
        assert d.water_temps == {"main_tank": 60, "spa": 38}
