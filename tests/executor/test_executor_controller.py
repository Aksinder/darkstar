"""
Tests for Executor Controller Logic

Tests the controller decision-making that determines what actions to take
based on slot plan and override state.
"""

import pytest

from executor.config import (
    ControllerConfig,
    InverterConfig,
    WaterHeaterDeviceConfig,
    WaterHeaterGlobalConfig,
)
from executor.controller import Controller, ControllerDecision, make_decision
from executor.override import OverrideResult, OverrideType, SlotPlan, SystemState
from executor.profiles import InverterProfile, ProfileBehavior, ProfileMetadata


class TestControllerDecision:
    """Test the ControllerDecision dataclass."""

    def test_required_fields(self):
        """ControllerDecision requires mode_intent."""
        decision = ControllerDecision(
            mode_intent="export",
            charge_value=100.0,
            discharge_value=50.0,
            soc_target=80,
            water_temp=60,
        )
        assert decision.mode_intent == "export"
        assert decision.charge_value == 100.0
        assert decision.soc_target == 80

    def test_default_flags(self):
        """ControllerDecision has sensible default flags."""
        decision = ControllerDecision(
            mode_intent="idle",
            charge_value=0.0,
            discharge_value=0.0,
            soc_target=50,
            water_temp=40,
        )
        assert decision.write_charge_current is False
        assert decision.write_discharge_current is False
        assert decision.source == "plan"
        assert decision.reason == ""


class TestControllerConfig:
    """Test the ControllerConfig dataclass."""

    def test_default_values(self):
        """ControllerConfig has sensible defaults."""
        config = ControllerConfig()
        assert config.battery_capacity_kwh == 27.0
        assert config.nominal_voltage_v == 48.0
        assert config.min_charge_a == 10.0
        assert config.max_charge_a == 185.0
        assert config.max_discharge_a == 185.0
        assert config.round_step_a == 5.0

    def test_custom_values(self):
        """ControllerConfig can be initialized with custom values."""
        config = ControllerConfig(
            battery_capacity_kwh=30.0,
            nominal_voltage_v=50.0,
            min_charge_a=5.0,
            max_charge_a=200.0,
            max_discharge_a=200.0,
            round_step_a=10.0,
        )
        assert config.battery_capacity_kwh == 30.0
        assert config.nominal_voltage_v == 50.0
        assert config.min_charge_a == 5.0
        assert config.max_charge_a == 200.0
        assert config.max_discharge_a == 200.0
        assert config.round_step_a == 10.0


class TestControllerFollowPlan:
    """Test Controller._follow_plan behavior."""

    @pytest.fixture
    def controller(self):
        """Create a controller with default config."""
        return Controller(ControllerConfig(), InverterConfig())

    def test_export_mode_when_export_planned(self, controller):
        """When export is planned, use export mode intent."""
        # Battery export requires both export_kw > 0 AND discharge_kw > 0
        slot = SlotPlan(export_kw=5.0, discharge_kw=5.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "export"
        assert decision.source == "plan"

    def test_export_blocked_at_soc_floor(self, controller):
        """R1 runtime guard: export intent at/below the SoC floor downgrades to
        self_consumption — the MILP floor alone cannot protect against stale plans
        or SoC drift between replans (Sungrow Forced discharge runs to BMS cutoff)."""
        slot = SlotPlan(export_kw=5.0, discharge_kw=5.0)
        state = SystemState(current_soc_percent=20.0)  # default floor is 20

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    def test_export_allowed_just_above_soc_floor(self, controller):
        """Just above the floor the export intent goes through unchanged."""
        slot = SlotPlan(export_kw=5.0, discharge_kw=5.0)
        state = SystemState(current_soc_percent=21.0)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "export"

    def test_force_export_override_respects_the_soc_floor(self, controller):
        """R1 applies to MANUAL exports too — live-caught: a 15-min force_export
        quick action ran 29 -> 17.8 %, straight through the 20 % floor (only the
        timer stopped it). The override path must downgrade exactly like the plan
        path and must NOT force discharge when downgraded."""
        slot = SlotPlan()
        state = SystemState(current_soc_percent=19.0)
        override = OverrideResult(
            override_needed=True, override_type=OverrideType.FORCE_EXPORT, actions={}
        )

        decision = controller._apply_override(slot, state, override)

        assert decision.mode_intent == "self_consumption"
        assert decision.discharge_value == 0.0

    def test_force_export_override_allowed_above_floor(self, controller):
        slot = SlotPlan()
        state = SystemState(current_soc_percent=45.0)
        override = OverrideResult(
            override_needed=True, override_type=OverrideType.FORCE_EXPORT, actions={}
        )

        decision = controller._apply_override(slot, state, override)

        assert decision.mode_intent == "export"
        assert decision.discharge_value > 0.0

    def test_charge_mode_when_charging_only(self, controller):
        """When charging only, use charge mode intent."""
        slot = SlotPlan(export_kw=0.0, charge_kw=3.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "charge"

    def test_idle_mode_when_at_soc_target(self, controller):
        """When at or below SoC target, use idle mode intent."""
        slot = SlotPlan(export_kw=0.0, charge_kw=0.0, soc_target=50)
        state = SystemState(current_soc_percent=50)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "idle"

    def test_idle_mode_when_soc_target_has_decimal_precision(self, controller):
        """Regression test: SoC drift from decimal precision mismatch.

        Bug: Planner outputs 78.3% target, executor stores int(78.3)=78.
        Battery at 78.3% compared: 78.3 <= 78 was FALSE, stayed in
        self_consumption instead of idle, causing gradual drift.

        Fix: Round both planner output to int AND round current_soc at
        comparison. This test validates the comparison fix.
        """
        slot = SlotPlan(export_kw=0.0, charge_kw=0.0, soc_target=78)
        # Battery at 78.3% should trigger idle since rounded(78.3) = 78 <= 78
        state = SystemState(current_soc_percent=78.3)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "idle"

    def test_idle_mode_when_soc_slightly_above_target(self, controller):
        """When SoC rounds above target, use self_consumption (not idle)."""
        slot = SlotPlan(export_kw=0.0, charge_kw=0.0, soc_target=78)
        # Battery at 78.6% rounds to 79, which is > 78, so self_consumption
        state = SystemState(current_soc_percent=78.6)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    def test_self_consumption_when_above_soc_target(self, controller):
        """When above SoC target, use self_consumption mode intent."""
        slot = SlotPlan(export_kw=0.0, charge_kw=0.0, soc_target=50)
        state = SystemState(current_soc_percent=80)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    # Tests for PV surplus vs battery export mode selection (Bug fix)
    def test_pv_surplus_uses_self_consumption_not_export(self, controller):
        """PV surplus (charge + export, no discharge) uses self_consumption mode.

        Bug: Previously, any export_kw > 0 triggered export mode, causing
        battery to discharge when it should charge during PV surplus.
        Fix: PV surplus uses self_consumption mode (grid_charging OFF,
        max_charge_current limited to enable PV export).
        """
        slot = SlotPlan(charge_kw=3.5, export_kw=1.96, discharge_kw=0.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        # Should use self_consumption, not export (grid_charging OFF for PV surplus)
        assert decision.mode_intent == "self_consumption"

    def test_battery_export_uses_export_mode(self, controller):
        """Battery discharge to grid uses export mode."""
        slot = SlotPlan(charge_kw=0.0, export_kw=2.0, discharge_kw=2.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "export"

    def test_grid_charge_uses_charge_mode(self, controller):
        """Grid charging (no export) uses charge mode."""
        slot = SlotPlan(charge_kw=5.0, export_kw=0.0, discharge_kw=0.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "charge"

    def test_grid_charge_requires_pv_shortfall(self, controller):
        """Genuine grid charge: forecast PV can't cover charge + load (e.g. an
        overnight cheap-price top-up), so the shortfall is imported -> charge."""
        slot = SlotPlan(charge_kw=4.0, export_kw=0.0, discharge_kw=0.0, pv_kw=0.0, load_kw=1.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "charge"

    def test_near_full_pv_topup_no_export_uses_self_consumption(self, controller):
        """Regression: a near-full battery topping off the last % from PV emits a
        tiny charge_kw with export_kw == 0. Previously this was misclassified as
        grid charge (Forced charge), which set max_charge to the tiny value and
        blocked discharge -> a full battery sat idle while single-phase loads were
        bought from the grid ("köper el fast batteriet är fullt"). When PV covers
        charge + load, no grid import is needed -> use self_consumption so the
        battery still covers real-time/phase deficits."""
        slot = SlotPlan(charge_kw=0.1, export_kw=0.0, discharge_kw=0.0, pv_kw=1.5, load_kw=0.9)
        state = SystemState(current_soc_percent=99)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    def test_pv_charge_no_export_when_pv_exactly_covers_charge_and_load(self, controller):
        """Boundary: PV exactly equals charge + load (no import needed) -> the
        battery is free to self-consume, not force-grid-charged."""
        slot = SlotPlan(charge_kw=2.0, export_kw=0.0, discharge_kw=0.0, pv_kw=3.0, load_kw=1.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    def test_idle_when_at_soc_target_no_activity(self, controller):
        """When at SoC target with no charging/discharging, use idle."""
        slot = SlotPlan(charge_kw=0.0, export_kw=0.0, discharge_kw=0.0, soc_target=50)
        state = SystemState(current_soc_percent=50)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "idle"

    def test_idle_when_below_soc_target_no_activity(self, controller):
        """When below SoC target with no activity, use idle to prevent discharge."""
        slot = SlotPlan(charge_kw=0.0, export_kw=0.0, discharge_kw=0.0, soc_target=80)
        state = SystemState(current_soc_percent=70)

        decision = controller.decide(slot, state)

        # Below target with no charge/export planned - use idle to hold battery
        assert decision.mode_intent == "idle"

    def test_pv_surplus_hold_uses_self_consumption_not_idle(self, controller):
        """At SoC target during a PV-surplus slot, use self_consumption so the
        battery can cover load deficits (transient/single-phase loads the 3-phase
        PV feed can't reach) instead of freezing and forcing a grid import."""
        slot = SlotPlan(
            charge_kw=0.0,
            export_kw=0.0,
            discharge_kw=0.0,
            soc_target=100,
            pv_kw=11.0,
            load_kw=1.5,
        )
        state = SystemState(current_soc_percent=100)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"

    def test_hold_without_pv_surplus_stays_idle(self, controller):
        """At SoC target with no PV surplus (e.g. overnight reserve hold), keep
        idle so the battery is preserved at the safety floor."""
        slot = SlotPlan(
            charge_kw=0.0,
            export_kw=0.0,
            discharge_kw=0.0,
            soc_target=30,
            pv_kw=0.0,
            load_kw=1.5,
        )
        state = SystemState(current_soc_percent=30)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "idle"

    def test_pv_surplus_hold_with_ev_charging_stays_idle(self, controller):
        """A PV-surplus hold while an EV is charging must stay idle to preserve
        battery→EV source isolation (battery must not discharge into the EV)."""
        slot = SlotPlan(
            charge_kw=0.0,
            export_kw=0.0,
            discharge_kw=0.0,
            soc_target=100,
            pv_kw=11.0,
            load_kw=1.5,
            ev_charging_kw=7.0,
        )
        state = SystemState(current_soc_percent=100)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "idle"

    def test_soc_target_from_plan(self, controller):
        """SoC target comes from slot plan."""
        slot = SlotPlan(soc_target=85)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.soc_target == 85

    def test_water_temp_active_when_water_planned(self, controller):
        """Water temp set to 60 when water heating planned."""
        slot = SlotPlan(water_kw=3.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.water_temp == 60  # Normal heating temp

    def test_water_temp_off_when_no_water(self, controller):
        """Water temp set to 40 (off) when no water heating."""
        slot = SlotPlan(water_kw=0.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        assert decision.water_temp == 40  # Off/minimum

    def test_self_consumption_fallback_uses_max_charge(self, controller):
        """Verify default self_consumption fallback sets charge_value to max_charge_a (not 0)
        when nothing is planned and SoC is above target.
        """
        # When charge_kw is 0 and soc > target, it falls back to self_consumption
        slot = SlotPlan(charge_kw=0.0, soc_target=50)
        state = SystemState(current_soc_percent=80)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"
        assert decision.charge_value == controller.config.max_charge_a
        assert decision.write_charge_current is True

    def test_self_consumption_fallback_uses_max_charge_w(self):
        """Verify default self_consumption fallback sets charge_value to max_charge_w
        when control_unit is 'W'.
        """
        config = ControllerConfig()
        profile = InverterProfile(
            metadata=ProfileMetadata(name="test", version="1.0"),
            behavior=ProfileBehavior(control_unit="W"),
        )
        controller = Controller(config, InverterConfig(), profile=profile)

        slot = SlotPlan(charge_kw=0.0, soc_target=50, export_kw=0.0, discharge_kw=0.0)
        state = SystemState(current_soc_percent=80)

        decision = controller.decide(slot, state)

        assert decision.mode_intent == "self_consumption"
        assert decision.charge_value == config.max_charge_w
        assert decision.control_unit == "W"
        assert decision.write_charge_current is True

    def test_pv_surplus_uses_planned_charge_limit(self, controller):
        """Verify PV surplus path (charge_kw > 0) still uses the planned charge_value
        and is NOT overridden to max.
        """
        # When charge_kw > 0 and export_kw > 0, it uses self_consumption mode
        # but should keep the planned charge limit.
        planned_kw = 2.0
        slot = SlotPlan(charge_kw=planned_kw, export_kw=1.0, discharge_kw=0.0)
        state = SystemState()

        decision = controller.decide(slot, state)

        # 2kW at default 46V ≈ 43.47A -> rounds to 45A with default 5A step
        expected_amps = 45.0

        assert decision.mode_intent == "self_consumption"
        assert decision.charge_value == expected_amps
        assert decision.charge_value < controller.config.max_charge_a


class TestControllerApplyOverride:
    """Test Controller._apply_override behavior."""

    @pytest.fixture
    def controller(self):
        """Create a controller with default config."""
        return Controller(ControllerConfig(), InverterConfig())

    def test_override_uses_override_values(self, controller):
        """When override is active, use override values."""
        slot = SlotPlan(export_kw=5.0)  # Plan says export
        state = SystemState()
        override = OverrideResult(
            override_needed=True,
            override_type=OverrideType.FORCE_CHARGE,
            actions={
                "soc_target": 30,
            },
        )

        decision = controller.decide(slot, state, override)

        assert decision.mode_intent == "charge"  # Override sets charge intent
        assert decision.soc_target == 30
        assert decision.source == "override"

    def test_override_source_is_override(self, controller):
        """Decision source is 'override' when override is active."""
        slot = SlotPlan()
        state = SystemState()
        override = OverrideResult(
            override_needed=True,
            override_type=OverrideType.MANUAL_OVERRIDE,
            reason="User override",
        )

        decision = controller.decide(slot, state, override)

        assert decision.source == "override"
        assert decision.reason == "User override"

    def test_force_charge_sets_max_current(self, controller):
        """Force charge override sets max charging current."""
        slot = SlotPlan()
        state = SystemState()
        override = OverrideResult(
            override_needed=True,
            override_type=OverrideType.FORCE_CHARGE,
            actions={},
        )

        decision = controller.decide(slot, state, override)

        # Default is Amps
        assert decision.mode_intent == "charge"
        assert decision.charge_value == controller.config.max_charge_a
        assert decision.write_charge_current is True

    def test_force_export_sets_max_discharge(self, controller):
        """Force export override sets max discharge current."""
        slot = SlotPlan()
        state = SystemState()
        override = OverrideResult(
            override_needed=True,
            override_type=OverrideType.FORCE_EXPORT,
            actions={},
        )

        decision = controller.decide(slot, state, override)

        assert decision.mode_intent == "export"
        assert decision.discharge_value == controller.config.max_discharge_a
        assert decision.write_discharge_current is True

    def test_no_override_follows_plan(self, controller):
        """When no override, decision follows plan."""
        # Battery export requires both export_kw > 0 AND discharge_kw > 0
        slot = SlotPlan(export_kw=5.0, discharge_kw=5.0)
        state = SystemState()
        override = None

        decision = controller.decide(slot, state, override)

        assert decision.source == "plan"
        assert decision.mode_intent == "export"


class TestCalculateChargeCurrent:
    """Test Controller._calculate_charge_current."""

    def test_no_charge_planned_returns_zero(self):
        """When no charge planned, return 0."""
        config = ControllerConfig(min_voltage_v=46.0)
        controller = Controller(config, InverterConfig())
        slot = SlotPlan(charge_kw=0.0)
        state = SystemState()

        current, should_write = controller._calculate_charge_limit(slot, state)

        assert current == 0.0
        assert should_write is False

    def test_kw_to_amps_conversion(self):
        """Correctly converts kW to Amps."""
        config = ControllerConfig(
            min_voltage_v=46.0,
            round_step_a=5.0,
            min_charge_a=10.0,
            max_charge_a=185.0,
        )
        controller = Controller(config, InverterConfig())
        # 5 kW at 46V = 5000/46 ≈ 108.7A → rounds to 110A
        slot = SlotPlan(charge_kw=5.0)
        state = SystemState()

        current, _ = controller._calculate_charge_limit(slot, state)

        assert current == 110.0  # Rounded to step

    def test_respects_min_limit(self):
        """Current is clamped to minimum."""
        config = ControllerConfig(
            min_voltage_v=46.0,
            round_step_a=5.0,
            min_charge_a=10.0,
            max_charge_a=185.0,
        )
        controller = Controller(config, InverterConfig())
        # Very small charge → would be below min
        slot = SlotPlan(charge_kw=0.1)  # 0.1kW at 46V ≈ 2.2A
        state = SystemState()

        current, _ = controller._calculate_charge_limit(slot, state)

        assert current >= config.min_charge_a

    def test_respects_max_limit(self):
        """Current is clamped to maximum."""
        config = ControllerConfig(
            min_voltage_v=46.0,
            round_step_a=5.0,
            min_charge_a=10.0,
            max_charge_a=185.0,
        )
        controller = Controller(config, InverterConfig())
        # Very high charge → would exceed max
        slot = SlotPlan(charge_kw=20.0)  # 20kW at 46V ≈ 435A
        state = SystemState()

        current, _ = controller._calculate_charge_limit(slot, state)

        assert current <= config.max_charge_a


class TestCalculateDischargeCurrent:
    """Test Controller._calculate_discharge_current."""

    def test_export_mode_limits_discharge(self):
        """When exporting, discharge is limited to export rate."""
        config = ControllerConfig(
            min_voltage_v=46.0,
            round_step_a=5.0,
            min_charge_a=10.0,
            max_charge_a=185.0,
            max_discharge_a=185.0,
        )
        controller = Controller(config, InverterConfig())
        # 3 kW export at 46V ≈ 65A → rounds to 65A
        slot = SlotPlan(export_kw=3.0)
        state = SystemState()

        current, should_write = controller._calculate_discharge_limit(slot, state)

        # Bug fix #1: Even when exporting, we set discharge to MAX
        # so local load spikes are handled by battery.
        assert current == 185.0
        assert should_write is True

    def test_no_export_sets_max_discharge(self):
        """When not exporting, discharge is set to max for load handling."""
        config = ControllerConfig(max_discharge_a=185.0)
        controller = Controller(config, InverterConfig())
        slot = SlotPlan(export_kw=0.0)
        state = SystemState()

        current, should_write = controller._calculate_discharge_limit(slot, state)

        assert current == 185.0  # Max for load spikes
        assert should_write is True


class TestGenerateReason:
    """Test Controller._generate_reason."""

    def test_charge_reason(self):
        """Reason includes charge info."""
        controller = Controller(ControllerConfig(), InverterConfig())
        slot = SlotPlan(charge_kw=5.0)

        reason = controller._generate_reason(slot, "charge")

        assert "Charge 5.0kW" in reason
        assert "Charge" in reason

    def test_export_reason(self):
        """Reason includes export info."""
        controller = Controller(ControllerConfig(), InverterConfig())
        slot = SlotPlan(export_kw=3.0)

        reason = controller._generate_reason(slot, "export")

        assert "Export 3.0kW" in reason
        assert "Export" in reason

    def test_idle_reason(self):
        """Reason shows idle when nothing planned."""
        controller = Controller(ControllerConfig(), InverterConfig())
        slot = SlotPlan()

        reason = controller._generate_reason(slot, "idle")

        assert "Hold/Idle" in reason
        assert "Idle" in reason


class TestExportWithLoadCalculation:
    """Test export_with_load_w calculation for Fronius export mode."""

    def test_export_with_load_calculation(self):
        """export_with_load_w = export_power_w + (load_kw * 1000), rounded to step."""
        config = ControllerConfig()
        inverter_config = InverterConfig()
        controller = Controller(config, inverter_config)

        # Export 2kW, load 0.5kW -> total 2.5kW = 2500W
        slot = SlotPlan(export_kw=2.0, load_kw=0.5)
        state = SystemState()

        decision = controller._follow_plan(slot, state)

        assert decision.export_power_w == 2000.0  # 2kW in Watts
        assert decision.export_with_load_w == 2500.0  # (2 + 0.5) * 1000, rounded to 100

    def test_export_with_load_rounding(self):
        """export_with_load_w is rounded to round_step_w (100W default)."""
        config = ControllerConfig()
        inverter_config = InverterConfig()
        controller = Controller(config, inverter_config)

        # Export 1.23kW, load 0.456kW -> total 1.686kW = 1686W
        # Should round to 1700W (nearest 100W)
        slot = SlotPlan(export_kw=1.23, load_kw=0.456)
        state = SystemState()

        decision = controller._follow_plan(slot, state)

        assert decision.export_with_load_w == 1700.0  # Rounded to nearest 100W

    def test_no_export_sets_export_with_load_to_zero(self):
        """When not exporting, export_with_load_w should be 0."""
        config = ControllerConfig()
        inverter_config = InverterConfig()
        controller = Controller(config, inverter_config)

        # No export, just charging
        slot = SlotPlan(charge_kw=3.0, load_kw=1.0)
        state = SystemState()

        decision = controller._follow_plan(slot, state)

        assert decision.export_power_w == 0.0
        assert decision.export_with_load_w == 0.0

    def test_export_with_zero_load(self):
        """export_with_load_w equals export_power_w when load is 0."""
        config = ControllerConfig()
        inverter_config = InverterConfig()
        controller = Controller(config, inverter_config)

        slot = SlotPlan(export_kw=3.0, load_kw=0.0)
        state = SystemState()

        decision = controller._follow_plan(slot, state)

        assert decision.export_power_w == 3000.0
        assert decision.export_with_load_w == 3000.0


class TestMakeDecisionConvenience:
    """Test the make_decision convenience function."""

    def test_with_defaults(self):
        """Works with default config."""
        # Battery export requires both export_kw > 0 AND discharge_kw > 0
        slot = SlotPlan(export_kw=5.0, discharge_kw=5.0)
        state = SystemState()

        decision = make_decision(slot, state)

        assert isinstance(decision, ControllerDecision)
        assert decision.mode_intent == "export"

    def test_with_custom_config(self):
        """Works with custom config."""
        slot = SlotPlan(charge_kw=5.0)
        state = SystemState()
        config = ControllerConfig(max_charge_a=100.0)

        decision = make_decision(slot, state, config=config)

        # Charge current should respect custom max
        assert decision.charge_value <= 100.0

    def test_with_override(self):
        """Works with override."""
        slot = SlotPlan(export_kw=5.0)
        state = SystemState()
        override = OverrideResult(
            override_needed=True,
            override_type=OverrideType.FORCE_CHARGE,
            actions={},
        )

        decision = make_decision(slot, state, override)

        assert decision.source == "override"
        assert decision.mode_intent == "charge"


class TestSlotPlanWaterHeaterPlans:
    """Task 6.5: SlotPlan parses per-device water data."""

    def test_slot_plan_default_empty_water_heater_plans(self):
        """SlotPlan defaults water_heater_plans to empty dict."""
        slot = SlotPlan()
        assert slot.water_heater_plans == {}

    def test_slot_plan_accepts_water_heater_plans(self):
        """SlotPlan stores water_heater_plans dict."""
        slot = SlotPlan(water_heater_plans={"wh1": 3.0, "wh2": 0.0})
        assert slot.water_heater_plans["wh1"] == 3.0
        assert slot.water_heater_plans["wh2"] == 0.0


class TestControllerPerDeviceWaterTemps:
    """Task 6.5: controller produces per-device water temps."""

    def _make_devices(self):
        return [
            WaterHeaterDeviceConfig(id="wh1", target_entity="climate.wh1", power_kw=3.0),
            WaterHeaterDeviceConfig(id="wh2", target_entity="climate.wh2", power_kw=2.0),
        ]

    def _make_global_config(self):
        return WaterHeaterGlobalConfig(temp_normal=60, temp_off=35)

    def test_active_heater_gets_temp_normal(self):
        """Heater with planned kW > 0 gets temp_normal."""
        slot = SlotPlan(water_heater_plans={"wh1": 3.0, "wh2": 0.0})
        state = SystemState()

        decision = make_decision(
            slot,
            state,
            water_heater_config=self._make_global_config(),
            water_heater_devices=self._make_devices(),
        )

        assert decision.water_temps["wh1"] == 60
        assert decision.water_temps["wh2"] == 35

    def test_all_heaters_off_when_not_planned(self):
        """All heaters get temp_off when not planned."""
        slot = SlotPlan(water_heater_plans={"wh1": 0.0, "wh2": 0.0})
        state = SystemState()

        decision = make_decision(
            slot,
            state,
            water_heater_config=self._make_global_config(),
            water_heater_devices=self._make_devices(),
        )

        assert decision.water_temps == {"wh1": 35, "wh2": 35}

    def test_old_format_fallback_empty_water_heater_plans(self):
        """Old schedule format with empty water_heater_plans: all devices get temp_off."""
        slot = SlotPlan(water_kw=3.0, water_heater_plans={})
        state = SystemState()

        decision = make_decision(
            slot,
            state,
            water_heater_config=self._make_global_config(),
            water_heater_devices=self._make_devices(),
        )

        # All devices default to temp_off since no per-device plans
        assert decision.water_temps == {"wh1": 35, "wh2": 35}

    def test_no_devices_produces_empty_water_temps(self):
        """When no water_heater_devices, water_temps is empty."""
        slot = SlotPlan(water_heater_plans={"wh1": 3.0})
        state = SystemState()

        decision = make_decision(slot, state)

        assert decision.water_temps == {}

    def test_scalar_water_temp_still_populated(self):
        """Scalar water_temp is still set for backward compat."""
        slot = SlotPlan(water_kw=3.0)
        state = SystemState()

        decision = make_decision(
            slot,
            state,
            water_heater_config=self._make_global_config(),
        )

        assert decision.water_temp == 60


class TestPerDeviceWaterTemps:
    """Thermostatic loads don't share a temperature range: the spa runs 20-40 C via
    an input_number bridge (>30 = heat to value, <=30 = fan_only), so the tanks'
    global temp_off=40 would read as 'heat to 40' and temp_normal=60 exceeds its
    max. Per-device overrides fix that; unset devices keep the globals."""

    @staticmethod
    def _controller():
        from executor.config import (
            ControllerConfig,
            InverterConfig,
            WaterHeaterDeviceConfig,
            WaterHeaterGlobalConfig,
        )
        from executor.controller import Controller

        return Controller(
            config=ControllerConfig(),
            inverter_config=InverterConfig(),
            water_heater_config=WaterHeaterGlobalConfig(
                temp_off=40, temp_normal=60, temp_boost=70, temp_max=85
            ),
            water_heater_devices=[
                WaterHeaterDeviceConfig(id="main_tank", target_entity="switch.vvb"),
                WaterHeaterDeviceConfig(
                    id="spa",
                    target_entity="input_number.spa_darkstar_target_temp",
                    temp_off=20, temp_normal=38, temp_boost=40, temp_max=40,
                ),
            ],
        )

    def test_off_uses_per_device_override(self):
        c = self._controller()
        temps = c._determine_water_temps(SlotPlan())
        assert temps["spa"] == 20      # fan_only via the bridge
        assert temps["main_tank"] == 40  # global

    def test_planned_heat_uses_per_device_normal(self):
        c = self._controller()
        slot = SlotPlan(water_heater_plans={"spa": 1.8, "main_tank": 3.4})
        temps = c._determine_water_temps(slot)
        assert temps["spa"] == 38
        assert temps["main_tank"] == 60

    def test_boost_uses_per_device_max(self):
        c = self._controller()
        slot = SlotPlan(
            water_heater_plans={"spa": 1.8, "main_tank": 3.4},
            water_heating_boost={"spa": True, "main_tank": True},
        )
        temps = c._determine_water_temps(slot)
        assert temps["spa"] == 40   # never the tanks' 85
        assert temps["main_tank"] == 85

    def test_zero_is_a_real_override_not_absent(self):
        from executor.config import WaterHeaterDeviceConfig

        c = self._controller()
        c.water_heater_devices = [
            WaterHeaterDeviceConfig(id="x", target_entity="switch.x", temp_off=0)
        ]
        assert c._determine_water_temps(SlotPlan())["x"] == 0
