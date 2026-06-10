"""Tests for the EV surplus-follow controller (executor/ev_surplus.py)."""

from executor.ev_surplus import (
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    compute_ev_surplus,
)


def _charger(**kw) -> ChargerState:
    base = dict(
        id="tesla",
        plugged=True,
        at_home=True,
        enabled=True,
        current_power_w=0.0,
        max_current_a=16.0,  # 16 A * 230 V * 3ph ≈ 11 kW
        min_current_a=6.0,  # 6 A * 230 * 3 ≈ 4.14 kW
        phases=3,
        voltage_v=230.0,
        controllable=True,
        priority=0,
    )
    base.update(kw)
    return ChargerState(**base)


def _inputs(**kw) -> EVSurplusInputs:
    base = dict(
        pv_w=12000.0,
        grid_w=0.0,
        battery_w=0.0,
        battery_soc_percent=90.0,
        import_price_sek=1.0,
        remaining_solar_kwh=10.0,
        chargers=[_charger()],
    )
    base.update(kw)
    return EVSurplusInputs(**base)


def _cfg(**kw) -> EVSurplusConfig:
    base = dict(enabled=True, gain=0.5, deadband_w=250.0)
    base.update(kw)
    return EVSurplusConfig(**base)


class TestGating:
    def test_disabled_returns_no_commands(self):
        assert compute_ev_surplus(_inputs(grid_w=-10000), _cfg(enabled=False)) == []

    def test_no_active_chargers(self):
        # Unplugged / away / disabled chargers are not controlled.
        away = _charger(at_home=False)
        unplugged = _charger(id="easee", plugged=False)
        out = compute_ev_surplus(_inputs(chargers=[away, unplugged]), _cfg())
        assert out == []


class TestSolarSurplus:
    def test_export_raises_current(self):
        # Exporting 10 kW => big positive headroom => the car is told to charge.
        out = compute_ev_surplus(_inputs(grid_w=-10000.0), _cfg())
        assert len(out) == 1
        cmd = out[0]
        assert cmd.switch_on and cmd.set_current_a is not None
        assert cmd.target_power_w > 4000  # above the 3ph 6A floor

    def test_small_surplus_below_min_keeps_off(self):
        # Only 2 kW export: a 3-phase 6 A charger can't run that low => off.
        out = compute_ev_surplus(_inputs(grid_w=-2000.0), _cfg())
        assert out[0].switch_on is False
        assert out[0].target_power_w == 0.0


class TestBatteryProtection:
    def test_battery_discharge_backs_off(self):
        # Battery discharging 3 kW to feed a car already pulling 6 kW => reduce.
        ch = _charger(current_power_w=6000.0)
        out = compute_ev_surplus(
            _inputs(grid_w=0.0, battery_w=-3000.0, chargers=[ch]), _cfg()
        )
        # target = 6000 + 0.5*(0 + (-3000)) = 4500 -> still on but lower than 6000.
        assert out[0].target_power_w < 6000.0

    def test_battery_charging_is_handed_to_car(self):
        # Battery absorbing 6 kW of surplus => give that to the car instead (above the
        # 3-phase 6 A start floor).
        out = compute_ev_surplus(_inputs(grid_w=0.0, battery_w=6000.0), _cfg())
        assert out[0].switch_on and out[0].target_power_w > 0


class TestCheapGridTier:
    def test_cheap_grid_allows_import(self):
        cfg = _cfg(cheap_grid_price_sek=0.5, cheap_grid_allowance_w=6000.0)
        # Price below threshold, grid balanced, battery idle => setpoint pulls EV up.
        out = compute_ev_surplus(_inputs(import_price_sek=0.3, grid_w=0.0), cfg)
        assert out[0].switch_on and out[0].target_power_w > 0

    def test_expensive_grid_no_import(self):
        cfg = _cfg(cheap_grid_price_sek=0.5, cheap_grid_allowance_w=6000.0)
        # Price above threshold, nothing exporting => no budget => off.
        out = compute_ev_surplus(_inputs(import_price_sek=1.5, grid_w=0.0), cfg)
        assert out[0].switch_on is False


class TestBatteryAssistTier:
    def _cfg_assist(self, **kw):
        base = dict(
            enabled=True,
            battery_assist_enabled=True,
            battery_assist_max_price_sek=0.0,
            battery_assist_min_remaining_solar_kwh=8.0,
            battery_assist_floor_soc=30.0,
            battery_assist_allowance_w=6000.0,  # above the 3-phase 6 A start floor (~4.1 kW)
        )
        base.update(kw)
        return EVSurplusConfig(**base)

    def test_negative_price_plus_solar_headroom_allows_battery(self):
        out = compute_ev_surplus(
            _inputs(import_price_sek=-0.1, remaining_solar_kwh=10.0, battery_soc_percent=80.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on and out[0].target_power_w > 0

    def test_blocked_when_soc_below_floor(self):
        out = compute_ev_surplus(
            _inputs(import_price_sek=-0.1, remaining_solar_kwh=10.0, battery_soc_percent=25.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on is False  # no battery assist below floor, no other budget

    def test_blocked_when_not_enough_remaining_solar(self):
        out = compute_ev_surplus(
            _inputs(import_price_sek=-0.1, remaining_solar_kwh=3.0, battery_soc_percent=80.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on is False

    def test_blocked_when_price_not_negative(self):
        out = compute_ev_surplus(
            _inputs(import_price_sek=0.5, remaining_solar_kwh=10.0, battery_soc_percent=80.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on is False


class TestDynamics:
    def test_deadband_holds(self):
        ch = _charger(current_power_w=5000.0)
        # Tiny imbalance within deadband => hold at current.
        out = compute_ev_surplus(_inputs(grid_w=-100.0, chargers=[ch]), _cfg(deadband_w=250.0))
        assert out[0].target_power_w == 5000.0

    def test_clamped_to_charger_max(self):
        # Huge export can't exceed the charger's 11 kW ceiling.
        out = compute_ev_surplus(_inputs(grid_w=-50000.0), _cfg(gain=1.0))
        assert out[0].target_power_w <= 16.0 * 230.0 * 3 + 1
        assert out[0].set_current_a <= 16.0


class TestDistribution:
    def test_priority_fills_first_charger_first(self):
        tesla = _charger(id="tesla", priority=0, current_power_w=0.0)
        easee = _charger(id="easee", priority=1, max_current_a=16.0, current_power_w=0.0)
        # Export ~8 kW, gain 1.0 => ~8 kW budget: Tesla (priority 0) fills first.
        out = compute_ev_surplus(_inputs(grid_w=-8000.0, chargers=[tesla, easee]), _cfg(gain=1.0))
        by_id = {c.id: c for c in out}
        assert by_id["tesla"].target_power_w >= by_id["easee"].target_power_w

    def test_binary_charger_gets_no_current(self):
        binc = _charger(id="easee", controllable=False)
        out = compute_ev_surplus(_inputs(grid_w=-10000.0, chargers=[binc]), _cfg())
        assert out[0].set_current_a is None
        assert out[0].switch_on is True

    def test_easee_priority_one_fills_before_tesla(self):
        # User's config: Easee = #1 (priority 0), Tesla = #2 (priority 1).
        easee = _charger(id="easee", priority=0, min_current_a=6.0, current_power_w=0.0)
        tesla = _charger(id="tesla", priority=1, min_current_a=5.0, current_power_w=0.0)
        out = compute_ev_surplus(_inputs(grid_w=-8000.0, chargers=[easee, tesla]), _cfg(gain=1.0))
        by_id = {c.id: c for c in out}
        assert by_id["easee"].target_power_w >= by_id["tesla"].target_power_w


class TestManualOverride:
    def test_force_off_never_charges(self):
        ch = _charger(override="force_off")
        out = compute_ev_surplus(_inputs(grid_w=-12000.0, chargers=[ch]), _cfg())
        assert out[0].switch_on is False and out[0].target_power_w == 0.0
        assert "force_off" in out[0].reason

    def test_force_on_charges_at_max_despite_no_surplus(self):
        # Importing, battery draining — surplus control would say OFF, but force_on overrules.
        ch = _charger(override="force_on")
        out = compute_ev_surplus(_inputs(grid_w=5000.0, battery_w=-3000.0, chargers=[ch]), _cfg())
        assert out[0].switch_on is True
        assert out[0].set_current_a == ch.max_current_a
        assert "force_on" in out[0].reason

    def test_forced_on_does_not_consume_auto_budget(self):
        # Tesla force_on (its draw shows in grid/battery); Easee auto still gets the surplus.
        tesla = _charger(id="tesla", override="force_on")
        easee = _charger(id="easee", override="auto", current_power_w=0.0)
        out = compute_ev_surplus(_inputs(grid_w=-9000.0, chargers=[tesla, easee]), _cfg(gain=1.0))
        by_id = {c.id: c for c in out}
        assert by_id["tesla"].set_current_a == tesla.max_current_a  # forced
        assert by_id["easee"].switch_on is True  # auto still charges on the surplus


class TestWriteGuard:
    """Rate-limit charge-current writes to protect the car (and be conservative)."""

    def _cfg(self):
        from executor.ev_surplus import WriteGuardConfig

        return WriteGuardConfig(min_step_a=1.0, min_interval_s=90.0)

    def test_first_write_always_proceeds(self):
        from executor.ev_surplus import should_write_current

        assert should_write_current(None, None, 10.0, 1000.0, self._cfg()) is True

    def test_small_step_is_skipped(self):
        from executor.ev_surplus import should_write_current

        # 10.0 -> 10.5 (<1 A) => no write even after a long interval.
        assert should_write_current(10.0, 0.0, 10.5, 10_000.0, self._cfg()) is False

    def test_too_soon_is_skipped(self):
        from executor.ev_surplus import should_write_current

        # Big change but only 30 s since last write => wait.
        assert should_write_current(10.0, 1000.0, 16.0, 1030.0, self._cfg()) is False

    def test_real_change_after_interval_writes(self):
        from executor.ev_surplus import should_write_current

        assert should_write_current(10.0, 1000.0, 16.0, 1100.0, self._cfg()) is True

    def test_stop_bypasses_interval(self):
        from executor.ev_surplus import should_write_current

        # Dropping to 0 (stop) is allowed immediately even 1 s after the last write.
        assert should_write_current(16.0, 1000.0, 0.0, 1001.0, self._cfg()) is True
