"""Tests for the EV surplus-follow controller (executor/ev_surplus.py)."""

import pytest

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
    def test_deadband_holds_stable(self):
        # Small opposite perturbations both within the deadband => identical command
        # (the budget doesn't drift on noise; the deadband's job is stability).
        ch = _charger(current_power_w=5000.0)
        a = compute_ev_surplus(_inputs(grid_w=-100.0, chargers=[ch]), _cfg(deadband_w=250.0))
        b = compute_ev_surplus(_inputs(grid_w=120.0, chargers=[ch]), _cfg(deadband_w=250.0))
        assert a[0].set_current_a == b[0].set_current_a


class TestChunkySteps:
    def test_current_snaps_to_step_grid(self):
        # 2 A grid => commanded current is an even multiple, not a jittery 7.2 A.
        out = compute_ev_surplus(_inputs(grid_w=-10000.0), _cfg(gain=1.0, current_step_a=2.0))
        amps = out[0].set_current_a
        assert amps is not None and abs((amps / 2.0) - round(amps / 2.0)) < 1e-9

    def test_off_hysteresis_needs_extra_to_start(self):
        # Charger OFF, budget just at its bare min (4.14 kW) but below min*1.15 => stays off.
        off = _charger(current_power_w=0.0)
        out = compute_ev_surplus(_inputs(grid_w=-4300.0, chargers=[off]), _cfg(gain=1.0, start_hysteresis=0.15))
        assert out[0].switch_on is False

    def test_on_hysteresis_keeps_running_at_min(self):
        # Same budget but the charger is already ON => it keeps running down to its true min.
        on = _charger(current_power_w=4200.0)
        out = compute_ev_surplus(_inputs(grid_w=0.0, battery_w=0.0, chargers=[on]), _cfg(gain=1.0, start_hysteresis=0.15))
        # headroom ~0 => hold at ~4200, above min 4140 => stays on.
        assert out[0].switch_on is True

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


class TestSocCapAndDeadline:
    """SoC caps (never overcharge) + grid-backed deadline floors (make departure)."""

    def test_soc_at_or_above_target_caps_off(self):
        # FMB at 50% with a 15% vacation target => stop, even with huge surplus.
        ch = _charger(id="easee", soc_percent=50.0, target_soc_percent=15.0)
        out = compute_ev_surplus(_inputs(grid_w=-12000.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is False and out[0].target_power_w == 0.0
        assert "cap" in out[0].reason

    def test_soc_without_target_does_not_cap(self):
        # A known SoC but no target => legacy opportunistic charging (no cap).
        ch = _charger(soc_percent=50.0, target_soc_percent=None)
        out = compute_ev_surplus(_inputs(grid_w=-12000.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is True

    def test_below_target_still_charges_on_surplus(self):
        ch = _charger(soc_percent=10.0, target_soc_percent=80.0)
        out = compute_ev_surplus(_inputs(grid_w=-12000.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is True

    def test_deadline_behind_forces_grid_without_surplus(self):
        # 10%->80% of 60 kWh in 10 h => ~4.7 kW avg floor (> the 6 A min). No surplus
        # (importing) => it must still charge from grid to make the deadline.
        ch = _charger(
            soc_percent=10.0, target_soc_percent=80.0, capacity_kwh=60.0, deadline_hours=10.0
        )
        out = compute_ev_surplus(_inputs(grid_w=5000.0, battery_w=0.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is True
        assert out[0].set_current_a is not None and out[0].set_current_a >= 6.0
        assert "deadline" in out[0].reason

    def test_plenty_of_time_does_not_force_grid(self):
        # Only 10% to go over 100 h => required power is a trickle (< the charger min),
        # so don't pay for grid — wait for solar. No surplus => off.
        ch = _charger(
            soc_percent=70.0, target_soc_percent=80.0, capacity_kwh=60.0, deadline_hours=100.0
        )
        out = compute_ev_surplus(_inputs(grid_w=5000.0, battery_w=0.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is False

    def test_deadline_car_overtakes_priority(self):
        # Easee is priority 0 (normally first), but the Tesla (priority 1) is behind its
        # departure deadline => the Tesla overtakes and gets the limited budget.
        easee = _charger(id="easee", priority=0, current_power_w=0.0)
        tesla = _charger(
            id="tesla", priority=1, current_power_w=0.0,
            soc_percent=10.0, target_soc_percent=80.0, capacity_kwh=60.0, deadline_hours=2.0,
        )
        out = compute_ev_surplus(
            _inputs(grid_w=-6000.0, chargers=[easee, tesla]), _cfg(gain=1.0)
        )
        by_id = {c.id: c for c in out}
        assert by_id["tesla"].target_power_w > by_id["easee"].target_power_w
        assert by_id["tesla"].switch_on is True

    def test_cap_takes_precedence_over_deadline(self):
        # Already at/above target => capped off, deadline ignored (nothing left to do).
        ch = _charger(
            soc_percent=80.0, target_soc_percent=80.0, capacity_kwh=60.0, deadline_hours=0.5
        )
        out = compute_ev_surplus(_inputs(grid_w=5000.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is False
        assert "cap" in out[0].reason

    def test_missing_capacity_disables_deadline_floor(self):
        # Behind on time but no capacity configured => can't size a floor => surplus only.
        ch = _charger(
            soc_percent=10.0, target_soc_percent=80.0, capacity_kwh=0.0, deadline_hours=1.0
        )
        out = compute_ev_surplus(_inputs(grid_w=5000.0, chargers=[ch]), _cfg(gain=1.0))
        assert out[0].switch_on is False


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

    def test_start_from_stop_bypasses_interval(self):
        from executor.ev_surplus import should_write_current

        # Starting from 0 (e.g. deadline-forced or fresh surplus) is allowed immediately too,
        # so a restart isn't stranded at 0 for the whole interval after a stop.
        assert should_write_current(0.0, 1000.0, 8.0, 1001.0, self._cfg()) is True


class TestEvPriorityBatteryCap:
    """The return channel the reserve gate never had (2026-08-23).

    Live: a Tesla at 45 % pinned to 6 A under 14.5 kW of PV while the home battery
    (82 %, reserve inactive) took the rest — the servo raised the car when the
    battery paused, the battery took it back. The cap tells the battery to leave
    the cars what they were just told to take THIS tick.
    """

    def _run(self, *, battery_w, car_w, grid_w=0.0, plan_w=0.0, soc=80.0,
             remaining=30.0, enabled=True, chargers=None, cfg_kw=None):
        from executor.ev_surplus import EVSurplusTick, ev_priority_battery_cap_w

        base_cfg = dict(
            ev_priority_battery_cap_enabled=enabled,
            # A deadline-style reserve that is clearly INACTIVE: full-ish battery,
            # lots of sun still to come.
            battery_yield_soc=95.0, battery_capacity_kwh=16.0,
            battery_fill_margin_kwh=3.0,
        )
        base_cfg.update(cfg_kw or {})
        cfg = _cfg(**base_cfg)
        inputs = _inputs(
            battery_w=battery_w, grid_w=grid_w, battery_soc_percent=soc,
            remaining_solar_kwh=remaining,
            chargers=chargers or [_charger(current_power_w=car_w)],
            plan_battery_charge_w=plan_w,
        )
        tick = EVSurplusTick()
        compute_ev_surplus(inputs, cfg, tick_out=tick)
        return ev_priority_battery_cap_w(inputs, cfg, tick, plan_w), tick

    def test_the_battery_gets_what_the_cars_leave(self):
        """spare 10 kW: car 3, battery 7, grid 0. Target = 3 + 0.5*7 = 6.5 kW, but
        the car is COMMANDED 9.0 A = 6.21 kW (amps are quantized), so the battery
        gets 3.79 kW. Deriving the cap from the commanded draw rather than the raw
        target is what makes car + battery sum to the spare exactly — the old
        target-derived 3.5 kW left 290 W allocated to nobody, i.e. exported."""
        cap, tick = self._run(battery_w=7000.0, car_w=3000.0)
        assert tick.computed and tick.demand and not tick.reserve_active
        assert tick.target_total_w == pytest.approx(6500.0)
        assert tick.commanded_on_total_w == pytest.approx(6210.0)
        assert cap == pytest.approx(3790.0)
        assert tick.commanded_on_total_w + cap == pytest.approx(10000.0)

    def test_off_by_default(self):
        cap, _ = self._run(battery_w=7000.0, car_w=3000.0, enabled=False)
        assert cap is None

    def test_reserve_active_means_hands_off(self):
        """Battery nearly empty with little sun left: the reserve claims the inflow,
        and the cap must step aside — the same verdict, not a second rule."""
        cap, tick = self._run(battery_w=7000.0, car_w=3000.0, soc=20.0, remaining=2.0)
        assert tick.reserve_active
        assert cap is None

    def test_a_discharging_battery_is_never_capped(self):
        cap, _ = self._run(battery_w=-2000.0, car_w=3000.0)
        assert cap is None

    def test_no_demand_no_cap(self):
        """A fleet pinned at its max has nothing to gain from starving the battery."""
        full = _charger(current_power_w=11040.0)  # 16 A x 3 x 230 V
        cap, tick = self._run(battery_w=2000.0, car_w=11040.0, chargers=[full])
        assert not tick.demand
        assert cap is None

    def test_the_plan_is_a_floor(self):
        """The solver booked 4 kW for the battery (an evening peak ahead). A car
        does not get to undercut that."""
        cap, _ = self._run(battery_w=7000.0, car_w=3000.0, plan_w=4000.0)
        assert cap == pytest.approx(4000.0)

    def test_a_car_being_stopped_has_no_claim(self):
        """Target 3 kW is below the charger's 4.14 kW minimum, so the servo commands
        the car OFF. This test previously asserted cap == 100 W — pinning the home
        battery to its floor while the car was being told to stop, which is the
        daylight half of the 2026-08-24 flap. Nobody is drawing: hands off."""
        cap, tick = self._run(battery_w=100.0, car_w=3000.0)
        assert tick.target_total_w == pytest.approx(3000.0)
        assert tick.commanded_on_total_w == 0.0
        assert cap is None

    def test_grid_import_is_not_spare(self):
        """Battery 4 kW, car 5 kW, but 2 kW of that is being IMPORTED: only 7 kW is
        really surplus, so the cap is what is left of seven after the target."""
        cap, tick = self._run(battery_w=4000.0, car_w=5000.0, grid_w=2000.0)
        # headroom = -2000 + 4000 = 2000 -> target 6000, commanded 6210 (9.0 A);
        # spare 7000 -> cap 790, and 6210 + 790 == 7000 exactly.
        assert tick.target_total_w == pytest.approx(6000.0)
        assert cap == pytest.approx(790.0)

    def test_convergence_without_oscillation(self):
        """THE anti-hunt proof. Iterate the closed loop: each tick the car draws what
        it was COMMANDED last tick and the battery charges at last tick's cap. The car
        must climb monotonically and the cap fall monotonically — no reversal — to a
        fixed point where car + battery is exactly the surplus (nothing exported,
        nothing imported, no hunting)."""
        spare = 10000.0
        car, batt = 3000.0, 7000.0
        cars, caps = [], []
        for _ in range(10):
            cap, tick = self._run(battery_w=batt, car_w=car)
            assert cap is not None
            cars.append(tick.commanded_on_total_w)
            caps.append(cap)
            car = tick.commanded_on_total_w
            batt = min(cap, spare - car)  # the inverter takes at most the cap
            # Every tick allocates the whole surplus and no more.
            assert car + cap == pytest.approx(spare)
        assert cars == sorted(cars), cars
        assert caps == sorted(caps, reverse=True), caps
        # Settles: the last several ticks are one repeated fixed point.
        assert len(set(cars[-5:])) == 1, cars
        assert len(set(caps[-5:])) == 1, caps
        assert cars[-1] > 8500.0, "the car ends up with nearly all of it"


class TestPriceGateStrictness:
    """The 2026-08-24 incident class: a dead price sensor coerced through float(0)
    reads as a legitimate 0.0. The tiers' gates are STRICTLY less-than so that
    exactly-zero — the dead-sensor signature — never satisfies 'only negative'."""

    def _cfg_assist(self, **kw):
        base = dict(
            enabled=True,
            battery_assist_enabled=True,
            battery_assist_max_price_sek=0.0,
            battery_assist_min_remaining_solar_kwh=8.0,
            battery_assist_floor_soc=30.0,
            battery_assist_allowance_w=6000.0,
        )
        base.update(kw)
        return EVSurplusConfig(**base)

    def test_exact_zero_does_not_open_battery_assist(self):
        # price == max_price_sek == 0.0 must NOT engage (documented '0 = only negative').
        out = compute_ev_surplus(
            _inputs(import_price_sek=0.0, remaining_solar_kwh=10.0,
                    battery_soc_percent=80.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on is False

    def test_truly_negative_still_opens_battery_assist(self):
        out = compute_ev_surplus(
            _inputs(import_price_sek=-0.01, remaining_solar_kwh=10.0,
                    battery_soc_percent=80.0, grid_w=0.0),
            self._cfg_assist(),
        )
        assert out[0].switch_on and out[0].target_power_w > 0

    def test_exact_threshold_does_not_open_cheap_grid(self):
        cfg = _cfg(cheap_grid_price_sek=0.30, cheap_grid_allowance_w=6000.0)
        out = compute_ev_surplus(_inputs(import_price_sek=0.30, grid_w=0.0), cfg)
        assert out[0].switch_on is False

    def test_below_threshold_opens_cheap_grid(self):
        cfg = _cfg(cheap_grid_price_sek=0.30, cheap_grid_allowance_w=6000.0)
        out = compute_ev_surplus(_inputs(import_price_sek=0.29, grid_w=0.0), cfg)
        assert out[0].switch_on and out[0].target_power_w > 0


class TestCapIdleCarRegression:
    """Live 2026-08-24: the cap fired 353 times overnight and flapped the inverter's
    charge limit 9500<->100 W every ~2 min, with a Tesla merely parked on the charger
    and the servo feeding it nothing. `demand` ("plugged and below its ceiling") is
    true all night; the missing condition is that the servo actually ALLOCATED power."""

    def _cap(self, *, battery_w, car_w, grid_w, plan_w=0.0, price=1.0):
        from executor.ev_surplus import EVSurplusTick, ev_priority_battery_cap_w

        cfg = _cfg(
            ev_priority_battery_cap_enabled=True,
            battery_yield_soc=95.0, battery_capacity_kwh=16.0, battery_fill_margin_kwh=3.0,
        )
        inputs = _inputs(
            battery_w=battery_w, grid_w=grid_w, battery_soc_percent=80.0,
            remaining_solar_kwh=30.0, import_price_sek=price,
            chargers=[_charger(current_power_w=car_w)], plan_battery_charge_w=plan_w,
        )
        tick = EVSurplusTick()
        compute_ev_surplus(inputs, cfg, tick_out=tick)
        return ev_priority_battery_cap_w(inputs, cfg, tick, plan_w), tick

    def test_idle_night_plugged_car_does_not_cap(self):
        # The overnight signature: importing, battery flat, expensive price => the
        # servo allocates nothing. The battery must be left alone.
        cap, tick = self._cap(battery_w=0.0, car_w=0.0, grid_w=1465.0)
        assert tick.demand is True  # the car IS plugged and below its ceiling...
        assert tick.target_total_w == 0.0  # ...but gets nothing
        assert cap is None  # ...so no claim on the battery

    def test_battery_hovering_at_zero_does_not_flap(self):
        # Sign-only discharge test made None<->value alternate tick to tick around
        # 0 W; each flip is an inverter register write. This must exercise the
        # DISCHARGE guard, so the car has to be genuinely commanded on — otherwise
        # the commanded-allocation gate returns None first and the assertion is
        # vacuous (it was, before adversarial review caught it).
        caps = []
        for battery_w in (-99.0, -50.0, -1.0, 0.0, 1.0, 50.0, 99.0):
            cap, tick = self._cap(battery_w=battery_w, car_w=6000.0, grid_w=-3000.0)
            assert tick.commanded_on_total_w > 0.0, "the guard under test must be reached"
            caps.append(cap)
        # Every one of them resolves the same way: no None<->value alternation across
        # the zero crossing, so no register write is provoked by meter noise.
        assert all(c is not None for c in caps), caps
        assert max(caps) - min(caps) < 200.0, caps  # inside the runtime's hysteresis

    def test_clear_discharge_below_the_deadband_hands_off(self):
        cap, tick = self._cap(battery_w=-500.0, car_w=6000.0, grid_w=-3000.0)
        assert tick.commanded_on_total_w > 0.0
        assert cap is None

    def test_clear_discharge_still_hands_off(self):
        cap, _ = self._cap(battery_w=-3000.0, car_w=6000.0, grid_w=0.0)
        assert cap is None

    def test_real_surplus_still_caps(self):
        # The feature must still work: sun, battery absorbing, car commanded ON.
        cap, tick = self._cap(battery_w=7000.0, car_w=3000.0, grid_w=0.0)
        assert tick.commanded_on_total_w > 0.0
        assert cap == pytest.approx(3790.0)

    def test_surplus_below_the_start_threshold_leaves_the_battery_alone(self):
        # The daylight twin of the overnight bug (found in adversarial review): the
        # cold-start kick reports the FULL headroom as target_total_w even when the
        # car cannot start on it, so a target-derived gate pinned the battery to
        # 100 W across the whole morning ramp while the PV exported. It is a stable
        # fixed point, not a transient: capping the battery does not change the
        # car's start decision, because headroom counts the battery's inflow either way.
        for surplus in (500.0, 1500.0, 2500.0, 3500.0, 4000.0):
            cap, tick = self._cap(battery_w=surplus, car_w=0.0, grid_w=0.0)
            assert tick.commanded_on_total_w == 0.0, f"car should not start on {surplus} W"
            assert cap is None, f"surplus {surplus} W: battery must stay free, got {cap}"

    def test_jitter_across_the_deadband_does_not_flap_the_cap(self):
        # Meter noise around the 250 W deadband made the cap alternate None<->0,
        # i.e. an inverter register write per tick. With no car commanded on, every
        # tick is None regardless of which side of the band the noise lands.
        caps = [
            self._cap(battery_w=w, car_w=0.0, grid_w=0.0)[0]
            for w in (240.0, 265.0, 235.0, 270.0, 245.0, 260.0, 238.0, 262.0)
        ]
        assert caps == [None] * 8, caps

    def test_plan_floor_survives_the_new_gate(self):
        # No allocation to cars => None, even with a plan floor: the controller's own
        # 9500 W write already implements "battery takes what it wants".
        cap, _ = self._cap(battery_w=0.0, car_w=0.0, grid_w=1465.0, plan_w=4000.0)
        assert cap is None
