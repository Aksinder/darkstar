"""The battery-yield gate as a DEADLINE, not an SoC threshold.

Owner, 2026-08-15: "vi ska ladda Teslan om solen skiner före batteriet om vi räknar
med att hinna ladda upp batteriet senare ... Teslan kan försvinna, det kan inte
batteriet." A car that drives off with the surplus unspent has lost it for good; the
battery can still be filled from later sun. So while the forecast shows slack, the
cars go first, and the battery claims its own inflow only at the latest safe moment.
"""

from __future__ import annotations

from executor.ev_surplus import (
    EVSurplusConfig,
    EVSurplusInputs,
    battery_fill_slack_kwh,
    battery_reserve_active,
    compute_ev_surplus,
)
from tests.executor.test_ev_surplus_priority import _tesla

# 16 kWh battery, reserve target 95%, 3 kWh of slack demanded.
CFG = EVSurplusConfig(
    enabled=True,
    battery_yield_soc=95.0,
    battery_capacity_kwh=16.0,
    battery_charge_efficiency=0.95,
    battery_fill_margin_kwh=3.0,
    battery_fill_margin_hysteresis_kwh=2.0,
)


def _inputs(soc, remaining_solar, *, prev=False, battery_w=7000.0, chargers=None):
    return EVSurplusInputs(
        pv_w=9000.0, grid_w=-800.0, battery_w=battery_w,
        battery_soc_percent=soc, import_price_sek=0.14,
        remaining_solar_kwh=remaining_solar,
        battery_reserve_active_prev=prev,
        chargers=chargers if chargers is not None else [],
        plan_battery_charge_w=4000.0,
    )


class TestSlack:
    def test_need_is_soc_deficit_over_efficiency(self):
        # 95 - 45 = 50% of 16 kWh = 8 kWh, / 0.95 = 8.42 needed; 20 forecast.
        slack = battery_fill_slack_kwh(_inputs(45.0, 20.0), CFG)
        assert round(slack, 2) == round(20.0 - 8.0 / 0.95, 2)

    def test_battery_already_full_has_infinite_slack(self):
        assert battery_fill_slack_kwh(_inputs(96.0, 0.0), CFG) == float("inf")

    def test_unknown_capacity_is_unanswerable(self):
        assert battery_fill_slack_kwh(_inputs(45.0, 20.0), EVSurplusConfig(enabled=True)) is None


class TestReserve:
    def test_sunny_morning_lets_the_car_go_first(self):
        """SoC 50, 20 kWh still forecast — plenty of time to fill later."""
        assert battery_reserve_active(_inputs(50.0, 20.0), CFG) is False

    def test_late_afternoon_claims_the_inflow(self):
        """Same SoC, only 6 kWh left: 6 - 8.42 is negative, start filling NOW."""
        assert battery_reserve_active(_inputs(50.0, 6.0), CFG) is True

    def test_the_margin_is_what_separates_them(self):
        """Just enough to fill but no slack for house load — battery wins."""
        need = 8.0 / 0.95
        assert battery_reserve_active(_inputs(50.0, need + 0.5), CFG) is True
        assert battery_reserve_active(_inputs(50.0, need + 3.5), CFG) is False

    def test_full_battery_never_reserves(self):
        assert battery_reserve_active(_inputs(95.0, 0.0), CFG) is False

    def test_no_capacity_configured_falls_back_to_battery_first(self):
        """Unanswerable must not silently hand the surplus to the cars."""
        legacy = EVSurplusConfig(enabled=True, battery_yield_soc=95.0)
        assert battery_reserve_active(_inputs(50.0, 20.0), legacy) is True


class TestHysteresis:
    def test_release_needs_more_slack_than_engaging(self):
        need = 8.0 / 0.95
        borderline = _inputs(50.0, need + 4.0)          # slack 4.0
        assert battery_reserve_active(borderline, CFG) is False
        engaged = _inputs(50.0, need + 4.0, prev=True)  # needs 3 + 2 = 5
        assert battery_reserve_active(engaged, CFG) is True

    def test_clear_slack_releases_even_when_engaged(self):
        need = 8.0 / 0.95
        assert battery_reserve_active(_inputs(50.0, need + 9.0, prev=True), CFG) is False


class TestEndToEnd:
    def _tesla_at(self, soc):
        return _tesla(soc_percent=soc, deadline_hours=None)

    def test_car_gets_the_battery_inflow_while_slack_lasts(self):
        cmds = compute_ev_surplus(
            _inputs(50.0, 20.0, chargers=[self._tesla_at(40.0)]), CFG
        )
        assert cmds[0].switch_on, "sunny morning: the car should charge"

    def test_car_drops_out_when_the_battery_must_start(self):
        """Identical instant except the forecast — only ~800 W of export is left."""
        cmds = compute_ev_surplus(
            _inputs(50.0, 6.0, chargers=[self._tesla_at(40.0)]), CFG
        )
        assert not cmds[0].switch_on, "late afternoon: the battery claims its inflow"

    def test_a_discharging_battery_still_backs_the_cars_off(self):
        """Protective and unconditional — unchanged by the deadline logic."""
        cmds = compute_ev_surplus(
            _inputs(50.0, 20.0, battery_w=-3000.0, chargers=[self._tesla_at(40.0)]), CFG
        )
        assert not cmds[0].switch_on


class TestReserveAndTheBatteryCap:
    """The EV-priority battery cap (2026-08-23) reuses THIS gate as its 'battery
    wins' verdict — one rule, one hysteresis. Pinned here so the two can never
    disagree: whenever the reserve claims the inflow, the cap steps aside."""

    def _cap(self, soc, remaining, prev=False):
        from dataclasses import replace

        from executor.ev_surplus import (
            EVSurplusTick,
            compute_ev_surplus,
            ev_priority_battery_cap_w,
        )

        cfg = replace(CFG, ev_priority_battery_cap_enabled=True)
        inputs = _inputs(soc, remaining, prev=prev, chargers=[_tesla(current_power_w=3000.0)])
        tick = EVSurplusTick()
        compute_ev_surplus(inputs, cfg, tick_out=tick)
        return ev_priority_battery_cap_w(inputs, cfg, tick, 4000.0), tick

    def test_sunny_morning_caps_the_battery_for_the_car(self):
        cap, tick = self._cap(soc=50.0, remaining=20.0)
        assert not tick.reserve_active
        assert cap is not None

    def test_late_afternoon_reserve_releases_the_battery(self):
        cap, tick = self._cap(soc=50.0, remaining=6.0)
        assert tick.reserve_active
        assert cap is None

    def test_the_cap_follows_the_gates_hysteresis(self):
        """Engaged at 3 kWh slack, released only at 5: a cap that used its own
        threshold would flap against the gate in the 3-5 kWh band."""
        cap_engaged, tick = self._cap(soc=50.0, remaining=12.5, prev=True)
        assert tick.reserve_active and cap_engaged is None
        cap_released, tick2 = self._cap(soc=50.0, remaining=12.5, prev=False)
        assert not tick2.reserve_active and cap_released is not None
