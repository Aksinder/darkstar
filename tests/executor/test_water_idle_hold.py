"""Idle-hold for self-thermostatted water heaters (the spa)."""

from __future__ import annotations

from typing import ClassVar

from executor.water_hold import (
    battery_charge_w,
    price_percentile,
    should_hold_off_write,
)

SPA_W = 1800.0


def _hold(**kw):
    """Default scene: night, no sun, spa idle, cheap. Override one thing per test."""
    base = {
        "power_w": 0.0,
        "grid_w": 300.0,
        "battery_w": 0.0,
        "import_price_sek_kwh": 0.30,
        "export_price_sek_kwh": 0.20,
        "heater_power_w": SPA_W,
        "max_price_sek_kwh": 0.9,
    }
    base.update(kw)
    return should_hold_off_write(**base)


class TestOwnerRule:
    def test_idle_and_cheap_holds(self):
        """No need to turn it off when it isn't drawing anything."""
        hold, reason = _hold()
        assert hold is True
        assert "idle" in reason

    def test_pump_only_counts_as_idle(self):
        assert _hold(power_w=60.0)[0] is True

    def test_drawing_without_surplus_writes_off(self):
        hold, reason = _hold(power_w=SPA_W, grid_w=2400.0)
        assert hold is False
        assert "without surplus" in reason

    def test_drawing_on_export_holds(self):
        hold, reason = _hold(power_w=SPA_W, grid_w=-2500.0)
        assert hold is True
        assert "surplus" in reason

    def test_high_price_forces_off_even_when_idle(self):
        """The 'högt elpris' half of the rule outranks idleness."""
        hold, reason = _hold(import_price_sek_kwh=2.40)
        assert hold is False
        assert "2.40" in reason

    def test_price_at_ceiling_still_holds(self):
        assert _hold(import_price_sek_kwh=0.9, max_price_sek_kwh=0.9)[0] is True


class TestSurplusIsNotExport:
    """The morning trap: meter ~0 while the battery soaks 8 kW of PV."""

    def test_battery_soaking_pv_counts_as_surplus(self):
        hold, reason = _hold(power_w=SPA_W, grid_w=13.0, battery_w=8000.0)
        assert hold is True
        assert "surplus" in reason

    def test_battery_charge_smaller_than_the_heater_is_not_surplus(self):
        """Only spare capacity the spa could actually eat counts (heater idle)."""
        reason = _hold(power_w=0.0, grid_w=13.0, battery_w=900.0)[1]
        assert "surplus" not in reason  # held for idleness, not for spare energy

    def test_battery_discharging_is_never_surplus(self):
        """Settled before the own-draw credit: the marginal kWh comes from storage."""
        assert _hold(power_w=SPA_W, grid_w=13.0, battery_w=-3000.0)[0] is False

    def test_meter_noise_near_zero_is_not_export(self):
        """With the heater IDLE, -50 W really is noise and not spare energy."""
        reason = _hold(power_w=0.0, grid_w=-50.0)[1]
        assert "surplus" not in reason


class TestOwnDrawIsNotScarcity:
    """The meter reads ~0 BECAUSE the element is on. Counting a heater's own draw
    against itself makes the surplus vanish the instant it starts — the same blindness
    that made the EV servo switch off the load it had just created."""

    def test_a_heater_eating_the_export_still_sees_surplus(self):
        hold, reason = _hold(power_w=SPA_W, grid_w=-50.0)
        assert hold is True
        assert "surplus" in reason

    def test_the_credit_is_only_its_own_draw(self):
        """2.4 kW of import is more than the spa can account for."""
        assert _hold(power_w=SPA_W, grid_w=2400.0)[0] is False

    def test_the_credit_does_not_survive_a_discharging_battery(self):
        assert _hold(power_w=SPA_W, grid_w=-50.0, battery_w=-3000.0)[0] is False


class TestEffectivePrice:
    """Spare PV costs the export revenue foregone, not the import price."""

    def test_sunny_noon_holds_despite_a_high_import_price(self):
        hold, _ = _hold(
            power_w=SPA_W, battery_w=8000.0,
            import_price_sek_kwh=1.20, export_price_sek_kwh=0.40,
        )
        assert hold is True

    def test_expensive_export_still_forces_off(self):
        """Winter: selling at 2.10 beats keeping the spa warm."""
        hold, reason = _hold(
            power_w=SPA_W, grid_w=-4000.0,
            import_price_sek_kwh=2.60, export_price_sek_kwh=2.10,
        )
        assert hold is False
        assert "2.10" in reason

    def test_import_price_applies_without_surplus(self):
        """A cheap export price must not excuse buying at 2.40."""
        assert _hold(import_price_sek_kwh=2.40, export_price_sek_kwh=0.10)[0] is False

    def test_missing_export_price_falls_back_to_import(self):
        assert _hold(
            power_w=SPA_W, battery_w=8000.0,
            import_price_sek_kwh=2.40, export_price_sek_kwh=None,
        )[0] is False


class TestFailsClosed:
    def test_unknown_price_with_ceiling_writes_off(self):
        hold, reason = _hold(import_price_sek_kwh=None, export_price_sek_kwh=None)
        assert hold is False
        assert "unknown" in reason

    def test_unknown_price_without_ceiling_is_fine(self):
        assert _hold(import_price_sek_kwh=None, max_price_sek_kwh=None)[0] is True

    def test_unreadable_power_writes_off(self):
        hold, reason = _hold(power_w=None)
        assert hold is False
        assert "unreadable" in reason

    def test_unreadable_power_is_moot_under_surplus(self):
        """Surplus decides on its own; the idle test is never reached."""
        assert _hold(power_w=None, grid_w=-3000.0)[0] is True

    def test_unreadable_grid_and_battery_is_no_surplus(self):
        assert _hold(power_w=SPA_W, grid_w=None, battery_w=None)[0] is False


class TestBatterySignConvention:
    """Two conventions coexist and disagree in sign — reading the wrong one inverts
    the whole surplus test. Live values at one instant: the servo's
    sensor.battery_charging_power_signed read +4797 while charging, and
    input_sensors' sensor.battery_power read -4797."""

    def test_servo_entity_is_already_charge_positive(self):
        assert battery_charge_w(servo_signed_w=4797.0, house_signed_w=-4797.0) == 4797.0

    def test_house_sensor_is_flipped(self):
        assert battery_charge_w(servo_signed_w=None, house_signed_w=-4797.0) == 4797.0

    def test_house_discharge_stays_negative(self):
        assert battery_charge_w(servo_signed_w=None, house_signed_w=3000.0) == -3000.0

    def test_both_unreadable(self):
        assert battery_charge_w(servo_signed_w=None, house_signed_w=None) is None

    def test_the_morning_case_end_to_end(self):
        """PV 9.8 kW, meter +303 W (importing!), battery charging: still surplus."""
        charge_w = battery_charge_w(servo_signed_w=None, house_signed_w=-4797.0)
        hold, reason = _hold(power_w=SPA_W, grid_w=303.0, battery_w=charge_w)
        assert hold is True
        assert "surplus" in reason

    def test_the_same_reading_misread_would_have_failed(self):
        """Guards the regression: the raw house value inverts the test."""
        assert _hold(power_w=SPA_W, grid_w=303.0, battery_w=-4797.0)[0] is False


class TestDynamicCeiling:
    """A fixed ceiling tuned in a cheap week stops working when the market moves.

    Owner, 2026-08-18: "spatak borde vara dynamiskt mot period och inte per dygn."
    Live proof of the failure: 0.9 SEK/kWh permitted heating through the summer, then
    sat below EVERY hour once spot reached 1.43 (export 1.36-2.44, import 2.42-3.78) —
    so the spa quietly went cold and all three of its features stood down. A percentile
    keeps the same intent, "only the cheap part of what is coming", at any price level.
    """

    CHEAP_WEEK: ClassVar[list[float]] = [0.20, 0.30, 0.45, 0.60, 0.90, 1.20]
    EXPENSIVE_WEEK: ClassVar[list[float]] = [2.42, 2.60, 2.90, 3.10, 3.40, 3.78]

    def test_the_ceiling_tracks_a_cheap_window(self):
        cap = price_percentile(self.CHEAP_WEEK, 30.0)
        assert 0.30 <= cap <= 0.60

    def test_the_same_percentile_rises_with_the_market(self):
        cap = price_percentile(self.EXPENSIVE_WEEK, 30.0)
        assert 2.6 <= cap <= 2.95, "must follow the market, not the old absolute level"

    def test_a_percentile_never_permits_the_dearest_hours(self):
        for window in (self.CHEAP_WEEK, self.EXPENSIVE_WEEK):
            assert price_percentile(window, 30.0) < max(window)

    def test_p100_is_the_top_and_p0_the_floor(self):
        assert price_percentile(self.CHEAP_WEEK, 100.0) == max(self.CHEAP_WEEK)
        assert price_percentile(self.CHEAP_WEEK, 0.0) == min(self.CHEAP_WEEK)

    def test_an_empty_window_yields_no_ceiling(self):
        """Callers must fall back to the absolute value, not drop the ceiling."""
        assert price_percentile([], 30.0) is None

    def test_a_single_price_is_its_own_percentile(self):
        assert price_percentile([1.43], 30.0) == 1.43

    def test_the_expensive_week_would_have_blocked_a_static_ceiling(self):
        """The regression this exists to prevent, stated as a test."""
        static = 0.9
        assert all(p > static for p in self.EXPENSIVE_WEEK)
        assert price_percentile(self.EXPENSIVE_WEEK, 30.0) > static


class TestWindowLength:
    """Owner, 2026-08-18: "för spa skulle vi behöva längre fönster än 24h".

    A 24 h window still lets a comfort load heat during a uniformly expensive day — it
    just picks that day's cheapest hours. Waiting out the whole expensive stretch needs
    a longer span. The hard ceiling is the price feed: Nordpool publishes today and
    tomorrow and nothing older, so the forward series runs out after ~12-36 h and the
    remainder can only come from hours already passed.
    """

    def test_a_longer_window_dilutes_one_expensive_day(self):
        """The same P30 permits nothing on the dear day once cheap history is included."""
        dear_day = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
        cheap_history = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        short = price_percentile(dear_day, 30.0)
        long = price_percentile(cheap_history + dear_day, 30.0)
        assert short > 3.0, "a day-long window sets the bar at that day's own level"
        assert long < 1.0, "a longer window remembers the stretch was expensive"

    def test_a_longer_window_also_lifts_the_bar_after_a_dear_stretch(self):
        """Symmetric: the same mechanism must not starve a load after cheap days end."""
        cheap_day = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        dear_history = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
        assert price_percentile(dear_history + cheap_day, 30.0) > price_percentile(
            cheap_day, 30.0
        )
