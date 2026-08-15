"""Idle-hold for self-thermostatted water heaters (the spa)."""

from __future__ import annotations

from executor.water_hold import should_hold_off_write

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
        """Only spare capacity the spa could actually eat counts."""
        assert _hold(power_w=SPA_W, grid_w=13.0, battery_w=900.0)[0] is False

    def test_battery_discharging_is_never_surplus(self):
        assert _hold(power_w=SPA_W, grid_w=13.0, battery_w=-3000.0)[0] is False

    def test_meter_noise_near_zero_is_not_export(self):
        assert _hold(power_w=SPA_W, grid_w=-50.0)[0] is False


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
