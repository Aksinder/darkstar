"""Idle-hold for self-thermostatted water heaters (the spa)."""

from __future__ import annotations

from executor.water_hold import should_hold_off_write


def _hold(**kw):
    base = {
        "power_w": 0.0,
        "grid_w": -3000.0,
        "price_sek_kwh": 0.30,
        "max_price_sek_kwh": 0.9,
    }
    base.update(kw)
    return should_hold_off_write(**base)


def test_idle_and_cheap_holds():
    """Owner rule: no need to turn it off when it isn't drawing anything."""
    hold, reason = _hold(power_w=0.0)
    assert hold is True
    assert "idle" in reason


def test_pump_only_counts_as_idle():
    hold, _ = _hold(power_w=60.0)
    assert hold is True


def test_drawing_on_export_holds():
    """Heating on surplus is exactly what we want to preserve."""
    hold, reason = _hold(power_w=1800.0, grid_w=-2500.0)
    assert hold is True
    assert "surplus" in reason


def test_drawing_without_surplus_writes_off():
    hold, reason = _hold(power_w=1800.0, grid_w=+1200.0)
    assert hold is False
    assert "without surplus" in reason


def test_meter_noise_near_zero_is_not_surplus():
    hold, _ = _hold(power_w=1800.0, grid_w=-50.0)
    assert hold is False


def test_high_price_forces_off_even_when_idle():
    """The 'högt elpris' half of the rule outranks idleness."""
    hold, reason = _hold(power_w=0.0, price_sek_kwh=2.40)
    assert hold is False
    assert "2.40" in reason


def test_high_price_forces_off_even_on_surplus():
    hold, _ = _hold(power_w=1800.0, grid_w=-4000.0, price_sek_kwh=2.40)
    assert hold is False


def test_price_at_ceiling_still_holds():
    hold, _ = _hold(power_w=0.0, price_sek_kwh=0.9, max_price_sek_kwh=0.9)
    assert hold is True


def test_unknown_price_with_ceiling_fails_closed():
    hold, reason = _hold(power_w=0.0, price_sek_kwh=None)
    assert hold is False
    assert "unknown" in reason


def test_unknown_price_without_ceiling_is_fine():
    hold, _ = _hold(power_w=0.0, price_sek_kwh=None, max_price_sek_kwh=None)
    assert hold is True


def test_unreadable_power_fails_closed():
    hold, reason = _hold(power_w=None)
    assert hold is False
    assert "unreadable" in reason


def test_unreadable_grid_fails_closed_only_when_drawing():
    """Blind on grid is harmless while idle; only the surplus test needs it."""
    assert _hold(power_w=0.0, grid_w=None)[0] is True
    assert _hold(power_w=1800.0, grid_w=None)[0] is False
