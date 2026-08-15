"""Push a self-thermostatted heater to its boost target on measured surplus.

Owner, 2026-08-15: "spa borde tryckas upp till 40 om det finns överskott och elpriset
inte är dyrt." The planner cannot answer this — its boost path needs excess_pv_sink ==
"water_heater_boost", deliberately off since it was the mechanism behind the 36 kWh/day
phantom-water incident. So the decision is made from MEASURED surplus each tick, which
is both truer to the request and self-limiting: the boost lasts as long as the surplus.
"""

from __future__ import annotations

from executor.water_hold import should_boost_on_surplus

SPA_W = 1800.0


def _boost(**kw):
    """Default scene: sunny, exporting, cheap, well under the daily cap."""
    base = {
        "power_w": 0.0,
        "grid_w": -3000.0,
        "battery_w": 0.0,
        "import_price_sek_kwh": 1.20,
        "export_price_sek_kwh": 0.30,
        "heater_power_w": SPA_W,
        "max_price_sek_kwh": 0.9,
        "heated_today_kwh": 2.0,
        "absorb_cap_kwh_per_day": 8.0,
    }
    base.update(kw)
    return should_boost_on_surplus(**base)


class TestOwnerRule:
    def test_surplus_and_cheap_boosts(self):
        boost, reason = _boost()
        assert boost is True
        assert "surplus" in reason

    def test_no_surplus_no_boost(self):
        boost, reason = _boost(grid_w=1500.0)
        assert boost is False
        assert "no surplus" in reason

    def test_expensive_surplus_does_not_boost(self):
        """Winter: selling at 2.10 beats a warmer spa."""
        boost, reason = _boost(export_price_sek_kwh=2.10)
        assert boost is False
        assert "2.10" in reason

    def test_stored_pv_counts_as_surplus(self):
        """Meter ~0 while the battery soaks 8 kW — the sunny-morning case."""
        assert _boost(grid_w=13.0, battery_w=8000.0)[0] is True

    def test_discharging_battery_never_boosts(self):
        assert _boost(grid_w=-3000.0, battery_w=-4000.0)[0] is False


class TestItKeepsItselfGoing:
    """Once heating, the heater's own draw must not read as the surplus running out."""

    def test_boost_survives_eating_its_own_export(self):
        boost, _ = _boost(power_w=SPA_W, grid_w=-50.0)
        assert boost is True, "would flap between 38 and 40 every tick"


class TestDailyBound:
    """This path commands heat, so it carries the bound the planner would have."""

    def test_daily_cap_stops_the_boost(self):
        boost, reason = _boost(heated_today_kwh=8.0)
        assert boost is False
        assert "daily cap" in reason

    def test_just_under_the_cap_still_boosts(self):
        assert _boost(heated_today_kwh=7.9)[0] is True

    def test_no_cap_configured_means_no_bound(self):
        assert _boost(heated_today_kwh=99.0, absorb_cap_kwh_per_day=None)[0] is True

    def test_unknown_consumption_does_not_block(self):
        assert _boost(heated_today_kwh=None)[0] is True


class TestFailsClosed:
    def test_unknown_price_with_a_ceiling_does_not_boost(self):
        boost, reason = _boost(
            import_price_sek_kwh=None, export_price_sek_kwh=None
        )
        assert boost is False
        assert "unknown" in reason

    def test_missing_export_price_falls_back_to_import(self):
        assert _boost(export_price_sek_kwh=None, import_price_sek_kwh=1.20)[0] is False

    def test_unreadable_grid_and_battery_is_no_surplus(self):
        assert _boost(grid_w=None, battery_w=None)[0] is False
