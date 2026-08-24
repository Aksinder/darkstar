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


class TestSurplusBoostHold:
    """Live 2026-08-24 18:29-18:36: the spa's TRUE surplus (export + battery charge +
    its own draw) went 3589 W -> -63 -> -102 -> -186 -> -251 -> -287 -> 3086 W as other
    loads came and went at sunset, and the boost followed it tick for tick — writing
    the appliance's mode every minute or two. The measurement was right; the decision
    just had no anti-short-cycle hold. (The relay path's min-on dwell cannot cover this:
    it sits behind a switch./input_boolean. branch and a hardcoded 50 C on-threshold,
    so a 20-40 C helper-driven appliance never had any.)"""

    def _engine(self, hold_minutes=15.0):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from executor.engine import ExecutorEngine

        eng = ExecutorEngine.__new__(ExecutorEngine)
        eng._water_boost_state = {}
        eng._water_boost_hold_until = {}
        eng.config = MagicMock()
        eng.config.water_heater.temp_boost = 40
        eng._heater_price_ceiling = lambda device, ctx: 0.9
        return eng, SimpleNamespace(
            id="spa", surplus_boost=True, temp_boost=40, power_kw=1.8,
            absorb_cap_kwh_per_day=8.0, surplus_boost_min_minutes=hold_minutes,
        )

    def _ctx(self, grid_w, battery_w, price=0.3):
        return {
            "vacation": False, "grid_w": grid_w, "battery_w": battery_w,
            "import_price": price, "export_price": price,
        }

    def test_the_live_sunset_sequence_no_longer_flaps(self):
        eng, dev = self._engine()
        # (grid_w, battery_w, own_draw) straight from the incident.
        seq = [(-1789, 0, 1800), (201, -1662, 1800), (-29, -1931, 1800),
               (59, -1927, 1800), (-73, -2124, 1800), (13, -2074, 1800)]
        out = [eng._water_surplus_boost(dev, own, self._ctx(g, b), 0.0)
               for g, b, own in seq]
        assert out[0] == 40, "real surplus starts the boost"
        assert all(v == 40 for v in out), f"boost must hold through the dip: {out}"

    def test_without_the_hold_it_flaps(self):
        eng, dev = self._engine(hold_minutes=0.0)
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0), 0.0) == 40
        assert eng._water_surplus_boost(dev, 1800, self._ctx(201, -1662), 0.0) is None

    def test_price_ceiling_wins_instantly_over_the_hold(self):
        # The hold rides out NOISE, never an expensive hour.
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        # Surplus is still there, but the price left the ceiling: refuse on this tick.
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 2.5), 0.0) is None

    def test_price_refusal_during_a_dip_is_not_held(self):
        # Both conditions at once: surplus gone AND price high. Price must win.
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        assert eng._water_surplus_boost(dev, 1800, self._ctx(201, -1662, 2.5), 0.0) is None

    def test_vacation_wins_instantly_over_the_hold(self):
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        ctx = self._ctx(-1789, 0, 0.3)
        ctx["vacation"] = True
        assert eng._water_surplus_boost(dev, 1800, ctx, 0.0) is None

    def test_daily_cap_wins_instantly_over_the_hold(self):
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        # heated_today past absorb_cap (8.0): refuse now; the hold must not override it.
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 8.5) is None

    def test_hold_expires(self):
        import time as _t
        eng, dev = self._engine(hold_minutes=15.0)
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0), 0.3) == 40
        # Expire the hold by hand (monotonic clock, no sleeping in tests).
        eng._water_boost_hold_until["spa"] = _t.monotonic() - 1.0
        assert eng._water_surplus_boost(dev, 1800, self._ctx(201, -1662), 0.3) is None
        assert "spa" not in eng._water_boost_hold_until, "expired hold is cleared"

    def test_hold_respects_the_import_price_during_a_dip(self):
        # With the surplus gone the held kWh is genuinely BOUGHT, so the import price
        # is what the ceiling must judge — export price is irrelevant when nothing is
        # being exported. (should_boost_on_surplus checks surplus first and would
        # otherwise mask the price refusal behind "no surplus".)
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        ctx = self._ctx(201, -1662, 0.3)
        ctx["import_price"] = 2.5  # dear to buy, even though export was cheap
        assert eng._water_surplus_boost(dev, 1800, ctx, 0.0) is None

    def test_hold_does_not_fire_on_unknown_price(self):
        eng, dev = self._engine()
        assert eng._water_surplus_boost(dev, 1800, self._ctx(-1789, 0, 0.3), 0.0) == 40
        ctx = self._ctx(201, -1662, 0.3)
        ctx["import_price"] = None
        assert eng._water_surplus_boost(dev, 1800, ctx, 0.0) is None
