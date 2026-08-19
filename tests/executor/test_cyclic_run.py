"""Opportunistic run gates for the pool pump and the bog filter.

Owner, 2026-08-19, on the bog filter: "6 timmar, far ocksa kora vid overskott om
det inte ar extra dyra timmar samt nar vi ar hemma om utrymme finns."
"""

from __future__ import annotations

from executor.cyclic_run import anyone_home, should_run_opportunistically

FILTER_W = 390.0
# A day of prices, cheap night to dear evening. P80 = 2.16, P50 = 1.10.
WINDOW: list[float] = [0.20, 0.40, 0.60, 0.90, 1.30, 1.80, 2.40, 3.60]


def _run(**kw):
    """Default scene: no surplus, nobody home, mid price, budget untouched."""
    base = {
        "plan_wants_on": False,
        "power_w": 0.0,
        "grid_w": 300.0,
        "battery_w": 0.0,
        "load_power_w": FILTER_W,
        "import_price_sek_kwh": 0.60,
        "export_price_sek_kwh": 0.40,
        "price_window": WINDOW,
        "surplus_run": True,
        "max_price_percentile": 80.0,
        "presence_home": False,
        "presence_max_price_percentile": 50.0,
        "extra_hours_today": 0.0,
        "max_extra_hours_per_day": 4.0,
    }
    base.update(kw)
    return should_run_opportunistically(**base)


class TestSurplusGate:
    """"far ocksa kora vid overskott" -- measured, not forecast."""

    def test_export_runs_it(self):
        run, why = _run(grid_w=-2000.0)
        assert run is True
        assert "surplus" in why

    def test_battery_soaking_pv_counts_as_surplus(self):
        """The morning trap: meter ~0 while the battery eats 8 kW."""
        assert _run(grid_w=13.0, battery_w=8000.0)[0] is True

    def test_no_surplus_and_nobody_home_stays_off(self):
        run, why = _run()
        assert run is False
        assert "no surplus" in why

    def test_its_own_draw_is_not_scarcity(self):
        """Once running, the meter reads ~0 BECAUSE the pump is on."""
        assert _run(power_w=FILTER_W, grid_w=-50.0)[0] is True

    def test_a_discharging_battery_is_never_surplus(self):
        assert _run(grid_w=-50.0, battery_w=-3000.0)[0] is False


class TestNotTheDearestHours:
    """"om det inte ar extra dyra timmar" -- a HIGH ceiling: block only the tail."""

    def test_surplus_at_a_dear_hour_stands_down(self):
        """Winter export at 3.60: selling beats filtering."""
        run, why = _run(
            grid_w=-4000.0, import_price_sek_kwh=3.80, export_price_sek_kwh=3.60
        )
        assert run is False
        assert "3.60" in why

    def test_an_ordinary_hour_is_not_extra_dear(self):
        """1.30 sits below P80 -- 'inte extra dyra' means the tail, not the middle."""
        assert _run(grid_w=-4000.0, export_price_sek_kwh=1.30)[0] is True

    def test_the_ceiling_follows_the_market(self):
        """The same P80 permits more in an expensive week -- intent, not a level."""
        dear = [2.4, 2.6, 2.9, 3.1, 3.4, 3.8]
        assert _run(grid_w=-4000.0, export_price_sek_kwh=2.90, price_window=dear)[0] is True
        assert _run(grid_w=-4000.0, export_price_sek_kwh=2.90)[0] is False

    def test_surplus_uses_the_export_price_not_import(self):
        """Spare PV costs the revenue foregone; a 3.80 import must not block it."""
        assert _run(grid_w=-4000.0, import_price_sek_kwh=3.80,
                    export_price_sek_kwh=0.40)[0] is True


class TestPresenceGate:
    """"samt nar vi ar hemma om utrymme finns" -- tighter ceiling than surplus."""

    def test_home_and_cheap_runs_it(self):
        run, why = _run(presence_home=True)
        assert run is True
        assert "home" in why

    def test_home_at_a_middling_price_does_not(self):
        """1.80 clears the surplus P80 but not the presence P50: company is not free."""
        assert _run(presence_home=True, import_price_sek_kwh=1.80)[0] is False
        assert _run(grid_w=-4000.0, export_price_sek_kwh=1.80)[0] is True

    def test_away_does_not_run_it(self):
        assert _run(presence_home=False)[0] is False

    def test_unreadable_presence_declines(self):
        """A dead tracker must not read as 'home' -- nor as a decided 'away'."""
        run, why = _run(presence_home=None)
        assert run is False
        assert "unreadable" in why

    def test_presence_still_works_with_the_surplus_gate_off(self):
        assert _run(presence_home=True, surplus_run=False)[0] is True

    def test_one_person_home_is_enough(self):
        assert anyone_home(["not_home", "home"]) is True

    def test_both_away(self):
        assert anyone_home(["not_home", "not_home"]) is False

    def test_all_trackers_dead_is_unknown(self):
        assert anyone_home(["unavailable", None]) is None

    def test_one_live_tracker_decides(self):
        assert anyone_home(["unavailable", "not_home"]) is False


class TestTheBudgetIsWhatMakesItOpportunistic:
    """"om utrymme finns" -- without a cap, a sunny week at home runs it 24/7."""

    def test_a_spent_budget_stops_both_gates(self):
        for scene in ({"grid_w": -4000.0}, {"presence_home": True}):
            run, why = _run(extra_hours_today=4.0, **scene)
            assert run is False
            assert "budget" in why

    def test_a_partly_spent_budget_still_runs(self):
        assert _run(grid_w=-4000.0, extra_hours_today=3.9)[0] is True

    def test_no_budget_configured_disables_the_extras(self):
        """Unset means no room, never unlimited -- the safe reading of a forgotten knob."""
        run, why = _run(grid_w=-4000.0, max_extra_hours_per_day=None)
        assert run is False
        assert "budget" in why


class TestItOnlyEverAddsRuntime:
    def test_a_planned_block_is_not_an_extra_run(self):
        """The plan already said yes; this gate has nothing to add or subtract."""
        run, why = _run(plan_wants_on=True, grid_w=-4000.0)
        assert run is False
        assert "plan" in why

    def test_no_gates_configured_is_a_no_op(self):
        assert _run(surplus_run=False, presence_max_price_percentile=None,
                    grid_w=-4000.0, presence_home=True)[0] is False


class TestFailsClosed:
    def test_unknown_price_under_a_ceiling_declines(self):
        run, why = _run(grid_w=-4000.0, import_price_sek_kwh=None,
                        export_price_sek_kwh=None)
        assert run is False
        assert "unknown" in why

    def test_an_empty_price_window_declines(self):
        run, why = _run(grid_w=-4000.0, price_window=[])
        assert run is False
        assert "price window" in why

    def test_no_ceiling_configured_runs_on_surplus_alone(self):
        """Explicitly unset (not uncomputable) means the owner wants no price gate."""
        assert _run(grid_w=-4000.0, max_price_percentile=None, price_window=[])[0] is True
