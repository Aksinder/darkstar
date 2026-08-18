"""Per-load window for the dynamic WTP percentile.

The percentile answers "is this among the cheap hours", and the window decides cheap
COMPARED TO WHAT. At 24 h a comfort load still heats during a uniformly expensive day —
it just picks that day's cheapest hours, which is not what sitting out an expensive
stretch means (owner, 2026-08-18: "för spa skulle vi behöva längre fönster än 24h").

Unlike the executor the planner cannot look backwards; it only has its forward horizon,
so a longer ask is clipped to that rather than refused.
"""

from __future__ import annotations

from planner.solver.adapter import _wtp_window_hours, dynamic_wtp_from_prices

CHEAP = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
DEAR = [3.00, 3.20, 3.40, 3.60, 3.80, 4.00]


class TestWindowChangesTheVerdict:
    def test_a_short_window_sets_the_bar_at_the_dear_days_own_level(self):
        cap = dynamic_wtp_from_prices(DEAR, 30.0, len(DEAR))
        assert cap > 3.0, "judged only against itself, an expensive day looks normal"

    def test_a_longer_window_remembers_the_cheap_hours(self):
        cap = dynamic_wtp_from_prices(CHEAP + DEAR, 30.0, len(CHEAP + DEAR))
        assert cap < 1.0
        assert all(p > cap for p in DEAR), "no hour of the dear day is permitted"

    def test_the_window_takes_the_FIRST_n_prices(self):
        """It reads forward from now, so the window is a prefix, not a sample."""
        assert dynamic_wtp_from_prices(CHEAP + DEAR, 100.0, len(CHEAP)) == max(CHEAP)


class TestParsingTheOverride:
    def test_a_per_load_value_wins_over_the_tier(self):
        assert _wtp_window_hours({"wtp_window_hours": 48}, {"wtp_window_hours": 24}, "spa") == 48.0

    def test_it_falls_back_to_the_tier(self):
        assert _wtp_window_hours({}, {"wtp_window_hours": 36}, "spa") == 36.0

    def test_absent_means_the_default(self):
        assert _wtp_window_hours({}, {}, "spa") is None

    def test_nonsense_is_ignored_rather_than_fatal(self):
        assert _wtp_window_hours({"wtp_window_hours": "soon"}, {}, "spa") is None
        assert _wtp_window_hours({"wtp_window_hours": 0}, {}, "spa") is None
        assert _wtp_window_hours({"wtp_window_hours": -5}, {}, "spa") is None
