"""may_skip_day: track the market AND still sit out an expensive period.

Until now "may this load skip a day" was an accident of "dynamic_percentile is None".
Setting a percentile on the spa therefore silently removed its right to skip, and the
15 SEK/kWh reliability floor forced its 4 kWh daily need into the only hours left in the
calendar day — the 3.6 SEK evening peak, bought from the grid. Observed live 2026-08-18:
the spa was scheduled 19:45-20:45 at import 3.57-3.78 with no PV, drawing ~1 kW of extra
import, at a WTP cap of 2.71 that should have refused every one of those slots.

The two properties are independent: the ceiling says WHICH hours are cheap, the floor
says whether the load may go without. They are now expressed independently.
"""

from __future__ import annotations

from planner.solver.adapter import build_load_priorities


def _lp(spec: dict, tier: dict | None = None):
    """Resolve one load's priority through the real config path."""
    load = {"tier": "comfort", "rank": 3}
    load.update(spec)
    cfg = {
        "load_priority": {
            "enabled": True,
            "tiers": {"comfort": {"base_wtp_sek_per_kwh": 0.9, **(tier or {})}},
            "loads": {"spa": load},
        }
    }
    return build_load_priorities(cfg)[1]["spa"]


class TestParsing:
    def test_absent_defaults_to_false(self):
        assert _lp({}).may_skip_day is False

    def test_a_per_load_value_is_read(self):
        assert _lp({"may_skip_day": True}).may_skip_day is True

    def test_it_falls_back_to_the_tier(self):
        assert _lp({}, {"may_skip_day": True}).may_skip_day is True

    def test_a_per_load_value_wins_over_the_tier(self):
        assert _lp({"may_skip_day": False}, {"may_skip_day": True}).may_skip_day is False

    def test_it_is_independent_of_the_percentile(self):
        """The whole point: a dynamic ceiling no longer implies a mandatory floor."""
        lp = _lp({"wtp_percentile": 30, "may_skip_day": True})
        assert lp.dynamic_percentile == 30.0
        assert lp.may_skip_day is True
