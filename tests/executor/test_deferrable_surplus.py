"""Surplus-aware window scoring.

Live 2026-09-14 07:05: the dishwasher started into the day's cheapest IMPORT window
(07:00, 2.65 SEK/kWh) while the 12:00 window on planned PV surplus would have cost the
export price, 1.68. The scorer only knew import prices, so every sunny day a morning
dip beat nearly-free midday sun. A slot the plan expects to export prices the share of
the appliance's draw that surplus covers at the export price.
"""

from __future__ import annotations

from executor.deferrable import WindowSlot, cheapest_window_start, recommend_appliance_action

SLOT = 900.0


def _slots(imp, exp=None, export_kw=None):
    exp = exp or [None] * len(imp)
    export_kw = export_kw or [0.0] * len(imp)
    return [
        WindowSlot(start_ts=i * SLOT, import_price_sek_kwh=p,
                   export_price_sek_kwh=e, export_kw=k)
        for i, (p, e, k) in enumerate(zip(imp, exp, export_kw, strict=True))
    ]


class TestEffectivePrice:
    def test_no_surplus_data_is_import(self):
        s = WindowSlot(0.0, 2.65)
        assert s.effective_price(1.0) == 2.65

    def test_full_cover_is_export(self):
        s = WindowSlot(0.0, 2.80, export_price_sek_kwh=1.68, export_kw=3.0)
        assert s.effective_price(0.6) == 1.68

    def test_partial_cover_blends(self):
        s = WindowSlot(0.0, 2.0, export_price_sek_kwh=1.0, export_kw=0.5)
        assert s.effective_price(1.0) == 1.5  # half at export, half at import

    def test_zero_appliance_power_is_legacy(self):
        s = WindowSlot(0.0, 2.80, export_price_sek_kwh=1.68, export_kw=3.0)
        assert s.effective_price(0.0) == 2.80

    def test_export_price_above_import_gives_no_credit(self):
        """A pathological price pair must never make surplus cost MORE than import."""
        s = WindowSlot(0.0, 1.0, export_price_sek_kwh=1.5, export_kw=3.0)
        assert s.effective_price(1.0) == 1.0


class TestSurplusWindowWins:
    """Morning import dip vs. midday surplus — the 2026-09-14 shape, 8 slots each."""

    def _day(self):
        imp = [2.65] * 8 + [3.0] * 8 + [2.80] * 8 + [4.5] * 8   # 07:00 dip, midday, evening peak
        exp = [1.2] * 8 + [1.5] * 8 + [1.68] * 8 + [3.0] * 8
        kw = [0.0] * 8 + [0.0] * 8 + [3.0] * 8 + [0.0] * 8       # surplus only at "midday"
        return _slots(imp, exp, kw)

    def test_legacy_scorer_picks_the_import_dip(self):
        start = cheapest_window_start(self._day(), now_ts=0.0, duration_slots=8, deadline_ts=None)
        assert start == 0.0

    def test_surplus_aware_scorer_picks_midday(self):
        start = cheapest_window_start(
            self._day(), now_ts=0.0, duration_slots=8, deadline_ts=None, appliance_kw=0.6,
        )
        assert start == 16 * SLOT

    def test_recommendation_defers_to_the_surplus_window(self):
        action, start = recommend_appliance_action(
            self._day(), now_ts=0.0, duration_slots=8, deadline_ts=None, appliance_kw=0.6,
        )
        assert action == "defer" and start == 16 * SLOT

    def test_wait_penalty_still_applies_against_surplus(self):
        """A wait price large enough to eat the surplus saving keeps 'run now'."""
        action, _ = recommend_appliance_action(
            self._day(), now_ts=0.0, duration_slots=8, deadline_ts=None,
            energy_kwh=1.2, wait_cost_sek_per_hour=5.0, appliance_kw=0.6,
        )
        assert action == "run"
