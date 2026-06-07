"""get_nordpool_data must resolve sensor-backed export components for EVERY caller.

Previously only the planner path resolved export_*_entity; the recorder (which writes the
price model's training label) and the schedule/forecast APIs used raw spot. _resolve_pricing_overrides
centralizes the resolution so the recorded export label matches the planner's effective price.
"""

import pytest

from backend.core import ha_client, prices


class TestResolvePricingOverrides:
    @pytest.mark.asyncio
    async def test_returns_none_without_entities(self):
        # Only literals configured -> nothing to resolve, and no HA calls.
        cfg = {"pricing": {"export_premium_sek_kwh": 0.10}}
        assert await prices._resolve_pricing_overrides(cfg) is None

    @pytest.mark.asyncio
    async def test_resolves_configured_entities(self, monkeypatch):
        states = {
            "input_number.darkstar_export_premium": {"state": "0.10"},
            "input_number.darkstar_export_grid_benefit": {"state": "0.05"},
        }

        async def fake_get(entity_id):
            return states.get(entity_id)

        monkeypatch.setattr(ha_client, "get_ha_entity_state", fake_get)
        cfg = {
            "pricing": {
                "export_premium_entity": "input_number.darkstar_export_premium",
                "export_grid_benefit_entity": "input_number.darkstar_export_grid_benefit",
            }
        }
        overrides = await prices._resolve_pricing_overrides(cfg)
        assert overrides is not None
        assert overrides["export_premium_sek_kwh"] == pytest.approx(0.10)
        assert overrides["export_grid_benefit_sek_kwh"] == pytest.approx(0.05)

        # And the resolved overrides feed the export price end-to-end (spot -0.05 + 0.15 = 0.10).
        _imp, exp = prices.calculate_import_export_prices(-50.0, {"pricing": overrides})
        assert exp == pytest.approx(0.10)

    @pytest.mark.asyncio
    async def test_unreadable_entity_is_skipped(self, monkeypatch):
        async def fake_get(entity_id):
            return {"state": "unavailable"}

        monkeypatch.setattr(ha_client, "get_ha_entity_state", fake_get)
        cfg = {"pricing": {"export_premium_entity": "sensor.x"}}
        # 'unavailable' is not float-parseable -> no values -> None (falls back to literals).
        assert await prices._resolve_pricing_overrides(cfg) is None
