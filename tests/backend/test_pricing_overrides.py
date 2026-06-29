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

    @pytest.mark.asyncio
    async def test_resolves_import_adders_with_ore_conversion(self, monkeypatch):
        # Swedish nätavgift/energiskatt helpers are in öre/kWh -> must convert to SEK (/100).
        states = {
            "input_number.elnat_transfer_cost": {
                "state": "37.5", "attributes": {"unit_of_measurement": "öre/kWh"}
            },
            "input_number.elnat_energy_tax": {
                "state": "53.5", "attributes": {"unit_of_measurement": "öre/kWh"}
            },
        }

        async def fake_get(entity_id):
            return states.get(entity_id)

        monkeypatch.setattr(ha_client, "get_ha_entity_state", fake_get)
        cfg = {
            "pricing": {
                "grid_transfer_fee_entity": "input_number.elnat_transfer_cost",
                "energy_tax_entity": "input_number.elnat_energy_tax",
            }
        }
        overrides = await prices._resolve_pricing_overrides(cfg)
        assert overrides is not None
        assert overrides["grid_transfer_fee_sek"] == pytest.approx(0.375)
        assert overrides["energy_tax_sek"] == pytest.approx(0.535)
        # End-to-end: import = (spot + transfer + tax) * 1.25.  spot 1.0 SEK/kWh:
        imp, _exp = prices.calculate_import_export_prices(1000.0, {"pricing": overrides})
        assert imp == pytest.approx((1.0 + 0.375 + 0.535) * 1.25)

    @pytest.mark.asyncio
    async def test_import_only_entity_triggers_resolution(self, monkeypatch):
        # Guard must include the import keys: an import-only config still resolves
        # (previously returned None because only export keys were checked).
        async def fake_get(entity_id):
            return {"state": "0.30", "attributes": {"unit_of_measurement": "SEK/kWh"}}

        monkeypatch.setattr(ha_client, "get_ha_entity_state", fake_get)
        cfg = {"pricing": {"grid_transfer_fee_entity": "input_number.x"}}
        overrides = await prices._resolve_pricing_overrides(cfg)
        assert overrides is not None
        assert overrides["grid_transfer_fee_sek"] == pytest.approx(0.30)


class TestPriceEntityToSek:
    def test_ore_converted(self):
        st = {"state": "37.5", "attributes": {"unit_of_measurement": "öre/kWh"}}
        assert prices.price_entity_to_sek(st) == pytest.approx(0.375)

    def test_sek_passthrough(self):
        st = {"state": "0.44", "attributes": {"unit_of_measurement": "SEK/kWh"}}
        assert prices.price_entity_to_sek(st) == pytest.approx(0.44)

    def test_no_unit_small_value_passthrough(self):
        # No unit + plausible SEK magnitude -> trust it as SEK.
        assert prices.price_entity_to_sek({"state": "0.44"}) == pytest.approx(0.44)

    def test_no_unit_large_value_rescued_as_ore(self):
        # Bare input_number in öre WITHOUT a unit must NOT be read as 37.5 SEK (100x error).
        assert prices.price_entity_to_sek({"state": "37.5"}) == pytest.approx(0.375)

    def test_kr_unit_passthrough(self):
        st = {"state": "0.44", "attributes": {"unit_of_measurement": "kr/kWh"}}
        assert prices.price_entity_to_sek(st) == pytest.approx(0.44)

    def test_unrecognized_unit_small_value_passthrough(self):
        # An odd unit that merely contains 'ore' (e.g. 'more') must not trigger /100 for a SEK value.
        st = {"state": "0.30", "attributes": {"unit_of_measurement": "more/kWh"}}
        assert prices.price_entity_to_sek(st) == pytest.approx(0.30)

    def test_unparseable_and_none(self):
        assert prices.price_entity_to_sek({"state": "unavailable"}) is None
        assert prices.price_entity_to_sek(None) is None
