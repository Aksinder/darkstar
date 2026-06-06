"""Tests for the configurable export-compensation model (C1).

Export price must be ``(spot if include_spot) + premium + grid_benefit - fee`` so it
reflects the user's real contract instead of raw spot, and stays backward compatible
(all components default to 0 -> export == spot).
"""

import pytest

from backend.core.prices import (
    calculate_import_export_prices,
    resolve_export_price_components,
)


def _export(spot_kwh: float, pricing: dict | None = None) -> float:
    """Export price (SEK/kWh) for a given spot (SEK/kWh) and pricing config."""
    _imp, exp = calculate_import_export_prices(spot_kwh * 1000.0, {"pricing": pricing or {}})
    return exp


class TestExportComponents:
    def test_default_export_equals_spot(self):
        # No components configured -> legacy behaviour.
        assert _export(1.00) == pytest.approx(1.00)
        assert _export(-0.05) == pytest.approx(-0.05)

    def test_premium_grid_benefit_and_fee_applied(self):
        pricing = {
            "export_premium_sek_kwh": 0.10,
            "export_grid_benefit_sek_kwh": 0.04,
            "export_fee_sek_kwh": 0.01,
        }
        # 1.00 + 0.10 + 0.04 - 0.01
        assert _export(1.00, pricing) == pytest.approx(1.13)

    def test_include_spot_false_uses_components_only(self):
        pricing = {
            "export_includes_spot": False,
            "export_premium_sek_kwh": 0.10,
            "export_grid_benefit_sek_kwh": 0.04,
        }
        assert _export(1.00, pricing) == pytest.approx(0.14)

    def test_shallow_negative_spot_stays_profitable_with_premium(self):
        # spot -0.05 + 0.10 premium + 0.04 nätnytta = +0.09 -> still worth exporting.
        pricing = {
            "export_premium_sek_kwh": 0.10,
            "export_grid_benefit_sek_kwh": 0.04,
        }
        assert _export(-0.05, pricing) == pytest.approx(0.09)
        assert _export(-0.05, pricing) > 0

    def test_deep_negative_spot_turns_export_negative(self):
        # spot -0.30 + 0.14 comp = -0.16 -> we now pay to export (curtail signal).
        pricing = {
            "export_premium_sek_kwh": 0.10,
            "export_grid_benefit_sek_kwh": 0.04,
        }
        assert _export(-0.30, pricing) == pytest.approx(-0.16)
        assert _export(-0.30, pricing) < 0

    def test_import_price_unchanged_by_export_components(self):
        base_imp, _ = calculate_import_export_prices(1000.0, {"pricing": {}})
        with_comp_imp, _ = calculate_import_export_prices(
            1000.0, {"pricing": {"export_premium_sek_kwh": 0.5}}
        )
        assert with_comp_imp == pytest.approx(base_imp)


class TestResolveExportComponents:
    def test_entity_overrides_literal(self):
        pricing = {
            "export_premium_sek_kwh": 0.10,
            "export_premium_entity": "input_number.export_premium",
        }
        states = {"input_number.export_premium": 0.155}
        resolved = resolve_export_price_components(pricing, lambda e: states.get(e))
        assert resolved["export_premium_sek_kwh"] == pytest.approx(0.155)
        # original is not mutated
        assert pricing["export_premium_sek_kwh"] == pytest.approx(0.10)

    def test_unset_entity_keeps_literal(self):
        pricing = {"export_premium_sek_kwh": 0.10, "export_premium_entity": ""}
        resolved = resolve_export_price_components(pricing, lambda e: 9.99)
        assert resolved["export_premium_sek_kwh"] == pytest.approx(0.10)

    def test_unreadable_entity_keeps_literal(self):
        pricing = {
            "export_grid_benefit_sek_kwh": 0.04,
            "export_grid_benefit_entity": "sensor.natnytta",
        }
        resolved = resolve_export_price_components(pricing, lambda e: None)
        assert resolved["export_grid_benefit_sek_kwh"] == pytest.approx(0.04)

    def test_resolved_components_feed_export_price(self):
        pricing = {
            "export_premium_entity": "input_number.export_premium",
            "export_grid_benefit_entity": "input_number.natnytta",
        }
        states = {
            "input_number.export_premium": 0.10,
            "input_number.natnytta": 0.04,
        }
        resolved = resolve_export_price_components(pricing, lambda e: states.get(e))
        _imp, exp = calculate_import_export_prices(-50.0, {"pricing": resolved})  # spot -0.05
        assert exp == pytest.approx(0.09)
