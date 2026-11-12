"""Supplier markup in the import price, the price cache key, and fixed fees in /energy/range.

Idea from upstream 91b219c1, adapted to the fork's ``fees_include_vat`` switch: the
markup and the fixed fees are in the same VAT basis as the other configured fees.
"""

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from backend.api.routers.config import _validate_config_for_save
from backend.api.routers.energy import get_energy_range
from backend.core.fixed_fees import fixed_fees_for_range
from backend.core.prices import _nordpool_cache_key, calculate_import_export_prices

SPOTS_MWH = [-123.4, 0.0, 0.1, 317.77, 999.99, 1234.5678, 4821.3]


def _legacy_import(spot_mwh: float, pricing: dict) -> float:
    """The import formula exactly as it was before the markup existed."""
    vat_multiplier = 1 + pricing.get("vat_percent", 25.0) / 100.0
    spot = spot_mwh / 1000.0
    transfer = pricing.get("grid_transfer_fee_sek", 0.2456)
    tax = pricing.get("energy_tax_sek", 0.439)
    if pricing.get("fees_include_vat", False):
        return spot * vat_multiplier + transfer + tax
    return (spot + transfer + tax) * vat_multiplier


class TestSupplierMarkup:
    @pytest.mark.parametrize("fees_include_vat", [False, True])
    @pytest.mark.parametrize("markup", ["absent", 0, 0.0, None, ""])
    def test_no_markup_is_bit_identical_to_the_old_formula(self, fees_include_vat, markup):
        pricing: dict = {
            "vat_percent": 25.0,
            "grid_transfer_fee_sek": 0.37,
            "energy_tax_sek": 0.45,
            "fees_include_vat": fees_include_vat,
        }
        if markup != "absent":
            pricing["supplier_markup_sek_kwh"] = markup
        for spot in SPOTS_MWH:
            imp, _ = calculate_import_export_prices(spot, {"pricing": pricing})
            assert imp == _legacy_import(spot, pricing)  # exact, not approx

    def test_default_config_ships_a_zero_markup(self):
        with Path("config.default.yaml").open(encoding="utf-8") as f:
            pricing = yaml.safe_load(f)["pricing"]
        assert pricing["supplier_markup_sek_kwh"] == 0.0
        assert pricing["subscription_fee_sek_per_month"] == 0.0
        assert pricing["supplier_fee_sek_per_year"] == 0.0
        for spot in SPOTS_MWH:
            imp, _ = calculate_import_export_prices(spot, {"pricing": pricing})
            assert imp == _legacy_import(spot, pricing)

    def test_markup_excl_vat_gets_vat_when_fees_exclude_vat(self):
        pricing = {"vat_percent": 25.0, "grid_transfer_fee_sek": 0.30, "energy_tax_sek": 0.40}
        base, _ = calculate_import_export_prices(1000.0, {"pricing": pricing})
        imp, _ = calculate_import_export_prices(
            1000.0, {"pricing": {**pricing, "supplier_markup_sek_kwh": 0.02}}
        )
        # (1.00 + 0.30 + 0.40 + 0.02) * 1.25
        assert imp == pytest.approx(2.15)
        assert imp - base == pytest.approx(0.025)

    def test_markup_incl_vat_is_added_as_is_when_fees_include_vat(self):
        pricing = {
            "vat_percent": 25.0,
            "grid_transfer_fee_sek": 0.375,
            "energy_tax_sek": 0.50,
            "fees_include_vat": True,
        }
        base, _ = calculate_import_export_prices(1000.0, {"pricing": pricing})
        imp, _ = calculate_import_export_prices(
            1000.0, {"pricing": {**pricing, "supplier_markup_sek_kwh": 0.025}}
        )
        # 1.00 * 1.25 + 0.375 + 0.50 + 0.025
        assert imp == pytest.approx(2.15)
        assert imp - base == pytest.approx(0.025)

    def test_both_conventions_agree_for_the_same_markup(self):
        excl = {"grid_transfer_fee_sek": 0.30, "energy_tax_sek": 0.40}
        incl = {"grid_transfer_fee_sek": 0.375, "energy_tax_sek": 0.50, "fees_include_vat": True}
        a, _ = calculate_import_export_prices(
            800.0, {"pricing": {**excl, "supplier_markup_sek_kwh": 0.02}}
        )
        b, _ = calculate_import_export_prices(
            800.0, {"pricing": {**incl, "supplier_markup_sek_kwh": 0.025}}
        )
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("fees_include_vat", [False, True])
    def test_markup_never_changes_the_export_price(self, fees_include_vat):
        pricing = {
            "fees_include_vat": fees_include_vat,
            "export_grid_benefit_sek_kwh": 0.044,
            "export_premium_sek_kwh": 0.01,
        }
        for spot in SPOTS_MWH:
            _, base = calculate_import_export_prices(spot, {"pricing": pricing})
            _, exp = calculate_import_export_prices(
                spot, {"pricing": {**pricing, "supplier_markup_sek_kwh": 0.5}}
            )
            assert exp == base

    def test_negative_markup_is_a_discount(self):
        base, _ = calculate_import_export_prices(1000.0, {"pricing": {}})
        imp, _ = calculate_import_export_prices(
            1000.0, {"pricing": {"supplier_markup_sek_kwh": -0.04}}
        )
        assert imp - base == pytest.approx(-0.05)


CACHE_KEY_BASE = {
    "vat_percent": 25.0,
    "fees_include_vat": False,
    "grid_transfer_fee_sek": 0.37,
    "energy_tax_sek": 0.45,
    "supplier_markup_sek_kwh": 0.0,
}


class TestPriceCacheKey:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("vat_percent", 6.0),
            ("fees_include_vat", True),
            ("supplier_markup_sek_kwh", 0.02),
            ("grid_transfer_fee_sek", 0.65),
            ("energy_tax_sek", 0.36),
            ("export_grid_benefit_sek_kwh", 0.044),
        ],
    )
    def test_each_formula_input_invalidates_the_key(self, key, value):
        assert _nordpool_cache_key({**CACHE_KEY_BASE, key: value}) != _nordpool_cache_key(
            CACHE_KEY_BASE
        )

    def test_defaults_and_explicit_defaults_share_a_key(self):
        # An absent key and its default must not split the cache.
        assert _nordpool_cache_key({}) == _nordpool_cache_key(
            {"vat_percent": 25.0, "fees_include_vat": False, "supplier_markup_sek_kwh": 0.0}
        )
        assert _nordpool_cache_key({"supplier_markup_sek_kwh": None}) == _nordpool_cache_key({})


class TestFixedFees:
    def test_no_fees_configured_is_zero(self):
        assert fixed_fees_for_range(date(2026, 1, 1), date(2026, 1, 31), {"pricing": {}}) == 0.0
        assert fixed_fees_for_range(date(2026, 1, 1), date(2026, 1, 31), {}) == 0.0

    def test_full_month_excl_vat_gets_vat(self):
        cfg = {"pricing": {"subscription_fee_sek_per_month": 400.0, "vat_percent": 25.0}}
        assert fixed_fees_for_range(date(2026, 2, 1), date(2026, 2, 28), cfg) == pytest.approx(
            500.0
        )

    def test_full_month_incl_vat_is_used_as_billed(self):
        cfg = {
            "pricing": {
                "subscription_fee_sek_per_month": 500.0,
                "vat_percent": 25.0,
                "fees_include_vat": True,
            }
        }
        assert fixed_fees_for_range(date(2026, 2, 1), date(2026, 2, 28), cfg) == pytest.approx(
            500.0
        )

    def test_yearly_fee_is_prorated_per_day_and_leap_aware(self):
        cfg = {"pricing": {"supplier_fee_sek_per_year": 365.0, "fees_include_vat": True}}
        assert fixed_fees_for_range(date(2026, 3, 1), date(2026, 3, 10), cfg) == pytest.approx(10.0)
        leap = {"pricing": {"supplier_fee_sek_per_year": 366.0, "fees_include_vat": True}}
        assert fixed_fees_for_range(date(2028, 1, 1), date(2028, 12, 31), leap) == pytest.approx(
            366.0
        )

    def test_range_spanning_months_uses_each_months_length(self):
        cfg = {"pricing": {"subscription_fee_sek_per_month": 310.0, "fees_include_vat": True}}
        # Jan 31 (310/31 = 10) + Feb 1 (310/28)
        assert fixed_fees_for_range(date(2026, 1, 31), date(2026, 2, 1), cfg) == pytest.approx(
            10.0 + 310.0 / 28
        )

    def test_garbage_and_negative_values_count_as_zero(self):
        cfg = {
            "pricing": {
                "subscription_fee_sek_per_month": "abc",
                "supplier_fee_sek_per_year": -5,
            }
        }
        assert fixed_fees_for_range(date(2026, 1, 1), date(2026, 1, 2), cfg) == 0.0
        assert fixed_fees_for_range(date(2026, 1, 2), date(2026, 1, 1), cfg) == 0.0


def _store_returning(row: tuple) -> MagicMock:
    result = MagicMock()
    result.fetchone.return_value = row
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    store = MagicMock()
    store.AsyncSession.return_value = ctx
    return store


class TestEnergyRangeFixedFees:
    # import 10 kWh, export 4 kWh, ..., import cost 20, export revenue 3, ..., 96 slots
    ROW = (10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 3.0, 0.0, 0.0, 96)

    async def _range(self, pricing: dict) -> dict:
        config = {"timezone": "Europe/Stockholm", "pricing": pricing}
        with patch("backend.api.routers.energy.load_yaml", return_value=config):
            return await get_energy_range(
                period="custom",
                start_date="2026-02-01",
                end_date="2026-02-28",
                store=_store_returning(self.ROW),
            )

    @pytest.mark.asyncio
    async def test_net_cost_keeps_its_meaning_and_fees_get_their_own_fields(self):
        data = await self._range(
            {"subscription_fee_sek_per_month": 400.0, "supplier_fee_sek_per_year": 365.0}
        )
        assert "error" not in data
        assert data["net_cost_sek"] == pytest.approx(17.0)  # energy only, unchanged
        # (400 + 28 days of 1 SEK) * 1.25
        assert data["fixed_fees_sek"] == pytest.approx(535.0)
        assert data["net_cost_incl_fixed_fees_sek"] == pytest.approx(552.0)

    @pytest.mark.asyncio
    async def test_fees_include_vat_is_honoured(self):
        data = await self._range(
            {"subscription_fee_sek_per_month": 500.0, "fees_include_vat": True}
        )
        assert data["fixed_fees_sek"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_without_fees_the_new_fields_are_neutral(self):
        data = await self._range({})
        assert data["fixed_fees_sek"] == 0.0
        assert data["net_cost_incl_fixed_fees_sek"] == data["net_cost_sek"] == pytest.approx(17.0)


VALIDATION_BASE = {"system": {"has_battery": False, "has_solar": False, "has_water_heater": False}}


class TestPricingValidation:
    def _pricing_issues(self, pricing: dict) -> list[dict]:
        issues = _validate_config_for_save({**VALIDATION_BASE, "pricing": pricing})
        return [i for i in issues if i["message"].startswith("pricing.")]

    def test_valid_values_are_accepted(self):
        assert not self._pricing_issues(
            {
                "supplier_markup_sek_kwh": 0.02,
                "subscription_fee_sek_per_month": 395,
                "supplier_fee_sek_per_year": 540.0,
            }
        )
        assert not self._pricing_issues({})
        assert not self._pricing_issues({"supplier_markup_sek_kwh": None})
        assert not self._pricing_issues({"supplier_markup_sek_kwh": -0.04})  # a discount

    def test_non_numbers_and_negative_fixed_fees_are_errors(self):
        for pricing in (
            {"supplier_markup_sek_kwh": "2 öre"},
            {"supplier_markup_sek_kwh": True},
            {"subscription_fee_sek_per_month": -1},
            {"supplier_fee_sek_per_year": "x"},
        ):
            issues = self._pricing_issues(pricing)
            assert [i["severity"] for i in issues] == ["error"], pricing


@pytest.mark.parametrize("bad", ["0,02", "abc", float("nan"), float("inf"), True, [], {}])
def test_unusable_markup_counts_as_zero_and_never_raises(bad):
    """A hand-edited YAML value must not raise out of the price fetch or poison prices."""
    from backend.core import prices

    assert prices._supplier_markup_sek({"supplier_markup_sek_kwh": bad}) == 0.0
    assert prices._nordpool_cache_key({"supplier_markup_sek_kwh": bad}) == (
        prices._nordpool_cache_key({})
    )
