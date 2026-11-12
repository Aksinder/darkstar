"""Fixed fees (grid subscription, supplier yearly fee) prorated per local calendar day.

Idea from upstream 91b219c1, with one fork adaptation: the fees follow
``pricing.fees_include_vat`` like every other configured fee (upstream always adds VAT,
which would charge it twice on fees entered "as billed").
"""

import calendar
from datetime import date, timedelta
from typing import Any, cast


def _non_negative(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))


def fixed_fees_for_range(start_date: date, end_date: date, config: dict[str, Any]) -> float:
    """Total fixed fees in SEK incl. VAT for the inclusive local date range.

    The grid monthly fee (``pricing.subscription_fee_sek_per_month``) is charged as
    monthly / days_in_month per calendar day and the supplier yearly fee
    (``pricing.supplier_fee_sek_per_year``) as yearly / days_in_year per calendar day.
    Both are in the VAT basis of ``pricing.fees_include_vat``: false (default) = entered
    excl. VAT and VAT is added here; true = entered incl. VAT and used as-is.

    Every calendar day of the range counts in full, whether or not slot data exists (a
    DST day is one day, today is a full day).
    """
    pricing: Any = config.get("pricing")
    pricing_cfg = cast("dict[str, Any]", pricing) if isinstance(pricing, dict) else {}
    monthly = _non_negative(pricing_cfg.get("subscription_fee_sek_per_month"))
    yearly = _non_negative(pricing_cfg.get("supplier_fee_sek_per_year"))
    if end_date < start_date or (monthly == 0.0 and yearly == 0.0):
        return 0.0

    total = 0.0
    day = start_date
    while day <= end_date:
        days_in_month = calendar.monthrange(day.year, day.month)[1]
        days_in_year = 366 if calendar.isleap(day.year) else 365
        total += monthly / days_in_month + yearly / days_in_year
        day += timedelta(days=1)
    if bool(pricing_cfg.get("fees_include_vat", False)):
        return total
    return total * (1 + _non_negative(pricing_cfg.get("vat_percent", 25.0)) / 100.0)
