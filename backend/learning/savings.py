"""No-battery counterfactual savings — the "did Darkstar actually earn money?" metric.

For every recorded slot we compare what the house ACTUALLY paid on the grid with what
the SAME house would have paid with the battery layer switched off:

    actual   = import_kwh * import_price - export_kwh * export_price
    house    = load_kwh + water_kwh + ev_charging_kwh
    baseline = max(house - pv, 0) * import_price - max(pv - house, 0) * export_price

i.e. the baseline is a plain PV house: PV serves load first, any surplus exports, any
deficit imports — no battery arbitrage, no stored-solar time shifting. Note that
``SlotObservation.load_kwh`` is BASE load (the recorder subtracts water-heater and EV
energy before storing), so the counterfactual house must add those columns back to
consume the same whole-house energy the real house did.

Honesty notes (what this measures and what it does not):
- It measures the value of the BATTERY layer (arbitrage + stored-solar self-consumption)
  against the realized load/PV profile. Loads that Darkstar time-shifted (VVB, EV,
  appliances) appear at their shifted time in BOTH worlds, so scheduling value of
  load-shifting is NOT captured here — this is a floor, not the full story
  (see backend/learning/savings_loadshift.py for the load-shift streams).
- Slots without a recorded import price cannot be valued and are skipped; the summary
  reports coverage so a thin day can't masquerade as a bad day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SavingsSummary:
    """Aggregated actual-vs-baseline grid cost over a set of observed slots."""

    actual_cost_sek: float
    baseline_cost_sek: float
    savings_sek: float  # baseline - actual (positive = Darkstar saved money)
    n_slots: int  # observations considered
    n_priced_slots: int  # observations that carried an import price (valued)

    @property
    def coverage(self) -> float:
        """Fraction of slots that could be valued (0..1)."""
        return self.n_priced_slots / self.n_slots if self.n_slots else 0.0


def compute_savings(rows: Sequence[Mapping[str, Any]]) -> SavingsSummary:
    """Compute the no-battery counterfactual over observation rows.

    Rows need: import_kwh, export_kwh, pv_kwh, load_kwh, import_price_sek_kwh
    (export_price_sek_kwh optional, treated as 0 when absent — conservative for
    the baseline's export revenue and the actual's alike). load_kwh is BASE load,
    so water_kwh and ev_charging_kwh (0 when absent) are added back to reconstruct
    the whole-house load the baseline house must serve.
    """
    actual = 0.0
    baseline = 0.0
    n_priced = 0
    for row in rows:
        import_price = row.get("import_price_sek_kwh")
        if import_price is None:
            continue
        p_imp = float(import_price)
        p_exp = float(row.get("export_price_sek_kwh") or 0.0)
        imp = float(row.get("import_kwh") or 0.0)
        exp = float(row.get("export_kwh") or 0.0)
        pv = float(row.get("pv_kwh") or 0.0)
        # Whole-house load: load_kwh is base load (recorder subtracts water + EV).
        load = (
            float(row.get("load_kwh") or 0.0)
            + float(row.get("water_kwh") or 0.0)
            + float(row.get("ev_charging_kwh") or 0.0)
        )

        actual += imp * p_imp - exp * p_exp
        net = load - pv
        baseline += max(net, 0.0) * p_imp - max(-net, 0.0) * p_exp
        n_priced += 1

    return SavingsSummary(
        actual_cost_sek=round(actual, 4),
        baseline_cost_sek=round(baseline, 4),
        savings_sek=round(baseline - actual, 4),
        n_slots=len(rows),
        n_priced_slots=n_priced,
    )
