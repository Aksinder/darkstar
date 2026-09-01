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
- It values grid CASH only. Energy still sitting in the battery at a window edge is not
  valued at all, so a partial window cut mid-cycle understates a charging day and
  overstates a draining one. ``compute_stored_energy_delta`` quantifies exactly that
  edge effect. It is applied to the TODAY sensor only and deliberately NOT folded into
  the 30-day figure, because over 30 days the effect self-cancels: measured over
  2026-08-03..09-02 the term is +2.87 SEK against +323.21 SEK of savings (0.9%), while
  on a single day it reaches +7.47 SEK and flips 2026-09-01 from -5.08 to +2.39.

INVARIANT — ``compute_savings`` is an exact per-slot sum and is therefore ADDITIVE over
any partition of slots: savings(A) + savings(B) == savings(A + B) for disjoint A, B
(measured residual 9e-13 SEK over 28 daily partitions). Do NOT "fix" the day boundary by
introducing a window-level term here — a window-weighted price is not additive, and that
is precisely the bug an earlier design of this feature would have shipped. The
inventory term lives in its own function, applied by the publisher to one window.
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


@dataclass(frozen=True)
class StoredEnergyDelta:
    """Value of the energy the battery GAINED (or lost) across a window.

    This is the edge term compute_savings cannot see: it books grid cash, so energy
    that entered the battery inside the window and has not come out yet reads as pure
    cost. Positive value_sek = the window ended holding energy that has already been
    paid for and will be earned back later.
    """

    net_stored_kwh: float  # efficiency-adjusted, slot-netted; + = ended holding more
    basis_sek_kwh: float | None  # source-cost of what went in; None when undeterminable
    value_sek: float  # net_stored_kwh * basis_sek_kwh, or 0.0 when basis is None
    charge_kwh: float  # netted inflow = the basis denominator
    n_slots: int  # observations considered
    n_battery_slots: int  # observations where BOTH flow columns were measured

    @property
    def battery_coverage(self) -> float:
        """Fraction of slots carrying real battery telemetry (0..1)."""
        return self.n_battery_slots / self.n_slots if self.n_slots else 0.0


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


def compute_stored_energy_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    roundtrip_efficiency: float = 0.95,
    basis_rows: Sequence[Mapping[str, Any]] | None = None,
) -> StoredEnergyDelta:
    """Value the energy the battery gained across a window, at what it cost to store.

    Why each clause is here -- every one answers a measurement, not a guess:

    EFFICIENCY. The raw flow difference sum(charge) - sum(discharge) is NOT the energy
    the battery gained; it also contains the round-trip loss. Over 2026-08-04..09-01 the
    raw difference is +28.49 kWh while the battery's own SoC moved +58.5 points, which
    on this site's measured ~15.5 kWh is +9.07 kWh. Crediting the raw difference would
    book roughly two thirds of it as savings that never existed. Applying the efficiency
    to the charge leg reconciles the two: 0.95 * 370.13 - 341.64 = +9.98 kWh, against
    +9.07 measured. (Capacity is measured, not assumed: a slot-level regression of SoC
    change on the two flow columns gives 15.25 kWh from the discharge leg and 15.69 from
    the charge leg. Note config.default.yaml:50 declares 34.2 kWh, which is wrong for
    this site -- and the reason this function takes the FLOW route and never touches a
    capacity constant.)

    SLOT NETTING. 216 real slots carry both a charge and a discharge. Netting per slot
    keeps energy that merely passed through out of the basis denominator.

    SOURCE PRICING. Energy charged from PV surplus displaced an EXPORT; energy charged
    from the grid displaced nothing -- its alternative was simply not buying, at the
    IMPORT price. Pricing both at a single export rate misprices grid-charged arbitrage.

    NO PRICE GATE. Unlike compute_savings, this loop must NOT skip unpriced slots: the
    energy physically moved whether or not a price was recorded, and dropping it would
    silently break the correspondence between this term and the battery's real content.
    An unpriced slot contributes to net_stored_kwh but not to the basis.

    NULL IS NOT ZERO. A slot whose flow columns were never recorded is skipped, never
    coalesced to 0.0. The NULLs are anti-correlated with charging (they cluster in the
    recorder's daylight gaps), so coalescing would fabricate a large phantom debit while
    priced coverage still read 1.000. ``battery_coverage`` makes such a window visible.

    ``basis_rows`` supplies a fallback basis for the dominant degenerate case: a window
    cut early in the morning has discharged overnight but not yet charged, so it has no
    inflow of its own to price. Recursion is depth-1 only.
    """
    eta = float(roundtrip_efficiency)
    net = 0.0
    num = 0.0
    den = 0.0
    n_batt = 0

    for row in rows:
        raw_charge = row.get("batt_charge_kwh")
        raw_discharge = row.get("batt_discharge_kwh")
        if raw_charge is None or raw_discharge is None:
            continue  # not measured -- never coalesce to 0.0
        n_batt += 1
        charge = float(raw_charge)
        discharge = float(raw_discharge)
        inflow = max(charge - discharge, 0.0)
        outflow = max(discharge - charge, 0.0)
        net += eta * inflow - outflow

        if inflow <= 0.0:
            continue
        raw_p_imp = row.get("import_price_sek_kwh")
        raw_p_exp = row.get("export_price_sek_kwh")
        # Either price alone is enough to value the inflow; when only one is recorded it
        # stands in for both. With neither, the slot is basis-neutral: its energy still
        # counts in net_stored_kwh above, but it cannot price anything.
        if raw_p_exp is not None:
            p_exp = float(raw_p_exp)
            p_imp = float(raw_p_imp) if raw_p_imp is not None else p_exp
        elif raw_p_imp is not None:
            p_imp = float(raw_p_imp)
            p_exp = p_imp
        else:
            continue

        house = (
            float(row.get("load_kwh") or 0.0)
            + float(row.get("water_kwh") or 0.0)
            + float(row.get("ev_charging_kwh") or 0.0)
        )
        surplus = max(float(row.get("pv_kwh") or 0.0) - house, 0.0)
        from_pv = min(inflow, surplus)
        from_grid = inflow - from_pv
        num += from_pv * p_exp + from_grid * p_imp
        den += inflow

    basis: float | None
    if den > 1e-9:
        basis = num / den
    elif basis_rows is not None:
        basis = compute_stored_energy_delta(
            basis_rows, roundtrip_efficiency=eta, basis_rows=None
        ).basis_sek_kwh
    else:
        basis = None

    value = net * basis if basis is not None else 0.0

    return StoredEnergyDelta(
        net_stored_kwh=round(net, 6),
        basis_sek_kwh=basis,
        value_sek=round(value, 4),
        charge_kwh=round(den, 6),
        n_slots=len(rows),
        n_battery_slots=n_batt,
    )
