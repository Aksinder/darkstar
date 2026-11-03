"""Unseen-price-day reserve (upstream 894d6596, adapted to this fork).

The planning horizon ends where prices end: end of tomorrow, either published or
filled by the D+1 ML forecast fallback. Nothing in the objective prices the day
AFTER the horizon. The safety floor reserves for its load deficit (capped), and
``battery_value`` credits leftover energy at <= the cheapest import price in the
NEXT 12 h. Neither asks "will that unseen day's expensive hours cost more than
charging now?" — so the planner happily runs the pack to the floor at the horizon
end even when the forecast says the following evening is expensive and the
cheapest chance to refill it is inside the current horizon.

Upstream answers with a terminal SoC TARGET (enforced at 200 SEK/kWh) sized by a
greedy that also models the known window's charging cost. This fork instead hands
Kepler a concave piecewise TERMINAL CREDIT: per stored kWh, what it would avoid
paying on the unseen day. The MILP then compares that against the real known-window
costs itself — grid prices, PV surplus at its export opportunity cost, limits — so
no known-window cost curve is needed here and the credit cannot force grid buying
that does not pay. The credit is incremental over ``battery_value`` (no double
counting) and continuous (no binaries; a handful of bounded variables).

Value of carrying one STORED kWh into the unseen day, for an unseen slot h:

    v = discharge_eff * (min(P_h, own_day_cost) - 0.5 * wear - threshold)

``own_day_cost`` is the marginal cost of instead charging on the unseen day itself
(cheapest-first over its own slots: grid at import/(eta_c*eta_d) + wear, PV surplus
at export/(eta_c*eta_d) + wear), which grows as energy accumulates. Ordering within
the unseen day is ignored (as upstream): that overstates the own-day option, so the
reserve errs small. Everything here is pure; the pipeline gathers the inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

_EPS = 1e-9


@dataclass(frozen=True)
class UnseenSlot:
    """One slot of the unseen window (forecast prices, forecast net load)."""

    hours: float
    import_price: float  # SEK/kWh, forecast P50
    export_price: float  # SEK/kWh, forecast P50 (PV-surplus opportunity cost)
    net_load_kwh: float  # max(0, load - pv)
    pv_surplus_kwh: float  # max(0, pv - load)


@dataclass(frozen=True)
class UnseenReserveResult:
    segments: list[tuple[float, float]]  # (stored kWh, extra SEK/kWh over battery_value)
    reason: str
    total_kwh: float = 0.0
    top_value_sek_per_kwh: float = 0.0

    @property
    def active(self) -> bool:
        return self.total_kwh > _EPS


class _OwnDayCost:
    """Marginal cost per DELIVERED kWh of charging on the unseen day itself."""

    def __init__(
        self,
        slots: list[UnseenSlot],
        max_charge_kw: float,
        charge_eff: float,
        discharge_eff: float,
        wear: float,
    ) -> None:
        rt = charge_eff * discharge_eff
        segs: list[tuple[float, float]] = []
        if rt > 0:
            for s in slots:
                cap_in = max_charge_kw * s.hours
                if cap_in <= _EPS:
                    continue
                pv_in = min(s.pv_surplus_kwh, cap_in)
                if pv_in > _EPS:
                    segs.append((s.export_price / rt + wear, pv_in * rt))
                grid_in = cap_in - pv_in
                if grid_in > _EPS:
                    segs.append((s.import_price / rt + wear, grid_in * rt))
        segs.sort(key=lambda x: x[0])
        self._costs = [c for c, _ in segs]
        self._ends: list[float] = []
        acc = 0.0
        for _, cap in segs:
            acc += cap
            self._ends.append(acc)

    def at(self, delivered_kwh: float) -> tuple[float, float]:
        """(marginal cost, delivered kWh left in that segment); inf past capacity."""
        for cost, end in zip(self._costs, self._ends, strict=True):
            if delivered_kwh < end - _EPS:
                return cost, end - delivered_kwh
        return math.inf, math.inf


def _merge(raw: list[tuple[float, float]], max_segments: int) -> list[tuple[float, float]]:
    """Merge consecutive (kWh, value) pieces into <= max_segments by energy-weighted
    value. Input values are non-increasing, so the merged ones are too (concave)."""
    if not raw:
        return []
    n = max(1, int(max_segments))
    total = sum(e for e, _ in raw)
    if len(raw) <= n:
        return [(round(e, 4), round(v, 4)) for e, v in raw]
    target = total / n
    out: list[tuple[float, float]] = []
    acc_e = acc_ev = 0.0
    for e, v in raw:
        acc_e += e
        acc_ev += e * v
        if len(out) < n - 1 and acc_e >= target - _EPS:
            out.append((acc_e, acc_ev / acc_e))
            acc_e = acc_ev = 0.0
    if acc_e > _EPS:
        out.append((acc_e, acc_ev / acc_e))
    return [(round(e, 4), round(v, 4)) for e, v in out if e > _EPS and v > _EPS]


def size_unseen_reserve(
    slots: list[UnseenSlot],
    *,
    max_charge_kw: float,
    max_discharge_kw: float,
    charge_eff: float,
    discharge_eff: float,
    wear_cost_sek_per_kwh: float,
    threshold_sek_per_kwh: float,
    battery_value_sek_per_kwh: float,
    usable_kwh: float,
    max_segments: int = 6,
) -> UnseenReserveResult:
    """Size the concave terminal credit (stored kWh above min_soc) for the window."""
    if not slots:
        return UnseenReserveResult([], "no_window")
    if discharge_eff <= 0 or usable_kwh <= _EPS:
        return UnseenReserveResult([], "no_capacity")

    own = _OwnDayCost(slots, max_charge_kw, charge_eff, discharge_eff, wear_cost_sek_per_kwh)
    candidates = sorted(
        ((s.import_price, min(s.net_load_kwh, max_discharge_kw * s.hours)) for s in slots),
        key=lambda c: c[0],
        reverse=True,
    )
    candidates = [c for c in candidates if c[1] > _EPS]
    if not candidates:
        return UnseenReserveResult([], "no_net_load")

    raw: list[tuple[float, float]] = []  # (stored kWh, incremental value)
    delivered = 0.0
    stored = 0.0
    prev_value = math.inf
    done = False
    for price, energy in candidates:
        remaining = energy
        while remaining > _EPS:
            own_cost, own_left = own.at(delivered)
            alt = min(price, own_cost)
            value = discharge_eff * (alt - 0.5 * wear_cost_sek_per_kwh - threshold_sek_per_kwh)
            # Concavity: never let a later piece be worth more than an earlier one.
            value = min(value, prev_value)
            extra = value - battery_value_sek_per_kwh
            if extra <= _EPS:
                done = True
                break
            step = min(remaining, own_left)
            step_stored = step / discharge_eff
            if stored + step_stored > usable_kwh:
                step_stored = usable_kwh - stored
                step = step_stored * discharge_eff
                done = True
            if step_stored <= _EPS:
                done = True
                break
            raw.append((step_stored, extra))
            stored += step_stored
            delivered += step
            remaining -= step
            prev_value = value
            if done:
                break
        if done:
            break

    segments = _merge(raw, max_segments)
    total = sum(e for e, _ in segments)
    if total <= _EPS:
        return UnseenReserveResult([], "below_threshold")
    return UnseenReserveResult(segments, "active", round(total, 3), segments[0][1])


def build_unseen_slots(
    unseen_prices: Mapping[Any, tuple[float, float]],
    full_forecast_df: pd.DataFrame | None,
    window_start: Any,
    window_end: Any,
    *,
    slot_hours: float = 0.25,
    min_coverage: float = 0.9,
) -> tuple[list[UnseenSlot], str | None]:
    """Join forecast prices {slot_start: (import, export)} with the extended
    load/PV forecast over (window_start, window_end]. Returns (slots, None) or
    ([], reason) when either input covers less than ``min_coverage`` of the window.
    """
    import pandas as pd

    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end)
    expected = max(1, round((end_ts - start_ts).total_seconds() / 3600.0 / slot_hours))

    prices = {pd.Timestamp(k): v for k, v in unseen_prices.items()}
    in_window = sorted(ts for ts in prices if start_ts < ts <= end_ts)
    if len(in_window) < min_coverage * expected:
        return [], "insufficient_price_forecast"

    if (
        full_forecast_df is None
        or full_forecast_df.empty
        or not {"load_forecast_kwh", "pv_forecast_kwh"} <= set(full_forecast_df.columns)
    ):
        return [], "insufficient_load_forecast"
    win = full_forecast_df.loc[
        (full_forecast_df.index > start_ts) & (full_forecast_df.index <= end_ts),
        ["load_forecast_kwh", "pv_forecast_kwh"],
    ].dropna()
    if len(win) < min_coverage * expected:
        return [], "insufficient_load_forecast"

    slots: list[UnseenSlot] = []
    for ts in in_window:
        if ts not in win.index:
            continue
        load = float(win.at[ts, "load_forecast_kwh"])
        pv = float(win.at[ts, "pv_forecast_kwh"])
        imp, exp = prices[ts]
        slots.append(
            UnseenSlot(
                hours=slot_hours,
                import_price=float(imp),
                export_price=float(exp),
                net_load_kwh=max(0.0, load - pv),
                pv_surplus_kwh=max(0.0, pv - load),
            )
        )
    if len(slots) < min_coverage * expected:
        return [], "insufficient_overlap"
    return slots, None


def load_unseen_prices(
    config: dict[str, Any],
    db_path: str,
    window_start: Any,
    window_end: Any,
) -> dict[Any, tuple[float, float]]:
    """Latest-issued ML price forecast (P50) per slot in (window_start, window_end],
    as consumer prices {tz-aware slot_start: (import, export)}. Blocking (SQLite);
    call via asyncio.to_thread. Raises on DB errors (the pipeline catches and
    leaves the credit off).
    """
    from datetime import datetime

    import pandas as pd
    import pytz

    from ml.price_forecast import _latest_forecasts_for_local_date, derive_consumer_prices
    from utils.time_utils import dst_safe_localize

    tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))

    def _local(x: Any) -> pd.Timestamp:
        t = pd.Timestamp(x)
        if t.tzinfo is None:
            return pd.Timestamp(dst_safe_localize(t.to_pydatetime(), tz))
        return t.tz_convert(tz)

    start_ts = _local(window_start)
    end_ts = _local(window_end)
    out: dict[Any, tuple[float, float]] = {}
    day = start_ts.date()
    while day <= end_ts.date():
        for row in _latest_forecasts_for_local_date(db_path, day, tz):
            # Already tz-aware local ISO (_latest_forecasts_for_local_date localizes).
            ts = _local(datetime.fromisoformat(row["slot_start"]))
            if not (start_ts < ts <= end_ts) or row.get("spot_p50") is None:
                continue
            p50 = float(row["spot_p50"])
            prices = derive_consumer_prices(p50, p50, p50, config, slot_start=row["slot_start"])
            out[ts] = (prices["import_p50"], prices["export_p50"])
        day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
    return out
