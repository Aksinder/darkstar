"""
Phase rebalancing recommendations (Recommend phase of phase-aware-load-modeling.md).

Phase 1 (``phase_learning``) learns which phase each single-phase device sits on and
publishes the per-phase load + imbalance. This module turns that into an *actionable*
one-time fix: which single-phase device should be moved to which phase to cut the
grid import the single-net-node MILP hides.

It is rigorous rather than hand-wavy: it replays the installation's real grid-meter
history and quantifies the *hidden per-phase waste* directly -

    waste(t) = sum_phase max(0, grid_phase(t))  -  max(0, grid_A+grid_B+grid_C)(t)

i.e. the power imported on heavy phases while the house is simultaneously exporting
on light phases (or net-exporting). A balanced inverter cannot fix this, so the only
levers are scheduling (Phase 3) and physically moving a load to another phase (here).

To evaluate "move device d from phase p to q" it uses the device's *own* metered
series: ``grid_p -= d`` and ``grid_q += d`` (moving a load within the house leaves the
net unchanged), recomputes the waste, and integrates the saved energy over the window
- capping the gap between samples so it never integrates across data outages. Results
are annualised and priced with the import/export spread.

Pure stdlib (no pandas, no I/O); safe on the read path. Only single-phase devices with
a confident learned mapping are considered; three-phase/unknown loads are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from backend.learning.cycle_learning import PowerSample
    from backend.learning.phase_learning import PhaseMapping

__all__ = [
    "MoveRecommendation",
    "recommend_phase_moves",
]

_PHASE_INDEX = {"A": 0, "B": 1, "C": 2}
_YEAR_SECONDS = 31_557_600.0  # 365.25 days


@dataclass(frozen=True)
class MoveRecommendation:
    """A suggested one-time move of a single-phase device to another phase."""

    device_id: str
    device_name: str
    from_phase: str
    to_phase: str
    device_avg_w: float
    imbalance_before_w: float
    imbalance_after_w: float
    window_import_avoided_kwh: float
    annual_import_avoided_kwh: float
    annual_saving_sek: float
    confidence: float


def _aligned(
    series_list: Sequence[Sequence[PowerSample]],
) -> list[tuple[datetime, tuple[float, ...]]]:
    """Forward-fill several PowerSample series onto their merged timeline.

    Each row is ``(timestamp, (value_per_series...))``; rows before every series has a
    value are skipped, so all tuples are fully populated.
    """
    if not series_list:
        return []
    timeline = sorted({s.ts for ser in series_list for s in ser})
    idx = [0] * len(series_list)
    last: list[float | None] = [None] * len(series_list)
    out: list[tuple[datetime, tuple[float, ...]]] = []
    for ts in timeline:
        for i, ser in enumerate(series_list):
            while idx[i] < len(ser) and ser[idx[i]].ts <= ts:
                last[i] = ser[idx[i]].power_w
                idx[i] += 1
        if all(v is not None for v in last):
            out.append((ts, tuple(cast("list[float]", list(last)))))
    return out


def _waste_w(grid: tuple[float, float, float]) -> float:
    """Hidden per-phase import: gross per-phase import minus the net import (W)."""
    gross = sum(max(0.0, g) for g in grid)
    net = max(0.0, grid[0] + grid[1] + grid[2])
    return gross - net


def _imbalance_w(grid: tuple[float, float, float]) -> float:
    mean = (grid[0] + grid[1] + grid[2]) / 3.0
    return max(grid) - mean


def recommend_phase_moves(
    phase_a: Sequence[PowerSample],
    phase_b: Sequence[PowerSample],
    phase_c: Sequence[PowerSample],
    devices: Sequence[tuple[PhaseMapping, Sequence[PowerSample]]],
    *,
    names: dict[str, str] | None = None,
    import_price_sek_kwh: float = 2.0,
    export_price_sek_kwh: float = 0.5,
    min_confidence: float = 0.5,
    max_gap_s: float = 900.0,
    min_benefit_kwh: float = 0.05,
) -> list[MoveRecommendation]:
    """Rank one-time device-to-phase moves by the grid import they would avoid.

    Replays the grid meters; for each confidently single-phase device evaluates moving
    it to each other phase using the device's own metered series, integrates the saved
    hidden-waste energy over the window (capping inter-sample gaps at ``max_gap_s``),
    annualises it, and prices it with the import/export spread. Returns the best target
    per device, best-first; moves below ``min_benefit_kwh`` over the window are dropped.
    """
    name_map = names or {}
    eligible = [
        (m, s)
        for m, s in devices
        if m.load_type == "single" and m.phase in _PHASE_INDEX and m.confidence >= min_confidence
    ]
    if not eligible:
        return []

    dev_series = [s for _, s in eligible]
    rows = _aligned([phase_a, phase_b, phase_c, *dev_series])
    if len(rows) < 2:
        return []

    window_s = max(0.0, (rows[-1][0] - rows[0][0]).total_seconds())
    annual_factor = (_YEAR_SECONDS / window_s) if window_s > 0 else 0.0
    price_spread = max(0.0, import_price_sek_kwh - export_price_sek_kwh)
    n_dev = len(eligible)

    # Accumulators over the window.
    benefit_ws: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n_dev)]  # per dev, per target
    imbalance_after_ws: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n_dev)]
    dev_sum: list[float] = [0.0] * n_dev
    dev_dt: list[float] = [0.0] * n_dev
    imbalance_before_ws = 0.0
    total_dt = 0.0

    for i in range(len(rows) - 1):
        ts, vals = rows[i]
        dt = min(max_gap_s, (rows[i + 1][0] - ts).total_seconds())
        if dt <= 0:
            continue
        grid = (vals[0], vals[1], vals[2])
        waste_before = _waste_w(grid)
        imbalance_before_ws += _imbalance_w(grid) * dt
        total_dt += dt
        for k, (m, _) in enumerate(eligible):
            d = vals[3 + k]
            dev_sum[k] += d * dt
            dev_dt[k] += dt
            pi = _PHASE_INDEX[cast("str", m.phase)]
            for qi in range(3):
                if qi == pi:
                    imbalance_after_ws[k][qi] += _imbalance_w(grid) * dt
                    continue
                moved = list(grid)
                moved[pi] -= d
                moved[qi] += d
                moved_t = (moved[0], moved[1], moved[2])
                benefit_ws[k][qi] += (waste_before - _waste_w(moved_t)) * dt
                imbalance_after_ws[k][qi] += _imbalance_w(moved_t) * dt

    if total_dt <= 0:
        return []

    recs: list[MoveRecommendation] = []
    for k, (m, _) in enumerate(eligible):
        pi = _PHASE_INDEX[cast("str", m.phase)]
        # Best target = max integrated benefit among the other two phases.
        best_qi = max((qi for qi in range(3) if qi != pi), key=lambda qi: benefit_ws[k][qi])
        benefit_kwh = benefit_ws[k][best_qi] / 3_600_000.0
        if benefit_kwh < min_benefit_kwh:
            continue
        to_phase = next(p for p, idx in _PHASE_INDEX.items() if idx == best_qi)
        annual_kwh = benefit_kwh * annual_factor
        avg_dev = (dev_sum[k] / dev_dt[k]) if dev_dt[k] > 0 else 0.0
        recs.append(
            MoveRecommendation(
                device_id=m.device_id,
                device_name=name_map.get(m.device_id, m.device_id),
                from_phase=cast("str", m.phase),
                to_phase=to_phase,
                device_avg_w=round(avg_dev, 1),
                imbalance_before_w=round(imbalance_before_ws / total_dt, 1),
                imbalance_after_w=round(imbalance_after_ws[k][best_qi] / total_dt, 1),
                window_import_avoided_kwh=round(benefit_kwh, 3),
                annual_import_avoided_kwh=round(annual_kwh, 1),
                annual_saving_sek=round(annual_kwh * price_spread, 0),
                confidence=m.confidence,
            )
        )

    recs.sort(key=lambda r: r.window_import_avoided_kwh, reverse=True)
    return recs
