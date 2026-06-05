"""
Phase-aware load modeling (Observe phase of the phase-aware blueprint).

A 3-phase hybrid inverter (Sungrow SH) delivers PV + battery support *balanced*
across the three phases, while many house loads are *single-phase*. So a heavy
single-phase load imports grid on its phase while the other two export - even with
a full battery and surplus PV. Darkstar's single-net-node MILP nets this to zero and
never sees the cost (see docs/designs/phase-aware-load-modeling.md).

This module, pure stdlib and fully unit-testable, provides the read-only "Observe"
primitives proven on live data:

- ``learn_device_phase`` maps a metered device to its electrical phase by correlating
  the device's power *changes* against the per-phase meter *changes*. It cancels the
  balanced inverter injection by subtracting the three-phase mean, so a single-phase
  load L on phase X shows the signature dRel_X ~ +2/3 L, dRel_other ~ -1/3 L. It also
  classifies the load as single-phase (an imbalance source) vs three-phase (balanced,
  phase-neutral) vs unknown, with a confidence score.
- ``phase_imbalance_w`` reports the current grid imbalance (max phase minus mean) -
  immune to the balanced inverter, so it is a clean "how unbalanced am I now" metric.
- ``reconstruct_phase_fractions`` estimates each phase's share of the *house load*
  (fractions summing to 1) for feeding the realism simulation
  (``planner/simulation.py``), which then reports the per-phase grid cost the LP hides.

No device control happens here; it is safe on the executor read path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.learning.cycle_learning import PowerSample

__all__ = [
    "PHASE_LABELS",
    "PhaseLoadEstimate",
    "PhaseMapping",
    "learn_device_phase",
    "load_device_phases",
    "load_phase_fractions",
    "phase_imbalance_w",
    "reconstruct_phase_fractions",
]

PHASE_LABELS = ("A", "B", "C")

# A single-phase load on phase X (no inverter change) shifts that phase's
# deviation-from-mean by +2/3 of the load and the other two by -1/3 each.
_SINGLE_PHASE_IDEAL_SLOPE = 2.0 / 3.0


@dataclass(frozen=True)
class PhaseMapping:
    """The learned phase of one metered device.

    ``phase`` is the dominant phase label ("A"/"B"/"C") for a single-phase load,
    or ``None`` for a balanced three-phase load (or when unknown). ``load_type`` is
    "single" | "three_phase" | "unknown". ``slopes`` holds the regression slope of
    each phase's deviation-from-mean against the device power change (for transparency
    on the dashboard). ``confidence`` is 0..1.
    """

    device_id: str
    phase: str | None
    load_type: str
    confidence: float
    slopes: dict[str, float] = field(default_factory=lambda: {})
    n_steps: int = 0


@dataclass(frozen=True)
class PhaseLoadEstimate:
    """Per-phase house load (W) and the load fractions summing to 1."""

    load_w: dict[str, float] = field(default_factory=lambda: {})
    fractions: dict[str, float] = field(default_factory=lambda: {})
    imbalance_w: float = 0.0
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Series alignment (forward-fill onto a common timeline)
# ---------------------------------------------------------------------------


def _forward_filled(
    series: Sequence[Sequence[PowerSample]],
) -> list[tuple[float, ...]]:
    """Align several PowerSample series onto their merged timeline.

    Each output row is the tuple of the most recent value of every series at that
    timestamp. Rows before *all* series have produced a first value are skipped, so
    every emitted tuple is fully populated. Input series must be sorted (as the
    ``*_from_ha_history`` parsers guarantee).
    """
    if not series:
        return []
    indices = [0] * len(series)
    last: list[float | None] = [None] * len(series)
    # Merge all timestamps in order.
    timeline = sorted({s.ts for one in series for s in one})
    rows: list[tuple[float, ...]] = []
    for ts in timeline:
        for i, one in enumerate(series):
            j = indices[i]
            while j < len(one) and one[j].ts <= ts:
                last[i] = one[j].power_w
                j += 1
            indices[i] = j
        if all(v is not None for v in last):
            rows.append(tuple(v for v in last if v is not None))
    return rows


def _slope_through_origin(pairs: Sequence[tuple[float, float]]) -> tuple[float, int]:
    """Least-squares slope of y on x forced through the origin, plus the count."""
    sxy = 0.0
    sxx = 0.0
    for x, y in pairs:
        sxy += x * y
        sxx += x * x
    if sxx <= 0.0:
        return 0.0, len(pairs)
    return sxy / sxx, len(pairs)


# ---------------------------------------------------------------------------
# Device -> phase learning
# ---------------------------------------------------------------------------


def learn_device_phase(
    device: Sequence[PowerSample],
    phase_a: Sequence[PowerSample],
    phase_b: Sequence[PowerSample],
    phase_c: Sequence[PowerSample],
    *,
    device_id: str = "",
    min_step_w: float = 200.0,
    min_steps: int = 4,
    single_slope_min: float = 0.45,
    separation_min: float = 0.30,
    three_phase_max_abs: float = 0.20,
) -> PhaseMapping:
    """Learn which phase a metered device sits on (or that it is balanced 3-phase).

    Method (proven on live data): align the device power and the three phase meters,
    subtract the three-phase mean to cancel the balanced inverter injection, then for
    every step where ``|d(device)| >= min_step_w`` regress each phase's
    deviation-from-mean change on the device change (least squares through origin).
    The phase whose deviation rises ~+2/3 with the device is its phase.

    Classification:
    - **single**: the top slope exceeds ``single_slope_min`` and beats the runner-up by
      ``separation_min`` (the +2/3 vs -1/3 signature).
    - **three_phase**: every slope is within ``three_phase_max_abs`` of zero (a balanced
      load creates no imbalance, so no phase stands out).
    - **unknown**: too few steps or an ambiguous signature.
    """
    rows = _forward_filled([device, phase_a, phase_b, phase_c])
    # Build per-phase deviation-from-mean and pair the step-changes with d(device).
    pairs: dict[str, list[tuple[float, float]]] = {ph: [] for ph in PHASE_LABELS}
    prev: tuple[float, float, float, float] | None = None
    for dev, a, b, c in rows:
        mean = (a + b + c) / 3.0
        rel = (a - mean, b - mean, c - mean)
        cur = (dev, rel[0], rel[1], rel[2])
        if prev is not None:
            d_dev = cur[0] - prev[0]
            if abs(d_dev) >= min_step_w:
                for k, ph in enumerate(PHASE_LABELS):
                    pairs[ph].append((d_dev, cur[1 + k] - prev[1 + k]))
        prev = cur

    slopes: dict[str, float] = {}
    n_steps = 0
    for ph in PHASE_LABELS:
        s, n = _slope_through_origin(pairs[ph])
        slopes[ph] = round(s, 3)
        n_steps = n

    if n_steps < min_steps:
        return PhaseMapping(device_id, None, "unknown", 0.0, slopes, n_steps)

    ordered = sorted(PHASE_LABELS, key=lambda p: slopes[p], reverse=True)
    best, second = ordered[0], ordered[1]
    best_slope = slopes[best]
    separation = best_slope - slopes[second]
    max_abs = max(abs(slopes[p]) for p in PHASE_LABELS)

    # Three-phase (balanced) load: nothing stands out in either direction.
    if max_abs <= three_phase_max_abs:
        conf = round(max(0.0, 1.0 - max_abs / three_phase_max_abs), 3)
        return PhaseMapping(device_id, None, "three_phase", conf, slopes, n_steps)

    # Single-phase load: one phase rises clearly and beats the runner-up.
    if best_slope >= single_slope_min and separation >= separation_min:
        slope_score = min(1.0, best_slope / _SINGLE_PHASE_IDEAL_SLOPE)
        sep_score = min(1.0, separation / _SINGLE_PHASE_IDEAL_SLOPE)
        conf = round(slope_score * sep_score, 3)
        return PhaseMapping(device_id, best, "single", conf, slopes, n_steps)

    return PhaseMapping(device_id, None, "unknown", 0.0, slopes, n_steps)


# ---------------------------------------------------------------------------
# Per-phase load model
# ---------------------------------------------------------------------------


def load_phase_fractions(config: dict[str, object]) -> dict[str, float] | None:
    """Read learned per-phase load fractions persisted by the phase observer.

    Returns the ``{"A": .., "B": .., "C": ..}`` fractions from
    ``<config_dir>/phase_model.json`` (written by ``PhaseObserverService``), or None
    if absent/unreadable. Lets the planner feed *measured* shares into the realism
    simulation without coupling the planner to the learner.
    """
    import json
    from pathlib import Path

    config_dir = str(config.get("config_dir") or "/config/darkstar")
    path = Path(config_dir) / "phase_model.json"
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    fractions_raw = cast("dict[str, Any]", raw).get("fractions")
    if not isinstance(fractions_raw, dict):
        return None
    fractions = cast("dict[str, Any]", fractions_raw)
    out: dict[str, float] = {}
    for ph in PHASE_LABELS:
        try:
            out[ph] = float(fractions[ph])
        except (KeyError, TypeError, ValueError):
            return None
    return out


def load_device_phases(
    config: dict[str, object], *, min_confidence: float = 0.6
) -> dict[str, str]:
    """Read confidently-learned device->phase mappings persisted by the observer.

    Returns ``{device_id: "A"|"B"|"C"}`` for devices the observer classified as
    single-phase with confidence >= ``min_confidence`` (from
    ``<config_dir>/phase_model.json``). Used by the planner to auto-assign a
    deferrable load's phase when config doesn't pin one (Phase 3 control). Empty when
    absent/unreadable; three-phase/unknown/low-confidence devices are excluded.
    """
    import json
    from pathlib import Path

    config_dir = str(config.get("config_dir") or "/config/darkstar")
    path = Path(config_dir) / "phase_model.json"
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    devices = cast("dict[str, Any]", raw).get("devices")
    if not isinstance(devices, list):
        return {}
    out: dict[str, str] = {}
    for entry in cast("list[Any]", devices):
        if not isinstance(entry, dict):
            continue
        d = cast("dict[str, Any]", entry)
        phase = d.get("phase")
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if d.get("load_type") == "single" and phase in PHASE_LABELS and conf >= min_confidence:
            out[str(d.get("id"))] = str(phase)
    return out


def phase_imbalance_w(a: float, b: float, c: float) -> float:
    """Current grid imbalance = max phase minus the three-phase mean (W).

    Because the inverter injects equally on all phases, this cancels the balanced
    supply and reflects only the load imbalance - a clean instantaneous metric.
    """
    mean = (a + b + c) / 3.0
    return max(a, b, c) - mean


def reconstruct_phase_fractions(
    phase_a: Sequence[PowerSample],
    phase_b: Sequence[PowerSample],
    phase_c: Sequence[PowerSample],
    inverter_total: Sequence[PowerSample],
    *,
    min_total_w: float = 300.0,
) -> PhaseLoadEstimate:
    """Estimate each phase's share of the house load from meters + inverter output.

    Per sample the house load on a phase is ``grid_phase + inverter_total/3`` (the
    inverter supplies its AC output balanced, so each phase gets a third). Averaging
    over samples whose total load exceeds ``min_total_w`` (to avoid dividing noise)
    yields stable per-phase fractions for the realism simulation, plus the mean
    imbalance. ``inverter_total`` is PV + battery discharge in W (negative while the
    battery charges).
    """
    rows = _forward_filled([phase_a, phase_b, phase_c, inverter_total])
    sums: dict[str, float] = dict.fromkeys(PHASE_LABELS, 0.0)
    imbalance_sum = 0.0
    n = 0
    for a, b, c, inv in rows:
        share = inv / 3.0
        load = {"A": a + share, "B": b + share, "C": c + share}
        total = load["A"] + load["B"] + load["C"]
        if total < min_total_w:
            continue
        for ph in PHASE_LABELS:
            sums[ph] += max(0.0, load[ph])
        imbalance_sum += phase_imbalance_w(a, b, c)
        n += 1

    if n == 0:
        return PhaseLoadEstimate({}, {}, 0.0, 0)

    avg_load = {ph: sums[ph] / n for ph in PHASE_LABELS}
    grand = sum(avg_load.values())
    if grand <= 0.0:
        fractions: dict[str, float] = dict.fromkeys(PHASE_LABELS, 1.0 / 3.0)
    else:
        fractions = {ph: avg_load[ph] / grand for ph in PHASE_LABELS}
    return PhaseLoadEstimate(
        load_w={ph: round(avg_load[ph], 1) for ph in PHASE_LABELS},
        fractions={ph: round(fractions[ph], 4) for ph in PHASE_LABELS},
        imbalance_w=round(imbalance_sum / n, 1),
        n_samples=n,
    )
