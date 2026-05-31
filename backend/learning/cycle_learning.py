"""
Cycle Learning for deferrable household loads (dishwasher, washing machine, ...).

Step 1 of the deferrable-loads blueprint (docs/designs/deferrable-loads.md):
read-only detection of appliance run cycles from raw Home Assistant signals,
producing per-load learned estimates (duration, energy, typical power profile)
that the planner will later consume. No device control happens here.

Why this is non-trivial (observed in live HA data):
- A dishwasher's plug power can read ~0 W (metered via a native integration
  instead), so run-state must come from a boolean sensor.
- Boolean run-state sensors flap on/off every few seconds.
- Plug power sensors under significant_changes_only drop the low-power phases.

The detector therefore uses a hysteresis state machine plus min-on filtering and
gap-merging so flapping and multi-phase cycles collapse into a single cycle, and
integrates energy trapezoidally while refusing to integrate across long data gaps.

Pure stdlib (datetime, statistics) — no pandas, no I/O — so it is fully unit
testable and safe to call from the executor's read path.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CycleStats",
    "DetectedCycle",
    "PowerSample",
    "RunSample",
    "detect_cycles_from_power",
    "detect_cycles_from_runstate",
    "power_samples_from_ha_history",
    "run_samples_from_ha_history",
]

# HA states that are not real readings and must be skipped.
_NON_NUMERIC_STATES = frozenset({"unavailable", "unknown", "none", ""})
_TRUE_STATES = frozenset({"on", "true", "running", "1", "active", "open", "home"})
_FALSE_STATES = frozenset({"off", "false", "idle", "0", "inactive", "closed", "not_home"})


@dataclass(frozen=True)
class PowerSample:
    """A single power reading (Watts) at a timestamp."""

    ts: datetime
    power_w: float


@dataclass(frozen=True)
class RunSample:
    """A single boolean run-state reading at a timestamp."""

    ts: datetime
    running: bool


@dataclass(frozen=True)
class DetectedCycle:
    """One detected appliance run.

    energy_kwh and peak_w are 0.0 when detection came from a boolean run-state
    signal without an assumed power. profile_kw holds mean kW per
    ``profile_bucket_min`` bucket across the cycle span.
    """

    start: datetime
    end: datetime
    duration_min: float
    energy_kwh: float = 0.0
    peak_w: float = 0.0
    profile_kw: list[float] = field(default_factory=lambda: [])
    complete: bool = True


# ---------------------------------------------------------------------------
# HA history parsing
# ---------------------------------------------------------------------------


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def power_samples_from_ha_history(states: Sequence[dict[str, Any]]) -> list[PowerSample]:
    """Convert HA history rows (``{state, last_changed}``) into PowerSamples.

    Non-numeric states (``unavailable``/``unknown``) are skipped. Output is
    sorted chronologically.
    """
    out: list[PowerSample] = []
    for row in states:
        state = str(row.get("state", "")).strip().lower()
        if state in _NON_NUMERIC_STATES:
            continue
        try:
            power = float(row["state"])
        except (TypeError, ValueError, KeyError):
            continue
        ts_raw = row.get("last_changed") or row.get("last_updated")
        if not ts_raw:
            continue
        out.append(PowerSample(ts=_parse_ts(ts_raw), power_w=power))
    out.sort(key=lambda s: s.ts)
    return out


def run_samples_from_ha_history(states: Sequence[dict[str, Any]]) -> list[RunSample]:
    """Convert HA history rows into RunSamples (boolean), skipping unknowns."""
    out: list[RunSample] = []
    for row in states:
        state = str(row.get("state", "")).strip().lower()
        if state in _TRUE_STATES:
            running = True
        elif state in _FALSE_STATES:
            running = False
        else:
            continue
        ts_raw = row.get("last_changed") or row.get("last_updated")
        if not ts_raw:
            continue
        out.append(RunSample(ts=_parse_ts(ts_raw), running=running))
    out.sort(key=lambda s: s.ts)
    return out


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------


def _merge_segments(
    segments: list[tuple[datetime, datetime]],
    merge_gap_minutes: float,
) -> list[tuple[datetime, datetime]]:
    """Merge segments separated by a gap shorter than merge_gap_minutes.

    Collapses flapping run-state and the inter-phase pauses of a real cycle
    (e.g. a dishwasher heating, pausing, then heating again) into one cycle.
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s[0])
    merged: list[tuple[datetime, datetime]] = [segments[0]]
    gap = timedelta(minutes=merge_gap_minutes)
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _bucket_profile(
    samples: Sequence[PowerSample],
    start: datetime,
    end: datetime,
    bucket_min: float,
) -> list[float]:
    """Mean kW per ``bucket_min`` bucket across [start, end] (step interpolation)."""
    if bucket_min <= 0 or end <= start:
        return []
    span_min = (end - start).total_seconds() / 60.0
    n_buckets = max(1, round(span_min / bucket_min))
    sums = [0.0] * n_buckets
    secs = [0.0] * n_buckets
    bucket = timedelta(minutes=bucket_min)
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        seg_start = max(a.ts, start)
        seg_end = min(b.ts, end)
        if seg_end <= seg_start:
            continue
        # Spread this interval (step-held at the left sample's power) across
        # every bucket it overlaps, so a sparse long hold is distributed
        # correctly rather than dumped into the start bucket.
        t = seg_start
        while t < seg_end:
            idx = min(max(int((t - start) / bucket), 0), n_buckets - 1)
            bucket_end = start + bucket * (idx + 1)
            chunk_end = min(seg_end, bucket_end)
            dt = (chunk_end - t).total_seconds()
            if dt <= 0:
                break
            sums[idx] += a.power_w * dt
            secs[idx] += dt
            t = chunk_end
    return [
        round((sums[i] / secs[i] / 1000.0) if secs[i] > 0 else 0.0, 3) for i in range(n_buckets)
    ]


# ---------------------------------------------------------------------------
# Detection from power series
# ---------------------------------------------------------------------------


def detect_cycles_from_power(
    samples: Sequence[PowerSample],
    *,
    on_threshold_w: float = 50.0,
    off_threshold_w: float | None = None,
    min_on_minutes: float = 3.0,
    merge_gap_minutes: float = 20.0,
    profile_bucket_min: float = 15.0,
    max_sample_gap_minutes: float = 30.0,
) -> list[DetectedCycle]:
    """Detect appliance cycles from a power (W) time series.

    Args:
        samples: PowerSamples, any order (sorted internally).
        on_threshold_w: power at/above which the appliance counts as running.
        off_threshold_w: hysteresis low threshold (default on_threshold/2) to
            avoid chattering at the boundary.
        min_on_minutes: discard runs whose merged span is shorter than this
            (filters momentary blips).
        merge_gap_minutes: merge runs separated by a shorter gap into one cycle
            (handles multi-phase cycles and flapping).
        profile_bucket_min: bucket width for the per-cycle power profile.
        max_sample_gap_minutes: never integrate energy across a gap longer than
            this (a stale sensor must not inflate energy).

    Returns:
        Chronological list of DetectedCycle. An open run at the end of the
        series is returned with complete=False.
    """
    if off_threshold_w is None:
        off_threshold_w = on_threshold_w / 2.0

    pts = sorted(samples, key=lambda s: s.ts)
    if len(pts) < 2:
        return []

    # 1) Raw on-segments via a hysteresis state machine.
    raw: list[tuple[datetime, datetime]] = []
    running = False
    seg_start: datetime | None = None
    last_above: datetime | None = None
    for s in pts:
        if not running and s.power_w >= on_threshold_w:
            running = True
            seg_start = s.ts
            last_above = s.ts
        elif running:
            if s.power_w >= off_threshold_w:
                last_above = s.ts
            else:
                assert seg_start is not None and last_above is not None
                raw.append((seg_start, last_above))
                running = False
                seg_start = None
    open_tail: tuple[datetime, datetime] | None = None
    if running and seg_start is not None and last_above is not None:
        open_tail = (seg_start, pts[-1].ts)
        raw.append(open_tail)

    # 2) Merge nearby segments, 3) filter by min span, 4) measure each cycle.
    merged = _merge_segments(raw, merge_gap_minutes)
    cycles: list[DetectedCycle] = []
    for start, end in merged:
        duration_min = (end - start).total_seconds() / 60.0
        if duration_min < min_on_minutes:
            continue
        window = [s for s in pts if start <= s.ts <= end]
        energy_kwh = _integrate_energy_kwh(window, max_sample_gap_minutes)
        peak_w = max((s.power_w for s in window), default=0.0)
        profile = _bucket_profile(window, start, end, profile_bucket_min)
        complete = not (open_tail is not None and end == open_tail[1])
        cycles.append(
            DetectedCycle(
                start=start,
                end=end,
                duration_min=round(duration_min, 1),
                energy_kwh=round(energy_kwh, 3),
                peak_w=round(peak_w, 1),
                profile_kw=profile,
                complete=complete,
            )
        )
    return cycles


def _integrate_energy_kwh(
    samples: Sequence[PowerSample],
    max_sample_gap_minutes: float,
) -> float:
    """Trapezoidal Wh integration, skipping intervals longer than the max gap."""
    max_gap = timedelta(minutes=max_sample_gap_minutes)
    wh = 0.0
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        dt = b.ts - a.ts
        if dt <= timedelta(0) or dt > max_gap:
            continue
        wh += (a.power_w + b.power_w) / 2.0 * dt.total_seconds() / 3600.0
    return wh / 1000.0


# ---------------------------------------------------------------------------
# Detection from boolean run-state
# ---------------------------------------------------------------------------


def detect_cycles_from_runstate(
    samples: Sequence[RunSample],
    *,
    min_on_minutes: float = 3.0,
    merge_gap_minutes: float = 20.0,
    assumed_power_kw: float | None = None,
    profile_bucket_min: float = 15.0,
) -> list[DetectedCycle]:
    """Detect cycles from a (flapping) boolean run-state signal.

    Energy is only estimated when ``assumed_power_kw`` is given (duration x power); otherwise energy_kwh stays 0.0 and the duration is still learned.
    """
    pts = sorted(samples, key=lambda s: s.ts)
    if not pts:
        return []

    # Raw on-segments: from a rising edge until the next falling edge.
    raw: list[tuple[datetime, datetime]] = []
    seg_start: datetime | None = None
    for s in pts:
        if s.running and seg_start is None:
            seg_start = s.ts
        elif not s.running and seg_start is not None:
            raw.append((seg_start, s.ts))
            seg_start = None
    open_tail: tuple[datetime, datetime] | None = None
    if seg_start is not None:
        open_tail = (seg_start, pts[-1].ts)
        raw.append(open_tail)

    merged = _merge_segments(raw, merge_gap_minutes)
    cycles: list[DetectedCycle] = []
    for start, end in merged:
        duration_min = (end - start).total_seconds() / 60.0
        if duration_min < min_on_minutes:
            continue
        energy_kwh = 0.0
        profile: list[float] = []
        if assumed_power_kw is not None:
            energy_kwh = assumed_power_kw * (duration_min / 60.0)
            n_buckets = max(1, round(duration_min / profile_bucket_min))
            profile = [round(assumed_power_kw, 3)] * n_buckets
        complete = not (open_tail is not None and end == open_tail[1])
        cycles.append(
            DetectedCycle(
                start=start,
                end=end,
                duration_min=round(duration_min, 1),
                energy_kwh=round(energy_kwh, 3),
                peak_w=round((assumed_power_kw or 0.0) * 1000.0, 1),
                profile_kw=profile,
                complete=complete,
            )
        )
    return cycles


# ---------------------------------------------------------------------------
# Rolling learned estimate
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[k]


@dataclass(frozen=True)
class CycleStats:
    """Rolling learned estimate for one deferrable load.

    Until at least ``min_cycles`` complete cycles are observed, the seed values
    (from config) are returned and ``learned`` is False, so the planner never
    relies on a single noisy run.
    """

    n_cycles: int
    learned: bool
    duration_min: float
    duration_min_p90: float
    energy_kwh: float
    energy_kwh_p90: float
    typical_profile_kw: list[float]

    @classmethod
    def from_cycles(
        cls,
        cycles: Sequence[DetectedCycle],
        *,
        seed_duration_min: float,
        seed_energy_kwh: float,
        min_cycles: int = 3,
        require_energy: bool = True,
    ) -> CycleStats:
        """Aggregate completed cycles into a rolling estimate.

        Args:
            cycles: detected cycles (incomplete ones are ignored).
            seed_duration_min / seed_energy_kwh: config fallback used until
                enough real cycles exist.
            min_cycles: cycles required before ``learned`` becomes True.
            require_energy: when True, only cycles with energy_kwh > 0 count
                toward the energy estimate (skip run-state-only detections).
        """
        complete = [c for c in cycles if c.complete]
        durations = [c.duration_min for c in complete if c.duration_min > 0]
        energies = [c.energy_kwh for c in complete if (not require_energy or c.energy_kwh > 0)]

        learned = len(durations) >= min_cycles
        if not learned:
            return cls(
                n_cycles=len(complete),
                learned=False,
                duration_min=seed_duration_min,
                duration_min_p90=seed_duration_min,
                energy_kwh=seed_energy_kwh,
                energy_kwh_p90=seed_energy_kwh,
                typical_profile_kw=[],
            )

        duration_med = statistics.median(durations)
        energy_med = statistics.median(energies) if energies else seed_energy_kwh
        return cls(
            n_cycles=len(complete),
            learned=True,
            duration_min=round(duration_med, 1),
            duration_min_p90=round(_percentile(durations, 90), 1),
            energy_kwh=round(energy_med, 3),
            energy_kwh_p90=round(_percentile(energies, 90), 3) if energies else seed_energy_kwh,
            typical_profile_kw=_median_profile([c.profile_kw for c in complete if c.profile_kw]),
        )


def _median_profile(profiles: list[list[float]]) -> list[float]:
    """Element-wise median profile across cycles (padded to the longest)."""
    if not profiles:
        return []
    width = max(len(p) for p in profiles)
    out: list[float] = []
    for i in range(width):
        col = [p[i] for p in profiles if i < len(p)]
        out.append(round(statistics.median(col), 3) if col else 0.0)
    return out
