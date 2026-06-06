"""
Come-home prediction for the EV (pure, no I/O, unit-testable). Step 1.

The hard presence gate (``ev_presence.ev_is_home``) decides whether to charge *now*.
This module decides whether to *pre-position* (reserve home-battery buffer) because the
car is likely to come home soon. Three distance zones:

- **home** - handled by the hard gate (charge now).
- **near** (``distance <= near_radius_km``): the car is close enough that we treat it as
  certainly arriving soon → ``p = 1.0`` (full reservation), no probability weighting.
- **extended** (``distance <= extended_radius_km``): the island/region where the car
  *might* come home → ``p`` from the learned arrival profile (weekday x hour fraction of
  time the car is home, from device_tracker history).
- **beyond** extended → ``p = 0`` (won't come home this horizon).

A manual override (``auto`` / ``force_reserve`` / ``force_off``) can pin the result, set
from a Home Assistant ``input_select`` so it is controllable from the dashboard.

The reservation is always soft, capped, and low-weight in the planner - a nudge, never a
forced grid purchase (the phantom-load safeguard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

__all__ = [
    "OVERRIDE_AUTO",
    "OVERRIDE_FORCE_OFF",
    "OVERRIDE_FORCE_RESERVE",
    "ArrivalProfile",
    "build_arrival_profile",
    "come_home_probability",
    "load_arrival_profile",
    "reserve_kwh",
    "save_arrival_profile",
]

OVERRIDE_AUTO = "auto"
OVERRIDE_FORCE_RESERVE = "force_reserve"
OVERRIDE_FORCE_OFF = "force_off"


@dataclass(frozen=True)
class ArrivalProfile:
    """Learned P(car is home) per (weekday, hour), from device_tracker history."""

    fraction: dict[str, float] = field(default_factory=lambda: {})  # "weekday:hour" -> [0,1]
    samples: int = 0
    default_p: float = 0.0

    @staticmethod
    def key(weekday: int, hour: int) -> str:
        return f"{weekday}:{hour}"

    def probability(self, when: datetime) -> float:
        """Historical P(home) for this moment's weekday/hour (default if unseen)."""
        return self.fraction.get(self.key(when.weekday(), when.hour), self.default_p)

    def to_dict(self) -> dict[str, Any]:
        return {"fraction": self.fraction, "samples": self.samples, "default_p": self.default_p}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrivalProfile:
        raw = data.get("fraction")
        fraction: dict[str, float] = {}
        if isinstance(raw, dict):
            for k, v in cast("dict[str, Any]", raw).items():
                try:
                    fraction[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            fraction=fraction,
            samples=int(data.get("samples", 0) or 0),
            default_p=float(data.get("default_p", 0.0) or 0.0),
        )


def build_arrival_profile(
    events: Sequence[tuple[datetime, bool]],
    *,
    step_minutes: int = 30,
    default_p: float = 0.0,
) -> ArrivalProfile:
    """Build the weekdayxhour home-probability profile from presence events.

    ``events`` are ``(timestamp, is_home)`` transitions (sorted or not). The presence
    state is forward-filled and sampled every ``step_minutes`` across the span; each
    sample is bucketed by (weekday, hour) and the bucket's fraction = home/total. This
    measures *fraction of time home*, robust to sparse/irregular state changes.
    """
    evs = sorted(events, key=lambda e: e[0])
    if not evs:
        return ArrivalProfile(samples=0, default_p=default_p)

    counts: dict[str, list[int]] = {}  # key -> [home_samples, total_samples]
    step = timedelta(minutes=max(1, step_minutes))
    start, end = evs[0][0], evs[-1][0]
    t = start
    idx = 0
    current = evs[0][1]
    # Safety bound on iterations (very wide windows / tiny steps).
    max_iters = 200_000
    iters = 0
    while t <= end and iters < max_iters:
        while idx < len(evs) and evs[idx][0] <= t:
            current = evs[idx][1]
            idx += 1
        bucket = counts.setdefault(ArrivalProfile.key(t.weekday(), t.hour), [0, 0])
        bucket[1] += 1
        if current:
            bucket[0] += 1
        t = t + step
        iters += 1

    fraction = {k: (c[0] / c[1]) for k, c in counts.items() if c[1] > 0}
    total = sum(c[1] for c in counts.values())
    return ArrivalProfile(fraction=fraction, samples=total, default_p=default_p)


def come_home_probability(
    now: datetime,
    distance_km: float | None,
    profile: ArrivalProfile | None,
    *,
    override: str = OVERRIDE_AUTO,
    near_radius_km: float = 0.0,
    extended_radius_km: float = 0.0,
) -> tuple[float, str, str]:
    """Return (p, zone, reason) for the come-home reservation.

    Zones by distance: near → 1.0, extended → profile, beyond → 0.0. The override pins
    the result: ``force_off`` → 0.0, ``force_reserve`` → 1.0.
    """
    if override == OVERRIDE_FORCE_OFF:
        return 0.0, "off", "override:force_off"
    if override == OVERRIDE_FORCE_RESERVE:
        return 1.0, "force", "override:force_reserve"

    if distance_km is None:
        return 0.0, "unknown", "no_distance"
    if near_radius_km > 0 and distance_km <= near_radius_km:
        return 1.0, "near", f"near (d={distance_km:.1f}<= {near_radius_km}km)"
    if extended_radius_km > 0 and distance_km <= extended_radius_km:
        p = profile.probability(now) if profile is not None else 0.0
        return max(0.0, min(1.0, p)), "extended", f"profile p={p:.2f} (d={distance_km:.1f}km)"
    return 0.0, "far", f"beyond extended (d={distance_km:.1f}km)"


def reserve_kwh(p: float, buffer_kwh: float, max_reserve_kwh: float) -> float:
    """Soft battery buffer to reserve for the (probable) arrival: p x buffer, capped."""
    return round(max(0.0, min(max_reserve_kwh, p * buffer_kwh)), 3)


# --- persistence (the only I/O; the logic above is pure) ----------------------


def _profile_path(config: dict[str, object], charger_id: str) -> Any:
    from pathlib import Path

    config_dir = str(config.get("config_dir") or "/config/darkstar")
    return Path(config_dir) / f"ev_arrival_{charger_id}.json"


def load_arrival_profile(config: dict[str, object], charger_id: str) -> ArrivalProfile | None:
    """Load a persisted arrival profile from ``<config_dir>/ev_arrival_<id>.json``."""
    import json

    try:
        data = json.loads(_profile_path(config, charger_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return ArrivalProfile.from_dict(cast("dict[str, Any]", data))


def save_arrival_profile(
    config: dict[str, object], charger_id: str, profile: ArrivalProfile
) -> None:
    """Persist an arrival profile for the planner/dashboard to read."""
    import json

    path = _profile_path(config, charger_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
