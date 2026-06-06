"""
Pure presence logic for the EV home-zone gate (no I/O, fully unit-testable).

The planner should only schedule EV charging when the car is (about to be) charging at
*our* house. A device_tracker's binary home/not_home is a coarse proxy: a small zone
plus GPS drift can read not_home while the car sits in the driveway/garage, which would
wrongly exclude a home-charging car. So ``ev_is_home`` is robust in both directions:

- **zone**: the tracker state is one of ``home_states`` (the normal case),
- **radius**: the car is within ``radius_km`` of the home coordinates (absorbs GPS drift
  / covers the property even when the HA zone radius is tight),
- **grace**: the tracker only just flipped away (debounce momentary drift).

Otherwise the car is treated as away and excluded from the plan.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

__all__ = ["ev_is_home", "haversine_km"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    radius_earth = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius_earth * math.asin(min(1.0, math.sqrt(a)))


def ev_is_home(
    zone_state: str,
    *,
    home_states: Sequence[str],
    home_lat: float | None = None,
    home_lon: float | None = None,
    car_lat: float | None = None,
    car_lon: float | None = None,
    radius_km: float = 0.0,
    last_changed: datetime | None = None,
    grace_minutes: float = 0.0,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Decide whether the EV counts as home-chargeable. Returns (at_home, reason).

    Checks, in order: zone membership, distance within ``radius_km``, then a grace
    window (still treated home for ``grace_minutes`` after the tracker flipped away).
    ``radius_km``/``grace_minutes`` of 0 disable those layers, so the default behaviour
    is the plain zone check.
    """
    allowed = {str(s).lower() for s in home_states}
    if str(zone_state or "").lower() in allowed:
        return True, "zone"

    if (
        radius_km > 0
        and home_lat is not None
        and home_lon is not None
        and car_lat is not None
        and car_lon is not None
    ):
        dist = haversine_km(car_lat, car_lon, home_lat, home_lon)
        if dist <= radius_km:
            return True, f"radius={dist:.2f}km"

    if grace_minutes > 0 and last_changed is not None and now is not None:
        mins = (now - last_changed).total_seconds() / 60.0
        if 0.0 <= mins < grace_minutes:
            return True, f"grace={mins:.1f}min"

    return False, "away"
