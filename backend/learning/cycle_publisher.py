"""
Publish deferrable-load cycle statistics and hot-water state to Home Assistant.

This is the clean alternative to per-appliance helper/automation sprawl in HA:
Darkstar computes per-cycle energy/duration/phase and daily draw (via the
Cycle Learning module) and the hot-water tank state (via the thermal estimator),
then publishes them as ``sensor.darkstar_*`` entities through the HA REST API.

Creating sensor states is additive and controls no hardware, so it is safe to
run from the executor's read path.

The state-building functions are pure (no I/O) and fully unit-testable; only
``publish_sensors`` performs the HTTP POST.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.learning.cycle_learning import CycleStats, DetectedCycle
    from backend.learning.phase_learning import PhaseLoadEstimate, PhaseMapping
    from backend.learning.phase_recommend import MoveRecommendation
    from planner.hot_water import HotWaterEstimator

logger = logging.getLogger("darkstar.cycle_publisher")

__all__ = [
    "PublishedSensor",
    "build_hot_water_sensors",
    "build_load_sensors",
    "build_phase_recommendation_sensors",
    "build_phase_sensors",
    "build_realism_sensors",
    "publish_sensors",
]

# Human-friendly state for a learned device phase (single-phase uses the phase label).
_PHASE_STATE = {"three_phase": "3-fas", "unknown": "okänd"}


@dataclass(frozen=True)
class PublishedSensor:
    """One HA sensor state to publish via POST /api/states/<entity_id>."""

    object_id: str  # without the "sensor." prefix
    state: str
    unit: str | None = None
    device_class: str | None = None
    state_class: str | None = None
    icon: str | None = None
    friendly_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def entity_id(self) -> str:
        return f"sensor.{self.object_id}"

    def to_payload(self) -> dict[str, Any]:
        """REST body for /api/states (state + merged attributes)."""
        attrs: dict[str, Any] = dict(self.attributes)
        if self.unit is not None:
            attrs["unit_of_measurement"] = self.unit
        if self.device_class is not None:
            attrs["device_class"] = self.device_class
        if self.state_class is not None:
            attrs["state_class"] = self.state_class
        if self.icon is not None:
            attrs["icon"] = self.icon
        if self.friendly_name is not None:
            attrs["friendly_name"] = self.friendly_name
        return {"state": self.state, "attributes": attrs}


def _slug(value: str) -> str:
    """Sanitise an id into a safe entity object_id fragment."""
    s = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return s.strip("_") or "load"


def build_realism_sensors(realism: dict[str, Any] | None) -> list[PublishedSensor]:
    """Surface the planner's forward per-phase imbalance cost as a sensor.

    The net-node MILP is phase-blind; the realism simulation re-prices the optimal plan against
    the measured per-phase load split and reports the hidden cost (``gap_sek``) and the extra
    per-phase import (``extra_import_kwh``) the optimizer cannot see. Publishing it makes that
    structural loss visible on a dashboard instead of buried in the schedule meta. Returns ``[]``
    when no realism data is available (e.g. balanced/no measured phase fractions).
    """
    if not realism:
        return []
    gap = float(realism.get("gap_sek", 0.0) or 0.0)
    extra_kwh = float(realism.get("extra_import_kwh", 0.0) or 0.0)
    return [
        PublishedSensor(
            object_id="darkstar_phase_imbalance_cost",
            state=f"{round(gap, 3)}",
            unit="SEK",
            device_class="monetary",
            state_class="measurement",
            icon="mdi:scale-unbalanced",
            friendly_name="Fas-obalans kostnad (plan)",
            attributes={
                "extra_import_kwh": round(extra_kwh, 3),
                "phase_flagged_slots": int(realism.get("phase_flagged_slots", 0) or 0),
                "idle_exposed_slots": int(realism.get("idle_exposed_slots", 0) or 0),
            },
        )
    ]


def build_load_sensors(
    load_id: str,
    name: str,
    stats: CycleStats,
    last_cycle: DetectedCycle | None,
    draw_today_kwh: float,
) -> list[PublishedSensor]:
    """Build the sensor set for one deferrable appliance.

    Produces last-cycle energy/duration, learned typical values, and today's
    total energy. The main sensor carries the rich detail (cycle count, learned
    flag, per-phase minutes, power profile) as attributes.
    """
    sid = _slug(load_id)
    last_energy = round(last_cycle.energy_kwh, 3) if last_cycle else 0.0
    last_minutes = round(last_cycle.duration_min, 1) if last_cycle else 0.0

    main_attrs: dict[str, Any] = {
        "load_id": load_id,
        "cycles_observed": stats.n_cycles,
        "learned": stats.learned,
        "typical_minutes": stats.duration_min,
        "typical_minutes_p90": stats.duration_min_p90,
        "typical_energy_kwh": stats.energy_kwh,
        "typical_profile_kw": stats.typical_profile_kw,
    }
    if last_cycle is not None:
        main_attrs["last_cycle_start"] = last_cycle.start.isoformat()
        if last_cycle.phase_minutes:
            main_attrs["last_cycle_phases"] = last_cycle.phase_minutes

    return [
        PublishedSensor(
            object_id=f"darkstar_{sid}_last_cycle_energy",
            state=f"{last_energy}",
            unit="kWh",
            device_class="energy",
            state_class="total",
            icon="mdi:lightning-bolt",
            friendly_name=f"{name} senaste cykel energi",
            attributes=main_attrs,
        ),
        PublishedSensor(
            object_id=f"darkstar_{sid}_last_cycle_minutes",
            state=f"{last_minutes}",
            unit="min",
            icon="mdi:timer-outline",
            friendly_name=f"{name} senaste cykel tid",
        ),
        PublishedSensor(
            object_id=f"darkstar_{sid}_typical_minutes",
            state=f"{stats.duration_min}",
            unit="min",
            icon="mdi:timer-sand",
            friendly_name=f"{name} typisk cykeltid",
            attributes={"learned": stats.learned, "cycles_observed": stats.n_cycles},
        ),
        PublishedSensor(
            object_id=f"darkstar_{sid}_draw_today",
            state=f"{round(draw_today_kwh, 3)}",
            unit="kWh",
            device_class="energy",
            state_class="total_increasing",
            icon="mdi:counter",
            friendly_name=f"{name} förbrukning idag",
        ),
    ]


def build_hot_water_sensors(
    load_id: str,
    name: str,
    estimator: HotWaterEstimator,
    draw_today_kwh: float,
    *,
    object_id_prefix: str = "darkstar_",
) -> list[PublishedSensor]:
    """Build the hot-water availability sensors for one tank (VVB).

    Object-id suffixes follow the canonical scheme also used by HA-native template
    sensors (``_hot_water_level`` / ``_liters_remaining`` / ``_estimated_temperature``),
    so there is one naming convention across the install. ``object_id_prefix`` defaults
    to ``"darkstar_"`` so this publisher never collides with existing template sensors;
    set it to ``""`` (per tank, via ``sensor_prefix``) only when Darkstar should OWN the
    tank's sensors and no template sensor of the same id exists.
    """
    base = f"{object_id_prefix}{_slug(load_id)}"
    return [
        PublishedSensor(
            object_id=f"{base}_hot_water_level",
            state=f"{round(estimator.soc_percent(), 1)}",
            unit="%",
            icon="mdi:water-percent",
            friendly_name=f"{name} varmvattennivå",
            attributes={"temperature_c": round(estimator.temperature_c(), 1)},
        ),
        PublishedSensor(
            object_id=f"{base}_liters_remaining",
            state=f"{round(estimator.liters_in_tank(), 0)}",
            unit="L",
            icon="mdi:water",
            friendly_name=f"{name} varmvatten kvar",
            attributes={"mixed_liters_at_comfort": round(estimator.mixed_liters_at(), 0)},
        ),
        PublishedSensor(
            object_id=f"{base}_estimated_temperature",
            state=f"{round(estimator.temperature_c(), 1)}",
            unit="°C",
            device_class="temperature",
            state_class="measurement",
            icon="mdi:thermometer-water",
            friendly_name=f"{name} temperatur",
        ),
        PublishedSensor(
            object_id=f"{base}_draw_today",
            state=f"{round(draw_today_kwh, 3)}",
            unit="kWh",
            device_class="energy",
            state_class="total_increasing",
            icon="mdi:counter",
            friendly_name=f"{name} förbrukning idag",
        ),
    ]


def build_phase_sensors(
    estimate: PhaseLoadEstimate,
    mappings: list[PhaseMapping],
    *,
    device_names: dict[str, str] | None = None,
    current_imbalance_w: float | None = None,
) -> list[PublishedSensor]:
    """Build the per-phase observability sensors (Observe phase).

    Publishes each phase's house load (W) and share (%), the load imbalance (W,
    the real money signal the net-node LP hides), and one sensor per learned device
    reporting its electrical phase plus the regression detail as attributes.
    """
    names = device_names or {}
    sensors: list[PublishedSensor] = []

    for ph in ("A", "B", "C"):
        load = estimate.load_w.get(ph)
        frac = estimate.fractions.get(ph)
        sensors.append(
            PublishedSensor(
                object_id=f"darkstar_phase_{ph.lower()}_load",
                state=f"{round(load, 0) if load is not None else 0.0}",
                unit="W",
                device_class="power",
                state_class="measurement",
                icon="mdi:flash",
                friendly_name=f"Fas {ph} last",
                attributes={"share_percent": round((frac or 0.0) * 100.0, 1)},
            )
        )

    imbalance = current_imbalance_w if current_imbalance_w is not None else estimate.imbalance_w
    sensors.append(
        PublishedSensor(
            object_id="darkstar_phase_imbalance",
            state=f"{round(imbalance, 0)}",
            unit="W",
            device_class="power",
            state_class="measurement",
            icon="mdi:scale-balance",
            friendly_name="Fasobalans",
            attributes={
                "average_imbalance_w": estimate.imbalance_w,
                "samples": estimate.n_samples,
                "fractions": estimate.fractions,
            },
        )
    )

    for m in mappings:
        if m.load_type == "single":
            state = m.phase or "okänd"
        else:
            state = _PHASE_STATE.get(m.load_type, "okänd")
        sensors.append(
            PublishedSensor(
                object_id=f"darkstar_{_slug(m.device_id)}_phase",
                state=state,
                icon="mdi:sine-wave",
                friendly_name=f"{names.get(m.device_id, m.device_id)} fas",
                attributes={
                    "load_type": m.load_type,
                    "confidence": m.confidence,
                    "slopes": m.slopes,
                    "steps_observed": m.n_steps,
                },
            )
        )

    return sensors


def build_phase_recommendation_sensors(
    recommendations: list[MoveRecommendation],
) -> list[PublishedSensor]:
    """Build the rebalancing-recommendation sensor (Recommend phase).

    A single ``sensor.darkstar_phase_recommendation`` whose state is the top
    suggestion (or "balanserat") and whose attributes carry the full ranked list, so
    a dashboard can show "move device X from phase P to Q, ~N kr/yr".
    """
    if not recommendations:
        return [
            PublishedSensor(
                object_id="darkstar_phase_recommendation",
                state="balanserat",
                icon="mdi:scale-balance",
                friendly_name="Fasbalans-rekommendation",
                attributes={"count": 0, "recommendations": []},
            )
        ]

    top = recommendations[0]
    state = (
        f"Flytta {top.device_name} {top.from_phase}→{top.to_phase} "
        f"(~{round(top.annual_saving_sek)} kr/år)"
    )
    items = [
        {
            "device_id": r.device_id,
            "device_name": r.device_name,
            "from_phase": r.from_phase,
            "to_phase": r.to_phase,
            "device_avg_w": r.device_avg_w,
            "annual_import_avoided_kwh": r.annual_import_avoided_kwh,
            "annual_saving_sek": r.annual_saving_sek,
            "confidence": r.confidence,
        }
        for r in recommendations
    ]
    return [
        PublishedSensor(
            object_id="darkstar_phase_recommendation",
            state=state[:255],
            icon="mdi:transit-connection-variant",
            friendly_name="Fasbalans-rekommendation",
            attributes={
                "count": len(recommendations),
                "top_saving_sek": top.annual_saving_sek,
                "recommendations": items,
            },
        )
    ]


async def publish_sensors(
    sensors: list[PublishedSensor],
    base_url: str,
    token: str,
    *,
    timeout_s: float = 10.0,
) -> int:
    """POST each sensor to HA /api/states. Returns the number published.

    Failures are logged and skipped (one bad sensor never blocks the rest).
    """
    import httpx

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    published = 0
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for s in sensors:
            endpoint = f"{base_url.rstrip('/')}/api/states/{s.entity_id}"
            try:
                resp = await client.post(endpoint, headers=headers, json=s.to_payload())
                resp.raise_for_status()
                published += 1
            except Exception as exc:
                logger.warning("Failed to publish %s: %s", s.entity_id, exc)
    return published
