"""
Orchestration for publishing deferrable-load cycle stats + hot-water state to HA.

Ties together Cycle Learning (detection), the thermal hot-water estimator, and
the sensor publisher into a single periodic task. Designed for dependency
injection (HA history reader, state reader, publisher, clock) so the whole
orchestration is unit-testable without a live Home Assistant.

Per tick (``run_once``):
- Appliances: read the signal entity's history, detect cycles, compute rolling
  stats + last cycle + today's draw, and publish ``sensor.darkstar_*``.
- Tanks (VVB): advance a persistent ``HotWaterEstimator`` by the elapsed time
  and current heating power, then publish SoC / litres / temperature / draw.

Publishing sensor states controls no hardware, so this is safe on the read path.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from backend.learning.cycle_learning import (
    CycleStats,
    DetectedCycle,
    detect_cycles_from_power,
    detect_cycles_from_runstate,
    detect_cycles_from_status,
    power_samples_from_ha_history,
    run_samples_from_ha_history,
    status_samples_from_ha_history,
)
from backend.learning.cycle_publisher import (
    PublishedSensor,
    build_hot_water_sensors,
    build_load_sensors,
)
from planner.hot_water import HotWaterEstimator
from planner.thermal import WaterTankModel

logger = logging.getLogger("darkstar.cycle_publisher_service")

# Injected I/O signatures.
FetchHistory = Callable[[str, int], Awaitable[list[dict[str, Any]]]]
FetchFloat = Callable[[str], Awaitable[float | None]]
Publish = Callable[[list[PublishedSensor]], Awaitable[int]]


@dataclass
class TrackedAppliance:
    """A deferrable appliance to learn cycles for and publish."""

    id: str
    name: str
    signal_entity: str
    signal_kind: str = "status"  # "status" | "power" | "runstate"
    power_attr: str = "power"
    power_scale: float = 1.0
    running_power_w: float | None = None
    seed_duration_min: float = 120.0
    seed_energy_kwh: float = 1.0
    assumed_power_kw: float | None = None  # runstate energy estimate


@dataclass
class TrackedTank:
    """A hot-water tank (VVB) to estimate and publish."""

    id: str
    name: str
    power_entity: str
    volume_litres: float
    t_cold_c: float = 10.0
    t_max_c: float = 85.0
    ua_w_per_k: float = 2.0
    power_scale: float = 1.0


class DeferrablePublisherService:
    """Periodic publisher of cycle stats and hot-water state."""

    def __init__(
        self,
        appliances: Sequence[TrackedAppliance],
        tanks: Sequence[TrackedTank],
        fetch_history: FetchHistory,
        fetch_float: FetchFloat,
        publish: Publish,
        *,
        history_hours: int = 336,  # 14 days
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.appliances = list(appliances)
        self.tanks = list(tanks)
        self._fetch_history = fetch_history
        self._fetch_float = fetch_float
        self._publish = publish
        self._history_hours = history_hours
        self._now_fn = now_fn or datetime.now

        # Persistent per-tank observer state.
        self._estimators: dict[str, HotWaterEstimator] = {}
        self._last_tick: datetime | None = None
        self._tank_heating_kwh_today: dict[str, float] = {}
        self._today: date | None = None

    # -- appliances ---------------------------------------------------------

    def _detect(self, app: TrackedAppliance, rows: list[dict[str, Any]]) -> list[DetectedCycle]:
        if app.signal_kind == "status":
            samples = status_samples_from_ha_history(rows, app.power_attr)
            return detect_cycles_from_status(
                samples, power_scale=app.power_scale, running_power_w=app.running_power_w
            )
        if app.signal_kind == "runstate":
            samples = run_samples_from_ha_history(rows)
            return detect_cycles_from_runstate(samples, assumed_power_kw=app.assumed_power_kw)
        samples = power_samples_from_ha_history(rows)
        return detect_cycles_from_power(samples)

    async def _run_appliance(self, app: TrackedAppliance, today: date) -> list[PublishedSensor]:
        rows = await self._fetch_history(app.signal_entity, self._history_hours)
        cycles = self._detect(app, rows)
        stats = CycleStats.from_cycles(
            cycles,
            seed_duration_min=app.seed_duration_min,
            seed_energy_kwh=app.seed_energy_kwh,
            require_energy=(app.signal_kind != "runstate"),
        )
        last = next((c for c in reversed(cycles) if c.complete), None)
        draw_today = round(
            sum(c.energy_kwh for c in cycles if c.complete and c.start.date() == today), 3
        )
        return build_load_sensors(app.id, app.name, stats, last, draw_today)

    # -- tanks --------------------------------------------------------------

    def _estimator_for(self, tank: TrackedTank) -> HotWaterEstimator:
        est = self._estimators.get(tank.id)
        if est is None:
            model = WaterTankModel(
                volume_litres=tank.volume_litres,
                t_cold_c=tank.t_cold_c,
                t_max_c=tank.t_max_c,
                ua_w_per_k=tank.ua_w_per_k,
            )
            est = HotWaterEstimator(model)  # starts full; auto-anchors on first cut-off
            self._estimators[tank.id] = est
        return est

    async def _run_tank(
        self, tank: TrackedTank, now: datetime, dt_minutes: float, today: date
    ) -> list[PublishedSensor]:
        est = self._estimator_for(tank)
        raw_power_w = await self._fetch_float(tank.power_entity)
        heating_kw = max(0.0, (raw_power_w or 0.0) * tank.power_scale) / 1000.0

        if dt_minutes > 0:
            est.update(dt_minutes, heating_kw)
            self._tank_heating_kwh_today[tank.id] = self._tank_heating_kwh_today.get(
                tank.id, 0.0
            ) + heating_kw * (dt_minutes / 60.0)

        # Daily draw ~= heating energy today minus standing losses (full->full window).
        heating_today = self._tank_heating_kwh_today.get(tank.id, 0.0)
        hours_today = (now.hour * 60 + now.minute) / 60.0 or 0.001
        losses = est.tank.avg_loss_kw(est.temperature_c()) * hours_today
        draw_today = max(0.0, heating_today - losses)
        return build_hot_water_sensors(tank.id, tank.name, est, draw_today)

    # -- tick ---------------------------------------------------------------

    async def run_once(self) -> int:
        """Run one publish cycle. Returns the number of sensors published."""
        now = self._now_fn()
        today = now.date()
        dt_minutes = 0.0
        if self._last_tick is not None:
            dt_minutes = max(0.0, (now - self._last_tick).total_seconds() / 60.0)
        # Reset daily accumulators at midnight rollover.
        if self._today != today:
            self._tank_heating_kwh_today = {}
            self._today = today

        sensors: list[PublishedSensor] = []
        for app in self.appliances:
            try:
                sensors.extend(await self._run_appliance(app, today))
            except Exception as exc:
                logger.warning("Cycle publish failed for appliance %s: %s", app.id, exc)
        for tank in self.tanks:
            try:
                sensors.extend(await self._run_tank(tank, now, dt_minutes, today))
            except Exception as exc:
                logger.warning("Hot-water publish failed for tank %s: %s", tank.id, exc)

        self._last_tick = now
        published = await self._publish(sensors)
        logger.info("Published %d/%d darkstar_* sensors", published, len(sensors))
        return published


def build_tracked_from_config(
    config: dict[str, Any],
) -> tuple[list[TrackedAppliance], list[TrackedTank]]:
    """Build tracked appliances/tanks from the darkstar config (config-driven).

    Appliances come from ``deferrable_loads`` entries that have a tracking
    signal; tanks come from ``water_heaters`` entries flagged ``type: thermal``
    with a power sensor and tank geometry.
    """
    appliances: list[TrackedAppliance] = []
    loads_cfg: list[dict[str, Any]] = config.get("deferrable_loads", []) or []
    for cfg in loads_cfg:
        if not cfg.get("enabled", True):
            continue
        signal = cfg.get("running_sensor") or cfg.get("status_sensor") or cfg.get("power_sensor")
        if not (cfg.get("id") and signal):
            continue
        kind = "status" if (cfg.get("running_sensor") or cfg.get("status_sensor")) else "power"
        appliances.append(
            TrackedAppliance(
                id=str(cfg["id"]),
                name=str(cfg.get("name", cfg["id"])),
                signal_entity=str(signal),
                signal_kind=str(cfg.get("signal_kind", kind)),
                power_attr=str(cfg.get("power_attr", "power")),
                power_scale=float(cfg.get("power_scale", 1.0)),
                running_power_w=cfg.get("running_power_w"),
                seed_duration_min=float(cfg.get("duration_min", 120.0)),
                seed_energy_kwh=float(cfg.get("energy_kwh", 1.0)),
                assumed_power_kw=cfg.get("assumed_power_kw"),
            )
        )

    tanks: list[TrackedTank] = []
    heaters_cfg: list[dict[str, Any]] = config.get("water_heaters", []) or []
    for cfg in heaters_cfg:
        if not cfg.get("enabled", True) or cfg.get("type") != "thermal":
            continue
        power = cfg.get("power_sensor") or cfg.get("sensor")
        if not (cfg.get("id") and power and cfg.get("volume_litres")):
            continue
        tanks.append(
            TrackedTank(
                id=str(cfg["id"]),
                name=str(cfg.get("name", cfg["id"])),
                power_entity=str(power),
                volume_litres=float(cfg["volume_litres"]),
                t_cold_c=float(cfg.get("t_cold_c", 10.0)),
                t_max_c=float(cfg.get("t_max_c", 85.0)),
                ua_w_per_k=float(cfg.get("ua_w_per_k", 2.0)),
                power_scale=float(cfg.get("power_scale", 1.0)),
            )
        )
    return appliances, tanks


async def run_publisher_loop(
    config: dict[str, Any],
    *,
    interval_s: float = 300.0,
    history_hours: int = 336,
) -> None:
    """Deploy-ready periodic loop wiring real HA I/O to the publisher service.

    Start it once from the add-on startup, e.g.::

        asyncio.create_task(run_publisher_loop(app_config))

    Reads HA history/state and POSTs ``sensor.darkstar_*`` (sensor writes only -
    controls no hardware). Returns immediately if nothing is configured.
    """
    import asyncio
    from datetime import timedelta

    import httpx

    from backend.core import secrets
    from backend.core.ha_client import get_ha_sensor_float, make_ha_headers
    from backend.learning.cycle_publisher import publish_sensors

    appliances, tanks = build_tracked_from_config(config)
    if not appliances and not tanks:
        logger.info("Cycle publisher: no tracked loads/tanks configured; not starting")
        return

    ha_cfg = secrets.load_home_assistant_config()
    base_url = ha_cfg.get("url")
    token = ha_cfg.get("token")
    if not base_url or not token:
        logger.warning("Cycle publisher: HA url/token missing; not starting")
        return

    async def fetch_history(entity_id: str, hours: int) -> list[dict[str, Any]]:
        now = datetime.now().astimezone()
        start = now - timedelta(hours=hours)
        api_url = f"{base_url.rstrip('/')}/api/history/period/{start.isoformat()}"
        params: dict[str, Any] = {
            "filter_entity_id": entity_id,
            "end_time": now.isoformat(),
            "significant_changes_only": False,
            "minimal_response": False,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(api_url, headers=make_ha_headers(token), params=params)
            resp.raise_for_status()
            data = resp.json()
        return data[0] if data else []

    async def publish(sensors: list[PublishedSensor]) -> int:
        return await publish_sensors(sensors, base_url, token)

    service = DeferrablePublisherService(
        appliances,
        tanks,
        fetch_history,
        get_ha_sensor_float,
        publish,
        history_hours=history_hours,
    )
    logger.info(
        "Cycle publisher started: %d appliance(s), %d tank(s), every %.0fs",
        len(appliances),
        len(tanks),
        interval_s,
    )
    while True:
        try:
            await service.run_once()
        except Exception as exc:
            logger.warning("Cycle publisher tick failed: %s", exc)
        await asyncio.sleep(interval_s)


__all__ = [
    "DeferrablePublisherService",
    "TrackedAppliance",
    "TrackedTank",
    "build_tracked_from_config",
    "run_publisher_loop",
]
