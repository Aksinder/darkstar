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
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from backend.learning.cycle_learning import (
    CycleStats,
    DetectedCycle,
    PowerSample,
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
    build_phase_recommendation_sensors,
    build_phase_sensors,
)
from backend.learning.phase_learning import (
    PhaseMapping,
    learn_device_phase,
    phase_imbalance_w,
    reconstruct_phase_fractions,
)
from backend.learning.phase_recommend import recommend_phase_moves
from planner.hot_water import HotWaterEstimator
from planner.thermal import WaterTankModel

logger = logging.getLogger("darkstar.cycle_publisher_service")

# Injected I/O signatures.
FetchHistory = Callable[[str, int], Awaitable[list[dict[str, Any]]]]
FetchFloat = Callable[[str], Awaitable[float | None]]
Publish = Callable[[list[PublishedSensor]], Awaitable[int]]


class _CumulativeMeter:
    """Step lookup over a cumulative-energy (kWh) sensor's history.

    Timestamps are kept timezone-aware (as HA provides them), so deltas compare
    cleanly against the detector's tz-aware cycle boundaries.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        pts: list[tuple[datetime, float]] = []
        for r in rows:
            try:
                value = float(r["state"])
            except (TypeError, ValueError, KeyError):
                continue
            ts_raw = r.get("last_changed") or r.get("last_updated")
            if not ts_raw:
                continue
            pts.append((datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")), value))
        pts.sort(key=lambda x: x[0])
        self._pts = pts

    def _value_at(self, ts: datetime) -> float | None:
        val: float | None = None
        for t, v in self._pts:
            if t <= ts:
                val = v
            else:
                break
        return val

    def delta(self, start: datetime, end: datetime) -> float | None:
        """Energy consumed between start and end (kWh, >= 0), or None if unknown."""
        a = self._value_at(start)
        b = self._value_at(end)
        if a is None or b is None:
            return None
        return max(0.0, b - a)


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
    # Optional cumulative-energy sensor (kWh) for accurate per-cycle energy. When
    # set, the energy of the last cycle and today's draw are taken from this
    # meter's delta instead of the (often unreliable) status power attribute.
    energy_sensor: str | None = None


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

        # Prefer a real cumulative-energy meter when configured: the status power
        # attribute is often unreliable, but the plug's kWh meter gives an
        # accurate per-cycle delta. Compared against the (tz-aware) cycle
        # boundaries, so there is no naive/aware datetime mismatch.
        if app.energy_sensor:
            energy_rows = await self._fetch_history(app.energy_sensor, self._history_hours)
            meter = _CumulativeMeter(energy_rows)
            if last is not None:
                d = meter.delta(last.start, last.end)
                if d is not None:
                    last = replace(last, energy_kwh=round(d, 3))
            today_deltas = [
                meter.delta(c.start, c.end)
                for c in cycles
                if c.complete and c.start.date() == today
            ]
            today_deltas = [d for d in today_deltas if d is not None]
            if today_deltas:
                draw_today = round(sum(today_deltas), 3)

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
                energy_sensor=cfg.get("energy_sensor"),
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


# ===========================================================================
# Phase-aware observability (Observe phase of phase-aware-load-modeling.md)
# ===========================================================================


@dataclass(frozen=True)
class TrackedPhaseDevice:
    """A metered device whose electrical phase we want to learn."""

    id: str
    name: str
    power_entity: str
    power_scale: float = 1.0


@dataclass(frozen=True)
class PhaseSources:
    """Entities describing the per-phase grid meter and the inverter AC output."""

    phase_a_entity: str
    phase_b_entity: str
    phase_c_entity: str
    # Inverter AC output S = sum(entity * scale); used to reconstruct per-phase
    # house load (load_phase = grid_phase + S/3). Typically battery + PV power.
    inverter_entities: tuple[tuple[str, float], ...] = ()


def _sum_series(
    serieses: Sequence[Sequence[PowerSample]], scales: Sequence[float]
) -> list[PowerSample]:
    """Forward-fill several PowerSample series onto their merged timeline and sum."""
    if not serieses:
        return []
    timeline = sorted({s.ts for ser in serieses for s in ser})
    idx = [0] * len(serieses)
    last: list[float | None] = [None] * len(serieses)
    out: list[PowerSample] = []
    for ts in timeline:
        for i, ser in enumerate(serieses):
            while idx[i] < len(ser) and ser[idx[i]].ts <= ts:
                last[i] = ser[idx[i]].power_w
                idx[i] += 1
        if all(v is not None for v in last):
            total = sum((v or 0.0) * scales[i] for i, v in enumerate(last))
            out.append(PowerSample(ts=ts, power_w=total))
    return out


PersistModel = Callable[[dict[str, Any]], Awaitable[None]]


class PhaseObserverService:
    """Periodic learner/publisher of per-phase load + device->phase mappings.

    Read-only: it learns each device's phase by correlation, reconstructs the
    per-phase house-load split, publishes ``sensor.darkstar_phase_*`` and
    ``sensor.darkstar_<device>_phase``, and (optionally) persists the learned load
    fractions so the planner's realism simulation can use real measured shares.
    Controls no hardware.
    """

    def __init__(
        self,
        devices: Sequence[TrackedPhaseDevice],
        sources: PhaseSources,
        fetch_history: FetchHistory,
        fetch_float: FetchFloat,
        publish: Publish,
        *,
        persist: PersistModel | None = None,
        history_hours: int = 72,
        import_price_sek_kwh: float = 2.0,
        export_price_sek_kwh: float = 0.5,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.devices = list(devices)
        self.sources = sources
        self._fetch_history = fetch_history
        self._fetch_float = fetch_float
        self._publish = publish
        self._persist = persist
        self._history_hours = history_hours
        self._import_price = import_price_sek_kwh
        self._export_price = export_price_sek_kwh
        self._now_fn = now_fn or datetime.now

    async def _phase_grid_series(
        self,
    ) -> tuple[list[PowerSample], list[PowerSample], list[PowerSample]]:
        a = power_samples_from_ha_history(
            await self._fetch_history(self.sources.phase_a_entity, self._history_hours)
        )
        b = power_samples_from_ha_history(
            await self._fetch_history(self.sources.phase_b_entity, self._history_hours)
        )
        c = power_samples_from_ha_history(
            await self._fetch_history(self.sources.phase_c_entity, self._history_hours)
        )
        return a, b, c

    async def _inverter_series(self) -> list[PowerSample]:
        if not self.sources.inverter_entities:
            return []
        serieses: list[list[PowerSample]] = []
        scales: list[float] = []
        for entity, scale in self.sources.inverter_entities:
            serieses.append(
                power_samples_from_ha_history(
                    await self._fetch_history(entity, self._history_hours)
                )
            )
            scales.append(scale)
        return _sum_series(serieses, scales)

    async def _current_imbalance(self) -> float | None:
        a = await self._fetch_float(self.sources.phase_a_entity)
        b = await self._fetch_float(self.sources.phase_b_entity)
        c = await self._fetch_float(self.sources.phase_c_entity)
        if a is None or b is None or c is None:
            return None
        return round(phase_imbalance_w(a, b, c), 1)

    async def run_once(self) -> int:
        """Run one learn+publish cycle. Returns the number of sensors published."""
        pa, pb, pc = await self._phase_grid_series()
        inv = await self._inverter_series()
        estimate = reconstruct_phase_fractions(pa, pb, pc, inv)

        mappings: list[PhaseMapping] = []
        names: dict[str, str] = {}
        dev_series: dict[str, list[PowerSample]] = {}
        for dev in self.devices:
            names[dev.id] = dev.name
            try:
                rows = await self._fetch_history(dev.power_entity, self._history_hours)
                series = power_samples_from_ha_history(rows)
                if dev.power_scale != 1.0:
                    series = [PowerSample(s.ts, s.power_w * dev.power_scale) for s in series]
                dev_series[dev.id] = series
                mappings.append(learn_device_phase(series, pa, pb, pc, device_id=dev.id))
            except Exception as exc:
                logger.warning("Phase learn failed for device %s: %s", dev.id, exc)

        # Rebalancing recommendations (Recommend phase): replay history to rank
        # one-time device-to-phase moves by the grid import they would avoid.
        reco_inputs = [(m, dev_series[m.device_id]) for m in mappings if m.device_id in dev_series]
        recommendations = recommend_phase_moves(
            pa,
            pb,
            pc,
            reco_inputs,
            names=names,
            import_price_sek_kwh=self._import_price,
            export_price_sek_kwh=self._export_price,
        )

        current_imbalance = await self._current_imbalance()
        sensors = build_phase_sensors(
            estimate, mappings, device_names=names, current_imbalance_w=current_imbalance
        )
        sensors.extend(build_phase_recommendation_sensors(recommendations))

        if self._persist and estimate.fractions:
            try:
                await self._persist(
                    {
                        "fractions": estimate.fractions,
                        "load_w": estimate.load_w,
                        "imbalance_w": estimate.imbalance_w,
                        "samples": estimate.n_samples,
                        "devices": [
                            {
                                "id": m.device_id,
                                "phase": m.phase,
                                "load_type": m.load_type,
                                "confidence": m.confidence,
                            }
                            for m in mappings
                        ],
                        "recommendations": [
                            {
                                "device_id": r.device_id,
                                "device_name": r.device_name,
                                "from_phase": r.from_phase,
                                "to_phase": r.to_phase,
                                "annual_import_avoided_kwh": r.annual_import_avoided_kwh,
                                "annual_saving_sek": r.annual_saving_sek,
                                "confidence": r.confidence,
                            }
                            for r in recommendations
                        ],
                        "updated": self._now_fn().isoformat(),
                    }
                )
            except Exception as exc:
                logger.warning("Persisting phase model failed: %s", exc)

        published = await self._publish(sensors)
        logger.info("Published %d phase observability sensor(s)", published)
        return published


def build_phase_observer_from_config(
    config: dict[str, Any],
) -> tuple[list[TrackedPhaseDevice], PhaseSources] | None:
    """Build phase-observer inputs from config; None if not enabled/configured.

    Reads ``phase_observer`` (phase meter + inverter entities, explicit devices) and
    additionally learns the phase of any ``deferrable_loads`` entry that exposes a
    ``power_sensor`` - so existing config is reused without duplication.
    """
    obs: dict[str, Any] = config.get("phase_observer", {}) or {}
    if not obs.get("enabled", False):
        return None
    a = obs.get("phase_a_sensor")
    b = obs.get("phase_b_sensor")
    c = obs.get("phase_c_sensor")
    if not (a and b and c):
        logger.warning("phase_observer enabled but phase_a/b/c_sensor missing; not starting")
        return None

    inverter: list[tuple[str, float]] = []
    if obs.get("battery_power_sensor"):
        inverter.append(
            (str(obs["battery_power_sensor"]), float(obs.get("battery_power_scale", 1.0)))
        )
    if obs.get("pv_power_sensor"):
        inverter.append((str(obs["pv_power_sensor"]), float(obs.get("pv_power_scale", 1.0))))
    sources = PhaseSources(str(a), str(b), str(c), tuple(inverter))

    devices: list[TrackedPhaseDevice] = []
    seen: set[str] = set()
    obs_devices: list[dict[str, Any]] = obs.get("devices", []) or []
    for cfg in obs_devices:
        power = cfg.get("power_sensor") or cfg.get("power_entity")
        if not (cfg.get("id") and power):
            continue
        devices.append(
            TrackedPhaseDevice(
                id=str(cfg["id"]),
                name=str(cfg.get("name", cfg["id"])),
                power_entity=str(power),
                power_scale=float(cfg.get("power_scale", 1.0)),
            )
        )
        seen.add(str(cfg["id"]))

    deferrable: list[dict[str, Any]] = config.get("deferrable_loads", []) or []
    for cfg in deferrable:
        power = cfg.get("power_sensor")
        cid = cfg.get("id")
        if not (cid and power) or str(cid) in seen:
            continue
        devices.append(
            TrackedPhaseDevice(
                id=str(cid),
                name=str(cfg.get("name", cid)),
                power_entity=str(power),
                power_scale=float(cfg.get("power_scale", 1.0)),
            )
        )
        seen.add(str(cid))

    return devices, sources


async def run_phase_observer_loop(
    config: dict[str, Any],
    *,
    interval_s: float = 900.0,
    history_hours: int = 72,
) -> None:
    """Deploy-ready periodic loop for phase observability. No-op if not configured.

    Persists the learned load fractions to ``<config_dir>/phase_model.json`` so the
    planner can feed real per-phase shares into the realism simulation.
    """
    import asyncio
    import json
    from datetime import timedelta
    from pathlib import Path

    import httpx

    from backend.core import secrets
    from backend.core.ha_client import get_ha_sensor_float, make_ha_headers
    from backend.learning.cycle_publisher import publish_sensors

    built = build_phase_observer_from_config(config)
    if built is None:
        logger.info("Phase observer: not enabled/configured; not starting")
        return
    devices, sources = built

    ha_cfg = secrets.load_home_assistant_config()
    base_url = ha_cfg.get("url")
    token = ha_cfg.get("token")
    if not base_url or not token:
        logger.warning("Phase observer: HA url/token missing; not starting")
        return

    config_dir = str(config.get("config_dir") or "/config/darkstar")
    model_path = Path(config_dir) / "phase_model.json"

    obs_cfg: dict[str, Any] = config.get("phase_observer", {}) or {}
    import_price = float(obs_cfg.get("import_price_sek_kwh", 2.0))
    export_price = float(obs_cfg.get("export_price_sek_kwh", 0.5))

    async def fetch_history(entity_id: str, hours: int) -> list[dict[str, Any]]:
        now = datetime.now().astimezone()
        start = now - timedelta(hours=hours)
        api_url = f"{base_url.rstrip('/')}/api/history/period/{start.isoformat()}"
        params: dict[str, Any] = {
            "filter_entity_id": entity_id,
            "end_time": now.isoformat(),
            "significant_changes_only": False,
            "minimal_response": True,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(api_url, headers=make_ha_headers(token), params=params)
            resp.raise_for_status()
            data = resp.json()
        return data[0] if data else []

    async def publish(sensors: list[PublishedSensor]) -> int:
        return await publish_sensors(sensors, base_url, token)

    async def persist(model: dict[str, Any]) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    service = PhaseObserverService(
        devices,
        sources,
        fetch_history,
        get_ha_sensor_float,
        publish,
        persist=persist,
        history_hours=history_hours,
        import_price_sek_kwh=import_price,
        export_price_sek_kwh=export_price,
    )
    logger.info(
        "Phase observer started: %d device(s), every %.0fs -> %s",
        len(devices),
        interval_s,
        model_path,
    )
    while True:
        try:
            await service.run_once()
        except Exception as exc:
            logger.warning("Phase observer tick failed: %s", exc)
        await asyncio.sleep(interval_s)


__all__ = [
    "DeferrablePublisherService",
    "PhaseObserverService",
    "PhaseSources",
    "TrackedAppliance",
    "TrackedPhaseDevice",
    "TrackedTank",
    "build_phase_observer_from_config",
    "build_tracked_from_config",
    "run_phase_observer_loop",
    "run_publisher_loop",
]
