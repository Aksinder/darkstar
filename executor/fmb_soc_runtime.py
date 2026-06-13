"""
FMB SoC estimator — runtime wiring (config + live read + persistence + publish).

Wraps the pure logic in ``fmb_soc_estimator.py``: reads the Easee sensors, advances
the estimate, persists a small JSON blob across add-on restarts, and publishes
``sensor.darkstar_fmb_soc_estimate`` (+ the learned consumption rate) so the planner
can read it as the FMB ``soc_sensor``. Default OFF.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from .fmb_soc_estimator import (
    FmbSocConfig,
    FmbSocInputs,
    FmbSocState,
    initial_state,
    update_fmb_soc,
)

logger = logging.getLogger("darkstar.fmb_soc")

_TRUEISH = {"on", "home", "true", "charging", "connected", "plugged", "1"}


@dataclass
class FmbSocRuntimeConfig:
    """Full ``executor.fmb_soc_estimator`` config: model params + entity wiring."""

    enabled: bool = False
    pure: FmbSocConfig = field(default_factory=FmbSocConfig)
    lifetime_energy_entity: str | None = None
    power_entity: str | None = None
    plug_entity: str | None = None
    enabled_switch_entity: str | None = None
    dynamic_limit_entity: str | None = None
    status_entity: str | None = None
    publish_entity_id: str = "sensor.darkstar_fmb_soc_estimate"
    publish_rate_entity_id: str = "sensor.darkstar_fmb_consumption_rate"
    state_path: str = "data/fmb_soc_state.json"
    save_interval_s: float = 120.0


def _f(v: Any) -> float | None:
    if v is None or v in ("unknown", "unavailable", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_fmb_soc_config(executor_data: dict[str, Any]) -> FmbSocRuntimeConfig | None:
    """Build runtime config from ``executor.fmb_soc_estimator``; None if absent."""
    raw_any = executor_data.get("fmb_soc_estimator")
    if not isinstance(raw_any, dict):
        return None
    raw = cast("dict[str, Any]", raw_any)
    enabled = bool(raw.get("enabled", False))
    pure = FmbSocConfig(
        enabled=enabled,
        capacity_kwh=float(raw.get("capacity_kwh", 28.0)),
        charge_efficiency=float(raw.get("charge_efficiency", 0.9)),
        prior_consumption_kwh_per_day=float(raw.get("prior_consumption_kwh_per_day", 5.6)),
        learn_alpha=float(raw.get("learn_alpha", 0.3)),
        min_consumption_kwh_per_day=float(raw.get("min_consumption_kwh_per_day", 0.5)),
        max_consumption_kwh_per_day=float(raw.get("max_consumption_kwh_per_day", 20.0)),
        min_anchor_energy_kwh=float(raw.get("min_anchor_energy_kwh", 2.0)),
        full_idle_min_s=float(raw.get("full_idle_min_s", 300.0)),
        full_offered_min_a=float(raw.get("full_offered_min_a", 5.0)),
        floor_soc=float(raw.get("floor_soc", 0.0)),
        initial_soc=float(raw.get("initial_soc", 50.0)),
    )
    return FmbSocRuntimeConfig(
        enabled=enabled,
        pure=pure,
        lifetime_energy_entity=raw.get("lifetime_energy_entity") or None,
        power_entity=raw.get("power_entity") or None,
        plug_entity=raw.get("plug_entity") or None,
        enabled_switch_entity=raw.get("enabled_switch_entity") or None,
        dynamic_limit_entity=raw.get("dynamic_limit_entity") or None,
        status_entity=raw.get("status_entity") or None,
        publish_entity_id=raw.get("publish_entity_id") or "sensor.darkstar_fmb_soc_estimate",
        publish_rate_entity_id=(
            raw.get("publish_rate_entity_id") or "sensor.darkstar_fmb_consumption_rate"
        ),
        state_path=raw.get("state_path") or "data/fmb_soc_state.json",
        save_interval_s=float(raw.get("save_interval_s", 120.0)),
    )


class FmbSocEstimator:
    """Stateful runtime: reads Easee, advances the estimate, persists, publishes."""

    def __init__(self, cfg: FmbSocRuntimeConfig):
        self.cfg = cfg
        self._state: FmbSocState | None = None
        self._last_save: float = 0.0

    # --- persistence ---
    def _load_state(self) -> FmbSocState:
        try:
            with Path(self.cfg.state_path).open(encoding="utf-8") as fh:
                data = cast("dict[str, Any]", json.load(fh))
            return FmbSocState(**data)
        except (OSError, ValueError, TypeError):
            logger.info("FMB SoC: no valid persisted state, seeding at %.0f%%", self.cfg.pure.initial_soc)
            return initial_state(self.cfg.pure)

    def _save_state(self, state: FmbSocState) -> None:
        try:
            p = Path(self.cfg.state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                json.dump(asdict(state), fh)
        except OSError as e:
            logger.warning("FMB SoC: failed to persist state: %s", e)

    # --- reads ---
    async def _read_f(self, ha: Any, entity: str | None, default: float | None = None) -> float | None:
        if not entity:
            return default
        v = _f(await ha.get_state_value(entity))
        return v if v is not None else default

    async def _read_bool(self, ha: Any, entity: str | None, default: bool) -> bool:
        if not entity:
            return default
        v = await ha.get_state_value(entity)
        if v is None:
            return default
        return str(v).strip().lower() in _TRUEISH

    async def _read_s(self, ha: Any, entity: str | None) -> str | None:
        if not entity:
            return None
        v = await ha.get_state_value(entity)
        return str(v) if v is not None else None

    async def run(self, ha: Any, now_ts: float, shadow: bool = False) -> dict[str, Any]:
        """One estimator cycle. Publishing happens even in shadow mode (it is observational)."""
        cfg = self.cfg
        if not cfg.enabled:
            return {"enabled": False}
        if self._state is None:
            self._state = self._load_state()

        energy, power, plug, sw, limit, status = await asyncio.gather(
            self._read_f(ha, cfg.lifetime_energy_entity),
            self._read_f(ha, cfg.power_entity, 0.0),
            self._read_bool(ha, cfg.plug_entity, True),
            self._read_bool(ha, cfg.enabled_switch_entity, False),
            self._read_f(ha, cfg.dynamic_limit_entity),
            self._read_s(ha, cfg.status_entity),
        )
        inp = FmbSocInputs(
            now_ts=now_ts,
            lifetime_energy_kwh=energy,
            power_w=power or 0.0,
            plugged=plug,
            charger_enabled=sw,
            dynamic_limit_a=limit,
            status=status,
        )
        new_state, dbg = update_fmb_soc(self._state, inp, cfg.pure)
        self._state = new_state

        await self._publish(ha, new_state)

        if (now_ts - self._last_save) >= cfg.save_interval_s or bool(dbg.get("learned")):
            self._save_state(new_state)
            self._last_save = now_ts

        logger.info(
            "FMB SoC: %.1f%% rate=%.1f kWh/d charging=%s full=%s (since_full=%.1f kWh)",
            new_state.soc_pct,
            new_state.learned_rate_kwh_per_day,
            dbg.get("charging", False),
            new_state.full_latched,
            new_state.energy_since_anchor_kwh,
        )
        return {"enabled": True, "soc": new_state.soc_pct, "debug": dbg}

    async def _publish(self, ha: Any, st: FmbSocState) -> None:
        await ha.set_state(
            self.cfg.publish_entity_id,
            str(round(st.soc_pct, 1)),
            {
                "unit_of_measurement": "%",
                "device_class": "battery",
                "state_class": "measurement",
                "friendly_name": "FMB uppskattad SoC",
                "icon": "mdi:battery-charging",
                "learned_rate_kwh_per_day": round(st.learned_rate_kwh_per_day, 2),
                "energy_since_full_kwh": round(st.energy_since_anchor_kwh, 2),
                "full": st.full_latched,
            },
        )
        await ha.set_state(
            self.cfg.publish_rate_entity_id,
            str(round(st.learned_rate_kwh_per_day, 2)),
            {
                "unit_of_measurement": "kWh/d",
                "state_class": "measurement",
                "friendly_name": "FMB inlärd förbrukning",
                "icon": "mdi:speedometer",
            },
        )
