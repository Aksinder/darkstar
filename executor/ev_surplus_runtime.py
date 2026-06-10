"""
EV surplus controller — runtime wiring (config + live read + actuation).

Wraps the pure control law in ``ev_surplus.py``: reads the live sensors and each charger's
state from HA, calls ``compute_ev_surplus``, then actuates through the write-guard. Default
OFF. Tesla current is set via ``number.set_value``; the Easee via its flash-SAFE
``easee.set_charger_dynamic_limit`` service (NEVER the non-dynamic max/circuit limits).

Kept separate from the engine so the integration there is a two-line construct-and-call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

from .ev_surplus import (
    ChargerState,
    EVSurplusConfig,
    EVSurplusInputs,
    WriteGuardConfig,
    compute_ev_surplus,
    should_write_current,
)

logger = logging.getLogger("darkstar.ev_surplus")

_TRUEISH = {"on", "home", "true", "charging", "connected", "plugged", "1"}


@dataclass
class EVSurplusChargerCfg:
    """One charger's control wiring for the surplus controller."""

    id: str
    switch_entity: str | None = None  # on/off
    current_entity: str | None = None  # number.* set_value path (e.g. Tesla)
    easee_device_id: str | None = None  # easee.set_charger_dynamic_limit device (e.g. Easee)
    power_entity: str | None = None
    plug_entity: str | None = None
    home_entity: str | None = None  # device_tracker; absent => assume home
    home_states: tuple[str, ...] = ("home",)
    override_entity: str | None = None  # input_select auto/force_on/force_off
    priority: int = 0
    min_current_a: float = 6.0
    max_current_a: float = 16.0
    phases: int = 3
    voltage_v: float = 230.0

    @property
    def controllable(self) -> bool:
        return bool(self.current_entity or self.easee_device_id)


@dataclass
class EVSurplusRuntimeConfig:
    """Full executor.ev_surplus config: sources + policy + chargers."""

    enabled: bool = False
    pv_power_entity: str | None = None
    grid_power_entity: str | None = None  # signed, + import
    battery_power_entity: str | None = None  # signed, + charge
    battery_soc_entity: str | None = None
    price_entity: str | None = None
    remaining_solar_entity: str | None = None  # optional; absent => battery-assist tier inert
    policy: EVSurplusConfig = field(default_factory=EVSurplusConfig)
    guard: WriteGuardConfig = field(default_factory=WriteGuardConfig)
    chargers: list[EVSurplusChargerCfg] = field(default_factory=lambda: [])


def _f(v: Any) -> float | None:
    if v is None or v in ("unknown", "unavailable", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_ev_surplus_config(executor_data: dict[str, Any]) -> EVSurplusRuntimeConfig | None:
    """Build the runtime config from ``executor.ev_surplus``; None if absent."""
    raw_any = executor_data.get("ev_surplus")
    if not isinstance(raw_any, dict):
        return None
    raw = cast("dict[str, Any]", raw_any)
    pol = cast("dict[str, Any]", raw.get("policy", {}) or {})
    ba = cast("dict[str, Any]", pol.get("battery_assist", {}) or {})
    guard_raw = cast("dict[str, Any]", raw.get("write_guard", {}) or {})
    policy = EVSurplusConfig(
        enabled=bool(raw.get("enabled", False)),
        cheap_grid_price_sek=_f(pol.get("cheap_grid_price_sek")),
        cheap_grid_allowance_w=float(pol.get("cheap_grid_allowance_w", 3680.0)),
        battery_assist_enabled=bool(ba.get("enabled", False)),
        battery_assist_max_price_sek=float(ba.get("max_price_sek", 0.0)),
        battery_assist_min_remaining_solar_kwh=float(ba.get("min_remaining_solar_kwh", 8.0)),
        battery_assist_floor_soc=float(ba.get("floor_soc", 40.0)),
        battery_assist_allowance_w=float(ba.get("allowance_w", 6000.0)),
        gain=float(pol.get("gain", 0.5)),
        deadband_w=float(pol.get("deadband_w", 250.0)),
        current_step_a=float(pol.get("current_step_a", 2.0)),
        start_hysteresis=float(pol.get("start_hysteresis", 0.15)),
    )
    guard = WriteGuardConfig(
        min_step_a=float(guard_raw.get("min_step_a", 2.0)),
        min_interval_s=float(guard_raw.get("min_interval_s", 90.0)),
    )
    chargers: list[EVSurplusChargerCfg] = []
    for c in cast("list[dict[str, Any]]", raw.get("chargers", []) or []):
        if not c.get("id"):
            continue
        chargers.append(
            EVSurplusChargerCfg(
                id=str(c["id"]),
                switch_entity=c.get("switch_entity") or None,
                current_entity=c.get("current_entity") or None,
                easee_device_id=c.get("easee_device_id") or None,
                power_entity=c.get("power_entity") or None,
                plug_entity=c.get("plug_entity") or None,
                home_entity=c.get("home_entity") or None,
                home_states=tuple(c.get("home_states", ["home"])),
                override_entity=c.get("override_entity") or None,
                priority=int(c.get("priority", 0)),
                min_current_a=float(c.get("min_current_a", 6.0)),
                max_current_a=float(c.get("max_current_a", 16.0)),
                phases=int(c.get("phases", 3)),
                voltage_v=float(c.get("voltage_v", 230.0)),
            )
        )
    return EVSurplusRuntimeConfig(
        enabled=bool(raw.get("enabled", False)),
        pv_power_entity=raw.get("pv_power_entity") or None,
        grid_power_entity=raw.get("grid_power_entity") or None,
        battery_power_entity=raw.get("battery_power_entity") or None,
        battery_soc_entity=raw.get("battery_soc_entity") or None,
        price_entity=raw.get("price_entity") or None,
        remaining_solar_entity=raw.get("remaining_solar_entity") or None,
        policy=policy,
        guard=guard,
        chargers=chargers,
    )


class EVSurplusController:
    """Stateful runtime: reads HA, computes, actuates through the write-guard."""

    def __init__(self, cfg: EVSurplusRuntimeConfig):
        self.cfg = cfg
        # Per-charger write-guard memory.
        self._last_a: dict[str, float] = {}
        self._last_ts: dict[str, float] = {}
        self._last_switch: dict[str, bool] = {}

    async def _read_f(self, ha: Any, entity: str | None, default: float | None = None) -> float | None:
        if not entity:
            return default
        v = _f(await ha.get_state_value(entity))
        return v if v is not None else default

    async def _read_on(self, ha: Any, entity: str | None, states: tuple[str, ...], default: bool) -> bool:
        if not entity:
            return default
        v = await ha.get_state_value(entity)
        if v is None:
            return default
        return str(v).lower() in {s.lower() for s in states}

    async def _read_override(self, ha: Any, entity: str | None) -> str:
        if not entity:
            return "auto"
        v = await ha.get_state_value(entity)
        val = str(v).lower() if v else "auto"
        return val if val in ("auto", "force_on", "force_off") else "auto"

    async def run(self, ha: Any, now_ts: float, shadow: bool = False) -> dict[str, Any]:
        """One control cycle. Returns a summary (for logging / UI)."""
        cfg = self.cfg
        if not cfg.enabled or not cfg.chargers:
            return {"enabled": False}

        pv_w = (await self._read_f(ha, cfg.pv_power_entity, 0.0)) or 0.0
        grid_w = (await self._read_f(ha, cfg.grid_power_entity, 0.0)) or 0.0
        battery_w = (await self._read_f(ha, cfg.battery_power_entity, 0.0)) or 0.0
        soc = (await self._read_f(ha, cfg.battery_soc_entity, 100.0)) or 100.0
        price = (await self._read_f(ha, cfg.price_entity, 999.0)) or 999.0
        remaining_solar = (await self._read_f(ha, cfg.remaining_solar_entity, 0.0)) or 0.0

        states: list[ChargerState] = []
        for c in cfg.chargers:
            power = (await self._read_f(ha, c.power_entity, 0.0)) or 0.0
            plugged = await self._read_on(ha, c.plug_entity, ("on", "true", "plugged", "connected"), True)
            at_home = await self._read_on(ha, c.home_entity, c.home_states, True)
            override = await self._read_override(ha, c.override_entity)
            states.append(
                ChargerState(
                    id=c.id, plugged=plugged, at_home=at_home, enabled=True,
                    current_power_w=power, max_current_a=c.max_current_a,
                    min_current_a=c.min_current_a, phases=c.phases, voltage_v=c.voltage_v,
                    controllable=c.controllable, priority=c.priority, override=override,
                )
            )

        inputs = EVSurplusInputs(
            pv_w=pv_w, grid_w=grid_w, battery_w=battery_w, battery_soc_percent=soc,
            import_price_sek=price, remaining_solar_kwh=remaining_solar, chargers=states,
        )
        commands = compute_ev_surplus(inputs, cfg.policy)
        cfg_by_id = {c.id: c for c in cfg.chargers}

        applied: list[dict[str, Any]] = []
        for cmd in commands:
            ccfg = cfg_by_id.get(cmd.id)
            if ccfg is None:
                continue
            await self._actuate(ha, ccfg, cmd, now_ts, shadow)
            applied.append({"id": cmd.id, "on": cmd.switch_on, "a": cmd.set_current_a, "why": cmd.reason})

        logger.info("EV surplus: grid=%.0fW batt=%.0fW soc=%.0f%% price=%.2f -> %s",
                    grid_w, battery_w, soc, price, [(a["id"], a["on"], a["a"]) for a in applied])
        return {"enabled": True, "applied": applied}

    async def _actuate(self, ha: Any, ccfg: EVSurplusChargerCfg, cmd: Any, now_ts: float, shadow: bool) -> None:
        # Switch: only toggle on change.
        if ccfg.switch_entity is not None and self._last_switch.get(ccfg.id) != cmd.switch_on:
            if not shadow:
                svc = "turn_on" if cmd.switch_on else "turn_off"
                await ha.call_service("switch", svc, ccfg.switch_entity)
            self._last_switch[ccfg.id] = cmd.switch_on

        # Current: only when on, controllable, and the write-guard allows it.
        if not (cmd.switch_on and ccfg.controllable and cmd.set_current_a is not None):
            return
        new_a = float(cmd.set_current_a)
        if not should_write_current(
            self._last_a.get(ccfg.id), self._last_ts.get(ccfg.id), new_a, now_ts, self.cfg.guard
        ):
            return
        if not shadow:
            if ccfg.current_entity:
                await ha.call_service("number", "set_value", ccfg.current_entity, {"value": new_a})
            elif ccfg.easee_device_id:
                # FLASH-SAFE dynamic limit only (volatile/RAM). Never the max/circuit limits.
                await ha.call_service(
                    "easee", "set_charger_dynamic_limit", None,
                    {"device_id": ccfg.easee_device_id, "current": round(new_a), "time_to_live": 0},
                )
        self._last_a[ccfg.id] = new_a
        self._last_ts[ccfg.id] = now_ts
