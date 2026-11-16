"""Configuration of spa bathing times (``water_heaters[id=spa].bathing``).

Shared by the planner (which turns it into a ``BathingSpec`` for kepler, or simulates
the legacy plan in shadow mode) and the executor (which runs the in-window, catch-up
and eco-floor rules), so both sides read the block the same way.

Defaults are applied HERE, in code, never by the template merge:
``backend/config_migration.template_aware_merge`` merges ``water_heaters`` by ``id``
with a shallow ``dict.update``, so a user's ``bathing:`` block replaces any template
block wholesale. The add-on config POST also replaces lists, so the owner always
supplies the whole spa entry including ``bathing``.

Nothing in here raises on bad input. A nonsensical value is clamped or replaced by
its default, with a logged warning; a block that cannot work at all (no calendar, no
temperature sensor) is reported by :func:`bathing_config_problem`, and the planner
then falls back to the legacy daily-kWh plan.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("darkstar.core.spa_bathing_config")

LEARNED_PARAMS_PATH = "data/spa_thermal_params.json"

# Frost guard: an eco floor below this would let the water get cold enough that the
# pump and pipes are at risk in a cold spell, so it is never planned for.
ECO_FLOOR_MIN_C = 20.0
# Plausible Newtonian cooling constants for an insulated, lidded spa (1/h). Outside
# this a typo (e.g. 0.21 for 0.021) would make the model wildly wrong.
COOLING_K_MIN = 0.005
COOLING_K_MAX = 0.08
AMBIENT_MIN_C = -40.0
AMBIENT_MAX_C = 45.0


@dataclass(frozen=True)
class BathingConfig:
    enabled: bool = False
    shadow: bool = True
    window_entities: tuple[str, ...] = ()
    window_event_match: str = ""
    ready_temp_c: float = 38.0
    eco_floor_c: float = 28.0
    temp_max_c: float = 40.0
    ready_margin_minutes: float = 30.0
    temp_sensor: str = ""
    ambient_sensor: str = ""
    ambient_default_c: float = 10.0
    heat_capacity_kwh_per_c: float = 0.9
    cooling_k_per_h: float = 0.021
    ready_shortfall_sek_per_c: float = 5.0
    floor_shortfall_sek_per_c: float = 2.0  # interpreted per degC-hour
    hold_tolerance_c: float = 1.0
    overshoot_penalty_sek_per_c: float = 10.0
    catch_up_margin_c: float = 0.5
    # Must exceed the bridge's 30 C fan_only threshold until the owner changes it (B3).
    floor_write_temp_c: float = 31.0
    learn: bool = True


_DEFAULTS = BathingConfig()


def _as_bool(raw: Any, default: bool, key: str) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int | float):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    logger.warning("spa bathing: %s=%r is not a boolean, using %s", key, raw, default)
    return default


def _as_float(raw: Any, default: float, key: str) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool):
        logger.warning("spa bathing: %s=%r is not a number, using %s", key, raw, default)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("spa bathing: %s=%r is not a number, using %s", key, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("spa bathing: %s=%r is not finite, using %s", key, raw, default)
        return default
    return value


def _non_negative(value: float, default: float, key: str) -> float:
    if value < 0.0:
        logger.warning("spa bathing: %s=%s is negative, using %s", key, value, default)
        return default
    return value


def _as_entities(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list | tuple):
        logger.warning("spa bathing: window_entities=%r is not a list, ignoring it", raw)
        return ()
    items = cast("list[Any] | tuple[Any, ...]", raw)
    return tuple(str(e).strip() for e in items if str(e or "").strip())


def parse_bathing_config(raw: Mapping[str, Any] | None) -> BathingConfig | None:
    """The ``bathing`` block of one water heater, defaults applied, values sanitised.

    ``None`` when the block is missing (the heater has no bathing times). A block that
    is present but not a mapping is treated as missing, with a warning.
    """
    if raw is None:
        return None
    obj = cast("object", raw)  # YAML can hold anything here, whatever the annotation says
    if not isinstance(obj, Mapping):
        logger.warning("spa bathing: block is %s, not a mapping - ignoring it", type(obj).__name__)
        return None
    r = cast("Mapping[str, Any]", obj)
    d = _DEFAULTS

    enabled = _as_bool(r.get("enabled"), d.enabled, "enabled")
    # Shadow is the safe default: a block switched on without saying otherwise only
    # predicts and publishes; the plan and the executor stay exactly as today.
    shadow = _as_bool(r.get("shadow"), d.shadow, "shadow")
    learn = _as_bool(r.get("learn"), d.learn, "learn")

    eco_floor = _as_float(r.get("eco_floor_c"), d.eco_floor_c, "eco_floor_c")
    if eco_floor < ECO_FLOOR_MIN_C:
        logger.warning(
            "spa bathing: eco_floor_c=%s is below the frost guard, clamped to %s",
            eco_floor,
            ECO_FLOOR_MIN_C,
        )
        eco_floor = ECO_FLOOR_MIN_C

    temp_max = _as_float(r.get("temp_max_c"), d.temp_max_c, "temp_max_c")
    if temp_max <= eco_floor:
        fixed = max(d.temp_max_c, eco_floor + 1.0)
        logger.warning(
            "spa bathing: temp_max_c=%s is not above eco_floor_c=%s, using %s",
            temp_max,
            eco_floor,
            fixed,
        )
        temp_max = fixed

    ready = _as_float(r.get("ready_temp_c"), d.ready_temp_c, "ready_temp_c")
    if not (eco_floor < ready <= temp_max):
        fixed = min(max(d.ready_temp_c, eco_floor + 0.5), temp_max)
        logger.warning(
            "spa bathing: ready_temp_c=%s must lie in (eco_floor_c=%s, temp_max_c=%s], using %s",
            ready,
            eco_floor,
            temp_max,
            fixed,
        )
        ready = fixed

    cap = _as_float(
        r.get("heat_capacity_kwh_per_c"), d.heat_capacity_kwh_per_c, "heat_capacity_kwh_per_c"
    )
    if cap <= 0.0:
        logger.warning(
            "spa bathing: heat_capacity_kwh_per_c=%s must be > 0, using %s",
            cap,
            d.heat_capacity_kwh_per_c,
        )
        cap = d.heat_capacity_kwh_per_c

    k = _as_float(r.get("cooling_k_per_h"), d.cooling_k_per_h, "cooling_k_per_h")
    if not (COOLING_K_MIN <= k <= COOLING_K_MAX):
        clamped = min(max(k, COOLING_K_MIN), COOLING_K_MAX)
        logger.warning(
            "spa bathing: cooling_k_per_h=%s is outside [%s, %s], clamped to %s",
            k,
            COOLING_K_MIN,
            COOLING_K_MAX,
            clamped,
        )
        k = clamped

    ambient_default = _as_float(
        r.get("ambient_default_c"), d.ambient_default_c, "ambient_default_c"
    )
    if not (AMBIENT_MIN_C <= ambient_default <= AMBIENT_MAX_C):
        logger.warning(
            "spa bathing: ambient_default_c=%s is implausible, using %s",
            ambient_default,
            d.ambient_default_c,
        )
        ambient_default = d.ambient_default_c

    def nn(key: str, default: float) -> float:
        return _non_negative(_as_float(r.get(key), default, key), default, key)

    return BathingConfig(
        enabled=enabled,
        shadow=shadow,
        window_entities=_as_entities(r.get("window_entities")),
        window_event_match=str(r.get("window_event_match") or "").strip(),
        ready_temp_c=ready,
        eco_floor_c=eco_floor,
        temp_max_c=temp_max,
        ready_margin_minutes=nn("ready_margin_minutes", d.ready_margin_minutes),
        temp_sensor=str(r.get("temp_sensor") or "").strip(),
        ambient_sensor=str(r.get("ambient_sensor") or "").strip(),
        ambient_default_c=ambient_default,
        heat_capacity_kwh_per_c=cap,
        cooling_k_per_h=k,
        ready_shortfall_sek_per_c=nn("ready_shortfall_sek_per_c", d.ready_shortfall_sek_per_c),
        floor_shortfall_sek_per_c=nn("floor_shortfall_sek_per_c", d.floor_shortfall_sek_per_c),
        hold_tolerance_c=nn("hold_tolerance_c", d.hold_tolerance_c),
        overshoot_penalty_sek_per_c=nn(
            "overshoot_penalty_sek_per_c", d.overshoot_penalty_sek_per_c
        ),
        catch_up_margin_c=nn("catch_up_margin_c", d.catch_up_margin_c),
        floor_write_temp_c=_as_float(
            r.get("floor_write_temp_c"), d.floor_write_temp_c, "floor_write_temp_c"
        ),
        learn=learn,
    )


def bathing_config_problem(cfg: BathingConfig) -> str | None:
    """Why this (enabled) block cannot drive a bathing plan, or ``None`` if it can.

    Clamping already fixed every numeric value; what is left are the inputs that have
    no safe default: without a calendar there are no windows, and without a
    temperature sensor there is no state to start the model from.
    """
    if not cfg.window_entities:
        return "no_window_entities"
    if not cfg.temp_sensor:
        return "no_temp_sensor"
    return None


def heater_bathing_config(heater_cfg: object) -> BathingConfig | None:
    """``parse_bathing_config`` of one ``water_heaters[]`` entry (None-safe)."""
    if not isinstance(heater_cfg, Mapping):
        return None
    return parse_bathing_config(cast("Mapping[str, Any]", heater_cfg).get("bathing"))


def any_bathing_enabled(config: object) -> bool:
    """True when any ``water_heaters[]`` entry has ``bathing.enabled: true``."""
    if not isinstance(config, Mapping):
        return False
    heaters: Any = cast("Mapping[str, Any]", config).get("water_heaters") or []
    if not isinstance(heaters, list):
        return False
    for h in cast("list[Any]", heaters):
        cfg = heater_bathing_config(h)
        if cfg is not None and cfg.enabled:
            return True
    return False


def load_learned_params(
    path: str = LEARNED_PARAMS_PATH,
    max_age_days: float = 45.0,
) -> dict[str, float] | None:
    """Learned ``heat_capacity_kwh_per_c`` / ``cooling_k_per_h`` (written by phase 2).

    ``None`` when the file is missing, unreadable, stale (``updated_at`` older than
    ``max_age_days``, or missing/naive), or holds an implausible value — the caller
    then uses the configured values. Never raises.
    """
    p = Path(path)
    try:
        if not p.is_file():
            return None
        data: Any = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("spa bathing: learned params %s unreadable (%s)", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("spa bathing: learned params %s is not an object", path)
        return None
    d = cast("dict[str, Any]", data)
    try:
        updated = datetime.fromisoformat(str(d.get("updated_at")))
    except (TypeError, ValueError):
        logger.warning("spa bathing: learned params %s has no valid updated_at", path)
        return None
    if updated.tzinfo is None:
        # The container runs local time; a naive stamp cannot be aged reliably.
        logger.warning("spa bathing: learned params %s updated_at is naive - ignored", path)
        return None
    age_days = (datetime.now(UTC) - updated).total_seconds() / 86400.0
    if age_days > max_age_days or age_days < -1.0:
        logger.info("spa bathing: learned params %s are %.1f days old - ignored", path, age_days)
        return None
    try:
        cap = float(d["heat_capacity_kwh_per_c"])
        k = float(d["cooling_k_per_h"])
    except (KeyError, TypeError, ValueError):
        logger.warning("spa bathing: learned params %s lack C or k", path)
        return None
    if not (math.isfinite(cap) and cap > 0.0):
        return None
    if not (math.isfinite(k) and COOLING_K_MIN <= k <= COOLING_K_MAX):
        return None
    return {"heat_capacity_kwh_per_c": cap, "cooling_k_per_h": k}
