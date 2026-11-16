"""Spa bathing times: the planner-side decision, shadow simulation and plan summary.

Pure functions (no I/O except the optional learned-params file read). The pipeline
calls :func:`decide_bathing` once per water heater that has a ``bathing`` block, just
before the solve, and

* mode ``solver``   -> attaches ``decision.spec`` to the heater (kepler then plans the
  spa on a temperature model instead of the daily-kWh floor);
* mode ``shadow``   -> leaves the heater untouched (the legacy plan ships unchanged)
  and, after the solve, runs :func:`simulate_plan` on that legacy plan so the
  predicted temperatures can be published and checked against the measured bath
  start temperature;
* mode ``fallback`` -> leaves the heater untouched and records why (unknown calendar,
  unreadable temperature, invalid block, ...). A window is NEVER invented;
* mode ``disabled`` -> nothing at all (byte-identical to a config without ``bathing``).

Datetimes: windows arrive as aware ISO strings from ``backend.core.ha_client`` and
``now`` must be aware. Slots are aware pandas Timestamps in production. A naive
``now`` or naive slots in the pipeline are a programming error here and lead to
``fallback`` (never a TypeError, never a mixed naive/aware comparison).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from backend.core.load_windows import windows_from_iso
from backend.core.spa_bathing_config import (
    AMBIENT_MAX_C,
    AMBIENT_MIN_C,
    LEARNED_PARAMS_PATH,
    BathingConfig,
    bathing_config_problem,
    load_learned_params,
    parse_bathing_config,
)
from backend.core.spa_thermal import simulate_temps_c
from planner.solver.types import (
    BathingReport,
    BathingSpec,
    BathWindow,
    WaterHeaterInput,
    planned_slot_hours,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger("darkstar.planner.spa_bathing")

BathingMode = Literal["solver", "shadow", "fallback", "disabled"]

# Measured spa temperatures outside this are a broken sensor, not water.
TEMP_VALID_MIN_C = 0.0
TEMP_VALID_MAX_C = 50.0
# A window whose predicted ready temperature is this close to the target is on track.
ON_TRACK_TOLERANCE_C = 0.5
# Heater kW above this counts as "element on" in the shadow simulation.
ON_THRESHOLD_KW = 0.01


class _Slot(Protocol):
    @property
    def start_time(self) -> datetime: ...

    @property
    def end_time(self) -> datetime: ...


@dataclass
class BathingDecision:
    heater_id: str
    mode: BathingMode
    reason: str | None
    spec: BathingSpec | None  # handed to the solver (mode "solver" only)
    windows: list[tuple[datetime, datetime]]  # all aware windows ending after now
    measured_temp_c: float | None
    ambient_c: float | None
    params_source: Literal["config", "learned"]
    # The spec as it would be (or is) handed to the solver: set in modes "solver" and
    # "shadow". Shadow mode simulates the legacy plan with it; the solver never sees it.
    spec_for_simulation: BathingSpec | None = None
    power_kw: float = 0.0
    config: BathingConfig | None = None


def _is_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() is not None


def _finite(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _disabled(heater_id: str, power_kw: float, cfg: BathingConfig | None) -> BathingDecision:
    return BathingDecision(
        heater_id=heater_id,
        mode="disabled",
        reason=None,
        spec=None,
        windows=[],
        measured_temp_c=None,
        ambient_c=None,
        params_source="config",
        power_kw=power_kw,
        config=cfg,
    )


def decide_bathing(
    *,
    heater_cfg: Mapping[str, Any],
    inputs: Mapping[str, Any] | None,
    heater: WaterHeaterInput,
    slots: Sequence[_Slot],
    first_slot_remaining_h: float | None,
    now: datetime,
    mean_outdoor_temp_c: float | None,
    learned_params_path: str = LEARNED_PARAMS_PATH,
) -> BathingDecision:
    """Which mode the spa is planned in this run, and the solver spec if any.

    ``first_slot_remaining_h`` is accepted for signature symmetry with the solver;
    the recursion's partial slot 0 is handled by kepler / :func:`simulate_plan`.
    """
    del first_slot_remaining_h  # the spec carries no slot geometry
    hid = heater.id
    cfg = parse_bathing_config(cast("Mapping[str, Any] | None", heater_cfg.get("bathing")))
    if cfg is None or not cfg.enabled or inputs is None:
        return _disabled(hid, heater.power_kw, cfg)

    def fallback(
        reason: str,
        windows: list[tuple[datetime, datetime]] | None = None,
        temp: float | None = None,
        ambient: float | None = None,
    ) -> BathingDecision:
        logger.warning("Spa bathing %s: FALLBACK to the legacy daily-kWh plan (%s)", hid, reason)
        return BathingDecision(
            heater_id=hid,
            mode="fallback",
            reason=reason,
            spec=None,
            windows=windows or [],
            measured_temp_c=temp,
            ambient_c=ambient,
            params_source="config",
            power_kw=heater.power_kw,
            config=cfg,
        )

    if not _is_aware(now):
        return fallback("naive_now")
    if slots and not _is_aware(slots[0].start_time):
        # Kepler compares slot times with window times; aware windows against naive
        # slots would be a mixed pairing. Never attempted.
        return fallback("naive_slots")

    # 1. Windows. None = the calendar (beyond load_windows' 6 h last-good) or a gate
    #    entity could not be read. An EMPTY list is a readable "no bath" answer.
    raw_windows = inputs.get("windows")
    parsed = windows_from_iso(raw_windows) if raw_windows is not None else None
    if parsed is None:
        return fallback("calendar_unknown")
    windows = [(s, e) for s, e in parsed if e > now]

    # 2. Measured temperature: the recursion's state 0.
    temp = _finite(inputs.get("temp_c"))
    if temp is None:
        return fallback("temp_unreadable", windows)
    if not (TEMP_VALID_MIN_C <= temp <= TEMP_VALID_MAX_C):
        return fallback("temp_out_of_range", windows, temp)

    # 3. A block that cannot work (no calendar / no sensor configured).
    problem = bathing_config_problem(cfg)
    if problem is not None:
        return fallback(f"invalid_config:{problem}", windows, temp)

    # 4. No element power, nothing to model.
    if not (math.isfinite(heater.power_kw) and heater.power_kw > 0.0):
        return fallback("no_power", windows, temp)

    measured_ambient = _finite(inputs.get("ambient_c"))
    mean_outdoor = _finite(mean_outdoor_temp_c)
    if measured_ambient is not None and AMBIENT_MIN_C <= measured_ambient <= AMBIENT_MAX_C:
        ambient = measured_ambient
    elif mean_outdoor is not None and AMBIENT_MIN_C <= mean_outdoor <= AMBIENT_MAX_C:
        ambient = mean_outdoor
    else:
        ambient = cfg.ambient_default_c

    cap = cfg.heat_capacity_kwh_per_c
    k = cfg.cooling_k_per_h
    source: Literal["config", "learned"] = "config"
    if cfg.learn:
        learned = load_learned_params(learned_params_path)
        if learned is not None:
            cap = learned["heat_capacity_kwh_per_c"]
            k = learned["cooling_k_per_h"]
            source = "learned"

    margin = timedelta(minutes=cfg.ready_margin_minutes)
    spec = BathingSpec(
        initial_temp_c=temp,
        ambient_c=ambient,
        heat_capacity_kwh_per_c=cap,
        cooling_k_per_h=k,
        ready_temp_c=cfg.ready_temp_c,
        eco_floor_c=cfg.eco_floor_c,
        temp_max_c=cfg.temp_max_c,
        windows=[BathWindow(start=s, end=e, ready_by=s - margin) for s, e in windows],
        ready_shortfall_sek_per_c=cfg.ready_shortfall_sek_per_c,
        hold_shortfall_sek_per_c_h=cfg.ready_shortfall_sek_per_c,
        floor_shortfall_sek_per_c_h=cfg.floor_shortfall_sek_per_c,
        hold_tolerance_c=cfg.hold_tolerance_c,
        overshoot_penalty_sek_per_c=cfg.overshoot_penalty_sek_per_c,
    )
    mode: BathingMode = "shadow" if cfg.shadow else "solver"
    logger.info(
        "Spa bathing %s: mode=%s, T=%.1f C, ambient=%.1f C, %d window(s), C=%.3f k=%.4f (%s)",
        hid,
        mode,
        temp,
        ambient,
        len(windows),
        cap,
        k,
        source,
    )
    return BathingDecision(
        heater_id=hid,
        mode=mode,
        reason=None,
        spec=spec if mode == "solver" else None,
        windows=windows,
        measured_temp_c=temp,
        ambient_c=ambient,
        params_source=source,
        spec_for_simulation=spec,
        power_kw=heater.power_kw,
        config=cfg,
    )


def simulate_plan(
    decision: BathingDecision,
    slots: Sequence[Any],
    heater_kw_per_slot: Sequence[float],
    first_slot_remaining_h: float | None,
) -> list[float]:
    """End-of-slot spa temperatures of a plan (same recursion as kepler).

    ``on_fraction[t]`` is 1 when the planned heater kW exceeds 0.01, else 0. Slot
    hours come from ``planned_slot_hours`` so a partial slot 0 is treated exactly as
    the solver treats it. Returns ``[]`` when the decision carries no spec.
    """
    spec = decision.spec_for_simulation
    if spec is None or not slots:
        return []
    if len(heater_kw_per_slot) != len(slots):
        raise ValueError(
            f"heater_kw_per_slot has {len(heater_kw_per_slot)} entries for {len(slots)} slots"
        )
    hours = planned_slot_hours(list(slots), first_slot_remaining_h)
    on = [1.0 if (_finite(kw) or 0.0) > ON_THRESHOLD_KW else 0.0 for kw in heater_kw_per_slot]
    return simulate_temps_c(
        spec.initial_temp_c,
        spec.ambient_c,
        spec.cooling_k_per_h,
        spec.heat_capacity_kwh_per_c,
        decision.power_kw,
        hours,
        on,
    )


def ready_state_index(slots: Sequence[_Slot], ready_by: datetime, now: datetime) -> int | None:
    """Recursion state index whose temperature is the one at ``ready_by``.

    Mirrors kepler's mapping: a ``ready_by`` already past or inside slot 0 maps to
    state 1 (soft, inevitably short); otherwise the last slot boundary at or before
    ``ready_by``; ``None`` when ``ready_by`` lies beyond the horizon.
    """
    if not slots:
        return None
    if ready_by > slots[-1].end_time:
        return None
    if ready_by <= now or ready_by < slots[0].end_time:
        return 1
    k = 1
    for idx, slot in enumerate(slots):
        if slot.end_time <= ready_by:
            k = idx + 1
        else:
            break
    return k


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _round(value: float | None, nd: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, nd)


def summarize(
    decision: BathingDecision,
    temps_c: Sequence[float] | None,
    report: BathingReport | None,
    slots: Sequence[_Slot],
    now: datetime,
) -> dict[str, Any]:
    """``schedule.json`` ``meta.spa_bathing`` (schema: build spec section C).

    ``temps_c`` are the end-of-slot predictions for ``slots`` (simulated in shadow
    mode, the solver's in solver mode). ``report`` is the solver's BathingReport
    (solver mode); its per-window figures win over a recomputation from ``temps_c``.
    """
    cfg = decision.config
    spec = decision.spec_for_simulation
    active = decision.mode in ("solver", "shadow") and spec is not None
    margin = timedelta(minutes=cfg.ready_margin_minutes if cfg else 0.0)
    ready_temp = spec.ready_temp_c if spec else (cfg.ready_temp_c if cfg else None)
    temps = list(temps_c or [])

    def predicted_at(window_idx: int, ready_by: datetime) -> tuple[float | None, float | None]:
        if not active or ready_temp is None:
            return None, None
        if report is not None and window_idx < len(report.window_ready_temp_c):
            pred = report.window_ready_temp_c[window_idx]
            short = (
                report.window_shortfall_c[window_idx]
                if window_idx < len(report.window_shortfall_c)
                else None
            )
            if pred is not None and short is None:
                short = max(0.0, ready_temp - pred)
            return pred, short
        idx = ready_state_index(slots, ready_by, now)
        if idx is None or idx - 1 >= len(temps) or idx < 1:
            return None, None
        pred = temps[idx - 1]
        return pred, max(0.0, ready_temp - pred)

    shortfalls: list[float] = []
    next_bath: dict[str, Any] | None = None
    for i, (start, end) in enumerate(decision.windows):
        ready_by = start - margin
        if start <= now:
            continue  # in progress: hold only, no ready target
        pred, short = predicted_at(i, ready_by)
        if short is not None:
            shortfalls.append(short)
        if next_bath is None:
            next_bath = {
                "start": _iso(start),
                "end": _iso(end),
                "ready_by": _iso(ready_by),
                "predicted_temp_c": _round(pred),
                "shortfall_c": _round(short),
            }

    on_track: bool | None = None
    if active:
        on_track = all(s < ON_TRACK_TOLERANCE_C for s in shortfalls)

    params: dict[str, Any] = {
        "heat_capacity_kwh_per_c": (
            spec.heat_capacity_kwh_per_c if spec else (cfg.heat_capacity_kwh_per_c if cfg else None)
        ),
        "cooling_k_per_h": spec.cooling_k_per_h if spec else (cfg.cooling_k_per_h if cfg else None),
        "source": decision.params_source,
    }
    return {
        "heater_id": decision.heater_id,
        "mode": decision.mode,
        "reason": decision.reason,
        "planned_at": _iso(now),
        "measured_temp_c": _round(decision.measured_temp_c),
        "ambient_c": _round(decision.ambient_c),
        "ready_temp_c": ready_temp,
        "eco_floor_c": spec.eco_floor_c if spec else (cfg.eco_floor_c if cfg else None),
        "params": params,
        "windows": [[_iso(s), _iso(e)] for s, e in decision.windows],
        "next_bath": next_bath,
        "on_track": on_track,
        # degC: a time-boxed plan misses a bath by more than the element alone
        # forces, unproven (see BathingReport); None otherwise.
        "unproven_shortfall_c": _round(
            report.incumbent_unproven_shortfall_c if report is not None else None
        ),
    }
