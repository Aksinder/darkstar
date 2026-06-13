"""
FMB state-of-charge estimator — pure dead-reckoning logic.

The FMB EV reports no real SoC, so we estimate it: integrate delivered energy UP
while charging (from the Easee lifetime-energy counter), and drift it DOWN
continuously at a *learned* consumption rate while not charging. Each time the
battery is charged back to full, the energy it took to refill since the previous
full IS the true consumption over that span — so the drift rate self-calibrates
from "how much we actually charged" (an EMA over full-to-full cycles), exactly as
specified. The user's 20 %/day (~5.6 kWh/day on a 28 kWh pack) is only the prior.

Pure functions only — the runtime wrapper (fmb_soc_runtime.py) does HA I/O and
persistence. Returning a fresh state (rather than mutating) keeps this trivially
testable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

_SECONDS_PER_DAY = 86400.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class FmbSocConfig:
    """Tuning for the FMB SoC estimator. capacity + prior come from the user."""

    enabled: bool = False
    capacity_kwh: float = 28.0
    charge_efficiency: float = 0.9  # grid->battery loss; delivered kWh times this = stored
    # Drift / learning
    prior_consumption_kwh_per_day: float = 5.6  # 20 %/day of 28 kWh (user-chosen prior)
    learn_alpha: float = 0.3  # EMA weight on each new full-to-full observation
    min_consumption_kwh_per_day: float = 0.5
    max_consumption_kwh_per_day: float = 20.0
    min_anchor_energy_kwh: float = 2.0  # need ≥ this refilled to trust a learned rate
    min_anchor_days: float = 0.04  # and ≥ ~1 h since last full (avoid divide-by-noise)
    # "Full" detection (the car stopped accepting while we were offering current)
    charging_power_w: float = 200.0  # above this = actively charging
    full_idle_power_w: float = 200.0  # below this *while offered current* = tapered/full
    full_offered_min_a: float = 5.0  # only treat taper as "full" if we offered ≥ this
    full_idle_min_s: float = 300.0  # taper must persist this long to latch full
    full_status_values: tuple[str, ...] = ("completed",)
    full_release_soc: float = 97.0  # drop below this (or unplug) clears the full latch
    # Seeding / bounds
    floor_soc: float = 0.0
    initial_soc: float = 50.0  # conservative seed when no persisted state exists
    max_step_kwh: float = 25.0  # reject implausible counter jumps (≈ full pack in one tick)


@dataclass
class FmbSocState:
    """Persisted estimator state (one JSON blob)."""

    soc_pct: float = 50.0
    last_energy_kwh: float | None = None  # last lifetime-energy reading; None = uninit
    last_ts: float | None = None
    energy_since_anchor_kwh: float = 0.0  # battery-side energy delivered since last full
    last_full_ts: float | None = None
    learned_rate_kwh_per_day: float = 5.6
    full_latched: bool = False
    idle_full_since_ts: float | None = None


@dataclass
class FmbSocInputs:
    """Live readings for one tick."""

    now_ts: float
    lifetime_energy_kwh: float | None  # Easee lifetime-energy counter (monotonic)
    power_w: float
    plugged: bool
    charger_enabled: bool  # the Easee enable switch is on
    dynamic_limit_a: float | None = None  # offered current (None = unknown → trust taper)
    status: str | None = None  # Easee status string


def initial_state(cfg: FmbSocConfig) -> FmbSocState:
    """Fresh state seeded from config (used when nothing is persisted yet)."""
    return FmbSocState(
        soc_pct=_clamp(cfg.initial_soc, cfg.floor_soc, 100.0),
        learned_rate_kwh_per_day=_clamp(
            cfg.prior_consumption_kwh_per_day,
            cfg.min_consumption_kwh_per_day,
            cfg.max_consumption_kwh_per_day,
        ),
    )


def _energy_gain_kwh(state: FmbSocState, inp: FmbSocInputs, cfg: FmbSocConfig) -> float:
    """Battery-side energy gained since the last reading (0 on first tick / counter reset)."""
    if state.last_energy_kwh is None or inp.lifetime_energy_kwh is None:
        return 0.0
    raw = inp.lifetime_energy_kwh - state.last_energy_kwh
    if raw < 0.0 or raw > cfg.max_step_kwh:
        # Counter reset (integration restart) or implausible jump — skip this delta.
        return 0.0
    return raw * cfg.charge_efficiency


def _is_full(state: FmbSocState, inp: FmbSocInputs, cfg: FmbSocConfig) -> tuple[bool, float | None]:
    """True-full detection: the car stopped accepting while we offered current, or status=completed.

    Returns (is_full, new_idle_full_since_ts). Deliberately does NOT treat an
    integrated SoC of 100 as full — a too-high seed could reach 100 before the
    real battery does, which would anchor (and learn) on a false full. Only a
    real taper / completed status is trusted to anchor.
    """
    if inp.status and inp.status.lower() in {s.lower() for s in cfg.full_status_values}:
        return True, state.idle_full_since_ts

    offering = (
        inp.plugged
        and inp.charger_enabled
        and (inp.dynamic_limit_a is None or inp.dynamic_limit_a >= cfg.full_offered_min_a)
    )
    tapered = offering and inp.power_w < cfg.full_idle_power_w
    if not tapered:
        return False, None  # reset the taper timer

    since = state.idle_full_since_ts if state.idle_full_since_ts is not None else inp.now_ts
    is_full = (inp.now_ts - since) >= cfg.full_idle_min_s
    return is_full, since


def update_fmb_soc(
    state: FmbSocState, inp: FmbSocInputs, cfg: FmbSocConfig
) -> tuple[FmbSocState, dict[str, Any]]:
    """Advance the estimate one tick. Returns (new_state, debug)."""
    cap = max(cfg.capacity_kwh, 1e-6)
    learned = _clamp(
        state.learned_rate_kwh_per_day,
        cfg.min_consumption_kwh_per_day,
        cfg.max_consumption_kwh_per_day,
    )

    # First-ever tick: just establish the counter/time baseline, no drift.
    if state.last_ts is None or state.last_energy_kwh is None:
        seeded = replace(
            state,
            soc_pct=_clamp(state.soc_pct, cfg.floor_soc, 100.0),
            last_energy_kwh=inp.lifetime_energy_kwh,
            last_ts=inp.now_ts,
            learned_rate_kwh_per_day=learned,
        )
        return seeded, {"reason": "init", "soc": seeded.soc_pct, "rate": learned}

    gained_kwh = _energy_gain_kwh(state, inp, cfg)
    charging = inp.power_w > cfg.charging_power_w or gained_kwh > 0.0

    soc = state.soc_pct
    energy_since_anchor = state.energy_since_anchor_kwh

    # Up: integrate delivered energy.
    if gained_kwh > 0.0:
        soc += gained_kwh / cap * 100.0
        energy_since_anchor += gained_kwh

    # Down: continuous learned drift, only while NOT charging.
    drift_pct = 0.0
    if not charging:
        dt_days = max(0.0, (inp.now_ts - state.last_ts) / _SECONDS_PER_DAY)
        drift_pct = learned * dt_days / cap * 100.0
        soc -= drift_pct

    soc = _clamp(soc, cfg.floor_soc, 100.0)

    # Full detection + learning.
    is_full, idle_since = _is_full(state, inp, cfg)
    full_latched = state.full_latched
    last_full_ts = state.last_full_ts
    learned_out = learned
    learned_event: dict[str, Any] | None = None

    if is_full and not full_latched:
        # Rising edge into full: the refill since the last full reveals true consumption.
        if last_full_ts is not None:
            days = (inp.now_ts - last_full_ts) / _SECONDS_PER_DAY
            if days >= cfg.min_anchor_days and energy_since_anchor >= cfg.min_anchor_energy_kwh:
                implied = _clamp(
                    energy_since_anchor / days,
                    cfg.min_consumption_kwh_per_day,
                    cfg.max_consumption_kwh_per_day,
                )
                learned_out = _clamp(
                    cfg.learn_alpha * implied + (1.0 - cfg.learn_alpha) * learned,
                    cfg.min_consumption_kwh_per_day,
                    cfg.max_consumption_kwh_per_day,
                )
                learned_event = {
                    "implied_kwh_per_day": round(implied, 2),
                    "energy_since_anchor_kwh": round(energy_since_anchor, 2),
                    "days": round(days, 3),
                    "rate_before": round(learned, 2),
                    "rate_after": round(learned_out, 2),
                }
        soc = 100.0
        energy_since_anchor = 0.0
        last_full_ts = inp.now_ts
        full_latched = True
    elif full_latched and (not inp.plugged or soc < cfg.full_release_soc):
        # Left full (driven off, or drifted down enough) — re-arm for the next cycle.
        full_latched = False

    new_state = FmbSocState(
        soc_pct=round(soc, 3),
        last_energy_kwh=inp.lifetime_energy_kwh,
        last_ts=inp.now_ts,
        energy_since_anchor_kwh=round(energy_since_anchor, 4),
        last_full_ts=last_full_ts,
        learned_rate_kwh_per_day=round(learned_out, 4),
        full_latched=full_latched,
        idle_full_since_ts=idle_since,
    )
    debug: dict[str, Any] = {
        "soc": new_state.soc_pct,
        "charging": charging,
        "gained_kwh": round(gained_kwh, 4),
        "drift_pct": round(drift_pct, 4),
        "rate_kwh_per_day": new_state.learned_rate_kwh_per_day,
        "full": full_latched,
        "energy_since_anchor_kwh": new_state.energy_since_anchor_kwh,
    }
    if learned_event is not None:
        debug["learned"] = learned_event
    return new_state, debug
