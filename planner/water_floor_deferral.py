"""Roll a small end-of-bucket floor residual into the next day-bucket.

The cliff this removes
----------------------
The daily floor is a kWh total per day-bucket (10:00 -> 10:00 at the live
``defer_up_to_hours: 10``). Whatever is still owed must be heated before the
boundary, and a shortfall is priced at the reliability penalty (15 SEK/kWh at
comfort level 3). Water is decided per wall-clock hour (``hourly_blocks``), so
the smallest thing the solver can buy is "the rest of this hour".

Put together: at 06:15, with only the morning peak left before 10:00, a residual
of 0.5 kWh books nothing and 0.6 kWh books 2.55 kWh at 06:15 — the dearest price
of the day, because a three-slot stub of an hour is the cheapest *total* on offer
(reproduced with the real solver on the 2026-10-01 price curve). The residual
itself is usually not a need: it is the estimator's headroom creeping up as the
learned draw depletes a tank that filled at night, capped by the physics cap in
``pipeline._apply_water_shortfall_gate``. It crosses the threshold again roughly
every two and a half hours, so the same morning can buy two such blocks.

The rule
--------
When ALL of the following hold, bucket 0 owes nothing more and the residual is
left to the next bucket, whose own floor (never lowered here) buys it in a cheap
hour::

    1. the bucket ends within ``max_window_hours``                       (short)
    2. every remaining slot before the boundary costs at least
       ``min_price_ratio`` x what an hour of heating costs right after it (dear)
    3. the tank was MEASURED saturated — relay confirmed on, element idle —
       no more than ``max_full_age_hours`` ago                         (evidence)
    4. the estimator state itself is fresh                              (alive)
    5. projected to the boundary with the learned draw and the standing loss,
       the tank still holds ``reserve_showers`` above ``comfort_c``     (comfort)

Why each guard is there
-----------------------
**(3) is a measurement, not the model.** ``stored_kwh`` re-pins to full whenever
any heating run ends, including one the plan cut short on a half-empty tank. The
saturation clock only runs while the relay is confirmed on and the element draws
nothing, which is the thermostat talking. It is the same evidence the STRUCK rule
already acts on.

**(5) is deliberately in showers.** The learned draw is a mean rate and cannot
see a shower (2.1 kWh in ten minutes); the reserve is the quantum that covers the
ones it missed. ``comfort_c`` is applied to a fully mixed tank, which understates
what a stratified tank delivers — the error is on the cold-shower-safe side.

**It only ever lowers bucket 0, and only until the boundary.** No other bucket is
touched and nothing is rewarded: the solver gets one constraint fewer, never a
new term, so heating cannot become profitable on its own account. A floor that
is only ever lowered cannot be one that never clears.

**It fails to the old behaviour.** Any missing or implausible input returns "do
not defer", which is exactly the pre-existing floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["DeferralConfig", "DeferralDecision", "bucket_boundary", "decide_floor_deferral"]

# Specific heat of water, Wh per litre-kelvin (mirrors planner/thermal.py).
_C_P_WH_PER_LITRE_K = 4186.0 / 3600.0


@dataclass(frozen=True)
class DeferralConfig:
    """Per-tank knobs. Disabled by default — opt in per heater."""

    enabled: bool = False
    # What must still be in the tank at the bucket boundary, in showers.
    reserve_showers: float = 2.0
    shower_litres: float = 60.0  # litres AT comfort_c
    comfort_c: float = 40.0
    t_cold_c: float = 10.0
    # Only the tail of a bucket qualifies.
    max_window_hours: float = 6.0
    # How old the last measured saturation may be.
    max_full_age_hours: float = 8.0
    # Estimator state older than this is a dead publisher, not a tank.
    max_state_age_min: float = 20.0
    # Cheapest slot before the boundary vs. an hour of heating after it.
    min_price_ratio: float = 1.25

    @classmethod
    def from_heater(cls, raw: dict[str, Any]) -> DeferralConfig:
        block: dict[str, Any] = raw.get("floor_deferral") or {}
        base = cls()

        def _f(key: str, default: float) -> float:
            try:
                v = block.get(key)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(block.get("enabled", False)),
            reserve_showers=max(0.0, _f("reserve_showers", base.reserve_showers)),
            shower_litres=max(0.0, _f("shower_litres", base.shower_litres)),
            comfort_c=_f("comfort_c", base.comfort_c),
            t_cold_c=float(raw.get("t_cold_c", base.t_cold_c) or base.t_cold_c),
            max_window_hours=max(0.0, _f("max_window_hours", base.max_window_hours)),
            max_full_age_hours=max(0.0, _f("max_full_age_hours", base.max_full_age_hours)),
            max_state_age_min=max(0.0, _f("max_state_age_min", base.max_state_age_min)),
            min_price_ratio=max(1.0, _f("min_price_ratio", base.min_price_ratio)),
        )


@dataclass(frozen=True)
class DeferralDecision:
    defer: bool
    reason: str


def bucket_boundary(now_local: datetime, defer_hours: float) -> datetime:
    """End of the day-bucket ``now_local`` belongs to (kepler's bucketing, verbatim).

    kepler puts a slot in the PREVIOUS calendar day's bucket when
    ``defer_hours > 0 and slot.hour < defer_hours``; the boundary is therefore the
    first later instant whose hour is no longer below ``defer_hours``.
    """
    from datetime import timedelta

    midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if defer_hours > 0:
        whole = int(defer_hours) if float(defer_hours).is_integer() else int(defer_hours) + 1
        whole = max(0, min(24, whole))
        if now_local.hour < defer_hours:
            return midnight + timedelta(hours=whole)
        return midnight + timedelta(days=1, hours=whole)
    return midnight + timedelta(days=1)


def _slot_hours(slot: Any) -> float:
    try:
        h = (slot.end_time - slot.start_time).total_seconds() / 3600.0
    except Exception:
        return 0.0
    return h if h > 0 else 0.0


def _unit_cost(slot: Any, heater_kw: float) -> float | None:
    """SEK per kWh of heating in this slot: forecast PV surplus at the export price it
    forgoes, the rest at the import price. Same arithmetic as kepler's anchor gate."""
    h = _slot_hours(slot)
    kwh = heater_kw * h
    if kwh <= 0:
        return None
    surplus = max(0.0, float(slot.pv_kwh) - float(slot.load_kwh))
    pv_used = min(surplus, kwh)
    return (
        pv_used * float(slot.export_price_sek_kwh)
        + (kwh - pv_used) * float(slot.import_price_sek_kwh)
    ) / kwh


def decide_floor_deferral(
    *,
    cfg: DeferralConfig,
    slots: list[Any],
    now_local: datetime,
    defer_hours: float,
    heater_kw: float,
    residual_kwh: float,
    stored_kwh: float | None,
    capacity_kwh: float | None,
    volume_litres: float | None,
    learned_draw_kw: float | None,
    min_since_saturated: float | None,
    state_age_min: float | None,
    standby_loss_kwh_fn: Any = None,
) -> DeferralDecision:
    """Whether bucket 0's remaining floor may wait for the next bucket.

    ``standby_loss_kwh_fn(stored_kwh, hours) -> kWh`` supplies the standing loss over
    the wait; None counts no loss only in tests — the pipeline always passes it.
    """
    if not cfg.enabled:
        return DeferralDecision(False, "disabled")
    if residual_kwh <= 1e-9:
        return DeferralDecision(False, "nothing owed")
    if not slots or heater_kw <= 0:
        return DeferralDecision(False, "no slots")

    # -- (4) the estimator is alive, (3) the tank was measured full recently --------
    if stored_kwh is None or capacity_kwh is None or not volume_litres:
        return DeferralDecision(False, "tank state unknown")
    if capacity_kwh <= 0 or volume_litres <= 0:
        return DeferralDecision(False, "unusable tank geometry")
    if state_age_min is None or state_age_min > cfg.max_state_age_min:
        return DeferralDecision(False, "estimator state stale or of unknown age")
    if min_since_saturated is None or min_since_saturated < 0:
        return DeferralDecision(False, "tank never measured saturated")
    if min_since_saturated > cfg.max_full_age_hours * 60.0:
        return DeferralDecision(
            False, f"last measured full {min_since_saturated / 60.0:.1f} h ago"
        )
    # A learned rate of zero is an estimator that never learned, not a household that
    # uses no hot water (same reading as water_shortfall.dynamic_floor_kwh).
    if learned_draw_kw is None or learned_draw_kw <= 0:
        return DeferralDecision(False, "draw rate unknown")

    # -- (1) the bucket ends soon ---------------------------------------------------
    try:
        boundary = bucket_boundary(now_local, defer_hours)
        wait_h = (boundary - now_local).total_seconds() / 3600.0
        before = [s for s in slots if s.start_time < boundary]
        after = [s for s in slots if s.start_time >= boundary]
    except TypeError:
        # Naive/aware mismatch: refuse rather than guess a timezone.
        return DeferralDecision(False, "timezone mismatch")
    if wait_h <= 0 or wait_h > cfg.max_window_hours:
        return DeferralDecision(False, f"bucket ends in {wait_h:.1f} h")
    if not before:
        return DeferralDecision(False, "no slots before the boundary")

    # -- (2) nothing before the boundary is cheap, and something after it is --------
    costs_before = [c for c in (_unit_cost(s, heater_kw) for s in before) if c is not None]
    after_pairs = sorted(
        (c, heater_kw * _slot_hours(s))
        for s in after
        if (c := _unit_cost(s, heater_kw)) is not None
    )
    if not costs_before:
        return DeferralDecision(False, "no priced slots before the boundary")
    # One full hour of this heater, at the dearest of the cheapest slots that make it
    # up. No such hour in view => the next bucket is not visible => nothing to roll to.
    need = heater_kw * 1.0
    ref_after: float | None = None
    for c, kwh in after_pairs:
        need -= kwh
        ref_after = c
        if need <= 1e-9:
            break
    if ref_after is None or need > 1e-9:
        return DeferralDecision(False, "next bucket not in view")
    cheapest_before = min(costs_before)
    if cheapest_before < cfg.min_price_ratio * ref_after:
        return DeferralDecision(
            False,
            f"a slot before the boundary is cheap enough "
            f"({cheapest_before:.2f} vs {ref_after:.2f} after)",
        )

    # -- (5) comfort: what is left at the boundary ----------------------------------
    loss = 0.0
    if standby_loss_kwh_fn is not None:
        try:
            loss = max(0.0, float(standby_loss_kwh_fn(float(stored_kwh), wait_h)))
        except Exception:
            return DeferralDecision(False, "standing loss unavailable")
    projected = min(float(stored_kwh), float(capacity_kwh)) - learned_draw_kw * wait_h - loss
    comfort_floor = volume_litres * _C_P_WH_PER_LITRE_K * (cfg.comfort_c - cfg.t_cold_c) / 1000.0
    shower = cfg.shower_litres * _C_P_WH_PER_LITRE_K * (cfg.comfort_c - cfg.t_cold_c) / 1000.0
    if comfort_floor <= 0 or shower <= 0:
        return DeferralDecision(False, "comfort temperature not above cold inlet")
    usable = projected - comfort_floor
    reserve = cfg.reserve_showers * shower
    if usable < reserve:
        return DeferralDecision(
            False,
            f"projected {usable:.2f} kWh above {cfg.comfort_c:.0f} C at the boundary "
            f"< reserve {reserve:.2f} ({cfg.reserve_showers:.1f} showers)",
        )

    return DeferralDecision(
        True,
        f"bucket ends in {wait_h:.1f} h, cheapest before {cheapest_before:.2f} vs "
        f"{ref_after:.2f} SEK/kWh after; measured full {min_since_saturated:.0f} min ago, "
        f"holds {float(stored_kwh):.2f} of {float(capacity_kwh):.2f} kWh, projected "
        f"{usable:.2f} kWh above {cfg.comfort_c:.0f} C at the boundary >= reserve "
        f"{reserve:.2f} ({cfg.reserve_showers:.1f} showers)",
    )
