"""Shortfall-risk gate for the daily hot-water floor.

.. warning::
   **NOT PRODUCTION READY — do not set ``enabled: true``.** Adversarial review
   2026-08-25 confirmed five critical defects, and they are not all typos: the
   ``need`` model is a mean draw RATE integrated over the wait, which cannot
   represent a shower. At the live ``learned_draw_kw`` of 0.15 kW the model
   predicts 0.025 kWh across the ten minutes in which a real shower draws
   2.09 kWh — an 84x under-forecast — and the owner's reserve, the only
   per-shower quantum in the model, is switched OFF for the entire morning
   (00:00-18:00 on a weekday). The day bucket also rolls at 06:00, so an
   evening booking is revoked at midnight before the cheap night hours arrive.
   Fixing this needs a demand model with a shower quantum, not a patch.
   See the review notes in the session transcript before resuming.

The blunt rule this replaces
---------------------------
``min_kwh_per_day`` is a hard household comfort promise expressed in kilowatt
hours, and the solver prices a shortfall at the reliability penalty — 15 SEK/kWh
at comfort level 3, which outvotes any Swedish spot hour by a factor of four. It
therefore books the tank at ANY price whenever the day-bucket total is unmet,
with no idea whether the tank is actually low. Live 2026-08-25 19:28: the house
tank was switched on at 3.56 SEK/kWh while sitting at 69.7 C with 388 litres of
40-degree water available — several showers' worth — purely because the bucket
had credited 0.000 of its 6.00 kWh.

The rule that replaces it
-------------------------
Ask what the household will actually draw before the next cheap window, compare
it with what the tank already holds, and book only the difference::

    need    = expected draw over the wait + standing loss + owner's reserve
    have    = energy stored ABOVE the comfort temperature
    floor   = clamp(need - have, 0, configured min_kwh_per_day)

The reserve is the owner's risk appetite, stated in showers rather than
kilowatt hours because that is the unit the decision is felt in (2026-08-25:
"1.5 showers, from 18:00 on a weekday and 14:00 at the weekend").

Three things this deliberately does NOT do
------------------------------------------
**It never raises the floor.** The configured value stays the ceiling, so the
gate can only ever relax a promise the household already accepted, never invent
a new one. That keeps the blast radius one-directional.

**It never reasons in mixed litres.** ``mixed_liters_at`` divides by
``comfort_c - t_cold``, so it falls off a cliff — 40.1 C reports 196 litres and
40.0 C reports zero — and a threshold near that discontinuity would chatter
between plans. Everything here is energy above comfort, which is smooth.

**It fails to the old behaviour.** Unknown tank state, unknown draw rate or a
missing horizon all return the configured floor unchanged. A gate that opens on
a dead sensor would trade cold showers for pennies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

# Specific heat of water, Wh per litre-kelvin (mirrors planner/thermal.py so this
# module stays importable without dragging the tank model in).
_C_P_WH_PER_LITRE_K = 4186.0 / 3600.0  # ~= 1.1628


@dataclass(frozen=True)
class ShortfallConfig:
    """Per-tank knobs for the gate. Disabled by default — opt in per heater."""

    enabled: bool = False
    # What "usable hot water" means. Energy below this is not a shower.
    comfort_c: float = 40.0
    # Cold inlet; with comfort_c it fixes the kWh cost of one litre of shower.
    t_cold_c: float = 10.0
    # One shower, in litres of water AT comfort_c (mixed, i.e. what leaves the head).
    shower_litres: float = 60.0
    # The owner's reserve, in showers, held back on top of the forecast draw.
    margin_showers: float = 1.5
    # The reserve applies from this local hour. The evening is when the choice is
    # real: the next cheap window is tomorrow's sun, and someone may still shower.
    margin_from_hour_weekday: float = 18.0
    margin_from_hour_weekend: float = 14.0
    # Cap on how far ahead to wait for a cheap window. Beyond this the gate stops
    # betting on a forecast it cannot see.
    max_horizon_hours: float = 16.0


def shower_kwh(cfg: ShortfallConfig) -> float:
    """Energy in one shower: litres at comfort_c, lifted from the cold inlet."""
    delta = cfg.comfort_c - cfg.t_cold_c
    if delta <= 0.0 or cfg.shower_litres <= 0.0:
        return 0.0
    return cfg.shower_litres * _C_P_WH_PER_LITRE_K * delta / 1000.0


def margin_showers_now(now_local: datetime, cfg: ShortfallConfig) -> float:
    """The reserve that applies at this local wall-clock time, in showers.

    ``now_local`` must already be in the site timezone — this function does no
    conversion, because silently assuming UTC is the recurring bug class in this
    codebase (a container running local time with UTC comparators).
    """
    if cfg.margin_showers <= 0.0:
        return 0.0
    is_weekend = now_local.weekday() >= 5
    from_hour = cfg.margin_from_hour_weekend if is_weekend else cfg.margin_from_hour_weekday
    hour_now = now_local.hour + now_local.minute / 60.0
    return cfg.margin_showers if hour_now >= from_hour else 0.0


def usable_kwh_above_comfort(
    stored_kwh: float | None,
    volume_litres: float | None,
    cfg: ShortfallConfig,
) -> float | None:
    """Stored energy that can still deliver water at comfort_c, or None if unknown.

    ``stored_kwh`` is the estimator's energy above the cold inlet. A tank sitting
    exactly at comfort holds plenty of energy but zero usable showers, so the
    comfort floor is subtracted rather than compared against.
    """
    if stored_kwh is None or volume_litres is None or volume_litres <= 0.0:
        return None
    floor_kwh = volume_litres * _C_P_WH_PER_LITRE_K * (cfg.comfort_c - cfg.t_cold_c) / 1000.0
    return max(0.0, float(stored_kwh) - floor_kwh)


def hours_to_cheap_window(
    slots: list[Any],
    now_local: datetime,
    *,
    heater_kw: float,
    max_horizon_hours: float,
    min_discount_frac: float = 0.25,
) -> float | None:
    """Hours until hot water can be made cheaply again, or None if never in range.

    "Cheaply" is either of two things, whichever comes first:

    * **Sun.** A slot whose PV forecast exceeds this heater's own draw — then the
      kWh is free at the margin. This is the case the owner asked for: wait for
      tomorrow's daylight instead of buying at the evening peak.
    * **A materially cheaper grid hour** — at least ``min_discount_frac`` below
      what this kWh costs right now. The comparison is against the CURRENT price
      rather than a percentile of the horizon, because the question is not "is
      this hour among the cheaper ones" but "does waiting actually save money".
      On a flat price curve nothing qualifies, waiting buys nothing, and the
      caller correctly falls back to heating now.

    Returns None when neither appears inside ``max_horizon_hours`` — the caller
    treats that as "no cheap window", which keeps the configured floor.
    """
    if not slots or heater_kw <= 0.0:
        return None
    horizon_end_h = max(0.0, float(max_horizon_hours))
    future: list[Any] = []
    for s in slots:
        st = getattr(s, "start_time", None)
        if st is None:
            continue
        try:
            dh = (st - now_local).total_seconds() / 3600.0
        except TypeError:
            # Naive/aware mismatch: refuse rather than guess a timezone.
            return None
        if 0.0 <= dh <= horizon_end_h:
            future.append((dh, s))
    if not future:
        return None

    ordered = sorted(future, key=lambda x: x[0])
    price_now = float(getattr(ordered[0][1], "import_price_sek_kwh", 0.0) or 0.0)
    cheap_price = price_now * (1.0 - max(0.0, min(1.0, min_discount_frac)))

    for dh, s in ordered:
        slot_h = _slot_hours(s)
        pv_kw = (float(getattr(s, "pv_kwh", 0.0) or 0.0) / slot_h) if slot_h > 0 else 0.0
        if pv_kw >= heater_kw:
            return dh
        if float(getattr(s, "import_price_sek_kwh", 0.0) or 0.0) <= cheap_price:
            return dh
    return None


def _slot_hours(slot: Any) -> float:
    st = getattr(slot, "start_time", None)
    et = getattr(slot, "end_time", None)
    if st is None or et is None:
        return 0.25
    try:
        h = (et - st).total_seconds() / 3600.0
    except TypeError:
        return 0.25
    return h if h > 0 else 0.25


def dynamic_floor_kwh(
    *,
    configured_min_kwh: float,
    heated_today_kwh: float,
    stored_kwh: float | None,
    volume_litres: float | None,
    learned_draw_kw: float | None,
    standby_loss_kwh: float,
    hours_to_cheap: float | None,
    now_local: datetime,
    cfg: ShortfallConfig,
) -> tuple[float, str]:
    """The floor to hand the solver, and a one-line reason for the log.

    The returned value is what ``min_kwh_per_day`` must be set to. It carries
    ``heated_today_kwh`` back in on purpose: the solver's day-bucket constraint
    is ``min_kwh_per_day - heated_today_kwh``, while the shortfall is computed
    from the tank as it is RIGHT NOW and therefore already reflects everything
    heated so far. Without the add-back today's heating would be deducted twice.
    """
    if not cfg.enabled:
        return configured_min_kwh, "gate disabled"
    if configured_min_kwh <= 0.0:
        return configured_min_kwh, "no floor configured"

    # Fail to the old behaviour on any missing input — see module docstring.
    usable = usable_kwh_above_comfort(stored_kwh, volume_litres, cfg)
    if usable is None:
        return configured_min_kwh, "tank state unknown"
    # A learned rate of exactly zero is not a measurement, it is an estimator that
    # never learned (or decayed to nothing): treating it as "this household uses no
    # hot water" books nothing for a stone-cold tank.
    if learned_draw_kw is None or learned_draw_kw <= 0.0:
        return configured_min_kwh, "draw rate unknown"
    if hours_to_cheap is None or hours_to_cheap <= 0.0:
        return configured_min_kwh, "no cheap window found"

    wait_h = min(float(hours_to_cheap), cfg.max_horizon_hours)
    draw_kwh = float(learned_draw_kw) * wait_h
    margin = margin_showers_now(now_local, cfg)
    reserve_kwh = margin * shower_kwh(cfg)

    need_kwh = draw_kwh + max(0.0, standby_loss_kwh) + reserve_kwh
    shortfall = max(0.0, need_kwh - usable)
    booked = min(shortfall, configured_min_kwh)

    reason = (
        f"wait {wait_h:.1f}h: need {need_kwh:.2f} kWh "
        f"(draw {draw_kwh:.2f} + loss {max(0.0, standby_loss_kwh):.2f} "
        f"+ reserve {reserve_kwh:.2f} = {margin:.1f} showers), "
        f"have {usable:.2f} kWh usable -> book {booked:.2f} of {configured_min_kwh:.2f}"
    )
    # The add-back is what makes the solver's own "- heated_today" cancel out.
    return booked + max(0.0, heated_today_kwh), reason
