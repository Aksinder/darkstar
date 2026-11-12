"""Price rules for ``mode: window`` cyclic loads (room heaters): cheap early start
and peak off.

Owner, 2026-11-12: "if the price is low earlier, from the night before, we of course
want to start the heater then, and we switch them off at all the high peaks."

ONE decision function, :func:`step`, used by both sides so the plan and the plug
agree: the executor calls it every tick with the live price, the planner walks it
forward over its slots with the forecast prices (:func:`simulate`). It is pure — no
I/O, no clock — and carries its little memory in a RuleState the caller keeps.

* Early start. From ``early_start_hours`` before the plug would otherwise come on
  (window start minus preheat), the heater starts as soon as the import price is at
  or below the ``early_start_price_percentile`` of the rolling price window. It then
  stays on — LATCHED — until the window begins; a price that rises above the early
  threshold again does not switch it off. Only the peak rule can.
* Peak off. While the price is at or above the ``peak_off_price_percentile`` the
  heater is off, also inside a window. Both room temperature sensors are dead, so
  TIME is the protection against a cold bedroom: one off-stretch lasts at most
  ``peak_off_max_minutes``, and the heater must then have been on for
  ``peak_off_min_on_minutes`` before the next one may begin. With a working
  thermostat the caller additionally passes ``keep_warm`` when the room is well
  below target, which forbids (and ends) a peak-off.

Fail direction: an unknown price (None) never starts early and never starts or
continues a peak-off. Inside a window the heater then simply stays on.

Thresholds are percentiles of a rolling price window (executor/water_hold.py
``price_percentile``, the same machinery as the pumps' opportunistic gates), spanning
``price_window_hours``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from .water_hold import price_percentile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PHASE_IDLE = "idle"  # nothing asks for heat
PHASE_WINDOW = "window"  # in a window (or preheat), heating allowed
PHASE_EARLY = "early_start"  # latched on before the window, on a cheap price
PHASE_PEAK_OFF = "peak_off"  # heat is asked for, but the price is at a peak


def _number(raw: Any, default: float) -> float:
    try:
        return default if raw is None or raw == "" else float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PriceRules:
    """The price-rule keys of one window-mode load. Everything off by default."""

    # How long before the plug would otherwise come on it MAY start early. 0 = off.
    early_start_hours: float = 0.0
    early_start_price_percentile: float = 30.0
    # None = the peak rule is off.
    peak_off_price_percentile: float | None = None
    peak_off_max_minutes: float = 180.0
    peak_off_min_on_minutes: float = 120.0
    # Thermostat guard: no peak-off below target - hysteresis - this margin.
    peak_off_temp_margin_c: float = 1.0
    # Span of the rolling price window the percentiles are taken over.
    price_window_hours: float = 24.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PriceRules:
        """Parse one ``cyclic_loads[]`` entry — the ONLY parser, so the planner (raw
        config) and the executor (CyclicLoadConfig) cannot read the keys differently."""
        peak = raw.get("peak_off_price_percentile")
        return cls(
            early_start_hours=max(0.0, _number(raw.get("early_start_hours"), 0.0)),
            early_start_price_percentile=_number(
                raw.get("early_start_price_percentile"), cls.early_start_price_percentile
            ),
            peak_off_price_percentile=(
                None if peak is None or peak == "" else _number(peak, 100.0)
            ),
            peak_off_max_minutes=max(
                0.0, _number(raw.get("peak_off_max_minutes"), cls.peak_off_max_minutes)
            ),
            peak_off_min_on_minutes=max(
                0.0, _number(raw.get("peak_off_min_on_minutes"), cls.peak_off_min_on_minutes)
            ),
            peak_off_temp_margin_c=max(
                0.0, _number(raw.get("peak_off_temp_margin_c"), cls.peak_off_temp_margin_c)
            ),
            price_window_hours=(
                _number(raw.get("price_window_hours"), cls.price_window_hours)
                or cls.price_window_hours
            ),
        )

    @property
    def early_enabled(self) -> bool:
        return self.early_start_hours > 0

    @property
    def peak_enabled(self) -> bool:
        return self.peak_off_price_percentile is not None

    @property
    def enabled(self) -> bool:
        return self.early_enabled or self.peak_enabled


@dataclass(frozen=True)
class RuleState:
    """What the rules remember between two steps. Frozen: a step returns a new one."""

    # Latched on ahead of a window.
    early_latched: bool = False
    # Start of the running peak-off stretch (None = not in one).
    peak_off_since: datetime | None = None
    # Since when heat has been allowed without interruption (None = not allowed now,
    # or a heat period that has not begun). The min-on clock.
    heat_since: datetime | None = None


def rule_state_from_attributes(attrs: Mapping[str, Any] | None) -> RuleState | None:
    """The executor's published rule memory (attributes of the load's status sensor)
    as a RuleState; None when there is nothing usable. Stale values are harmless: the
    next :func:`step` drops a latch outside its span and ends an over-long peak-off.
    """
    if not attrs or "early_start_latched" not in attrs:
        return None

    def _when(raw: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(raw)) if raw else None
        except ValueError:
            return None
        return parsed if parsed is not None and parsed.tzinfo is not None else None

    return RuleState(
        early_latched=attrs.get("early_start_latched") is True,
        peak_off_since=_when(attrs.get("peak_off_since")),
        heat_since=_when(attrs.get("heat_since")),
    )


@dataclass(frozen=True)
class Verdict:
    heat: bool  # may the heater be on (before the thermostat has its say)?
    phase: str
    reason: str


def thresholds(
    rules: PriceRules, price_window: Sequence[float | None]
) -> tuple[float | None, float | None]:
    """(early-start threshold, peak-off threshold) in SEK/kWh; None when the rule is
    off or the window has no such price.

    A percentile alone misbehaves on a price PLATEAU: when most of the window costs
    the same, that price is the 30th and the 85th percentile at once, and the heater
    would both start early and be "at a peak" for hours — 3 h off, 2 h on, all day,
    for no saving. So cheap must also be strictly below the window's median and a
    peak strictly above it; a flat window has neither.
    """
    window = [p for p in price_window if p is not None]
    median = price_percentile(window, 50.0)
    if median is None:
        return None, None
    early: float | None = None
    peak: float | None = None
    if rules.early_enabled:
        early = price_percentile(window, rules.early_start_price_percentile)
        if early is not None and early >= median:
            early = max((p for p in window if p < median), default=None)
    if rules.peak_off_price_percentile is not None:
        peak = price_percentile(window, rules.peak_off_price_percentile)
        if peak is not None and peak <= median:
            peak = min((p for p in window if p > median), default=None)
    return early, peak


def step(
    rules: PriceRules,
    state: RuleState,
    *,
    now: datetime,
    in_window: bool,
    next_on: datetime | None,
    price: float | None,
    early_threshold: float | None,
    peak_threshold: float | None,
    keep_warm: bool = False,
) -> tuple[RuleState, Verdict]:
    """One decision.

    Args:
        in_window: a window is active, or its unconditional preheat has begun.
        next_on: when the plug would next come on without these rules (the next
            window start minus preheat). None when unknown — non-calendar gates
            have no future, so early start is a no-op for them.
        price: current import price, None when unknown or stale.
        early_threshold / peak_threshold: from :func:`thresholds`.
        keep_warm: the room is too cold for a peak-off (thermostat guard).
    """
    # --- early start latch ---------------------------------------------------
    in_early_span = (
        rules.early_enabled
        and not in_window
        and next_on is not None
        and next_on - timedelta(hours=rules.early_start_hours) <= now < next_on
    )
    latched = state.early_latched and in_early_span
    if (
        in_early_span
        and not latched
        and price is not None
        and early_threshold is not None
        and price <= early_threshold
    ):
        latched = True

    if not (in_window or latched):
        return RuleState(), Verdict(False, PHASE_IDLE, "no window active")

    base_phase = PHASE_WINDOW if in_window else PHASE_EARLY
    base_reason = (
        "window active"
        if in_window
        else f"early start: price at or below the {rules.early_start_price_percentile:.0f}th "
        "percentile before the window"
    )

    # --- peak off --------------------------------------------------------------
    at_peak = (
        rules.peak_enabled
        and price is not None
        and peak_threshold is not None
        and price >= peak_threshold
        and not keep_warm
    )
    if state.peak_off_since is not None:
        off_for = now - state.peak_off_since
        if at_peak and off_for < timedelta(minutes=rules.peak_off_max_minutes):
            return (
                replace(state, early_latched=latched),
                Verdict(False, PHASE_PEAK_OFF, "price peak — off"),
            )
        if at_peak:
            why = f"; peak-off ended after {rules.peak_off_max_minutes:.0f} min"
        elif keep_warm:
            why = "; room too cold for a peak-off"
        else:
            why = ""
        return (
            RuleState(early_latched=latched, heat_since=now),
            Verdict(True, base_phase, base_reason + why),
        )

    heat_since = state.heat_since
    if at_peak:
        # A heat period that BEGINS at a peak may wait it out; one already running
        # must first have been on for the minimum time.
        on_long_enough = heat_since is None or now - heat_since >= timedelta(
            minutes=rules.peak_off_min_on_minutes
        )
        if on_long_enough and rules.peak_off_max_minutes > 0:
            return (
                RuleState(early_latched=latched, peak_off_since=now),
                Verdict(False, PHASE_PEAK_OFF, "price peak — off"),
            )
        base_reason += (
            f"; price peak, but on for less than {rules.peak_off_min_on_minutes:.0f} min"
        )
    return (
        RuleState(early_latched=latched, heat_since=heat_since or now),
        Verdict(True, base_phase, base_reason),
    )


def rolling_price_window(
    starts: Sequence[datetime],
    prices: Sequence[float | None],
    index: int,
    hours: float,
) -> list[float]:
    """The price window as seen FROM slot ``index`` of a price series: every price
    from that slot up to ``hours`` ahead and, when the series ends sooner, as much of
    the already-passed part as is needed to span ``hours`` — the same shape as the
    executor's live window (future first, backfilled from the past)."""
    span = timedelta(hours=hours)
    start = starts[index]
    forward = [
        p
        for s, p in zip(starts[index:], prices[index:], strict=False)
        if p is not None and s < start + span
    ]
    covered = (starts[-1] - start) if len(starts) > index else timedelta(0)
    missing = span - covered
    if missing <= timedelta(0) or index == 0:
        return forward
    past = [
        p
        for s, p in zip(starts[:index], prices[:index], strict=False)
        if p is not None and s >= start - missing
    ]
    return past + forward


def simulate(
    rules: PriceRules,
    state: RuleState,
    *,
    starts: Sequence[datetime],
    prices: Sequence[float | None],
    in_window: Sequence[bool],
    next_on: Sequence[datetime | None],
) -> list[Verdict]:
    """Walk :func:`step` forward over a slot series (the planner's horizon): one
    verdict per slot, evaluated at the slot's start with that slot's price, the latch
    and the peak-off clocks carried from slot to slot."""
    verdicts: list[Verdict] = []
    for i, now in enumerate(starts):
        early, peak = thresholds(
            rules, rolling_price_window(starts, prices, i, rules.price_window_hours)
        )
        state, verdict = step(
            rules,
            state,
            now=now,
            in_window=in_window[i],
            next_on=next_on[i],
            price=prices[i],
            early_threshold=early,
            peak_threshold=peak,
        )
        verdicts.append(verdict)
    return verdicts
