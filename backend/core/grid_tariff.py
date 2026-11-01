"""Time-of-use grid transfer fee (nätavgift per kWh) resolved per slot.

Config (``pricing.grid_transfer_fee_schedule``; absent or ``enabled: false`` = the flat
fee exactly as before)::

    pricing:
      grid_transfer_fee_schedule:
        enabled: true
        default_sek: 0.37            # fee outside every window
        windows:                     # first matching window wins
          - name: hoglast
            fee_sek: 0.65
            months: [1, 2, 3, 11, 12]
            weekdays: [0, 1, 2, 3, 4]  # Mon=0
            hours: "06-22"             # [start, end) local wall time
            exclude_holidays: true     # Swedish public holidays + eves never match

Rules:

* Fees are in the same VAT basis as ``pricing.fees_include_vat`` (they simply replace
  ``grid_transfer_fee_sek`` in the import-price formula).
* Windows are matched on the slot START converted to the site's LOCAL wall time
  (``config.timezone``, default Europe/Stockholm). Aware datetimes are converted; naive
  ones are taken as already-local wall time. DST-safe: the 02:00-03:00 hour that does not
  exist in spring simply never occurs, and the repeated autumn hour is matched twice by
  the same wall-clock hour.
* Precedence: an ENABLED, VALID schedule wins over both ``grid_transfer_fee_sek`` and
  ``grid_transfer_fee_entity`` (the entity override is ignored, logged once). When the
  schedule is disabled, absent or invalid, the flat fee (entity override, else literal)
  is used — an invalid schedule logs an error once and never crashes pricing.
* Without a slot time (``slot_start=None``) the flat fee is returned.
"""

from __future__ import annotations

import logging
from datetime import datetime, tzinfo
from typing import Any, cast

import pytz

from backend.core.time_windows import TimeWindow, matches, parse_window

logger = logging.getLogger("darkstar.core.grid_tariff")

SCHEDULE_KEY = "grid_transfer_fee_schedule"
DEFAULT_TZ = "Europe/Stockholm"


class _Compiled:
    __slots__ = ("default_sek", "windows")

    def __init__(self, default_sek: float | None, windows: list[tuple[TimeWindow, float]]):
        self.default_sek = default_sek
        self.windows = windows


# repr(schedule) -> compiled schedule, or None when disabled/invalid. Bounded: configs change
# rarely, but clear if it ever grows (e.g. a test suite churning configs).
_compiled_cache: dict[str, _Compiled | None] = {}
_logged_once: set[str] = set()


def _log_once(level: int, key: str, msg: str, *args: Any) -> None:
    if key in _logged_once:
        return
    _logged_once.add(key)
    logger.log(level, msg, *args)


def _parse_fee(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field} must be a number (SEK/kWh)")
    fee = float(raw)
    if fee < 0 or fee >= 5.0:
        raise ValueError(f"{field} {fee} is outside 0-5 SEK/kWh (öre given instead of SEK?)")
    return fee


def _compile(schedule: dict[str, Any]) -> _Compiled:
    """Validate + compile a schedule. Raises ``ValueError`` on any invalid part."""
    default_raw = schedule.get("default_sek")
    default_sek = None if default_raw is None else _parse_fee(default_raw, "default_sek")
    raw_windows: Any = schedule.get("windows") or []
    if not isinstance(raw_windows, list):
        raise ValueError("windows must be a list")
    windows: list[tuple[TimeWindow, float]] = []
    for i, raw in enumerate(cast("list[Any]", raw_windows)):
        try:
            if not isinstance(raw, dict):
                raise ValueError("window must be a mapping")
            win = cast("dict[str, Any]", raw)
            windows.append((parse_window(win), _parse_fee(win.get("fee_sek"), "fee_sek")))
        except ValueError as exc:
            name = raw.get("name", "") if isinstance(raw, dict) else ""  # type: ignore[union-attr]
            raise ValueError(f"windows[{i}]{f' ({name})' if name else ''}: {exc}") from exc
    return _Compiled(default_sek, windows)


def _active_schedule(pricing_config: dict[str, Any]) -> _Compiled | None:
    """The compiled schedule when enabled AND valid, else ``None`` (= use the flat fee)."""
    raw: Any = pricing_config.get(SCHEDULE_KEY)
    if not raw:
        return None
    key = repr(raw)
    if key in _compiled_cache:
        return _compiled_cache[key]
    compiled: _Compiled | None = None
    if not isinstance(raw, dict):
        _log_once(
            logging.ERROR,
            "notdict:" + key,
            "pricing.%s must be a mapping; using the flat grid transfer fee.",
            SCHEDULE_KEY,
        )
    else:
        sched = cast("dict[str, Any]", raw)
        if sched.get("enabled", False) is True:
            try:
                compiled = _compile(sched)
            except ValueError as exc:
                _log_once(
                    logging.ERROR,
                    "invalid:" + key,
                    "pricing.%s is invalid (%s); falling back to the flat grid transfer fee.",
                    SCHEDULE_KEY,
                    exc,
                )
            else:
                logger.info(
                    "Time-of-use grid transfer fee active: %d window(s), default %s SEK/kWh.",
                    len(compiled.windows),
                    compiled.default_sek if compiled.default_sek is not None else "flat",
                )
    if len(_compiled_cache) > 64:
        _compiled_cache.clear()
    _compiled_cache[key] = compiled
    return compiled


def schedule_enabled(pricing_config: dict[str, Any]) -> bool:
    return _active_schedule(pricing_config) is not None


def _to_local(slot_start: datetime | str, tz: str | tzinfo) -> datetime:
    if isinstance(slot_start, str):
        slot_start = datetime.fromisoformat(slot_start)
    if slot_start.tzinfo is None:
        return slot_start  # naive = already local wall time
    local_tz = pytz.timezone(tz) if isinstance(tz, str) else tz
    return slot_start.astimezone(local_tz)


def transfer_fee_for(
    slot_start: datetime | str | None,
    pricing_config: dict[str, Any],
    fallback_flat_fee: float,
    tz: str | tzinfo = DEFAULT_TZ,
) -> float:
    """Grid transfer fee (SEK/kWh, VAT basis per ``fees_include_vat``) for one slot.

    ``fallback_flat_fee`` is today's flat fee (already entity-resolved); it is returned
    unchanged whenever the schedule is absent, disabled, invalid, or ``slot_start`` is None.
    """
    compiled = _active_schedule(pricing_config)
    if compiled is None or slot_start is None:
        return float(fallback_flat_fee)
    if str(pricing_config.get("grid_transfer_fee_entity", "") or "").strip():
        _log_once(
            logging.INFO,
            "entity_ignored",
            "pricing.grid_transfer_fee_entity is ignored while pricing.%s is enabled.",
            SCHEDULE_KEY,
        )
    try:
        local_dt = _to_local(slot_start, tz)
    except (TypeError, ValueError) as exc:
        _log_once(
            logging.WARNING,
            f"badslot:{exc}",
            "grid tariff: unparseable slot_start %r (%s); using flat fee.",
            slot_start,
            exc,
        )
        return float(fallback_flat_fee)
    for window, fee in compiled.windows:
        if matches(window, local_dt):
            return fee
    return compiled.default_sek if compiled.default_sek is not None else float(fallback_flat_fee)


def config_for_slot(config: dict[str, Any], slot_start: datetime | str | None) -> dict[str, Any]:
    """Return ``config`` with ``pricing.grid_transfer_fee_sek`` pinned to the slot's fee.

    For consumers that only see a (spot, config) pair — e.g. the ML price-forecast
    derivation — so a single-line hook routes them through :func:`transfer_fee_for`.
    Returns ``config`` itself (no copy) when no schedule is active.
    """
    pricing: dict[str, Any] = config.get("pricing", {}) or {}
    if slot_start is None or _active_schedule(pricing) is None:
        return config
    fee = transfer_fee_for(
        slot_start,
        pricing,
        float(pricing.get("grid_transfer_fee_sek", 0.2456)),
        config.get("timezone", DEFAULT_TZ),
    )
    pinned = {k: v for k, v in pricing.items() if k != SCHEDULE_KEY}
    pinned["grid_transfer_fee_sek"] = fee
    return {**config, "pricing": pinned}


def cache_token(pricing_config: dict[str, Any]) -> str:
    """Stable fingerprint of the schedule for price-cache keys (changes invalidate)."""
    raw: Any = pricing_config.get(SCHEDULE_KEY)
    return repr(raw) if raw else ""
