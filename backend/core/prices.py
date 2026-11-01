import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytz
import yaml
from nordpool.elspot import Prices

from backend.core import grid_tariff
from backend.core.cache import cache_sync

logger = logging.getLogger("darkstar.core.prices")

# Price-component entity overrides, resolved into the matching ``pricing.*_sek_kwh`` literals.
# Single source for BOTH resolvers (prices._resolve_pricing_overrides and the forecasts.py
# forward-curve path) so the list can never drift between them.
PRICE_COMPONENT_ENTITY_KEYS = (
    "export_premium_entity",
    "export_grid_benefit_entity",
    "export_fee_entity",
    "grid_transfer_fee_entity",
    "energy_tax_entity",
)

# A per-kWh price adder (transfer / tax / premium / ...) is well under this in SEK. A value at
# or above it from a UNITLESS entity is almost certainly öre/kWh, so rescue it (÷100) + warn —
# this prevents a bare input_number (37.5 öre) being read as 37.5 SEK (a 100x pricing error).
_ORE_MAGNITUDE_CEILING_SEK = 5.0

# Negative-result memo: after a failed/empty Nordpool fetch, don't re-attempt for this
# long. Several executor-tick consumers (EV servo spot, water idle-hold, battery cost
# tracker) call get_nordpool_data every ~60 s; without this, an upstream outage makes
# every tick block on a doomed ~10 s fetch inside the actuation path. Monotonic clock;
# benign races (worst case one extra attempt) are acceptable.
_FETCH_FAIL_MEMO_S = 300.0
_fetch_fail_until: float = 0.0

# After the ~13:00 auction a cached series without tomorrow's real prices is re-polled,
# at most this often while tomorrow is simply unpublished (a failed D+1 fetch backs off
# for _FETCH_FAIL_MEMO_S instead).
_TOMORROW_UNPUBLISHED_RETRY_S = 60.0
_tomorrow_retry_at: float = 0.0

# Last successfully fetched+processed series (any pricing key). Served, minus past
# slots, when a fetch fails entirely -- stale-but-valid beats [] while it covers now.
_last_good_series: list[dict[str, Any]] = []
_stale_logged_at: float = float("-inf")

# Last good reading per price-override entity: (SEK/kWh, aware UTC read time). Used when
# HA cannot be read, so a hiccup neither silently swaps in the config literals (different
# fees) nor changes the cache key and forces a refetch. Entities currently being served
# from memory are in _override_stale_logged (one warning per outage, not per call).
_override_last_good: dict[str, tuple[float, datetime]] = {}
_override_stale_logged: set[str] = set()


async def get_nordpool_data(
    config_path: str = "config.yaml",
    *,
    pricing_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    # Resolve sensor-backed export-price components for EVERY caller (recorder ML labels,
    # schedule/forecast APIs, planner) — not just the planner path — so the effective export
    # price is consistent everywhere. Callers that already resolved them pass pricing_overrides;
    # otherwise we resolve here. No-op (and no HA calls) when no export_*_entity is configured.
    if pricing_overrides is None:
        pricing_overrides = await _resolve_pricing_overrides(config)
    if pricing_overrides:
        config["pricing"] = {**config.get("pricing", {}), **pricing_overrides}

    # --- Smart Cache Check ---
    # The cache key embeds the export components so a contract/sensor change invalidates any
    # stale price series instead of serving export prices computed under the old contract.
    _pc: dict[str, Any] = config.get("pricing", {}) or {}
    cache_key = "nordpool_data:{}:{}:{}:{}:{}:{}:{}".format(
        _pc.get("export_includes_spot", True),
        _pc.get("export_premium_sek_kwh", 0.0),
        _pc.get("export_grid_benefit_sek_kwh", 0.0),
        _pc.get("export_fee_sek_kwh", 0.0),
        _pc.get("grid_transfer_fee_sek", 0.2456),
        _pc.get("energy_tax_sek", 0.439),
        grid_tariff.cache_token(_pc),
    )
    cached = cache_sync.get(cache_key)

    local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
    now = datetime.now(local_tz)
    today = now.date()

    import time as _time

    if cached and len(cached) > 0:
        first_slot = cached[0]["start_time"]
        first_slot_date = first_slot.date() if hasattr(first_slot, "date") else today
        # Only PUBLISHED prices count as "tomorrow": ML-forecast fill-ins
        # (price_is_forecast) must not keep the cache alive past the ~13:00
        # auction, or the forecast is served for up to an hour after real
        # prices exist.
        has_tomorrow = any(
            s["start_time"].date() > today and not s.get("price_is_forecast")
            for s in cached
            if hasattr(s["start_time"], "date")
        )

        if first_slot_date < today:
            cached = None
        elif first_slot > now and now.hour < 23:
            current_slot_start = now.replace(
                minute=(now.minute // 15) * 15, second=0, microsecond=0
            )
            if first_slot > current_slot_start:
                cached = None

        # Re-poll for tomorrow's real prices after the auction, but throttled: a late
        # publication (or a failing D+1 endpoint) must not turn every executor tick
        # into a blocking Nordpool round-trip.
        if (
            cached
            and now.hour >= 13
            and not has_tomorrow
            and _time.monotonic() >= _tomorrow_retry_at
        ):
            cached = None

        if cached:
            return cached

    nordpool_config = config.get("nordpool", {})
    price_area = nordpool_config.get("price_area", "SE4")
    currency = nordpool_config.get("currency", "SEK")
    resolution_minutes = nordpool_config.get("resolution_minutes", 60)

    import asyncio

    global _fetch_fail_until
    if _time.monotonic() < _fetch_fail_until:
        return _stale_but_valid(now)

    prices_client = Prices(currency=currency)

    async def _fetch_day(day: date) -> list[dict[str, Any]]:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                prices_client.fetch,
                end_date=day,
                areas=[price_area],
                resolution=resolution_minutes,
            ),
            timeout=10.0,
        )
        if raw and "areas" in raw and price_area in raw["areas"]:
            values = raw["areas"][price_area].get("values", [])
            return [v for v in values if v["start"].astimezone(local_tz).date() == day]
        return []

    try:
        # Fetch prices for today and tomorrow using to_thread with timeout
        today_values = await _fetch_day(today)
    except TimeoutError:
        _fetch_fail_until = _time.monotonic() + _FETCH_FAIL_MEMO_S
        print("Warning: Nordpool price fetch timed out after 10 seconds")
        return _stale_but_valid(now)
    except Exception as exc:
        _fetch_fail_until = _time.monotonic() + _FETCH_FAIL_MEMO_S
        print(f"Warning: Failed to fetch Nordpool prices: {exc}")
        import traceback

        traceback.print_exc()
        return _stale_but_valid(now)

    try:
        tomorrow_values: list[dict[str, Any]] = []
        if now.hour >= 13:
            # A failing D+1 fetch must not throw away today's good prices: keep them
            # and let the forecast fallback cover tomorrow (retried after the memo).
            try:
                tomorrow_values = await _fetch_day(today + timedelta(days=1))
            except Exception as exc:
                _set_tomorrow_retry(_FETCH_FAIL_MEMO_S)
                logger.warning("Nordpool D+1 fetch failed, keeping today's prices: %s", exc)
            else:
                if not tomorrow_values:
                    _set_tomorrow_retry(_TOMORROW_UNPUBLISHED_RETRY_S)

        all_entries = today_values + tomorrow_values

        # Tomorrow's real prices not available (before the ~13:00 auction, a late
        # publication or a failed D+1 fetch): fill tomorrow from the price forecast.
        # Skipped when export_includes_spot is false: the model is trained on the
        # observed export price, which then carries no spot, so it cannot yield one.
        _forecast_usable = bool(
            (config.get("pricing", {}) or {}).get("export_includes_spot", True)
        )
        if not tomorrow_values and _forecast_usable:
            try:
                from ml.price_forecast import get_d1_price_forecast_fallback

                forecast_fallback = await get_d1_price_forecast_fallback(config)
                if forecast_fallback:
                    fc_entries = forecast_fallback_entries(
                        forecast_fallback, config, local_tz, int(resolution_minutes)
                    )
                    # Forecast only AFTER the last real slot: it never overrides a
                    # published price, not even a partially overlapping one.
                    if all_entries:
                        last_real_end = max(e["end"] for e in all_entries)
                        fc_entries = [e for e in fc_entries if e["start"] >= last_real_end]
                    all_entries.extend(fc_entries)
                    print(
                        f"[get_nordpool_data] Using D+1 price forecast fallback ({len(fc_entries)} slots)"
                    )
            except Exception as exc:
                print(f"Warning: Failed to get D+1 price forecast fallback: {exc}")

        if not today_values:
            # Nordpool answered without today's prices: nothing published to plan on.
            _fetch_fail_until = _time.monotonic() + _FETCH_FAIL_MEMO_S
            return _stale_but_valid(now)

        processed = _process_nordpool_data(all_entries, config)
        cache_sync.set(cache_key, processed, ttl_seconds=3600.0)
        _remember_good_series(processed)
        return processed
    except Exception as exc:
        _fetch_fail_until = _time.monotonic() + _FETCH_FAIL_MEMO_S
        print(f"Warning: Failed to process Nordpool prices: {exc}")
        import traceback

        traceback.print_exc()
        return _stale_but_valid(now)


def _set_tomorrow_retry(seconds: float) -> None:
    import time as _time

    global _tomorrow_retry_at
    _tomorrow_retry_at = _time.monotonic() + seconds


def _remember_good_series(series: list[dict[str, Any]]) -> None:
    global _last_good_series
    _last_good_series = list(series)


def _stale_but_valid(now: datetime) -> list[dict[str, Any]]:
    """Future part of the last good price series, or [] if it no longer covers now.

    Serves a whole-fetch outage from what was already known (e.g. yesterday's fetch
    that included today) instead of handing the planner/servo nothing. Returned
    slots keep their original prices and flags; only past slots are dropped.
    """
    if not _last_good_series:
        return []
    remaining = [s for s in _last_good_series if s["end_time"] > now]
    if not remaining or remaining[0]["start_time"] > now:
        return []
    import time as _time

    global _stale_logged_at
    if _time.monotonic() - _stale_logged_at >= _FETCH_FAIL_MEMO_S:
        _stale_logged_at = _time.monotonic()
        logger.warning(
            "Nordpool unavailable: serving last good price series (%d slots, until %s)",
            len(remaining),
            remaining[-1]["end_time"].isoformat(),
        )
    return remaining


async def get_nordpool_data_for_date(
    day: date, config_path: str = "config.yaml"
) -> list[dict[str, Any]]:
    """Processed slot prices for one PAST local day, straight from Nordpool.

    get_nordpool_data() only ever holds today (and tomorrow), so a recorded slot
    that missed its price could never be repaired afterwards. Not cached and
    not memoised: this is called a handful of times per repair run, for days
    whose prices no longer change. Returns [] on any failure.
    """
    import asyncio

    with Path(config_path).open() as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    # Same fee resolution as get_nordpool_data, so a repaired slot is priced
    # the way a live one would be.
    overrides = await _resolve_pricing_overrides(config)
    if overrides:
        config["pricing"] = {**config.get("pricing", {}), **overrides}
    nordpool_config = config.get("nordpool", {})
    price_area = nordpool_config.get("price_area", "SE4")
    local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                Prices(currency=nordpool_config.get("currency", "SEK")).fetch,
                end_date=day,
                areas=[price_area],
                resolution=nordpool_config.get("resolution_minutes", 60),
            ),
            timeout=10.0,
        )
    except Exception as exc:
        logger.warning("Nordpool fetch for %s failed: %s", day, exc)
        return []
    values = ((raw or {}).get("areas") or {}).get(price_area, {}).get("values", [])
    entries = [v for v in values if v["start"].astimezone(local_tz).date() == day]
    return _process_nordpool_data(entries, config) if entries else []


def forecast_value_to_spot_sek_kwh(model_value: float, config: dict[str, Any]) -> float:
    """Convert a price-model output back to raw spot (SEK/kWh).

    The price model is trained on ``slot_observations.export_price_sek_kwh`` -- the
    EFFECTIVE export price (spot + premium + grid_benefit - fee) -- not raw spot.
    Feeding that straight into :func:`calculate_import_export_prices` added the export
    adders a second time (and VAT on them into import). Subtracting the CURRENT adders
    inverts it, so the derived export price equals the model output exactly and import
    is built from spot as for a published slot.

    With ``export_includes_spot`` false the observed export price carries no spot at all,
    so nothing can be inverted; the value is returned unchanged.
    """
    pricing = config.get("pricing", {}) or {}
    if not bool(pricing.get("export_includes_spot", True)):
        return float(model_value)
    adders = (
        float(pricing.get("export_premium_sek_kwh", 0.0) or 0.0)
        + float(pricing.get("export_grid_benefit_sek_kwh", 0.0) or 0.0)
        - float(pricing.get("export_fee_sek_kwh", 0.0) or 0.0)
    )
    return float(model_value) - adders


def forecast_fallback_entries(
    forecasts: list[dict[str, Any]],
    config: dict[str, Any],
    local_tz: Any,
    resolution_minutes: int,
) -> list[dict[str, Any]]:
    """Turn D+1 price-forecast rows into raw Nordpool-style entries (SEK/MWh spot).

    Forecast rows are on a 15-minute grid. At a coarser Nordpool resolution they are
    averaged into one entry per resolution bucket -- emitting each quarter with a
    60-minute end time produced four overlapping slots per hour.
    """
    step = timedelta(minutes=max(15, resolution_minutes))
    buckets: dict[datetime, list[float]] = {}
    for fc in forecasts:
        slot_start = fc["slot_start"]
        if isinstance(slot_start, str):
            slot_start = datetime.fromisoformat(slot_start)
        slot_start = (
            local_tz.localize(slot_start)
            if slot_start.tzinfo is None
            else slot_start.astimezone(local_tz)
        )
        if resolution_minutes > 15:
            # Bucket on the UTC epoch: whole-hour buckets stay DST-safe.
            epoch = int(slot_start.timestamp())
            size = int(step.total_seconds())
            slot_start = datetime.fromtimestamp(epoch - epoch % size, tz=pytz.UTC).astimezone(
                local_tz
            )
        spot = forecast_value_to_spot_sek_kwh(float(fc.get("spot_p50") or 0.0), config)
        buckets.setdefault(slot_start, []).append(spot)

    return [
        {
            "start": start,
            "end": start + step,
            "value": sum(vals) / len(vals) * 1000.0,  # SEK/kWh -> SEK/MWh
            # Not a published price: flagged so the planner input,
            # schedule.json and logs can tell it apart.
            "price_is_forecast": True,
        }
        for start, vals in sorted(buckets.items())
    ]


def calculate_import_export_prices(
    spot_price_mwh: float, config: dict[str, Any], slot_start: datetime | None = None
) -> tuple[float, float]:
    """
    Calculate import and export prices from spot price.

    Export price is the *net compensation you actually receive* for exported energy,
    built from configurable components so it tracks your contract rather than assuming
    raw spot:

        export = (spot if export_includes_spot) + premium + grid_benefit - fee

    where premium (elhandlarens påslag), grid_benefit (nätnytta) and fee are SEK/kWh.
    All components default to 0.0, so an unconfigured system keeps the legacy behaviour
    ``export == spot``. Sensor-backed components are resolved upstream into these same
    ``pricing`` keys via :func:`resolve_export_price_components` so a contract change just
    means updating a sensor/helper — no code change.

    Import price is ``(spot + grid_transfer_fee + energy_tax) * (1 + VAT)``. When
    ``fees_include_vat`` is true the fees are treated as already VAT-inclusive, so VAT is
    applied to the spot only and the fees are added as-is — letting you point
    ``grid_transfer_fee_entity`` / ``energy_tax_entity`` straight at your "as billed" öre
    helpers (nätavgift / energiskatt) without the VAT being applied a second time.

    Args:
        spot_price_mwh: Spot price in SEK/MWh
        config: Configuration dictionary

    Returns:
        tuple: (import_price_sek_kwh, export_price_sek_kwh)
    """
    pricing_config = config.get("pricing", {})
    vat_percent = pricing_config.get("vat_percent", 25.0)
    # Per-slot time-of-use fee when pricing.grid_transfer_fee_schedule is enabled (else flat).
    grid_transfer_fee_sek = grid_tariff.transfer_fee_for(
        slot_start,
        pricing_config,
        pricing_config.get("grid_transfer_fee_sek", 0.2456),
        config.get("timezone", "Europe/Stockholm"),
    )
    energy_tax_sek = pricing_config.get("energy_tax_sek", 0.439)

    spot_price_sek_kwh = spot_price_mwh / 1000.0

    include_spot = bool(pricing_config.get("export_includes_spot", True))
    export_premium_sek = float(pricing_config.get("export_premium_sek_kwh", 0.0) or 0.0)
    export_grid_benefit_sek = float(pricing_config.get("export_grid_benefit_sek_kwh", 0.0) or 0.0)
    export_fee_sek = float(pricing_config.get("export_fee_sek_kwh", 0.0) or 0.0)
    export_price_sek_kwh = (
        (spot_price_sek_kwh if include_spot else 0.0)
        + export_premium_sek
        + export_grid_benefit_sek
        - export_fee_sek
    )

    # Import price. ``fees_include_vat`` selects which VAT convention the configured grid
    # fees are in: false (legacy) = VAT-exclusive fees, VAT applied to (spot + fees); true =
    # VAT-inclusive fees (e.g. Swedish öre helpers "as billed"), VAT applied to spot only and
    # the fees added as-is. Both give the same total for the same real tariff.
    fees_include_vat = bool(pricing_config.get("fees_include_vat", False))
    vat_multiplier = 1 + vat_percent / 100.0
    if fees_include_vat:
        import_price_sek_kwh = (
            spot_price_sek_kwh * vat_multiplier + grid_transfer_fee_sek + energy_tax_sek
        )
    else:
        import_price_sek_kwh = (
            spot_price_sek_kwh + grid_transfer_fee_sek + energy_tax_sek
        ) * vat_multiplier

    return import_price_sek_kwh, export_price_sek_kwh


def price_entity_to_sek(state: dict[str, Any] | None) -> float | None:
    """Read an HA price entity's state as SEK/kWh, converting öre/kWh -> SEK/kWh by unit.

    Swedish tariff helpers (nätavgift / energiskatt) are usually in öre/kWh, but Darkstar's
    price math is in SEK/kWh — so a value whose ``unit_of_measurement`` mentions öre is
    divided by 100. Returns ``None`` when the state is missing or not float-parseable.
    """
    if state is None:
        return None
    raw = state.get("state")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    attrs = cast("dict[str, Any]", state.get("attributes") or {})
    unit = str(attrs.get("unit_of_measurement", "")).strip().lower()
    if unit.startswith("öre") or unit.startswith("ore"):
        return value / 100.0
    if unit.startswith("sek") or unit.startswith("kr"):
        return value
    # No SEK/öre unit set: a per-kWh adder at/above the ceiling is almost certainly öre.
    if abs(value) >= _ORE_MAGNITUDE_CEILING_SEK:
        logger.warning(
            "price_entity_to_sek: %.3f has no SEK/öre unit_of_measurement; assuming öre/kWh "
            "(÷100). Set the helper's unit to SEK/kWh or öre/kWh to silence this.",
            value,
        )
        return value / 100.0
    return value


def resolve_export_price_components(
    pricing_config: dict[str, Any],
    get_state: Callable[[str], float | None],
) -> dict[str, Any]:
    """Resolve sensor-backed price components (export AND import) into literal SEK/kWh.

    Returns a shallow copy of ``pricing_config`` where each ``*_sek_kwh`` value is overwritten
    by the current reading of its companion ``*_entity`` HA sensor, when configured and
    readable. ``get_state`` maps an ``entity_id`` to a float in SEK/kWh (callers do any
    öre->SEK conversion via :func:`price_entity_to_sek` before this) or ``None``. Unset /
    unreadable entities leave the literal untouched, so this is always safe to call.

    This is the single-source price hook: point e.g. ``grid_transfer_fee_entity`` /
    ``energy_tax_entity`` (import) or ``export_premium_entity`` (export) at your own
    input_number/sensor and adjust it when the contract changes — no redeploy.
    """
    resolved = dict(pricing_config)
    for value_key, entity_key in (
        ("export_premium_sek_kwh", "export_premium_entity"),
        ("export_grid_benefit_sek_kwh", "export_grid_benefit_entity"),
        ("export_fee_sek_kwh", "export_fee_entity"),
        # Import-side adders (single source: point these at your nätavgift / energiskatt helpers).
        ("grid_transfer_fee_sek", "grid_transfer_fee_entity"),
        ("energy_tax_sek", "energy_tax_entity"),
    ):
        entity_id = str(pricing_config.get(entity_key, "") or "").strip()
        if not entity_id:
            continue
        value = get_state(entity_id)
        if value is not None:
            resolved[value_key] = float(value)
    return resolved


async def _resolve_pricing_overrides(config: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve sensor-backed export-price components from their HA entities into literal
    SEK/kWh overrides, so every price consumer (recorder ML labels, schedule/forecast APIs,
    planner) shares ONE effective export price. Returns ``None`` (making zero HA calls) when no
    ``export_*_entity`` is configured.

    An entity that cannot be read (HA down, ``unavailable``, exception) uses its last good
    value from this process, logged once per outage; only an entity never read successfully
    falls back to the config literal.
    """
    pricing_cfg: dict[str, Any] = config.get("pricing", {}) or {}
    if not any(str(pricing_cfg.get(k, "") or "").strip() for k in PRICE_COMPONENT_ENTITY_KEYS):
        return None

    from backend.core import ha_client  # lazy import to avoid any import cycle

    values: dict[str, float] = {}
    for ek in PRICE_COMPONENT_ENTITY_KEYS:
        eid = str(pricing_cfg.get(ek, "") or "").strip()
        if not eid:
            continue
        try:
            value = price_entity_to_sek(await ha_client.get_ha_entity_state(eid))
        except Exception as exc:
            logger.debug("Pricing: reading %s failed: %s", eid, exc)
            value = None
        if value is not None:
            values[eid] = value
            _override_last_good[eid] = (value, datetime.now(pytz.UTC))
            if eid in _override_stale_logged:
                _override_stale_logged.discard(eid)
                logger.info("Pricing: %s readable again (%.4f SEK/kWh)", eid, value)
            continue
        remembered = _override_last_good.get(eid)
        if remembered is not None:
            values[eid] = remembered[0]
            if eid not in _override_stale_logged:
                _override_stale_logged.add(eid)
                logger.warning(
                    "Pricing: %s unreadable; using last good value %.4f SEK/kWh from %s",
                    eid,
                    remembered[0],
                    remembered[1].isoformat(),
                )
        elif eid not in _override_stale_logged:
            _override_stale_logged.add(eid)
            logger.warning(
                "Pricing: %s unreadable and never read successfully; using the config literal.",
                eid,
            )
    if not values:
        return None
    return resolve_export_price_components(pricing_cfg, lambda e: values.get(e))


def _process_nordpool_data(
    all_entries: list[dict[str, Any]],
    config: dict[str, Any],
    today_values: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Process raw Nordpool API data into the required format.

    Args:
        all_entries: Combined list of raw price entries from today and tomorrow
        config: The full configuration dictionary, typed as dict[str, Any]

    Returns:
        list: Processed list of dictionaries with standardized format
    """
    result: list[dict[str, Any]] = []

    # Get local timezone
    local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))

    # Process the hourly data
    for i, entry in enumerate(all_entries):
        # Manual timezone conversion
        if today_values is not None and i < len(today_values):
            # Original entries - use their actual timestamps
            start_time = entry["start"].astimezone(local_tz)
            end_time = entry["end"].astimezone(local_tz)
        else:
            # Extended entries - calculate timestamps based on position
            if today_values is not None and len(today_values) > 0:
                base_start = today_values[0]["start"].astimezone(local_tz)
                slot_duration = today_values[0]["end"] - today_values[0]["start"]
                start_time = base_start + (slot_duration * i)
                end_time = start_time + slot_duration
            else:
                # Fallback if no today_values available
                start_time = entry["start"].astimezone(local_tz)
                end_time = entry["end"].astimezone(local_tz)

        import_price, export_price = calculate_import_export_prices(
            entry["value"], config, slot_start=start_time
        )

        result.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                # Raw day-ahead spot (excl. VAT/fees). Consumers whose thresholds are
                # written in spot terms (EV servo tiers, export curtailment) need this;
                # deriving it back out of the fee-and-VAT-laden totals is lossy.
                "spot_price_sek_kwh": round(entry["value"] / 1000.0, 5),
                "import_price_sek_kwh": import_price,
                "export_price_sek_kwh": export_price,
                # True only for D+1 slots synthesized from the ML price forecast
                # (pre-auction fallback); published Nordpool prices are False.
                "price_is_forecast": bool(entry.get("price_is_forecast", False)),
            }
        )

    # Sort by start time to ensure chronological order
    result.sort(key=lambda x: x["start_time"])

    # Deduplicate by start_time, keeping first occurrence (Nordpool wins over fallback)
    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for entry in result:
        key = entry["start_time"]
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    dup_count = len(result) - len(deduped)
    if dup_count > 0:
        logger.info(
            "_process_nordpool_data dropped %d duplicate start_time entries (kept first occurrence)",
            dup_count,
        )

    return deduped


async def get_known_spot_by_slot(config_path: str = "config.yaml") -> dict[datetime, float]:
    """Published (non-forecast) prices keyed by tz-aware local 15-minute slot start.

    Values are ``export_price_sek_kwh`` -- the same series the recorder stores in
    ``slot_observations`` and the price model is trained on, so they can stand in
    for an observation that does not exist yet. Slots synthesized from the ML
    forecast fallback (``price_is_forecast``) are excluded. Coarser resolutions are
    expanded to one key per covered quarter-hour. Empty on any failure.
    """
    try:
        price_data = await get_nordpool_data(config_path)
    except Exception as exc:
        logger.warning("Known prices unavailable: %s", exc)
        return {}

    quarter = timedelta(minutes=15)
    known: dict[datetime, float] = {}
    for slot in price_data or []:
        if slot.get("price_is_forecast"):
            continue
        value = slot.get("export_price_sek_kwh")
        if value is None:
            continue
        cursor = slot["start_time"]
        while cursor < slot["end_time"]:
            known[cursor] = float(value)
            cursor += quarter
    return known


async def get_current_slot_prices(config: dict[str, Any]) -> dict[str, float] | None:
    """
    Fetch prices for the current 15-minute slot.
    """
    try:
        price_data = await get_nordpool_data()
        if not price_data:
            return None

        local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
        now = datetime.now(local_tz)

        # Find the slot containing 'now'
        for slot in price_data:
            if slot["start_time"] <= now < slot["end_time"]:
                return {
                    "spot_price_sek_kwh": slot.get("spot_price_sek_kwh"),
                    "import_price_sek_kwh": slot["import_price_sek_kwh"],
                    "export_price_sek_kwh": slot["export_price_sek_kwh"],
                }
        return None
    except Exception as exc:
        print(f"Warning: Failed to get current slot prices: {exc}")
        return None
