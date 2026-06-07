import contextlib
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import yaml
from nordpool.elspot import Prices

from backend.core.cache import cache_sync

logger = logging.getLogger("darkstar.core.prices")


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
    cache_key = "nordpool_data:{}:{}:{}:{}".format(
        _pc.get("export_includes_spot", True),
        _pc.get("export_premium_sek_kwh", 0.0),
        _pc.get("export_grid_benefit_sek_kwh", 0.0),
        _pc.get("export_fee_sek_kwh", 0.0),
    )
    cached = cache_sync.get(cache_key)

    local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
    now = datetime.now(local_tz)
    today = now.date()

    if cached and len(cached) > 0:
        first_slot = cached[0]["start_time"]
        first_slot_date = first_slot.date() if hasattr(first_slot, "date") else today
        has_tomorrow = any(
            s["start_time"].date() > today for s in cached if hasattr(s["start_time"], "date")
        )

        if first_slot_date < today:
            cached = None
        elif first_slot > now and now.hour < 23:
            current_slot_start = now.replace(
                minute=(now.minute // 15) * 15, second=0, microsecond=0
            )
            if first_slot > current_slot_start:
                cached = None

        if cached and now.hour >= 13 and not has_tomorrow:
            cached = None

        if cached:
            return cached

    nordpool_config = config.get("nordpool", {})
    price_area = nordpool_config.get("price_area", "SE4")
    currency = nordpool_config.get("currency", "SEK")
    resolution_minutes = nordpool_config.get("resolution_minutes", 60)

    import asyncio

    prices_client = Prices(currency=currency)

    try:
        # Fetch prices for today and tomorrow using to_thread with timeout
        raw_today = await asyncio.wait_for(
            asyncio.to_thread(
                prices_client.fetch,
                end_date=today,
                areas=[price_area],
                resolution=resolution_minutes,
            ),
            timeout=10.0,
        )
        today_values = []
        if raw_today and "areas" in raw_today and price_area in raw_today["areas"]:
            today_raw = raw_today["areas"][price_area].get("values", [])
            today_values = [v for v in today_raw if v["start"].astimezone(local_tz).date() == today]

        tomorrow_values = []
        if now.hour >= 13:
            tomorrow = today + timedelta(days=1)
            raw_tomorrow = await asyncio.wait_for(
                asyncio.to_thread(
                    prices_client.fetch,
                    end_date=tomorrow,
                    areas=[price_area],
                    resolution=resolution_minutes,
                ),
                timeout=10.0,
            )
            if raw_tomorrow and "areas" in raw_tomorrow and price_area in raw_tomorrow["areas"]:
                all_raw = raw_tomorrow["areas"][price_area].get("values", [])
                tomorrow_values = [
                    v for v in all_raw if v["start"].astimezone(local_tz).date() == tomorrow
                ]

        all_entries = today_values + tomorrow_values

        # If no tomorrow data yet (before ~13:00 CET auction), try forecast fallback
        if not tomorrow_values and now.hour < 13:
            try:
                from ml.price_forecast import get_d1_price_forecast_fallback

                forecast_fallback = await get_d1_price_forecast_fallback(config)
                if forecast_fallback:
                    # Convert forecast format to match Nordpool format
                    for fc in forecast_fallback:
                        slot_start = fc["slot_start"]
                        if isinstance(slot_start, str):
                            slot_start = datetime.fromisoformat(slot_start)
                        slot_start = (
                            slot_start.replace(tzinfo=local_tz)
                            if slot_start.tzinfo is None
                            else slot_start.astimezone(local_tz)
                        )
                        end_time = slot_start + timedelta(hours=1)
                        all_entries.append(
                            {
                                "start": slot_start,
                                "end": end_time,
                                "value": fc.get("spot_p50", 0) * 1000,  # Convert SEK/kWh to SEK/MWh
                            }
                        )
                    print(
                        f"[get_nordpool_data] Using D+1 price forecast fallback ({len(forecast_fallback)} slots)"
                    )
            except Exception as exc:
                print(f"Warning: Failed to get D+1 price forecast fallback: {exc}")

        if not all_entries:
            return []

        processed = _process_nordpool_data(all_entries, config)
        cache_sync.set(cache_key, processed, ttl_seconds=3600.0)
        return processed
    except TimeoutError:
        print("Warning: Nordpool price fetch timed out after 10 seconds, returning empty data")
        return []
    except Exception as exc:
        print(f"Warning: Failed to fetch Nordpool prices: {exc}")
        import traceback

        traceback.print_exc()
        return []


def calculate_import_export_prices(
    spot_price_mwh: float, config: dict[str, Any]
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

    Args:
        spot_price_mwh: Spot price in SEK/MWh
        config: Configuration dictionary

    Returns:
        tuple: (import_price_sek_kwh, export_price_sek_kwh)
    """
    pricing_config = config.get("pricing", {})
    vat_percent = pricing_config.get("vat_percent", 25.0)
    grid_transfer_fee_sek = pricing_config.get("grid_transfer_fee_sek", 0.2456)
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

    import_price_sek_kwh = (spot_price_sek_kwh + grid_transfer_fee_sek + energy_tax_sek) * (
        1 + vat_percent / 100.0
    )

    return import_price_sek_kwh, export_price_sek_kwh


def resolve_export_price_components(
    pricing_config: dict[str, Any],
    get_state: Callable[[str], float | None],
) -> dict[str, Any]:
    """Resolve sensor-backed export-price components into literal SEK/kWh values.

    Returns a shallow copy of ``pricing_config`` where each ``export_*_sek_kwh`` value is
    overwritten by the current reading of its companion ``export_*_entity`` HA sensor,
    when that entity is configured and readable. ``get_state`` maps an ``entity_id`` to a
    float (SEK/kWh) or ``None`` when unavailable. Unset/unreadable entities leave the
    literal untouched, so this is always safe to call.

    This lets the effective export compensation follow the user's contract live: point an
    ``export_*_entity`` at an ``input_number``/sensor and adjust it when the contract
    changes — no redeploy.
    """
    resolved = dict(pricing_config)
    for value_key, entity_key in (
        ("export_premium_sek_kwh", "export_premium_entity"),
        ("export_grid_benefit_sek_kwh", "export_grid_benefit_entity"),
        ("export_fee_sek_kwh", "export_fee_entity"),
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
    """
    pricing_cfg: dict[str, Any] = config.get("pricing", {}) or {}
    entity_keys = ("export_premium_entity", "export_grid_benefit_entity", "export_fee_entity")
    if not any(str(pricing_cfg.get(k, "") or "").strip() for k in entity_keys):
        return None

    from backend.core import ha_client  # lazy import to avoid any import cycle

    values: dict[str, float] = {}
    for ek in entity_keys:
        eid = str(pricing_cfg.get(ek, "") or "").strip()
        if not eid:
            continue
        state = await ha_client.get_ha_entity_state(eid)
        raw = state.get("state") if state is not None else None
        if raw is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            values[eid] = float(raw)
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

        import_price, export_price = calculate_import_export_prices(entry["value"], config)

        result.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "import_price_sek_kwh": import_price,
                "export_price_sek_kwh": export_price,
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
                    "import_price_sek_kwh": slot["import_price_sek_kwh"],
                    "export_price_sek_kwh": slot["export_price_sek_kwh"],
                }
        return None
    except Exception as exc:
        print(f"Warning: Failed to get current slot prices: {exc}")
        return None
