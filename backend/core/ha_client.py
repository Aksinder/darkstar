import asyncio
import contextlib
import logging
import math
import re
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytz
import yaml

from backend.core import secrets
from backend.core.ev_arrival import (
    OVERRIDE_AUTO,
    OVERRIDE_FORCE_OFF,
    come_home_probability,
    load_arrival_profile,
    reserve_kwh,
)
from backend.core.ev_presence import ev_is_home, haversine_km
from backend.health import set_load_forecast_status

logger = logging.getLogger("darkstar.core.ha_client")


def _as_float(value: Any) -> float | None:
    """Best-effort float, or None (for optional lat/lon/attribute values)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def make_ha_headers(token: str) -> dict[str, str]:
    """Return headers for Home Assistant REST calls."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def gather_sensor_reads(
    reads: list[tuple[str, Callable[[], Coroutine[Any, Any, Any]]]],
    context: str = "sensor_batch",
) -> dict[str, Any]:
    """Run multiple sensor reads concurrently using asyncio.gather().

    Args:
        reads: List of (name, coroutine_factory) pairs. Each factory is called
               to produce a coroutine (e.g., lambda: get_ha_sensor_float(entity_id)).
        context: Label included in log messages to identify the call site.

    Returns:
        Dict mapping each name to its result value, or None if that read failed.
    """
    names = [name for name, _ in reads]
    coros = [fn() for _, fn in reads]
    raw = await asyncio.gather(*coros, return_exceptions=True)

    out: dict[str, Any] = {}
    failures = 0
    for name, result in zip(names, raw, strict=True):
        if isinstance(result, Exception):
            logger.warning("[%s] Sensor read failed for '%s': %s", context, name, result)
            out[name] = None
            failures += 1
        else:
            out[name] = result

    if failures > 0 and failures == len(reads):
        logger.warning("[%s] All %d sensor reads failed", context, failures)

    return out


async def get_ha_entity_state(entity_id: str) -> dict[str, Any] | None:
    """Fetch a single entity state from Home Assistant asynchronously."""
    ha_config = secrets.load_home_assistant_config()
    url = ha_config.get("url")
    token = ha_config.get("token")

    if not url or not token or not entity_id:
        print(
            f"[get_ha_entity_state] Missing config: url={bool(url)}, token={bool(token)}, entity={entity_id}"
        )
        return None

    endpoint = f"{url.rstrip('/')}/api/states/{entity_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(endpoint, headers=make_ha_headers(token))
            response.raise_for_status()
            data = response.json()
            return data
    except Exception as exc:
        print(f"Warning: Could not fetch HA entity {entity_id}: {exc}")
        return None


async def get_ha_sensor_float(entity_id: str) -> float | None:
    """Return numeric state of HA sensor asynchronously."""
    state = await get_ha_entity_state(entity_id)
    if not state:
        return None

    raw_value = state.get("state")
    if raw_value in (None, "unknown", "unavailable"):
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


async def get_ha_sensor_kw_normalized(entity_id: str) -> float | None:
    """Return numeric state of HA sensor normalized to kW (scales W to kW)."""
    state_data = await get_ha_entity_state(entity_id)
    if not state_data:
        return None

    raw_value = state_data.get("state")
    if raw_value in (None, "unknown", "unavailable"):
        return None

    try:
        value = float(raw_value)
        # Check units
        attributes = state_data.get("attributes", {})
        unit = str(attributes.get("unit_of_measurement", "")).upper()
        if unit == "W":
            return value / 1000.0
        return value
    except (TypeError, ValueError):
        return None


def _normalize_energy_to_kwh(value: float, unit: str | None) -> float:
    """Normalize energy value to kWh based on Home Assistant unit_of_measurement.

    Handles common energy units: Wh, kWh, MWh with case-insensitive matching.
    Uses magnitude-based heuristic when no unit is specified.

    Args:
        value: The raw numeric value from HA
        unit: The unit_of_measurement attribute from HA state

    Returns:
        Value normalized to kWh
    """
    if not unit:
        if value > 100_000:
            result = value / 1000.0
            logger.info(
                "Energy normalization: %s (no unit) → %s kWh (Wh inferred from magnitude)",
                value,
                result,
            )
            return result
        logger.debug("Energy normalization: %s (no unit) → %s kWh (assumed kWh)", value, value)
        return value

    unit_clean = re.sub(r"[^A-Z0-9]", "", str(unit).upper())

    if unit_clean in ("WH", "WATTHOUR", "WATTHOURS"):
        result = value / 1000.0
        logger.debug(
            "Energy normalization: %s %s → %s kWh (from unit_of_measurement)", value, unit, result
        )
        return result
    elif unit_clean in ("KWH", "KILOWATTHOUR", "KILOWATTHOURS"):
        logger.debug(
            "Energy normalization: %s %s → %s kWh (from unit_of_measurement)", value, unit, value
        )
        return value
    elif unit_clean in ("MWH", "MEGAWATTHOUR", "MEGAWATTHOURS"):
        result = value * 1000.0
        logger.debug(
            "Energy normalization: %s %s → %s kWh (from unit_of_measurement)", value, unit, result
        )
        return result
    else:
        logger.warning(
            "Energy normalization: unknown unit '%s' for value %s, assuming kWh", unit, value
        )
        return value


async def get_energy_from_power_history(
    entity_id: str,
    start: datetime,
    end: datetime,
) -> float | None:
    """Fetch power sensor history and compute energy via average power x time.

    Returns energy in kWh, or None if history unavailable.
    """
    ha_config = secrets.load_home_assistant_config()
    url = ha_config.get("url")
    token = ha_config.get("token")

    if not url or not token or not entity_id:
        return None

    api_url = f"{url.rstrip('/')}/api/history/period/{start.isoformat()}"
    params = {
        "filter_entity_id": entity_id,
        "end_time": end.isoformat(),
        "significant_changes_only": False,
        "minimal_response": False,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(api_url, headers=make_ha_headers(token), params=params)
            response.raise_for_status()
            data = response.json()

        if not data or not data[0]:
            return None

        return integrate_power_history_kwh(data[0], start, end)

    except Exception as exc:
        logger.warning("get_energy_from_power_history(%s): %s", entity_id, exc)
        return None


def integrate_power_history_kwh(
    states: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> float | None:
    """Time-weighted integral of a power-sensor history over [start, end], in kWh.

    The old implementation was an UNWEIGHTED mean of the state changes times the
    window length. State-change history is event-driven, so a 5-minute element
    burst contributes dozens of samples while a long idle stretch contributes
    one — the mean was dominated by whichever mode changed most often, not by
    time. Live consequence (2026-08-25): the house VVB's 3.5 kW bursts landed as
    0.000 kWh whenever the (mis-chosen) window caught the idle side, and would
    have been over-counted with a correct window. Each sample now covers exactly
    the span until the next sample (clamped to the window), i.e. the standard
    step-function integral of HA state history.

    A sample whose value is unknown/unavailable/unparsable is skipped and the
    previous numeric value carries across the gap — the sensor's last report is
    the best estimate of what the element was doing while the sensor was mute.

    Samples without a parsable timestamp fall back to the legacy mean-times-
    duration estimate, so callers that only have bare values still get an answer.
    Returns None when no numeric samples exist at all.
    """
    points: list[tuple[datetime, float]] = []
    bare_kw: list[float] = []
    unit: str | None = None

    for state in states:
        state_val = state.get("state", "")
        if state_val in ("unknown", "unavailable", "", None):
            continue
        try:
            value = float(state_val)
        except (TypeError, ValueError):
            continue

        if unit is None:
            attributes = state.get("attributes", {})
            unit = str(attributes.get("unit_of_measurement", "")).upper()
        if unit == "W":
            kw = value / 1000.0
        elif unit == "MW":
            kw = value * 1000.0
        else:
            kw = value  # Assume kW

        raw_ts = state.get("last_changed") or state.get("last_updated")
        ts: datetime | None = None
        if isinstance(raw_ts, str):
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        elif isinstance(raw_ts, datetime):
            ts = raw_ts
        if ts is not None and ts.tzinfo is None:
            # HA timestamps are UTC when the offset is missing.
            ts = ts.replace(tzinfo=UTC)

        if ts is None:
            bare_kw.append(kw)
        else:
            points.append((ts, kw))

    if not points:
        if not bare_kw:
            return None
        duration_hours = (end - start).total_seconds() / 3600.0
        return (sum(bare_kw) / len(bare_kw)) * duration_hours

    points.sort(key=lambda p: p[0])
    total_kwh = 0.0
    for i, (ts, kw) in enumerate(points):
        seg_start = max(ts, start)
        seg_end = min(points[i + 1][0], end) if i + 1 < len(points) else end
        if seg_end > seg_start:
            total_kwh += kw * (seg_end - seg_start).total_seconds() / 3600.0
    return total_kwh


async def get_ha_bool(entity_id: str) -> bool:
    """Return True if entity is 'on', 'true', 'armed', etc."""
    state = await get_ha_entity_state(entity_id)
    if not state:
        return False

    raw = str(state.get("state", "")).lower()
    # Common 'on' states in Home Assistant
    true_states = {"on", "true", "yes", "1", "armed_away", "armed_home", "armed_night"}
    is_true = raw in true_states
    if is_true and "vacation" in entity_id:
        print(f"DEBUG: Vacation mode detected TRUE. Raw state: '{raw}' from entity '{entity_id}'")
    return is_true


def _water_day_bucket_start(now: datetime, defer_hours: float, tz: Any) -> datetime:
    """Start of the water day-bucket that ``now`` belongs to, matching kepler.

    kepler (planner/solver/kepler.py:888-893) buckets each slot by its LOCAL date,
    moving slots whose local hour is < ``defer_up_to_hours`` into the PREVIOUS day.
    Equivalently the day boundary is at wall-clock hour ``ceil(defer_hours)`` local
    (00:00 when defer_hours == 0). This applies the identical rule to ``now`` so the
    heated-today window is exactly the current bucket [boundary_hour:00, now], never
    a naive local-midnight window when defer differs.
    """
    boundary_hour = math.ceil(defer_hours) if defer_hours > 0 else 0
    boundary_hour = max(0, min(23, boundary_hour))
    bucket_date = now.date()
    if defer_hours > 0 and now.hour < defer_hours:
        bucket_date = bucket_date - timedelta(days=1)
    naive = datetime(bucket_date.year, bucket_date.month, bucket_date.day, boundary_hour, 0, 0)
    # pytz tz => use localize; stdlib/zoneinfo tz => replace(tzinfo=...)
    localize = getattr(tz, "localize", None)
    if callable(localize):
        return cast("datetime", localize(naive))
    return naive.replace(tzinfo=tz)


async def get_water_heated_today_by_tank(config: dict[str, Any]) -> dict[str, float]:
    """MEASURED, per-tank water energy already delivered in the current day-bucket.

    COLD-SHOWER-SAFE by construction:
      * SOURCE is measured, per-tank: summed ONLY from the learning store's
        ``slot_device_energy`` rows (savings-v2, device_id == heater id) over the
        current day-bucket. That sum is integrated per 15-min slot by the recorder and
        cannot exceed what actually ran. We deliberately do NOT fall back to a direct
        power-history integral over the (multi-hour) bucket: mean-of-samples x a long
        window can OVER-estimate, and over-crediting would zero the reliability floor.
      * CONSERVATIVE: every value is clamped to ``[0, min_kwh_per_day]`` so a big
        draw-day can never credit MORE than the daily floor (which would zero the
        cold-shower backstop). When a tank has NO store rows (sparse DB / store read
        failed), it credits 0.0 — the safe OVER-heating direction, never under-heating.

    Returns ``{heater_id: heated_today_kwh}`` for enabled heaters; empty on failure.
    """
    result: dict[str, float] = {}
    try:
        water_heaters = [wh for wh in config.get("water_heaters", []) if wh.get("enabled", True)]
        # Cyclic loads are credited by the same rule: the recorder writes their
        # per-slot energy under their id, and the pre-scheduler deducts it from
        # today's need exactly as kepler does for a tank. Same clamp, same
        # zero-on-uncertainty direction (over-run a filter, never under-run it).
        water_heaters = water_heaters + [
            c for c in (config.get("cyclic_loads", []) or [])
            if isinstance(c, dict) and c.get("enabled", True)
        ]
        if not water_heaters:
            return {}

        wh_cfg = config.get("water_heating", {})
        defer_hours = float(wh_cfg.get("defer_up_to_hours", 0.0) or 0.0)
        tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
        now = datetime.now(tz)
        bucket_start = _water_day_bucket_start(now, defer_hours, tz)

        # 1) Preferred source: measured per-tank rows from the learning store.
        seen_ids: set[str] = set()
        measured_by_id: dict[str, float] = {}
        db_path = config.get("learning", {}).get("sqlite_path", "data/planner_learning.db")
        store: Any = None
        try:
            from backend.learning.store import LearningStore

            store = LearningStore(db_path, tz)
            rows = await store.get_device_energy_rows_between(
                bucket_start.isoformat(), now.isoformat()
            )
            for r in rows:
                did = str(r.get("device_id", ""))
                if not did:
                    continue
                measured_by_id[did] = measured_by_id.get(did, 0.0) + float(r.get("kwh", 0.0) or 0.0)
                seen_ids.add(did)
        except Exception as exc:  # store unavailable => fall back per-tank below
            logger.warning(
                "heated_today: slot_device_energy read failed (%s); using power history", exc
            )
        finally:
            if store is not None:
                with contextlib.suppress(Exception):
                    await store.close()

        # 2) Per-tank: credit ONLY the store's summed per-slot measured energy, clamped.
        # The store sum cannot exceed what actually ran (it is integrated per 15-min slot
        # by the recorder). A direct power-history integral over the whole multi-hour
        # bucket, by contrast, can OVER-estimate (mean-of-samples x long duration), and
        # over-crediting would drive kepler's day_min to 0 and zero the reliability floor
        # -> cold shower. So a tank with NO store rows (sparse DB, or the store read
        # raised above leaving seen_ids empty) credits 0.0 = the safe OVER-heat direction,
        # never under-heat. Do NOT reintroduce a long-window power-history fallback here.
        for wh in water_heaters:
            hid = str(wh.get("id", ""))
            if not hid:
                continue
            min_kwh = float(wh.get("min_kwh_per_day", 0.0) or 0.0)
            if hid in seen_ids:
                credited = max(0.0, min(measured_by_id.get(hid, 0.0), min_kwh))
                source = "store"
            else:
                credited = 0.0  # no trusted measurement -> over-heat, never under-heat
                source = "no_store_rows"

            result[hid] = credited
            logger.info(
                "heated_today[%s]: credited=%.3f kWh (min_kwh=%.2f, bucket_start=%s, source=%s)",
                hid,
                credited,
                min_kwh,
                bucket_start.isoformat(),
                source,
            )
        return result
    except Exception as exc:
        logger.warning("heated_today crediting failed; falling back to 0.0 for all tanks: %s", exc)
        return {}


async def get_water_absorption_stats(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-tank MEASURED absorption stats for the planner's absorption cap (2026-08-10).

    Returns ``{heater_id: {"absorbed_today_kwh": float, "absorbed_daily_avg_kwh": float | None}}``.

    - ``absorbed_today_kwh``: UNCLAMPED slot_device_energy sum for the current
      day-bucket. Deliberately separate from get_water_heated_today_by_tank, whose
      values are clamped to [0, min_kwh] for the cold-shower-safe reliability floor —
      the cap needs the raw number or a boost-heavy morning leaves it too generous.
    - ``absorbed_daily_avg_kwh``: trailing mean over the previous 7 COMPLETE
      day-buckets, divided by the fixed window length (a day the tank absorbed
      nothing is a real zero — the thermostat was satisfied — not missing data).
      This is thermostat-limited measured intake, i.e. the honest estimate of
      draw + standing losses. None when the window holds no rows for the tank
      (fresh install / new device) => the adapter applies NO cap => fail-open.

    Failure of any kind returns {} — callers must treat missing stats as "no cap".
    """
    result: dict[str, dict[str, Any]] = {}
    try:
        water_heaters = [wh for wh in config.get("water_heaters", []) if wh.get("enabled", True)]
        if not water_heaters:
            return {}

        wh_cfg = config.get("water_heating", {})
        defer_hours = float(wh_cfg.get("defer_up_to_hours", 0.0) or 0.0)
        tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))
        now = datetime.now(tz)
        bucket_start = _water_day_bucket_start(now, defer_hours, tz)
        window_start = bucket_start - timedelta(days=7)

        db_path = config.get("learning", {}).get("sqlite_path", "data/planner_learning.db")
        store: Any = None
        today_by_id: dict[str, float] = {}
        window_by_id: dict[str, float] = {}
        covered_days_by_id: dict[str, set[Any]] = {}
        try:
            from backend.learning.store import LearningStore

            store = LearningStore(db_path, tz)
            rows = await store.get_device_energy_rows_between(
                window_start.isoformat(), now.isoformat()
            )
            for r in rows:
                did = str(r.get("device_id", ""))
                if not did:
                    continue
                kwh = float(r.get("kwh", 0.0) or 0.0)
                # Datetime comparison, NOT string: slot_start is a local-offset ISO
                # string, and during the autumn DST fall-back the repeated hour makes
                # lexicographic order diverge from time order ("02:xx+02:00" sorts
                # above "02:yy+01:00") — the repo's recurring tz-trap class.
                try:
                    slot_dt = datetime.fromisoformat(str(r.get("slot_start", "")))
                except (TypeError, ValueError):
                    continue
                if slot_dt.tzinfo is None:
                    continue
                if slot_dt >= bucket_start:
                    today_by_id[did] = today_by_id.get(did, 0.0) + kwh
                else:
                    window_by_id[did] = window_by_id.get(did, 0.0) + kwh
                    # Coverage: which complete day-buckets the recorder actually
                    # covered for this device (it writes rows every slot when healthy,
                    # including 0.0 — so presence means "measured", absence means
                    # "recorder outage", and outage days must not bias the average low).
                    row_bucket = slot_dt.astimezone(tz).date()
                    if slot_dt.astimezone(tz).hour < defer_hours:
                        row_bucket = row_bucket - timedelta(days=1)
                    covered_days_by_id.setdefault(did, set()).add(row_bucket)
        finally:
            if store is not None:
                with contextlib.suppress(Exception):
                    await store.close()

        # The bucket the today-numbers belong to: the planner must NOT subtract them
        # from a NEWER bucket if the solve crosses the 10:00 boundary mid-flight.
        bucket_date_iso = bucket_start.astimezone(tz).date().isoformat()

        for wh in water_heaters:
            hid = str(wh.get("id", ""))
            if not hid:
                continue
            covered = covered_days_by_id.get(hid, set())
            # Average over days the recorder COVERED, not the fixed window length:
            # an outage day is missing data, while a covered day with only 0.0 rows
            # is a genuine "thermostat was satisfied" zero and still counts.
            avg = window_by_id.get(hid, 0.0) / len(covered) if covered else None
            result[hid] = {
                "absorbed_today_kwh": max(0.0, today_by_id.get(hid, 0.0)),
                "absorbed_daily_avg_kwh": avg,
                "absorbed_bucket_date": bucket_date_iso,
            }
        return result
    except Exception as exc:
        logger.warning("water absorption stats failed; planner runs uncapped: %s", exc)
        return {}


async def get_initial_state(
    config_path: str = "config.yaml",
    ev_plugged_in_override: bool | None = None,
    ev_plug_override_charger_id: str | None = None,
) -> dict[str, Any]:
    """
    Get the initial battery state (Asynchronous).

    Args:
        config_path: Path to config.yaml
        ev_plugged_in_override: If provided, use this value for the specific charger
            identified by ev_plug_override_charger_id (or all chargers if None).
        ev_plug_override_charger_id: Charger ID to apply the plug state override to.
            If None and ev_plugged_in_override is set, applies to the first enabled charger
            (legacy behaviour). With per-device replans, this should always be set.
    """
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    # Use system.battery if available, otherwise fall back to battery
    battery_config = config.get("system", {}).get("battery", config.get("battery", {}))
    capacity_kwh = battery_config.get("capacity_kwh", 10.0)
    battery_soc_percent = 50.0
    battery_cost_sek_per_kwh = config.get("battery_economics", {}).get(
        "battery_cycle_cost_kwh", 0.20
    )

    # HA Config
    ha_config = secrets.load_home_assistant_config()
    input_sensors = config.get("input_sensors", {})
    soc_entity_id = input_sensors.get("battery_soc", ha_config.get("soc_entity_id"))

    if soc_entity_id:
        ha_soc = await get_ha_sensor_float(soc_entity_id)
        if ha_soc is not None:
            battery_soc_percent = ha_soc
        else:
            # Critical safety check: Do not default to 50% if we expected a live reading.
            # This causes "phantom charging" when HA is down.
            raise RuntimeError(
                f"Critical: Failed to read battery SoC from {soc_entity_id}. "
                "Planning aborted to prevent unsafe assumptions."
            )

    battery_soc_percent = max(0.0, min(100.0, battery_soc_percent))
    battery_kwh = capacity_kwh * battery_soc_percent / 100.0

    system_config = config.get("system", {})
    # Per-tank MEASURED heated-today (kepler credits this against min_kwh_per_day so
    # it stops re-inserting the full daily floor every replan — the "walking block"
    # root cause). Cold-shower-safe: clamped per tank and 0.0 on any uncertainty.
    water_heater_states: list[dict[str, Any]] = []
    water_heated_today_kwh = 0.0
    try:
        heated_by_id = await get_water_heated_today_by_tank(config)
        # Absorption-cap stats (2026-08-10): unclamped today + trailing 7d average.
        # {} on any failure => the adapter applies no cap (fail-open).
        absorption_stats = await get_water_absorption_stats(config)
        for hid, kwh in heated_by_id.items():
            entry: dict[str, Any] = {"id": hid, "heated_today_kwh": kwh}
            stats = absorption_stats.get(hid)
            if stats:
                entry["absorbed_today_kwh"] = stats.get("absorbed_today_kwh", 0.0)
                entry["absorbed_daily_avg_kwh"] = stats.get("absorbed_daily_avg_kwh")
                entry["absorbed_bucket_date"] = stats.get("absorbed_bucket_date")
            water_heater_states.append(entry)
        water_heated_today_kwh = sum(heated_by_id.values())
    except Exception as exc:
        logger.warning("heated_today: initial-state crediting failed, using 0.0: %s", exc)
        water_heater_states = []
        water_heated_today_kwh = 0.0

    # Cyclic loads: when did each last run? The pre-scheduler's max-gap rule is
    # anchored here, so a filter that last ran at 16:11 is not left idle until
    # tomorrow's cheap hours just because the first planned hour lies beyond the
    # gap. Read from the switch: ON = running now; OFF = its last_changed.
    cyclic_load_states: list[dict[str, Any]] = []
    for cyclic in config.get("cyclic_loads", []) or []:
        if not isinstance(cyclic, dict) or not cyclic.get("enabled", True):
            continue
        cid = str(cyclic.get("id", ""))
        sw = cyclic.get("switch_entity")
        if not cid or not sw:
            continue
        entry_c: dict[str, Any] = {"id": cid, "heated_today_kwh": 0.0, "last_run_end": None}
        for st in water_heater_states:
            if st.get("id") == cid:
                entry_c["heated_today_kwh"] = float(st.get("heated_today_kwh", 0.0) or 0.0)
        try:
            raw = await get_ha_entity_state(str(sw))
            if raw:
                if str(raw.get("state", "")).lower() == "on":
                    entry_c["last_run_end"] = datetime.now(UTC).isoformat()
                elif raw.get("last_changed"):
                    entry_c["last_run_end"] = str(raw["last_changed"])
        except Exception as exc:
            logger.debug("cyclic %s: switch read failed (%s); no gap anchor", cid, exc)
        cyclic_load_states.append(entry_c)

    # Per-device EV state fetching
    has_ev_charger = system_config.get("has_ev_charger", False)
    ev_chargers_cfg = config.get("ev_chargers", [])
    enabled_ev_chargers = [ev for ev in ev_chargers_cfg if ev.get("enabled", True)]

    # Build per-device EV state list
    ev_charger_states: list[dict[str, Any]] = []

    if has_ev_charger and enabled_ev_chargers:
        # Build batch reads for all enabled chargers
        per_device_reads: list[tuple[str, Any]] = []
        for ev in enabled_ev_chargers:
            charger_id = ev.get("id", "")
            soc_sensor = ev.get("soc_sensor", "")
            plug_sensor = ev.get("plug_sensor", "")

            if soc_sensor:
                key = f"ev_soc_{charger_id}"
                per_device_reads.append((key, lambda e=soc_sensor: get_ha_sensor_float(e)))

            # Only fetch plug from HA if no override applies to this charger
            is_override_charger = ev_plug_override_charger_id == charger_id or (
                ev_plug_override_charger_id is None and ev is enabled_ev_chargers[0]
            )
            if plug_sensor and not (ev_plugged_in_override is not None and is_override_charger):
                key = f"ev_plug_{charger_id}"
                per_device_reads.append((key, lambda e=plug_sensor: get_ha_bool(e)))

            # Home-zone presence (e.g. device_tracker for the car). When configured we
            # read its state so the EV can be excluded from the plan while away — the
            # car's API reports plug/charging regardless of location, which would
            # otherwise become a phantom load.
            home_entity = ev.get("home_entity", "")
            if home_entity:
                key = f"ev_home_{charger_id}"
                per_device_reads.append((key, lambda e=home_entity: get_ha_entity_state(e)))

            # Manual override helper (input_select: auto / force_reserve / force_off) for
            # the come-home prediction + gate, so it can be driven from the dashboard.
            ch_cfg: dict[str, Any] = ev.get("come_home", {}) or {}
            override_entity = ch_cfg.get("override_entity", "")
            if override_entity:
                key = f"ev_override_{charger_id}"
                per_device_reads.append((key, lambda e=override_entity: get_ha_entity_state(e)))

        per_device_results: dict[str, Any] = {}
        if per_device_reads:
            per_device_results = await gather_sensor_reads(
                per_device_reads, context="ev_initial_state"
            )

        for ev in enabled_ev_chargers:
            charger_id = ev.get("id", "")
            soc_sensor = ev.get("soc_sensor", "")
            plug_sensor = ev.get("plug_sensor", "")

            # SoC
            soc_percent = 0.0
            if soc_sensor:
                ha_soc_val = per_device_results.get(f"ev_soc_{charger_id}")
                if ha_soc_val is not None:
                    soc_percent = float(ha_soc_val)
                else:
                    logger.warning(
                        "EV %s SoC sensor %s returned no data, defaulting to 0%%",
                        charger_id,
                        soc_sensor,
                    )

            # Plug state
            is_override_charger = ev_plug_override_charger_id == charger_id or (
                ev_plug_override_charger_id is None and ev is enabled_ev_chargers[0]
            )
            if ev_plugged_in_override is not None and is_override_charger:
                plugged_in = ev_plugged_in_override
                logger.debug(
                    "EV %s: using plug state override=%s", charger_id, ev_plugged_in_override
                )
            elif plug_sensor:
                plugged_in = bool(per_device_results.get(f"ev_plug_{charger_id}", False))
            else:
                # No plug sensor → assume plugged in (let enabled flag be the control)
                plugged_in = True

            # Home-zone gate (robust both ways): zone OR distance-within-radius OR a
            # grace window after the tracker flips away. See backend/core/ev_presence.py.
            at_home = True
            ev_reserve = 0.0
            come_home_zone = "home"
            home_entity = ev.get("home_entity", "")
            if home_entity:
                home_obj = per_device_results.get(f"ev_home_{charger_id}")
                zone_state = ""
                car_lat: float | None = None
                car_lon: float | None = None
                last_changed: datetime | None = None
                if isinstance(home_obj, dict):
                    ho = cast("dict[str, Any]", home_obj)
                    zone_state = str(ho.get("state", ""))
                    attrs = ho.get("attributes")
                    if isinstance(attrs, dict):
                        a = cast("dict[str, Any]", attrs)
                        car_lat = _as_float(a.get("latitude"))
                        car_lon = _as_float(a.get("longitude"))
                    lc = ho.get("last_changed")
                    if isinstance(lc, str):
                        last_changed = datetime.fromisoformat(lc.replace("Z", "+00:00"))
                loc = config.get("system", {}).get("location", {})
                home_lat = _as_float(loc.get("latitude"))
                home_lon = _as_float(loc.get("longitude"))
                at_home, reason = ev_is_home(
                    zone_state,
                    home_states=ev.get("home_states") or ["home"],
                    home_lat=home_lat,
                    home_lon=home_lon,
                    car_lat=car_lat,
                    car_lon=car_lon,
                    radius_km=float(ev.get("home_radius_km", 0.0) or 0.0),
                    last_changed=last_changed,
                    grace_minutes=float(ev.get("home_grace_minutes", 0.0) or 0.0),
                    now=datetime.now(UTC),
                )

                # Manual override helper (auto / force_reserve / force_off).
                override = OVERRIDE_AUTO
                ov_obj = per_device_results.get(f"ev_override_{charger_id}")
                if isinstance(ov_obj, dict):
                    override = str(cast("dict[str, Any]", ov_obj).get("state", "")) or OVERRIDE_AUTO
                if override == OVERRIDE_FORCE_OFF:
                    at_home = False  # force off also blocks charging now
                    reason = "override:force_off"
                if not at_home:
                    logger.info("EV %s not home (%s); excluding from plan", charger_id, reason)

                # Come-home prediction (Step 1): soft, capped, low-weight battery buffer
                # to pre-position for a likely arrival. Only when away (home -> charge now).
                ch_cfg2: dict[str, Any] = ev.get("come_home", {}) or {}
                if ch_cfg2.get("enabled", False) and not at_home:
                    dist: float | None = None
                    if (
                        car_lat is not None
                        and car_lon is not None
                        and home_lat is not None
                        and home_lon is not None
                    ):
                        dist = haversine_km(car_lat, car_lon, home_lat, home_lon)
                    p, come_home_zone, ch_reason = come_home_probability(
                        datetime.now(UTC),
                        dist,
                        load_arrival_profile(config, charger_id),
                        override=override,
                        near_radius_km=float(ch_cfg2.get("near_radius_km", 0.0) or 0.0),
                        extended_radius_km=float(ch_cfg2.get("extended_radius_km", 0.0) or 0.0),
                    )
                    ev_reserve = reserve_kwh(
                        p,
                        float(ch_cfg2.get("buffer_kwh", 0.0) or 0.0),
                        float(ch_cfg2.get("max_reserve_kwh", 0.0) or 0.0),
                    )
                    if ev_reserve > 0:
                        logger.info(
                            "EV %s come-home: zone=%s p=%.2f reserve=%.2f kWh (%s)",
                            charger_id,
                            come_home_zone,
                            p,
                            ev_reserve,
                            ch_reason,
                        )

            ev_charger_states.append(
                {
                    "id": charger_id,
                    "soc_percent": soc_percent,
                    "plugged_in": plugged_in,
                    "at_home": at_home,
                    "reserve_kwh": ev_reserve,
                    "come_home_zone": come_home_zone,
                }
            )

    # Build aggregate values for backward compatibility
    ev_soc_percent = ev_charger_states[0]["soc_percent"] if ev_charger_states else 0.0
    ev_plugged_in = ev_charger_states[0]["plugged_in"] if ev_charger_states else False

    # Effekttariff inputs (only fetched when the feature is on: cost > 0).
    grid_cfg = cast("dict[str, Any]", config.get("grid", {}) or {})
    peak_power_baseline_kw = float(grid_cfg.get("peak_power_baseline_kw", 0.0) or 0.0)
    peak_hour_elapsed_import_kwh = 0.0
    if float(grid_cfg.get("peak_power_cost_sek_per_kw", 0.0) or 0.0) > 0.0:
        # Baseline: month-to-date peak import (kW) so the planner only pays for RAISING
        # the monthly peak. Unit-aware (W/kW via unit_of_measurement); a unitless value
        # falls back to a magnitude heuristic; an unreadable entity keeps the literal.
        peak_entity = str(grid_cfg.get("peak_power_baseline_entity", "") or "").strip()
        if peak_entity:
            peak_state = await get_ha_entity_state(peak_entity)
            live_peak: float | None = None
            peak_unit = ""
            if peak_state:
                raw_peak = peak_state.get("state")
                if raw_peak not in (None, "unknown", "unavailable"):
                    try:
                        live_peak = float(raw_peak)
                    except (TypeError, ValueError):
                        live_peak = None
                peak_unit = str(
                    cast("dict[str, Any]", peak_state.get("attributes", {}) or {}).get(
                        "unit_of_measurement", ""
                    )
                ).strip()
            if live_peak is None:
                logger.warning(
                    "peak_power_baseline_entity %s unreadable; using literal %.2f kW",
                    peak_entity,
                    peak_power_baseline_kw,
                )
            else:
                if peak_unit.upper() == "W":
                    live_peak = live_peak / 1000.0
                elif not peak_unit and abs(live_peak) > 1000.0:
                    # No unit set: a household month-peak above 1000 is almost surely W.
                    logger.warning(
                        "peak_power_baseline_entity %s = %.0f has no unit; assuming W "
                        "(set the sensor's unit_of_measurement to kW or W to silence this)",
                        peak_entity,
                        live_peak,
                    )
                    live_peak = live_peak / 1000.0
                peak_power_baseline_kw = float(live_peak)

        # Elapsed import of the in-progress clock hour up to the horizon start (the
        # planner floors "now" to the plan resolution), so a mid-hour replan prices the
        # billed FULL-hour mean instead of just the remaining fraction. Best-effort:
        # unavailable history => 0.0 (the pre-fix behaviour for this hour only).
        import_entity = str(
            input_sensors.get("grid_import_power") or input_sensors.get("grid_power") or ""
        ).strip()
        if import_entity:
            res_min = int(config.get("nordpool", {}).get("resolution_minutes", 15) or 15)
            now_local = datetime.now().astimezone()
            hour_start = now_local.replace(minute=0, second=0, microsecond=0)
            horizon_start = now_local.replace(
                minute=(now_local.minute // res_min) * res_min, second=0, microsecond=0
            )
            if horizon_start > hour_start:
                elapsed = await get_energy_from_power_history(
                    import_entity, hour_start, horizon_start
                )
                if elapsed is None:
                    logger.warning(
                        "effekttariff: no import history for %s in the current hour; "
                        "assuming 0 kWh elapsed",
                        import_entity,
                    )
                else:
                    # net meter can go negative (export); billed import can't
                    peak_hour_elapsed_import_kwh = max(0.0, float(elapsed))

    return {
        "battery_soc_percent": battery_soc_percent,
        "battery_kwh": battery_kwh,
        "battery_cost_sek_per_kwh": battery_cost_sek_per_kwh,
        "water_heated_today_kwh": water_heated_today_kwh,
        # Per-tank measured heated-today (pipeline consumes this into
        # water_heated_today_by_id; kepler subtracts it from min_kwh_per_day).
        "water_heater_states": water_heater_states,
        "cyclic_load_states": cyclic_load_states,
        # Legacy scalar fields (backward compat)
        "ev_soc_percent": ev_soc_percent,
        "ev_plugged_in": ev_plugged_in,
        # Per-device EV state list
        "ev_charger_states": ev_charger_states,
        # Effekttariff: month-to-date peak import (kW) + elapsed import of the
        # in-progress clock hour before the horizon start (kWh)
        "peak_power_baseline_kw": peak_power_baseline_kw,
        "peak_hour_elapsed_import_kwh": peak_hour_elapsed_import_kwh,
    }


async def get_load_profile_from_ha(config: dict[str, Any]) -> list[float]:
    """Fetch actual load profile from Home Assistant historical data (Async)."""
    ha_config = secrets.load_home_assistant_config()
    url: str | None = cast("str | None", ha_config.get("url"))
    token = cast("str", ha_config.get("token", ""))

    _sensors_cfg: Any = config.get("input_sensors", {})
    if isinstance(_sensors_cfg, dict):
        input_sensors: dict[str, Any] = cast("dict[str, Any]", _sensors_cfg)
    else:
        input_sensors = {}

    entity_id: str | None = input_sensors.get(
        "total_load_consumption", ha_config.get("consumption_entity_id")
    )

    if not all([url, token, entity_id]):
        print("Warning: Missing Home Assistant configuration for load profile")
        return get_dummy_load_profile(config)

    headers = make_ha_headers(token)
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=7)

    url_str: str = cast("str", url)
    api_url = f"{url_str.rstrip('/')}/api/history/period/{start_time.isoformat()}"
    params = {
        "filter_entity_id": entity_id,
        "end_time": end_time.isoformat(),
        "significant_changes_only": False,
        "minimal_response": False,
    }

    try:
        print(f"Fetching {entity_id} data from Home Assistant...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(api_url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
        if not data or not data[0]:
            print(f"Warning: No data received from Home Assistant for {entity_id}")
            return get_dummy_load_profile(config)

        states = data[0]
        if len(states) < 2:
            print(f"Warning: Insufficient data points from Home Assistant for {entity_id}")
            return get_dummy_load_profile(config)

        # Convert to local timezone for processing
        local_tz = pytz.timezone("Europe/Stockholm")

        # Calculate energy consumption between state changes
        time_buckets = [0.0] * (96 * 7)  # 7 days * 96 slots per day
        prev_state = None
        prev_time = None
        cached_unit: str | None = None

        start_time_local = start_time.astimezone(local_tz)

        for state in states:
            try:
                # Skip unavailable/unknown/null states silently
                state_val = state.get("state", "")
                if state_val in ("unavailable", "unknown", "null", "", None):
                    continue

                current_time = datetime.fromisoformat(state["last_changed"])
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=pytz.UTC)
                current_time = current_time.astimezone(local_tz)
                current_value = float(state_val)

                # Normalize energy unit to kWh (handles Wh, kWh, MWh)
                attributes = state.get("attributes", {})
                unit = attributes.get("unit_of_measurement")
                if unit is not None and unit != "":
                    cached_unit = unit
                if unit is None or unit == "":
                    unit = cached_unit
                current_value = _normalize_energy_to_kwh(current_value, unit)

                if prev_state is not None and prev_time is not None:
                    # Calculate energy delta (ensure positive)
                    energy_delta = max(0, current_value - prev_state)

                    # Distribute across time buckets
                    time_diff = current_time - prev_time
                    minutes_diff = time_diff.total_seconds() / 60

                    if minutes_diff > 0 and energy_delta > 0:
                        # Calculate which 15-minute buckets this spans
                        start_slot = int((prev_time.hour * 60 + prev_time.minute) // 15)
                        end_slot = int((current_time.hour * 60 + current_time.minute) // 15)
                        day_offset = int(
                            (prev_time - start_time_local).total_seconds() / (24 * 3600)
                        )

                        # Calculate start and end times for each slot
                        for slot_idx in range(max(0, start_slot), min(96, end_slot + 1)):
                            # Calculate slot start time relative to the day start
                            slot_start_minutes = slot_idx * 15
                            day_start = prev_time.replace(hour=0, minute=0, second=0, microsecond=0)
                            slot_start_time = day_start + timedelta(minutes=slot_start_minutes)
                            slot_end_time = slot_start_time + timedelta(minutes=15)

                            # Calculate overlap between this slot and the energy consumption period
                            overlap_start = max(prev_time, slot_start_time)
                            overlap_end = min(current_time, slot_end_time)
                            overlap_minutes = max(
                                0, (overlap_end - overlap_start).total_seconds() / 60
                            )

                            if overlap_minutes > 0:
                                # Distribute energy proportionally to time overlap
                                energy_fraction = overlap_minutes / minutes_diff
                                energy_for_slot = energy_delta * energy_fraction

                                bucket_idx = day_offset * 96 + slot_idx
                                if 0 <= bucket_idx < len(time_buckets):
                                    time_buckets[bucket_idx] += energy_for_slot

                prev_state = current_value
                prev_time = current_time

            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Skipping invalid state data for {entity_id}: {e}")
                continue

        # Create average daily profile from the 7 days of data (divide by 7 days)
        daily_profile = [0.0] * 96
        for slot in range(96):
            slot_sum = 0.0
            for day in range(7):
                bucket_idx = day * 96 + slot
                if 0 <= bucket_idx < len(time_buckets):
                    slot_sum += time_buckets[bucket_idx]
            daily_profile[slot] = slot_sum / 7.0

        # Validate and clean the profile
        total_daily = sum(daily_profile)
        if total_daily > 500:
            print(
                f"Warning: Daily total {total_daily:.1f} kWh/day for {entity_id} exceeds 500 kWh sanity bound, using dummy profile"
            )
            return get_dummy_load_profile(config)
        if total_daily <= 0:
            print(f"Warning: No valid energy consumption data found for {entity_id}")
            return get_dummy_load_profile(config)

        print(f"Successfully loaded HA data: {total_daily:.2f} kWh/day average")

        # Ensure all values are positive and reasonable
        for i in range(96):
            if daily_profile[i] < 0:
                daily_profile[i] = 0
            elif daily_profile[i] > 10:  # Cap at 10kW per 15min
                daily_profile[i] = 10

        return daily_profile

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"Warning: Failed to fetch data from Home Assistant for {entity_id}: {e}")
        return get_dummy_load_profile(config)
    except Exception as e:
        print(f"Warning: Error processing Home Assistant data for {entity_id}: {e}")
        return get_dummy_load_profile(config)


def get_dummy_load_profile(config: dict[str, Any]) -> list[float]:
    """Create a dummy load profile or a synthetic scaled profile.

    If config.input_sensors.total_load_consumption is a number (estimated daily kWh),
    we generate a synthetic winter heat-pump curve scaled to that daily total.
    Otherwise, we fall back to a 0.5 kWh flat dummy profile.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check if the user provided an estimated daily kWh (from Startup Wizard)
    # The wizard stores this as a string, e.g. "20", in the total_load_consumption field
    # if they selected 'synthetic' mode.
    estimated_daily_kwh = None
    sensors = config.get("input_sensors", {})
    raw_val = sensors.get("total_load_consumption")

    if raw_val is not None:
        try:
            val = float(raw_val)
            if val > 0 and not str(raw_val).startswith(("sensor.", "input_")):
                estimated_daily_kwh = val
        except (ValueError, TypeError):
            pass

    if estimated_daily_kwh is not None:
        logger.info(
            f"Generating Synthetic Heat Pump profile scaled to {estimated_daily_kwh} kWh/day."
        )
        set_load_forecast_status("synthetic", "estimated")

        # Base normalized heat pump curve (higher in night/morning, lower in afternoon)
        # 96 slots representing a standard winter day shape. Sums to ~1.0.
        base_curve = [
            1.2,
            1.2,
            1.1,
            1.1,
            1.1,
            1.1,
            1.2,
            1.2,  # 00:00 - 02:00
            1.2,
            1.3,
            1.3,
            1.3,
            1.4,
            1.4,
            1.5,
            1.6,  # 02:00 - 04:00
            1.7,
            1.8,
            1.9,
            1.9,
            2.0,
            2.0,
            1.9,
            1.8,  # 04:00 - 06:00
            1.7,
            1.6,
            1.5,
            1.4,
            1.3,
            1.2,
            1.1,
            1.0,  # 06:00 - 08:00
            0.9,
            0.9,
            0.8,
            0.8,
            0.8,
            0.7,
            0.7,
            0.7,  # 08:00 - 10:00
            0.7,
            0.6,
            0.6,
            0.6,
            0.6,
            0.5,
            0.5,
            0.5,  # 10:00 - 12:00
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.6,  # 12:00 - 14:00
            0.6,
            0.6,
            0.7,
            0.7,
            0.8,
            0.8,
            0.9,
            1.0,  # 14:00 - 16:00
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.7,  # 16:00 - 18:00
            1.6,
            1.5,
            1.4,
            1.3,
            1.2,
            1.1,
            1.0,
            1.0,  # 18:00 - 20:00
            0.9,
            0.9,
            0.9,
            1.0,
            1.0,
            1.0,
            1.1,
            1.1,  # 20:00 - 22:00
            1.1,
            1.1,
            1.1,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,  # 22:00 - 00:00
        ]

        curve_sum = sum(base_curve)
        # Scale the curve so its integral (sum) equals the estimated daily kWh
        return [(val / curve_sum) * estimated_daily_kwh for val in base_curve]

    logger.warning(
        "⚠️ Using DEMO load profile (0.5 kWh flat) - no historical data available. Configure total_load_consumption sensor for accurate forecasts."
    )

    # REV F65 Phase 5b: Set degraded status when using demo data
    set_load_forecast_status("degraded", "demo")

    return [0.5] * 96
