"""Historical repair of corrupted slot observations from Home Assistant statistics.

Why this exists (2026-08-03 incident): the recorder used to SKIP the whole
observation whenever the Sungrow ``load_power`` register read 0 — which it does
for hours during any strong export — and the gap backfill then zero-filled the
skipped slots (PV has no cumulative source). Result: every sunny midday since
early July is stored as ``pv_kwh=0, load_kwh=0`` while the plant demonstrably
produced 50+ kWh/day. Those artifact rows poisoned /api/forecast/eval and
starved ML training of exactly the strong-export slots the PV residual model
needs most (train.py's ``load_kwh > 0.001`` filter drops them).

The truth still exists: HA long-term statistics (hourly, permanent retention)
for the combined PV power sensor, and the cumulative grid/battery counters —
all of which stay healthy through the load-register glitch. This module
rebuilds the artifact slots from that data:

    pv_kwh      = solpaneler hourly mean / 4 (flat within the hour)
    import/export/battery = hourly counter delta ("change") / 4
    water/ev    = device power sensor hourly mean / 4
    load_kwh    = pv + import - export + discharge - charge - water - ev
                  (energy balance; clamped >= 0 — same BASE-load semantics as
                  the live recorder, which subtracts EV + water)

Hourly-flat 15-min slots are a huge upgrade over fake zeros, and only ARTIFACT
rows (load_kwh <= 0.001) or missing rows are touched — healthy high-resolution
rows are never overwritten. Writes go through the store's F35 upsert (energy
overwrites only when the new value > 0), and repaired rows are tagged in
``quality_flags``.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger("darkstar.repair")

# Guard: never touch the live edge — the recorder owns the most recent slots.
_LIVE_EDGE_MINUTES = 60

# One WS call per sensor for the whole window; hourly points are small.
_WS_TIMEOUT_S = 30.0


def _ws_url(http_url: str) -> str:
    """Derive the websocket API URL from the configured HA base URL."""
    base = http_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/api/websocket"
    return "ws://" + base[len("http://") :] + "/api/websocket"


async def fetch_statistics_during_period(
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    period: str = "hour",
    types: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """One-shot ``recorder/statistics_during_period`` call over a fresh WS connection.

    Returns ``{entity_id: [{"start": epoch_ms, "mean": ..., "change": ...}, ...]}``.
    Raises on auth/transport errors — the caller decides how loud to be.
    """
    import websockets

    from backend.core.secrets import load_home_assistant_config

    ha = load_home_assistant_config()
    url = _ws_url(str(ha.get("url", "")))
    token = str(ha.get("token", ""))

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), _WS_TIMEOUT_S))
        if hello.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected WS hello: {hello.get('type')}")
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth = json.loads(await asyncio.wait_for(ws.recv(), _WS_TIMEOUT_S))
        if auth.get("type") != "auth_ok":
            raise RuntimeError(f"HA WS auth failed: {auth.get('type')}")

        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start.astimezone(UTC).isoformat(),
                    "end_time": end.astimezone(UTC).isoformat(),
                    "statistic_ids": entity_ids,
                    "period": period,
                    "types": types or ["mean", "change"],
                }
            )
        )
        reply = json.loads(await asyncio.wait_for(ws.recv(), _WS_TIMEOUT_S))
        if not reply.get("success"):
            raise RuntimeError(f"statistics_during_period failed: {reply.get('error')}")
        result: dict[str, list[dict[str, Any]]] = reply.get("result") or {}
        return result


def _hour_key(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)


def _hourly_kwh(
    stats: dict[str, list[dict[str, Any]]], entity: str | None, kind: str
) -> dict[datetime, float]:
    """Hourly energy (kWh) for one entity from its statistics rows.

    kind "mean_w": power sensor in W — hourly kWh = mean / 1000.
    kind "change": cumulative kWh counter — hourly kWh = change.
    Missing entity / missing field -> empty dict (treated as 0 downstream).
    """
    if not entity:
        return {}
    out: dict[datetime, float] = {}
    for row in stats.get(entity, []):
        start_raw = row.get("start")
        if start_raw is None:
            continue
        hour = _hour_key(int(start_raw))
        if kind == "mean_w":
            v = row.get("mean")
            if v is not None:
                out[hour] = max(0.0, float(v)) / 1000.0
        else:
            v = row.get("change")
            if v is not None:
                out[hour] = max(0.0, float(v))
    return out


def compute_slot_record(
    slot_start: datetime,
    hourly: dict[str, dict[datetime, float]],
) -> dict[str, Any] | None:
    """Build one repaired observation record for a 15-min slot from hourly series.

    ``hourly`` maps series name -> {hour_start_utc: kWh_for_that_hour}. Each slot
    gets 1/4 of its hour's energy (flat-within-hour). Returns None when the hour
    has NO pv and NO import data at all (statistics gap — nothing to write).
    """
    hour = slot_start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)

    def q(name: str) -> float | None:
        series = hourly.get(name) or {}
        v = series.get(hour)
        return None if v is None else v / 4.0

    pv = q("pv")
    imp = q("imp")
    if pv is None and imp is None:
        return None

    exp = q("exp") or 0.0
    cha = q("cha") or 0.0
    dis = q("dis") or 0.0
    water = q("water") or 0.0
    ev = q("ev") or 0.0
    pv_v = pv or 0.0
    imp_v = imp or 0.0

    load = max(0.0, pv_v + imp_v - exp + dis - cha - water - ev)
    return {
        "slot_start": slot_start,
        "slot_end": slot_start + timedelta(minutes=15),
        "pv_kwh": round(pv_v, 4),
        "load_kwh": round(load, 4),
        "import_kwh": round(imp_v, 4),
        "export_kwh": round(exp, 4),
        "water_kwh": round(water, 4),
        "ev_charging_kwh": round(ev, 4),
        "batt_charge_kwh": round(cha, 4),
        "batt_discharge_kwh": round(dis, 4),
        "quality_flags": json.dumps({"repaired": "statistics_backfill"}),
    }


async def repair_observations(
    config: dict[str, Any],
    start: datetime,
    end: datetime,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Repair artifact observation slots (load_kwh <= 0.001, or missing) in [start, end).

    Sources everything from HA long-term statistics (hourly). Returns a summary;
    with dry_run=True nothing is written.
    """
    from backend.learning import get_learning_engine

    now = datetime.now(UTC)
    end = min(end, now - timedelta(minutes=_LIVE_EDGE_MINUTES))
    if end <= start:
        return {"error": "empty window after live-edge clamp", "repaired": 0}

    input_sensors: dict[str, Any] = config.get("input_sensors", {}) or {}
    water_sensors = [
        str(wh.get("power_sensor") or wh.get("sensor"))
        for wh in config.get("water_heaters", [])
        if wh.get("enabled", True) and (wh.get("power_sensor") or wh.get("sensor"))
    ]
    ev_sensors = [
        str(ev.get("sensor"))
        for ev in config.get("ev_chargers", [])
        if ev.get("enabled", True) and ev.get("sensor")
    ]

    series_spec: list[tuple[str, str | None, str]] = [
        ("pv", input_sensors.get("pv_power"), "mean_w"),
        ("imp", input_sensors.get("total_grid_import"), "change"),
        ("exp", input_sensors.get("total_grid_export"), "change"),
        ("cha", input_sensors.get("total_battery_charge"), "change"),
        ("dis", input_sensors.get("total_battery_discharge"), "change"),
    ]
    entity_ids = [e for _, e, _ in series_spec if e] + water_sensors + ev_sensors
    if not input_sensors.get("pv_power"):
        return {"error": "no pv_power sensor configured", "repaired": 0}

    stats = await fetch_statistics_during_period(entity_ids, start, end)

    hourly: dict[str, dict[datetime, float]] = {
        name: _hourly_kwh(stats, entity, kind) for name, entity, kind in series_spec
    }
    # Sum multi-device water/EV into single series.
    for name, sensors in (("water", water_sensors), ("ev", ev_sensors)):
        merged: dict[datetime, float] = {}
        for s in sensors:
            for hour, kwh in _hourly_kwh(stats, s, "mean_w").items():
                merged[hour] = merged.get(hour, 0.0) + kwh
        hourly[name] = merged

    # Existing rows in the window: find artifacts + missing slots.
    engine = get_learning_engine()
    tz = engine.timezone
    rows = await engine.store.get_observation_rows_between(
        start.astimezone(tz).isoformat(), end.astimezone(tz).isoformat()
    )
    existing: dict[datetime, float] = {}
    for r in rows:
        try:
            ts = pd.to_datetime(r["slot_start"]).astimezone(UTC).to_pydatetime()
            existing[ts] = float(r.get("load_kwh") or 0.0)
        except Exception:
            continue

    # Walk the 15-min slot grid.
    slot = start.astimezone(UTC).replace(second=0, microsecond=0)
    slot -= timedelta(minutes=slot.minute % 15)
    records: list[dict[str, Any]] = []
    scanned = artifacts = missing = no_data = 0
    while slot < end:
        scanned += 1
        load_existing = existing.get(slot)
        is_artifact = load_existing is not None and load_existing <= 0.001
        is_missing = load_existing is None
        if is_artifact or is_missing:
            rec = compute_slot_record(slot, hourly)
            if rec is None:
                no_data += 1
            else:
                artifacts += int(is_artifact)
                missing += int(is_missing)
                records.append(rec)
        slot += timedelta(minutes=15)

    summary: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "slots_scanned": scanned,
        "artifact_rows": artifacts,
        "missing_rows": missing,
        "slots_without_statistics": no_data,
        "records_to_write": len(records),
        "dry_run": dry_run,
        "repaired": 0,
        "sample": [
            {**r, "slot_start": r["slot_start"].isoformat(), "slot_end": r["slot_end"].isoformat()}
            for r in records[:5]
        ],
    }

    if not dry_run and records:
        df = pd.DataFrame(records)
        await engine.store.store_slot_observations(df)
        summary["repaired"] = len(records)
        logger.info(
            "Repaired %d observation slots (%d artifacts, %d missing) in [%s, %s)",
            len(records),
            artifacts,
            missing,
            start.isoformat(),
            end.isoformat(),
        )
    return summary
