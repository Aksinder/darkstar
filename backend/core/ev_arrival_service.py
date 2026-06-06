"""
Periodic builder for the EV come-home arrival profile (Step 1).

Reads the car's ``device_tracker`` state history from Home Assistant, turns it into
presence events, builds the weekday x hour home-probability profile, and persists it to
``<config_dir>/ev_arrival_<id>.json`` for ``get_initial_state`` to read. Read-only (one
JSON file written); no hardware control. No-op unless an EV has ``come_home.enabled`` and
a ``home_entity`` configured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from backend.core.ev_arrival import build_arrival_profile, save_arrival_profile

logger = logging.getLogger("darkstar.ev_arrival_service")

__all__ = ["events_from_history", "run_ev_arrival_loop"]


def events_from_history(
    rows: list[dict[str, Any]], home_states: list[str]
) -> list[tuple[datetime, bool]]:
    """Convert HA device_tracker history rows into (timestamp, is_home) events."""
    allowed = {str(s).lower() for s in home_states}
    out: list[tuple[datetime, bool]] = []
    for r in rows:
        state = str(r.get("state", "")).strip().lower()
        if state in ("unknown", "unavailable", ""):
            continue
        ts_raw = r.get("last_changed") or r.get("last_updated")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((ts, state in allowed))
    out.sort(key=lambda e: e[0])
    return out


def _come_home_chargers(config: dict[str, Any]) -> list[dict[str, Any]]:
    chargers: list[dict[str, Any]] = config.get("ev_chargers", []) or []
    out: list[dict[str, Any]] = []
    for ev in chargers:
        ch: dict[str, Any] = ev.get("come_home", {}) or {}
        if ev.get("enabled", True) and ch.get("enabled", False) and ev.get("home_entity") and ev.get("id"):
            out.append(ev)
    return out


async def run_ev_arrival_loop(
    config: dict[str, Any],
    *,
    interval_s: float = 21600.0,  # 6 h — the arrival profile changes slowly
    history_hours: int = 1344,  # 8 weeks
) -> None:
    """Deploy-ready loop: rebuild + persist each come-home charger's arrival profile."""
    import asyncio
    from datetime import timedelta

    import httpx

    from backend.core import secrets
    from backend.core.ha_client import make_ha_headers

    evs = _come_home_chargers(config)
    if not evs:
        logger.info("EV arrival: no come-home chargers configured; not starting")
        return

    ha_cfg = secrets.load_home_assistant_config()
    base_url = ha_cfg.get("url")
    token = ha_cfg.get("token")
    if not base_url or not token:
        logger.warning("EV arrival: HA url/token missing; not starting")
        return

    async def fetch_history(entity_id: str, hours: int) -> list[dict[str, Any]]:
        now = datetime.now().astimezone()
        start = now - timedelta(hours=hours)
        api_url = f"{base_url.rstrip('/')}/api/history/period/{start.isoformat()}"
        params: dict[str, Any] = {
            "filter_entity_id": entity_id,
            "end_time": now.isoformat(),
            "minimal_response": True,
            "significant_changes_only": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(api_url, headers=make_ha_headers(token), params=params)
            resp.raise_for_status()
            data = resp.json()
        return data[0] if data else []

    logger.info(
        "EV arrival profile builder started: %d charger(s), every %.0fs", len(evs), interval_s
    )
    while True:
        for ev in evs:
            cid = str(ev.get("id", ""))
            try:
                rows = await fetch_history(str(ev["home_entity"]), history_hours)
                events = events_from_history(rows, ev.get("home_states") or ["home"])
                profile = build_arrival_profile(events, step_minutes=30)
                save_arrival_profile(config, cid, profile)
                logger.info(
                    "EV arrival profile for %s: %d weekday/hour buckets, %d samples",
                    cid,
                    len(profile.fraction),
                    profile.samples,
                )
            except Exception as exc:
                logger.warning("EV arrival build failed for %s: %s", cid, exc)
        await asyncio.sleep(interval_s)
