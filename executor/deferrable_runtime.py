"""
Deferrable smart-appliance controller — runtime wiring (turnkey, observe-first).

Give Darkstar a ``power_sensor`` (+ the plug's ``switch_entity``) and it does the rest:
auto-arms when a cycle starts (power rises), picks the cheapest FORECAST window for the
WHOLE cycle before the deadline (duration-aware, unlike a "is it cheap right now?"
trigger), detects done, publishes ``sensor.<prefix><id>_state``, and notifies.

Default OFF. Even when enabled it runs **observe-only** by default: it publishes state +
the forecast recommendation and sends notifications, but does NOT gate the plug — so its
decisions can be verified against the user's existing HA automations before any cutover.

Pure logic lives in ``executor/deferrable.py`` (update_appliance_power_state,
recommend_appliance_action); this module is the I/O shell (read HA, read the planner
schedule, publish, notify, persist), mirroring ``ev_surplus_runtime`` / ``fmb_soc_runtime``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from .deferrable import (
    AppliancePowerConfig,
    AppliancePowerState,
    WindowSlot,
    recommend_appliance_action,
    update_appliance_power_state,
)

logger = logging.getLogger("darkstar.deferrable")

_STATE_FILE = "data/deferrable_state.json"


def _f(v: Any) -> float | None:
    if v is None or v in ("unknown", "unavailable", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class DeferrableApplianceCfg:
    """One appliance's control wiring (parsed from a ``deferrable_loads[]`` entry)."""

    id: str
    name: str
    power_sensor: str | None = None
    switch_entity: str | None = None
    override_entity: str | None = None
    typical_minutes_sensor: str | None = None  # learned duration; falls back to seed
    power: AppliancePowerConfig = field(default_factory=AppliancePowerConfig)
    seed_duration_min: float = 120.0
    deadline_mode: str = "cheapest_within_hours"  # or "hard_deadline"
    window_hours: float = 14.0
    hard_deadline: str | None = None  # "HH:MM"


@dataclass
class DeferrableRuntimeConfig:
    """``executor.deferrable_appliances`` config + the per-appliance list."""

    enabled: bool = False
    observe_only: bool = True
    notify_service: str | None = None
    publish_prefix: str = "darkstar_"
    slot_minutes: float = 15.0
    schedule_path: str = "schedule.json"
    timezone: str = "Europe/Stockholm"
    appliances: list[DeferrableApplianceCfg] = field(default_factory=lambda: [])


def parse_deferrable_runtime_config(
    full_config: dict[str, Any],
) -> DeferrableRuntimeConfig | None:
    """Build runtime config from ``executor.deferrable_appliances`` + top-level
    ``deferrable_loads``. None when the executor block is absent."""
    executor = cast("dict[str, Any]", full_config.get("executor", {}) or {})
    raw = executor.get("deferrable_appliances")
    if not isinstance(raw, dict):
        return None
    raw = cast("dict[str, Any]", raw)

    appliances: list[DeferrableApplianceCfg] = []
    for c in cast("list[dict[str, Any]]", full_config.get("deferrable_loads", []) or []):
        if not c.get("id") or not c.get("enabled", True):
            continue
        if not c.get("power_sensor"):
            continue  # turnkey path requires a power sensor
        lid = str(c["id"])
        prefix = str(raw.get("publish_prefix", "darkstar_"))
        appliances.append(
            DeferrableApplianceCfg(
                id=lid,
                name=str(c.get("name", lid)),
                power_sensor=c.get("power_sensor") or None,
                switch_entity=c.get("switch_entity") or None,
                override_entity=c.get("override_entity") or None,
                typical_minutes_sensor=f"sensor.{prefix}{_slug(lid)}_typical_minutes",
                power=AppliancePowerConfig(
                    on_threshold_w=float(c.get("on_threshold_w", 10.0)),
                    off_threshold_w=float(c.get("off_threshold_w", 3.0)),
                    start_debounce_s=float(c.get("start_debounce_s", 3.0)),
                    done_delay_s=float(c.get("done_delay_s", 300.0)),
                    power_scale=float(c.get("power_scale", 1.0)),
                ),
                seed_duration_min=float(c.get("duration_min", 120.0)),
                deadline_mode=str(c.get("deadline_mode", "cheapest_within_hours")),
                window_hours=float(c.get("window_hours", 14.0)),
                hard_deadline=c.get("hard_deadline") or None,
            )
        )

    return DeferrableRuntimeConfig(
        enabled=bool(raw.get("enabled", False)),
        observe_only=bool(raw.get("observe_only", True)),
        notify_service=raw.get("notify_service") or None,
        publish_prefix=str(raw.get("publish_prefix", "darkstar_")),
        slot_minutes=float(raw.get("slot_minutes", 15.0)),
        schedule_path=str(executor.get("schedule_path", "schedule.json")),
        timezone=str(full_config.get("timezone", "Europe/Stockholm")),
        appliances=appliances,
    )


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.strip().lower())


def load_forward_slots(schedule_path: str, now_ts: float, tz_name: str) -> list[WindowSlot]:
    """Read the planner schedule.json into forward price slots for window scheduling.

    Returns slots from one slot before ``now`` onward (so the in-progress slot counts),
    sorted by start. Empty on any read/parse failure (caller then can't recommend a defer).
    """
    try:
        import pytz

        path = Path(schedule_path)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            payload = cast("dict[str, Any]", json.load(fh))
        schedule = cast("list[dict[str, Any]]", payload.get("schedule", []) or [])
        tz = pytz.timezone(tz_name)
        out: list[WindowSlot] = []
        for s in schedule:
            start_str = s.get("start_time")
            if not start_str:
                continue
            try:
                start = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
                start = tz.localize(start) if start.tzinfo is None else start
            except (ValueError, TypeError):
                continue
            price = _f(s.get("import_price_sek_kwh"))
            if price is None:
                continue
            out.append(WindowSlot(start_ts=start.timestamp(), import_price_sek_kwh=price))
        out.sort(key=lambda w: w.start_ts)
        # Keep current + future (drop deep past).
        return [w for w in out if w.start_ts >= now_ts - 7200.0]
    except Exception as exc:  # never let a bad schedule break the tick
        logger.warning("Deferrable: failed to load schedule slots: %s", exc)
        return []


class DeferrableApplianceController:
    """Stateful runtime: reads HA + schedule, runs the state machine, publishes, notifies."""

    def __init__(self, cfg: DeferrableRuntimeConfig, state_file: str = _STATE_FILE):
        self.cfg = cfg
        self._state_file = state_file
        self._state: dict[str, AppliancePowerState] = {}
        self._load_state()

    def _load_state(self) -> None:
        try:
            p = Path(self._state_file)
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                for lid, d in raw.items():
                    self._state[lid] = AppliancePowerState(**d)
        except Exception as exc:
            logger.warning("Deferrable: could not load state: %s", exc)

    def _save_state(self) -> None:
        try:
            p = Path(self._state_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({lid: asdict(s) for lid, s in self._state.items()}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Deferrable: could not save state: %s", exc)

    async def _read_on(self, ha: Any, entity: str | None, default: bool = False) -> bool:
        if not entity:
            return default
        v = await ha.get_state_value(entity)
        if v is None:
            return default
        return str(v).lower() in ("on", "true", "1", "home", "open")

    def _deadline_ts(self, cfg: DeferrableApplianceCfg, now_dt: datetime) -> float | None:
        if cfg.deadline_mode == "hard_deadline" and cfg.hard_deadline:
            try:
                hh, mm = (int(x) for x in cfg.hard_deadline.split(":"))
            except ValueError:
                return None
            target = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target <= now_dt:
                target = target + timedelta(days=1)
            return target.timestamp()
        # cheapest_within_hours (soft): window_hours from now
        return now_dt.timestamp() + cfg.window_hours * 3600.0

    async def run(
        self, ha: Any, now_ts: float, now_dt: datetime, *, shadow: bool = False
    ) -> dict[str, Any]:
        """One control cycle (observe-first)."""
        cfg = self.cfg
        if not cfg.enabled or not cfg.appliances:
            return {"enabled": False}

        slots = load_forward_slots(cfg.schedule_path, now_ts, cfg.timezone)
        summaries: list[dict[str, Any]] = []

        for app in cfg.appliances:
            if not app.power_sensor:
                continue
            power = _f(await ha.get_state_value(app.power_sensor)) or 0.0
            override = await self._read_on(ha, app.override_entity)
            switch_on = await self._read_on(ha, app.switch_entity, default=True)

            prev = self._state.get(app.id, AppliancePowerState())
            new_state, event = update_appliance_power_state(prev, power, switch_on, now_ts, app.power)
            self._state[app.id] = new_state

            # Forecast recommendation (duration-aware) for an armed cycle.
            action, window_start = "run", None
            if new_state.pending and not override:
                duration_min = _f(await ha.get_state_value(app.typical_minutes_sensor or "")) or (
                    app.seed_duration_min
                )
                duration_slots = max(1, math.ceil(duration_min / cfg.slot_minutes))
                deadline_ts = self._deadline_ts(app, now_dt)
                action, window_start = recommend_appliance_action(
                    slots, now_ts, duration_slots, deadline_ts
                )

            label = self._label(new_state, action, override)
            await self._publish(ha, app, new_state, label, action, window_start, power, override)
            await self._maybe_notify(ha, app, event, action, window_start, override, shadow)

            summaries.append(
                {"id": app.id, "state": label, "power": round(power, 1),
                 "action": action, "event": event}
            )

        self._save_state()
        logger.info("Deferrable (observe=%s): %s", cfg.observe_only,
                    [(s["id"], s["state"], s["action"]) for s in summaries])
        return {"enabled": True, "observe_only": cfg.observe_only, "appliances": summaries}

    def _label(self, st: AppliancePowerState, action: str, override: bool) -> str:
        if not st.pending:
            return "idle"
        if st.running:
            return "running"
        if override:
            return "armed"
        return "waiting" if action == "defer" else "armed"

    async def _publish(
        self, ha: Any, app: DeferrableApplianceCfg, st: AppliancePowerState,
        label: str, action: str, window_start: float | None, power: float, override: bool,
    ) -> None:
        oid = f"{self.cfg.publish_prefix}{_slug(app.id)}_state"
        rec_start = (
            datetime.fromtimestamp(window_start).astimezone().isoformat()
            if window_start is not None
            else None
        )
        attrs: dict[str, Any] = {
            "friendly_name": f"{app.name} status",
            "icon": "mdi:washing-machine",
            "load_id": app.id,
            "power_w": round(power, 1),
            "recommended_action": action,
            "recommended_start": rec_start,
            "would_defer": bool(st.pending and action == "defer"),
            "override": override,
            "observe_only": self.cfg.observe_only,
            "armed_since": (
                datetime.fromtimestamp(st.start_ts).astimezone().isoformat()
                if st.start_ts is not None
                else None
            ),
        }
        try:
            await ha.set_state(f"sensor.{oid}", label, attrs)
        except Exception as exc:
            logger.warning("Deferrable: publish failed for %s: %s", app.id, exc)

    async def _maybe_notify(
        self, ha: Any, app: DeferrableApplianceCfg, event: str | None,
        action: str, window_start: float | None, override: bool, shadow: bool,
    ) -> None:
        if not self.cfg.notify_service or event is None or shadow:
            return
        title = app.name
        if event == "armed":
            if override:
                msg = "Started (override — running now)"
            elif action == "defer" and window_start is not None:
                when = datetime.fromtimestamp(window_start).astimezone().strftime("%H:%M")
                verb = "would wait" if self.cfg.observe_only else "waiting"
                msg = f"Started — cheaper window at {when}, {verb} (Darkstar)"
            else:
                msg = "Started (good window)"
        elif event == "done":
            msg = "Done"
        else:
            return
        try:
            await ha.send_notification(self.cfg.notify_service, title, msg)
        except Exception as exc:
            logger.warning("Deferrable: notify failed for %s: %s", app.id, exc)
