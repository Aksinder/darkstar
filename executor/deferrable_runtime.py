"""
Deferrable smart-appliance controller — runtime wiring (turnkey, observe-first).

Give Darkstar a ``power_sensor`` (+ the plug's ``switch_entity``) and it does the rest:
auto-arms when a cycle starts (power rises), picks the cheapest FORECAST window for the
WHOLE cycle before the deadline (duration-aware, unlike a "is it cheap right now?"
trigger), detects done, publishes ``sensor.<prefix><id>_state``, and notifies.

Default OFF. Even when enabled it runs **observe-only** by default: it publishes state +
the forecast recommendation and sends notifications without touching the plug, so its
decisions can be verified first. Setting ``observe_only: false`` arms actuation (Fas 3):
the plug is held OFF only for an armed-but-deferred cycle and re-enabled at the cheapest
window (resume-on-power continues the programme); a RUNNING cycle is never interrupted,
an idle plug always stays ON (manual starts keep working), and the override boolean
forces straight through.

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
# Append-only per-cycle ledger (one JSON object per line), written on every "done"
# event so the savings load-shift credit has a durable arm->done record — the
# state file above only holds CURRENT state and wipes start_ts on completion.
_LEDGER_FILE = "data/deferrable_cycles.jsonl"
# Reserved key for chain-tracking metadata inside the state file (not a load id).
_CHAINS_KEY = "_chains"
# The cycle detectors merge measured runs separated by <20 min (cycle_learning's
# fixed merge_gap_minutes=20.0). A silent re-arm whose gap from the PHYSICAL stop
# (~ last_done_ts - done_delay_s) exceeds this is detected as a SEPARATE run, so
# the ledger chain must split too — otherwise the new run inherits the previous
# programme's armed_ts (fabricated shift credit) and its done row erases the
# previous one via the (load_id, armed_ts) dedupe. At the default done_delay_s=300
# the silent-rearm horizon (done_delay_s + rearm_cooldown_s = 1200 s) equals this
# gap, so the split can only trigger when done_delay_s > 300.
_CYCLE_MERGE_GAP_S = 1200.0


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

    def __init__(
        self,
        cfg: DeferrableRuntimeConfig,
        state_file: str = _STATE_FILE,
        ledger_file: str | None = None,
    ):
        self.cfg = cfg
        self._state_file = state_file
        # Default the ledger next to the state file (data/ in production), so a
        # custom state location — e.g. a test tmp dir — keeps both together.
        self._ledger_file = ledger_file or str(
            Path(state_file).parent / Path(_LEDGER_FILE).name
        )
        self._state: dict[str, AppliancePowerState] = {}
        # Per-appliance cycle-chain metadata for the ledger: armed_ts of the FIRST
        # genuine arm (a rearm-cooldown continuation must not move it), whether we
        # ever held the plug, and the deadline anchored to that first arm.
        self._chains: dict[str, dict[str, Any]] = {}
        self._boot_recovered = False
        self._load_state()

    def _load_state(self) -> None:
        try:
            p = Path(self._state_file)
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                chains = raw.pop(_CHAINS_KEY, None)
                if isinstance(chains, dict):
                    self._chains = cast("dict[str, dict[str, Any]]", chains)
                for lid, d in raw.items():
                    self._state[lid] = AppliancePowerState(**d)
        except Exception as exc:
            logger.warning("Deferrable: could not load state: %s", exc)

    def _save_state(self) -> None:
        try:
            p = Path(self._state_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            blob: dict[str, Any] = {lid: asdict(s) for lid, s in self._state.items()}
            if self._chains:
                blob[_CHAINS_KEY] = self._chains
            p.write_text(json.dumps(blob), encoding="utf-8")
        except Exception as exc:
            logger.warning("Deferrable: could not save state: %s", exc)

    def _append_ledger(self, row: dict[str, Any]) -> None:
        """Append one completed-cycle record to the JSONL ledger (never raises)."""
        try:
            p = Path(self._ledger_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.warning(
                "Deferrable: cycle-ledger append failed for %s: %s", row.get("load_id"), exc
            )

    async def _read_on(self, ha: Any, entity: str | None, default: bool = False) -> bool:
        if not entity:
            return default
        v = await ha.get_state_value(entity)
        if v is None:
            return default
        return str(v).lower() in ("on", "true", "1", "home", "open")

    def _deadline_ts(
        self, cfg: DeferrableApplianceCfg, now_dt: datetime, start_ts: float | None
    ) -> float | None:
        """Deadline anchored to when the cycle ARMED, not to the current tick.

        A now-anchored deadline is a rolling horizon that never closes (a cheaper
        block always exists somewhere in the next N hours => hold forever) and a
        missed hard deadline would re-roll +24h each crossing. Anchoring to
        start_ts bounds every hold: once the anchored deadline passes,
        recommend_appliance_action fails open to "run".
        """
        anchor_dt = (
            datetime.fromtimestamp(start_ts, tz=now_dt.tzinfo)
            if start_ts is not None
            else now_dt
        )
        if cfg.deadline_mode == "hard_deadline" and cfg.hard_deadline:
            try:
                hh, mm = (int(x) for x in cfg.hard_deadline.split(":"))
            except ValueError:
                return None
            target = anchor_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target <= anchor_dt:
                target = target + timedelta(days=1)
            return target.timestamp()
        # cheapest_within_hours (soft): window_hours from ARMING
        return anchor_dt.timestamp() + cfg.window_hours * 3600.0

    async def run(
        self, ha: Any, now_ts: float, now_dt: datetime, *, shadow: bool = False
    ) -> dict[str, Any]:
        """One control cycle (observe-first)."""
        cfg = self.cfg
        if not cfg.enabled or not cfg.appliances:
            return {"enabled": False}

        # Boot recovery: if the persisted state was lost (an add-on update wipes
        # /app/data) while a plug was held OFF, no state entry exists and the cycle
        # can never re-arm through a dead plug — stranded until a human notices. On
        # the first actuating tick, power ON any configured plug that reads OFF and
        # has NO state entry (a live hold always has one). Trade-off: a deliberately
        # off idle plug is re-enabled once per add-on start — harmless for an idle
        # machine, and we notify.
        if not self._boot_recovered and not cfg.observe_only and not shadow:
            self._boot_recovered = True
            for app in cfg.appliances:
                if not app.switch_entity or app.id in self._state:
                    continue
                sw = await ha.get_state_value(app.switch_entity)
                if sw is not None and str(sw).lower() == "off":
                    try:
                        await ha.set_switch(app.switch_entity, True)
                        logger.warning(
                            "Deferrable: boot recovery — re-enabled %s (no persisted hold state)",
                            app.switch_entity,
                        )
                        if cfg.notify_service:
                            await ha.send_notification(
                                cfg.notify_service,
                                app.name,
                                "Plug re-enabled after restart (hold state lost)",
                            )
                    except Exception as exc:
                        logger.warning(
                            "Deferrable: boot recovery failed for %s: %s", app.id, exc
                        )

        slots = load_forward_slots(cfg.schedule_path, now_ts, cfg.timezone)
        summaries: list[dict[str, Any]] = []

        for app in cfg.appliances:
            if not app.power_sensor:
                continue
            power_raw = await ha.get_state_value(app.power_sensor)
            power_readable = power_raw is not None and str(power_raw) not in (
                "unknown",
                "unavailable",
                "",
            )
            power = _f(power_raw) or 0.0
            override = await self._read_on(ha, app.override_entity)
            sw_raw = (
                await ha.get_state_value(app.switch_entity) if app.switch_entity else None
            )
            switch_readable = sw_raw is not None and str(sw_raw) not in (
                "unknown",
                "unavailable",
            )
            switch_on = (
                str(sw_raw).lower() in ("on", "true", "1") if switch_readable else True
            )

            prev = self._state.get(app.id, AppliancePowerState())

            # FREEZE on unreadable sensors: a device that dropped off the network (its
            # power sensor and plug flap together) must not advance the state machine —
            # power 'unavailable'->0.0 with an hours-old below_since would fire a false
            # "done", drop pending, and strand a held plug OFF forever (the VVB
            # stuck-switch incident, inverted). Publish the last state as stale, touch
            # nothing, and resume when readings return.
            frozen = not power_readable or (bool(app.switch_entity) and not switch_readable)
            if frozen:
                label = self._label(prev, "run", override) + " (stale)"
                await self._publish(ha, app, prev, label, "run", None, power, override, None)
                summaries.append(
                    {"id": app.id, "state": label, "power": None,
                     "action": "frozen", "event": None, "plug": None}
                )
                continue

            new_state, event = update_appliance_power_state(prev, power, switch_on, now_ts, app.power)
            new_state.held_by_us = prev.held_by_us and not switch_on
            self._state[app.id] = new_state

            # Cycle-chain tracking for the ledger. A genuine "armed" event starts a
            # new chain; a silent re-arm within rearm_cooldown_s (continuation —
            # soak pause, resume after re-power) keeps the FIRST armed_ts so the
            # ledger row reflects the human's original start press. Deadline is
            # anchored to that first arm, mirroring _deadline_ts semantics.
            if event == "armed":
                self._chains[app.id] = {
                    "armed_ts": new_state.start_ts,
                    "held_ever": bool(new_state.held_by_us),
                    "deadline_ts": self._deadline_ts(app, now_dt, new_state.start_ts),
                }
            elif new_state.pending and not prev.pending:
                # Silent continuation re-arm. Keep the first-press chain ONLY
                # while cycle detection will merge the runs into one; past the
                # merge gap the runs split, and inheriting the old armed_ts would
                # price a distinct (never-deferred) programme against the previous
                # arm anchor. Also fresh when chain state was lost (restart/wipe):
                # no original press is known — record the resume honestly rather
                # than inventing an earlier timestamp.
                runs_will_split = (
                    prev.last_done_ts is not None
                    and (now_ts - (prev.last_done_ts - app.power.done_delay_s)) > _CYCLE_MERGE_GAP_S
                )
                if app.id not in self._chains or runs_will_split:
                    self._chains[app.id] = {
                        "armed_ts": new_state.start_ts,
                        "held_ever": bool(new_state.held_by_us),
                        "deadline_ts": self._deadline_ts(app, now_dt, new_state.start_ts),
                    }

            # Forecast recommendation (duration-aware) for an armed cycle.
            action, window_start = "run", None
            if new_state.pending and not override:
                duration_min = _f(await ha.get_state_value(app.typical_minutes_sensor or "")) or (
                    app.seed_duration_min
                )
                duration_slots = max(1, math.ceil(duration_min / cfg.slot_minutes))
                deadline_ts = self._deadline_ts(app, now_dt, new_state.start_ts)
                action, window_start = recommend_appliance_action(
                    slots, now_ts, duration_slots, deadline_ts
                )

            # Fas 3 — plug actuation (only outside observe/shadow, only on a readable
            # switch). Semantics:
            #  - PAUSE only at genuine start detection (the "armed" event — a re-arm
            #    within rearm_cooldown_s is a continuation and emits none) or while WE
            #    are already holding the plug OFF. Never mid-cycle.
            #  - RESUME (ON) only a hold WE own, when the window arrives / the anchored
            #    deadline forces / override demands. A plug the USER cut (held_by_us
            #    False) is never re-energized — manual off wins, pending or not.
            #  - IDLE / DONE: hands off entirely.
            plug_cmd: str | None = None
            if (
                not cfg.observe_only
                and not shadow
                and app.switch_entity
                and switch_readable
            ):
                desired_on: bool | None = None
                if new_state.pending:
                    if (
                        action == "defer"
                        and not override
                        and (event == "armed" or new_state.held_by_us)
                    ):
                        desired_on = False
                    elif switch_on or new_state.held_by_us:
                        # run/override: keep a powered plug on, or release OUR hold.
                        desired_on = True
                    # else: user-cut plug (off, not ours) — hands off.
                if desired_on is not None and desired_on != switch_on:
                    try:
                        ok = await ha.set_switch(app.switch_entity, desired_on)
                        plug_cmd = ("on" if desired_on else "off") if ok else "failed"
                        if ok:
                            # Ownership only — switch_was_on is deliberately NOT set
                            # here: the next tick's real switch read must see the
                            # OFF->ON transition so the state machine grants its
                            # re-power grace (below_since reset).
                            new_state.held_by_us = not desired_on
                    except Exception as exc:
                        logger.warning(
                            "Deferrable: plug actuation failed for %s: %s", app.id, exc
                        )
                        plug_cmd = "failed"

            chain = self._chains.get(app.id)
            if chain is not None and new_state.held_by_us:
                chain["held_ever"] = True

            if event == "done":
                # armed_ts comes from the chain (FIRST arm of the programme) with
                # prev.start_ts as fallback — read BEFORE the state machine cleared
                # it on the "done" transition. The chain is deliberately kept: a
                # rearm-cooldown continuation that completes later appends another
                # row with the SAME (load_id, armed_ts), and the ledger reader
                # collapses those to the final done. Only the next genuine "armed"
                # starts a fresh chain.
                self._append_ledger(
                    {
                        "load_id": app.id,
                        "armed_ts": (chain or {}).get("armed_ts") or prev.start_ts,
                        "done_ts": now_ts,
                        "held_by_us_ever": bool((chain or {}).get("held_ever", False)),
                        "deadline_ts": (chain or {}).get("deadline_ts"),
                        # Not cheaply available here (only instantaneous W reads);
                        # the publisher back-fills measured_kwh + run window from
                        # cycle detection at the first publish after done
                        # (savings_loadshift.enrich_cycle_ledger), so the row
                        # stays priceable beyond the detection history horizon.
                        "measured_kwh": None,
                    }
                )

            label = self._label(new_state, action, override)
            await self._publish(
                ha, app, new_state, label, action, window_start, power, override, plug_cmd
            )
            await self._maybe_notify(ha, app, event, action, window_start, override, shadow)

            summaries.append(
                {"id": app.id, "state": label, "power": round(power, 1),
                 "action": action, "event": event, "plug": plug_cmd}
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
        plug_cmd: str | None = None,
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
            "plug_commanded": plug_cmd,
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
