"""Daily EV-priority confirmation — a manual override must not quietly become policy.

Owner directive (2026-08-27): the ``input_select.darkstar_ev_priority`` helper
is a per-situation lever ("tesla_first tonight"), not a standing setting. Left
alone it silently keeps re-weighting every plan. So: once a day, at a suitable
time, ask the owner whether to keep the non-auto priority via an actionable
notification. A tap on "Behåll" keeps it; no answer within the timeout reverts
the select to ``auto``.

Mechanics
---------
* The ask fires at ``ask_time`` local wall-clock (default 20:00 — evening,
  before the night plan is decided), only when the select is not ``auto``.
* The tap arrives as a ``mobile_app_notification_action`` event on the HA
  websocket Darkstar already maintains (backend/ha_socket.py fans it out here).
* State survives restarts via a small JSON file, so a reboot inside the
  60-minute window neither loses the ask nor re-sends it.
* Reverting is a plain ``input_select.select_option`` service call — the same
  writeback channel the FMB SoC estimator already uses. The revert is announced
  with a second (non-actionable) notification.

All wall-clock math follows the repo's canonical pattern: naive local
candidates localized per-day (DST-safe), never arithmetic on aware datetimes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import requests

from backend.core import secrets

logger = logging.getLogger("darkstar.ev_priority_attendant")

KEEP_ACTION = "DARKSTAR_KEEP_EV_PRIORITY"
_STATE_PATH = Path("data/ev_priority_ask.json")


class EVPriorityAttendant:
    def __init__(self, config: dict[str, Any]):
        surplus = (config.get("executor", {}) or {}).get("ev_surplus", {}) or {}
        attendant = surplus.get("priority_attendant", {}) or {}
        self.priority_entity: str | None = surplus.get("priority_entity")
        self.enabled: bool = bool(attendant.get("enabled", True)) and bool(
            self.priority_entity
        )
        self.ask_time: str = str(attendant.get("ask_time", "20:00"))
        self.timeout_minutes: float = float(attendant.get("timeout_minutes", 60))
        # Resolution order covers every place a notify service actually lives in
        # this config (review: the first draft read a path that does not exist):
        # attendant-specific -> ev_surplus.notify_service -> executor.notifications
        # -> top-level notifications.
        self.notify_service: str | None = (
            attendant.get("notify_service")
            or surplus.get("notify_service")
            or ((config.get("executor", {}) or {}).get("notifications", {}) or {}).get(
                "service"
            )
            or (config.get("notifications", {}) or {}).get("service")
        )
        self.tz = pytz.timezone(str(config.get("timezone", "Europe/Stockholm")))
        self._state: dict[str, Any] = self._load_state()
        # Set from the HA-websocket THREAD (ha_socket runs its own loop in a
        # daemon thread); tick() folds it into state and re-checks it right
        # before any revert. A bare bool store is atomic under the GIL — the
        # callback must never touch the state dict or the file from that thread.
        self._kept_flag: bool = False

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict[str, Any]:
        try:
            if _STATE_PATH.exists():
                return json.loads(_STATE_PATH.read_text())
        except Exception:
            logger.warning("ev_priority_ask.json unreadable — starting fresh")
        return {}

    def _save_state(self) -> None:
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._state))
            tmp.replace(_STATE_PATH)
        except Exception:
            logger.exception("could not persist ev_priority_ask state")

    # ------------------------------------------------------------------ HA I/O

    def _ha(self) -> tuple[str, str] | None:
        cfg = secrets.load_home_assistant_config()
        url, token = cfg.get("url"), cfg.get("token")
        if not url or not token:
            return None
        return str(url).rstrip("/"), str(token)

    def _post(self, path: str, payload: dict[str, Any]) -> bool:
        ha = self._ha()
        if not ha:
            return False
        url, token = ha
        try:
            r = requests.post(
                f"{url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=10,
            )
            return r.status_code < 300
        except Exception as e:
            logger.warning("HA POST %s failed: %s", path, e)
            return False

    def _get_select_state(self) -> str | None:
        ha = self._ha()
        if not ha or not self.priority_entity:
            return None
        url, token = ha
        try:
            r = requests.get(
                f"{url}/api/states/{self.priority_entity}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if r.status_code == 200:
                return str(r.json().get("state", "")).lower()
        except Exception as e:
            logger.warning("HA GET %s failed: %s", self.priority_entity, e)
        return None

    def _notify(self, title: str, message: str, actionable: bool) -> bool:
        if not self.notify_service or "." not in self.notify_service:
            logger.warning("no notify service configured — cannot ask about EV priority")
            return False
        domain, service = self.notify_service.split(".", 1)
        data: dict[str, Any] = {"tag": "darkstar_ev_priority"}
        if actionable:
            data["actions"] = [{"action": KEEP_ACTION, "title": "Behåll prioriteten"}]
        return self._post(
            f"/api/services/{domain}/{service}",
            {"title": title, "message": message, "data": data},
        )

    # ------------------------------------------------------------------ events

    def on_notification_action(self, action: str) -> None:
        """Called from the HA websocket THREAD for every mobile notification tap.

        Cross-thread: only sets an atomic flag. tick() (attendant loop) folds it
        into the persisted state and re-checks it immediately before reverting,
        so a tap landing during the revert tick still wins.
        """
        if action == KEEP_ACTION:
            self._kept_flag = True
            logger.info("EV priority keep-tap received")

    # ------------------------------------------------------------------ ticks

    def _today_ask_ts(self, now_ts: float) -> float | None:
        """Epoch of TODAY's ask_time, DST-safe: build the naive local candidate
        for today's date and localize it (never next-day-minus-86400 — review
        proved that skips the whole day before spring-forward and fires an hour
        late before fall-back)."""
        try:
            hh, mm = (int(x) for x in self.ask_time.split(":", 1))
        except ValueError:
            logger.warning("invalid ask_time %r — attendant idle", self.ask_time)
            return None
        base_naive = datetime.fromtimestamp(now_ts, self.tz).replace(tzinfo=None)
        cand_naive = base_naive.replace(hour=hh, minute=mm, second=0, microsecond=0)
        try:
            return self.tz.localize(cand_naive).timestamp()
        except Exception:
            # Nonexistent/ambiguous wall-clock minute on a DST day: shift an hour.
            return self.tz.localize(cand_naive + timedelta(hours=1)).timestamp()

    # How long past ask_time we keep retrying a failed ask/read. Wide enough to
    # ride out an HA restart at exactly 20:00 (review: a 120 s window burned the
    # whole day on one failed read).
    _ASK_WINDOW_S = 900.0
    # A pending revert that cannot be delivered (HA down) is retried each tick;
    # after this long we give up for the day and log — the priority then stands
    # until tomorrow's ask, which is the honest outcome of an unreachable HA.
    _REVERT_GIVE_UP_S = 6 * 3600.0

    def tick_sync(self, now_ts: float) -> None:
        if not self.enabled:
            return

        # Fold in a tap from the websocket thread (atomic flag -> persisted state).
        if self._kept_flag and self._state.get("asked_at") and not self._state.get("kept"):
            self._state["kept"] = True
            self._save_state()
            logger.info("EV priority kept by owner tap (%s)", self._state.get("mode"))

        asked_at = self._state.get("asked_at")

        # Phase 1: is it time to ask?
        if not asked_at:
            today = datetime.fromtimestamp(now_ts, self.tz).strftime("%Y-%m-%d")
            if self._state.get("ask_day") == today:
                return
            ask_ts = self._today_ask_ts(now_ts)
            if ask_ts is None or not (0.0 <= now_ts - ask_ts < self._ASK_WINDOW_S):
                return
            mode = self._get_select_state()
            if mode is None:
                return  # HA unreachable — retry next tick inside the window
            if mode == "auto":
                self._state = {"ask_day": today}
                self._save_state()
                return
            self._kept_flag = False
            if self._notify(
                "EV-prioritet",
                f"EV-prioriteten står på '{mode}'. Behålla den? "
                f"Utan svar inom {int(self.timeout_minutes)} min återgår den "
                "till auto.",
                actionable=True,
            ):
                self._state = {
                    "asked_at": now_ts,
                    "mode": mode,
                    "kept": False,
                    "ask_day": today,
                }
                self._save_state()
                logger.info("asked owner about EV priority '%s'", mode)
            return

        # Phase 2: an ask is pending — resolve it.
        if self._state.get("kept"):
            self._state = {"ask_day": self._state.get("ask_day")}
            self._save_state()
            return
        overdue = now_ts - float(asked_at) - self.timeout_minutes * 60.0
        if overdue < 0.0:
            return
        if overdue > self._REVERT_GIVE_UP_S:
            logger.warning(
                "EV priority revert undeliverable for %.0f h — giving up until "
                "tomorrow's ask (priority '%s' stands)",
                overdue / 3600.0,
                self._state.get("mode"),
            )
            self._state = {"ask_day": self._state.get("ask_day")}
            self._save_state()
            return
        mode_now = self._get_select_state()
        if mode_now is None:
            return  # HA unreachable — keep the pending ask, retry next tick
        if mode_now != self._state.get("mode"):
            # The owner changed the select themselves during the window —
            # their explicit action supersedes the ask.
            logger.info("EV priority changed during ask window — leaving it alone")
            self._state = {"ask_day": self._state.get("ask_day")}
            self._save_state()
            return
        # Last-instant tap check: the flag is set from another thread and may
        # have landed after the fold at the top of this tick.
        if self._kept_flag:
            return
        if not self.priority_entity:
            return
        ok = self._post(
            "/api/services/input_select/select_option",
            {"entity_id": self.priority_entity, "option": "auto"},
        )
        if not ok:
            return  # HA POST failed — keep the pending ask, retry next tick
        self._notify(
            "EV-prioritet",
            f"Ingen bekräftelse — prioriteten '{self._state.get('mode')}' "
            "är återställd till auto.",
            actionable=False,
        )
        logger.info(
            "EV priority '%s' reverted to auto (no confirmation)",
            self._state.get("mode"),
        )
        self._state = {"ask_day": self._state.get("ask_day")}
        self._save_state()


async def run_ev_priority_attendant_loop(config: dict[str, Any]) -> None:
    attendant = EVPriorityAttendant(config)
    if not attendant.enabled:
        logger.info("EV priority attendant disabled (no priority_entity or turned off)")
        return
    try:
        from backend.ha_socket import register_notification_action_callback

        register_notification_action_callback(attendant.on_notification_action)
    except Exception:
        logger.exception("could not register notification-action callback")
    logger.info(
        "EV priority attendant running: ask %s, timeout %.0f min, notify %s",
        attendant.ask_time,
        attendant.timeout_minutes,
        attendant.notify_service,
    )
    while True:
        try:
            # tick_sync does blocking HTTP (requests); to_thread keeps the shared
            # backend event loop responsive even when HA is slow or down.
            await asyncio.to_thread(attendant.tick_sync, time.time())
        except Exception:
            logger.exception("EV priority attendant tick failed")
        await asyncio.sleep(60)
