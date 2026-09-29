"""A short, durable record of WHY the battery changed what it was doing.

2026-09-25 03:37 the executor flipped from idle (holding 37 % for the morning
peak) to self-consumption, and the battery was spent on the cheapest hour of the
day. By the time anybody asked, the add-on log had rotated (it holds ~2 hours)
and the plan that made the decision lived in a database inside the container.
The question "which plan said so, and what did the one before it say?" could
not be answered from outside the house.

This journal answers it. One JSON line per change of mind, written next to
config.yaml — a directory that is readable over the Home Assistant file API:

    {"ts": ..., "from": "idle", "to": "self_consumption", "source": "plan",
     "reason": ..., "soc": 37.7, "slot_start": ..., "planned_at": ...,
     "slot": {...}, "prev_slot": {...}, "prev_planned_at": ..., "ahead": [...]}

``prev_slot``/``prev_planned_at`` are what the SAME decision looked like on the
tick before, which is exactly the comparison the incident needed; ``ahead`` is
the battery's next couple of hours in the new plan.

A line is written when the mode or its source changes, or when a NEW plan moves
the current slot's SoC target by ``TARGET_SHIFT_PCT`` or more without changing
the mode. Steady state writes nothing. The journal never raises: it is a
convenience, and a full disk must not cost a control tick.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TARGET_SHIFT_PCT = 5
DEFAULT_KEEP = 300
_TRIM_SLACK = 50  # rewrite the file only when it has grown this far past `keep`


class DecisionJournal:
    """Append-only, bounded, best-effort."""

    def __init__(self, path: str | Path | None, keep: int = DEFAULT_KEEP):
        self.path = Path(path) if path else None
        self.keep = max(1, int(keep))
        self._last: dict[str, Any] | None = None
        self._lines: int | None = None  # unknown until the file has been counted

    @property
    def enabled(self) -> bool:
        """Only where the directory already exists: a development checkout or a test
        has no /config/darkstar, and must not grow one."""
        return self.path is not None and self.path.parent.is_dir()

    def observe(self, entry: dict[str, Any]) -> bool:
        """Offer this tick's decision. Returns True when a line was written.

        ``entry`` needs ``mode`` and ``source``; ``planned_at`` and ``slot`` (with
        ``soc_target``) make the replan comparison possible. Everything else is
        carried through as given.
        """
        try:
            return self._observe(entry)
        except Exception as e:  # the journal must never cost a tick
            logger.debug("Decision journal skipped an entry: %s", e)
            return False

    def _observe(self, entry: dict[str, Any]) -> bool:
        prev = self._last
        self._last = entry
        if prev is None:
            # First tick after a start: remember, do not report. A restart is not a
            # change of mind, and the line would carry no "from".
            return False

        changed = entry.get("mode") != prev.get("mode") or entry.get("source") != prev.get(
            "source"
        )
        shifted = False
        if not changed and entry.get("planned_at") != prev.get("planned_at"):
            new_t = _soc_target(entry)
            old_t = _soc_target(prev)
            same_slot = entry.get("slot_start") == prev.get("slot_start")
            shifted = (
                same_slot
                and new_t is not None
                and old_t is not None
                and abs(new_t - old_t) >= TARGET_SHIFT_PCT
            )
        if not (changed or shifted):
            return False
        if not self.enabled:
            return False

        line = {
            "ts": entry.get("ts"),
            "kind": "mode" if changed else "target",
            "from": prev.get("mode"),
            "to": entry.get("mode"),
            "source": entry.get("source"),
            "prev_source": prev.get("source"),
            "reason": entry.get("reason"),
            "prev_reason": prev.get("reason"),
            "soc": entry.get("soc"),
            "slot_start": entry.get("slot_start"),
            "prev_slot_start": prev.get("slot_start"),
            "planned_at": entry.get("planned_at"),
            "prev_planned_at": prev.get("planned_at"),
            "slot": entry.get("slot"),
            "prev_slot": prev.get("slot"),
            "ahead": entry.get("ahead"),
        }
        for key in ("import_price", "ev_isolation", "note"):
            if entry.get(key) is not None:
                line[key] = entry[key]
        self._append(line)
        return True

    def _append(self, line: dict[str, Any]) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        if self._lines is None:
            with self.path.open(encoding="utf-8") as fh:
                self._lines = sum(1 for _ in fh)
        else:
            self._lines += 1
        if self._lines > self.keep + _TRIM_SLACK:
            self._trim()

    def _trim(self) -> None:
        assert self.path is not None
        with self.path.open(encoding="utf-8") as fh:
            tail = fh.readlines()[-self.keep :]
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.writelines(tail)
            tmp.replace(self.path)
            self._lines = len(tail)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise


def _soc_target(entry: dict[str, Any]) -> float | None:
    slot = entry.get("slot")
    if not isinstance(slot, dict):
        return None
    value = slot.get("soc_target")  # type: ignore[reportUnknownMemberType]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
