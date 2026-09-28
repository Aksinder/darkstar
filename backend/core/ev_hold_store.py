"""Last-good EV readings that survive a restart.

The planner input path (backend/core/ha_client.py) and the surplus servo
(executor/ev_surplus_runtime.py) both hold the last READABLE plug / presence /
SoC answer for 24 h when a car's entities go blank — the Tesla Fleet integration
reports every entity ``unknown`` after a Home Assistant restart until the car
next comes online. Those holds lived in memory only.

2026-09-27: HA restarted at 19:47 with the car asleep, so its entities went
``unknown``; the holds carried it (plugged, home, 19 %). At 03:30 the nightly
install restarted the add-on, the holds were gone, and from then on the servo
read ``unknown`` -> "nothing to hold (never read) -> False": a car sitting in its
cable was "not plugged", got no start command, and — because the wake button is
only pressed after a FAILED start — was never woken. It left at 07:38 with 19 %.

This module is the disk behind those dicts, nothing more: the owners keep their
own dicts and their own expiry rules, and call ``save`` after a fresh reading.

Off by default. ``enable`` is called once from the application's startup; tests
and tools that never call it get the old in-memory behaviour and write nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# namespace -> kind -> key -> (value, epoch seconds of the reading)
Holds = dict[str, dict[str, tuple[Any, float]]]

# A reading that only got NEWER (same value, later timestamp) is written at most
# this often. The timestamp on disk may therefore lag the truth by this much —
# noise against holds measured in hours — while a changed VALUE is written at once.
SAVE_MIN_INTERVAL_S = 300.0

_lock = threading.Lock()
_path: Path | None = None
_last_saved_values: dict[str, dict[str, dict[str, Any]]] = {}
_last_saved_ts: dict[str, float] = {}


def enable(path: str | Path) -> None:
    """Turn persistence on, reading and writing ``path``."""
    global _path
    with _lock:
        _path = Path(path)
        _last_saved_values.clear()
        _last_saved_ts.clear()
    logger.info("EV holds: persisting last-good readings to %s", path)


def disable() -> None:
    """Back to memory-only (tests)."""
    global _path
    with _lock:
        _path = None
        _last_saved_values.clear()
        _last_saved_ts.clear()


def is_enabled() -> bool:
    return _path is not None


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        logger.warning("EV holds: %s unreadable (%s) — starting with nothing held", path, e)
        return {}
    return cast("dict[str, Any]", data) if isinstance(data, dict) else {}


def load(namespace: str) -> Holds:
    """The holds persisted for ``namespace``; empty when off, absent or unreadable.

    Malformed entries are dropped one by one rather than poisoning the rest: a
    hold is only ever a convenience, never worth failing a startup for.
    """
    with _lock:
        path = _path
        if path is None:
            return {}
        raw = _read_file(path).get(namespace)
    out: Holds = {}
    if not isinstance(raw, dict):
        return out
    for kind, entries in cast("dict[str, Any]", raw).items():
        if not isinstance(entries, dict):
            continue
        kept: dict[str, tuple[Any, float]] = {}
        for key, pair in cast("dict[str, Any]", entries).items():
            if not isinstance(pair, list | tuple):
                continue
            items: list[Any] = list(cast("Any", pair))
            if len(items) != 2 or isinstance(items[1], bool):
                continue
            if not isinstance(items[1], int | float):
                continue
            kept[str(key)] = (items[0], float(items[1]))
        if kept:
            out[str(kind)] = kept
    return out


def save(namespace: str, holds: Holds, now: float, *, force: bool = False) -> bool:
    """Persist ``holds`` for ``namespace``. Returns True when the file was written.

    Written when a value differs from what is on disk, when the last write is
    older than SAVE_MIN_INTERVAL_S, or on ``force``. Other namespaces in the file
    are preserved. A failed write is logged and swallowed.
    """
    with _lock:
        path = _path
        if path is None:
            return False
        values = {
            kind: {key: pair[0] for key, pair in entries.items()}
            for kind, entries in holds.items()
        }
        changed = values != _last_saved_values.get(namespace)
        due = (now - _last_saved_ts.get(namespace, float("-inf"))) >= SAVE_MIN_INTERVAL_S
        if not (force or changed or due):
            return False
        data = _read_file(path)
        data[namespace] = {
            kind: {key: [pair[0], pair[1]] for key, pair in entries.items()}
            for kind, entries in holds.items()
        }
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
            tmp.replace(path)  # atomic: a crash mid-write leaves the old file whole
        except OSError as e:
            logger.warning("EV holds: failed to persist to %s: %s", path, e)
            with contextlib.suppress(OSError):
                tmp.unlink()
            return False
        _last_saved_values[namespace] = values
        _last_saved_ts[namespace] = now
        return True
