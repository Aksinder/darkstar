import copy
import logging
import threading
from pathlib import Path
from typing import Any, cast

import yaml

logger = logging.getLogger("darkstar.core.secrets")

# Parsed secrets.yaml, keyed by absolute path and validated by (mtime_ns, size).
# Every HA call reads the URL/token through here; re-parsing YAML each time was a
# measurable cost on the 2-vCPU box. A rewritten file (add-on options sync, UI
# save) changes mtime/size and is re-read on the next call.
_secrets_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_secrets_cache_lock = threading.Lock()


def _load_secrets_file(path: str = "secrets.yaml") -> dict[str, Any]:
    """Return the parsed secrets file (a deep copy; callers may mutate it).

    Missing file -> {}. A parse error is logged and returns {} without caching,
    so a fixed file is picked up on the next call.
    """
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warning: Could not load {path}: {exc}")
        return {}

    key = str(p.resolve())
    stamp = (st.st_mtime_ns, st.st_size)
    with _secrets_cache_lock:
        cached = _secrets_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return copy.deepcopy(cached[1])

    try:
        with p.open() as file:
            raw_data: Any = yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warning: Could not load {path}: {exc}")
        return {}

    parsed: dict[str, Any] = cast("dict[str, Any]", raw_data) if isinstance(raw_data, dict) else {}
    with _secrets_cache_lock:
        _secrets_cache[key] = (stamp, parsed)
    return copy.deepcopy(parsed)


def clear_secrets_cache() -> None:
    with _secrets_cache_lock:
        _secrets_cache.clear()


def load_home_assistant_config() -> dict[str, Any]:
    """Read Home Assistant configuration from secrets.yaml."""
    secrets = _load_secrets_file()
    ha_config: Any = secrets.get("home_assistant")
    if not isinstance(ha_config, dict):
        return {}
    return cast("dict[str, Any]", ha_config)


def load_notifications_config() -> dict[str, Any]:
    """Read notification secrets (e.g., Discord webhook) from secrets.yaml."""
    secrets = _load_secrets_file()
    notif_secrets: Any = secrets.get("notifications")
    if not isinstance(notif_secrets, dict):
        return {}
    return cast("dict[str, Any]", notif_secrets)


def load_yaml(path: str) -> dict[str, Any]:
    try:
        with Path(path).open() as f:
            raw_data: Any = yaml.safe_load(f)
            return cast("dict[str, Any]", raw_data) if isinstance(raw_data, dict) else {}
    except FileNotFoundError:
        return {}
