"""Small, strict parsers for user-edited config values."""

from __future__ import annotations

from typing import Any

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def strict_bool(value: Any, default: bool = False) -> bool:
    """Parse a config flag without Python truthiness.

    ``bool("false")`` is True, so a YAML/UI value saved as the string "false" would
    silently ENABLE a feature. Only real booleans, 0/1 and the usual words count;
    anything else (None, unknown strings, other types) yields ``default``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return default
