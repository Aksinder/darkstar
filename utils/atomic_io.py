"""Atomic file writes for files another process reads while we write them.

The executor reads data/schedule.json every tick; a truncating ``open("w")``
rewrite lets it catch half a JSON document (live 2026-08-26: "Failed to load
schedule: Expecting value"). Write a unique sibling temp file, fsync it, then
``os.replace`` over the target — a reader sees either the old or the new file.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, data: Any, **dump_kwargs: Any) -> None:
    """Serialize ``data`` as JSON to ``path`` atomically (tmp + fsync + rename).

    The temp file lives in the target's directory (same filesystem, so the rename
    is atomic) under a unique name, so two concurrent writers never share one.
    On any failure the temp file is removed and the target is left untouched.
    """
    target = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent) or ".", prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise
