"""Atomic JSON writer + /api/schedule/save uses it (the executor once read torn JSON)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.atomic_io import atomic_write_json


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "schedule.json"
    target.write_text('{"old": true}', encoding="utf-8")
    with patch("utils.atomic_io.os.fsync", wraps=os.fsync) as fsync:
        atomic_write_json(target, {"new": 1}, indent=2)
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": 1}
    assert fsync.called
    assert sorted(p.name for p in tmp_path.iterdir()) == ["schedule.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path: Path) -> None:
    target = tmp_path / "schedule.json"
    target.write_text('{"old": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})  # not serializable mid-dump
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["schedule.json"]


def test_schedule_save_endpoint_writes_atomically(tmp_path: Path, monkeypatch) -> None:
    from backend.api.routers import schedule as sched_router

    (tmp_path / "data").mkdir()
    path = tmp_path / "data" / "schedule.json"
    path.write_text(
        json.dumps({"schedule": [{"start_time": "t0", "water_kw": 0}], "meta": {}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls: list[Path] = []

    def spy(p, data, **kw):
        calls.append(Path(p))
        atomic_write_json(p, data, **kw)

    with patch.object(sched_router, "atomic_write_json", spy):
        out = asyncio.run(
            sched_router.save_schedule({"overrides": [{"start_time": "t0", "water_kw": 3}]})
        )
    assert out["status"] == "success", out
    assert calls == [Path("data/schedule.json")]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schedule"][0]["water_kw"] == 3
    assert "last_manual_override" in saved["meta"]
