"""GET /api/system/db-snapshot: consistent SQLite online backup, streamed then deleted."""

import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.deps import get_learning_store
from backend.api.routers import system


def _make_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")  # as LearningStore does
    conn.execute("CREATE TABLE slot_observations (id INTEGER PRIMARY KEY, kwh REAL)")
    conn.executemany("INSERT INTO slot_observations (kwh) VALUES (?)", [(i,) for i in range(500)])
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "planner_learning.db"
    _make_db(path)
    return path


@pytest.fixture
def client(db_path):
    app = FastAPI()
    app.include_router(system.router)
    app.dependency_overrides[get_learning_store] = lambda: SimpleNamespace(db_path=str(db_path))
    system._snapshot_started = None
    with TestClient(app) as c:
        yield c
    system._snapshot_started = None


def _leftovers(directory):
    return list(directory.glob(f"{system._SNAPSHOT_PREFIX}*"))


def test_snapshot_is_a_consistent_standalone_copy(client, db_path, tmp_path):
    # Uncommitted work by another connection must not appear; committed WAL rows must.
    writer = sqlite3.connect(db_path)
    writer.execute("INSERT INTO slot_observations (kwh) VALUES (999)")
    writer.commit()
    writer.execute("BEGIN IMMEDIATE")  # an open write txn does not block the backup (WAL)
    writer.execute("INSERT INTO slot_observations (kwh) VALUES (-1)")

    resp = client.get("/api/system/db-snapshot")
    writer.rollback()
    writer.close()

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.sqlite3"
    assert 'filename="planner_learning-' in resp.headers["content-disposition"]
    assert int(resp.headers["content-length"]) == len(resp.content)

    out = tmp_path / "downloaded.db"
    out.write_bytes(resp.content)
    conn = sqlite3.connect(out)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT COUNT(*) FROM slot_observations").fetchone()[0] == 501
        assert (
            conn.execute("SELECT COUNT(*) FROM slot_observations WHERE kwh < 0").fetchone()[0] == 0
        )
    finally:
        conn.close()

    # Temp file deleted after streaming, slot released.
    assert _leftovers(db_path.parent) == []
    assert system._snapshot_started is None


def test_concurrent_snapshot_is_refused(client, db_path):
    system._snapshot_started = time.monotonic()
    resp = client.get("/api/system/db-snapshot")
    assert resp.status_code == 409
    assert _leftovers(db_path.parent) == []


def test_stale_busy_marker_does_not_lock_forever(client):
    system._snapshot_started = time.monotonic() - system._SNAPSHOT_STALE_S - 1
    assert client.get("/api/system/db-snapshot").status_code == 200


def test_missing_database_is_404_and_frees_the_slot(client, db_path):
    db_path.unlink()
    for suffix in ("-wal", "-shm"):
        (db_path.parent / (db_path.name + suffix)).unlink(missing_ok=True)
    assert client.get("/api/system/db-snapshot").status_code == 404
    assert system._snapshot_started is None
    assert _leftovers(db_path.parent) == []


def test_backup_failure_is_500_and_cleans_up(client, db_path, monkeypatch):
    def boom(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(system.sqlite3, "connect", boom)
    resp = client.get("/api/system/db-snapshot")
    assert resp.status_code == 500
    assert system._snapshot_started is None
    assert _leftovers(db_path.parent) == []


def test_old_orphaned_snapshot_files_are_swept(client, db_path):
    import os

    orphan = db_path.parent / f"{system._SNAPSHOT_PREFIX}orphan.db"
    orphan.write_bytes(b"x")
    old = time.time() - system._SNAPSHOT_STALE_S - 60
    os.utime(orphan, (old, old))
    fresh = db_path.parent / f"{system._SNAPSHOT_PREFIX}fresh.db"
    fresh.write_bytes(b"x")  # possibly still streaming: keep

    assert client.get("/api/system/db-snapshot").status_code == 200
    assert not orphan.exists()
    assert fresh.exists()
