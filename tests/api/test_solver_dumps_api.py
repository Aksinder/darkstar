"""Read-only solver-dump endpoints (GET /api/debug/solver_dumps[/{name}]).

They exist so real solver instances can be pulled off the box through the add-on's
ingress and replayed offline. The fetch endpoint takes a file name from the URL,
so it must never serve anything but a dump the solver wrote into its own directory.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import create_app

    app = create_app()
    fastapi_app = app.other_asgi_app if hasattr(app, "other_asgi_app") else app
    with (
        patch("backend.main.LearningStore", return_value=MagicMock(close=AsyncMock())),
        TestClient(fastapi_app) as client,
    ):
        yield client


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    d = tmp_path / "solver_dumps"
    d.mkdir()
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(d))
    return d


def test_list_is_empty_without_a_dump_directory(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(tmp_path / "missing"))
    response = client.get("/api/debug/solver_dumps")
    assert response.status_code == 200
    assert response.json() == {"dumps": [], "count": 0}


def test_list_returns_dumps_newest_first_and_nothing_else(client, dump_dir):
    (dump_dir / "kepler_fail_20260810T082200.json").write_text("{}")
    (dump_dir / "kepler_ok_20261112T040000.json").write_text('{"reason": "ok"}')
    (dump_dir / "kepler_ok_20261111T230000.json").write_text("{}")
    (dump_dir / "notes.txt").write_text("not a dump")
    (dump_dir / "kepler_ok_20261112T040000.json.bak").write_text("{}")

    body = client.get("/api/debug/solver_dumps").json()

    assert [d["name"] for d in body["dumps"]] == [
        "kepler_ok_20261112T040000.json",
        "kepler_ok_20261111T230000.json",
        "kepler_fail_20260810T082200.json",
    ]
    assert body["count"] == 3
    assert body["dumps"][0]["size_bytes"] == len('{"reason": "ok"}')


def test_fetch_returns_the_dump_verbatim(client, dump_dir):
    payload = {"reason": "ok", "input": {"slots": []}, "config": {"capacity_kwh": 16.0}}
    (dump_dir / "kepler_ok_20261112T040000.json").write_text(json.dumps(payload))

    response = client.get("/api/debug/solver_dumps/kepler_ok_20261112T040000.json")

    assert response.status_code == 200
    assert response.json() == payload


def test_fetch_unknown_dump_is_404(client, dump_dir):
    assert client.get("/api/debug/solver_dumps/kepler_ok_20991231T235959.json").status_code == 404


@pytest.mark.parametrize(
    "name",
    [
        "secret.json",  # a real file next to the dumps, wrong name
        "..%2Fsecret.json",
        "%2E%2E%2Fsecret.json",
        "..%5Csecret.json",
        "kepler_ok_20261112T040000.json%2F..%2Fsecret.json",
        "kepler_ok_20261112T040000.json.bak",
        "kepler_ok_..json",
        "kepler_debug.lp",
    ],
)
def test_fetch_rejects_anything_but_a_dump_name(client, dump_dir, name):
    (dump_dir.parent / "secret.json").write_text('{"token": "x"}')
    (dump_dir / "secret.json").write_text('{"token": "x"}')
    (dump_dir / "kepler_ok_20261112T040000.json").write_text("{}")
    (dump_dir / "kepler_ok_20261112T040000.json.bak").write_text('{"token": "x"}')

    response = client.get(f"/api/debug/solver_dumps/{name}")

    assert response.status_code == 404
    assert "token" not in response.text


def test_dump_name_pattern_is_exact():
    from backend.api.routers.debug import _SOLVER_DUMP_NAME

    assert _SOLVER_DUMP_NAME.fullmatch("kepler_fail_20260810T082200.json")
    assert _SOLVER_DUMP_NAME.fullmatch("kepler_ok_20261112T040000.json")
    for bad in (
        "../kepler_ok_20261112T040000.json",
        "kepler_ok_20261112T040000.json\n",
        "kepler_ok_20261112T040000.json/",
        "sub/kepler_ok_20261112T040000.json",
        "kepler_other_20261112T040000.json",
        "",
    ):
        assert not _SOLVER_DUMP_NAME.fullmatch(bad), bad
