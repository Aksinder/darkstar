"""Shared fixtures for the planner test suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_solver_dumps(tmp_path, monkeypatch):
    """Redirect failed-solve instance dumps to tmp for EVERY planner test.

    Several solver tests intentionally drive solve() into its failure paths, which
    (since the 2026-07-22 instrumentation) persists a dump. Without this fixture
    those dumps land in the developer's/CI's working-tree ``data/solver_dumps/``
    and the KEEP-5 retention would evict REAL incident dumps copied there for
    offline replay.
    """
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(tmp_path / "solver_dumps"))
