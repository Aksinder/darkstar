"""Regression tests for the 2026-10-11 audit, batch 4."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend.learning.savings import compute_savings
from executor.config import LoadGroupConfig
from executor.engine import ExecutorEngine


def _engine(groups: list[LoadGroupConfig]) -> Any:
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng.config = SimpleNamespace(load_groups=groups)  # type: ignore[assignment]
    return eng


def _dev(id_: str, kw: float, fuse_shed: bool = False) -> Any:
    return SimpleNamespace(id=id_, power_kw=kw, fuse_shed=fuse_shed, phase_map=("L2",))


def test_boost_admits_group_members_only_up_to_the_ceiling() -> None:
    eng = _engine([LoadGroupConfig(id="villavagn", max_power_kw=2.3, members=["spa", "vv_tank"])])
    used: dict[str, float] = {}
    assert eng._boost_blocked(_dev("main_tank", 3.4), None, used) is None  # not grouped
    assert eng._boost_blocked(_dev("spa", 1.8), None, used) is None
    why = eng._boost_blocked(_dev("vv_tank", 1.6), None, used)
    assert why is not None and "villavagn" in why
    assert used == {"villavagn": 1.8}, "a refused member must not consume budget"


def test_boost_yields_to_the_fuse() -> None:
    eng = _engine([])
    over = {"phase_currents": {"L1": 5.0, "L2": 24.5, "L3": 5.0}, "fuse_budget_a": 23.0, "grid_w": 9000.0}
    fine = {**over, "phase_currents": {"L1": 5.0, "L2": 6.0, "L3": 5.0}}
    assert eng._boost_blocked(_dev("vvb", 3.0, fuse_shed=True), over, {}) is not None
    assert eng._boost_blocked(_dev("vvb", 3.0, fuse_shed=True), fine, {}) is None
    assert eng._boost_blocked(_dev("vvb", 3.0, fuse_shed=False), over, {}) is None  # not opted in


def test_savings_ignore_rows_recorded_without_a_grid_reading() -> None:
    real = {"import_kwh": 1.0, "export_kwh": 0.0, "pv_kwh": 0.0, "load_kwh": 1.0,
            "import_price_sek_kwh": 2.0}
    outage = {"import_kwh": 0.0, "export_kwh": 0.0, "pv_kwh": 0.0, "load_kwh": 0.0,
              "water_kwh": 0.4, "ev_charging_kwh": 0.9, "import_price_sek_kwh": 2.0,
              "quality_flags": '{"grid_unmeasured": true, "soc_stale": true}'}
    assert compute_savings([real]).savings_sek == 0.0
    assert compute_savings([real, outage]).savings_sek == 0.0, "an outage is not a saving"
    # the same row WITHOUT the flag is what used to be booked: 1.3 kWh x 2 SEK
    unflagged = {k: v for k, v in outage.items() if k != "quality_flags"}
    assert compute_savings([real, unflagged]).savings_sek == 2.6


def test_workflows_pin_actions_by_commit() -> None:
    for wf in Path(".github/workflows").glob("*.yml"):
        for line in wf.read_text(encoding="utf-8").splitlines():
            m = re.search(r"uses:\s*(\S+)", line)
            if m and not m.group(1).startswith("./"):
                assert re.search(r"@[0-9a-f]{40}$", m.group(1)), f"{wf.name}: {line.strip()}"
