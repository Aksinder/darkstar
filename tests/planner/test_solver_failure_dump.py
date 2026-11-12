"""Failed-solve instrumentation + slow-NotSolved CBC fallback (2026-07-22).

The 2026-07-15 SOLVER_TIMEOUT freeze ran 7 h on a stale plan because (a) a SLOW
HiGHS NotSolved (no incumbent within the budget) skipped the CBC fallback and
failed loud, and (b) the failing KeplerInput was discarded with the error, so the
incident could never be replayed offline. These tests pin both fixes:

- _dump_failed_solve_instance persists a replayable JSON artifact (input + config
  + outcome), honors DARKSTAR_SOLVER_DUMP_DIR, enforces retention, and NEVER
  raises (a dump failure must not mask the PlannerError being raised).
- A slow no-incumbent HiGHS timeout now falls back to CBC and returns a real plan
  instead of raising SOLVER_TIMEOUT.
"""

import json
import time as real_time
from datetime import datetime, timedelta
from pathlib import Path

import pulp
import pytest

import planner.solver.kepler as kepler_mod
from planner.solver.kepler import (
    _SOLVER_DUMP_KEEP,
    KeplerSolver,
    _dump_failed_solve_instance,
)
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    LoadGroup,
    WaterHeaterInput,
)


def _slots(n: int = 4) -> list[KeplerInputSlot]:
    base = datetime(2026, 7, 15, 9, 0)
    out: list[KeplerInputSlot] = []
    for i in range(n):
        s = base + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=0.5,
                pv_kwh=1.0,
                import_price_sek_kwh=1.5,
                export_price_sek_kwh=0.8,
            )
        )
    return out


def _cfg() -> KeplerConfig:
    return KeplerConfig(
        capacity_kwh=16.0,
        min_soc_percent=10.0,
        max_soc_percent=100.0,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        wear_cost_sek_per_kwh=0.0,
        water_heaters=[
            WaterHeaterInput(
                id="main_tank",
                power_kw=3.4,
                min_kwh_per_day=6.0,
                max_hours_between_heating=0.0,
                min_spacing_hours=0.0,
            )
        ],
    )


def _dump(**overrides):
    kwargs = {
        "reason": "not_optimal",
        "status": "Not Solved",
        "sol_status": 0,
        "used_solver": "highs",
        "solve_duration_s": 239.9,
        "chain_duration_s": 241.2,
    }
    kwargs.update(overrides)
    return _dump_failed_solve_instance(
        KeplerInput(slots=_slots(), initial_soc_kwh=9.1), _cfg(), **kwargs
    )


# -- the dump artifact ------------------------------------------------------


def test_dump_writes_replayable_json(tmp_path, monkeypatch):
    """The artifact carries outcome metadata plus the FULL input and config."""
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(tmp_path / "dumps"))
    path = _dump(extra={"var_count": 3305})

    assert path is not None
    payload = json.loads(Path(path).read_text())
    assert payload["reason"] == "not_optimal"
    assert payload["used_solver"] == "highs"
    assert payload["solve_duration_s"] == 239.9
    assert payload["var_count"] == 3305
    # Config and input survive with enough fidelity to rebuild the instance.
    assert payload["config"]["capacity_kwh"] == 16.0
    assert payload["config"]["water_heaters"][0]["id"] == "main_tank"
    assert len(payload["input"]["slots"]) == 4
    assert payload["input"]["initial_soc_kwh"] == 9.1
    # Datetimes were stringified, not dropped.
    assert "2026-07-15" in payload["input"]["slots"][0]["start_time"]


def test_dump_honors_env_dir_override(tmp_path, monkeypatch):
    """DARKSTAR_SOLVER_DUMP_DIR redirects dumps away from data/solver_dumps."""
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(override))
    path = _dump()

    assert path is not None
    assert Path(path).parent == override
    assert Path(path).exists()


def test_dump_retention_keeps_newest(tmp_path, monkeypatch):
    """Old dumps are pruned so the directory never grows unbounded."""
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(dump_dir))
    dump_dir.mkdir(parents=True)
    for i in range(_SOLVER_DUMP_KEEP + 2):
        (dump_dir / f"kepler_fail_20260101T00000{i}.json").write_text("{}")

    path = _dump()

    remaining = sorted(dump_dir.glob("kepler_fail_*.json"))
    assert len(remaining) == _SOLVER_DUMP_KEEP
    # The just-written dump (lexically newest) survives the pruning.
    assert Path(path).name in {p.name for p in remaining}


def test_dump_failure_returns_none_never_raises(tmp_path, monkeypatch):
    """An unwritable dump dir degrades to None — it must never mask the PlannerError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")  # mkdir of blocker/dumps will fail
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(blocker / "dumps"))

    assert _dump() is None


# -- slow-NotSolved -> CBC fallback (the 2026-07-15 freeze regression) ------


class _OffsetClock:
    """time.time() replacement whose reading can be advanced instantly."""

    def __init__(self) -> None:
        self.offset = 0.0
        self._t0 = real_time.time()

    def time(self) -> float:
        return self._t0 + self.offset


class _SlowNoIncumbentHiGHS(pulp.LpSolver):
    """Fake HiGHS: burns (almost) the whole budget and returns NO incumbent."""

    def __init__(self, clock: _OffsetClock, budget_s: float) -> None:
        super().__init__()
        self._clock = clock
        self._budget_s = budget_s

    def available(self) -> bool:  # pragma: no cover - pulp API surface
        return True

    def actualSolve(self, lp):
        self._clock.offset += 0.95 * self._budget_s  # slow: >= 0.9x budget
        lp.assignStatus(
            pulp.LpStatusNotSolved, pulp.constants.LpSolutionNoSolutionFound
        )
        return lp.status


def test_slow_notsolved_falls_back_to_cbc(monkeypatch):
    """REGRESSION (2026-07-15 freeze): a slow no-incumbent HiGHS timeout must fall
    back to CBC and return a real plan — NOT raise SOLVER_TIMEOUT and strand the
    executor on a stale plan for hours."""
    clock = _OffsetClock()
    monkeypatch.setattr(real_time, "time", clock.time)
    monkeypatch.setattr(
        pulp,
        "HiGHS",
        lambda **_kw: _SlowNoIncumbentHiGHS(clock, kepler_mod.SOLVER_TIME_LIMIT_S),
    )

    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(n=8), initial_soc_kwh=9.1), _cfg()
    )

    # CBC rescued the solve: a usable plan came back despite HiGHS "failing".
    assert result.is_optimal
    assert len(result.slots) == 8


def test_fast_notsolved_still_falls_back_to_cbc(monkeypatch):
    """The pre-existing fast-NotSolved (malfunction) path still reaches CBC."""
    clock = _OffsetClock()
    monkeypatch.setattr(real_time, "time", clock.time)

    class _FastNotSolvedHiGHS(_SlowNoIncumbentHiGHS):
        def actualSolve(self, lp):
            self._clock.offset += 1.0  # fast: far under 0.9x budget
            lp.assignStatus(
                pulp.LpStatusNotSolved, pulp.constants.LpSolutionNoSolutionFound
            )
            return lp.status

    monkeypatch.setattr(
        pulp,
        "HiGHS",
        lambda **_kw: _FastNotSolvedHiGHS(clock, kepler_mod.SOLVER_TIME_LIMIT_S),
    )

    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(n=8), initial_soc_kwh=9.1), _cfg()
    )

    assert result.is_optimal
    assert len(result.slots) == 8


# -- opt-in dumps of SUCCESSFUL solves (kepler.solver.dump_all_instances) ----


def _solve_ok(dump_all: int | None = None):
    cfg = _cfg()
    if dump_all is not None:
        cfg.solver_dump_all_instances = dump_all
    return KeplerSolver().solve(KeplerInput(slots=_slots(n=8), initial_soc_kwh=9.1), cfg)


def test_successful_solves_are_not_dumped_by_default(tmp_path, monkeypatch):
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(dump_dir))
    assert _solve_ok().is_optimal
    assert not dump_dir.exists()


def test_dump_all_instances_writes_a_replayable_ok_dump(tmp_path, monkeypatch):
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(dump_dir))

    result = _solve_ok(dump_all=3)

    files = list(dump_dir.glob("*.json"))
    assert [p.name.startswith("kepler_ok_") for p in files] == [True]
    payload = json.loads(files[0].read_text())
    assert payload["reason"] == "ok"
    assert payload["status"] == "Optimal"
    assert payload["objective_cost_sek"] == result.objective_cost_sek
    assert payload["var_count"] > 0
    assert len(payload["input"]["slots"]) == 8
    assert payload["config"]["water_heaters"][0]["id"] == "main_tank"


def test_ok_dumps_have_their_own_retention(tmp_path, monkeypatch):
    """dump_all_instances=N keeps the newest N ok dumps and never evicts failure dumps."""
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(dump_dir))
    dump_dir.mkdir(parents=True)
    for i in range(4):
        (dump_dir / f"kepler_ok_20260101T00000{i}.json").write_text("{}")
    for i in range(_SOLVER_DUMP_KEEP):
        (dump_dir / f"kepler_fail_20260101T00000{i}.json").write_text("{}")

    _solve_ok(dump_all=2)

    ok = sorted(p.name for p in dump_dir.glob("kepler_ok_*.json"))
    assert len(ok) == 2
    assert ok[0] == "kepler_ok_20260101T000003.json"  # newest pre-existing survives
    assert len(list(dump_dir.glob("kepler_fail_*.json"))) == _SOLVER_DUMP_KEEP

    # ...and a failure dump does not evict ok dumps either.
    _dump()
    assert len(list(dump_dir.glob("kepler_ok_*.json"))) == 2


def test_ok_dump_failure_never_breaks_the_solve(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(blocker / "dumps"))
    result = _solve_ok(dump_all=5)
    assert result.is_optimal
    assert len(result.slots) == 8


def test_adapter_reads_dump_all_instances():
    from planner.solver.adapter import config_to_kepler_config

    base = {
        "system": {
            "battery": {
                "capacity_kwh": 10.0,
                "max_charge_a": 100.0,
                "max_discharge_a": 100.0,
                "nominal_voltage_v": 48.0,
            }
        },
        "executor": {"inverter": {"control_unit": "A"}},
    }
    assert config_to_kepler_config(base).solver_dump_all_instances == 0
    for raw, expected in [(20, 20), ("7", 7), (0, 0), (-3, 0), (None, 0), ("x", 0), (True, 0)]:
        cfg = {**base, "kepler": {"solver": {"dump_all_instances": raw}}}
        assert config_to_kepler_config(cfg).solver_dump_all_instances == expected, raw


def test_failed_solve_writes_debug_lp_next_to_the_dumps_not_the_cwd(tmp_path, monkeypatch):
    """The debug LP used to land in the working directory (the repo root in tests)."""
    from planner.errors import PlannerError

    dump_dir = tmp_path / "data" / "solver_dumps"
    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(dump_dir))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    cfg = _cfg()
    # A forced-on 3.4 kW heater behind a 1 kW shared feed: infeasible.
    cfg.water_heaters[0].force_on_slots = [0]
    cfg.load_groups = [LoadGroup(id="feed", max_power_kw=1.0, members=["main_tank"])]
    with pytest.raises(PlannerError):
        KeplerSolver().solve(KeplerInput(slots=_slots(), initial_soc_kwh=9.1), cfg)
    assert (tmp_path / "data" / "kepler_debug.lp").exists()
    assert not list(cwd.iterdir())


def test_retention_ignores_stray_files_and_count_setting_is_bounded(tmp_path, monkeypatch):
    """Review follow-ups: a stray kepler_ok_*.json must not evict a real dump, and a
    garbage dump count (inf) must not raise out of the adapter."""
    from planner.solver import kepler as kepler_mod
    from planner.solver.adapter import _COUNT_SETTING_MAX, _non_negative_int

    assert _non_negative_int(float("inf")) == 0
    assert _non_negative_int("1e999") == 0
    assert _non_negative_int(10**9) == _COUNT_SETTING_MAX
    assert _non_negative_int(20) == 20

    assert kepler_mod._DUMP_STAMP_RE.fullmatch("kepler_ok_20261112T201500")
    assert not kepler_mod._DUMP_STAMP_RE.fullmatch("kepler_ok_notes")
