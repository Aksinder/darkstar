"""Battery-only fallback planner, independent plan validator and the plan guard."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from planner.errors import PlannerError, PlannerErrorCode
from planner.plan_guard import (
    PlanGuardSettings,
    read_last_plan,
    settings_from_config,
    solve_guarded,
)
from planner.plan_validator import validate_plan
from planner.solver.battery_lp import BenchmarkTerms, solve_battery_only
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    KeplerResult,
    WaterHeaterInput,
)

if TYPE_CHECKING:
    from pathlib import Path

START = datetime(2026, 10, 21, 0, 0)
CAP = 16.0


def _config(**kw: Any) -> KeplerConfig:
    base: dict[str, Any] = {
        "capacity_kwh": CAP,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 9.5,
        "max_discharge_power_kw": 9.5,
        "charge_efficiency": 0.975,
        "discharge_efficiency": 0.975,
        "wear_cost_sek_per_kwh": 0.45,
        "max_export_power_kw": 16.0,
        "max_import_power_kw": 16.0,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _day_slots(n: int = 192) -> list[KeplerInputSlot]:
    """Two days: cheap nights, expensive evening peaks, midday PV."""
    out: list[KeplerInputSlot] = []
    for i in range(n):
        t = START + timedelta(minutes=15 * i)
        hr = t.hour + t.minute / 60
        price = 0.4 if hr < 6 else (2.8 if 17 <= hr < 21 else 1.2)
        pv = max(0.0, 2.0 * (1 - abs(hr - 13) / 5)) if 8 <= hr <= 18 else 0.0
        out.append(
            KeplerInputSlot(
                start_time=t,
                end_time=t + timedelta(minutes=15),
                load_kwh=0.3 + (0.4 if 17 <= hr < 22 else 0.0),
                pv_kwh=pv,
                import_price_sek_kwh=price,
                export_price_sek_kwh=price * 0.6,
            )
        )
    return out


# ---------------------------------------------------------------- battery_lp


def test_fallback_solves_two_days_fast_and_arbitrages():
    inp = KeplerInput(slots=_day_slots(), initial_soc_kwh=4.0)
    started = time.monotonic()
    res = solve_battery_only(inp, _config(), mode="live")
    assert time.monotonic() - started < 2.0
    assert len(res.slots) == 192
    # Charges in the cheap night, discharges into the evening peak.
    night = sum(s.charge_kwh for s in res.slots[:24])
    evening = sum(s.discharge_kwh for s in res.slots[68:84])
    assert night > 1.0
    assert evening > 1.0
    assert all(not s.water_heater_results and not s.ev_charger_results for s in res.slots)
    assert validate_plan(res, inp, _config()).ok


def test_fallback_respects_recovery_window_floor():
    # Initial SoC 2 pp below min_soc: inside the recovery window it must not drain
    # further, and it must be back at min_soc after the window.
    cfg = _config()
    initial = CAP * 0.08
    slots = _day_slots(96)
    res = solve_battery_only(KeplerInput(slots=slots, initial_soc_kwh=initial), cfg)
    assert min(s.soc_kwh for s in res.slots[:24]) >= initial - 1e-6
    assert all(s.soc_kwh >= CAP * 0.10 - 1e-6 for s in res.slots[24:])


def test_fallback_honours_export_toggle_and_limits():
    cfg = _config(enable_export=False, max_import_power_kw=5.0)
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=8.0)
    res = solve_battery_only(inp, cfg)
    assert all(s.grid_export_kwh <= 1e-9 for s in res.slots)
    assert all(s.grid_import_kwh <= 5.0 * 0.25 + 1e-6 for s in res.slots)
    assert validate_plan(res, inp, cfg).ok


def test_benchmark_mode_hits_terminal_and_blocks_discharge():
    slots = _day_slots(96)
    no_dis = [17 <= s.start_time.hour < 19 for s in slots]
    res = solve_battery_only(
        KeplerInput(slots=slots, initial_soc_kwh=8.0),
        _config(),
        mode="benchmark",
        benchmark=BenchmarkTerms(terminal_soc_kwh=6.0, no_discharge_slots=no_dis),
    )
    assert res.slots[-1].soc_kwh == pytest.approx(6.0, abs=1e-6)
    assert all(s.discharge_kwh <= 1e-9 for s, nd in zip(res.slots, no_dis, strict=True) if nd)


# ---------------------------------------------------------------- validator


def test_validator_accepts_kepler_plan():
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    cfg = _config()
    res = KeplerSolver().solve(inp, cfg)
    report = validate_plan(res, inp, cfg)
    assert report.ok, report.errors
    assert report.recomputed_cost_sek == pytest.approx(res.total_cost_sek, abs=0.05)


def test_validator_accepts_kepler_plan_with_water():
    cfg = _config(
        water_heaters=[
            WaterHeaterInput(
                id="main",
                power_kw=3.0,
                min_kwh_per_day=6.0,
                max_hours_between_heating=24.0,
                min_spacing_hours=0.0,
            )
        ],
        water_reliability_penalty_sek=100.0,
    )
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    res = KeplerSolver().solve(inp, cfg)
    assert sum(s.water_heat_kw for s in res.slots) > 0
    report = validate_plan(res, inp, cfg)
    assert report.ok, report.errors


def _solved() -> tuple[KeplerResult, KeplerInput, KeplerConfig]:
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    cfg = _config()
    return solve_battery_only(inp, cfg), inp, cfg


def test_validator_flags_soc_out_of_range_and_trajectory():
    res, inp, cfg = _solved()
    res.slots[10] = replace(res.slots[10], soc_kwh=CAP + 1.0)
    report = validate_plan(res, inp, cfg)
    assert not report.ok
    assert any("outside" in e for e in report.errors)
    assert any("does not follow" in e for e in report.errors)


def test_validator_flags_energy_balance_and_power_limits():
    res, inp, cfg = _solved()
    s = res.slots[5]
    # Phantom import with nowhere to go, above the grid limit.
    res.slots[5] = replace(s, grid_import_kwh=s.grid_import_kwh + 10.0)
    report = validate_plan(res, inp, cfg)
    assert not report.ok
    assert any("surplus" in e for e in report.errors)
    assert any("grid limit" in e for e in report.errors)
    assert any("recomputed" in e for e in report.errors)


def test_validator_flags_unserved_demand_from_device_load():
    res, inp, cfg = _solved()
    s = res.slots[3]
    res.slots[3] = replace(s, water_heater_results={"main": 3.0}, water_heat_kw=3.0)
    report = validate_plan(res, inp, cfg)
    assert any("short" in e for e in report.errors)


def test_validator_warnings_do_not_fail():
    res, inp, cfg = _solved()
    tight = replace(cfg, max_soc_percent=20.0)
    report = validate_plan(res, inp, tight)
    assert report.ok
    assert report.warning_count >= 1


def test_validator_slot_count_mismatch():
    res, inp, cfg = _solved()
    res.slots.pop()
    assert not validate_plan(res, inp, cfg).ok


# ---------------------------------------------------------------- guard


def _write_plan(path: Path, age_min: float, source: str | None = None) -> None:
    meta: dict[str, Any] = {"planned_at": (datetime.now() - timedelta(minutes=age_min)).isoformat()}
    if source:
        meta["plan_source"] = source
    path.write_text(json.dumps({"schedule": [], "meta": meta}))


def _timeout(*_a: Any) -> KeplerResult:
    raise PlannerError(
        code=PlannerErrorCode.SOLVER_TIMEOUT,
        details={"reason": "time-boxed incumbent violates water comfort floors"},
    )


def test_settings_defaults_off_and_log():
    s = settings_from_config({})
    assert s == PlanGuardSettings(False, 30.0, "log")
    s = settings_from_config(
        {
            "kepler": {
                "fallback": {"enabled": True, "min_plan_age_minutes": 45},
                "plan_validator": {"mode": "ENFORCE"},
            }
        }
    )
    assert s == PlanGuardSettings(True, 45.0, "enforce")
    assert (
        settings_from_config({"kepler": {"plan_validator": {"mode": "bogus"}}}).validator_mode
        == "log"
    )


def test_guard_success_path_marks_kepler(tmp_path: Path):
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    res, meta = solve_guarded(
        KeplerSolver().solve, inp, _config(), PlanGuardSettings(), tmp_path / "s.json"
    )
    assert meta["plan_source"] == "kepler"
    assert meta["plan_validation"]["ok"] is True
    assert res.slots


def test_guard_disabled_fallback_reraises(tmp_path: Path):
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    with pytest.raises(PlannerError) as ei:
        solve_guarded(_timeout, inp, _config(), PlanGuardSettings(), tmp_path / "s.json")
    assert ei.value.code == PlannerErrorCode.SOLVER_TIMEOUT


def test_guard_keeps_fresh_kepler_plan(tmp_path: Path):
    path = tmp_path / "s.json"
    _write_plan(path, age_min=10)
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    with pytest.raises(PlannerError):
        solve_guarded(_timeout, inp, _config(), PlanGuardSettings(fallback_enabled=True), path)


def test_guard_falls_back_when_last_plan_is_stale(tmp_path: Path):
    path = tmp_path / "s.json"
    _write_plan(path, age_min=95)
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    res, meta = solve_guarded(
        _timeout, inp, _config(), PlanGuardSettings(fallback_enabled=True), path
    )
    assert meta["plan_source"] == "fallback"
    assert meta["fallback"]["reason"] == "SOLVER_TIMEOUT"
    assert meta["fallback"]["last_plan_age_minutes"] == pytest.approx(95, abs=1)
    assert meta["fallback"]["devices_scheduled"] is False
    assert len(res.slots) == 96


def test_guard_falls_back_without_any_plan(tmp_path: Path):
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    _res, meta = solve_guarded(
        _timeout, inp, _config(), PlanGuardSettings(fallback_enabled=True), tmp_path / "none.json"
    )
    assert meta["plan_source"] == "fallback"
    assert meta["fallback"]["last_plan_age_minutes"] is None


def test_guard_refreshes_previous_fallback_regardless_of_age(tmp_path: Path):
    path = tmp_path / "s.json"
    _write_plan(path, age_min=5, source="fallback")
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    _res, meta = solve_guarded(
        _timeout, inp, _config(), PlanGuardSettings(fallback_enabled=True), path
    )
    assert meta["plan_source"] == "fallback"


def test_guard_does_not_fallback_on_input_errors(tmp_path: Path):
    def bad(*_a: Any) -> KeplerResult:
        raise PlannerError(code=PlannerErrorCode.PRICES_UNAVAILABLE)

    inp = KeplerInput(slots=_day_slots(8), initial_soc_kwh=6.0)
    with pytest.raises(PlannerError) as ei:
        solve_guarded(bad, inp, _config(), PlanGuardSettings(fallback_enabled=True), tmp_path / "x")
    assert ei.value.code == PlannerErrorCode.PRICES_UNAVAILABLE


def _corrupt_solver(inp: KeplerInput, cfg: KeplerConfig) -> KeplerResult:
    res = solve_battery_only(inp, cfg)
    res.slots[0] = replace(res.slots[0], soc_kwh=-5.0)
    return res


def test_guard_log_mode_ships_invalid_plan(tmp_path: Path):
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    _res, meta = solve_guarded(
        _corrupt_solver, inp, _config(), PlanGuardSettings(), tmp_path / "s.json"
    )
    assert meta["plan_source"] == "kepler"
    assert meta["plan_validation"]["ok"] is False


def test_guard_enforce_mode_rejects_invalid_plan(tmp_path: Path):
    inp = KeplerInput(slots=_day_slots(96), initial_soc_kwh=6.0)
    with pytest.raises(PlannerError) as ei:
        solve_guarded(
            _corrupt_solver,
            inp,
            _config(),
            PlanGuardSettings(validator_mode="enforce"),
            tmp_path / "s.json",
        )
    assert ei.value.code == PlannerErrorCode.INVALID_SCHEDULE
    # ...and with fallback enabled the rejected plan is replaced.
    _res, meta = solve_guarded(
        _corrupt_solver,
        inp,
        _config(),
        PlanGuardSettings(fallback_enabled=True, validator_mode="enforce"),
        tmp_path / "s.json",
    )
    assert meta["plan_source"] == "fallback"
    assert meta["fallback"]["rejected_kepler_validation"]["ok"] is False


def test_guard_fallback_failure_reraises_original(tmp_path: Path, monkeypatch):
    import planner.plan_guard as pg

    def boom(*_a: Any, **_k: Any) -> KeplerResult:
        raise RuntimeError("lp exploded")

    monkeypatch.setattr(pg, "solve_battery_only", boom)
    inp = KeplerInput(slots=_day_slots(8), initial_soc_kwh=6.0)
    with pytest.raises(PlannerError) as ei:
        solve_guarded(
            _timeout, inp, _config(), PlanGuardSettings(fallback_enabled=True), tmp_path / "x"
        )
    assert ei.value.code == PlannerErrorCode.SOLVER_TIMEOUT


def test_read_last_plan_naive_and_aware(tmp_path: Path):
    path = tmp_path / "s.json"
    _write_plan(path, age_min=40, source="kepler")
    age, src = read_last_plan(path)
    assert age == pytest.approx(40, abs=1) and src == "kepler"
    from datetime import UTC

    aware = (datetime.now(UTC) - timedelta(minutes=70)).isoformat()
    path.write_text(json.dumps({"meta": {"planned_at": aware}}))
    age, src = read_last_plan(path)
    assert age == pytest.approx(70, abs=1) and src == "kepler"
    assert read_last_plan(tmp_path / "missing.json") == (None, "none")


@pytest.mark.asyncio
async def test_schedule_meta_carries_plan_source(tmp_path: Path):
    from planner.output.schedule import save_schedule_to_json
    from tests.planner.test_schedule_atomic_write import _df

    out = tmp_path / "schedule.json"
    await save_schedule_to_json(
        _df(),
        config={"timezone": "Europe/Stockholm"},
        now_slot=None,
        forecast_meta={},
        s_index_debug=None,
        window_responsibilities=[],
        planner_state={},
        output_path=str(out),
        extra_meta={"plan_source": "fallback", "planned_at": "must-not-override"},
    )
    meta = json.loads(out.read_text())["meta"]
    assert meta["plan_source"] == "fallback"
    assert meta["planned_at"] != "must-not-override"
    age, source = read_last_plan(out)
    assert source == "fallback" and age is not None and age < 1
