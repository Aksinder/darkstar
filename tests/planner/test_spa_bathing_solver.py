"""Spa bathing times in kepler (WP1): a continuous temperature state, no new binaries.

A heater with a valid BathingSpec gets the exact Newtonian recursion of
backend/core/spa_thermal.py driven by its existing water_heat binaries, plus soft
ready/hold/floor/cap rows. Evidence here:
  * the solver's per-slot temperature equals the pure-Python simulation for any
    heating pattern (uniform, mixed coarse-tail and partial-slot-0 grids);
  * the soft rows do what they say (cap, ready at least cost, eco floor, hold) and
    never make the model infeasible;
  * bathing=None (or a dropped spec) builds the byte-identical legacy LP, and
    bathing never adds an integer variable.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pulp
import pytest

from backend.core.spa_thermal import simulate_temps_c
from planner.errors import PlannerError, PlannerErrorCode
from planner.solver.bathing import map_windows, validate_bathing_spec
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    BathingSpec,
    BathWindow,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    KeplerResult,
    LoadGroup,
    WaterHeaterInput,
    planned_slot_hours,
)

BASE = datetime(2026, 1, 15, 6, 0)
TIGHT_GAP = 1e-9  # brute-force / formulation-equality tests
GAP = 1e-6  # everything else (1e-9 makes degenerate instances crawl)
_REAL_SOLVE = pulp.LpProblem.solve
TOL = 1e-4  # degC: HiGHS primal feasibility (1e-7) accumulated over the recursion


# ------------------------------------------------------------------ instances


def _price(i: int) -> float:
    return round(1.2 + 0.9 * ((i * 7) % 11) / 10.0 - (0.7 if 10 <= i < 18 else 0.0), 2)


def _slots(
    fine: int,
    coarse: int = 0,
    start: datetime = BASE,
    price: Any = None,
) -> list[KeplerInputSlot]:
    """``fine`` x 15 min followed by ``coarse`` x 1 h (the coarse-tail shape)."""
    out: list[KeplerInputSlot] = []
    t = start
    for i in range(fine + coarse):
        hours = 0.25 if i < fine else 1.0
        p = price(i) if price is not None else _price(i)
        out.append(
            KeplerInputSlot(
                start_time=t,
                end_time=t + timedelta(hours=hours),
                load_kwh=0.4 * hours,
                pv_kwh=0.0,
                import_price_sek_kwh=p,
                export_price_sek_kwh=round(p * 0.4, 2),
            )
        )
        t += timedelta(hours=hours)
    return out


def _spa(**kw: Any) -> WaterHeaterInput:
    base: dict[str, Any] = {
        "id": "spa",
        "power_kw": 1.8,
        "min_kwh_per_day": 4.0,
        "max_hours_between_heating": 0.0,
        "min_spacing_hours": 0.0,
        "absorb_cap_kwh_per_day": 6.0,
    }
    base.update(kw)
    return WaterHeaterInput(**base)


def _spec(windows: list[BathWindow] | None = None, **kw: Any) -> BathingSpec:
    base: dict[str, Any] = {
        "initial_temp_c": 30.0,
        "ambient_c": 10.0,
        "heat_capacity_kwh_per_c": 0.9,
        "cooling_k_per_h": 0.021,
        "ready_temp_c": 38.0,
        "eco_floor_c": 28.0,
        "temp_max_c": 40.0,
        "windows": windows or [],
    }
    base.update(kw)
    return BathingSpec(**base)


def _window(start: datetime, hours: float = 2.0, margin_min: float = 30.0) -> BathWindow:
    return BathWindow(
        start=start,
        end=start + timedelta(hours=hours),
        ready_by=start - timedelta(minutes=margin_min),
    )


def _cfg(heaters: list[WaterHeaterInput], **overrides: Any) -> KeplerConfig:
    kwargs: dict[str, Any] = {
        "capacity_kwh": 8.0,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 4.0,
        "max_discharge_power_kw": 4.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "wear_cost_sek_per_kwh": 0.1,
        "water_heaters": heaters,
        "water_reliability_penalty_sek": 50.0,
        "water_block_start_penalty_sek": 1.5,
        "water_block_penalty_sek": 0.8,
        "max_block_hours": 1.0,
        "water_hourly_blocks": False,
        "defer_up_to_hours": 0.0,
        "solver_gap_rel": GAP,
    }
    kwargs.update(overrides)
    return KeplerConfig(**kwargs)


def _solve(
    monkeypatch: pytest.MonkeyPatch,
    ki: KeplerInput,
    cfg: KeplerConfig,
    fix: dict[str, int] | None = None,
) -> tuple[KeplerResult, Any]:
    """Solve through KeplerSolver; optionally pin variables by name.

    Returns (KeplerResult, the LpProblem kepler built)."""
    captured: dict[str, Any] = {}

    def _spy(self: Any, solver: Any = None, **kw: Any) -> Any:
        captured["prob"] = self
        if fix:
            for var in self.variables():
                if var.name in fix:
                    var.lowBound = var.upBound = fix[var.name]
        return _REAL_SOLVE(self, solver, **kw)

    monkeypatch.setattr(pulp.LpProblem, "solve", _spy)
    try:
        result = KeplerSolver().solve(ki, cfg)
    finally:
        monkeypatch.setattr(pulp.LpProblem, "solve", _REAL_SOLVE)
    return result, captured["prob"]


def _lp_text(prob: Any, tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.lp"
    prob.writeLP(str(path))
    return path.read_text()


def _int_count(prob: Any) -> int:
    return sum(1 for v in prob.variables() if v.cat == pulp.LpInteger)


def _temps(result: KeplerResult, hid: str = "spa") -> list[float]:
    return [s.water_heater_temp_c[hid] for s in result.slots]


def _on(result: KeplerResult, hid: str = "spa") -> list[int]:
    return [1 if s.water_heater_results.get(hid, 0.0) > 0 else 0 for s in result.slots]


# ------------------------------------------------- 1-2: recursion exactness


@pytest.mark.parametrize(
    ("grid", "first_rem"),
    [("uniform", None), ("mixed", None), ("partial0", 0.1), ("mixed_partial0", 0.1)],
)
def test_recursion_matches_simulation(
    grid: str, first_rem: float | None, monkeypatch: pytest.MonkeyPatch
):
    slots = _slots(16) if grid in ("uniform", "partial0") else _slots(8, 6)
    T = len(slots)
    spec = _spec(temp_max_c=60.0, eco_floor_c=5.0, ready_temp_c=38.0)
    patterns = [
        [0] * T,
        [1 if t % 3 == 0 else 0 for t in range(T)],
        [1 if 2 <= t < 9 else 0 for t in range(T)],
    ]
    hours = planned_slot_hours(slots, first_rem)
    for pattern in patterns:
        ki = KeplerInput(slots, 4.0, first_slot_remaining_h=first_rem)
        cfg = _cfg([_spa(bathing=copy.deepcopy(spec))])
        fix = {f"water_heat_spa_{t}": pattern[t] for t in range(T)}
        result, _ = _solve(monkeypatch, ki, cfg, fix=fix)
        assert _on(result) == pattern
        expected = simulate_temps_c(30.0, 10.0, 0.021, 0.9, 1.8, hours, pattern)
        assert _temps(result) == pytest.approx(expected, abs=TOL)


def test_force_on_slots_drive_the_recursion(monkeypatch: pytest.MonkeyPatch):
    # Expensive energy and nothing to aim for: only the forced slots heat.
    slots = _slots(12, price=lambda i: 5.0)
    forced = [2, 3, 7]
    spec = _spec(eco_floor_c=5.0, initial_temp_c=30.0)
    cfg = _cfg([_spa(bathing=spec, force_on_slots=forced)])
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), cfg)
    pattern = [1 if t in forced else 0 for t in range(len(slots))]
    assert _on(result) == pattern
    expected = simulate_temps_c(30.0, 10.0, 0.021, 0.9, 1.8, [0.25] * len(slots), pattern)
    assert _temps(result) == pytest.approx(expected, abs=TOL)


def test_free_cooling_is_newtonian(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(8, 10)
    spec = _spec(eco_floor_c=5.0)  # no window, floor far below: nothing heats
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    assert sum(_on(result)) == 0
    elapsed = 0.0
    for s, temp in zip(result.slots, _temps(result), strict=True):
        elapsed += (s.end_time - s.start_time).total_seconds() / 3600.0
        assert temp == pytest.approx(10.0 + 20.0 * math.exp(-0.021 * elapsed), abs=TOL)


# ------------------------------------------------------------------- 3: cap


def test_negative_price_never_settles_above_temp_max(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(16, price=lambda i: -1.0)
    spec = _spec(initial_temp_c=37.0)
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    temps = _temps(result)
    assert sum(_on(result)) >= 6  # it does soak the paid energy (37 -> 40)...
    assert max(temps) <= 40.0 + 0.01  # ...but only up to the cap
    assert result.bathing_reports["spa"].overshoot_c <= 0.01


def test_forced_heat_past_temp_max_is_soft(monkeypatch: pytest.MonkeyPatch):
    # 16 forced quarter-hours from 39 degC run well past the cap: overshoot slack,
    # never infeasible.
    slots = _slots(20, price=lambda i: -1.0)
    spec = _spec(initial_temp_c=39.0)
    cfg = _cfg([_spa(bathing=spec, force_on_slots=list(range(16)))])
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), cfg)
    assert result.is_optimal
    assert result.bathing_reports["spa"].overshoot_c > 1.0


# --------------------------------------------------- 4: ready at least cost


def _ready_instance() -> tuple[KeplerInput, KeplerConfig]:
    prices = [2.5, 0.6, 2.2, 0.8, 1.9, 0.7, 2.4, 2.6]
    slots = _slots(0, 8, price=lambda i: prices[i])
    start = BASE + timedelta(hours=7, minutes=30)  # ready_by 13:00 = end of slot 6
    spec = _spec([_window(start)], initial_temp_c=34.0, hold_shortfall_sek_per_c_h=0.0)
    cfg = _cfg([_spa(bathing=spec)], water_block_start_penalty_sek=0.0, solver_gap_rel=TIGHT_GAP)
    return KeplerInput(slots, 4.0), cfg


def test_ready_by_is_met_at_least_cost(monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _ready_instance()
    result, _ = _solve(monkeypatch, ki, copy.deepcopy(cfg))
    rep = result.bathing_reports["spa"]
    assert rep.window_ready_temp_c[0] is not None
    assert rep.window_ready_temp_c[0] >= 38.0 - 0.05
    assert rep.window_shortfall_c[0] == pytest.approx(0.0, abs=0.05)
    best = math.inf
    for pattern in itertools.product((0, 1), repeat=len(ki.slots)):
        fix = {f"water_heat_spa_{t}": v for t, v in enumerate(pattern)}
        r, _ = _solve(monkeypatch, ki, copy.deepcopy(cfg), fix=fix)
        assert r.objective_cost_sek is not None
        best = min(best, r.objective_cost_sek)
    assert result.objective_cost_sek == pytest.approx(best, abs=1e-6)
    # The cheap hours (0.6 / 0.7 / 0.8) carry the heat.
    assert _on(result)[1] == 1 and _on(result)[3] == 1 and _on(result)[5] == 1


# ------------------------------------------------ 5: late schedule => slack


def test_late_schedule_gives_slack_not_infeasible(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(16)
    start = BASE + timedelta(hours=2)  # ready_by 01:30 from now: 20 -> 38 impossible
    spec = _spec([_window(start)], initial_temp_c=20.0)
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    assert result.is_optimal and result.status_msg == "Optimal"
    rep = result.bathing_reports["spa"]
    short, unavoidable = rep.window_shortfall_c[0], rep.unavoidable_shortfall_c[0]
    assert short is not None and unavoidable is not None
    assert short > 10.0
    assert short == pytest.approx(unavoidable, abs=0.01)  # all of it is physics
    assert _on(result)[:6] == [1] * 6  # full power from now to ready_by


# ------------------------------------------------------------ 6: horizon


def test_window_beyond_horizon_has_no_target(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(8, 6)  # ends 14:00
    spec = _spec([_window(BASE + timedelta(hours=9))], initial_temp_c=35.0)  # ready 14:30
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    rep = result.bathing_reports["spa"]
    assert rep.window_ready_temp_c == [None]
    assert rep.window_shortfall_c == [None]
    assert rep.unavoidable_shortfall_c == [None]
    assert sum(_on(result)) == 0


# ----------------------------------------------------------- 7: eco floor


def test_eco_floor_books_heat_before_crossing(monkeypatch: pytest.MonkeyPatch):
    # Hourly: free cooling from 28.5 passes 28 during the 2nd hour.
    slots = _slots(0, 24)
    spec = _spec(initial_temp_c=28.5, floor_shortfall_sek_per_c_h=50.0)
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    on = _on(result)
    assert on[0] == 1 or on[1] == 1
    assert min(_temps(result)) >= 28.0 - 0.01
    assert result.bathing_reports["spa"].floor_shortfall_c_h == pytest.approx(0.0, abs=0.01)


# ----------------------------------------------------------- 8: hold


def test_hold_in_window_at_minimum_energy(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(12, price=lambda i: 1.0)
    # In progress since 30 min ago, 3 h long: hold only, no ready target.
    win = BathWindow(
        start=BASE - timedelta(minutes=30),
        end=BASE + timedelta(hours=3),
        ready_by=BASE - timedelta(hours=1),
    )
    spec = _spec([win], initial_temp_c=38.0, hold_shortfall_sek_per_c_h=50.0)
    cfg = _cfg([_spa(bathing=spec)], water_block_start_penalty_sek=0.0)
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), cfg)
    rep = result.bathing_reports["spa"]
    assert rep.window_ready_temp_c == [None]
    assert all(t >= 37.0 - 0.01 for t in _temps(result))
    # Free cooling from 38 over 3 h ends ~36.3: 0.7 degC net is two quarter-hours.
    assert sum(_on(result)) == 2


# --------------------------------------------------- 9: legacy identity


def _legacy_case() -> tuple[KeplerInput, KeplerConfig]:
    slots = _slots(16, 12)
    vvb = WaterHeaterInput(
        id="vvb",
        power_kw=3.0,
        min_kwh_per_day=4.5,
        max_hours_between_heating=0.0,
        min_spacing_hours=1.0,
    )
    cfg = _cfg(
        [_spa(), vvb],
        load_groups=[LoadGroup(id="villavagn", max_power_kw=4.0, members=["spa", "vvb"])],
        water_hourly_blocks=True,
    )
    return KeplerInput(slots, 4.0), cfg


# Pinned against the pre-bathing kepler (verified byte-identical LP text at HEAD
# 971bfab9 when the feature was built).
_LEGACY_VARS = 510
_LEGACY_ROWS = 420
_LEGACY_INTS = 112


def test_no_bathing_is_the_legacy_model(monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _legacy_case()
    result, prob = _solve(monkeypatch, ki, cfg)
    assert (len(prob.variables()), len(prob.constraints), _int_count(prob)) == (
        _LEGACY_VARS,
        _LEGACY_ROWS,
        _LEGACY_INTS,
    )
    assert result.bathing_reports == {}
    assert all(s.water_heater_temp_c == {} for s in result.slots)


@pytest.mark.parametrize("flow", [False, True])
def test_bathing_adds_no_integer_variables(flow: bool, monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _legacy_case()
    cfg.water_flow_formulation = flow
    _, legacy = _solve(monkeypatch, ki, copy.deepcopy(cfg))
    bath_cfg = copy.deepcopy(cfg)
    bath_cfg.water_heaters[0].bathing = _spec([_window(BASE + timedelta(hours=10))])
    result, bathing = _solve(monkeypatch, ki, bath_cfg)
    assert "spa" in result.bathing_reports
    assert _int_count(bathing) == _int_count(legacy)
    assert len(bathing.variables()) > len(legacy.variables())  # continuous state added


def test_bathing_drops_the_spa_legacy_economics(monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _legacy_case()
    cfg.water_heaters[0].bathing = _spec([_window(BASE + timedelta(hours=10))])
    _, prob = _solve(monkeypatch, ki, cfg)
    names = {v.name for v in prob.variables()}
    assert not any(n.startswith("water_absorb_overage_spa") for n in names)
    assert not any(n.startswith("block_overshoot_spa") for n in names)
    assert not any(n.startswith("wtp_served_spa") for n in names)
    assert any(n.startswith("block_overshoot_vvb") for n in names)  # the vvb keeps its rows
    assert any(n.startswith("bath_temp_spa") for n in names)


# ------------------------------------- 10: other tank + load group unaffected


def test_second_tank_and_load_group_still_hold(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(16, 12)
    villa = WaterHeaterInput(
        id="villavagn_tank",
        power_kw=1.5,
        min_kwh_per_day=3.0,
        max_hours_between_heating=0.0,
        min_spacing_hours=0.0,
    )
    spec = _spec([_window(BASE + timedelta(hours=10))])
    cfg = _cfg(
        [_spa(bathing=spec), villa],
        load_groups=[
            LoadGroup(id="villavagn", max_power_kw=2.3, members=["spa", "villavagn_tank"])
        ],
    )
    result, prob = _solve(monkeypatch, KeplerInput(slots, 4.0), cfg)
    assert result.bathing_reports["spa"].window_shortfall_c[0] == pytest.approx(0.0, abs=0.05)
    villa_kwh = 0.0
    for s in result.slots:
        h = (s.end_time - s.start_time).total_seconds() / 3600.0
        spa_kw = s.water_heater_results["spa"]
        villa_kw = s.water_heater_results["villavagn_tank"]
        assert spa_kw + villa_kw <= 2.3 + 1e-9  # never co-on
        villa_kwh += villa_kw * h
    assert villa_kwh >= 3.0 - 1e-6  # its daily floor still binds
    for v in prob.variables():
        if v.name.startswith("water_min_kwh_violation_villavagn_tank"):
            assert (v.varValue or 0.0) <= 1e-6


# ------------------------------------------- 11: naive / aware / mixed


def test_all_aware_pairing_is_modelled(monkeypatch: pytest.MonkeyPatch):
    tz = ZoneInfo("Europe/Stockholm")
    start = BASE.replace(tzinfo=tz)
    slots = _slots(16, 12, start=start)
    spec = _spec([_window(start + timedelta(hours=10))])
    result, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=spec)]))
    assert result.bathing_reports["spa"].window_ready_temp_c[0] is not None


def test_mixed_pairing_is_dropped_and_legacy_equal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    slots = _slots(16, 12)  # naive
    aware_start = (BASE + timedelta(hours=10)).replace(tzinfo=ZoneInfo("Europe/Stockholm"))
    mixed = _spec([_window(aware_start)])
    with caplog.at_level(logging.ERROR, logger="darkstar.kepler"):
        result, prob = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa(bathing=mixed)]))
    assert result.bathing_reports == {}
    assert any("bathing spec for spa dropped" in r.getMessage() for r in caplog.records)
    legacy, legacy_prob = _solve(monkeypatch, KeplerInput(slots, 4.0), _cfg([_spa()]))
    assert _lp_text(prob, tmp_path, "mixed") == _lp_text(legacy_prob, tmp_path, "legacy")
    assert result.objective_cost_sek == pytest.approx(legacy.objective_cost_sek)


def test_validator_rejects_bad_specs():
    slots = _slots(4)
    good = _spa(bathing=_spec())
    assert validate_bathing_spec(good, slots) is None
    bad: list[WaterHeaterInput] = [
        _spa(bathing=_spec(cooling_k_per_h=0.0)),
        _spa(bathing=_spec(heat_capacity_kwh_per_c=-1.0)),
        _spa(bathing=_spec(ready_temp_c=41.0)),  # above temp_max
        _spa(bathing=_spec(eco_floor_c=38.0)),  # not below ready
        _spa(bathing=_spec(initial_temp_c=float("nan"))),
        _spa(power_kw=0.0, bathing=_spec()),
    ]
    for heater in bad:
        assert validate_bathing_spec(heater, slots) is not None


def test_window_mapping_rules():
    slots = _slots(4, 4)  # 06:00-07:00 quarters, then 07:00-11:00 hourly
    hours = planned_slot_hours(slots, 0.1)  # now = 06:09
    windows = [
        BathWindow(BASE - timedelta(hours=2), BASE, BASE - timedelta(hours=3)),  # past
        BathWindow(BASE, BASE + timedelta(hours=1), BASE - timedelta(hours=1)),  # in progress
        BathWindow(  # ready_by inside slot 0 -> state 1
            BASE + timedelta(minutes=30), BASE + timedelta(hours=1), BASE + timedelta(minutes=12)
        ),
        BathWindow(  # ready 08:40 snaps to the 08:00 boundary (state 5)
            BASE + timedelta(hours=2, minutes=50),
            BASE + timedelta(hours=4),
            BASE + timedelta(hours=2, minutes=40),
        ),
        BathWindow(  # beyond the horizon end (11:00)
            BASE + timedelta(hours=6),
            BASE + timedelta(hours=7),
            BASE + timedelta(hours=5, minutes=30),
        ),
    ]
    wmap = map_windows(_spec(windows), slots, hours)
    assert wmap.ready_state == [None, None, 1, 5, None]
    # In-progress window holds slot 0 (counted from now) to 07:00; 08:50-10:00 holds
    # only the fully covered 09:00-10:00 hour.
    assert wmap.hold_slots == [0, 1, 2, 3, 6]


# ------------------------------------------------------------ 12: tripwire


def test_tripwire_rejects_avoidable_shortfall(monkeypatch: pytest.MonkeyPatch):
    # A near-free ready penalty makes the "optimal" plan skip the bath entirely —
    # exactly the garbage a time-boxed incumbent can ship.
    slots = _slots(8, 8)
    spec = _spec(
        [_window(BASE + timedelta(hours=8))],
        ready_shortfall_sek_per_c=0.001,
        hold_shortfall_sek_per_c_h=0.0,
    )
    cfg = _cfg([_spa(bathing=spec)])
    monkeypatch.setattr("planner.solver.kepler.solve_is_time_boxed", lambda *a, **k: True)
    with pytest.raises(PlannerError) as exc:
        KeplerSolver().solve(KeplerInput(slots, 4.0), cfg)
    assert exc.value.code == PlannerErrorCode.SOLVER_TIMEOUT
    assert "bathing" in str(exc.value.details.get("reason", ""))
    assert exc.value.details.get("avoidable_shortfall_c", 0) > 1.0


def test_tripwire_accepts_unavoidable_shortfall(monkeypatch: pytest.MonkeyPatch):
    slots = _slots(16)
    spec = _spec([_window(BASE + timedelta(hours=2))], initial_temp_c=20.0)
    cfg = _cfg([_spa(bathing=spec)])
    monkeypatch.setattr("planner.solver.kepler.solve_is_time_boxed", lambda *a, **k: True)
    result = KeplerSolver().solve(KeplerInput(slots, 4.0), cfg)
    assert "time-boxed incumbent" in result.status_msg
    assert (result.bathing_reports["spa"].window_shortfall_c[0] or 0.0) > 10.0


# -------------------------------------------------------- 13: flow on/off


@pytest.mark.parametrize("hourly", [False, True])
def test_flow_formulation_agrees_with_bathing(hourly: bool, monkeypatch: pytest.MonkeyPatch):
    slots = _slots(16, 12)
    spec = _spec([_window(BASE + timedelta(hours=10))])
    cfg = _cfg(
        [_spa(bathing=spec, min_spacing_hours=1.0)],
        water_hourly_blocks=hourly,
        solver_gap_rel=TIGHT_GAP,
    )
    off_cfg = copy.deepcopy(cfg)
    on_cfg = copy.deepcopy(cfg)
    on_cfg.water_flow_formulation = True
    off, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), off_cfg)
    on, _ = _solve(monkeypatch, KeplerInput(slots, 4.0), on_cfg)
    assert off.objective_cost_sek == pytest.approx(on.objective_cost_sek, abs=1e-5)


# ---------------------------------------------- 14b: dump round trip


def _load_replay() -> Any:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "replay_solver_dump.py"
    spec = importlib.util.spec_from_file_location("replay_solver_dump", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dump_round_trip_rebuilds_the_spec():
    import json

    replay = _load_replay()
    tz = ZoneInfo("Europe/Stockholm")
    spec = _spec([_window((BASE + timedelta(hours=10)).replace(tzinfo=tz))])
    cfg = _cfg([_spa(bathing=spec), _spa(id="vvb")])
    # Same serialisation as kepler._dump_failed_solve_instance.
    dumped = json.loads(json.dumps(asdict(cfg), default=str))
    back = replay.rehydrate_config(dumped)
    assert back.water_heaters[0].bathing == spec
    assert back.water_heaters[0].bathing.windows[0].start.utcoffset() == timedelta(hours=1)
    assert back.water_heaters[1].bathing is None


def test_replay_mutations_attach_and_drop_bathing():
    replay = _load_replay()
    ki = KeplerInput(_slots(8, 60), 4.0)
    cfg = _cfg([])
    ki, cfg = replay.MUTATIONS["synth_bathing"](ki, cfg, None)
    spa = cfg.water_heaters[0]
    assert spa.id == "spa" and spa.power_kw == 1.8 and spa.bathing is not None
    starts = [w.start for w in spa.bathing.windows]
    assert starts == [BASE.replace(hour=19) + timedelta(days=d) for d in range(3)]
    ki, cfg = replay.MUTATIONS["no_bathing"](ki, cfg, None)
    assert cfg.water_heaters[0].bathing is None
