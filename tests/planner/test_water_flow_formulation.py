"""Arc-flow water formulation (water_heating.flow_formulation) must be EXACT.

The flag swaps the start binaries, the big-M spacing rows and the block-window
overshoot rows for a unit flow through a small state graph on the water decision
grid (planner.solver.kepler.add_water_flow). It is a performance change only:
every on/off pattern must cost exactly what it costs in the default rows, and be
feasible exactly when it is feasible there — on the uniform grid, on the mixed
(coarse-tail) grid, with and without hourly blocks, for spacing 0, spacing below
the block cap and spacing above it.

Three layers of evidence:
  * solve both formulations to a tight gap -> equal objective, and the pattern
    each one picks costs the same when fixed into the other;
  * brute force on tiny horizons: EVERY pattern, both models, same cost or both
    infeasible;
  * flag off -> the model is the pre-flag one (pinned variable/row counts).
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any

import pulp
import pytest

from planner.solver.adapter import config_to_kepler_config
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    LoadGroup,
    LoadPriority,
    WaterHeaterInput,
)

BASE = datetime(2026, 1, 15, 6, 0)
TIGHT_GAP = 1e-9
_REAL_SOLVE = pulp.LpProblem.solve


# ------------------------------------------------------------------ instances


def _price(i: int) -> float:
    """Deterministic öre-rounded price with two valleys so decisions are non-trivial."""
    return round(1.2 + 0.9 * ((i * 7) % 11) / 10.0 - (0.7 if 10 <= i < 18 else 0.0), 2)


def _slots(
    fine: int, coarse: int = 0, start: datetime = BASE, pv_peak: float = 0.0
) -> list[KeplerInputSlot]:
    """``fine`` x 15 min followed by ``coarse`` x 1 h (the coarse-tail shape)."""
    out: list[KeplerInputSlot] = []
    t = start
    for i in range(fine + coarse):
        hours = 0.25 if i < fine else 1.0
        pv = pv_peak * hours if 6 <= i < 16 else 0.0
        out.append(
            KeplerInputSlot(
                start_time=t,
                end_time=t + timedelta(hours=hours),
                load_kwh=0.4 * hours,
                pv_kwh=pv,
                import_price_sek_kwh=_price(i),
                export_price_sek_kwh=round(_price(i) * 0.4, 2),
            )
        )
        t += timedelta(hours=hours)
    return out


def _heater(hid: str = "vvb", **kw: Any) -> WaterHeaterInput:
    base: dict[str, Any] = {
        "id": hid,
        "power_kw": 3.0,
        "min_kwh_per_day": 4.5,
        "max_hours_between_heating": 0.0,
        "min_spacing_hours": 0.0,
    }
    base.update(kw)
    return WaterHeaterInput(**base)


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
        "solver_gap_rel": TIGHT_GAP,
    }
    kwargs.update(overrides)
    return KeplerConfig(**kwargs)


def _case(name: str) -> tuple[KeplerInput, KeplerConfig]:
    """Small instances covering every rule the flow graph replaces or touches."""
    uniform = _slots(32)
    mixed = _slots(16, 12)
    mid_hour = _slots(18, 10, start=BASE + timedelta(minutes=30))  # partial first hour group
    if name == "uniform_spacing_0":
        return KeplerInput(uniform, 4.0), _cfg([_heater()])
    if name == "uniform_spacing_below_cap":
        return KeplerInput(uniform, 4.0), _cfg([_heater(min_spacing_hours=0.5)])
    if name == "uniform_spacing_above_cap":
        return KeplerInput(uniform, 4.0), _cfg([_heater(min_spacing_hours=2.0)])
    if name == "uniform_hourly_blocks":
        return KeplerInput(uniform, 4.0), _cfg(
            [_heater(min_spacing_hours=2.0)], water_hourly_blocks=True
        )
    if name == "uniform_hourly_blocks_window_inside_hour":
        # Block cap shorter than an hour: windows that lie inside ONE hour group.
        return KeplerInput(uniform, 4.0), _cfg(
            [_heater(min_spacing_hours=1.0)], water_hourly_blocks=True, max_block_hours=0.5
        )
    if name == "uniform_odd_block_hours":
        return KeplerInput(uniform, 4.0), _cfg(
            [_heater(min_spacing_hours=0.75)], max_block_hours=0.6
        )
    if name == "uniform_no_penalties_spacing_only":
        return KeplerInput(uniform, 4.0), _cfg(
            [_heater(min_spacing_hours=1.5)],
            water_block_start_penalty_sek=0.0,
            water_block_penalty_sek=0.0,
        )
    if name == "mixed_spacing_0":
        return KeplerInput(mixed, 4.0), _cfg([_heater()])
    if name == "mixed_spacing":
        return KeplerInput(mixed, 4.0), _cfg([_heater(min_spacing_hours=2.0)])
    if name == "mixed_hourly_blocks":
        return KeplerInput(mixed, 4.0), _cfg(
            [_heater(min_spacing_hours=2.0)], water_hourly_blocks=True
        )
    if name == "mixed_hourly_blocks_mid_hour_start":
        return KeplerInput(mid_hour, 4.0), _cfg(
            [_heater(min_spacing_hours=1.5)], water_hourly_blocks=True, max_block_hours=1.5
        )
    if name == "mixed_partial_first_slot":
        return KeplerInput(mixed, 4.0, first_slot_remaining_h=0.1), _cfg(
            [_heater(min_spacing_hours=1.0)], water_hourly_blocks=True
        )
    if name == "force_on_aligned":
        return KeplerInput(mixed, 4.0), _cfg(
            [_heater(min_spacing_hours=1.0, force_on_slots=[0, 1, 2, 3])],
            water_hourly_blocks=True,
        )
    if name == "force_on_misaligned":
        # Slots 2-5 straddle two hour groups: hourly blocks widen it to both hours.
        return KeplerInput(mixed, 4.0), _cfg(
            [_heater(min_spacing_hours=1.0, force_on_slots=[2, 3, 4, 5])],
            water_hourly_blocks=True,
        )
    if name == "force_on_no_hourly_blocks":
        return KeplerInput(uniform, 4.0), _cfg(
            [_heater(min_spacing_hours=1.0, force_on_slots=[9, 10])]
        )
    if name == "heated_today":
        return KeplerInput(mixed, 4.0), _cfg(
            [_heater(min_spacing_hours=2.0, heated_today_kwh=2.0)], water_hourly_blocks=True
        )
    if name == "boost":
        slots = _slots(16, 12, pv_peak=6.0)
        return KeplerInput(slots, 7.5), _cfg(
            [_heater(min_spacing_hours=2.0, absorb_cap_kwh_per_day=7.0)],
            water_hourly_blocks=True,
            excess_pv_sink="water_heater_boost",
            excess_pv_slots=[6 <= i < 16 for i in range(len(slots))],
            excess_pv_reward_sek_per_kwh=0.5,
            excess_pv_soc_threshold_percent=50.0,
        )
    if name == "anchor":
        return KeplerInput(mixed, 4.0), _cfg(
            [_heater(min_spacing_hours=2.0, anchor_on_slots=[12, 13, 14, 15])],
            water_hourly_blocks=True,
            water_anchor_bonus_sek_per_slot=0.2,
        )
    if name in ("spa_may_skip_day", "multi_heater"):
        heaters = [
            _heater("vvb", min_spacing_hours=2.0),
            _heater("spa", power_kw=1.8, min_kwh_per_day=3.6),  # spacing 0
        ]
        priorities = {
            "vvb": LoadPriority(base_wtp_sek_per_kwh=2.5, dynamic_percentile=50.0),
            # may_skip_day: start penalty AND reliability floor waived.
            "spa": LoadPriority(
                base_wtp_sek_per_kwh=1.4, dynamic_percentile=30.0, may_skip_day=True
            ),
        }
        if name == "multi_heater":
            heaters.append(
                _heater("vagn", power_kw=1.5, min_kwh_per_day=3.0, min_spacing_hours=5.0)
            )
            # Static WTP: start penalty waived too.
            priorities["vagn"] = LoadPriority(base_wtp_sek_per_kwh=1.6)
        return KeplerInput(mixed, 4.0), _cfg(
            heaters,
            water_hourly_blocks=True,
            load_priority_enabled=True,
            load_priorities=priorities,
        )
    raise KeyError(name)


CASES = [
    "uniform_spacing_0",
    "uniform_spacing_below_cap",
    "uniform_spacing_above_cap",
    "uniform_hourly_blocks",
    "uniform_hourly_blocks_window_inside_hour",
    "uniform_odd_block_hours",
    "uniform_no_penalties_spacing_only",
    "mixed_spacing_0",
    "mixed_spacing",
    "mixed_hourly_blocks",
    "mixed_hourly_blocks_mid_hour_start",
    "mixed_partial_first_slot",
    "force_on_aligned",
    "force_on_misaligned",
    "force_on_no_hourly_blocks",
    "heated_today",
    "boost",
    "anchor",
    "spa_may_skip_day",
    "multi_heater",
]


# -------------------------------------------------------------------- helpers


def _heat_var_name(heater_id: str, t: int) -> str:
    return f"water_heat_{heater_id}_{t}"


def _solve(
    monkeypatch: pytest.MonkeyPatch,
    ki: KeplerInput,
    cfg: KeplerConfig,
    flow: bool,
    fix: dict[str, int] | None = None,
) -> tuple[Any, Any]:
    """Solve through KeplerSolver; optionally pin water_heat variables by name.

    Returns (KeplerResult, the LpProblem kepler built).
    """
    captured: dict[str, Any] = {}

    def _spy(self: Any, solver: Any = None, **kw: Any) -> Any:
        captured["prob"] = self
        if fix:
            for var in self.variables():
                if var.name in fix:
                    var.lowBound = var.upBound = fix[var.name]
        return _REAL_SOLVE(self, solver, **kw)

    monkeypatch.setattr(pulp.LpProblem, "solve", _spy)
    cfg.water_flow_formulation = flow
    try:
        result = KeplerSolver().solve(ki, cfg)
    finally:
        monkeypatch.setattr(pulp.LpProblem, "solve", _REAL_SOLVE)
    return result, captured["prob"]


def _pattern(prob: Any) -> dict[str, int]:
    """The solved water_heat values, by variable name (boost is NOT heat)."""
    return {
        v.name: round(v.varValue or 0.0)
        for v in prob.variables()
        if v.name.startswith("water_heat_")
    }


# ------------------------------------------------------- solve-level equality


@pytest.mark.parametrize("case", CASES)
def test_flow_and_default_formulations_agree(case: str, monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _case(case)
    off, prob_off = _solve(monkeypatch, ki, cfg, flow=False)
    on, prob_on = _solve(monkeypatch, ki, cfg, flow=True)
    assert off.is_optimal and on.is_optimal
    assert on.objective_cost_sek == pytest.approx(off.objective_cost_sek, abs=1e-5)

    # Exactness: the pattern each formulation chose costs the same in the other.
    pattern_off, pattern_on = _pattern(prob_off), _pattern(prob_on)
    assert sum(pattern_on.values()) > 0
    on_in_off, _ = _solve(monkeypatch, ki, cfg, flow=False, fix=pattern_on)
    assert on_in_off.objective_cost_sek == pytest.approx(on.objective_cost_sek, abs=1e-6)
    off_in_on, _ = _solve(monkeypatch, ki, cfg, flow=True, fix=pattern_off)
    assert off_in_on.objective_cost_sek == pytest.approx(off.objective_cost_sek, abs=1e-6)


def test_cases_exercise_the_rules():
    """Guard the fixtures: the spread really contains blocks, restarts and overshoot."""
    ki, cfg = _case("uniform_spacing_below_cap")
    cfg.water_flow_formulation = True
    res = KeplerSolver().solve(ki, cfg)
    on = [rs.water_heater_results.get("vvb", 0.0) > 0.0 for rs in res.slots]
    assert sum(on) >= 6
    assert sum(1 for a, b in itertools.pairwise([False, *on]) if b and not a) >= 1


# ---------------------------------------------------------------- brute force


def _tiny(
    grid: str, hourly: bool, spacing: float, block_h: float
) -> tuple[KeplerInput, KeplerConfig]:
    # "mixed": 4 x 15 min from hh:30 then 4 x 1 h, so the first hour group is
    # partial and the first coarse slot shares a clock hour with fine slots.
    if grid == "uniform":
        slots = _slots(20 if hourly else 8)  # 5 hour groups / 8 slots
    else:
        slots = _slots(4, 4, start=BASE + timedelta(minutes=30))
    cfg = _cfg(
        [_heater(power_kw=2.0, min_kwh_per_day=2.0, min_spacing_hours=spacing)],
        max_block_hours=block_h,
        water_hourly_blocks=hourly,
        capacity_kwh=2.0,
        max_charge_power_kw=2.0,
        max_discharge_power_kw=2.0,
    )
    return KeplerInput(slots, 1.0), cfg


def _units(ki: KeplerInput, hourly: bool) -> list[list[int]]:
    units: list[list[int]] = []
    for t, s in enumerate(ki.slots):
        prev = ki.slots[t - 1] if t else None
        if hourly and prev is not None and prev.start_time.hour == s.start_time.hour:
            units[-1].append(t)
        else:
            units.append([t])
    return units


def _resolve_fixed(prob: Any, pattern: dict[str, int]) -> float | None:
    """Objective of a captured model with water_heat pinned; None if infeasible."""
    for var in prob.variables():
        if var.name in pattern:
            var.lowBound = var.upBound = pattern[var.name]
    _REAL_SOLVE(prob, pulp.HiGHS(msg=False, gapRel=TIGHT_GAP))  # type: ignore[reportUnknownMemberType]
    if prob.status != pulp.LpStatusOptimal:
        return None
    return float(pulp.value(prob.objective))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("grid", "hourly", "spacing", "block_h", "spacing_bites"),
    [
        ("uniform", False, 0.0, 0.5, False),  # spacing 0 (the spa shape)
        ("uniform", False, 0.5, 0.75, True),  # spacing < cap
        ("uniform", False, 0.5, 0.5, True),  # spacing == cap
        ("uniform", False, 1.0, 0.25, True),  # spacing > cap, cap of one slot
        ("uniform", False, 0.75, 0.6, True),  # non-divisible block hours
        ("uniform", True, 1.0, 1.0, False),  # hour groups; one-unit lookback is vacuous
        ("uniform", True, 2.0, 1.0, True),
        ("uniform", True, 0.0, 0.5, False),  # windows inside one hour group
        ("mixed", False, 0.0, 0.5, False),
        ("mixed", False, 0.75, 1.0, True),  # lookback and window straddle fine/coarse
        ("mixed", False, 2.0, 1.5, True),
        ("mixed", True, 1.5, 1.0, True),  # partial first hour group + coarse tail
        ("mixed", True, 2.0, 2.0, True),
    ],
)
def test_every_pattern_costs_the_same(
    grid: str,
    hourly: bool,
    spacing: float,
    block_h: float,
    spacing_bites: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Enumerate every on/off pattern of a tiny horizon in both models."""
    ki, cfg = _tiny(grid, hourly, spacing, block_h)
    _, prob_off = _solve(monkeypatch, ki, cfg, flow=False)
    _, prob_on = _solve(monkeypatch, ki, cfg, flow=True)
    units = _units(ki, hourly)
    feasible = infeasible = 0
    costs: set[float] = set()
    for bits in itertools.product((0, 1), repeat=len(units)):
        pattern = {
            _heat_var_name("vvb", t): bit
            for unit, bit in zip(units, bits, strict=True)
            for t in unit
        }
        off = _resolve_fixed(prob_off, pattern)
        on = _resolve_fixed(prob_on, pattern)
        assert (off is None) == (on is None), f"feasibility differs for {bits}"
        if off is None or on is None:
            infeasible += 1
            continue
        feasible += 1
        costs.add(round(off, 3))
        assert on == pytest.approx(off, abs=1e-6), f"cost differs for {bits}"
    assert feasible > 1
    assert len(costs) > 3  # the patterns really are priced differently
    assert (infeasible > 0) == spacing_bites  # and the spacing rule really forbids some


# ------------------------------------------------------------ model structure

# (variables, constraints) of the default-formulation model, captured on the
# commit BEFORE the flag existed (e2113a06). The flag must not move them.
PRE_FLAG_MODEL_SIZE = {
    "uniform_spacing_above_cap": (484, 352),
    "mixed_hourly_blocks": (424, 323),
    "boost": (463, 383),
    "multi_heater": (597, 492),
}


@pytest.mark.parametrize("case", sorted(PRE_FLAG_MODEL_SIZE))
def test_flag_off_model_is_unchanged(case: str, monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _case(case)
    _, prob = _solve(monkeypatch, ki, cfg, flow=False)
    assert (len(prob.variables()), len(prob.constraints)) == PRE_FLAG_MODEL_SIZE[case]
    assert not [v.name for v in prob.variables() if v.name.startswith("wflow_")]


@pytest.mark.parametrize("case", ["mixed_hourly_blocks", "boost", "multi_heater"])
def test_flow_model_keeps_only_heat_binaries(case: str, monkeypatch: pytest.MonkeyPatch):
    ki, cfg = _case(case)
    _, off = _solve(monkeypatch, ki, cfg, flow=False)
    _, on = _solve(monkeypatch, ki, cfg, flow=True)
    names_on = [v.name for v in on.variables()]
    assert not [n for n in names_on if n.startswith(("water_start_", "block_overshoot_"))]
    flows = [v for v in on.variables() if v.name.startswith("wflow_")]
    assert flows and all(v.cat == pulp.LpContinuous for v in flows)

    def binaries(prob: Any) -> int:
        return sum(1 for v in prob.variables() if v.cat == pulp.LpInteger)

    T = len(ki.slots)
    with_start = sum(
        1
        for h in cfg.water_heaters
        if h.min_spacing_hours > 0 or cfg.water_block_start_penalty_sek > 0
    )
    assert binaries(off) - binaries(on) == with_start * T


def test_flow_does_nothing_for_a_heater_without_rules(monkeypatch: pytest.MonkeyPatch):
    """Spacing 0 and no start/block penalties: there is nothing to model."""
    ki, cfg = _case("uniform_spacing_0")
    cfg.water_block_start_penalty_sek = 0.0
    cfg.water_block_penalty_sek = 0.0
    off, _ = _solve(monkeypatch, ki, cfg, flow=False)
    on, prob = _solve(monkeypatch, ki, cfg, flow=True)
    assert not [v.name for v in prob.variables() if v.name.startswith("wflow_")]
    assert on.objective_cost_sek == pytest.approx(off.objective_cost_sek, abs=1e-6)


def test_spacing_infeasible_against_force_on_fails_the_same_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    """Two forced blocks closer than the spacing are infeasible in both models,
    and the flow path still leaves a replayable dump."""
    from planner.errors import PlannerError, PlannerErrorCode

    monkeypatch.setenv("DARKSTAR_SOLVER_DUMP_DIR", str(tmp_path / "dumps"))
    for flow in (False, True):
        ki, cfg = _case("uniform_spacing_above_cap")
        # A shared 3 kW feed: the second heater's forced slot 4 switches the first
        # one off between its own forced slots 2 and 6 — a restart inside 2 h.
        cfg.water_heaters[0].force_on_slots = [2, 6]
        cfg.water_heaters.append(_heater("other", min_kwh_per_day=0.0, force_on_slots=[4]))
        cfg.load_groups = [LoadGroup(id="feed", max_power_kw=3.0, members=["vvb", "other"])]
        cfg.water_flow_formulation = flow
        with pytest.raises(PlannerError) as exc:
            KeplerSolver().solve(ki, cfg)
        assert exc.value.code == PlannerErrorCode.SOLVER_INFEASIBLE
        assert exc.value.details["dump_path"]
    assert len(list((tmp_path / "dumps").glob("kepler_fail_*.json"))) >= 1


# -------------------------------------------------------------------- adapter


def _planner_config(**water_heating: Any) -> dict[str, Any]:
    return {
        "water_heating": dict(water_heating),
        "system": {
            "battery": {
                "capacity_kwh": 10.0,
                "max_charge_a": 100.0,
                "max_discharge_a": 100.0,
                "nominal_voltage_v": 48.0,
            },
        },
        "executor": {"inverter": {"control_unit": "A"}},
    }


def test_adapter_flag_defaults_off_and_follows_config():
    assert config_to_kepler_config(_planner_config()).water_flow_formulation is False
    assert (
        config_to_kepler_config(_planner_config(flow_formulation=True)).water_flow_formulation
        is True
    )
    assert (
        KeplerConfig(
            capacity_kwh=10.0,
            min_soc_percent=10.0,
            max_soc_percent=100.0,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            wear_cost_sek_per_kwh=0.0,
        ).water_flow_formulation
        is False
    )


def test_config_default_ships_the_flag_off_and_a_save_keeps_both_keys():
    """config.default.yaml carries flow_formulation: false, and the config-save
    merge (template + user values) keeps a user's opt-in — including
    kepler.solver.dump_all_instances, which lives under a commented-out block."""
    from pathlib import Path

    import yaml

    from backend.config_migration import template_aware_merge

    template = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config.default.yaml").read_text()
    )
    assert template["water_heating"]["flow_formulation"] is False
    assert config_to_kepler_config(template).water_flow_formulation is False

    user = {
        "water_heating": {"flow_formulation": True},
        "kepler": {"solver": {"dump_all_instances": 20}},
    }
    template_aware_merge(template, user)
    assert template["water_heating"]["flow_formulation"] is True
    assert template["kepler"]["solver"]["dump_all_instances"] == 20
    merged = config_to_kepler_config(template)
    assert merged.water_flow_formulation is True
    assert merged.solver_dump_all_instances == 20
