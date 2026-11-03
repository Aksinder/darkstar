"""DC-coupled PV charging efficiency (battery.pv_charge_efficiency).

PV on the hybrid inverter's own DC strings is stored DC->DC, skipping the inverter's
AC->DC stage, so it charges the battery more efficiently than grid/AC charging.
Kepler (and the battery-only fallback) split each slot's charge into a DC-PV part
``charge_pv`` stored at pv_charge_efficiency and the rest at charge_efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pulp  # type: ignore[import]
import pytest

from planner.coarse_tail import expand_result
from planner.plan_validator import validate_plan
from planner.solver.adapter import config_to_kepler_config, parse_pv_charge_efficiency
from planner.solver.battery_lp import solve_battery_only
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    KeplerResult,
    KeplerResultSlot,
)

START = datetime(2026, 6, 15, 0, 0)
CAP = 16.0
ETA_C = 0.92
ETA_D = 0.95
ETA_PV = 0.985
HYBRID = 0.43
TOL = 1e-4


def _config(**kw: Any) -> KeplerConfig:
    base: dict[str, Any] = {
        "capacity_kwh": CAP,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "max_charge_power_kw": 10.0,
        "max_discharge_power_kw": 10.0,
        "charge_efficiency": ETA_C,
        "discharge_efficiency": ETA_D,
        "wear_cost_sek_per_kwh": 0.1,
        "max_export_power_kw": 16.0,
        "max_import_power_kw": 16.0,
        "hybrid_pv_fraction": HYBRID,
        "enable_export": True,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _sunny_day(
    n: int = 96, *, export_factor: float = 0.6, negative_midday: bool = False
) -> list[KeplerInputSlot]:
    """Cheap night, expensive evening peak, strong midday PV, export price > 0."""
    out: list[KeplerInputSlot] = []
    for i in range(n):
        t = START + timedelta(minutes=15 * i)
        hr = t.hour + t.minute / 60
        price = 0.3 if hr < 5 else (3.0 if 17 <= hr < 21 else 1.0)
        pv = max(0.0, 3.0 * (1 - abs(hr - 13) / 6)) if 7 <= hr <= 19 else 0.0
        export = price * export_factor
        if negative_midday and 10 <= hr < 15:
            export = -0.5
            price = 0.05
        out.append(
            KeplerInputSlot(
                start_time=t,
                end_time=t + timedelta(minutes=15),
                load_kwh=0.3 + (0.5 if 17 <= hr < 22 else 0.0),
                pv_kwh=pv,
                import_price_sek_kwh=price,
                export_price_sek_kwh=export,
            )
        )
    return out


def _solve(slots: list[KeplerInputSlot], config: KeplerConfig, soc0: float = 3.0) -> KeplerResult:
    res = KeplerSolver().solve(KeplerInput(slots=slots, initial_soc_kwh=soc0), config)
    assert res.is_optimal
    return res


def _curtailment(s: KeplerInputSlot, r: KeplerResultSlot) -> float:
    """Balance residual = curtailment (no water/EV/sinks in these fixtures)."""
    return (
        s.pv_kwh
        + r.discharge_kwh
        + r.grid_import_kwh
        - (s.load_kwh + r.charge_kwh + r.grid_export_kwh)
    )


def _var_names(monkeypatch: pytest.MonkeyPatch, slots: list[KeplerInputSlot], cfg: KeplerConfig):
    names: list[set[str]] = []
    orig = pulp.LpProblem.solve

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        names.append({v.name for v in self.variables()})
        return orig(self, *a, **k)

    monkeypatch.setattr(pulp.LpProblem, "solve", spy)
    _solve(slots, cfg)
    monkeypatch.setattr(pulp.LpProblem, "solve", orig)
    assert names
    return names[0]


# ------------------------------------------------------------------ regression guard


def test_unset_or_equal_efficiency_keeps_model_identical(monkeypatch: pytest.MonkeyPatch):
    slots = _sunny_day()
    base = _var_names(monkeypatch, slots, _config())
    same = _var_names(monkeypatch, slots, _config(pv_charge_efficiency=ETA_C))
    split = _var_names(monkeypatch, slots, _config(pv_charge_efficiency=ETA_PV))
    assert base == same
    assert not any(n.startswith("charge_pv") for n in base)
    assert len(split - base) == len(slots)
    assert all(n.startswith("charge_pv_kwh") for n in split - base)

    legacy = _solve(slots, _config())
    equal = _solve(slots, _config(pv_charge_efficiency=ETA_C))
    assert legacy.objective_cost_sek == pytest.approx(equal.objective_cost_sek, abs=1e-6)
    for a, b in zip(legacy.slots, equal.slots, strict=True):
        assert a.charge_pv_kwh == 0.0 and b.charge_pv_kwh == 0.0
        assert a.soc_kwh == pytest.approx(b.soc_kwh, abs=1e-6)
    assert validate_plan(legacy, KeplerInput(slots=slots, initial_soc_kwh=3.0), _config()).ok


# ------------------------------------------------------------------ solver behaviour


def test_dc_pv_is_stored_more_efficiently_and_soc_follows_split():
    slots = _sunny_day()
    cfg = _config(pv_charge_efficiency=ETA_PV)
    legacy = _solve(slots, _config())
    split = _solve(slots, cfg)

    # charge_pv = 0 reproduces the legacy plan, so the split can never be worse.
    assert split.objective_cost_sek is not None and legacy.objective_cost_sek is not None
    assert split.objective_cost_sek <= legacy.objective_cost_sek + 1e-6
    assert sum(r.charge_pv_kwh for r in split.slots) > 1.0

    # Stored PV energy is at least what the legacy plan stored from PV hours.
    def stored_from_pv_hours(res: KeplerResult, eta_pv: float) -> float:
        return sum(
            (r.charge_kwh - r.charge_pv_kwh) * ETA_C + r.charge_pv_kwh * eta_pv
            for s, r in zip(slots, res.slots, strict=True)
            if s.pv_kwh > 0
        )

    assert stored_from_pv_hours(split, ETA_PV) >= stored_from_pv_hours(legacy, ETA_C) - 1e-6

    soc = 3.0
    for r in split.slots:
        soc += (r.charge_kwh - r.charge_pv_kwh) * ETA_C + r.charge_pv_kwh * ETA_PV
        soc -= r.discharge_kwh / ETA_D
        assert r.soc_kwh == pytest.approx(soc, abs=1e-4)
    assert validate_plan(split, KeplerInput(slots=slots, initial_soc_kwh=3.0), cfg).ok


def test_charge_pv_bounded_by_hybrid_share_and_total_charge():
    slots = _sunny_day()
    split = _solve(slots, _config(pv_charge_efficiency=ETA_PV))
    night_grid_charge = 0
    for s, r in zip(slots, split.slots, strict=True):
        assert r.charge_pv_kwh >= 0.0
        assert r.charge_pv_kwh <= r.charge_kwh + TOL
        assert r.charge_pv_kwh <= s.pv_kwh * HYBRID + TOL
        if s.pv_kwh == 0.0 and r.charge_kwh > 0.01:
            night_grid_charge += 1
            assert r.charge_pv_kwh == pytest.approx(0.0, abs=TOL)
    # The fixture must actually exercise a night grid charge for that check to bite.
    assert night_grid_charge > 0
    # And the hybrid share must actually bind somewhere (charge > hybrid PV).
    assert any(
        r.charge_kwh > s.pv_kwh * HYBRID + 0.05
        and r.charge_pv_kwh == pytest.approx(s.pv_kwh * HYBRID, abs=1e-3)
        for s, r in zip(slots, split.slots, strict=True)
    )


def test_curtailed_pv_is_never_also_credited_as_dc_charge():
    slots = _sunny_day(negative_midday=True)
    cfg = _config(pv_charge_efficiency=ETA_PV, max_charge_power_kw=3.0)
    res = _solve(slots, cfg, soc0=10.0)
    curtailed_any = False
    for s, r in zip(slots, res.slots, strict=True):
        curt = _curtailment(s, r)
        if curt > 0.01:
            curtailed_any = True
        # Curtailed PV is never also stored: the DC credit is capped at the hybrid
        # share of the PV that is left after curtailment (pro rata).
        assert r.charge_pv_kwh <= HYBRID * (s.pv_kwh - curt) + 1e-3
        assert r.charge_pv_kwh + curt <= s.pv_kwh + 1e-3
    assert curtailed_any, "fixture must curtail for the check to bite"
    assert validate_plan(res, KeplerInput(slots=slots, initial_soc_kwh=10.0), cfg).ok


# ------------------------------------------------------------------ validator


def _split_plan() -> tuple[KeplerResult, KeplerInput, KeplerConfig]:
    slots = _sunny_day()
    cfg = _config(pv_charge_efficiency=ETA_PV)
    inp = KeplerInput(slots=slots, initial_soc_kwh=3.0)
    return _solve(slots, cfg), inp, cfg


def _tamper(res: KeplerResult, idx: int, **kw: Any) -> KeplerResult:
    slots = list(res.slots)
    slots[idx] = replace(slots[idx], **kw)
    return replace(res, slots=slots)


def test_validator_catches_tampered_dc_pv_charge():
    res, inp, cfg = _split_plan()
    assert validate_plan(res, inp, cfg).ok
    i = next(k for k, r in enumerate(res.slots) if r.charge_pv_kwh > 0.05)
    r = res.slots[i]
    pv = inp.slots[i].pv_kwh

    over_charge = validate_plan(_tamper(res, i, charge_pv_kwh=r.charge_kwh + 0.5), inp, cfg)
    assert not over_charge.ok
    assert any("total charge" in e for e in over_charge.errors)

    night = next(k for k, s in enumerate(inp.slots) if s.pv_kwh == 0.0)
    over_pv = validate_plan(
        _tamper(res, night, charge_kwh=res.slots[night].charge_kwh + 0.2, charge_pv_kwh=0.2),
        inp,
        cfg,
    )
    assert not over_pv.ok
    assert any("hybrid-inverter PV" in e for e in over_pv.errors)

    above_hybrid = validate_plan(
        _tamper(res, i, charge_pv_kwh=min(r.charge_kwh, pv * HYBRID + 0.3)), inp, cfg
    )
    if r.charge_kwh > pv * HYBRID + 0.01:
        assert not above_hybrid.ok

    # Curtailed PV credited as stored: extra import in a DC-charging slot raises the
    # balance residual (= implied curtailment) until charge_pv + curtailment > PV.
    j = next(
        k
        for k, (s, r) in enumerate(zip(inp.slots, res.slots, strict=True))
        if r.charge_pv_kwh > 0.05 and s.pv_kwh > 0
    )
    rj = res.slots[j]
    sj = inp.slots[j]
    extra = sj.pv_kwh - rj.charge_pv_kwh * 0.5 - max(0.0, _curtailment(sj, rj))
    double_use = validate_plan(
        _tamper(res, j, grid_import_kwh=rj.grid_import_kwh + extra), inp, cfg
    )
    assert not double_use.ok
    assert any("curtailed PV credited" in e for e in double_use.errors), double_use.errors

    negative = validate_plan(_tamper(res, i, charge_pv_kwh=-0.1), inp, cfg)
    assert not negative.ok
    assert any("negative DC PV" in e for e in negative.errors)

    # Shifting energy between the two paths without fixing SoC breaks the replay.
    relabel = validate_plan(_tamper(res, i, charge_pv_kwh=0.0), inp, cfg)
    assert not relabel.ok
    assert any("does not follow flows" in e for e in relabel.errors)

    # A split plan checked under the legacy single efficiency must fail the replay:
    # the validator really uses the split, it doesn't just ignore the field.
    legacy_view = validate_plan(res, inp, replace(cfg, pv_charge_efficiency=None))
    assert not legacy_view.ok


# ------------------------------------------------------------------ fallback parity


def test_battery_lp_fallback_models_the_same_split():
    slots = _sunny_day()
    cfg = _config(pv_charge_efficiency=ETA_PV)
    inp = KeplerInput(slots=slots, initial_soc_kwh=3.0)
    legacy = solve_battery_only(inp, _config(), mode="live")
    split = solve_battery_only(inp, cfg, mode="live")

    assert all(r.charge_pv_kwh == 0.0 for r in legacy.slots)
    assert sum(r.charge_pv_kwh for r in split.slots) > 1.0
    assert split.objective_cost_sek is not None and legacy.objective_cost_sek is not None
    assert split.objective_cost_sek <= legacy.objective_cost_sek + 1e-6
    for s, r in zip(slots, split.slots, strict=True):
        assert r.charge_pv_kwh <= r.charge_kwh + TOL
        assert r.charge_pv_kwh <= s.pv_kwh * HYBRID + TOL
    report = validate_plan(split, inp, cfg)
    assert report.ok, report.errors
    assert validate_plan(legacy, inp, _config()).ok


# ------------------------------------------------------------------ coarse tail


def test_coarse_split_divides_dc_pv_charge_with_charge():
    slots = _sunny_day(8)
    coarse = KeplerResultSlot(
        start_time=slots[0].start_time,
        end_time=slots[3].end_time,
        charge_kwh=4.0,
        charge_pv_kwh=1.6,
        discharge_kwh=0.0,
        grid_import_kwh=2.4,
        grid_export_kwh=0.0,
        soc_kwh=8.0,
        cost_sek=0.0,
    )
    result = KeplerResult(slots=[coarse], total_cost_sek=0.0, is_optimal=True, status_msg="ok")
    out = expand_result(result, [[0, 1, 2, 3]], slots[:4])
    assert [s.charge_pv_kwh for s in out.slots] == pytest.approx([0.4] * 4)
    assert [s.charge_kwh for s in out.slots] == pytest.approx([1.0] * 4)


# ------------------------------------------------------------------ adapter


@pytest.mark.parametrize(
    "raw", [0, 0.0, -0.5, 1.01, 2, float("nan"), "high", "", "1.5", [0.9], True, False]
)
def test_adapter_rejects_invalid_values_with_warning(raw: Any, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        assert parse_pv_charge_efficiency(raw) is None
    assert any("pv_charge_efficiency" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", [0.97, 1, 1.0, "0.97", " 0.96 "])
def test_adapter_accepts_valid_values(raw: Any):
    assert parse_pv_charge_efficiency(raw) == pytest.approx(float(raw))


def _planner_config(**battery: Any) -> dict[str, Any]:
    return {
        "system": {
            "solar_arrays": [
                {"name": "sungrow", "kwp": 4.3, "on_battery_inverter": True},
                {"name": "fronius", "kwp": 5.7},
            ]
        },
        "battery": {
            "capacity_kwh": 16.0,
            "max_charge_a": 100,
            "max_discharge_a": 100,
            "charge_efficiency": 0.92,
            "discharge_efficiency": 0.95,
            **battery,
        },
    }


def test_adapter_passes_valid_value_through():
    cfg = config_to_kepler_config(_planner_config(pv_charge_efficiency=0.96))
    assert cfg.pv_charge_efficiency == pytest.approx(0.96)
    assert cfg.hybrid_pv_fraction == pytest.approx(0.43)


def test_adapter_unset_and_invalid_mean_none(caplog: pytest.LogCaptureFixture):
    assert config_to_kepler_config(_planner_config()).pv_charge_efficiency is None
    with caplog.at_level(logging.WARNING):
        cfg = config_to_kepler_config(_planner_config(pv_charge_efficiency=1.5))
    assert cfg.pv_charge_efficiency is None
    assert any("pv_charge_efficiency" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ edge cases


def test_partial_first_slot_scales_the_dc_pv_bound():
    from planner.solver.adapter import scale_first_slot_to_remaining

    slots = _sunny_day(export_factor=0.2)[60:]  # 15:00: waning PV, store it all now
    cfg = _config(pv_charge_efficiency=ETA_PV)
    inp = scale_first_slot_to_remaining(
        KeplerInput(slots=slots, initial_soc_kwh=3.0, first_slot_remaining_h=0.1)
    )
    assert inp.slots[0].pv_kwh == pytest.approx(slots[0].pv_kwh * 0.4)
    res = KeplerSolver().solve(inp, cfg)
    assert res.is_optimal
    r0 = res.slots[0]
    assert r0.charge_kwh <= cfg.max_charge_power_kw * 0.1 + TOL
    assert r0.charge_pv_kwh > 0.0
    assert r0.charge_pv_kwh <= inp.slots[0].pv_kwh * HYBRID + TOL
    assert r0.charge_pv_kwh <= r0.charge_kwh + TOL
    report = validate_plan(res, inp, cfg)
    assert report.ok, report.errors


def test_no_hybrid_fraction_lets_dc_charge_use_all_pv():
    slots = _sunny_day(negative_midday=True)
    cfg = _config(pv_charge_efficiency=ETA_PV, hybrid_pv_fraction=None, max_charge_power_kw=3.0)
    res = _solve(slots, cfg, soc0=10.0)
    for s, r in zip(slots, res.slots, strict=True):
        curt = _curtailment(s, r)
        assert r.charge_pv_kwh <= r.charge_kwh + TOL
        assert r.charge_pv_kwh <= s.pv_kwh - curt + 1e-3
    # With frac = 1.0 the whole charge can be DC PV in a sunny slot (above 0.43 share).
    assert any(
        r.charge_pv_kwh > s.pv_kwh * HYBRID + 0.05
        for s, r in zip(slots, res.slots, strict=True)
        if s.pv_kwh > 0
    )
    report = validate_plan(res, KeplerInput(slots=slots, initial_soc_kwh=10.0), cfg)
    assert report.ok, report.errors


def test_lower_pv_efficiency_warns_and_matches_legacy(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        adapted = config_to_kepler_config(_planner_config(pv_charge_efficiency=0.90))
    assert adapted.pv_charge_efficiency == pytest.approx(0.90)
    assert any("below charge_efficiency" in r.getMessage() for r in caplog.records)

    slots = _sunny_day()
    cfg = _config(pv_charge_efficiency=0.90)
    inp = KeplerInput(slots=slots, initial_soc_kwh=3.0)
    legacy = _solve(slots, _config())
    lower = _solve(slots, cfg)
    assert lower.objective_cost_sek is not None and legacy.objective_cost_sek is not None
    # The DC path is optional: a worse one is simply not used.
    assert lower.objective_cost_sek == pytest.approx(legacy.objective_cost_sek, abs=1e-4)
    report = validate_plan(lower, inp, cfg)
    assert report.ok, report.errors
