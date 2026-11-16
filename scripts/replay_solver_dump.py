#!/usr/bin/env python3
"""Replay a solver dump from data/solver_dumps/ locally.

The dump writer (planner/solver/kepler.py::_dump_failed_solve_instance) was built
for exactly this — "persisted for offline replay" — but no reader existed until
2026-08-26, when five production instances failed in one night and the box spent
78 minutes producing one accepted plan. This harness rehydrates a dump back into
KeplerConfig + KeplerInput and re-solves it on the local machine (~15x faster
than the production VM: --time-limit 16 approximates the box's 240 s budget).

Usage:
    replay_solver_dump.py DUMP.json [DUMP2.json ...]
        [--solver highs|cbc|glpk] [--time-limit N] [--gap G] [--threads N]
        [--highs-opt key=val ...] [--mutate NAME[:ARG] ...]
        [--sweep] [--repeat K] [--seed N] [--json]

Dumps: kepler_fail_*.json (every failed solve) and kepler_ok_*.json (successful
solves, collected when kepler.solver.dump_all_instances > 0). Pull them off the
box with GET /api/debug/solver_dumps and /api/debug/solver_dumps/<name>.

Mutations: no_hourly_blocks | no_anchor | no_spacing | no_absorb_cap
           | flow_water (water_heating.flow_formulation on; flow_water:off forces it off)
           | truncate:H (drop slots beyond H hours)
           | retail:H  (coarsen the tail beyond H hours to whole clock hours)
           | synth_bathing[:HEATER_ID]  (attach a synthetic spa BathingSpec; default
             heater "spa", appended at 1.8 kW when absent: T0 30, ambient 10, C 0.9,
             k 0.021, ready 38, floor 28, max 40, windows 19:00-21:00 on the next 3
             evenings after slot 0, ready_by = start - 30 min)
           | no_bathing (drop every heater's BathingSpec: the legacy baseline)
           | bath_set:FIELD=V[,FIELD=V]  (override BathingSpec fields after
             synth_bathing; windows=N keeps the first N windows)
           | heater_floor:ID=KWH (set a heater's daily floor, heated_today 0:
             the energy-matched legacy baseline for a bathing comparison)

Dumps written with a BathingSpec carry it as a nested dict (datetimes as str);
rehydrate_config rebuilds BathingSpec/BathWindow from it. Windows are
time-based, so truncate/retail need no index fixing for them.

Known limitation: dumps are captured AFTER coarse-tail coarsening
(planner/pipeline.py replaces kepler_input.slots before solve), so the original
15-minute tail slots are gone — a finer tail cannot be replayed, only a shorter
(truncate) or coarser (retail) one.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.resolve()))

import pulp

import planner.solver.kepler as kepler_mod
from planner.errors import PlannerError
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    BathingSpec,
    BathWindow,
    DeferrableLoadInput,
    EVChargerInput,
    EVDepartureTarget,
    ExcessPVSinkSpec,
    IncentiveBucket,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    LoadGroup,
    LoadPriority,
    WaterHeaterInput,
)

# ---------------------------------------------------------------- rehydration


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Dataclass constructor that tolerates schema drift between dump and code."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(data) - names)
    if dropped:
        print(f"  note: {cls.__name__} dropping unknown dump fields: {dropped}")
    return cls(**{k: v for k, v in data.items() if k in names})


def _dt(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def rehydrate_input(inp: dict[str, Any]) -> KeplerInput:
    slots = [
        _build(
            KeplerInputSlot,
            {**s, "start_time": _dt(s["start_time"]), "end_time": _dt(s["end_time"])},
        )
        for s in inp["slots"]
    ]
    rem = inp.get("first_slot_remaining_h")
    return KeplerInput(
        slots=slots,
        initial_soc_kwh=float(inp["initial_soc_kwh"]),
        # Partial first slot (kepler.partial_first_slot): dumps carry the remainder
        # and the already-scaled slot 0, so replay solves the same instance.
        first_slot_remaining_h=float(rem) if rem is not None else None,
    )


def _rehydrate_bathing(b: Any) -> BathingSpec | None:
    """Nested dump dict -> BathingSpec (windows as BathWindow, datetimes parsed)."""
    if b is None or isinstance(b, BathingSpec):
        return b
    windows = [
        _build(
            BathWindow,
            {
                **w,
                "start": _dt(w.get("start")),
                "end": _dt(w.get("end")),
                "ready_by": _dt(w.get("ready_by")),
            },
        )
        for w in b.get("windows") or []
    ]
    return _build(BathingSpec, {**b, "windows": windows})


def rehydrate_config(cfg: dict[str, Any]) -> KeplerConfig:
    cfg = dict(cfg)
    cfg["water_heaters"] = [
        _build(
            WaterHeaterInput,
            {**w, "bathing": _rehydrate_bathing(w.get("bathing"))} if "bathing" in w else w,
        )
        for w in cfg.get("water_heaters") or []
    ]
    cfg["ev_chargers"] = [
        _build(
            EVChargerInput,
            {
                **e,
                "deadline": _dt(e.get("deadline")),
                "incentive_buckets": [
                    _build(IncentiveBucket, b) for b in e.get("incentive_buckets") or []
                ],
                "departure_targets": [
                    _build(EVDepartureTarget, {**t, "deadline": _dt(t.get("deadline"))})
                    for t in e.get("departure_targets") or []
                ],
            },
        )
        for e in cfg.get("ev_chargers") or []
    ]
    cfg["excess_pv_sinks"] = [_build(ExcessPVSinkSpec, s) for s in cfg.get("excess_pv_sinks") or []]
    cfg["deferrable_loads"] = [
        _build(DeferrableLoadInput, d) for d in cfg.get("deferrable_loads") or []
    ]
    cfg["load_groups"] = [_build(LoadGroup, g) for g in cfg.get("load_groups") or []]
    cfg["load_priorities"] = {
        k: _build(LoadPriority, v) for k, v in (cfg.get("load_priorities") or {}).items()
    }
    # JSON stringifies dict[int, ...] keys; kepler looks hours up with int keys,
    # so string keys would silently disable the profile.
    if cfg.get("phase_load_profile"):
        cfg["phase_load_profile"] = {int(k): v for k, v in cfg["phase_load_profile"].items()}
    return _build(KeplerConfig, cfg)


# ------------------------------------------------------------------ mutations


def _mut_no_hourly_blocks(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    cfg.water_hourly_blocks = False
    return ki, cfg


def _mut_flow_water(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    """A/B the arc-flow water formulation (water_heating.flow_formulation)."""
    cfg.water_flow_formulation = (arg or "on").lower() not in ("off", "0", "false")
    return ki, cfg


def _mut_no_anchor(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    for h in cfg.water_heaters:
        h.anchor_on_slots = None
    return ki, cfg


def _mut_no_spacing(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    for h in cfg.water_heaters:
        h.min_spacing_hours = 0.0
    return ki, cfg


def _mut_no_absorb_cap(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    for h in cfg.water_heaters:
        h.absorb_cap_kwh_per_day = None
    return ki, cfg


def _mut_synth_bathing(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    """Attach a synthetic spa BathingSpec (replay check for spa bathing times)."""
    from datetime import timedelta

    hid = arg or "spa"
    heater = next((h for h in cfg.water_heaters if h.id == hid), None)
    if heater is None:
        heater = WaterHeaterInput(
            id=hid,
            power_kw=1.8,
            min_kwh_per_day=0.0,
            max_hours_between_heating=24.0,
            min_spacing_hours=0.0,
        )
        cfg.water_heaters.append(heater)
    t0 = ki.slots[0].start_time
    first = t0.replace(hour=19, minute=0, second=0, microsecond=0)
    if first <= t0:
        first = first + timedelta(days=1)
    windows: list[BathWindow] = []
    for day in range(3):
        start = first + timedelta(days=day)
        windows.append(
            BathWindow(
                start=start, end=start + timedelta(hours=2), ready_by=start - timedelta(minutes=30)
            )
        )
    heater.bathing = BathingSpec(
        initial_temp_c=30.0,
        ambient_c=10.0,
        heat_capacity_kwh_per_c=0.9,
        cooling_k_per_h=0.021,
        ready_temp_c=38.0,
        eco_floor_c=28.0,
        temp_max_c=40.0,
        windows=windows,
    )
    return ki, cfg


def _mut_bath_set(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    """Override BathingSpec fields on every bathing heater (apply after
    synth_bathing): bath_set:initial_temp_c=37,cooling_k_per_h=0.005. The pseudo
    field windows=N keeps only the first N windows (fair-comparison runs)."""
    for kv in (arg or "").split(","):
        if not kv:
            continue
        key, _, val = kv.partition("=")
        for h in cfg.water_heaters:
            if h.bathing is None:
                continue
            if key == "windows":
                h.bathing.windows = h.bathing.windows[: int(val)]
            else:
                setattr(h.bathing, key, float(val))
    return ki, cfg


def _mut_heater_floor(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    """Legacy fair-comparison: heater_floor:ID=KWH sets that heater's daily floor
    (and zeroes heated_today_kwh so the first bucket carries the full floor)."""
    hid, _, val = (arg or "").partition("=")
    for h in cfg.water_heaters:
        if h.id == hid:
            h.min_kwh_per_day = float(val)
            h.heated_today_kwh = 0.0
    return ki, cfg


def _mut_no_bathing(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    for h in cfg.water_heaters:
        h.bathing = None
    return ki, cfg


def _truncate_indices(ki: KeplerInput, cfg: KeplerConfig, keep: int):
    ki.slots = ki.slots[:keep]
    if cfg.excess_pv_slots:
        cfg.excess_pv_slots = cfg.excess_pv_slots[:keep]
    for h in cfg.water_heaters:
        for attr in ("force_on_slots", "anchor_on_slots"):
            v = getattr(h, attr)
            if v is not None:
                setattr(h, attr, [i for i in v if i < keep] or None)


def _mut_truncate(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    hours = float(arg or 24)
    t0 = ki.slots[0].start_time
    keep = sum(1 for s in ki.slots if (s.start_time - t0).total_seconds() / 3600.0 < hours)
    _truncate_indices(ki, cfg, max(keep, 1))
    return ki, cfg


def _mut_retail(ki: KeplerInput, cfg: KeplerConfig, arg: str | None):
    """Coarsen the tail beyond ARG hours into whole clock hours (mirror of
    planner/coarse_tail.py semantics: energies summed, prices averaged)."""
    hours = float(arg or 12)
    t0 = ki.slots[0].start_time
    head: list[KeplerInputSlot] = []
    tail: list[KeplerInputSlot] = []
    for s in ki.slots:
        (head if (s.start_time - t0).total_seconds() / 3600.0 < hours else tail).append(s)
    merged: list[KeplerInputSlot] = []
    flags_head = (cfg.excess_pv_slots or [])[: len(head)]
    flags_tail = (cfg.excess_pv_slots or [])[len(head) : len(head) + len(tail)]
    new_flags: list[bool] = list(flags_head)
    i = 0
    while i < len(tail):
        j = i
        group = [tail[i]]
        while (
            j + 1 < len(tail)
            and tail[j + 1].start_time.hour == tail[i].start_time.hour
            and tail[j + 1].start_time.date() == tail[i].start_time.date()
        ):
            j += 1
            group.append(tail[j])
        n = len(group)
        merged.append(
            KeplerInputSlot(
                start_time=group[0].start_time,
                end_time=group[-1].end_time,
                load_kwh=sum(g.load_kwh for g in group),
                pv_kwh=sum(g.pv_kwh for g in group),
                import_price_sek_kwh=sum(g.import_price_sek_kwh for g in group) / n,
                export_price_sek_kwh=sum(g.export_price_sek_kwh for g in group) / n,
            )
        )
        if flags_tail:
            new_flags.append(any(flags_tail[i : j + 1]))
        i = j + 1
    ki.slots = head + merged
    if cfg.excess_pv_slots:
        cfg.excess_pv_slots = new_flags
    for h in cfg.water_heaters:
        h.force_on_slots = None
        h.anchor_on_slots = None  # index maps are invalid after re-gridding
    return ki, cfg


MUTATIONS = {
    "no_hourly_blocks": _mut_no_hourly_blocks,
    "flow_water": _mut_flow_water,
    "no_anchor": _mut_no_anchor,
    "no_spacing": _mut_no_spacing,
    "no_absorb_cap": _mut_no_absorb_cap,
    "truncate": _mut_truncate,
    "retail": _mut_retail,
    "synth_bathing": _mut_synth_bathing,
    "no_bathing": _mut_no_bathing,
    "bath_set": _mut_bath_set,
    "heater_floor": _mut_heater_floor,
}

# ------------------------------------------------------------------- solving


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@contextlib.contextmanager
def _solver_overrides(args: argparse.Namespace):
    """Patch the module time budget and wrap pulp.HiGHS/PULP_CBC_CMD so kepler's
    own chain runs with our options. Restores everything on exit."""
    extra: dict[str, Any] = {}
    for kv in args.highs_opt or []:
        k, _, v = kv.partition("=")
        extra[k] = _coerce(v)
    if args.threads is not None:
        extra["threads"] = args.threads
    if args.seed is not None:
        extra["random_seed"] = args.seed

    real_highs = pulp.HiGHS
    real_cbc = pulp.PULP_CBC_CMD
    limit = args.time_limit

    def highs_factory(*a: Any, **kw: Any):
        if args.solver != "highs":
            raise pulp.PulpSolverError("replay: forced fallback past HiGHS")
        kw["timeLimit"] = limit
        kw["gapRel"] = args.gap
        return real_highs(*a, **kw, **extra)

    def cbc_factory(*a: Any, **kw: Any):
        if args.solver == "glpk":
            raise pulp.PulpSolverError("replay: forced fallback past CBC")
        kw["timeLimit"] = limit
        kw["gapRel"] = args.gap
        return real_cbc(*a, **kw)

    old_limit = kepler_mod.SOLVER_TIME_LIMIT_S
    kepler_mod.SOLVER_TIME_LIMIT_S = limit
    pulp.HiGHS = highs_factory
    pulp.PULP_CBC_CMD = cbc_factory
    try:
        yield
    finally:
        kepler_mod.SOLVER_TIME_LIMIT_S = old_limit
        pulp.HiGHS = real_highs
        pulp.PULP_CBC_CMD = real_cbc


def _floor_report(ki: KeplerInput, cfg: KeplerConfig, result: Any) -> list[str]:
    """Per heater: floor vs credited (avg-hours accounting) vs physical kWh."""
    lines = []
    hours = [(s.end_time - s.start_time).total_seconds() / 3600.0 for s in ki.slots]
    avg_h = sum(hours) / len(hours)
    for h in cfg.water_heaters:
        on = [
            t for t, rs in enumerate(result.slots) if rs.water_heater_results.get(h.id, 0.0) > 0.0
        ]
        credited = len(on) * h.power_kw * avg_h
        physical = sum(h.power_kw * hours[t] for t in on)
        # A heater kepler planned in bathing mode has no daily-kWh floor row.
        bathing = h.id in (getattr(result, "bathing_reports", None) or {})
        floor = "bathing" if bathing else f"floor {h.min_kwh_per_day:.2f}"
        lines.append(
            f"    {h.id}: {floor} | credited {credited:.2f}"
            f" | physical {physical:.2f} kWh ({len(on)} ON-slots)"
        )
    return lines


@contextlib.contextmanager
def _model_stats(stats: dict[str, int]):
    """Record the LP's size (and its integer-variable count: spa bathing must add
    none) at every prob.solve() kepler makes; the last attempt wins."""
    real_solve = pulp.LpProblem.solve

    def solve(self: Any, *a: Any, **kw: Any):
        variables = self.variables()
        stats["vars"] = len(variables)
        stats["int_vars"] = sum(1 for v in variables if v.cat == pulp.LpInteger)
        stats["constraints"] = len(self.constraints)
        return real_solve(self, *a, **kw)

    pulp.LpProblem.solve = solve  # type: ignore[method-assign]
    try:
        yield
    finally:
        pulp.LpProblem.solve = real_solve  # type: ignore[method-assign]


def _bathing_rows(result: Any) -> dict[str, Any]:
    """Per bathing heater: ready T, shortfall and unavoidable part per window, and
    the planned kWh (phantom-load check against temp_max)."""
    out: dict[str, Any] = {}
    for hid, rep in (getattr(result, "bathing_reports", None) or {}).items():
        kwh = sum(
            rs.water_heater_results.get(hid, 0.0)
            * (rs.duration_h or (rs.end_time - rs.start_time).total_seconds() / 3600.0)
            for rs in result.slots
        )
        out[hid] = {
            "ready_temp_c": rep.window_ready_temp_c,
            "shortfall_c": rep.window_shortfall_c,
            "unavoidable_c": rep.unavoidable_shortfall_c,
            "overshoot_c": rep.overshoot_c,
            "planned_kwh": round(kwh, 2),
        }
    return out


def run_replay(path: Path, args: argparse.Namespace, label: str = "") -> dict[str, Any]:
    dump = json.loads(path.read_text())
    ki = rehydrate_input(dump["input"])
    cfg = rehydrate_config(dump["config"])
    cfg.solver_dump_all_instances = 0  # a replay is not a new instance to collect
    for m in args.mutate or []:
        name, _, marg = m.partition(":")
        ki, cfg = MUTATIONS[name](ki, cfg, marg or None)

    row: dict[str, Any] = {
        "dump": path.name,
        "label": label or "baseline",
        "slots": len(ki.slots),
        "orig_solver": dump.get("used_solver"),
        "orig_seconds": dump.get("solve_duration_s"),
        "orig_violation": dump.get("floor_violation_kwh"),
    }
    tmp = tempfile.mkdtemp(prefix="replay_")
    old_cwd = Path.cwd()
    # Never evict real incident dumps; kepler_debug.lp lands next to this dir (in tmp).
    os.environ["DARKSTAR_SOLVER_DUMP_DIR"] = str(Path(tmp) / "solver_dumps")
    os.chdir(tmp)
    t0 = time.monotonic()
    stats: dict[str, int] = {}
    try:
        with _solver_overrides(args), _model_stats(stats):
            result = KeplerSolver().solve(ki, cfg)
        row.update(stats)
        row["bathing"] = _bathing_rows(result)
        row.update(
            status="shipped",
            seconds=round(time.monotonic() - t0, 2),
            status_msg=result.status_msg,
            objective=result.objective_cost_sek,
            total_cost=round(result.total_cost_sek, 2),
            floor_lines=_floor_report(ki, cfg, result),
        )
    except PlannerError as e:
        row.update(stats)
        row.update(
            status="rejected",
            seconds=round(time.monotonic() - t0, 2),
            error_code=str(e.code.name if hasattr(e.code, "name") else e.code),
            reason=str(e.details.get("reason", "")),
            violation=e.details.get("floor_violation_kwh"),
        )
    finally:
        os.chdir(old_cwd)
        os.environ.pop("DARKSTAR_SOLVER_DUMP_DIR", None)
    return row


SWEEP = [
    ("baseline", {}),
    ("heur=0.25", {"highs_opt": ["mip_heuristic_effort=0.25"]}),
    ("heur=0.5", {"highs_opt": ["mip_heuristic_effort=0.5"]}),
    ("sym=off", {"highs_opt": ["mip_detect_symmetry=false"]}),
    ("gap=0.02", {"gap": 0.02}),
    ("seed=2", {"seed": 2}),
    ("seed=3", {"seed": 3}),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--solver", choices=["highs", "cbc", "glpk"], default="highs")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="seconds (default: the dump's own time_limit_s). "
        "This machine is ~15x faster than the box: 16 =~ prod 240",
    )
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="HiGHS threads (kepler deliberately omits this in prod)",
    )
    ap.add_argument(
        "--highs-opt",
        action="append",
        metavar="KEY=VAL",
        help="raw HiGHS option, typed coercion (repeatable)",
    )
    ap.add_argument(
        "--mutate", action="append", metavar="NAME[:ARG]", help=f"one of: {', '.join(MUTATIONS)}"
    )
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.dumps:
        dump_limit = json.loads(path.read_text()).get("time_limit_s", 240)
        variants = SWEEP if args.sweep else [("", {})]
        for label, overrides in variants:
            a = argparse.Namespace(**vars(args))
            for k, v in overrides.items():
                setattr(a, k, v)
            if a.time_limit is None:
                a.time_limit = float(dump_limit)
            for _ in range(args.repeat):
                row = run_replay(path, a, label)
                rows.append(row)
                if not args.as_json:
                    _print_row(row)
    if args.as_json:
        print(json.dumps(rows, indent=2, default=str))


def _print_row(r: dict[str, Any]) -> None:
    head = f"{r['dump']}  [{r['label']}]  T={r['slots']}"
    if r["status"] == "shipped":
        print(
            f"{head}  SHIPPED in {r['seconds']}s  obj={r.get('objective')}  "
            f"cost={r.get('total_cost')}  ({r.get('status_msg', '')[:60]})"
        )
        for line in r.get("floor_lines", []):
            print(line)
        if "int_vars" in r:
            print(
                f"    model: {r['vars']} vars ({r['int_vars']} integer), "
                f"{r['constraints']} constraints"
            )
        for hid, b in (r.get("bathing") or {}).items():
            print(
                f"    bathing {hid}: ready T {b['ready_temp_c']} short {b['shortfall_c']} "
                f"unavoidable {b['unavoidable_c']} planned {b['planned_kwh']} kWh"
            )
    else:
        print(
            f"{head}  REJECTED in {r['seconds']}s  {r.get('error_code')}  "
            f"reason={r.get('reason', '')[:50]}  violation={r.get('violation')}"
        )
    orig = f"    prod: {r['orig_solver']} {r['orig_seconds']}s violation={r['orig_violation']}"
    print(orig)


if __name__ == "__main__":
    main()
