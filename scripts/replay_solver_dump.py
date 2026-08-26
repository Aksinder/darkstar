#!/usr/bin/env python3
"""Replay a solver-failure dump from data/solver_dumps/ locally.

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

Mutations: no_hourly_blocks | no_anchor | no_spacing | no_absorb_cap
           | truncate:H (drop slots beyond H hours)
           | retail:H  (coarsen the tail beyond H hours to whole clock hours)

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
    DeferrableLoadInput,
    EVChargerInput,
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
    return KeplerInput(slots=slots, initial_soc_kwh=float(inp["initial_soc_kwh"]))


def rehydrate_config(cfg: dict[str, Any]) -> KeplerConfig:
    cfg = dict(cfg)
    cfg["water_heaters"] = [_build(WaterHeaterInput, w) for w in cfg.get("water_heaters") or []]
    cfg["ev_chargers"] = [
        _build(
            EVChargerInput,
            {
                **e,
                "deadline": _dt(e.get("deadline")),
                "incentive_buckets": [
                    _build(IncentiveBucket, b) for b in e.get("incentive_buckets") or []
                ],
            },
        )
        for e in cfg.get("ev_chargers") or []
    ]
    cfg["excess_pv_sinks"] = [
        _build(ExcessPVSinkSpec, s) for s in cfg.get("excess_pv_sinks") or []
    ]
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
    "no_anchor": _mut_no_anchor,
    "no_spacing": _mut_no_spacing,
    "no_absorb_cap": _mut_no_absorb_cap,
    "truncate": _mut_truncate,
    "retail": _mut_retail,
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
            t
            for t, rs in enumerate(result.slots)
            if rs.water_heater_results.get(h.id, 0.0) > 0.0
        ]
        credited = len(on) * h.power_kw * avg_h
        physical = sum(h.power_kw * hours[t] for t in on)
        lines.append(
            f"    {h.id}: floor {h.min_kwh_per_day:.2f} | credited {credited:.2f}"
            f" | physical {physical:.2f} kWh ({len(on)} ON-slots)"
        )
    return lines


def run_replay(path: Path, args: argparse.Namespace, label: str = "") -> dict[str, Any]:
    dump = json.loads(path.read_text())
    ki = rehydrate_input(dump["input"])
    cfg = rehydrate_config(dump["config"])
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
    os.environ["DARKSTAR_SOLVER_DUMP_DIR"] = tmp  # never evict real incident dumps
    os.chdir(tmp)  # kepler_debug.lp lands here, not in the repo
    t0 = time.monotonic()
    try:
        with _solver_overrides(args):
            result = KeplerSolver().solve(ki, cfg)
        row.update(
            status="shipped",
            seconds=round(time.monotonic() - t0, 2),
            status_msg=result.status_msg,
            objective=result.objective_cost_sek,
            total_cost=round(result.total_cost_sek, 2),
            floor_lines=_floor_report(ki, cfg, result),
        )
    except PlannerError as e:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--solver", choices=["highs", "cbc", "glpk"], default="highs")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds (default: the dump's own time_limit_s). "
                    "This machine is ~15x faster than the box: 16 =~ prod 240")
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=None,
                    help="HiGHS threads (kepler deliberately omits this in prod)")
    ap.add_argument("--highs-opt", action="append", metavar="KEY=VAL",
                    help="raw HiGHS option, typed coercion (repeatable)")
    ap.add_argument("--mutate", action="append", metavar="NAME[:ARG]",
                    help=f"one of: {', '.join(MUTATIONS)}")
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
        print(f"{head}  SHIPPED in {r['seconds']}s  obj={r.get('objective')}  "
              f"cost={r.get('total_cost')}  ({r.get('status_msg', '')[:60]})")
        for line in r.get("floor_lines", []):
            print(line)
    else:
        print(f"{head}  REJECTED in {r['seconds']}s  {r.get('error_code')}  "
              f"reason={r.get('reason', '')[:50]}  violation={r.get('violation')}")
    orig = f"    prod: {r['orig_solver']} {r['orig_seconds']}s violation={r['orig_violation']}"
    print(orig)


if __name__ == "__main__":
    main()
