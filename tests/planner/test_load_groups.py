"""Shared-feed ceilings: two loads behind one sub-panel must not be co-scheduled.

The solver optimises against ONE net node and knows nothing about sub-panels, so it
happily books two loads whose combined draw exceeds the feed they share. Measured at
Burgbyn 2026-08-16: the spa (1.8 kW = 7.8 A) and the villavagn tank (1.6 kW = 6.7 A)
both sit on the villavagn's L2 and together reach 14.2 A against a 10 A guard — leaving
the phase guard to clean up after the plan rather than acting as the backstop it is.
"""

from datetime import datetime, timedelta

from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    LoadGroup,
    WaterHeaterInput,
)

SPA_KW = 1.8
TANK_KW = 1.6
VILLAVAGN_CAP_KW = 2.3  # 10 A x 230 V


def _slots(n: int = 4, price: float = 0.5):
    start = datetime(2026, 8, 16, 12, 0)
    return [
        KeplerInputSlot(
            start_time=start + timedelta(hours=i),
            end_time=start + timedelta(hours=i + 1),
            load_kwh=0.0,
            pv_kwh=10.0,          # plenty of surplus: price never forces the choice
            import_price_sek_kwh=price,
            export_price_sek_kwh=0.02,
        )
        for i in range(n)
    ]


def _heaters():
    return [
        WaterHeaterInput(
            id="spa", power_kw=SPA_KW, min_kwh_per_day=SPA_KW * 2,
            max_hours_between_heating=24.0, min_spacing_hours=0.0,
        ),
        WaterHeaterInput(
            id="villavagn_tank", power_kw=TANK_KW, min_kwh_per_day=TANK_KW * 2,
            max_hours_between_heating=24.0, min_spacing_hours=0.0,
        ),
    ]


def _config(groups: list[LoadGroup] | None = None):
    return KeplerConfig(
        capacity_kwh=16.0,
        max_charge_power_kw=9.5,
        max_discharge_power_kw=9.5,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        wear_cost_sek_per_kwh=0.05,
        curtailment_penalty_sek=0.001,
        enable_export=True,
        # Without a reliability penalty the solver has no reason to heat at all —
        # heating is pure cost, so every heater would sit idle and the test would
        # pass vacuously.
        water_reliability_penalty_sek=50.0,
        water_heaters=_heaters(),
        load_groups=groups or [],
    )


def _overlaps(result) -> list[int]:
    """Slot indices where both villavagn loads are scheduled at once."""
    out = []
    for i, s in enumerate(result.slots):
        plans = s.water_heater_results or {}
        if (plans.get("spa") or 0) > 0 and (plans.get("villavagn_tank") or 0) > 0:
            out.append(i)
    return out


def _served(result, heater_id: str) -> float:
    return sum((s.water_heater_results or {}).get(heater_id, 0.0) for s in result.slots)


def test_without_a_group_the_solver_co_schedules_them():
    """Baseline: nothing stops the collision, which is the whole problem."""
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=8.0), _config()
    )
    assert result.is_optimal
    assert _overlaps(result), "expected the unconstrained solver to overlap them"


def test_a_group_cap_below_their_sum_separates_them():
    group = LoadGroup(
        id="villavagn", max_power_kw=VILLAVAGN_CAP_KW,
        members=["spa", "villavagn_tank"],
    )
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=8.0), _config([group])
    )
    assert result.is_optimal
    assert _overlaps(result) == [], "group cap must forbid the shared-feed overlap"


def test_both_loads_still_get_their_energy():
    """Separation, not starvation — there are enough slots for both."""
    group = LoadGroup(
        id="villavagn", max_power_kw=VILLAVAGN_CAP_KW,
        members=["spa", "villavagn_tank"],
    )
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(n=6), initial_soc_kwh=8.0), _config([group])
    )
    assert result.is_optimal
    assert _served(result, "spa") > 0
    assert _served(result, "villavagn_tank") > 0


def test_a_cap_above_their_sum_changes_nothing():
    """The constraint must bind only when the feed is genuinely too small."""
    group = LoadGroup(id="villavagn", max_power_kw=10.0,
                      members=["spa", "villavagn_tank"])
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=8.0), _config([group])
    )
    assert result.is_optimal
    assert _overlaps(result), "a slack cap should not separate them"


def test_unknown_members_are_ignored_not_fatal():
    """A disabled or absent heater must not drop the bound for the rest."""
    group = LoadGroup(
        id="villavagn", max_power_kw=VILLAVAGN_CAP_KW,
        members=["spa", "villavagn_tank", "does_not_exist"],
    )
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=8.0), _config([group])
    )
    assert result.is_optimal
    assert _overlaps(result) == []


def test_a_single_member_over_the_cap_is_simply_never_scheduled():
    """A feed too small for even one load is a config error the plan must not hide."""
    group = LoadGroup(id="tiny", max_power_kw=1.0, members=["spa"])
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(), initial_soc_kwh=8.0), _config([group])
    )
    assert result.is_optimal
    assert _served(result, "spa") == 0.0
