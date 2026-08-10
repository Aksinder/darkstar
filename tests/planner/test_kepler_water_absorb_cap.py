"""Water absorption cap + heat/boost exclusivity (2026-08-10).

The phantom this pins down: the solver has no tank model, and the boost reward
(>= export price everywhere) paid it to "consume" 30+ kWh/day into a 195+40 L tank
pair whose measured intake is ~4-5 kWh/day. The phantom load ate the entire modeled
PV surplus, so the battery never charged and nights ran on imports.

Design intent (owner decision 2026-08-10): boost SHOULD fire whenever surplus exists
and filling beats selling — the hardware thermostat is the physical guard. The cap
only stops the PLAN from booking absorption that cannot physically happen, so the
battery/night decisions are grounded in reality.
"""

from datetime import datetime, timedelta

from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    WaterHeaterInput,
)


def _wh(
    id: str = "wh1",
    power_kw: float = 3.4,
    min_kwh_per_day: float = 0.0,
    absorb_cap: float | None = None,
    absorbed_today: float = 0.0,
    heated_today: float = 0.0,
    force_on_slots: list[int] | None = None,
):
    return WaterHeaterInput(
        id=id,
        power_kw=power_kw,
        min_kwh_per_day=min_kwh_per_day,
        max_hours_between_heating=24.0,
        min_spacing_hours=0.0,
        heated_today_kwh=heated_today,
        force_on_slots=force_on_slots,
        absorb_cap_kwh_per_day=absorb_cap,
        absorbed_today_kwh=absorbed_today,
    )


def _make_slots(
    n: int,
    pv_kwh: float = 10.0,
    load_kwh: float = 0.5,
    export_price: float = 0.2,
    import_price: float = 1.0,
    start: datetime | None = None,
) -> list[KeplerInputSlot]:
    t0 = start or datetime(2025, 6, 1, 8, 0)
    return [
        KeplerInputSlot(
            start_time=t0 + timedelta(minutes=15 * i),
            end_time=t0 + timedelta(minutes=15 * (i + 1)),
            load_kwh=load_kwh,
            pv_kwh=pv_kwh,
            import_price_sek_kwh=import_price,
            export_price_sek_kwh=export_price,
        )
        for i in range(n)
    ]


def _config(n_slots: int, heaters: list[WaterHeaterInput], reward: float = 1.0) -> KeplerConfig:
    """Battery pinned full + big surplus + boost reward above export = the live
    reward-farming regime that produced the phantom."""
    return KeplerConfig(
        capacity_kwh=10.0,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        wear_cost_sek_per_kwh=0.45,
        excess_pv_slots=[True] * n_slots,
        excess_pv_sink="water_heater_boost",
        excess_pv_reward_sek_per_kwh=reward,
        excess_pv_soc_threshold_percent=85.0,
        water_heaters=heaters,
        # Production always has a reliability floor via COMFORT_MAP (level 3 -> 15
        # SEK/kWh shortfall). The soft cap's overage penalty (~1.2) must be dwarfed by
        # it — with the default 0.0 the floor would be free to violate and these tests
        # would measure nothing real.
        water_reliability_penalty_sek=15.0,
    )


def _water_kwh(result, heater_id: str, power_kw: float) -> float:
    """Planned water energy for one heater over the whole horizon.

    Extraction reports power once whether heat, boost or (formerly) both are on,
    and the exclusivity constraint now guarantees extraction == LP energy charge.
    """
    total = 0.0
    for s in result.slots:
        kw = s.water_heater_results.get(heater_id, s.water_heat_kw)
        total += kw * 0.25
    return total


class TestAbsorbCap:
    def test_uncapped_reproduces_the_phantom(self):
        """Control: without a cap, reward farming books far more than any tank takes.

        This is the pre-fix behavior — if this test ever starts failing the regime
        has changed and the cap tests below must be revisited.
        """
        n = 48  # 12 h of surplus
        heaters = [_wh(absorb_cap=None)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        assert _water_kwh(result, "wh1", 3.4) > 15.0, (
            "expected uncapped reward farming to book >15 kWh in 12 surplus hours"
        )

    def test_cap_bounds_the_day(self):
        n = 48
        heaters = [_wh(absorb_cap=6.0)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked <= 6.0 + 1e-6, f"cap 6.0 exceeded: {booked:.2f}"

    def test_absorbed_today_shrinks_the_first_bucket(self):
        """5.5 of 6.0 already absorbed (unclamped) => at most ~0.5 more today."""
        n = 48
        heaters = [_wh(absorb_cap=6.0, absorbed_today=5.5)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked <= 0.5 + 0.85 + 1e-6, (
            f"first-bucket remaining ~0.5 (one 15-min slot granularity 0.85) exceeded: {booked:.2f}"
        )

    def test_cap_never_squeezes_the_comfort_floor(self):
        """min_kwh above the cap: the floor runs OVER the soft cap, min is met.

        The soft-cap contract: any floor shortfall is bounded by the point where one
        more slot's overage (lambda x kwh_per_slot = 1.2 x 0.85) costs more than the
        remaining violation (reliability x shortfall). Bound here: 1.2*0.85/15 = 0.068
        kWh = 68 Wh — physically negligible, and it shrinks as the reliability penalty
        grows (production runs 15-1000 SEK/kWh). Exact equality is NOT the contract;
        priced-and-bounded is.
        """
        n = 48
        heaters = [_wh(min_kwh_per_day=6.0, absorb_cap=2.0)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        shortfall_bound = 1.2 * 0.85 / 15.0 + 1e-6
        assert booked >= 6.0 - shortfall_bound, (
            f"comfort floor materially lost to the cap: {booked:.2f} "
            f"(allowed shortfall {shortfall_bound:.3f})"
        )

    def test_force_on_beyond_cap_stays_feasible(self):
        """Anti-legionella style force_on must never make the MILP infeasible."""
        n = 16
        heaters = [_wh(absorb_cap=0.5, force_on_slots=[0, 1, 2, 3])]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal, "force_on beyond the cap made the problem infeasible"

    def test_second_day_bucket_gets_the_full_cap(self):
        """absorbed_today only discounts the FIRST bucket; day 2 gets cap again."""
        n = 192  # 48 h
        heaters = [_wh(absorb_cap=4.0, absorbed_today=4.0)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        # Day 1 (first ~64 slots to midnight from 08:00): ~0 more.
        day1 = sum((s.water_heat_kw or 0.0) * 0.25 for s in result.slots[:64])
        day2 = sum((s.water_heat_kw or 0.0) * 0.25 for s in result.slots[64:160])
        assert day1 <= 0.85 + 1e-6, f"day-1 bucket should be exhausted: {day1:.2f}"
        assert day2 <= 4.0 + 1e-6, f"day-2 cap exceeded: {day2:.2f}"

    def test_freed_surplus_goes_to_export_not_water(self):
        """The point of it all: with the cap, surplus stops vanishing into phantom
        water and shows up as export (battery already full here)."""
        n = 48
        uncapped = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0),
            _config(n, [_wh(absorb_cap=None)]),
        )
        capped = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0),
            _config(n, [_wh(absorb_cap=6.0)]),
        )
        assert uncapped.is_optimal and capped.is_optimal
        exp_uncapped = sum(s.grid_export_kwh or 0.0 for s in uncapped.slots)
        exp_capped = sum(s.grid_export_kwh or 0.0 for s in capped.slots)
        assert exp_capped > exp_uncapped + 5.0, (
            f"capping water should free surplus to export: {exp_uncapped:.1f} -> {exp_capped:.1f}"
        )


class TestHeatBoostExclusivity:
    def test_heat_and_boost_never_double_charge_the_balance(self):
        """Per-slot energy balance must close: (pv + import + discharge) ==
        (load + water + export + charge). Before the exclusivity constraint the LP
        charged 2x power in double-on slots while extraction reported power once —
        ~5 kWh/day of hidden phantom load."""
        n = 24
        heaters = [_wh(min_kwh_per_day=3.0, absorb_cap=None)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        for i, s in enumerate(result.slots):
            pv_used = _make_slots(n)[i].pv_kwh
            lhs = pv_used + (s.grid_import_kwh or 0.0) + (s.discharge_kwh or 0.0)
            water = (s.water_heat_kw or 0.0) * 0.25
            rhs = (
                _make_slots(n)[i].load_kwh
                + water
                + (s.grid_export_kwh or 0.0)
                + (s.charge_kwh or 0.0)
            )
            assert abs(lhs - rhs) < 1e-4, (
                f"slot {i}: balance residual {lhs - rhs:+.4f} "
                f"(hidden double-charged water would show here)"
            )


class TestSoftCapReviewRegressions:
    """Regressions from the 2026-08-10 adversarial review of the HARD-cap design.

    Each of these reproduced a MAJOR/CRITICAL defect against the hard cap; the soft
    cap + demand ratchet must keep them green forever.
    """

    def test_hourly_blocks_floor_stays_reachable(self):
        """CRITICAL repro: hourly blocks quantize heat to 4-slot hour groups, so a
        hard cap raised to a slot-granular floor FORBADE the next attainable group
        (attainable {0, 3.4, 6.8}; floor 5.1; hard cap 5.1 banned 6.8 -> booked 3.4
        + 1.7 kWh violation at 15-1000 SEK/kWh, every replan). Soft cap: books 6.8,
        pays ~2 SEK overage, floor met."""
        n = 48
        heaters = [_wh(min_kwh_per_day=6.0, absorb_cap=6.0, heated_today=0.9, absorbed_today=0.9)]
        config = _config(n, heaters)
        config.water_hourly_blocks = True
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), config
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        remaining_floor = 6.0 - 0.9
        assert booked >= remaining_floor - 0.075, (
            f"hour-lattice floor unreachable again: booked {booked:.2f} < {remaining_floor:.2f}"
        )

    def test_misaligned_force_on_never_infeasible(self):
        """MAJOR repro: force_on covering 2 slots of an interior hour group, hourly
        tying expands the group to 4 slots; the hard cap's forced-energy raise
        undercounted -> hard INFEASIBLE -> PlannerError, stale plan kept. Soft cap:
        always feasible, overage priced."""
        n = 16
        heaters = [
            _wh(absorb_cap=0.5, absorbed_today=0.5, force_on_slots=[4, 5]),
        ]
        config = _config(n, heaters)
        config.water_hourly_blocks = True
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), config
        )
        assert result.is_optimal, "misaligned force_on under hourly tying went infeasible"

    def test_draw_day_ratchet_reopens_boost(self):
        """MAJOR repro (lockout): on a big-draw day the hard cap hit zero remaining
        and nothing could book for up to 23 h with a stone-cold tank. The ratchet
        expands the cap with measured absorption (adapter: absorbed x 1.3), so
        remaining headroom exists and boost books within it."""
        n = 48
        # Adapter-computed cap for absorbed 6.5, trailing 2.5, min 6:
        # max(6, 3.75, 6.5*1.3=8.45) = 8.45 -> remaining 1.95 after subtraction.
        heaters = [
            _wh(min_kwh_per_day=6.0, absorb_cap=8.45, heated_today=6.0, absorbed_today=6.5),
        ]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked >= 0.85 - 1e-6, (
            f"draw-day headroom exists (1.95 kWh) but nothing booked: {booked:.2f} "
            "— the lockout is back"
        )

    def test_reward_farming_beyond_cap_is_unprofitable(self):
        """The soft cap's core economics: overage penalty (reward x 1.2) makes every
        kWh beyond the cap net-negative for pure reward farming, so booking stays
        near the cap instead of filling all 12 surplus hours."""
        n = 48
        heaters = [_wh(absorb_cap=6.0)]
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked <= 6.0 + 0.85 + 1e-6, (
            f"reward farming pierced the soft cap: {booked:.2f} (cap 6.0 + one-slot slack)"
        )

    def test_stale_bucket_stats_are_not_subtracted_from_a_new_day(self):
        """MINOR repro (bucket-boundary race): stats measured in the OLD bucket must
        not shrink the NEW bucket's cap when the solve crosses the 10:00 boundary."""
        n = 48
        heaters = [
            _wh(absorb_cap=6.0, absorbed_today=6.0),
        ]
        # Slots start 2025-06-01; stats claim they were measured in a different bucket.
        heaters[0].absorbed_bucket_date = "2025-05-31"
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked > 0.85, f"stale old-bucket absorption locked out the new bucket: {booked:.2f}"

    def test_matching_bucket_stats_are_subtracted(self):
        """Control for the race fix: matching dates keep the subtraction."""
        n = 48
        heaters = [
            _wh(absorb_cap=6.0, absorbed_today=6.0),
        ]
        heaters[0].absorbed_bucket_date = "2025-06-01"  # slots start 2025-06-01 08:00
        result = KeplerSolver().solve(
            KeplerInput(slots=_make_slots(n), initial_soc_kwh=10.0), _config(n, heaters)
        )
        assert result.is_optimal
        booked = _water_kwh(result, "wh1", 3.4)
        assert booked <= 0.85 + 1e-6, f"same-bucket absorption should exhaust the cap: {booked:.2f}"
