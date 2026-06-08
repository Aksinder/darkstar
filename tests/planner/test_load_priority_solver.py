"""Tests for the load-priority / willingness-to-pay (WTP) layer (increment 1).

The WTP layer assigns each controllable (deferrable) load a reservation price built
from tier + intra-tier rank + a linear time-urgency ramp. A priority-bearing load is
OPTIONAL (at most once): it runs only while its WTP for the slot meets the marginal
energy price, so low-priority loads defer/skip under scarcity and soak cheap/surplus
energy, while a high-WTP load keeps running. When the flag is off (or a load has no
priority entry) behaviour is byte-identical to the legacy mandatory run-once model.
"""

from datetime import datetime, timedelta

import pytest

from planner.solver.adapter import build_load_priorities
from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    DeferrableLoadInput,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
    LoadPriority,
    WaterHeaterInput,
)
from tests.planner.test_deferrable_loads_solver import _run_slots, _slots


def _config(loads, **kw):
    """KeplerConfig with the battery neutralised (capacity 0) to isolate loads."""
    base = {
        "capacity_kwh": 0.0,
        "max_charge_power_kw": 1.0,
        "max_discharge_power_kw": 1.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        "wear_cost_sek_per_kwh": 0.0,
        "deferrable_loads": loads,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _solve(prices, loads, pv=None, **kw):
    return KeplerSolver().solve(KeplerInput(_slots(prices, pv), 0.0), _config(loads, **kw))


def _spa(**kw):
    return DeferrableLoadInput(id="spa", energy_kwh=1.0, duration_slots=1, **kw)


class TestWtpSolver:
    def test_off_parity_priorities_ignored_when_disabled(self):
        """Flag OFF => identical schedule + cost even if priorities are populated."""
        prices = [2.0, 2.0, 0.1, 2.0]
        baseline = _solve(prices, [_spa()])
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.6)}
        off = _solve(prices, [_spa()], load_priority_enabled=False, load_priorities=prio)
        assert _run_slots(baseline, "spa") == _run_slots(off, "spa")
        assert baseline.total_cost_sek == pytest.approx(off.total_cost_sek)

    def test_low_wtp_load_skips_when_all_slots_too_expensive(self):
        """A comfort load (base_wtp 0.4) never runs if every slot costs more."""
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve([1.0, 1.0, 1.0, 1.0], [_spa()], load_priority_enabled=True, load_priorities=prio)
        assert r.is_optimal
        assert _run_slots(r, "spa") == []  # skipped — optional, never cheap enough

    def test_low_wtp_load_runs_in_cheap_window(self):
        """The same comfort load runs once when a slot drops below its WTP."""
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve([1.0, 1.0, 0.1, 1.0], [_spa()], load_priority_enabled=True, load_priorities=prio)
        assert _run_slots(r, "spa") == [2]

    def test_high_wtp_load_runs_even_under_scarcity(self):
        """An important load (base_wtp 3.0) runs despite all slots being expensive."""
        load = DeferrableLoadInput(id="heatpump", energy_kwh=1.0, duration_slots=1)
        prio = {"heatpump": LoadPriority(base_wtp_sek_per_kwh=3.0, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve([1.0, 1.0, 1.0, 1.0], [load], load_priority_enabled=True, load_priorities=prio)
        assert len(_run_slots(r, "heatpump")) == 1

    def test_urgency_ramp_runs_an_otherwise_skipped_load_late(self):
        """With base_wtp below every price, the urgency ramp lifts WTP past the price
        in later slots, so the load runs late instead of skipping entirely."""
        prices = [1.0] * 8
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=2.0)}
        r = _solve(prices, [_spa(deadline_slot=7)], load_priority_enabled=True, load_priorities=prio)
        run = _run_slots(r, "spa")
        assert run != []  # urgency made it run (would skip with urgency_wtp=0)
        assert min(run) >= 3  # ran in the later, higher-urgency region

    def test_low_wtp_load_soaks_surplus_pv(self):
        """Free PV makes a slot's marginal energy ~0, so even a low-WTP load runs there."""
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve(
            [1.0, 1.0, 1.0, 1.0],
            [_spa()],
            pv=[0.0, 0.0, 1.0, 0.0],
            load_priority_enabled=True,
            load_priorities=prio,
        )
        assert _run_slots(r, "spa") == [2]

    def test_priority_load_ignores_legacy_soft_deadline_penalty(self):
        """No double-count: a priority load is NOT pulled early by the 30-SEK soft
        tardiness penalty — it still picks the cheap-but-late slot. Without suppression
        the penalty (60 SEK) would dominate and force it to run at slot 0."""
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve(
            [1.0, 1.0, 0.1],
            [_spa(deadline_slot=0, deadline_hard=False)],
            load_priority_enabled=True,
            load_priorities=prio,
        )
        assert _run_slots(r, "spa") == [2]  # cheap slot, soft deadline ignored

    def test_mixed_priority_and_legacy_loads_coexist(self):
        """A priority load is optional; a sibling load WITHOUT an entry keeps the
        legacy mandatory run-once behaviour in the same solve."""
        prices = [1.0, 1.0, 1.0, 1.0]  # all above the spa's WTP
        spa = _spa()  # comfort, will skip
        dishwasher = DeferrableLoadInput(id="dishwasher", energy_kwh=1.0, duration_slots=1)
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.0)}
        r = _solve(prices, [spa, dishwasher], load_priority_enabled=True, load_priorities=prio)
        assert _run_slots(r, "spa") == []  # optional, skipped
        assert len(_run_slots(r, "dishwasher")) == 1  # legacy mandatory run-once


class TestBuildLoadPriorities:
    def test_disabled_returns_empty(self):
        enabled, p = build_load_priorities(
            {"load_priority": {"enabled": False, "loads": {"spa": {"tier": "comfort"}}}}
        )
        assert enabled is False
        assert p == {}

    def test_missing_block_returns_empty(self):
        enabled, p = build_load_priorities({})
        assert enabled is False
        assert p == {}

    def test_tier_inheritance(self):
        enabled, p = build_load_priorities(
            {
                "load_priority": {
                    "enabled": True,
                    "tiers": {"comfort": {"base_wtp_sek_per_kwh": 0.4, "urgency_wtp_sek_per_kwh": 0.6}},
                    "loads": {"spa": {"tier": "comfort"}},
                }
            }
        )
        assert enabled is True
        assert p["spa"].base_wtp_sek_per_kwh == 0.4
        assert p["spa"].urgency_wtp_sek_per_kwh == 0.6

    def test_per_load_override_beats_tier(self):
        _, p = build_load_priorities(
            {
                "load_priority": {
                    "enabled": True,
                    "tiers": {"comfort": {"base_wtp_sek_per_kwh": 0.4, "urgency_wtp_sek_per_kwh": 0.6}},
                    "loads": {"spa": {"tier": "comfort", "base_wtp_sek_per_kwh": 0.9}},
                }
            }
        )
        assert p["spa"].base_wtp_sek_per_kwh == 0.9  # overridden
        assert p["spa"].urgency_wtp_sek_per_kwh == 0.6  # inherited

    def test_rank_maps_to_negative_epsilon(self):
        _, p = build_load_priorities(
            {
                "load_priority": {
                    "enabled": True,
                    "rank_step_sek_per_kwh": 0.001,
                    "tiers": {"comfort": {"base_wtp_sek_per_kwh": 0.4}},
                    "loads": {"a": {"tier": "comfort", "rank": 0}, "b": {"tier": "comfort", "rank": 2}},
                }
            }
        )
        assert p["a"].rank_epsilon_sek_per_kwh == 0.0
        assert p["b"].rank_epsilon_sek_per_kwh == pytest.approx(-0.002)

    def test_unknown_tier_is_skipped_not_raised(self):
        enabled, p = build_load_priorities(
            {"load_priority": {"enabled": True, "tiers": {}, "loads": {"spa": {"tier": "nope"}}}}
        )
        assert enabled is True
        assert "spa" not in p


def _wslots(prices, pv=None):
    """30-min slots on a single date (so the daily-minimum logic applies)."""
    base = datetime(2026, 1, 15, 0, 0)
    out = []
    for i, price in enumerate(prices):
        s = base + timedelta(minutes=30 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=30),
                load_kwh=0.0,
                pv_kwh=(pv[i] if pv else 0.0),
                import_price_sek_kwh=price,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def _wcfg(heaters, **kw):
    """KeplerConfig with battery neutralised; reliability penalty on so the legacy
    model would FORCE the daily minimum (the contrast the WTP layer overrides)."""
    base = {
        "capacity_kwh": 0.0,
        "max_charge_power_kw": 1.0,
        "max_discharge_power_kw": 1.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "min_soc_percent": 0.0,
        "max_soc_percent": 100.0,
        "wear_cost_sek_per_kwh": 0.0,
        "water_heaters": heaters,
        "water_reliability_penalty_sek": 100.0,
    }
    base.update(kw)
    return KeplerConfig(**base)


def _heated(result, heater_id):
    """Total kWh heated by a heater (water_heater_results is kW; 30-min slots => x0.5)."""
    return sum(s.water_heater_results.get(heater_id, 0.0) * 0.5 for s in result.slots)


def _wh(heater_id="spa", power_kw=2.0, min_kwh_per_day=2.0):
    return WaterHeaterInput(
        id=heater_id,
        power_kw=power_kw,
        min_kwh_per_day=min_kwh_per_day,
        max_hours_between_heating=0.0,
        min_spacing_hours=0.0,
    )


class TestWtpWaterHeaters:
    def test_low_priority_heater_is_forced_without_priority(self):
        """Baseline contrast: without the WTP layer the reliability penalty FORCES the
        spa to meet its daily minimum even at a high uniform price."""
        r = KeplerSolver().solve(KeplerInput(_wslots([1.0] * 8), 0.0), _wcfg([_wh()]))
        assert r.is_optimal
        assert _heated(r, "spa") == pytest.approx(2.0)

    def test_low_priority_heater_skips_when_expensive(self):
        """With the WTP layer on, a low-priority heater (base_wtp 0.4) skips a day when
        every slot costs more than its reservation price — no reliability forcing."""
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4)}
        r = KeplerSolver().solve(
            KeplerInput(_wslots([1.0] * 8), 0.0),
            _wcfg([_wh()], load_priority_enabled=True, load_priorities=prio),
        )
        assert r.is_optimal
        assert _heated(r, "spa") == pytest.approx(0.0)

    def test_low_priority_heater_fills_need_in_cheap_window(self):
        """The same low-priority heater fills its daily need when a cheap window appears."""
        prices = [1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0]
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4)}
        r = KeplerSolver().solve(
            KeplerInput(_wslots(prices), 0.0),
            _wcfg([_wh()], load_priority_enabled=True, load_priorities=prio),
        )
        assert _heated(r, "spa") == pytest.approx(2.0)  # need met
        # ...and only in the cheap slots (3, 4).
        hot = [i for i, s in enumerate(r.slots) if s.water_heater_results.get("spa", 0.0) > 0]
        assert hot == [3, 4]

    def test_satiation_prevents_overheating(self):
        """An important heater (base_wtp 3.0) at dirt-cheap prices heats EXACTLY its daily
        need, not every slot — the credit is satiated at min_kwh_per_day."""
        prio = {"vvb": LoadPriority(base_wtp_sek_per_kwh=3.0)}
        r = KeplerSolver().solve(
            KeplerInput(_wslots([0.1] * 8), 0.0),
            _wcfg(
                [_wh("vvb", min_kwh_per_day=2.0)],
                load_priority_enabled=True,
                load_priorities=prio,
            ),
        )
        assert _heated(r, "vvb") == pytest.approx(2.0)  # need only — no over-heating

    def test_off_parity_water_heater(self):
        """Flag OFF with priorities populated => identical heating + cost to baseline."""
        prices = [1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
        baseline = KeplerSolver().solve(KeplerInput(_wslots(prices), 0.0), _wcfg([_wh()]))
        prio = {"spa": LoadPriority(base_wtp_sek_per_kwh=0.4, urgency_wtp_sek_per_kwh=0.6)}
        off = KeplerSolver().solve(
            KeplerInput(_wslots(prices), 0.0),
            _wcfg([_wh()], load_priority_enabled=False, load_priorities=prio),
        )
        assert _heated(baseline, "spa") == pytest.approx(_heated(off, "spa"))
        assert baseline.total_cost_sek == pytest.approx(off.total_cost_sek)
