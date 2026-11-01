"""EV departure energy target + charging-loss modelling (both config-gated, default off)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from planner.ev_priority import add_departure_energy_target
from planner.pipeline import _apply_ev_owner_levers
from planner.solver.adapter import build_ev_charger_inputs
from planner.solver.kepler import KeplerSolver, ev_departure_need_grid_kwh
from planner.solver.types import (
    EVChargerInput,
    EVDepartureTarget,
    IncentiveBucket,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
)

TZ = timezone(timedelta(hours=1))
START = datetime(2026, 11, 2, 0, 0, tzinfo=TZ)


def _slots(n: int = 16, price: float = 1.0) -> list[KeplerInputSlot]:
    return [
        KeplerInputSlot(
            start_time=START + timedelta(minutes=15 * i),
            end_time=START + timedelta(minutes=15 * (i + 1)),
            load_kwh=0.1,
            pv_kwh=0.0,
            import_price_sek_kwh=price,
            export_price_sek_kwh=0.0,
        )
        for i in range(n)
    ]


def _config(chargers: list[EVChargerInput]) -> KeplerConfig:
    # No battery: the EV is the only flexible load, so the arithmetic is exact.
    return KeplerConfig(
        capacity_kwh=0.0,
        min_soc_percent=0.0,
        max_soc_percent=100.0,
        max_charge_power_kw=0.0,
        max_discharge_power_kw=0.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        wear_cost_sek_per_kwh=0.0,
        max_import_power_kw=30.0,
        enable_export=False,
        ev_chargers=chargers,
    )


def _ev(**kw) -> EVChargerInput:
    base: dict = {
        "id": "tesla",
        "max_power_kw": 4.0,  # 1 kWh per 15-min slot
        "battery_capacity_kwh": 50.0,
        "current_soc_percent": 40.0,
        "plugged_in": True,
        "deadline": None,
    }
    base.update(kw)
    return EVChargerInput(**base)


def _ev_kwh(result, charger_id: str = "tesla", before: datetime | None = None) -> float:
    return sum(
        s.ev_charger_results.get(charger_id, 0.0) * 0.25
        for s in result.slots
        if before is None or s.end_time <= before
    )


def _solve(chargers: list[EVChargerInput], n: int = 16, price: float = 1.0):
    result = KeplerSolver().solve(
        KeplerInput(slots=_slots(n, price), initial_soc_kwh=0.0), _config(chargers)
    )
    assert result.is_optimal
    return result


class TestNeedArithmetic:
    def test_need_is_soc_gap_times_capacity_over_efficiency(self):
        ev = _ev(current_soc_percent=40.0, charge_efficiency=0.8)
        # (60 - 40)% x 50 kWh = 10 kWh into the car = 12.5 kWh from the grid
        assert abs(ev_departure_need_grid_kwh(ev, 60.0) - 12.5) < 1e-9

    def test_target_at_or_below_current_needs_nothing(self):
        assert ev_departure_need_grid_kwh(_ev(current_soc_percent=70.0), 60.0) == 0.0

    def test_target_clamped_to_full(self):
        assert abs(ev_departure_need_grid_kwh(_ev(current_soc_percent=90.0), 150.0) - 5.0) < 1e-9

    def test_bad_efficiency_falls_back_to_lossless(self):
        for bad in (0.0, -1.0, 1.5):
            ev = _ev(charge_efficiency=bad)
            assert abs(ev_departure_need_grid_kwh(ev, 50.0) - 5.0) < 1e-9


class TestDepartureTargetSolver:
    def test_default_no_target_no_buckets_charges_nothing(self):
        assert _ev_kwh(_solve([_ev()])) == 0.0

    def test_target_charges_the_need_before_the_deadline(self):
        deadline = START + timedelta(hours=2)  # 8 slots
        ev = _ev(
            departure_targets=[
                EVDepartureTarget(
                    deadline=deadline, target_soc_percent=48.0, shortfall_penalty_sek_per_kwh=3.0
                )
            ]
        )
        result = _solve([ev])
        # (48 - 40)% x 50 = 4 kWh = 4 slots at 1 kWh, all inside the window
        assert abs(_ev_kwh(result, before=deadline) - 4.0) < 1e-6
        assert abs(_ev_kwh(result) - 4.0) < 1e-6

    def test_penalty_below_price_leaves_the_shortfall(self):
        ev = _ev(
            departure_targets=[
                EVDepartureTarget(
                    deadline=START + timedelta(hours=2),
                    target_soc_percent=48.0,
                    shortfall_penalty_sek_per_kwh=0.5,
                )
            ]
        )
        assert _ev_kwh(_solve([ev], price=1.0)) == 0.0

    def test_unreachable_target_is_soft_not_infeasible(self):
        deadline = START + timedelta(minutes=30)  # 2 slots = 2 kWh max
        ev = _ev(
            departure_targets=[
                EVDepartureTarget(
                    deadline=deadline, target_soc_percent=80.0, shortfall_penalty_sek_per_kwh=3.0
                )
            ]
        )
        result = _solve([ev])
        assert abs(_ev_kwh(result, before=deadline) - 2.0) < 1e-6

    def test_deadline_before_first_slot_is_inert(self):
        ev = _ev(
            departure_targets=[
                EVDepartureTarget(
                    deadline=START - timedelta(hours=1),
                    target_soc_percent=80.0,
                    shortfall_penalty_sek_per_kwh=3.0,
                )
            ]
        )
        assert _ev_kwh(_solve([ev])) == 0.0

    def test_target_is_capped_by_the_penalty_ladder(self):
        # Ladder allows 40 -> 44 % (2 kWh); the 48 % target can only be partly met.
        ev = _ev(
            incentive_buckets=[IncentiveBucket(threshold_soc=44.0, value_sek=0.1)],
            departure_targets=[
                EVDepartureTarget(
                    deadline=START + timedelta(hours=2),
                    target_soc_percent=48.0,
                    shortfall_penalty_sek_per_kwh=3.0,
                )
            ],
        )
        assert abs(_ev_kwh(_solve([ev])) - 2.0) < 1e-6


class TestChargeEfficiency:
    def _ladder_ev(self, eff: float) -> EVChargerInput:
        # 40 -> 48 % = 4 kWh into the car, valued far above the 1.0 import price
        return _ev(
            max_power_kw=0.8,  # 0.2 kWh per slot for a fine quantum
            charge_efficiency=eff,
            incentive_buckets=[IncentiveBucket(threshold_soc=48.0, value_sek=5.0)],
        )

    def test_lossless_default_buys_exactly_the_band(self):
        assert abs(_ev_kwh(_solve([self._ladder_ev(1.0)], n=40)) - 4.0) < 1e-6

    def test_losses_buy_more_grid_energy_for_the_same_band(self):
        grid = _ev_kwh(_solve([self._ladder_ev(0.8)], n=40))
        # 4 kWh into the car / 0.8 = 5 kWh from the grid
        assert abs(grid - 5.0) < 1e-6


class TestAdapter:
    def _build(self, dep_cfg, ev_extra=None, deadline=START + timedelta(hours=7)):
        cfg = [{"id": "tesla", "max_power_kw": 11.0, "battery_capacity_kwh": 60.0}]
        cfg[0].update(ev_extra or {})
        state = [{"id": "tesla", "soc_percent": 30.0, "plugged_in": True, "deadline": deadline}]
        return build_ev_charger_inputs(cfg, state, departure_target_cfg=dep_cfg)[0]

    def test_defaults_are_legacy(self):
        ev = self._build(None)
        assert ev.charge_efficiency == 1.0
        assert ev.departure_targets == []

    def test_charge_efficiency_read_from_config(self):
        assert self._build(None, {"charge_efficiency": 0.88}).charge_efficiency == 0.88

    def test_disabled_block_builds_no_target(self):
        ev = self._build(
            {"enabled": False, "shortfall_penalty_sek_per_kwh": 3.0}, {"departure_soc": 80}
        )
        assert ev.departure_targets == []

    def test_enabled_builds_target_at_the_recurring_deadline(self):
        ev = self._build(
            {"enabled": True, "shortfall_penalty_sek_per_kwh": 3.0}, {"departure_soc": 80}
        )
        assert len(ev.departure_targets) == 1
        tgt = ev.departure_targets[0]
        assert tgt.deadline == START + timedelta(hours=7)
        assert tgt.target_soc_percent == 80.0
        assert tgt.shortfall_penalty_sek_per_kwh == 3.0

    def test_no_deadline_or_no_soc_builds_no_target(self):
        on = {"enabled": True, "shortfall_penalty_sek_per_kwh": 3.0}
        assert self._build(on, {"departure_soc": 80}, deadline=None).departure_targets == []
        assert self._build(on, {}).departure_targets == []


class TestOneOffDeparture:
    def test_helper_appends_target(self):
        ev = _ev()
        dl = START + timedelta(hours=5)
        notes = add_departure_energy_target([ev], "tesla", dl, 75.0, 3.0)
        assert notes and ev.departure_targets[0].deadline == dl

    def test_helper_inert_cases(self):
        ev = _ev()
        dl = START + timedelta(hours=5)
        assert add_departure_energy_target([ev], "other", dl, 75.0, 3.0) == []
        assert add_departure_energy_target([ev], "tesla", None, 75.0, 3.0) == []
        assert add_departure_energy_target([ev], "tesla", dl, None, 3.0) == []
        assert add_departure_energy_target([ev], "tesla", dl, 75.0, 0.0) == []
        assert ev.departure_targets == []

    def _run(self, active_config, *, vacation=False, dep_hours=5.0):
        ev = _ev()
        kc = SimpleNamespace(ev_chargers=[ev])
        ki = SimpleNamespace(slots=_slots(96))
        dep_ts = START.timestamp() + dep_hours * 3600.0

        async def _state(entity):
            return {"state": "x", "attributes": {"timestamp": dep_ts}}

        async def _float(entity):
            return 75.0

        with (
            patch("backend.core.ha_client.get_ha_entity_state", _state),
            patch("backend.core.ha_client.get_ha_sensor_float", _float),
        ):
            asyncio.run(_apply_ev_owner_levers(kc, active_config, ki, vacation_enabled=vacation))
        return ev

    @staticmethod
    def _cfg(enabled: bool) -> dict:
        return {
            "ev_departure_target": {"enabled": enabled, "shortfall_penalty_sek_per_kwh": 3.0},
            "executor": {
                "ev_surplus": {
                    "chargers": [
                        {
                            "id": "tesla",
                            "departure_entity": "input_datetime.dep",
                            "departure_target_entity": "input_number.dep_soc",
                        }
                    ]
                }
            },
        }

    def test_pipeline_adds_one_off_target_when_enabled(self):
        ev = self._run(self._cfg(True))
        assert len(ev.departure_targets) == 1
        assert ev.departure_targets[0].deadline == START + timedelta(hours=5)
        assert ev.departure_targets[0].target_soc_percent == 75.0

    def test_pipeline_default_off(self):
        assert self._run(self._cfg(False)).departure_targets == []

    def test_pipeline_vacation_and_out_of_horizon_are_inert(self):
        assert self._run(self._cfg(True), vacation=True).departure_targets == []
        assert self._run(self._cfg(True), dep_hours=-1.0).departure_targets == []
        assert self._run(self._cfg(True), dep_hours=48.0).departure_targets == []
