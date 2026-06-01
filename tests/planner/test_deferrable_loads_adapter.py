"""Tests for building DeferrableLoadInput from config + HA state."""

from datetime import datetime, timedelta

from planner.solver.adapter import build_deferrable_load_inputs
from planner.solver.types import KeplerInputSlot

START = datetime(2026, 6, 1, 18, 0)  # 18:00


def _slots(n=96):
    out = []
    for i in range(n):
        s = START + timedelta(minutes=15 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=15),
                load_kwh=0.0,
                pv_kwh=0.0,
                import_price_sek_kwh=1.0,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def test_only_pending_runs_are_scheduled():
    cfg = [{"id": "dishwasher", "enabled": True, "duration_min": 120, "energy_kwh": 1.2}]
    # No state / not pending -> nothing scheduled.
    assert build_deferrable_load_inputs(cfg, None, _slots()) == []
    assert (
        build_deferrable_load_inputs(cfg, [{"id": "dishwasher", "pending": False}], _slots()) == []
    )
    # Pending -> scheduled.
    out = build_deferrable_load_inputs(cfg, [{"id": "dishwasher", "pending": True}], _slots())
    assert len(out) == 1
    assert out[0].id == "dishwasher"
    assert out[0].duration_slots == 8  # 120 min / 15
    assert out[0].energy_kwh == 1.2


def test_learned_values_override_config_seed():
    cfg = [{"id": "washer", "enabled": True, "duration_min": 120, "energy_kwh": 1.2}]
    state = [{"id": "washer", "pending": True, "duration_min": 90, "energy_kwh": 0.8}]
    out = build_deferrable_load_inputs(cfg, state, _slots())
    assert out[0].duration_slots == 6  # learned 90 min / 15
    assert out[0].energy_kwh == 0.8


def test_disabled_load_skipped():
    cfg = [{"id": "washer", "enabled": False, "duration_min": 90}]
    out = build_deferrable_load_inputs(cfg, [{"id": "washer", "pending": True}], _slots())
    assert out == []


def test_hard_deadline_resolves_clock_time():
    # Slots start at 18:00; deadline 07:00 -> next day 07:00 = 13 h later = 52 slots.
    cfg = [
        {
            "id": "washer",
            "enabled": True,
            "duration_min": 60,
            "deadline_mode": "hard_deadline",
            "hard_deadline": "07:00",
        }
    ]
    out = build_deferrable_load_inputs(cfg, [{"id": "washer", "pending": True}], _slots())
    assert out[0].deadline_hard is True
    # 18:00 -> 07:00 next day = 13 h = 52 slots; last slot ending <= 07:00 is index 51.
    assert out[0].deadline_slot == 51


def test_cheapest_within_hours_is_soft():
    cfg = [
        {
            "id": "dishwasher",
            "enabled": True,
            "duration_min": 120,
            "deadline_mode": "cheapest_within_hours",
            "window_hours": 10,
        }
    ]
    out = build_deferrable_load_inputs(cfg, [{"id": "dishwasher", "pending": True}], _slots())
    assert out[0].deadline_hard is False
    assert out[0].deadline_slot == 39  # 10 h = 40 slots, finish-by index 39


def test_phase_passthrough():
    cfg = [{"id": "dishwasher", "enabled": True, "duration_min": 60, "phase": "A"}]
    out = build_deferrable_load_inputs(cfg, [{"id": "dishwasher", "pending": True}], _slots())
    assert out[0].phase == "A"
