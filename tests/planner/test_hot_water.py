"""Tests for the open-loop hot-water availability estimator."""

import pytest

from planner.hot_water import HotWaterEstimator, estimate_draw_kwh
from planner.thermal import WaterTankModel


def _tank():
    return WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=80, ua_w_per_k=2.0)


def test_starts_full():
    est = HotWaterEstimator(_tank())
    assert est.soc_percent() == pytest.approx(100.0)
    assert est.temperature_c() == pytest.approx(80.0)
    assert est.liters_in_tank() == pytest.approx(200.0)


def test_from_temperature_seed():
    tank = _tank()
    est = HotWaterEstimator.from_temperature(tank, 45.0)
    assert est.temperature_c() == pytest.approx(45.0)
    # 45 over [10,80] -> 50% SoC
    assert est.soc_percent() == pytest.approx(50.0, abs=0.5)


def test_standing_loss_lowers_state_when_idle():
    est = HotWaterEstimator.from_temperature(_tank(), 70.0)
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0)  # idle 1 h
    assert est.stored_kwh < before
    assert est.temperature_c() < 70.0


def test_heating_raises_state_and_caps_at_full():
    est = HotWaterEstimator.from_temperature(_tank(), 40.0)
    for _ in range(60):
        est.update(dt_minutes=10, heating_kw=3.0)  # 10 h of 3 kW
    assert est.soc_percent() == pytest.approx(100.0)  # capped at full
    assert est.temperature_c() == pytest.approx(80.0)


def test_auto_anchors_to_full_when_thermostat_satisfied():
    # Seed deliberately wrong-low; a real heat-up that then cuts off must re-pin
    # the estimate to full regardless of the accumulated drift.
    tank = _tank()
    est = HotWaterEstimator.from_temperature(tank, 30.0, full_anchor_after_min=8.0)
    # Heat for 20 min (sustained), then element switches off (thermostat satisfied).
    est.update(dt_minutes=20, heating_kw=3.0)
    assert est.soc_percent() < 100.0  # not full yet after 20 min from 30C
    est.update(dt_minutes=1, heating_kw=0.0)  # switch-off after sustained heat
    assert est.soc_percent() == pytest.approx(100.0)  # anchored to full


def test_brief_heating_blip_does_not_anchor_full():
    est = HotWaterEstimator.from_temperature(_tank(), 30.0, full_anchor_after_min=8.0)
    est.update(dt_minutes=2, heating_kw=3.0)  # only 2 min < anchor threshold
    est.update(dt_minutes=1, heating_kw=0.0)
    assert est.soc_percent() < 100.0  # must NOT jump to full on a blip


def test_manual_anchor_full():
    est = HotWaterEstimator.from_temperature(_tank(), 20.0)
    est.anchor_full()
    assert est.soc_percent() == pytest.approx(100.0)


def test_mixed_liters_exceeds_volume_and_floors_at_comfort():
    est = HotWaterEstimator.from_temperature(_tank(), 70.0)
    # At 70C with 40C comfort and 10C cold: 200 * (70-10)/(40-10) = 400 L.
    assert est.mixed_liters_at(40.0) == pytest.approx(400.0, abs=1.0)
    cold = HotWaterEstimator.from_temperature(_tank(), 35.0)
    assert cold.mixed_liters_at(40.0) == 0.0  # below comfort -> nothing usable


def test_estimate_draw_is_heating_minus_losses():
    tank = _tank()
    # 5 kWh heated in over 6 h at ~70C avg; losses = UA*dT*h.
    losses = tank.avg_loss_kw(70.0, 20.0) * 6.0  # 2 W/K * 50 K = 100 W -> 0.6 kWh
    draw = estimate_draw_kwh(tank, heating_energy_kwh=5.0, avg_temp_c=70.0, hours=6.0)
    assert draw == pytest.approx(5.0 - losses, abs=1e-6)
    assert losses == pytest.approx(0.6, abs=0.01)


def test_estimate_draw_clamped_at_zero():
    tank = _tank()
    # Less heating than losses (impossible draw) -> clamp to 0.
    assert estimate_draw_kwh(tank, heating_energy_kwh=0.1, avg_temp_c=70.0, hours=6.0) == 0.0


# -- learned draw (the FMB-style down-force) + persistence ------------------


def test_learned_draw_depletes_between_heating_runs():
    """A configured draw must visibly deplete the tank while idle (not just standing loss)."""
    est = HotWaterEstimator(_tank(), prior_draw_kw=1.0)  # ~1 kW average draw
    est.stored_kwh = est.tank.capacity_kwh()
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0)  # idle 1 h, no heating
    drop = before - est.stored_kwh
    # ~1 kWh from draw + a little standing loss; standing loss alone would be <0.1 kWh.
    assert drop > 0.9
    assert est.soc_percent() < 100.0


def test_full_anchor_learns_draw_rate():
    """Over a full->full window, the draw rate is learned from heating_in - losses."""
    tank = _tank()
    est = HotWaterEstimator(tank, prior_draw_kw=0.1, draw_learn_alpha=1.0, full_anchor_after_min=8.0)
    # Start full, then idle 10 h while ~2 kWh is tapped (we deplete via the prior, but the
    # learner uses the energy we put back in). Simulate: idle draws it down, then a heat-up.
    est.update(dt_minutes=600, heating_kw=0.0)  # 10 h idle (minutes_since_anchor=600)
    # Now heat for 1 h at 3 kW (3 kWh in) then cut off -> anchor + learn.
    est.update(dt_minutes=60, heating_kw=3.0)   # heating run = 60 min >= 8 -> eligible
    est.update(dt_minutes=5, heating_kw=0.0)    # cut-off -> _anchor_full_and_learn
    # Window ~11 h, 3 kWh in, losses small -> implied draw ~3/11 ~= 0.27 kW. alpha=1 -> adopt.
    assert est.learned_draw_kw > 0.15  # moved up from the 0.1 prior toward the observed rate
    assert est.soc_percent() == pytest.approx(100.0)  # re-pinned to full


def test_state_dict_roundtrips_through_apply_state():
    tank = _tank()
    a = HotWaterEstimator(tank, prior_draw_kw=0.1)
    a.stored_kwh = 5.0
    a.learned_draw_kw = 0.42
    a._energy_in_since_anchor_kwh = 1.3
    a._minutes_since_anchor = 123.0
    blob = a.state_dict()

    b = HotWaterEstimator(tank)  # starts full
    b.apply_state(blob)
    assert b.stored_kwh == pytest.approx(5.0)
    assert b.learned_draw_kw == pytest.approx(0.42)
    assert b._energy_in_since_anchor_kwh == pytest.approx(1.3)
    assert b._minutes_since_anchor == pytest.approx(123.0)


def test_apply_state_ignores_missing_keys_and_clamps():
    tank = _tank()
    est = HotWaterEstimator(tank)
    est.apply_state({"stored_kwh": 999.0})  # over capacity -> clamped; other keys untouched
    assert est.stored_kwh == pytest.approx(tank.capacity_kwh())
    assert est.learned_draw_kw == pytest.approx(est.prior_draw_kw)
