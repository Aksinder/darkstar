"""Tests for the hot-water tank thermal model."""

import math

import pytest

from planner.thermal import WaterTankModel


def test_heat_capacity():
    tank = WaterTankModel(volume_litres=200)
    # 200 L * 1.1628 Wh/(L*K) ~= 232.6 Wh/K
    assert tank.heat_capacity_wh_per_k == pytest.approx(232.6, abs=0.5)


def test_energy_to_heat_known_value():
    # Heating 200 L by 10 K = 200 * 1.1628 * 10 / 1000 ~= 2.326 kWh
    tank = WaterTankModel(volume_litres=200)
    assert tank.energy_to_heat_kwh(50, 60) == pytest.approx(2.326, abs=0.01)


def test_stored_and_capacity():
    tank = WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=85)
    assert tank.stored_kwh(10) == 0.0
    assert tank.stored_kwh(60) == pytest.approx(tank.energy_to_heat_kwh(10, 60), abs=1e-9)
    assert tank.capacity_kwh() == pytest.approx(tank.energy_to_heat_kwh(10, 85), abs=1e-9)


def test_soc_percent():
    tank = WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=85)
    assert tank.soc_percent(10) == 0.0
    assert tank.soc_percent(85) == 100.0
    assert tank.soc_percent(47.5) == pytest.approx(50.0, abs=0.5)
    # Clamped outside the range.
    assert tank.soc_percent(5) == 0.0
    assert tank.soc_percent(95) == 100.0


def test_heating_round_trips_with_energy():
    tank = WaterTankModel(volume_litres=150)
    e = tank.energy_to_heat_kwh(45, 65)
    assert tank.temp_after_heating(45, e) == pytest.approx(65.0, abs=0.01)


def test_heating_capped_at_max():
    tank = WaterTankModel(volume_litres=100, t_max_c=80)
    assert tank.temp_after_heating(78, 5.0) == 80.0  # capped


def test_standing_loss_cools_toward_ambient():
    tank = WaterTankModel(volume_litres=200, ua_w_per_k=2.0)
    t1 = tank.temp_after_loss(70, hours=1, t_ambient_c=20)
    assert 20 < t1 < 70  # cooled but not to ambient in 1 h
    # After one time constant, ~63.2% of the gap is lost.
    tau = tank.time_constant_hours()
    t_tau = tank.temp_after_loss(70, hours=tau, t_ambient_c=20)
    expected = 20 + (70 - 20) * math.exp(-1)
    assert t_tau == pytest.approx(expected, abs=0.1)


def test_zero_loss_when_no_ua():
    tank = WaterTankModel(volume_litres=200, ua_w_per_k=0.0)
    assert tank.temp_after_loss(70, hours=5, t_ambient_c=20) == 70
    assert tank.standby_loss_kwh(70, hours=5) == 0.0
    assert math.isinf(tank.time_constant_hours())


def test_standby_loss_energy_positive_and_consistent():
    tank = WaterTankModel(volume_litres=200, ua_w_per_k=2.0)
    loss = tank.standby_loss_kwh(70, hours=2, t_ambient_c=20)
    assert loss > 0
    # Loss equals stored(now) - stored(after-cooling).
    t_after = tank.temp_after_loss(70, hours=2, t_ambient_c=20)
    assert loss == pytest.approx(tank.stored_kwh(70) - tank.stored_kwh(t_after), abs=1e-9)


def test_avg_loss_kw():
    tank = WaterTankModel(volume_litres=200, ua_w_per_k=2.0)
    # 2 W/K * (70-20) K = 100 W = 0.1 kW
    assert tank.avg_loss_kw(70, t_ambient_c=20) == pytest.approx(0.1, abs=1e-6)
    assert tank.avg_loss_kw(15, t_ambient_c=20) == 0.0  # below ambient -> no loss
