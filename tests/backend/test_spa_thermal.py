"""backend/core/spa_thermal.py: the one spa physics shared by planner and executor."""

from __future__ import annotations

import math

import pytest

from backend.core.spa_thermal import (
    decay_factor,
    heat_gain_c,
    heat_rate_c_per_h,
    hours_to_reach_c,
    max_reachable_temp_c,
    net_heat_rate_c_per_h,
    simulate_temps_c,
    step_temp_c,
)
from planner.thermal import WaterTankModel

C, K, P, AMB = 0.9, 0.021, 1.8, 10.0


def test_decay_and_gain_closed_forms():
    assert decay_factor(K, 2.0) == pytest.approx(math.exp(-0.042))
    assert decay_factor(K, -1.0) == 1.0  # negative hours clamp to 0
    a = math.exp(-K * 0.25)
    assert heat_gain_c(P, C, K, 0.25) == pytest.approx((1 - a) / (K * C) * P)
    # k -> 0 limit is P*h/C, and the closed form converges to it.
    assert heat_gain_c(P, C, 0.0, 0.25) == pytest.approx(0.5)
    assert heat_gain_c(P, C, 1e-7, 0.25) == pytest.approx(0.5, rel=1e-6)
    assert heat_gain_c(P, 0.0, K, 1.0) == 0.0


def test_step_matches_continuous_ode():
    # Exact discretisation: one 2 h step == 480 fifteen-second steps.
    one = step_temp_c(30.0, AMB, K, 2.0, power_kw=P, capacity_kwh_per_c=C, on_fraction=1.0)
    t = 30.0
    for _ in range(480):
        t = step_temp_c(t, AMB, K, 2.0 / 480, power_kw=P, capacity_kwh_per_c=C, on_fraction=1.0)
    assert one == pytest.approx(t, abs=1e-9)
    # ... and to a fine Euler integration of dT/dt = P/C - k(T-amb).
    e, dt = 30.0, 1e-4
    for _ in range(int(2.0 / dt)):
        e += dt * net_heat_rate_c_per_h(e, AMB, P, C, K)
    assert one == pytest.approx(e, abs=1e-3)


def test_on_fraction_scales_the_gain_only():
    off = step_temp_c(30.0, AMB, K, 1.0, power_kw=P, capacity_kwh_per_c=C)
    half = step_temp_c(30.0, AMB, K, 1.0, power_kw=P, capacity_kwh_per_c=C, on_fraction=0.5)
    full = step_temp_c(30.0, AMB, K, 1.0, power_kw=P, capacity_kwh_per_c=C, on_fraction=1.0)
    assert half - off == pytest.approx((full - off) / 2)


def test_simulate_mixed_grid():
    hours = [0.1, 0.25, 0.25, 1.0, 1.0]
    on = [1.0, 0.0, 1.0, 1.0]  # shorter than hours: padded with off
    temps = simulate_temps_c(30.0, AMB, K, C, P, hours, on)
    assert len(temps) == len(hours)
    t = 30.0
    for i, h in enumerate(hours):
        t = step_temp_c(
            t, AMB, K, h, power_kw=P, capacity_kwh_per_c=C, on_fraction=on[i] if i < len(on) else 0
        )
        assert temps[i] == pytest.approx(t)


@pytest.mark.parametrize(("hours", "t0", "amb"), [(1.0, 38.0, 10.0), (12.0, 40.0, -5.0)])
def test_free_cooling_equals_water_tank_model(hours: float, t0: float, amb: float):
    # Same Newtonian loss: volume = C*1000/1.1628 litres, UA = k*C*1000 W/K.
    tank = WaterTankModel(volume_litres=C * 1000 / (4186.0 / 3600.0), ua_w_per_k=K * C * 1000)
    expected = tank.temp_after_loss(t0, hours, t_ambient_c=amb)
    assert step_temp_c(t0, amb, K, hours) == pytest.approx(expected, abs=1e-9)


def test_rates():
    assert heat_rate_c_per_h(P, C) == pytest.approx(2.0)
    assert heat_rate_c_per_h(P, 0.0) == 0.0
    assert net_heat_rate_c_per_h(30.0, AMB, P, C, K) == pytest.approx(2.0 - K * 20.0)


def test_hours_to_reach_round_trips_with_max_reachable():
    h = hours_to_reach_c(30.0, 38.0, AMB, P, C, K)
    assert h is not None and h > 4.0  # 8 degC at < 2 degC/h
    assert max_reachable_temp_c(30.0, AMB, P, C, K, h) == pytest.approx(38.0)
    assert hours_to_reach_c(38.0, 38.0, AMB, P, C, K) == 0.0
    assert hours_to_reach_c(40.0, 38.0, AMB, P, C, K) == 0.0


def test_unreachable_targets():
    asymptote = AMB + P / (K * C)  # ~105 degC
    assert hours_to_reach_c(30.0, asymptote, AMB, P, C, K) is None
    assert hours_to_reach_c(30.0, asymptote + 1, AMB, P, C, K) is None
    assert hours_to_reach_c(30.0, 38.0, AMB, 0.0, C, K) is None
    # k ~ 0: linear heating, never unreachable.
    assert hours_to_reach_c(30.0, 38.0, AMB, P, C, 0.0) == pytest.approx(4.0)


def test_max_reachable_is_the_all_on_trajectory():
    assert max_reachable_temp_c(30.0, AMB, P, C, K, 0.0) == 30.0
    reach = max_reachable_temp_c(30.0, AMB, P, C, K, 3.0)
    assert reach == pytest.approx(simulate_temps_c(30.0, AMB, K, C, P, [1.0] * 3, [1.0] * 3)[-1])
    assert max_reachable_temp_c(30.0, AMB, -1.0, C, K, 3.0) == step_temp_c(30.0, AMB, K, 3.0)
