"""A HELD (unreadable) SoC rolls forward while the charger delivers.

A parked car's SoC does not fall, so the hold was safe for a parked car — but a
charging car's SoC rises, and a frozen value blinds the deadline floor and the
target cap. Fresh readings always win.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.executor.test_blind_wake import T0, FakeHA, _ctrl

SEEN = {"binary_sensor.tesla_plug": "on", "sensor.tesla_soc": "40", "sensor.tesla_power": "0"}


def _read(ctrl: Any, states: dict[str, str], now: float) -> Any:
    return asyncio.run(ctrl._read_charger(FakeHA(states), ctrl.cfg.chargers[0], now, False))


def _blind(power_w: float) -> dict[str, str]:
    return {
        "binary_sensor.tesla_plug": "on",
        "sensor.tesla_soc": "unavailable",
        "sensor.tesla_power": str(power_w),
    }


def test_held_soc_rolls_forward_by_delivered_energy() -> None:
    ctrl = _ctrl(capacity_kwh=60.0, charge_efficiency=0.9)
    assert _read(ctrl, SEEN, T0).soc_percent == 40.0
    # Hold begins; 11 kW for one hour in 60 s ticks = 11 kWh plug -> 9.9 kWh battery
    # = 16.5 SoC points on 60 kWh.
    st = _read(ctrl, _blind(11000.0), T0 + 60)
    assert st.soc_percent == 40.0  # first held tick: no interval to credit yet
    for i in range(2, 62):
        st = _read(ctrl, _blind(11000.0), T0 + 60 * i)
    assert st.soc_percent == pytest.approx(56.5, abs=0.01)


def test_idle_car_keeps_the_plain_hold() -> None:
    ctrl = _ctrl(capacity_kwh=60.0)
    _read(ctrl, SEEN, T0)
    for i in range(1, 30):
        st = _read(ctrl, _blind(0.0), T0 + 60 * i)
    assert st.soc_percent == 40.0


def test_rolled_soc_caps_at_100() -> None:
    ctrl = _ctrl(capacity_kwh=10.0, charge_efficiency=1.0)
    _read(ctrl, {**SEEN, "sensor.tesla_soc": "95"}, T0)
    for i in range(1, 20):
        st = _read(ctrl, _blind(11000.0), T0 + 60 * i)
    assert st.soc_percent == 100.0


def test_long_gap_is_not_credited() -> None:
    ctrl = _ctrl(capacity_kwh=60.0, charge_efficiency=1.0)
    _read(ctrl, SEEN, T0)
    _read(ctrl, _blind(6000.0), T0 + 60)
    st = _read(ctrl, _blind(6000.0), T0 + 60 + 3600)  # stalled loop: credit <= 120 s
    assert st.soc_percent == pytest.approx(40.0 + 6.0 * 120 / 3600 / 60 * 100, abs=0.01)


def test_fresh_reading_wins_and_resets_the_roll() -> None:
    ctrl = _ctrl(capacity_kwh=60.0, charge_efficiency=1.0)
    _read(ctrl, SEEN, T0)
    for i in range(1, 30):
        _read(ctrl, _blind(11000.0), T0 + 60 * i)
    st = _read(ctrl, {**SEEN, "sensor.tesla_soc": "47"}, T0 + 1800)
    assert st.soc_percent == 47.0
    # A new blackout starts from the fresh reading, not from the old accumulator.
    st = _read(ctrl, _blind(0.0), T0 + 1860)
    assert st.soc_percent == 47.0


def test_without_capacity_hold_is_dropped_after_30_min_of_drawing() -> None:
    ctrl = _ctrl()  # capacity_kwh defaults to 0
    _read(ctrl, SEEN, T0)
    st = None
    for i in range(1, 31):  # 29 credited minutes of drawing
        st = _read(ctrl, _blind(7000.0), T0 + 60 * i)
    assert st is not None and st.soc_percent == 40.0
    for i in range(31, 34):
        st = _read(ctrl, _blind(7000.0), T0 + 60 * i)
    assert st.soc_percent is None
    # ...and a fresh reading restores trust immediately.
    assert _read(ctrl, {**SEEN, "sensor.tesla_soc": "52"}, T0 + 60 * 35).soc_percent == 52.0
