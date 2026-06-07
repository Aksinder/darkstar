"""The S-Index load margin must never fall below the operator's load_safety_margin_percent.

risk_appetite=5 maps to the p25 load percentile, which the uncertainty term can drive toward
~0.5 (planning for half the forecast load) — systematic under-reservation and grid buys. The
floor wires forecasting.load_safety_margin_percent (previously read by no code) as a hard
guarantee; risk_appetite tunes conservatism above it.
"""

import pytest

from planner.inputs.data_prep import floored_load_margin


def test_clamps_aggressive_risk_to_floor():
    cfg = {"forecasting": {"load_safety_margin_percent": 117.0}}
    # risk_appetite=5 produced 0.5 live -> must be lifted to the 1.17 floor.
    assert floored_load_margin(0.5, cfg) == pytest.approx(1.17)


def test_keeps_margin_above_floor():
    cfg = {"forecasting": {"load_safety_margin_percent": 117.0}}
    assert floored_load_margin(1.40, cfg) == pytest.approx(1.40)


def test_default_floor_never_below_forecast():
    # No config -> default 100% -> never plan for less than the forecast load.
    assert floored_load_margin(0.5, {}) == pytest.approx(1.0)


def test_floor_of_one_hundred_percent():
    cfg = {"forecasting": {"load_safety_margin_percent": 100.0}}
    assert floored_load_margin(0.8, cfg) == pytest.approx(1.0)
    assert floored_load_margin(1.3, cfg) == pytest.approx(1.3)
