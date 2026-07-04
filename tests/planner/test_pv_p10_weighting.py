"""Tests for per-slot p10-weighted PV planning (forecasting.pv_p10_weight)."""

import pandas as pd
import pytest

from planner.inputs.data_prep import apply_safety_margins


def _df(pv=(4.0, 2.0), p10=(3.0, 0.5)):
    return pd.DataFrame(
        {
            "pv_forecast_kwh": list(pv),
            "pv_p10": list(p10),
            "load_forecast_kwh": [1.0] * len(pv),
        }
    )


def _cfg(**forecasting):
    base = {"pv_confidence_percent": 80.0}
    base.update(forecasting)
    return {"forecasting": base}


def test_weight_zero_keeps_legacy_scalar_haircut():
    df = apply_safety_margins(_df(), _cfg(), {}, 1.0)
    assert df["adjusted_pv_kwh"].tolist() == pytest.approx([3.2, 1.6])  # x0.8


def test_blend_discounts_uncertain_slots_harder():
    """w=0.5: slot with tight band (4->3) loses less than the wide-band slot (2->0.5)."""
    df = apply_safety_margins(_df(), _cfg(pv_p10_weight=0.5), {}, 1.0)
    # slot0: 0.5*4 + 0.5*3 = 3.5 ; slot1: 0.5*2 + 0.5*0.5 = 1.25
    assert df["adjusted_pv_kwh"].tolist() == pytest.approx([3.5, 1.25])


def test_missing_p10_falls_back_to_scalar_per_slot():
    df = _df(pv=(4.0, 2.0), p10=(3.0, None))
    out = apply_safety_margins(df, _cfg(pv_p10_weight=0.5), {}, 1.0)
    assert out["adjusted_pv_kwh"].tolist() == pytest.approx([3.5, 1.6])  # blend, scalar


def test_weight_clamped_to_one():
    df = apply_safety_margins(_df(), _cfg(pv_p10_weight=5.0), {}, 1.0)
    assert df["adjusted_pv_kwh"].tolist() == pytest.approx([3.0, 0.5])  # pure p10


def test_no_p10_column_is_legacy():
    df = pd.DataFrame({"pv_forecast_kwh": [4.0], "load_forecast_kwh": [1.0]})
    out = apply_safety_margins(df, _cfg(pv_p10_weight=0.5), {}, 1.0)
    assert out["adjusted_pv_kwh"].tolist() == pytest.approx([3.2])
