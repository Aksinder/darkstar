"""Charge-block soc_target during a planned partial recovery below min_soc.

Ported with upstream ergetie/darkstar 8a234bb0: a charge block whose projected
end SoC is still below min_soc (kepler's recovery window) must not have its target
raised to min_soc, or the executor charges beyond the plan.
"""

import pandas as pd

from planner.output.soc_target import apply_soc_target_percent

CONFIG = {"battery": {"min_soc_percent": 12.0, "max_soc_percent": 100.0, "capacity_kwh": 16.0}}


def _df(projected: list[float], actions: list[str], entry: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-10-21 00:00", periods=len(actions), freq="15min")
    return pd.DataFrame(
        {"action": actions, "projected_soc_percent": projected, "_entry_soc_percent": entry},
        index=index,
    )


def test_partial_recovery_block_keeps_projected_target():
    df = _df(
        projected=[9.0, 10.0, 11.0, 11.0],
        actions=["Charge", "Charge", "Charge", "Hold"],
        entry=[8.0, 9.0, 10.0, 11.0],
    )
    out = apply_soc_target_percent(df, CONFIG)
    assert out["soc_target_percent"].iloc[:3].tolist() == [11.0, 11.0, 11.0]


def test_block_reaching_above_min_is_unchanged():
    df = _df(
        projected=[20.0, 30.0, 40.0],
        actions=["Charge", "Charge", "Charge"],
        entry=[15.0, 20.0, 30.0],
    )
    out = apply_soc_target_percent(df, CONFIG)
    assert out["soc_target_percent"].tolist() == [40.0, 40.0, 40.0]


def test_missing_projection_still_floors_at_min():
    df = _df(
        projected=[None, None],  # type: ignore[list-item]
        actions=["Charge", "Charge"],
        entry=[8.0, 9.0],
    )
    out = apply_soc_target_percent(df, CONFIG)
    assert min(out["soc_target_percent"].tolist()) >= 12.0
