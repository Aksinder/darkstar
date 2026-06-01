import pandas as pd

from planner.strategy.s_index import (
    calculate_bridging_reserve,
    calculate_deficit_ratio,
    calculate_safety_floor,
    calculate_temporal_deficit,
)


def test_calculate_deficit_ratio():
    # Scenario 1: Load > PV (Deficit)
    # Load: 100, PV: 60 -> Deficit: 40 -> Ratio: 0.4
    assert calculate_deficit_ratio(100.0, 60.0) == 0.4

    # Scenario 2: PV > Load (Surplus)
    # Load: 100, PV: 120 -> Deficit: -20 -> Ratio: 0.0
    assert calculate_deficit_ratio(100.0, 120.0) == 0.0

    # Scenario 3: Zero Load
    assert calculate_deficit_ratio(0.0, 10.0) == 0.0

    # Scenario 4: Max Ratio (Zero PV)
    assert calculate_deficit_ratio(100.0, 0.0) == 1.0


def test_calculate_temporal_deficit():
    """Test temporal deficit calculation (sum of max(0, load - pv) per slot)."""
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [5.0, 10.0, 3.0],
            "pv_forecast_kwh": [2.0, 5.0, 5.0],
        }
    )

    # Expected: (5-2) + (10-5) + max(0, 3-5) = 3 + 5 + 0 = 8.0
    deficit = calculate_temporal_deficit(df)
    assert deficit == 8.0


def test_bridging_reserve_equals_deficit_for_pure_night():
    """With no intervening surplus, the bridging reserve equals the naive
    temporal deficit (cumulative balance only ever decreases)."""
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [1.0, 2.0, 1.5],
            "pv_forecast_kwh": [0.0, 0.0, 0.0],
        }
    )
    # net = [-1, -2, -1.5], cumulative = [-1, -3, -4.5], deepest dip = 4.5
    assert calculate_bridging_reserve(df) == 4.5
    assert calculate_bridging_reserve(df) == calculate_temporal_deficit(df)


def test_bridging_reserve_credits_incoming_surplus():
    """Surplus that arrives before a deficit cancels it out, so the bridging
    reserve is strictly smaller than the naive sum-of-deficits. This is the
    'enough sun is coming' case — the planner should not over-reserve."""
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [0.0, 0.0, 3.0, 3.0],
            "pv_forecast_kwh": [5.0, 5.0, 0.0, 0.0],
        }
    )
    # net = [+5, +5, -3, -3], cumulative = [5, 10, 7, 4]; never dips below 0.
    assert calculate_bridging_reserve(df) == 0.0
    # Naive deficit would have demanded a 6.0 kWh reserve.
    assert calculate_temporal_deficit(df) == 6.0


def test_bridging_reserve_deficit_then_surplus():
    """When the deficit comes first, surplus afterwards cannot retroactively
    cover it — the reserve still reflects the early dip."""
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [3.0, 3.0, 0.0, 0.0],
            "pv_forecast_kwh": [0.0, 0.0, 5.0, 5.0],
        }
    )
    # net = [-3, -3, +5, +5], cumulative = [-3, -6, -1, 4], deepest dip = 6
    assert calculate_bridging_reserve(df) == 6.0


def test_bridging_reserve_empty_df():
    """Empty DataFrame returns 0."""
    df = pd.DataFrame({"load_forecast_kwh": [], "pv_forecast_kwh": []})
    assert calculate_bridging_reserve(df) == 0.0


def test_calculate_safety_floor_defaults():
    """Test safety floor with temporal deficit calculation."""
    # Base Config
    battery_config = {"capacity_kwh": 10.0, "min_soc_percent": 10.0}  # Min 1.0 kWh
    s_index_cfg = {
        "risk_appetite": 3,
        "max_safety_buffer_percent": 50.0,
    }  # Neutral: 15% margin, 10% min buffer

    # DataFrame with temporal deficit
    # 2 slots: each with 5 kWh load, 2 kWh PV
    # Temporal deficit = (5-2) + (5-2) = 6 kWh
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [5.0, 5.0],
            "pv_forecast_kwh": [2.0, 2.0],
        }
    )

    floor, debug = calculate_safety_floor(df, battery_config, s_index_cfg, "UTC")

    # Expected calculation:
    # Temporal deficit = 6.0 kWh
    # Risk 3 margin = 1.15
    # Base reserve = 6.0 * 1.15 = 6.9 kWh
    # Min buffer = 10% of 10 kWh = 1.0 kWh
    # Max buffer = 50% of 10 kWh = 5.0 kWh
    # Effective reserve = min(6.9, 5.0) = 5.0 kWh (capped)
    # Floor = 1.0 (min_soc) + 5.0 = 6.0 kWh
    expected_floor = 6.0

    assert floor == expected_floor
    assert debug["method"] == "temporal_deficit"
    assert debug["temporal_deficit_kwh"] == 6.0
    assert debug["risk_appetite"] == 3


def test_calculate_safety_floor_weather_adders():
    """Test safety floor with weather adders."""
    battery_config = {"capacity_kwh": 10.0, "min_soc_percent": 10.0}  # Min 1.0 kWh
    s_index_cfg = {"risk_appetite": 3, "max_safety_buffer_percent": 50.0}

    # DataFrame with PV surplus (no temporal deficit) but cold/cloudy weather
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [3.0],
            "pv_forecast_kwh": [5.0],  # PV > Load, no temporal deficit
            "temperature_c": [-5.0],  # Adder +1.0
            "snow_prob": [0.0],
            "cloud_cover": [80.0],  # Adder +0.5
        }
    )

    floor, debug = calculate_safety_floor(df, battery_config, s_index_cfg, "UTC")

    # Expected:
    # Temporal deficit = 0 (PV > Load in all slots)
    # Min buffer = 10% of 10 kWh = 1.0 kWh
    # Weather Buffer = 1.0 (Temp) + 0.5 (Cloud) = 1.5 kWh
    # Floor = 1.0 (min_soc) + max(0, 1.0) + 1.5 = 3.5 kWh
    expected_floor = 3.5

    assert floor == expected_floor
    assert debug["weather_buffer_kwh"] == 1.5


def test_calculate_safety_floor_risk_multipliers():
    """Test safety floor with different risk levels."""
    battery_config = {"capacity_kwh": 10.0, "min_soc_percent": 0.0}

    # DataFrame with temporal deficit
    # 2 slots: each with 5 kWh load, 2 kWh PV
    # Temporal deficit = (5-2) + (5-2) = 6 kWh
    df = pd.DataFrame(
        {
            "load_forecast_kwh": [5.0, 5.0],
            "pv_forecast_kwh": [2.0, 2.0],
        }
    )

    # Max buffer = 50% of 10 kWh = 5.0 kWh for all tests

    # Risk 1 (Safety): 30% margin, 25% min buffer
    # Base reserve = 6.0 * 1.30 = 7.8, capped at 5.0
    # Min buffer = 2.5 kWh
    # Floor = max(5.0, 2.5) = 5.0 kWh
    cfg_risk1 = {"risk_appetite": 1, "max_safety_buffer_percent": 50.0}
    floor1, _ = calculate_safety_floor(df, battery_config, cfg_risk1, "UTC")
    assert floor1 == 5.0

    # Risk 5 (Gambler): 0% margin, 0% min buffer
    # Base reserve = 6.0 * 1.00 = 6.0, capped at 5.0
    # Min buffer = 0 kWh
    # Floor = max(5.0, 0) = 5.0 kWh
    cfg_risk5 = {"risk_appetite": 5, "max_safety_buffer_percent": 50.0}
    floor5, _ = calculate_safety_floor(df, battery_config, cfg_risk5, "UTC")
    assert floor5 == 5.0

    # Actually, with temporal deficit capped at max_buffer, both hit the cap.
    # Let's test with smaller deficit to see risk difference
    df_small = pd.DataFrame(
        {
            "load_forecast_kwh": [2.0, 2.0],
            "pv_forecast_kwh": [1.0, 1.0],
        }
    )
    # Temporal deficit = 2.0 kWh

    floor1_small, _ = calculate_safety_floor(df_small, battery_config, cfg_risk1, "UTC")
    floor5_small, _ = calculate_safety_floor(df_small, battery_config, cfg_risk5, "UTC")

    # Risk 1: 2.0 * 1.30 = 2.6 kWh + min buffer 2.5 = 5.1, capped at 5.0
    # Risk 5: 2.0 * 1.00 = 2.0 kWh + min buffer 0 = 2.0
    # Actually, the effective reserve = max(temporal_deficit * margin, min_buffer)
    # So Risk 1: max(2.6, 2.5) = 2.6 -> floor = 2.6
    # Risk 5: max(2.0, 0) = 2.0 -> floor = 2.0

    assert floor1_small > floor5_small
