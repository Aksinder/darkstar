"""
Unit tests for physics-based PV calculation functions.
"""

from datetime import datetime

import pytz

from ml.weather import (
    _calculate_poa_irradiance,  # type: ignore[reportPrivateUsage]
    _calculate_solar_position,  # type: ignore[reportPrivateUsage]
    calculate_physics_for_slots,
    calculate_physics_pv,
    calculate_physics_pv_simple,
)


class TestSolarPosition:
    """Tests for solar position calculation."""

    def test_solar_position_noon_stockholm_summer(self):
        """Test solar position at noon in Stockholm during summer."""
        # June 21, 2024 at 12:00 local time (approximately summer solstice)
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        pos = _calculate_solar_position(59.3, 18.1, dt)

        # At noon on summer solstice, sun should be high in the sky
        assert pos["elevation"] > 50.0, f"Expected high elevation, got {pos['elevation']}"
        assert pos["elevation"] < 60.0, f"Expected elevation < 60, got {pos['elevation']}"
        # Azimuth should be approximately south (180°) at noon
        assert 150 < pos["azimuth"] < 210, f"Expected south-facing azimuth, got {pos['azimuth']}"

    def test_solar_position_noon_stockholm_winter(self):
        """Test solar position at noon in Stockholm during winter."""
        # December 21, 2024 at 12:00 local time (approximately winter solstice)
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 12, 21, 12, 0, 0))

        pos = _calculate_solar_position(59.3, 18.1, dt)

        # At noon on winter solstice, sun should be low in the sky
        assert pos["elevation"] > 0, f"Expected sun above horizon at noon, got {pos['elevation']}"
        assert pos["elevation"] < 15.0, f"Expected low elevation, got {pos['elevation']}"

    def test_solar_position_nighttime(self):
        """Test solar position at night."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 2, 0, 0))

        pos = _calculate_solar_position(59.3, 18.1, dt)

        # Sun should be below horizon at 2 AM
        assert pos["elevation"] < 0, f"Expected sun below horizon, got {pos['elevation']}"


class TestPoaIrradiance:
    """Tests for POA irradiance calculation."""

    def test_poa_horizontal_panel(self):
        """Test POA for horizontal panel."""
        # Horizontal panel (tilt=0) should receive GHI
        poa = _calculate_poa_irradiance(
            radiation_w_m2=800.0,
            panel_tilt=0.0,
            panel_azimuth=0.0,
            solar_elevation=45.0,
            solar_azimuth=180.0,
        )
        # For horizontal panel, POA should be close to GHI
        assert 600 < poa < 900, f"Expected POA close to GHI, got {poa}"

    def test_poa_vertical_panel(self):
        """Test POA for vertical panel."""
        # Vertical south-facing panel
        poa = _calculate_poa_irradiance(
            radiation_w_m2=800.0,
            panel_tilt=90.0,
            panel_azimuth=0.0,  # South
            solar_elevation=45.0,
            solar_azimuth=0.0,  # Sun in south
        )
        # Should receive some irradiance
        assert poa > 0, f"Expected positive POA, got {poa}"

    def test_poa_zero_radiation(self):
        """Test POA with zero radiation."""
        poa = _calculate_poa_irradiance(
            radiation_w_m2=0.0,
            panel_tilt=30.0,
            panel_azimuth=0.0,
            solar_elevation=45.0,
            solar_azimuth=180.0,
        )
        assert poa == 0.0

    def test_poa_sun_below_horizon(self):
        """Test POA when sun is below horizon."""
        poa = _calculate_poa_irradiance(
            radiation_w_m2=800.0,
            panel_tilt=30.0,
            panel_azimuth=0.0,
            solar_elevation=-5.0,
            solar_azimuth=180.0,
        )
        assert poa == 0.0


class TestPoaIrradianceDniDhi:
    """Tests for the corrected DNI/DHI transposition path in _calculate_poa_irradiance."""

    def _morning_ne_geometry(self):
        """Solar geometry for the validated clear morning (Gotland, 2026-07-08 07:00)."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2026, 7, 8, 7, 0, 0))
        pos = _calculate_solar_position(57.6097, 18.4146, dt)
        solar_elevation = pos["elevation"]
        # South-convention solar azimuth as used by calculate_physics_pv
        solar_azimuth = pos["azimuth"] - 180.0
        # NE array: HA azimuth 50 -> South-convention (50 % 360) - 180
        panel_azimuth = (50 % 360) - 180
        return solar_elevation, solar_azimuth, panel_azimuth

    def test_dni_dhi_fixes_low_morning_sun(self):
        """DNI/DHI path reproduces the validated morning POA (~2x the buggy GHI path).

        Clear morning, low elevation (~20°), NE array (tilt 30, HA az 50):
        Open-Meteo GHI=179, DNI=257, DHI=108. The buggy GHI-only transposition
        understates POA; the DNI/DHI path recovers the beam geometry.
        """
        elev, saz, paz = self._morning_ne_geometry()

        new_poa = _calculate_poa_irradiance(
            radiation_w_m2=179.0,
            panel_tilt=30.0,
            panel_azimuth=paz,
            solar_elevation=elev,
            solar_azimuth=saz,
            dni_w_m2=257.0,
            dhi_w_m2=108.0,
        )
        old_poa = _calculate_poa_irradiance(
            radiation_w_m2=179.0,
            panel_tilt=30.0,
            panel_azimuth=paz,
            solar_elevation=elev,
            solar_azimuth=saz,
        )

        # Validated against Open-Meteo GTI (~278 W/m²); allow a generous band.
        assert 260.0 < new_poa < 310.0, f"Expected ~285 W/m² POA, got {new_poa}"
        # The bug understated morning POA ~2-3x.
        assert new_poa > 2.0 * old_poa, f"new={new_poa} not >2x old={old_poa}"

    def test_dni_dhi_overcast_no_phantom_beam(self):
        """Diffuse-only (DNI=0) overcast POA must not exceed GHI (no phantom beam)."""
        elev, saz, paz = self._morning_ne_geometry()

        poa = _calculate_poa_irradiance(
            radiation_w_m2=100.0,  # GHI
            panel_tilt=30.0,
            panel_azimuth=paz,
            solar_elevation=elev,
            solar_azimuth=saz,
            dni_w_m2=0.0,
            dhi_w_m2=100.0,
        )
        assert poa > 0.0
        assert poa <= 100.0, f"Diffuse-only POA {poa} should not exceed GHI 100"

    def test_dni_dhi_zero_returns_zero(self):
        """Both DNI and DHI zero -> no POA even if GHI is passed."""
        elev, saz, paz = self._morning_ne_geometry()
        poa = _calculate_poa_irradiance(
            radiation_w_m2=50.0,
            panel_tilt=30.0,
            panel_azimuth=paz,
            solar_elevation=elev,
            solar_azimuth=saz,
            dni_w_m2=0.0,
            dhi_w_m2=0.0,
        )
        assert poa == 0.0

    def test_none_dni_dhi_matches_legacy_path(self):
        """Regression: with DNI/DHI omitted, the legacy GHI transposition is unchanged."""
        elev, saz, paz = self._morning_ne_geometry()

        # Legacy path via omitted DNI/DHI
        legacy = _calculate_poa_irradiance(
            radiation_w_m2=179.0,
            panel_tilt=30.0,
            panel_azimuth=paz,
            solar_elevation=elev,
            solar_azimuth=saz,
        )
        # Recompute the legacy formula inline to pin the exact behaviour.
        import math

        diffuse_fraction = 0.2 if elev > 15.0 else 0.4
        dni = 179.0 * (1.0 - diffuse_fraction)
        dhi = 179.0 * diffuse_fraction
        cos_aoi = max(
            0.0,
            math.sin(math.radians(elev)) * math.cos(math.radians(30.0))
            + math.cos(math.radians(elev))
            * math.sin(math.radians(30.0))
            * math.cos(math.radians(saz) - math.radians(paz)),
        )
        expected = dni * cos_aoi + dhi * (1.0 + math.cos(math.radians(30.0))) / 2.0
        assert abs(legacy - expected) < 1e-9, f"legacy={legacy} expected={expected}"

    def test_calculate_physics_pv_dni_dhi_raises_morning(self):
        """calculate_physics_pv with DNI/DHI yields higher morning PV than GHI-only."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2026, 7, 8, 7, 0, 0))
        arrays = [{"name": "NE", "kwp": 4.0, "tilt": 30.0, "azimuth": 50.0}]

        with_dni, _ = calculate_physics_pv(
            radiation_w_m2=179.0,
            solar_arrays=arrays,
            slot_start=dt,
            latitude=57.6097,
            longitude=18.4146,
            dni_w_m2=257.0,
            dhi_w_m2=108.0,
        )
        without_dni, _ = calculate_physics_pv(
            radiation_w_m2=179.0,
            solar_arrays=arrays,
            slot_start=dt,
            latitude=57.6097,
            longitude=18.4146,
        )
        assert with_dni is not None and without_dni is not None
        assert with_dni > without_dni

    def test_calculate_physics_pv_nan_dni_falls_back(self):
        """NaN DNI/DHI degrade to the legacy GHI path (no NaN output)."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2026, 7, 8, 7, 0, 0))
        arrays = [{"name": "NE", "kwp": 4.0, "tilt": 30.0, "azimuth": 50.0}]

        nan_result, _ = calculate_physics_pv(
            radiation_w_m2=179.0,
            solar_arrays=arrays,
            slot_start=dt,
            latitude=57.6097,
            longitude=18.4146,
            dni_w_m2=float("nan"),
            dhi_w_m2=float("nan"),
        )
        legacy_result, _ = calculate_physics_pv(
            radiation_w_m2=179.0,
            solar_arrays=arrays,
            slot_start=dt,
            latitude=57.6097,
            longitude=18.4146,
        )
        assert nan_result is not None
        assert nan_result == legacy_result


class TestCalculatePhysicsPv:
    """Tests for calculate_physics_pv function."""

    def test_physics_pv_basic(self):
        """Test basic physics PV calculation."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        solar_arrays = [{"name": "South Roof", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}]

        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=800.0,
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        assert total_kwh is not None
        assert total_kwh > 0, f"Expected positive PV, got {total_kwh}"
        assert len(per_array) == 1
        assert per_array[0]["name"] == "South Roof"
        assert "poa_w_m2" in per_array[0]

    def test_physics_pv_multi_array(self):
        """Test physics PV with multiple arrays."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        solar_arrays = [
            {"name": "East Roof", "kwp": 3.0, "tilt": 30.0, "azimuth": 90.0},
            {"name": "West Roof", "kwp": 3.0, "tilt": 30.0, "azimuth": 270.0},
        ]

        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=800.0,
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        assert total_kwh is not None
        assert total_kwh > 0
        assert len(per_array) == 2
        # Total should be sum of both arrays
        expected_sum = sum(arr["kwh"] for arr in per_array)
        assert abs(total_kwh - expected_sum) < 0.001

    def test_physics_pv_nighttime(self):
        """Test physics PV at night returns None (no radiation)."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 2, 0, 0))

        solar_arrays = [{"name": "South Roof", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}]

        # With 0 radiation, returns None (no production data)
        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=0.0,
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        # When radiation is 0, the function returns None early
        assert total_kwh is None
        assert len(per_array) == 0

    def test_physics_pv_sun_below_horizon(self):
        """Test physics PV when sun is below horizon but radiation > 0 (edge case)."""
        tz = pytz.timezone("Europe/Stockholm")
        # 2 AM in June in Stockholm - sun might still be up at this latitude in summer
        # Let's use winter to ensure sun is down
        dt = tz.localize(datetime(2024, 12, 21, 2, 0, 0))

        solar_arrays = [{"name": "South Roof", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}]

        # Even if we pass radiation > 0, sun being below horizon should return 0
        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=100.0,  # Some radiation data
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        # Sun below horizon -> 0 production
        assert total_kwh == 0.0
        assert len(per_array) == 0

    def test_physics_pv_no_radiation(self):
        """Test physics PV with no radiation data."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        solar_arrays = [{"name": "South Roof", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}]

        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=None,
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        assert total_kwh is None
        assert len(per_array) == 0

    def test_physics_pv_no_arrays(self):
        """Test physics PV with no solar arrays configured."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=800.0,
            solar_arrays=[],
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        assert total_kwh is None
        assert len(per_array) == 0

    def test_physics_pv_zero_kwp_array(self):
        """Test physics PV with zero kwp array (should be skipped)."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        solar_arrays = [
            {"name": "Empty", "kwp": 0.0, "tilt": 30.0, "azimuth": 180.0},
            {"name": "Valid", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0},
        ]

        total_kwh, per_array = calculate_physics_pv(
            radiation_w_m2=800.0,
            solar_arrays=solar_arrays,
            slot_start=dt,
            latitude=59.3,
            longitude=18.1,
        )

        assert total_kwh is not None
        assert len(per_array) == 1  # Only valid array
        assert per_array[0]["name"] == "Valid"


class TestCalculatePhysicsPvSimple:
    """Tests for calculate_physics_pv_simple fallback function."""

    def test_simple_pv_basic(self):
        """Test simple physics PV calculation."""
        pv = calculate_physics_pv_simple(
            radiation_w_m2=800.0,
            total_capacity_kw=10.0,
        )
        assert pv is not None
        # Formula: (800/1000) * 10 * 0.85 * 0.25 = 1.7 kWh
        expected = (800.0 / 1000.0) * 10.0 * 0.85 * 0.25
        assert abs(pv - expected) < 0.01

    def test_simple_pv_no_radiation(self):
        """Test simple physics PV with no radiation."""
        pv = calculate_physics_pv_simple(
            radiation_w_m2=None,
            total_capacity_kw=10.0,
        )
        assert pv is None

    def test_simple_pv_no_capacity(self):
        """Test simple physics PV with no capacity."""
        pv = calculate_physics_pv_simple(
            radiation_w_m2=800.0,
            total_capacity_kw=0.0,
        )
        assert pv is None

    def test_simple_pv_custom_efficiency(self):
        """Test simple physics PV with custom efficiency."""
        pv = calculate_physics_pv_simple(
            radiation_w_m2=800.0,
            total_capacity_kw=10.0,
            efficiency=0.75,
        )
        assert pv is not None
        # Lower efficiency = lower output
        pv_default = calculate_physics_pv_simple(
            radiation_w_m2=800.0,
            total_capacity_kw=10.0,
        )
        assert pv_default is not None
        assert pv < pv_default


class TestCalculatePhysicsForSlots:
    """Tests for calculate_physics_for_slots function."""

    def test_physics_for_slots_basic(self):
        """Test calculating physics for multiple slots."""
        tz = pytz.timezone("Europe/Stockholm")
        dt1 = tz.localize(datetime(2024, 6, 21, 10, 0, 0))
        dt2 = tz.localize(datetime(2024, 6, 21, 12, 0, 0))
        dt3 = tz.localize(datetime(2024, 6, 21, 14, 0, 0))

        slots = [
            {"slot_start": dt1.isoformat(), "shortwave_radiation_w_m2": 600.0},
            {"slot_start": dt2.isoformat(), "shortwave_radiation_w_m2": 800.0},
            {"slot_start": dt3.isoformat(), "shortwave_radiation_w_m2": 700.0},
        ]

        config = {
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_arrays": [{"name": "South", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}],
            }
        }

        results = calculate_physics_for_slots(slots, config)

        assert len(results) == 3
        for r in results:
            assert "physics_kwh" in r
            assert "physics_arrays" in r
            # All should have positive physics during daytime
            if r.get("physics_kwh") is not None:
                assert r["physics_kwh"] >= 0

    def test_physics_for_slots_no_arrays(self):
        """Test calculating physics with no arrays configured."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        slots = [
            {"slot_start": dt.isoformat(), "shortwave_radiation_w_m2": 800.0},
        ]

        config = {
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
            }
        }

        results = calculate_physics_for_slots(slots, config)

        assert len(results) == 1
        assert results[0]["physics_kwh"] is None

    def test_physics_for_slots_missing_radiation(self):
        """Test calculating physics with missing radiation data."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        slots = [
            {"slot_start": dt.isoformat()},  # No radiation
        ]

        config = {
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_arrays": [{"name": "South", "kwp": 5.0, "tilt": 30.0, "azimuth": 180.0}],
            }
        }

        results = calculate_physics_for_slots(slots, config)

        assert len(results) == 1
        # Should return None when radiation is missing
        assert results[0]["physics_kwh"] is None or results[0]["physics_kwh"] == 0.0

    def test_physics_for_slots_legacy_array(self):
        """Test calculating physics with legacy single array config."""
        tz = pytz.timezone("Europe/Stockholm")
        dt = tz.localize(datetime(2024, 6, 21, 12, 0, 0))

        slots = [
            {"slot_start": dt.isoformat(), "shortwave_radiation_w_m2": 800.0},
        ]

        config = {
            "system": {
                "location": {"latitude": 59.3, "longitude": 18.1},
                "solar_array": {"kwp": 6.0, "tilt": 25.0, "azimuth": 180.0},
            }
        }

        results = calculate_physics_for_slots(slots, config)

        assert len(results) == 1
        assert results[0]["physics_kwh"] is not None
        assert results[0]["physics_kwh"] > 0
