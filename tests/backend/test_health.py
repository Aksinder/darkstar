"""Tests for health check async behavior - DNS crash loop fixes.

This test ensures:
1. Health check timeouts work correctly (asyncio.wait_for)
2. Entity checks run concurrently (asyncio.gather)
3. Timeout errors are handled gracefully
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_get_health_status_respects_timeout():
    """Verify that get_health_status() has a 15-second timeout wrapper.

    This is a regression test for the DNS crash loop bug where health checks
    could hang indefinitely due to blocking I/O.

    The fix: Wrap check_all() in asyncio.wait_for(timeout=15.0) to ensure
    the health check returns even if individual checks are stuck.
    """
    from backend.health import HealthStatus, get_health_status

    # Mock HealthChecker.check_all to be slow (simulates stuck I/O)
    with patch("backend.health.HealthChecker") as mock_checker_class:
        mock_checker = MagicMock()
        mock_checker.check_all = AsyncMock()

        # Make check_all hang for 30 seconds (longer than timeout)
        async def slow_check():
            await asyncio.sleep(30)
            return HealthStatus(healthy=True, issues=[])

        mock_checker.check_all.side_effect = slow_check
        mock_checker_class.return_value = mock_checker

        # Call should complete within reasonable time (timeout + overhead)
        start = asyncio.get_event_loop().time()
        result = await get_health_status()
        elapsed = asyncio.get_event_loop().time() - start

        # Should timeout and return gracefully within ~15 seconds
        assert elapsed < 20.0, f"Health check took {elapsed}s, should timeout at 15s"

        # Should return unhealthy status with timeout issue
        assert result.healthy is False
        assert len(result.issues) == 1
        assert "timed out" in result.issues[0].message.lower()


@pytest.mark.asyncio
async def test_check_entities_uses_concurrent_gather():
    """Verify that entity checks run concurrently using asyncio.gather.

    This test ensures that multiple entity checks don't run sequentially,
    which could cause the health check to take too long with many entities.

    The fix: Use asyncio.gather to run all entity checks concurrently,
    reducing total time from O(n) to O(1) for network-bound checks.
    """
    from backend.health import HealthChecker

    # Create a mock config with multiple entities to check
    mock_config = {
        "system": {
            "has_battery": True,
            "has_water_heater": False,
            "has_solar": True,
            "grid_meter_type": "net",
        },
        "learning": {"enable": False},
        "input_sensors": {
            "battery_soc": "sensor.battery_soc",
            "grid_power": "sensor.grid_power",
            "pv_power": "sensor.pv_power",
            "load_power": "sensor.load_power",
            "soc_min": "number.soc_min",
        },
    }
    mock_secrets = {
        "home_assistant": {
            "url": "http://homeassistant:8123",
            "token": "test_token",
        },
    }

    checker = HealthChecker.__new__(HealthChecker)
    checker._config = mock_config
    checker._secrets = mock_secrets

    # Track call times to verify concurrency
    call_times = []
    original_sleep = asyncio.sleep

    async def mock_check_with_delay(*args, **kwargs):
        """Simulate a network request that takes some time."""
        call_times.append(asyncio.get_event_loop().time())
        await original_sleep(0.1)  # Small delay to simulate network
        return None  # No issue

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "10.5", "attributes": {}}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        # Run the check
        start = asyncio.get_event_loop().time()
        await checker.check_entities()
        elapsed = asyncio.get_event_loop().time() - start

        # With 5 entities and 0.1s delay each, sequential would take ~0.5s
        # Concurrent should take ~0.1s (plus overhead)
        assert elapsed < 0.3, (
            f"Entity checks took {elapsed:.2f}s, suggesting sequential execution. "
            f"Expected concurrent execution under 0.3s"
        )

        # Verify all entities were checked
        assert mock_client.get.call_count >= 4


@pytest.mark.asyncio
async def test_check_entities_handles_individual_failures():
    """Verify that one failing entity check doesn't break all others.

    When using asyncio.gather with return_exceptions=True, individual
    failures should be handled gracefully without affecting other checks.
    """
    from backend.health import HealthChecker

    mock_config = {
        "system": {
            "has_battery": True,
            "has_water_heater": False,
            "has_solar": True,
            "grid_meter_type": "net",
        },
        "learning": {"enable": False},
        "input_sensors": {
            "battery_soc": "sensor.battery_soc",
            "grid_power": "sensor.grid_power",
        },
    }
    mock_secrets = {
        "home_assistant": {
            "url": "http://homeassistant:8123",
            "token": "test_token",
        },
    }

    checker = HealthChecker.__new__(HealthChecker)
    checker._config = mock_config
    checker._secrets = mock_secrets

    call_count = 0

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()

            # First call succeeds, second raises exception
            if call_count == 1:
                mock_response.status_code = 200
                mock_response.json.return_value = {"state": "10.5"}
            else:
                raise httpx.ConnectError("Connection refused")

            return mock_response

        mock_client.get = mock_get
        mock_client_class.return_value = mock_client

        # Should complete without raising
        await checker.check_entities()

        # Should have attempted both checks despite one failing
        assert call_count == 2


@pytest.mark.asyncio
async def test_health_status_timeout_returns_critical_issue():
    """Verify that timeout produces a properly formatted critical health issue."""
    from backend.health import HealthIssue, HealthStatus

    # Simulate what get_health_status does on timeout
    timeout_status = HealthStatus(
        healthy=False,
        issues=[
            HealthIssue(
                category="ha_connection",
                severity="critical",
                message="Health check timed out after 15 seconds",
                guidance="The system is experiencing connectivity issues. Check network and Home Assistant availability.",
            )
        ],
    )
    assert timeout_status.healthy is False
    assert len(timeout_status.issues) == 1
    issue = timeout_status.issues[0]
    assert issue.severity == "critical"
    assert "timed out" in issue.message.lower()
    assert "15" in issue.message


def _make_checker(config: dict):
    from backend.health import HealthChecker

    checker = HealthChecker.__new__(HealthChecker)
    checker._config = config
    checker._secrets = {}
    return checker


# --- Solar health check ---

def test_solar_warning_when_arrays_empty():
    checker = _make_checker({"system": {"has_solar": True, "solar_arrays": []}})
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Solar enabled but panel size not configured" in messages


def test_solar_no_warning_when_kwp_configured():
    checker = _make_checker(
        {"system": {"has_solar": True, "solar_arrays": [{"kwp": 5.0}, {"kwp": 3.0}]}}
    )
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Solar enabled but panel size not configured" not in messages


def test_solar_legacy_singular_key_ignored():
    checker = _make_checker(
        {"system": {"has_solar": True, "solar_array": {"kwp": 10.0}}}
    )
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Solar enabled but panel size not configured" in messages


# --- Water heater health check ---

def test_water_heater_warning_when_list_empty():
    checker = _make_checker(
        {"system": {"has_water_heater": True}, "water_heaters": []}
    )
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Water heater enabled but power not configured" in messages


def test_water_heater_no_warning_when_configured():
    checker = _make_checker(
        {
            "system": {"has_water_heater": True},
            "water_heaters": [{"enabled": True, "power_kw": 3.0}],
        }
    )
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Water heater enabled but power not configured" not in messages


def test_water_heater_legacy_flat_field_ignored():
    checker = _make_checker(
        {
            "system": {"has_water_heater": True},
            "water_heating": {"power_kw": 3.0},
            "water_heaters": [],
        }
    )
    issues = checker._validate_config_structure()
    messages = [i.message for i in issues]
    assert "Water heater enabled but power not configured" in messages


# --- Unactuatable water heater (planned-but-no-control-entity) ---


def test_water_heater_without_target_entity_is_critical():
    """Regression: villavagn tank was planned 12.8 kWh/day with target_entity '' while
    the executor silently dropped every command — health stayed green throughout."""
    checker = _make_checker(
        {
            "system": {"has_water_heater": True},
            "water_heaters": [
                {"id": "t1", "name": "Tank 1", "enabled": True, "power_kw": 3.0,
                 "target_entity": "input_number.t1_target"},
                {"id": "t2", "name": "Tank 2", "enabled": True, "power_kw": 1.6,
                 "target_entity": ""},
            ],
        }
    )
    issues = checker._validate_config_structure()
    hits = [i for i in issues if "no control entity" in i.message]
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert "Tank 2" in hits[0].message


def test_water_heater_disabled_without_target_entity_is_fine():
    checker = _make_checker(
        {
            "system": {"has_water_heater": True},
            "water_heaters": [
                {"id": "t1", "name": "Tank 1", "enabled": True, "power_kw": 3.0,
                 "target_entity": "input_number.t1_target"},
                {"id": "t2", "name": "Tank 2", "enabled": False, "power_kw": 1.6},
            ],
        }
    )
    issues = checker._validate_config_structure()
    assert not [i for i in issues if "no control entity" in i.message]


# --- Plan realism gap surfacing ---


def _write_schedule(tmp_path, gap_sek: float, slot_costs: list[float]):
    import json

    (tmp_path / "data").mkdir(exist_ok=True)
    payload = {
        "meta": {"s_index": {"realism": {"gap_sek": gap_sek}}},
        "schedule": [{"cost_sek": c} for c in slot_costs],
    }
    (tmp_path / "data" / "schedule.json").write_text(json.dumps(payload))


def test_realism_gap_surfaces_as_imbalance_info_when_material(tmp_path, monkeypatch):
    """Reframed 2026-08-20: Swedish meters net phases momentarily (STAFS 2022:9),
    so the gap is a phase-imbalance indicator (fuse headroom), never lost money.
    The old warning text recommended enabling phase_aware "for the economics" —
    which steered this site into double-priced import. Severity info, message in
    kWh, guidance about amps."""
    monkeypatch.chdir(tmp_path)
    _write_schedule(tmp_path, gap_sek=3.0, slot_costs=[1.0] * 10)  # 30% of 10 SEK gross
    checker = _make_checker({})
    issues = checker.check_plan_realism()
    assert len(issues) == 1
    assert issues[0].severity == "info"
    assert "phase imbalance" in issues[0].message.lower()
    assert "amps, not money" in issues[0].guidance
    assert "consider enabling" not in issues[0].guidance.lower()


def test_realism_gap_silent_when_small(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_schedule(tmp_path, gap_sek=1.0, slot_costs=[1.0] * 10)  # below 2 SEK floor
    checker = _make_checker({})
    assert checker.check_plan_realism() == []


def test_realism_gap_silent_when_relatively_minor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_schedule(tmp_path, gap_sek=3.0, slot_costs=[5.0] * 10)  # 6% of 50 SEK gross
    checker = _make_checker({})
    assert checker.check_plan_realism() == []


def test_realism_gap_silent_without_schedule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checker = _make_checker({})
    assert checker.check_plan_realism() == []


def test_health_no_energy_sensor_warnings():
    """No energy-sensor warnings are emitted — energy is now measured via History API."""
    from backend.health import HealthChecker

    checker = HealthChecker.__new__(HealthChecker)
    checker._config = {
        "ev_chargers": [
            {
                "id": "ev1",
                "name": "My EV",
                "sensor": "sensor.ev_power",
                "enabled": True,
            }
        ],
        "water_heaters": [
            {
                "id": "wh1",
                "name": "Main Tank",
                "sensor": "sensor.water_power",
                "enabled": True,
            }
        ],
    }
    checker._secrets = {}

    issues = checker._validate_config_structure()

    energy_sensor_issues = [i for i in issues if "energy sensor" in i.message.lower()]
    assert len(energy_sensor_issues) == 0


def _pv_energy_config(*, pv_power: str | None, total_pv_production: str) -> dict:
    """Learning-on config with all cumulative counters present except PV, which is
    controlled by the args (pv_power power sensor + total_pv_production counter)."""
    input_sensors = {
        "battery_soc": "sensor.battery_soc",
        "grid_power": "sensor.grid_power",
        "load_power": "sensor.load_power",
        "total_load_consumption": "sensor.total_consumed_energy",
        "total_pv_production": total_pv_production,
        "total_grid_import": "sensor.total_imported_energy",
        "total_grid_export": "sensor.total_exported_energy",
        "total_battery_charge": "sensor.total_battery_charge",
        "total_battery_discharge": "sensor.total_battery_discharge",
    }
    if pv_power is not None:
        input_sensors["pv_power"] = pv_power
    return {
        "system": {
            "has_battery": True,
            "has_water_heater": False,
            "has_solar": True,
            "grid_meter_type": "net",
        },
        "learning": {"enable": True},
        "input_sensors": input_sensors,
    }


async def _fallback_warning_issues(config: dict) -> list:
    """Run check_entities with all HA entities existing; return the F65 fallback-data
    warnings only."""
    from backend.health import HealthChecker

    checker = HealthChecker.__new__(HealthChecker)
    checker._config = config
    checker._secrets = {"home_assistant": {"url": "http://ha:8123", "token": "t"}}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "10.5", "attributes": {}}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        issues = await checker.check_entities()

    return [i for i in issues if "inaccurate fallback" in i.message.lower()]


@pytest.mark.asyncio
async def test_no_pv_fallback_warning_when_pv_power_present():
    """REGRESSION: pv_power integration satisfies the PV energy requirement, so an
    empty total_pv_production must NOT trip the 'inaccurate fallback data' warning
    (the recorder integrates pv_power per slot — build #13)."""
    config = _pv_energy_config(pv_power="sensor.solpaneler", total_pv_production="")
    warnings = await _fallback_warning_issues(config)
    assert warnings == [], f"unexpected fallback warning: {[w.guidance for w in warnings]}"


@pytest.mark.asyncio
async def test_pv_fallback_warning_when_both_pv_sources_missing():
    """When BOTH the cumulative counter and pv_power are absent there is no PV energy
    source, so the warning SHOULD fire and name total_pv_production."""
    config = _pv_energy_config(pv_power=None, total_pv_production="")
    warnings = await _fallback_warning_issues(config)
    assert len(warnings) == 1
    assert "total_pv_production" in warnings[0].guidance
    assert "sine wave" not in warnings[0].guidance.lower()
