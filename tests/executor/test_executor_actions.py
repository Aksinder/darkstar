"""
Tests for Executor Actions (HAClient and ActionDispatcher)

Tests with mocked HTTP requests to avoid needing a live Home Assistant instance.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from executor.actions import ActionResult, HACallError, HAClient


class TestActionResult:
    """Test the ActionResult dataclass."""

    def test_required_fields(self):
        """ActionResult requires action_type and success."""
        result = ActionResult(action_type="work_mode", success=True)
        assert result.action_type == "work_mode"
        assert result.success is True

    def test_default_values(self):
        """ActionResult has sensible defaults."""
        result = ActionResult(action_type="test", success=True)
        assert result.message == ""
        assert result.previous_value is None
        assert result.new_value is None
        assert result.skipped is False
        assert result.duration_ms == 0


class TestHAClientGetState:
    """Test HAClient.get_state and get_state_value."""

    @pytest.mark.asyncio
    async def test_get_state_success(self):
        """get_state returns parsed JSON on success."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={
                "entity_id": "switch.test",
                "state": "on",
            }
        )
        mock_response.raise_for_status = MagicMock()

        # Create mock session that returns mock_response as context manager
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm

        # Patch _get_session to return our mock
        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.get_state("switch.test")
            assert result == {"entity_id": "switch.test", "state": "on"}

    @pytest.mark.asyncio
    async def test_get_state_failure_returns_none(self):
        """get_state returns None on request error."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock session that raises an error
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.get_state("switch.test")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_state_value_extracts_state(self):
        """get_state_value returns just the state string."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"state": "Export First"})
        mock_response.raise_for_status = MagicMock()

        # Create mock session
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.get_state_value("select.work_mode")
            assert result == "Export First"


class TestHAClientCallService:
    """Test HAClient.call_service."""

    @pytest.mark.asyncio
    async def test_call_service_success(self):
        """call_service returns True on success."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        # Create mock session
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.call_service("switch", "turn_on", "switch.test")
            assert result is True

    @pytest.mark.asyncio
    async def test_call_service_failure(self):
        """call_service raises HACallError on request exception (REV F52 Phase 5)."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock session that raises an error
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            with pytest.raises(HACallError) as exc_info:
                await client.call_service("switch", "turn_on", "switch.test")

            assert exc_info.value.exception_type == "ClientError"

    @pytest.mark.asyncio
    async def test_call_service_timeout_raises_ha_call_error(self):
        """call_service raises HACallError on TimeoutError."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock session that raises TimeoutError
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=TimeoutError("Request timed out"))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            with pytest.raises(HACallError) as exc_info:
                await client.call_service("switch", "turn_on", "switch.test")

            assert exc_info.value.exception_type == "TimeoutError"


class TestHAClientSetMethods:
    """Test HAClient setter methods."""

    @pytest.mark.asyncio
    async def test_set_select_option(self):
        """set_select_option calls select_option service."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        # Create mock session
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.set_select_option("select.mode", "Self Use")
            assert result is True

    @pytest.mark.asyncio
    async def test_set_switch(self):
        """set_switch calls turn_on/turn_off service."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        # Create mock session
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.set_switch("switch.test", True)
            assert result is True

    @pytest.mark.asyncio
    async def test_set_number(self):
        """set_number calls set_value service."""
        client = HAClient("http://ha:8123", "token123")

        # Create mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        # Create mock session
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.set_number("number.soc_target", 80.0)
            assert result is True


class TestHAClientValidation:
    """Test HAClient input validation."""

    @pytest.mark.asyncio
    async def test_get_state_with_none_entity(self):
        """get_state returns None for None entity_id."""
        client = HAClient("http://ha:8123", "token123")
        result = await client.get_state(None)  # type: ignore[arg-type]
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_with_empty_entity(self):
        """get_state returns None for empty entity_id."""
        client = HAClient("http://ha:8123", "token123")
        result = await client.get_state("")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_with_whitespace_entity(self):
        """get_state returns None for whitespace-only entity_id."""
        client = HAClient("http://ha:8123", "token123")
        result = await client.get_state("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_with_literal_none_string(self):
        """get_state returns None for literal 'None' string."""
        client = HAClient("http://ha:8123", "token123")
        result = await client.get_state("None")
        assert result is None


class TestHAClientSafetyGuards:
    """Test HAClient safety guards."""

    @pytest.mark.asyncio
    async def test_cannot_control_sensor_entity(self):
        """Safety guard prevents controlling sensor entities."""
        client = HAClient("http://ha:8123", "token123")

        with pytest.raises(HACallError) as exc_info:
            await client.set_number("sensor.temperature", 25.0)

        assert (
            "read-only" in str(exc_info.value).lower()
            or "invalid domain" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_cannot_control_binary_sensor(self):
        """Safety guard prevents controlling binary_sensor entities."""
        client = HAClient("http://ha:8123", "token123")

        with pytest.raises(HACallError) as exc_info:
            await client.set_switch("binary_sensor.motion", True)

        assert (
            "read-only" in str(exc_info.value).lower()
            or "invalid domain" in str(exc_info.value).lower()
        )


class TestHAClientCrossThreadSafety:
    """Test HAClient handles cross-thread event loop usage correctly.

    These tests verify the fix for: RuntimeError: Timeout context manager
    should be used inside a task, which occurred when the executor's
    background thread tried to use an HTTP client session created in
    the FastAPI main thread's event loop.
    """

    @pytest.mark.asyncio
    async def test_session_recreated_on_different_event_loop(self):
        """Session is recreated when used from a different event loop."""
        from unittest.mock import MagicMock, patch

        client = HAClient("http://ha:8123", "token123")

        # Create mock sessions for loop 1 and loop 2
        mock_session1 = MagicMock()
        mock_session1.closed = False
        mock_session2 = MagicMock()
        mock_session2.closed = False

        # Track which session was created
        sessions_created = []

        def mock_session_factory(*args, **kwargs):
            if len(sessions_created) == 0:
                sessions_created.append(mock_session1)
                return mock_session1
            else:
                sessions_created.append(mock_session2)
                return mock_session2

        # First call: Create session in loop 1
        loop1 = MagicMock()
        with (
            patch("executor.actions.aiohttp.ClientSession", side_effect=mock_session_factory),
            patch("asyncio.get_running_loop", return_value=loop1),
        ):
            session1 = await client._get_session()

        # Mark session1 as closed to trigger recreation
        mock_session1.closed = True

        # Second call: Use from loop 2 (simulates executor thread)
        loop2 = MagicMock()
        with (
            patch("executor.actions.aiohttp.ClientSession", side_effect=mock_session_factory),
            patch("asyncio.get_running_loop", return_value=loop2),
        ):
            session2 = await client._get_session()

        # Verify we got a different session
        assert session1 is mock_session1
        assert session2 is mock_session2
        assert session1 is not session2
        assert client._session_loop == loop2

    @pytest.mark.asyncio
    async def test_session_reused_on_same_event_loop(self):
        """Session is reused when called from the same event loop."""
        from unittest.mock import MagicMock, patch

        client = HAClient("http://ha:8123", "token123")

        # Create mock session
        mock_session = MagicMock()
        mock_session.closed = False

        # Use same loop for both calls
        loop = MagicMock()

        with (
            patch("executor.actions.aiohttp.ClientSession", return_value=mock_session),
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            session1 = await client._get_session()
            session2 = await client._get_session()

        # Verify we got the same session
        assert session1 is mock_session
        assert session2 is mock_session
        assert session1 is session2
        assert client._session_loop == loop


class TestSetWaterTemp:
    """Test ActionDispatcher.set_water_temp() method."""

    @pytest.fixture
    def base_config(self):
        """Create base config for water heater tests."""
        from executor.config import (
            ControllerConfig,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
            WaterHeaterDeviceConfig,
        )

        return ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(
                temp_normal=50,
                temp_off=40,
                # Anti-short-cycle dwell OFF by default here so these tests keep
                # exercising ratify/ownership/enforcement behaviour in isolation.
                # The dedicated dwell tests (TestWaterDwell) set explicit values.
                min_on_minutes=0.0,
                min_off_minutes=0.0,
            ),
            water_heater_devices=[
                WaterHeaterDeviceConfig(
                    id="main",
                    name="Main Heater",
                    target_entity="input_number.water_heater_target",
                    power_kw=3.0,
                )
            ],
            notifications=NotificationConfig(),
        )

    @pytest.mark.asyncio
    async def test_set_water_temp_skips_when_already_at_target(self, base_config):
        """Idempotency: skip when current temperature equals target."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        # Mock HA client returns current temp = 50 (same as target)
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="50")

        dispatcher = ActionDispatcher(
            ha_client=ha_client,
            config=base_config,
            shadow_mode=False,
        )

        result = await dispatcher.set_water_temp(50, "input_number.water_heater_target")

        # Assert skipped because already at target
        assert result.success is True
        assert result.skipped is True
        assert result.action_type == "water_temp"
        assert result.previous_value == 50
        assert result.new_value == 50
        assert "Already at 50°C" in result.message

        # Assert no HA write was attempted
        ha_client.set_input_number.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_target_heats_when_commanded_above_threshold(self, base_config):
        """A switch. target_entity is driven directly: 60°C command => turn_on."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="off")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(60, "switch.vvb")

        assert result.success is True
        assert result.skipped is False
        assert result.new_value == "on"
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", True)
        ha_client.set_input_number.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_target_turns_off_at_temp_off(self, base_config):
        """40°C (temp_off) command on a switch target => turn_off. This is the exact
        write the stranded HA bridge dropped for 3 days — now executor-owned.
        (Plan-ON first: an ON the executor ratified is its own to turn off — a
        never-ratified ON would be manual-respected instead.)"""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        await dispatcher.set_water_temp(60, "switch.vvb")  # plan ON: ratifies ownership
        result = await dispatcher.set_water_temp(40, "switch.vvb")

        assert result.success is True
        assert result.new_value == "off"
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_switch_target_skips_when_already_in_desired_state(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(70, "switch.vvb")

        assert result.success is True
        assert result.skipped is True
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_target_respects_shadow_mode(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        base_config.water_heater.manual_on_respect_minutes = 0.0  # isolate shadow path
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=True)
        result = await dispatcher.set_water_temp(40, "switch.vvb")

        assert result.success is True
        assert result.skipped is True
        assert "[SHADOW]" in result.message
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_input_boolean_target_supported(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="off")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(60, "input_boolean.vvb_heat")

        assert result.success is True
        ha_client.set_switch.assert_awaited_once_with("input_boolean.vvb_heat", True)

    @pytest.mark.asyncio
    async def test_manual_on_is_respected_as_implicit_boost(self, base_config):
        """A HUMAN turning the relay on while the plan wants OFF must not be reverted
        (the 2026-07-05 incident: user heated an empty tank, executor flipped it off
        within a tick). We never commanded ON => manual => honor for the window."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(40, "switch.vvb")  # plan: OFF

        assert result.success is True
        assert result.skipped is True
        assert "Manual ON respected" in result.message
        ha_client.set_switch.assert_not_called()

        # Subsequent ticks inside the window keep respecting it.
        result2 = await dispatcher.set_water_temp(40, "switch.vvb")
        assert result2.skipped is True and "Manual ON respected" in result2.message
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_on_window_expiry_resumes_enforcement(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        await dispatcher.set_water_temp(40, "switch.vvb")  # window opens
        # Backdate the window: it has expired.
        dispatcher._manual_on_until["switch.vvb"] = 1.0
        result = await dispatcher.set_water_temp(40, "switch.vvb")
        assert result.success is True and result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_own_on_command_is_enforced_not_respected(self, base_config):
        """States the executor set itself are never treated as manual: boost/plan turns
        it on, plan later says off => off, immediately."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="off")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        await dispatcher.set_water_temp(70, "switch.vvb")  # WE command ON (boost)
        ha_client.get_state_value = AsyncMock(return_value="on")
        result = await dispatcher.set_water_temp(40, "switch.vvb")  # plan: OFF
        assert result.skipped is False
        ha_client.set_switch.assert_awaited_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_plan_on_ratifies_ownership(self, base_config):
        """Plan agrees with an already-on relay => ownership ratified, so the eventual
        plan-off is enforced instead of opening a fresh manual window."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        await dispatcher.set_water_temp(60, "switch.vvb")  # plan ON, already on => ratify
        result = await dispatcher.set_water_temp(40, "switch.vvb")  # plan OFF
        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_manual_on_respect_disabled_is_legacy(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        base_config.water_heater.manual_on_respect_minutes = 0.0
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(40, "switch.vvb")
        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_switch_target_reports_service_failure(self, base_config):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        base_config.water_heater.manual_on_respect_minutes = 0.0  # isolate failure path
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="on")
        ha_client.set_switch = AsyncMock(side_effect=RuntimeError("HA API 502"))

        dispatcher = ActionDispatcher(ha_client=ha_client, config=base_config, shadow_mode=False)
        result = await dispatcher.set_water_temp(40, "switch.vvb")

        assert result.success is False
        assert result.error_details is not None
        assert "502" in result.error_details

    @pytest.mark.asyncio
    async def test_set_water_temp_respects_shadow_mode(self, base_config):
        """Shadow mode: return skipped result without HA call."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        # Mock HA client returns current temp = 40
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="40")

        dispatcher = ActionDispatcher(
            ha_client=ha_client,
            config=base_config,
            shadow_mode=True,  # Enable shadow mode
        )

        result = await dispatcher.set_water_temp(50, "input_number.water_heater_target")

        # Assert skipped due to shadow mode
        assert result.success is True
        assert result.skipped is True
        assert result.action_type == "water_temp"
        assert result.previous_value == 40
        assert result.new_value == 50
        assert "[SHADOW]" in result.message
        assert "40°C → 50°C" in result.message

        # Assert no HA write was attempted
        ha_client.set_input_number.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_water_temp_skips_when_entity_not_configured(self, base_config):
        """Skip when target entity is not configured."""
        from unittest.mock import MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()

        dispatcher = ActionDispatcher(
            ha_client=ha_client,
            config=base_config,
            shadow_mode=False,
        )

        # Pass None explicitly (or call without entity) to test not-configured path
        result = await dispatcher.set_water_temp(50, None)

        # Assert skipped due to entity not configured
        assert result.success is True
        assert result.skipped is True
        assert result.action_type == "water_temp"
        assert "not configured" in result.message.lower()

        # Assert no HA calls were made
        ha_client.get_state_value.assert_not_called()
        ha_client.set_input_number.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_water_temp_success(self, base_config):
        """Successfully set water temperature when conditions are met."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        # Mock HA client returns current temp = 40, set succeeds
        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value="40")
        ha_client.set_input_number = AsyncMock(return_value=True)

        dispatcher = ActionDispatcher(
            ha_client=ha_client,
            config=base_config,
            shadow_mode=False,
        )

        result = await dispatcher.set_water_temp(50, "input_number.water_heater_target")

        # Assert successful execution
        assert result.success is True
        assert result.skipped is False
        assert result.action_type == "water_temp"
        assert result.previous_value == 40
        assert result.new_value == 50
        assert "Changed 40°C → 50°C" in result.message

        # Assert HA write was attempted
        ha_client.set_input_number.assert_called_once_with("input_number.water_heater_target", 50.0)


class TestWaterDwell:
    """Build #15 PART A: anti-short-cycle min-on/min-off dwell on switch targets.

    Reproduces the overnight short-cycling (ON 02:00:17 -> OFF 02:03:10) and proves
    the dwell suppresses it, while boost/safety bypass and manual-ON respect keep
    their precedence and a genuine sustained change still wins after the dwell.
    """

    def _config(self, min_on: float = 30.0, min_off: float = 15.0, manual_respect: float = 90.0):
        from executor.config import (
            ControllerConfig,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
            WaterHeaterDeviceConfig,
        )

        return ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(
                temp_normal=60,
                temp_off=40,
                temp_boost=70,
                min_on_minutes=min_on,
                min_off_minutes=min_off,
                manual_on_respect_minutes=manual_respect,
            ),
            water_heater_devices=[
                WaterHeaterDeviceConfig(
                    id="main",
                    name="Main Heater",
                    target_entity="switch.vvb",
                    power_kw=3.0,
                )
            ],
            notifications=NotificationConfig(),
        )

    def _dispatcher(self, current_state: str, **cfg_over):
        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value=current_state)
        ha_client.set_switch = AsyncMock(return_value=True)
        dispatcher = ActionDispatcher(
            ha_client=ha_client, config=self._config(**cfg_over), shadow_mode=False
        )
        return dispatcher, ha_client

    @pytest.mark.asyncio
    async def test_dwell_holds_on_against_early_off(self):
        """Relay ON (executor-owned) turned OFF 3 min later => HELD (no set_switch).

        This is the exact 02:00:17 ON -> 02:03:10 OFF flip from the incident.
        """
        import time

        dispatcher, ha_client = self._dispatcher("on")
        # Executor owns the ON (so manual-ON respect does NOT intercept) and it just
        # turned on ~3 min ago.
        dispatcher._last_water_cmd["switch.vvb"] = "on"
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 3 * 60

        result = await dispatcher.set_water_temp(40, "switch.vvb")  # plan flips OFF

        assert result.success is True
        assert result.skipped is True
        assert "Dwell hold" in result.message
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_dwell_holds_off_against_early_on(self):
        """Relay OFF 5 min ago, min_off=15 => an ON flip is HELD."""
        import time

        dispatcher, ha_client = self._dispatcher("off")
        dispatcher._last_water_cmd["switch.vvb"] = "off"
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 5 * 60

        result = await dispatcher.set_water_temp(60, "switch.vvb")  # plan flips ON

        assert result.success is True
        assert result.skipped is True
        assert "Dwell hold" in result.message
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_dwell_allows_flip_after_min_on(self):
        """Once min_on has elapsed a sustained OFF change wins."""
        import time

        dispatcher, ha_client = self._dispatcher("on")
        dispatcher._last_water_cmd["switch.vvb"] = "on"
        # ON happened 31 min ago; min_on=30 => elapsed.
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 31 * 60

        result = await dispatcher.set_water_temp(40, "switch.vvb")

        assert result.success is True
        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_oscillating_plan_bounded_to_one_toggle(self):
        """An oscillating 60/40/60/40 plan within one dwell window => <=1 relay flip."""
        from executor.actions import ActionDispatcher

        state = {"v": "off"}

        async def _get(_entity):
            return state["v"]

        async def _set(_entity, on):
            state["v"] = "on" if on else "off"
            return True

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(side_effect=_get)
        ha_client.set_switch = AsyncMock(side_effect=_set)
        dispatcher = ActionDispatcher(
            ha_client=ha_client, config=self._config(), shadow_mode=False
        )

        # Cold start: first ON flips immediately; every later flip is dwell-held.
        for target in (60, 40, 60, 40, 60, 40):
            await dispatcher.set_water_temp(target, "switch.vvb")

        assert ha_client.set_switch.await_count == 1  # only the initial ON
        assert state["v"] == "on"

    @pytest.mark.asyncio
    async def test_boost_bypasses_dwell(self):
        """A boost (bypass_dwell=True) turns the relay ON even inside min_off."""
        import time

        dispatcher, ha_client = self._dispatcher("off")
        dispatcher._last_water_cmd["switch.vvb"] = "off"
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time()  # just turned off

        result = await dispatcher.set_water_temp(70, "switch.vvb", bypass_dwell=True)

        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", True)

    @pytest.mark.asyncio
    async def test_safety_off_bypasses_dwell(self):
        """A safety/override OFF (bypass_dwell=True) turns off even inside min_on."""
        import time

        dispatcher, ha_client = self._dispatcher("on")
        dispatcher._last_water_cmd["switch.vvb"] = "on"  # executor-owned ON
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time()  # just turned on

        result = await dispatcher.set_water_temp(40, "switch.vvb", bypass_dwell=True)

        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", False)

    @pytest.mark.asyncio
    async def test_manual_on_respect_wins_over_dwell(self):
        """A HUMAN's ON (not executor-owned) is respected first; dwell never runs."""
        import time

        dispatcher, ha_client = self._dispatcher("on")
        # We never commanded this ON => manual. Seed a dwell epoch to prove the dwell
        # branch is not what produces the skip.
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 999 * 60

        result = await dispatcher.set_water_temp(40, "switch.vvb")

        assert result.skipped is True
        assert "Manual ON respected" in result.message
        assert "Dwell hold" not in result.message
        ha_client.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_cold_start_first_flip_immediate(self):
        """Empty _last_water_switch_ts => the first flip actuates with no delay."""
        dispatcher, ha_client = self._dispatcher("off")
        assert dispatcher._last_water_switch_ts == {}

        result = await dispatcher.set_water_temp(60, "switch.vvb")

        assert result.skipped is False
        ha_client.set_switch.assert_awaited_once_with("switch.vvb", True)

    @pytest.mark.asyncio
    async def test_contiguous_block_not_starved_by_dwell(self):
        """min_kwh floor is never starved: a contiguous ON block (plan keeps
        commanding ON while the relay is already ON) produces only idempotent
        already-equal skips, never a dwell skip, and the eventual OFF after min_on
        is honored so exactly the planned energy is delivered then cleanly stops."""
        import time

        from executor.actions import ActionDispatcher

        state = {"v": "off"}

        async def _get(_entity):
            return state["v"]

        async def _set(_entity, on):
            state["v"] = "on" if on else "off"
            return True

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(side_effect=_get)
        ha_client.set_switch = AsyncMock(side_effect=_set)
        dispatcher = ActionDispatcher(
            ha_client=ha_client, config=self._config(), shadow_mode=False
        )

        # Tick 1: plan ON, relay off => flip ON (delivers the block).
        r1 = await dispatcher.set_water_temp(60, "switch.vvb")
        assert r1.skipped is False and state["v"] == "on"
        # Ticks 2-3: plan still ON, relay already on => idempotent skip, NOT dwell.
        for _ in range(2):
            r = await dispatcher.set_water_temp(60, "switch.vvb")
            assert r.skipped is True
            assert "Dwell hold" not in r.message
        # After min_on elapses the plan-OFF is honored (block ends cleanly).
        dispatcher._last_water_switch_ts["switch.vvb"] = time.time() - 31 * 60
        r_off = await dispatcher.set_water_temp(40, "switch.vvb")
        assert r_off.skipped is False and state["v"] == "off"


class TestClimateSink:
    """ActionDispatcher.set_custom_entity climate path (villavagn AC cooling sink)."""

    def _config(self, **ce_over):
        from executor.config import (
            ControllerConfig,
            ExcessPVConfig,
            ExcessPVCustomEntityConfig,
            ExcessPVSinkType,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
        )

        ce_kwargs = dict(
            entity="climate.villavagn",
            on_value="1",
            off_value="0",
            climate_mode="cool",
            target_temp=22.0,
            comfort_min_temp=20.0,
        )
        ce_kwargs.update(ce_over)
        return ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(),
            notifications=NotificationConfig(),
            excess_pv=ExcessPVConfig(
                sink=ExcessPVSinkType.CUSTOM_ENTITY,
                custom_entity=ExcessPVCustomEntityConfig(**ce_kwargs),
            ),
        )

    def _dispatcher(self, ha_client, config, shadow_mode=False):
        from executor.actions import ActionDispatcher

        return ActionDispatcher(ha_client=ha_client, config=config, shadow_mode=shadow_mode)

    @pytest.mark.asyncio
    async def test_on_sets_cool_mode_and_temperature(self):
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config()).set_custom_entity("1")

        assert result.success and not result.skipped
        ha.call_service.assert_any_call(
            "climate", "set_hvac_mode", "climate.villavagn", {"hvac_mode": "cool"}
        )
        ha.call_service.assert_any_call(
            "climate", "set_temperature", "climate.villavagn", {"temperature": 22.0}
        )

    @pytest.mark.asyncio
    async def test_comfort_floor_blocks_cooling(self):
        # Already at/below the comfort floor (20) → force off, never cool.
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 19.5}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config()).set_custom_entity("1")

        assert result.success
        # desired_mode resolved to "off"; already off → skipped, no service call.
        assert result.new_value == "off"
        ha.call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_off_sets_hvac_off(self):
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "cool", "attributes": {"current_temperature": 21.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config()).set_custom_entity("0")

        assert result.success and not result.skipped
        ha.call_service.assert_called_once_with(
            "climate", "set_hvac_mode", "climate.villavagn", {"hvac_mode": "off"}
        )

    @pytest.mark.asyncio
    async def test_idempotent_when_already_in_mode(self):
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "cool", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config()).set_custom_entity("1")

        assert result.success and result.skipped
        ha.call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_shadow_mode_no_write(self):
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config(), shadow_mode=True).set_custom_entity("1")

        assert result.success and result.skipped
        assert "[SHADOW]" in result.message
        ha.call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_with_boost_sink_when_enabled(self):
        # Primary sink is water_heater_boost; custom_entity.enabled=True must still actuate.
        from executor.config import (
            ControllerConfig,
            ExcessPVConfig,
            ExcessPVCustomEntityConfig,
            ExcessPVSinkType,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
        )

        config = ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(),
            notifications=NotificationConfig(),
            excess_pv=ExcessPVConfig(
                sink=ExcessPVSinkType.WATER_HEATER_BOOST,
                custom_entity=ExcessPVCustomEntityConfig(
                    entity="climate.villavagn",
                    enabled=True,
                    climate_mode="cool",
                    target_temp=22.0,
                    comfort_min_temp=20.0,
                ),
            ),
        )
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, config).set_custom_entity("1")

        assert result.success and not result.skipped
        ha.call_service.assert_any_call(
            "climate", "set_hvac_mode", "climate.villavagn", {"hvac_mode": "cool"}
        )


class TestSetSink:
    """ActionDispatcher.set_sink — one rung of the excess-PV sink ladder."""

    def _config(self, sinks):
        from executor.config import (
            ControllerConfig,
            ExcessPVConfig,
            ExecutorConfig,
            InverterConfig,
            NotificationConfig,
            WaterHeaterConfig,
        )

        return ExecutorConfig(
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            water_heater=WaterHeaterConfig(),
            notifications=NotificationConfig(),
            excess_pv=ExcessPVConfig(sinks=sinks),
        )

    def _dispatcher(self, ha_client, config, shadow_mode=False):
        from executor.actions import ActionDispatcher

        return ActionDispatcher(ha_client=ha_client, config=config, shadow_mode=shadow_mode)

    def _switch_sink(self, **over):
        from executor.config import ExcessPVSinkSpec

        kwargs = {"id": "poolpump", "entity": "switch.poolpump", "power_kw": 0.25, "enabled": True}
        kwargs.update(over)
        return ExcessPVSinkSpec(**kwargs)

    def _climate_sink(self, **over):
        from executor.config import ExcessPVSinkSpec

        kwargs = {
            "id": "villavagn_ac",
            "entity": "climate.villavagn",
            "power_kw": 1.0,
            "enabled": True,
            "climate_mode": "cool",
            "target_temp": 22.0,
            "comfort_min_temp": 20.0,
        }
        kwargs.update(over)
        return ExcessPVSinkSpec(**kwargs)

    @pytest.mark.asyncio
    async def test_switch_sink_on(self):
        sink = self._switch_sink()
        ha = MagicMock()
        ha.get_state_value = AsyncMock(side_effect=["off", "on"])
        ha.set_switch = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert result.success and not result.skipped
        assert result.action_type == "sink:poolpump"
        # Boolean write path: bool("0") is True, so the dispatcher must pass the
        # real on/off boolean, never a coerced string.
        ha.set_switch.assert_called_once_with("switch.poolpump", True)
        assert result.new_value == "on"

    @pytest.mark.asyncio
    async def test_switch_sink_off(self):
        sink = self._switch_sink()
        ha = MagicMock()
        ha.get_state_value = AsyncMock(side_effect=["on", "off"])
        ha.set_switch = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, False)

        assert result.success and not result.skipped
        ha.set_switch.assert_called_once_with("switch.poolpump", False)

    @pytest.mark.asyncio
    async def test_switch_sink_idempotent(self):
        sink = self._switch_sink()
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value="on")
        ha.set_switch = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert result.success and result.skipped
        ha.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_sink_shadow_mode(self):
        sink = self._switch_sink()
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value="off")
        ha.set_switch = AsyncMock(return_value=True)
        dispatcher = self._dispatcher(ha, self._config([sink]), shadow_mode=True)
        result = await dispatcher.set_sink(sink, True)

        assert result.success and result.skipped
        assert "[SHADOW]" in result.message
        ha.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_climate_sink_on_via_set_sink(self):
        sink = self._climate_sink()
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert result.success and not result.skipped
        assert result.action_type == "sink:villavagn_ac"
        ha.call_service.assert_any_call(
            "climate", "set_hvac_mode", "climate.villavagn", {"hvac_mode": "cool"}
        )
        ha.call_service.assert_any_call(
            "climate", "set_temperature", "climate.villavagn", {"temperature": 22.0}
        )

    @pytest.mark.asyncio
    async def test_climate_sink_comfort_floor_blocks_via_set_sink(self):
        # Already at/below the comfort floor (20) → force off, never cool.
        sink = self._climate_sink()
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 19.5}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert result.success
        assert result.new_value == "off"
        ha.call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_sink_without_entity_is_skipped(self):
        sink = self._switch_sink(entity=None)
        ha = MagicMock()
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert result.success and result.skipped

    @pytest.mark.asyncio
    async def test_number_sink_non_numeric_on_value_fails_without_raising(self):
        # A number-domain rung misconfigured with switch-style on_value ("on"):
        # float("on") must surface as a failed ActionResult, never an exception
        # that would abort sibling rungs and the inverter actuation.
        sink = self._switch_sink(
            id="heater", entity="number.heater_power", on_value="on", off_value="off"
        )
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value="0")
        ha.set_number = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_sink(sink, True)

        assert not result.success and not result.skipped
        ha.set_number.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_wrapper_skips_disabled_first_sink(self):
        # Observe-first ladder shape: sinks[0].enabled=False must NEVER be
        # actuated via the legacy wrapper (regression: `or bool(sinks)` made
        # the enabled check dead code).
        sink = self._switch_sink(enabled=False)
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value="off")
        ha.set_switch = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_custom_entity("1")

        assert result.success and result.skipped
        ha.set_switch.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_wrapper_uses_first_sink(self):
        # set_custom_entity must keep working, delegating to sinks[0].
        sink = self._climate_sink()
        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": "off", "attributes": {"current_temperature": 24.0}}
        )
        ha.call_service = AsyncMock(return_value=True)
        result = await self._dispatcher(ha, self._config([sink])).set_custom_entity("1")

        assert result.success and not result.skipped
        assert result.action_type == "sink:villavagn_ac"
        ha.call_service.assert_any_call(
            "climate", "set_hvac_mode", "climate.villavagn", {"hvac_mode": "cool"}
        )


class TestCyclicLoadDispatch:
    """set_cyclic_load on the REAL dispatcher.

    The original implementation said self.call_service — a method that lives on the
    HA CLIENT, not the dispatcher — and every test passed anyway, because the engine
    tests replaced set_cyclic_load with an AsyncMock and nothing ever ran the real
    method. Live, the first tick where a pump needed a write (2026-08-20 18:39)
    raised AttributeError, the engine's outer try logged it as 'Failed to execute
    async actions', and everything downstream in the block — the EV servo included —
    was dead for over an hour while a car charged at 14 A through the evening peak
    on the home battery. These tests exist so a self.<missing-attr> in this method
    can never again hide behind a mocked dispatcher.
    """

    @pytest.fixture
    def base_config(self):
        from executor.config import ExecutorConfig, NotificationConfig

        return ExecutorConfig(notifications=NotificationConfig())

    def _dispatcher(self, base_config, current="on"):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.get_state_value = AsyncMock(return_value=current)
        ha_client.call_service = AsyncMock(return_value=True)
        return ActionDispatcher(
            ha_client=ha_client, config=base_config, shadow_mode=False
        ), ha_client

    @pytest.mark.asyncio
    async def test_a_real_write_goes_through_the_ha_client(self, base_config):
        """The exact live failure: switch on, plan wants off."""
        dispatcher, ha = self._dispatcher(base_config, current="on")
        result = await dispatcher.set_cyclic_load("switch.poolpump", False, name="Poolpump")
        assert result.success is True
        assert result.skipped is False
        ha.call_service.assert_awaited_once_with("switch", "turn_off", "switch.poolpump")

    @pytest.mark.asyncio
    async def test_already_correct_state_writes_nothing(self, base_config):
        dispatcher, ha = self._dispatcher(base_config, current="off")
        result = await dispatcher.set_cyclic_load("switch.poolpump", False)
        assert result.skipped is True
        ha.call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_unverified_also_runs_for_real(self, base_config):
        """The other method added in the same change — same blind spot, same sweep."""
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher

        ha_client = MagicMock()
        ha_client.call_service = AsyncMock(return_value=True)
        base_config.notifications.on_write_unverified = True
        base_config.notifications.service = "notify.notify_robert_emilia"
        dispatcher = ActionDispatcher(
            ha_client=ha_client, config=base_config, shadow_mode=False
        )
        await dispatcher.notify_unverified("Spa", "3 misslyckade rättningar")


class TestForceHeaterClimateMode:
    """Direct climate correction for a thermostatted heater (the spa). Idempotency
    matters: this runs on every drifted tick, and each service call wakes the tub."""

    def _dispatcher(self, state, setpoint=None, shadow=False):
        from unittest.mock import AsyncMock, MagicMock

        from executor.actions import ActionDispatcher
        from executor.config import ExecutorConfig, InverterConfig

        ha = MagicMock()
        ha.get_state = AsyncMock(
            return_value={"state": state, "attributes": {"temperature": setpoint}}
        )
        ha.call_service = AsyncMock(return_value=True)
        d = ActionDispatcher(
            ha_client=ha,
            config=ExecutorConfig(inverter=InverterConfig()),
            shadow_mode=shadow,
        )
        return d, ha

    def _services(self, ha):
        return [(c.args[0], c.args[1]) for c in ha.call_service.call_args_list]

    @pytest.mark.asyncio
    async def test_fan_only_is_driven_to_heat(self):
        d, ha = self._dispatcher("fan_only", setpoint=20)
        res = await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert res.success and not res.skipped
        assert ("climate", "set_hvac_mode") in self._services(ha)
        assert ("climate", "set_temperature") in self._services(ha)

    @pytest.mark.asyncio
    async def test_already_correct_writes_nothing(self):
        d, ha = self._dispatcher("heat", setpoint=40)
        res = await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert res.skipped is True
        ha.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_right_mode_wrong_setpoint_only_writes_setpoint(self):
        d, ha = self._dispatcher("heat", setpoint=38)
        await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert self._services(ha) == [("climate", "set_temperature")]

    @pytest.mark.asyncio
    async def test_setpoint_tolerance(self):
        d, ha = self._dispatcher("heat", setpoint=40.4)
        res = await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert res.skipped is True
        ha.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shadow_mode_writes_nothing(self):
        d, ha = self._dispatcher("fan_only", setpoint=20, shadow=True)
        res = await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert res.skipped is True
        ha.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ha_failure_is_reported_not_raised(self):
        from executor.actions import HACallError

        d, ha = self._dispatcher("fan_only", setpoint=20)
        ha.call_service = AsyncMock(side_effect=HACallError("boom"))
        res = await d.force_heater_climate_mode("climate.spa", "heat", setpoint_c=40.0)
        assert res.success is False and "boom" in (res.error_details or "")
