"""
Action Dispatcher

Executes actions by calling Home Assistant services asynchronously using aiohttp.
Handles idempotent execution (skip if already set), notification dispatch per action type,
and automatic retry with exponential backoff for transient network failures.

Key Features:
- Async HTTP client (aiohttp) for non-blocking HA API calls
- 5-second timeout on all requests to prevent executor freezing
- Exponential backoff retry (3 attempts) for transient network errors
- Graceful degradation when HA is unreachable
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import ExcessPVCustomEntityConfig, ExecutorConfig
from .controller import ControllerDecision
from .profiles import InverterProfile, ModeAction

logger = logging.getLogger(__name__)


class HACallError(Exception):
    """Home Assistant API call error with detailed context."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        exception_type: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.exception_type = exception_type

        error_parts = [message]
        if status_code is not None:
            error_parts.append(f"HTTP {status_code}")
        if response_body:
            error_parts.append(f"Response: {response_body}")
        if exception_type:
            error_parts.append(f"({exception_type})")

        super().__init__(" | ".join(error_parts))


def _is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable (transient network error).

    Retryable errors include:
    - Connection errors (connection reset, refused, etc.)
    - Timeout errors
    - Server errors (5xx)
    - Temporary network issues

    Non-retryable errors include:
    - Client errors (4xx except 429)
    - Authentication errors
    - Invalid URL errors
    """
    import aiohttp

    # Server errors (5xx) are retryable
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status >= 500 or exception.status == 429  # 429 = Too Many Requests

    # Connection and timeout errors are retryable
    return isinstance(exception, aiohttp.ClientError | asyncio.TimeoutError)


async def _retry_with_backoff(
    operation: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Execute an async operation with exponential backoff retry.

    Args:
        operation: Async callable to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 10.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)

    Returns:
        Result of the operation

    Raises:
        HACallError: If all retries are exhausted
        Exception: If the error is not retryable
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            last_exception = e

            # Check if this is the last attempt
            if attempt >= max_retries:
                break

            # Check if error is retryable
            if not _is_retryable_error(e):
                # Not retryable, raise immediately
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor**attempt), max_delay)
            logger.warning(
                "HA API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1,
                max_retries + 1,
                e,
                delay,
            )
            await asyncio.sleep(delay)

    # All retries exhausted
    raise HACallError(
        message=f"HA API call failed after {max_retries + 1} attempts",
        exception_type=type(last_exception).__name__ if last_exception else "Unknown",
    ) from last_exception


def _is_entity_configured(entity: str | None) -> bool:
    """Check if an entity ID is properly configured.

    Returns False if entity is:
    - None
    - Empty string
    - Whitespace only
    - Literal string "None" (case-insensitive)
    """
    if not entity:
        return False
    stripped = entity.strip()
    return stripped != "" and stripped.lower() != "none"


# Standard inverter entity keys that live directly in executor.inverter.*
# Any profile entity key NOT in this set goes into executor.inverter.custom_entities.*
_STANDARD_INVERTER_KEYS: frozenset[str] = frozenset(
    [
        "work_mode",
        "soc_target",
        "grid_charging_enable",
        "grid_charge_power",
        "minimum_reserve",
        "grid_max_export_power",
        "grid_max_export_power_switch",
        "max_charge_current",
        "max_discharge_current",
        "max_charge_power",
        "max_discharge_power",
    ]
)


@dataclass
class ActionResult:
    """Result of executing an action."""

    action_type: str
    success: bool
    message: str = ""
    previous_value: Any | None = None
    new_value: Any | None = None
    entity_id: str | None = None  # NEW: The HA entity being controlled
    verified_value: Any | None = None  # NEW: Value read back after setting
    verification_success: bool | None = None  # NEW: Whether verification matched expected value
    skipped: bool = False  # True if action was skipped (already at target)
    duration_ms: int = 0
    error_details: str | None = None  # REV F52 Phase 5: HA API error details (status, body, etc.)
    # ARC16: Track the controller's intended mode vs applied mode
    requested_mode: str | None = None  # The mode_intent from controller (e.g., "idle")
    applied_mode: str | None = None  # The actual mode whose entities were applied


class HAClient:
    """
    Async Home Assistant API client for executing actions.

    Uses aiohttp for non-blocking HTTP communication with Home Assistant.
    All methods are async and should be awaited.

    Features:
    - Connection pooling via aiohttp.ClientSession
    - Configurable timeout (default: 5 seconds)
    - Automatic retry with exponential backoff for transient errors
    - Graceful error handling with HACallError exceptions

    Usage:
        client = HAClient("http://homeassistant:8123", "token")
        state = await client.get_state("sensor.battery_soc")
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        import asyncio

        current_loop = asyncio.get_running_loop()

        # Check if we need a new session (closed, None, or different event loop)
        need_new_session = (
            self._session is None
            or self._session.closed
            or getattr(self, "_session_loop", None) != current_loop
        )

        if need_new_session:
            # Close old session if it exists and belongs to a different loop
            if self._session and not self._session.closed:
                with contextlib.suppress(Exception):
                    await self._session.close()

            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.timeout,
            )
            self._session_loop = current_loop

        assert self._session is not None
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        """Get the current state of an entity with retry logic."""
        # Early validation: catch None/invalid entity_id before hitting HA API
        if not entity_id or entity_id.strip().lower() in ("", "none"):
            logger.error(
                "get_state called with invalid entity_id: %r (type: %s) - "
                "check config.yaml for missing entity configuration",
                entity_id,
                type(entity_id).__name__,
            )
            return None

        async def _fetch() -> dict[str, Any]:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/states/{entity_id}",
            ) as response:
                response.raise_for_status()
                return await response.json()

        try:
            return await _retry_with_backoff(_fetch, max_retries=3, base_delay=1.0)
        except HACallError:
            # All retries exhausted, return None for graceful degradation
            return None

    async def get_state_value(self, entity_id: str) -> str | None:
        """Get just the state value of an entity."""
        state = await self.get_state(entity_id)
        if state:
            return state.get("state")
        return None

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., 'switch', 'select', 'number')
            service: Service name (e.g., 'turn_on', 'select_option', 'set_value')
            entity_id: Target entity ID (optional)
            data: Additional service data (optional)

        Returns:
            True if successful

        Raises:
            HACallError: If the API call fails
        """
        payload = data or {}
        if entity_id:
            payload["entity_id"] = entity_id

        logger.debug(
            "HA call_service: %s.%s on %s with payload: %s", domain, service, entity_id, payload
        )

        async def _post() -> None:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/services/{domain}/{service}",
                json=payload,
            ) as response:
                response.raise_for_status()

        try:
            await _retry_with_backoff(_post, max_retries=3, base_delay=1.0)
            return True
        except aiohttp.ClientResponseError as e:
            raise HACallError(
                message=f"Failed to call service {domain}.{service} on {entity_id}",
                status_code=e.status,
                response_body=str(e.message),
                exception_type=type(e).__name__,
            ) from e
        except (aiohttp.ClientError, TimeoutError) as e:
            raise HACallError(
                message=f"Failed to call service {domain}.{service} on {entity_id}",
                exception_type=type(e).__name__,
            ) from e

    def _get_safe_domain(self, entity_id: str, allowed_domains: set[str]) -> str | None:
        """
        Get the domain from an entity ID and validate it is safe to control.

        Args:
            entity_id: The HA entity ID (e.g., 'input_select.mode')
            allowed_domains: Set of allowed domains (e.g., {'select', 'input_select'})

        Returns:
            The domain string if valid, None if invalid or unsafe.
        """
        if not entity_id:
            return None

        parts = entity_id.split(".", 1)
        if len(parts) != 2:
            logger.error("Invalid entity_id format: %s", entity_id)
            return None

        domain = parts[0]

        # Explicit safety guard against sensors
        if domain in ("sensor", "binary_sensor"):
            logger.error(
                "SAFETY GUARD: Cannot control read-only entity '%s'. "
                "Check config.yaml and use a controllable entity (e.g., input_number, helper).",
                entity_id,
            )
            return None

        if domain not in allowed_domains:
            logger.error(
                "Domain '%s' not allowed for this action. Allowed: %s. Entity: %s",
                domain,
                allowed_domains,
                entity_id,
            )
            return None

        return domain

    async def set_select_option(self, entity_id: str, option: str) -> bool:
        """Set a select entity to a specific option."""
        domain = self._get_safe_domain(entity_id, {"select", "input_select"})
        if not domain:
            raise HACallError(
                message=f"Invalid domain for select entity {entity_id}",
                exception_type="DomainValidationError",
            )
        return await self.call_service(domain, "select_option", entity_id, {"option": option})

    async def set_switch(self, entity_id: str, state: bool) -> bool:
        """Turn a switch on or off."""
        domain = self._get_safe_domain(entity_id, {"switch", "input_boolean"})
        if not domain:
            raise HACallError(
                message=f"Invalid domain for switch entity {entity_id}",
                exception_type="DomainValidationError",
            )
        service = "turn_on" if state else "turn_off"
        return await self.call_service(domain, service, entity_id)

    async def set_number(self, entity_id: str, value: float) -> bool:
        """Set a number entity to a specific value."""
        domain = self._get_safe_domain(entity_id, {"number", "input_number"})
        if not domain:
            raise HACallError(
                message=f"Invalid domain for number entity {entity_id}",
                exception_type="DomainValidationError",
            )
        return await self.call_service(domain, "set_value", entity_id, {"value": value})

    async def set_input_number(self, entity_id: str, value: float) -> bool:
        """Set an input_number entity to a specific value."""
        # Alias to set_number which now handles both
        return await self.set_number(entity_id, value)

    async def send_notification(
        self,
        service: str | None,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send a notification via a notify service.

        Args:
            service: Full notification service name (e.g., 'notify.mobile_app_phone')
            title: Notification title
            message: Notification message
            data: Additional notification data (optional)

        Returns:
            True if successful, False otherwise
        """
        if not service:
            return False

        # Parse service name (e.g., "notify.mobile_app_phone" -> domain="notify", service="mobile_app_phone")
        parts = service.split(".", 1)
        if len(parts) != 2:
            logger.error("Invalid notification service format: %s", service)
            return False

        domain, svc_name = parts
        payload: dict[str, Any] = {
            "title": title,
            "message": message,
        }
        if data:
            payload["data"] = data

        return await self.call_service(domain, svc_name, data=payload)


class ActionDispatcher:
    """
    Dispatches actions to Home Assistant based on controller decisions.

    Uses profile-driven architecture where each mode defines an ordered list
    of entity+value actions. The executor is a generic loop.

    Features:
    - Idempotent execution (skip if already at target)
    - Configurable notifications per action type
    - Action result tracking
    """

    def __init__(
        self,
        ha_client: HAClient,
        config: ExecutorConfig,
        shadow_mode: bool = False,
        profile: InverterProfile | None = None,
    ):
        self.ha = ha_client
        self.config = config
        self.shadow_mode = shadow_mode
        self.profile = profile
        # C3: feed-in limit to restore to after a curtailment window. Captured lazily the moment
        # before the first clamp when export_curtailment.restore_limit_w is left at 0 (auto).
        self._restore_export_limit_w: float | None = None

    def _resolve_entity_id(self, key: str) -> str | None:
        """
        Resolve entity key to actual HA entity ID.

        Resolution order:
        1. User override: executor.inverter.custom_entities[key]
        2. Standard config: executor.inverter[key]
        3. Profile default: entities[key].default_entity
        """
        if not self.profile:
            return None

        entity_def = self.profile.entities.get(key)
        if not entity_def:
            return None

        override = self.config.inverter.custom_entities.get(key)
        if override:
            return override

        standard = getattr(self.config.inverter, key, None)
        if standard:
            return standard

        return entity_def.default_entity

    def _resolve_value(self, value: str | int | float | bool, decision: ControllerDecision) -> Any:
        """
        Resolve dynamic template values from ControllerDecision.

        Templates are strings in the form {{field_name}} where field_name
        is a property on ControllerDecision.
        """
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            field_name = value[2:-2]
            if not hasattr(decision, field_name):
                logger.error("Unknown template variable: %s", field_name)
                return value
            return getattr(decision, field_name)
        return value

    async def _write_entity(
        self,
        entity_id: str,
        value: Any,
        domain: str,
    ) -> bool:
        """
        Write value to HA entity using appropriate service call.

        Args:
            entity_id: The HA entity ID to write to
            value: The value to write
            domain: The HA domain (select, number, switch, input_number)

        Returns:
            True if successful
        """
        try:
            if domain in ("number", "input_number"):
                return await self.ha.set_number(entity_id, float(value))
            elif domain == "select":
                return await self.ha.set_select_option(entity_id, str(value))
            elif domain in ("switch", "input_boolean"):
                return await self.ha.set_switch(entity_id, bool(value))
            else:
                logger.error("Unknown entity domain: %s", domain)
                return False
        except HACallError as e:
            logger.error("Failed to write to %s: %s", entity_id, e)
            return False

    def _values_match(self, current: str | None, target: Any) -> bool:
        """Check if current value matches target value."""
        if current is None:
            return False
        try:
            current_float = float(current)
            target_float = float(target)
            return abs(current_float - target_float) < 0.01
        except (ValueError, TypeError):
            if isinstance(target, bool):
                current_lower = str(current).strip().lower()
                if target and current_lower == "on":
                    return True
                if not target and current_lower == "off":
                    return True
            return str(current).strip().lower() == str(target).strip().lower()

    async def _verify_action(self, entity_id: str, expected: Any) -> tuple[Any, bool | None]:
        """Verify that an action was applied correctly."""
        state = await self.ha.get_state_value(entity_id)
        if state is None:
            return None, None

        matches = self._values_match(state, expected)
        return state, matches

    async def execute(self, decision: ControllerDecision) -> list[ActionResult]:
        """
        Execute all actions from a controller decision using profile-driven approach.

        Args:
            decision: The controller's decision with mode_intent

        Returns:
            List of ActionResult for each action attempted
        """
        if not self.profile:
            logger.error("No profile loaded - cannot execute actions")
            return [
                ActionResult(
                    action_type="error",
                    success=False,
                    message="No inverter profile loaded",
                )
            ]

        mode_intent = decision.mode_intent

        try:
            mode_def = self.profile.get_mode(mode_intent)
        except Exception as e:
            logger.error("Failed to get mode '%s' from profile: %s", mode_intent, e)
            return [
                ActionResult(
                    action_type="error",
                    success=False,
                    message=f"Profile error: {e}",
                )
            ]

        logger.info(
            "Executing mode '%s' (%s) for profile '%s'",
            mode_intent,
            mode_def.description,
            self.profile.metadata.name,
        )

        results: list[ActionResult] = []

        for action in mode_def.actions:
            result = await self._execute_action(action, decision, mode_intent)
            results.append(result)

            if action.settle_ms and action.settle_ms > 0:
                logger.debug("Settle delay: %dms after %s", action.settle_ms, action.entity)
                await asyncio.sleep(action.settle_ms / 1000.0)

        # C3: price-conditioned export curtailment. Run AFTER the mode's own actions so it sets
        # the resting export limit, except in explicit grid-export mode (which manages the limit
        # itself and is only chosen by the planner at a profitable price).
        if self.config.export_curtailment.enabled and mode_intent != "export":
            clamp = await self._apply_export_curtailment(decision.export_price_sek_kwh)
            if clamp is not None:
                results.append(clamp)

        if results:
            successful = sum(1 for r in results if r.success)
            logger.info(
                "Mode '%s' executed: %d/%d actions successful",
                mode_intent,
                successful,
                len(results),
            )

        return results

    async def _execute_action(
        self,
        action: ModeAction,
        decision: ControllerDecision,
        mode_intent: str,
    ) -> ActionResult:
        """Execute a single mode action."""
        start_time = time.time()

        if not self.profile:
            return ActionResult(
                action_type=action.entity,
                success=False,
                message="No profile loaded",
                requested_mode=mode_intent,
                applied_mode=mode_intent,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        entity_def = self.profile.entities.get(action.entity)
        if not entity_def:
            return ActionResult(
                action_type=action.entity,
                success=False,
                message=f"Entity '{action.entity}' not defined in profile",
                requested_mode=mode_intent,
                applied_mode=mode_intent,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        entity_id = self._resolve_entity_id(action.entity)

        if not entity_id:
            return ActionResult(
                action_type=action.entity,
                success=False,
                message=f"Entity '{action.entity}' not configured",
                requested_mode=mode_intent,
                applied_mode=mode_intent,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        resolved_value = self._resolve_value(action.value, decision)

        previous_value = await self.ha.get_state_value(entity_id)

        if self._values_match(previous_value, resolved_value):
            return ActionResult(
                action_type=action.entity,
                success=True,
                message=f"Already at {resolved_value}",
                previous_value=previous_value,
                new_value=resolved_value,
                entity_id=entity_id,
                skipped=True,
                requested_mode=mode_intent,
                applied_mode=mode_intent,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        if self.shadow_mode:
            logger.info(
                "[SHADOW] Would set %s to %s (current: %s)",
                entity_id,
                resolved_value,
                previous_value,
            )
            return ActionResult(
                action_type=action.entity,
                success=True,
                message=f"[SHADOW] Would change {previous_value} → {resolved_value}",
                previous_value=previous_value,
                new_value=resolved_value,
                entity_id=entity_id,
                skipped=True,
                requested_mode=mode_intent,
                applied_mode=mode_intent,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        success = await self._write_entity(entity_id, resolved_value, entity_def.domain)

        verified_value = None
        verification_success = None
        if success:
            verified_value, verification_success = await self._verify_action(
                entity_id, resolved_value
            )

        duration_ms = int((time.time() - start_time) * 1000)

        if success:
            await self._maybe_notify(action.entity, f"Set {action.entity} to {resolved_value}")

        return ActionResult(
            action_type=action.entity,
            success=success,
            message=f"{previous_value} → {resolved_value}"
            if success
            else f"Failed to set {action.entity}",
            previous_value=previous_value,
            new_value=resolved_value,
            entity_id=entity_id,
            verified_value=verified_value,
            verification_success=verification_success,
            requested_mode=mode_intent,
            applied_mode=mode_intent,
            duration_ms=duration_ms,
        )

    async def set_water_temp(self, target: int, target_entity: str | None = None) -> ActionResult:
        """Set water heater target temperature.

        Args:
            target: Target temperature in °C
            target_entity: HA entity to control. If None, falls back to
                           config.water_heater.target_entity (legacy single-heater path).
        """
        start = time.time()
        # Use passed entity; fall back to legacy single-entity config for backward compat
        entity = (
            target_entity
            if target_entity is not None
            else getattr(self.config.water_heater, "target_entity", None)
        )

        if not _is_entity_configured(entity):
            logger.debug("Skipping water_temp action: entity not configured")
            return ActionResult(
                action_type="water_temp",
                success=True,
                message="Water heater target entity not configured. Configure in Settings → System → HA Entities",
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        if entity is None:
            return ActionResult(
                action_type="water_temp",
                success=False,
                message="Entity is None after validation",
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        current = await self.ha.get_state_value(entity)
        try:
            current_val = int(float(current)) if current else None
        except (ValueError, TypeError):
            current_val = None

        if current_val == target:
            return ActionResult(
                action_type="water_temp",
                success=True,
                message=f"Already at {target}°C",
                previous_value=current_val,
                new_value=target,
                entity_id=entity,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        if self.shadow_mode:
            logger.info(
                "[SHADOW] Would set water_temp to %s°C (current: %s°C)", target, current_val
            )
            return ActionResult(
                action_type="water_temp",
                success=True,
                message=f"[SHADOW] Would change {current_val}°C → {target}°C",
                previous_value=current_val,
                new_value=target,
                entity_id=entity,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        error_details = None
        try:
            success = await self.ha.set_input_number(entity, float(target))  # type: ignore[arg-type]
        except HACallError as e:
            success = False
            error_details = str(e)
            logger.error("Failed to set water_temp: %s", error_details)

        # Verification
        verified_value = None
        verification_success = None
        if success:
            v_val, v_ok = await self._verify_action(entity, target)  # type: ignore[arg-type]
            verification_success = v_ok
            try:
                verified_value = int(float(v_val)) if v_val else None
            except (ValueError, TypeError):
                verified_value = v_val

        duration = int((time.time() - start) * 1000)

        # Determine if this is start or stop
        is_heating = target > self.config.water_heater.temp_off
        action = "start" if is_heating else "stop"
        if success:
            await self._maybe_notify(f"water_heat_{action}", f"Water heater target: {target}°C")

        return ActionResult(
            action_type="water_temp",
            success=success,
            message=(
                f"Changed {current_val}°C → {target}°C"
                if success
                else f"Failed: {error_details}"
                if error_details
                else "Failed to set water temp"
            ),
            previous_value=current_val,
            new_value=target,
            entity_id=entity,
            verified_value=verified_value,
            verification_success=verification_success,
            duration_ms=duration,
            error_details=error_details,
        )

    async def set_custom_entity(self, value: str) -> ActionResult:
        """Toggle the excess PV custom entity sink on/off.

        Args:
            value: The value to set (on_value or off_value from config).

        Returns:
            ActionResult indicating success or failure.
        """
        from .config import ExcessPVSinkType

        start = time.time()
        excess_pv = self.config.excess_pv

        if excess_pv.sink != ExcessPVSinkType.CUSTOM_ENTITY or not excess_pv.custom_entity.entity:
            return ActionResult(
                action_type="custom_entity",
                success=True,
                message="Custom entity sink not configured",
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
            )

        entity = excess_pv.custom_entity.entity

        # Climate entities (e.g. the villavagn AC as a surplus cooling sink) need
        # hvac_mode/temperature services, not a plain state write. Delegate.
        if entity.split(".", 1)[0] == "climate":
            turn_on = self._values_match(value, excess_pv.custom_entity.on_value)
            return await self._set_climate_sink(entity, turn_on, excess_pv.custom_entity, start)

        current = await self.ha.get_state_value(entity)

        if self._values_match(current, value):
            return ActionResult(
                action_type="custom_entity",
                success=True,
                message=f"Already at {value}",
                previous_value=current,
                new_value=value,
                entity_id=entity,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
            )

        if self.shadow_mode:
            logger.info(
                "[SHADOW] Would set custom entity %s to %s (current: %s)",
                entity,
                value,
                current,
            )
            return ActionResult(
                action_type="custom_entity",
                success=True,
                message=f"[SHADOW] Would change {current} → {value}",
                previous_value=current,
                new_value=value,
                entity_id=entity,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
            )

        error_details = None
        try:
            domain = entity.split(".", 1)[0] if "." in entity else "switch"
            success = await self._write_entity(entity, value, domain)
        except HACallError as e:
            success = False
            error_details = str(e)
            logger.error("Failed to set custom entity %s: %s", entity, error_details)

        verified_value = None
        verification_success = None
        if success:
            verified_value, verification_success = await self._verify_action(entity, value)

        duration = int((time.time() - start) * 1000)

        return ActionResult(
            action_type="custom_entity",
            success=success,
            message=(
                f"Changed {current} → {value}"
                if success
                else f"Failed: {error_details}"
                if error_details
                else "Failed to set custom entity"
            ),
            previous_value=current,
            new_value=value,
            entity_id=entity,
            verified_value=verified_value,
            verification_success=verification_success,
            duration_ms=duration,
            error_details=error_details,
        )

    async def _set_climate_sink(
        self,
        entity: str,
        turn_on: bool,
        cfg: ExcessPVCustomEntityConfig,
        start: float,
    ) -> ActionResult:
        """Actuate a climate.* excess-PV sink (e.g. the villavagn AC).

        ON  => climate.set_hvac_mode(cfg.climate_mode) + optional set_temperature, BUT
               skipped (and forced OFF) when current_temperature <= comfort_min_temp so
               surplus never over-cools the space.
        OFF => climate.set_hvac_mode("off").

        Idempotent: a no-op when the unit is already in the desired mode.
        """
        state = await self.ha.get_state(entity)
        current_mode = state.get("state") if state else None
        attrs: dict[str, Any] = (state or {}).get("attributes", {}) or {}
        current_temp = attrs.get("current_temperature")

        # Comfort floor: if already cool enough, never run the AC for surplus.
        floor_blocked = (
            turn_on
            and cfg.comfort_min_temp is not None
            and isinstance(current_temp, int | float)
            and float(current_temp) <= cfg.comfort_min_temp
        )
        desired_mode = cfg.climate_mode if (turn_on and not floor_blocked) else "off"

        def _result(success: bool, message: str, *, skipped: bool = False) -> ActionResult:
            return ActionResult(
                action_type="custom_entity",
                success=success,
                message=message,
                previous_value=current_mode,
                new_value=desired_mode,
                entity_id=entity,
                skipped=skipped,
                duration_ms=int((time.time() - start) * 1000),
            )

        if floor_blocked:
            logger.info(
                "Climate sink %s: current %.1f°C <= comfort floor %.1f°C, not cooling",
                entity,
                float(current_temp),  # type: ignore[arg-type]
                cfg.comfort_min_temp,
            )

        if current_mode == desired_mode:
            return _result(True, f"Already {desired_mode}", skipped=True)

        if self.shadow_mode:
            logger.info(
                "[SHADOW] Would set climate sink %s %s -> %s", entity, current_mode, desired_mode
            )
            return _result(True, f"[SHADOW] Would change {current_mode} -> {desired_mode}", skipped=True)

        try:
            ok = await self.ha.call_service(
                "climate", "set_hvac_mode", entity, {"hvac_mode": desired_mode}
            )
            if ok and desired_mode != "off" and cfg.target_temp is not None:
                await self.ha.call_service(
                    "climate", "set_temperature", entity, {"temperature": cfg.target_temp}
                )
        except HACallError as e:
            logger.error("Failed to set climate sink %s: %s", entity, e)
            return ActionResult(
                action_type="custom_entity",
                success=False,
                message=f"Failed: {e}",
                previous_value=current_mode,
                new_value=desired_mode,
                entity_id=entity,
                error_details=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

        return _result(True, f"Changed {current_mode} -> {desired_mode}")

    async def _read_current_export_limit(self) -> float | None:
        """Read the inverter's current export-power limit (W), or None if unavailable."""
        entity = self.config.inverter.grid_max_export_power or self._resolve_entity_id(
            "export_power_limit"
        )
        if not _is_entity_configured(entity) or entity is None:
            return None
        try:
            raw = await self.ha.get_state_value(entity)
            return float(raw) if raw is not None else None
        except (ValueError, TypeError, HACallError):
            return None

    async def _apply_export_curtailment(self, export_price_sek_kwh: float) -> ActionResult | None:
        """Clamp grid export to 0 W when the effective export price is below the threshold (you
        would pay to export), and restore the feed-in limit otherwise. C3."""
        cc = self.config.export_curtailment
        if export_price_sek_kwh < cc.threshold_sek_per_kwh:
            # Capture the resting feed-in limit before overriding it, so restore is exact.
            if cc.restore_limit_w <= 0 and self._restore_export_limit_w is None:
                current = await self._read_current_export_limit()
                if current is not None and current > 0:
                    self._restore_export_limit_w = current
                    logger.info("Export curtailment: captured restore limit %.0f W", current)
            logger.info(
                "Export curtailment ACTIVE: effective export %.3f < %.3f SEK/kWh -> clamp 0 W",
                export_price_sek_kwh,
                cc.threshold_sek_per_kwh,
            )
            return await self._set_max_export_power(0.0)

        # Not curtailing: restore the feed-in limit if we know it (config value or captured).
        restore = (
            cc.restore_limit_w if cc.restore_limit_w > 0 else (self._restore_export_limit_w or 0.0)
        )
        if restore > 0:
            return await self._set_max_export_power(restore)
        return None

    async def _set_max_export_power(self, watts: float) -> ActionResult | None:
        """Set max grid export power."""
        start = time.time()

        # Resolve the export-limit entity: explicit executor config wins, otherwise fall back to
        # the inverter profile's export_power_limit entity (e.g. Sungrow number.export_power_limit)
        # so price-conditioned curtailment works without extra entity configuration.
        entity = self.config.inverter.grid_max_export_power or self._resolve_entity_id(
            "export_power_limit"
        )

        if not _is_entity_configured(entity):
            logger.debug("Skipping max_export_power action: no export-limit entity available")
            return None  # Silent skip - no entry in execution history

        # Check current value and apply write threshold to prevent EEPROM wear
        if entity is None:
            return ActionResult(
                action_type="max_export_power",
                success=False,
                message="Entity is None after validation",
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        current = await self.ha.get_state_value(entity)
        try:
            current_val = float(current) if current else None
        except (ValueError, TypeError):
            current_val = None

        if current_val is not None:
            change = abs(watts - current_val)
            if change < self.config.controller.write_threshold_w:
                return ActionResult(
                    action_type="max_export_power",
                    success=True,
                    message=f"Change {change:.0f}W < threshold {self.config.controller.write_threshold_w:.0f}W, skipping",
                    previous_value=current_val,
                    new_value=watts,
                    entity_id=entity,
                    skipped=True,
                    duration_ms=int((time.time() - start) * 1000),
                    error_details=None,
                )

        if self.shadow_mode:
            logger.info("[SHADOW] Would set max_export_power to %s W", watts)
            return ActionResult(
                action_type="max_export_power",
                success=True,
                message=f"[SHADOW] Would set to {watts} W",
                new_value=watts,
                entity_id=entity,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        error_details = None
        try:
            success = await self.ha.set_number(entity, watts)
        except HACallError as e:
            success = False
            error_details = str(e)
            logger.error("Failed to set max_export_power: %s", error_details)

        # Verification
        verified_value = None
        verification_success = None
        if success:
            verified_value, verification_success = await self._verify_action(entity, watts)

        # 5. Handle Export Switch (F49)
        # If a switch is configured, turn it ON when setting a limit.
        # This ensures that inverter actually enforces the numeric value.
        switch_entity = self.config.inverter.grid_max_export_power_switch or self._resolve_entity_id(
            "export_power_limit_switch"
        )
        if success and _is_entity_configured(switch_entity) and switch_entity is not None:
            logger.info("Enabling export power limit switch: %s", switch_entity)
            try:
                await self.ha.set_switch(switch_entity, True)
            except HACallError as e:
                logger.warning("Failed to enable export power limit switch: %s", str(e))

        duration = int((time.time() - start) * 1000)

        logger.info("Set max_export_power: %.0f W on %s (success=%s)", watts, entity, success)

        return ActionResult(
            action_type="max_export_power",
            success=success,
            message=f"Set to {watts} W"
            if success
            else f"Failed: {error_details}"
            if error_details
            else "Failed to set export power",
            previous_value=current_val,
            new_value=watts,
            entity_id=entity,
            verified_value=verified_value,
            verification_success=verification_success,
            duration_ms=duration,
            error_details=error_details,
        )

    async def set_ev_charger_switch(
        self, entity_id: str, turn_on: bool, charging_kw: float = 0.0
    ) -> ActionResult:
        """
        Control EV charger switch with shadow mode support.

        Args:
            entity_id: The HA switch entity ID for the EV charger
            turn_on: True to turn on, False to turn off
            charging_kw: Planned charging power in kW (for logging/notifications)

        Returns:
            ActionResult with details of the action
        """
        start = time.time()
        action_type = "ev_charge_start" if turn_on else "ev_charge_stop"
        action_label = "ON" if turn_on else "OFF"

        # Check current state
        current_state = await self.ha.get_state_value(entity_id)
        is_currently_on = current_state == "on" if current_state else False

        # Idempotent skip
        if turn_on == is_currently_on:
            return ActionResult(
                action_type=action_type,
                success=True,
                message=f"EV charger already {action_label}",
                previous_value=current_state,
                new_value=turn_on,
                entity_id=entity_id,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        # Shadow mode check
        if self.shadow_mode:
            logger.info(
                "[SHADOW] EV Charger: Would turn %s %s (current: %s)",
                action_label,
                entity_id,
                current_state,
            )
            return ActionResult(
                action_type=action_type,
                success=True,
                message=f"[SHADOW] Would turn {action_label}",
                previous_value=current_state,
                new_value=turn_on,
                entity_id=entity_id,
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
                error_details=None,
            )

        # Execute action
        error_details = None
        try:
            await self.ha.set_switch(entity_id, turn_on)
            success = True
        except HACallError as e:
            success = False
            error_details = str(e)
            logger.error("Failed to control EV charger %s: %s", entity_id, error_details)

        # Verification
        verified_value = None
        verification_success = None
        if success:
            verified_value, verification_success = await self._verify_action(
                entity_id, "on" if turn_on else "off"
            )

        duration = int((time.time() - start) * 1000)

        # Notification (via _maybe_notify)
        if turn_on:
            await self._maybe_notify(
                "ev_charge_start", f"EV charging started ({charging_kw:.1f} kW)"
            )
        else:
            await self._maybe_notify("ev_charge_stop", "EV charging stopped")

        return ActionResult(
            action_type=action_type,
            success=success,
            message=f"EV charger turned {action_label}"
            if success
            else f"Failed: {error_details}"
            if error_details
            else f"Failed to turn {action_label} EV charger",
            previous_value=current_state,
            new_value=turn_on,
            entity_id=entity_id,
            verified_value=verified_value,
            verification_success=verification_success,
            duration_ms=duration,
            error_details=error_details,
        )

    async def _maybe_notify(self, action_type: str, message: str) -> None:
        """Send notification if enabled for this action type."""
        notif = self.config.notifications

        # Map action types to notification flags
        should_notify = {
            "charge_start": notif.on_charge_start,
            "charge_stop": notif.on_charge_stop,
            "export_start": notif.on_export_start,
            "export_stop": notif.on_export_stop,
            "water_heat_start": notif.on_water_heat_start,
            "water_heat_stop": notif.on_water_heat_stop,
            "work_mode": notif.on_export_start or notif.on_export_stop,
            "override": notif.on_override_activated,
            "error": notif.on_error,
        }.get(action_type, False)

        if should_notify:
            await self._send_notification(message)

    async def _send_notification(self, message: str, title: str = "Darkstar Executor") -> None:
        """Send a notification via the configured service."""
        if self.shadow_mode:
            logger.info("[SHADOW] Would send notification: %s", message)
            return

        try:
            await self.ha.send_notification(
                self.config.notifications.service,
                title,
                message,
            )
        except Exception as e:
            logger.warning("Failed to send notification: %s", e)

    async def notify_override(self, override_type: str, reason: str) -> None:
        """Send notification about an override activation."""
        if self.config.notifications.on_override_activated:
            await self._send_notification(
                f"Override: {override_type}\n{reason}",
                title="Darkstar Override Active",
            )

    async def notify_error(self, error: str) -> None:
        """Send notification about an error."""
        if self.config.notifications.on_error:
            await self._send_notification(
                f"Error: {error}",
                title="Darkstar Executor Error",
            )
