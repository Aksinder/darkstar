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

from .config import ExcessPVSinkSpec, ExecutorConfig
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

# Water heaters whose target_entity is a relay (switch./input_boolean.) are driven
# directly: commanded temperature above this = heat ON, at/below = OFF. Matches the
# temp semantics (temp_off 40 / temp_normal 60 / boost 70 / max 85) and the >50°C
# convention of the HA bridge automations this replaces.
WATER_SWITCH_ON_THRESHOLD_C = 50.0


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

    async def set_state(
        self, entity_id: str, state: str, attributes: dict[str, Any] | None = None
    ) -> bool:
        """Publish an entity state to HA via POST /api/states/<entity_id>.

        Used to surface Darkstar-computed sensors (e.g. the FMB SoC estimate) that have
        no backing integration. Returns False on failure (graceful — a missed publish is
        recoverable next tick) rather than raising.
        """
        if not entity_id or entity_id.strip().lower() in ("", "none"):
            logger.error("set_state called with invalid entity_id: %r", entity_id)
            return False
        body: dict[str, Any] = {"state": state, "attributes": attributes or {}}

        async def _post() -> None:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/states/{entity_id}",
                json=body,
            ) as response:
                response.raise_for_status()

        try:
            await _retry_with_backoff(_post, max_retries=2, base_delay=1.0)
            return True
        except (aiohttp.ClientError, TimeoutError, HACallError) as e:
            logger.warning("Failed to publish state for %s: %s", entity_id, e)
            return False

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
        # Manual-ON respect for switch-type water targets: what WE last commanded per
        # entity (so a human's ON is distinguishable from our own), and until when a
        # detected manual ON is honored as an implicit boost. In-memory: an add-on
        # restart forgets an active window (bounded downside — the plan resumes).
        self._last_water_cmd: dict[str, str] = {}
        self._manual_on_until: dict[str, float] = {}
        # Anti-short-cycle dwell: wall-clock time.time() of the last relay state
        # CHANGE we actually commanded, keyed by target_entity (per-switch, so
        # switch.vvb and switch.villavagn_vvb are independent). Empty at cold start
        # => the first flip per entity is always allowed immediately; only
        # subsequent flips within the dwell window are gated.
        self._last_water_switch_ts: dict[str, float] = {}
        # Build #16 block-commit latch: wall-clock time.time() UNTIL which a
        # just-started heating block is committed ON, keyed by target_entity. Set on a
        # rising edge (OFF->ON) to the planned block length (falls back to min_on) so a
        # momentary plan OFF mid-block cannot chop the block into fragments (the planner
        # "walk" safety net). Boost/safety pass bypass_dwell=True and thus break the
        # commit; a rising edge is NOT committed once the daily floor is already met
        # (over-heat guard). In-memory: an add-on restart forgets an active commit
        # (bounded downside — min_on still bounds the block).
        self._water_commit_until: dict[str, float] = {}

    async def _read_control_pause(self, entity_id: str) -> bool:
        """Read one control-pause entity: True only if it is definitively 'on'.

        Fail-safe direction: None (missing), 'unavailable'/'unknown', or any read
        error => False (NOT paused = normal control), so a transient HA glitch on a
        local input_boolean never silently strands a device unmanaged.
        """
        try:
            value = await self.ha.get_state_value(entity_id)
        except Exception as exc:  # fail-safe: any read error => not paused (normal control)
            logger.warning(
                "control_pause read failed for %s (%s) - treating as NOT paused",
                entity_id,
                exc,
            )
            return False
        if value is None:
            return False
        return str(value).strip().lower() == "on"

    async def control_pause_entity(
        self,
        entities: list[str] | None,
        cache: dict[str, bool] | None = None,
    ) -> str | None:
        """Return the first control-pause entity that is 'on' (PAUSED), else None.

        A device wired with ``control_pause_entities`` is PAUSED (hands-off — the
        executor skips all actuation and leaves whatever a human set) if ANY of its
        entities reads state ``on``. Unreadable/unknown entities are treated as NOT
        paused (see :meth:`_read_control_pause`).

        ``cache`` (optional, per-tick) memoizes each entity's paused-ness so an
        input_boolean shared across several devices (e.g. the villavagn master
        toggle gating both the VVB and the AC sink) is read at most once per tick.
        """
        if not entities:
            return None
        for entity_id in entities:
            if not entity_id:
                continue
            if cache is not None and entity_id in cache:
                paused = cache[entity_id]
            else:
                paused = await self._read_control_pause(entity_id)
                if cache is not None:
                    cache[entity_id] = paused
            if paused:
                return entity_id
        return None

    async def is_control_paused(
        self,
        entities: list[str] | None,
        cache: dict[str, bool] | None = None,
    ) -> bool:
        """True if any of ``entities`` is definitively 'on'. See control_pause_entity."""
        return (await self.control_pause_entity(entities, cache)) is not None

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
        # ValueError/TypeError: float(value) on a number-domain entity with a
        # non-numeric configured value (e.g. on_value: "on"). Return False so
        # the caller emits a failed (loud) ActionResult instead of the config
        # error escaping and aborting the whole actuation pass.
        except (HACallError, ValueError, TypeError) as e:
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

    async def set_water_temp(
        self,
        target: int,
        target_entity: str | None = None,
        *,
        bypass_dwell: bool = False,
        commit_minutes: float | None = None,
        heated_today_kwh: float | None = None,
        min_kwh_per_day: float | None = None,
    ) -> ActionResult:
        """Set water heater target temperature.

        Args:
            target: Target temperature in °C
            target_entity: HA entity to control. If None, falls back to
                           config.water_heater.target_entity (legacy single-heater path).
            bypass_dwell: When True, skip the anti-short-cycle min-on/min-off dwell
                          gate AND the block-commit latch (switch/input_boolean targets
                          only). Used by boost (force ON) and safety/override (force OFF)
                          paths that must actuate immediately. Normal plan-driven calls
                          leave this False so the dwell caps toggling.
            commit_minutes: Build #16 — planned length (minutes) of the heating block
                          this call belongs to, sourced from the current schedule slot's
                          contiguous ON run. On a rising edge (OFF->ON) the relay is
                          committed ON for this long so a mid-block plan OFF cannot chop
                          the block. None => fall back to min_on_minutes.
            heated_today_kwh: Build #16 over-heat guard — MEASURED energy already
                          delivered to THIS heater in the current day-bucket. Combined
                          with ``min_kwh_per_day``: once the daily floor is met, a rising
                          edge is NOT committed (so a met-floor block is not held ON
                          longer than the plan wants). None => guard inactive.
            min_kwh_per_day: Build #16 — this heater's daily minimum (the cold-shower
                          floor). See ``heated_today_kwh``.
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

        # Direct relay targets (switch./input_boolean.): drive the heater ourselves
        # instead of writing a temperature helper and trusting an HA bridge automation
        # to relay it — an unloaded/stuck bridge stranded the relay ON for days while
        # every "command" landed in a helper nothing read. Semantics match the bridges:
        # target above the threshold (temp_normal/boost/max) = heat, temp_off = off.
        # The executor re-reads and re-asserts every run, so any external drift
        # (device reboot, manual toggle) self-heals within one tick.
        if entity.startswith(("switch.", "input_boolean.")):
            desired_state = "on" if float(target) > WATER_SWITCH_ON_THRESHOLD_C else "off"
            current_state = str(current).strip().lower() if current is not None else None
            if current_state == desired_state:
                if desired_state == "on":
                    # Plan and reality agree on ON: ratify ownership (so the eventual
                    # plan-off is OURS to enforce) and clear any manual window.
                    self._last_water_cmd[entity] = "on"
                    self._manual_on_until.pop(entity, None)
                    # Seed the dwell epoch if we have none (e.g. after an add-on
                    # restart where reality already matches the plan) so the eventual
                    # OFF has a valid time-since-change to measure against.
                    self._last_water_switch_ts.setdefault(entity, time.time())
                return ActionResult(
                    action_type="water_temp",
                    success=True,
                    message=f"Already {desired_state} ({target}°C)",
                    previous_value=current_state,
                    new_value=desired_state,
                    entity_id=entity,
                    skipped=True,
                    duration_ms=int((time.time() - start) * 1000),
                    error_details=None,
                )
            # Manual-ON respect: the relay is ON but the plan wants OFF, and WE did not
            # turn it on — a human did. That's a signal, not drift: honor it as an
            # implicit boost for manual_on_respect_minutes instead of reverting on the
            # next tick (the switch on the wall becomes an honest boost button). Only
            # states the executor commanded itself are enforced against.
            if desired_state == "off" and current_state == "on":
                respect_min = float(
                    getattr(self.config.water_heater, "manual_on_respect_minutes", 0.0) or 0.0
                )
                if respect_min > 0 and self._last_water_cmd.get(entity) != "on":
                    until = self._manual_on_until.get(entity)
                    if until is None:
                        until = time.time() + respect_min * 60.0
                        self._manual_on_until[entity] = until
                        logger.info(
                            "Water heater %s manually turned ON — respecting as implicit "
                            "boost for %.0f min",
                            entity,
                            respect_min,
                        )
                    if time.time() < until:
                        remaining_min = int((until - time.time()) / 60)
                        return ActionResult(
                            action_type="water_temp",
                            success=True,
                            message=(
                                f"Manual ON respected (implicit boost, {remaining_min} min left)"
                            ),
                            previous_value=current_state,
                            new_value="on",
                            entity_id=entity,
                            skipped=True,
                            duration_ms=int((time.time() - start) * 1000),
                            error_details=None,
                        )
                    # Window expired: clear it and fall through to enforce OFF.
                    self._manual_on_until.pop(entity, None)
            # Anti-short-cycle dwell gate: a flip is needed (current != desired).
            # Hold the relay in its current state until it has dwelled long enough,
            # unless a boost (force ON) or safety/override (force OFF) bypasses it.
            # This bounds the toggle rate regardless of why the plan flip-flopped
            # (mid-slot replan relocation, marginal WTP flip-flop, external toggles,
            # verification retries). Placed AFTER the ratify and manual-ON blocks so
            # those short-circuit first and are never double-handled.
            if not bypass_dwell:
                # Build #16 block-commit latch: once a heating block has started, hold
                # the relay ON for the planned block length even if the plan momentarily
                # flips OFF (the planner-walk safety net). Checked BEFORE the min-on/off
                # dwell so a committed block is never cut short by a marginal plan flip.
                # Boost/safety pass bypass_dwell=True and so are never blocked here — the
                # floor and forced-OFF overrides always win.
                commit_until = self._water_commit_until.get(entity)
                if (
                    commit_until is not None
                    and desired_state == "off"
                    and current_state == "on"
                    and time.time() < commit_until
                ):
                    remaining_min = int((commit_until - time.time()) / 60)
                    logger.info(
                        "Water heater %s block-commit hold: %d min left in committed "
                        "block (plan wants OFF)",
                        entity,
                        remaining_min,
                    )
                    return ActionResult(
                        action_type="water_temp",
                        success=True,
                        message=(
                            f"Block-commit hold: {remaining_min} min left before OFF"
                        ),
                        previous_value=current_state,
                        new_value=current_state,
                        entity_id=entity,
                        skipped=True,
                        duration_ms=int((time.time() - start) * 1000),
                        error_details=None,
                    )
                last_ts = self._last_water_switch_ts.get(entity)
                if last_ts is not None:
                    if current_state == "on":
                        # About to turn OFF: honor minimum ON time.
                        required_s = (
                            float(getattr(self.config.water_heater, "min_on_minutes", 0.0) or 0.0)
                            * 60.0
                        )
                    else:
                        # About to turn ON: honor minimum OFF time.
                        required_s = (
                            float(getattr(self.config.water_heater, "min_off_minutes", 0.0) or 0.0)
                            * 60.0
                        )
                    elapsed = time.time() - last_ts
                    if required_s > 0 and elapsed < required_s:
                        remaining_min = int((required_s - elapsed) / 60)
                        logger.info(
                            "Water heater %s dwell hold: %d min left before %s "
                            "(current=%s, min_on=%.0f min, min_off=%.0f min)",
                            entity,
                            remaining_min,
                            desired_state,
                            current_state,
                            float(getattr(self.config.water_heater, "min_on_minutes", 0.0) or 0.0),
                            float(getattr(self.config.water_heater, "min_off_minutes", 0.0) or 0.0),
                        )
                        return ActionResult(
                            action_type="water_temp",
                            success=True,
                            message=(
                                f"Dwell hold: {remaining_min} min left before {desired_state}"
                            ),
                            previous_value=current_state,
                            new_value=current_state,
                            entity_id=entity,
                            skipped=True,
                            duration_ms=int((time.time() - start) * 1000),
                            error_details=None,
                        )
            if self.shadow_mode:
                logger.info(
                    "[SHADOW] Would switch water heater %s %s (%s°C)",
                    entity,
                    desired_state,
                    target,
                )
                return ActionResult(
                    action_type="water_temp",
                    success=True,
                    message=f"[SHADOW] Would switch {current_state} → {desired_state}",
                    previous_value=current_state,
                    new_value=desired_state,
                    entity_id=entity,
                    skipped=True,
                    duration_ms=int((time.time() - start) * 1000),
                    error_details=None,
                )
            switch_error: str | None = None
            try:
                switch_ok = await self.ha.set_switch(entity, desired_state == "on")
            except Exception as exc:
                switch_ok = False
                switch_error = str(exc)
            if switch_ok:
                self._last_water_cmd[entity] = desired_state
                # Record the dwell epoch on every actual commanded state change.
                self._last_water_switch_ts[entity] = time.time()
                if desired_state == "on":
                    self._manual_on_until.pop(entity, None)
                    # Build #16: rising edge (OFF->ON). Latch a block commitment so a
                    # mid-block plan OFF cannot chop this block — UNLESS the daily floor
                    # is already met (over-heat guard: don't hold a met-floor top-up ON
                    # longer than the plan wants). When bypass_dwell (boost) we still
                    # record the epoch but do NOT commit — boost is re-asserted every
                    # tick and must never trap the relay past its own logic.
                    floor_met = (
                        heated_today_kwh is not None
                        and min_kwh_per_day is not None
                        and min_kwh_per_day > 0
                        and heated_today_kwh >= min_kwh_per_day
                    )
                    if not bypass_dwell and not floor_met:
                        min_on_s = (
                            float(
                                getattr(self.config.water_heater, "min_on_minutes", 0.0) or 0.0
                            )
                            * 60.0
                        )
                        block_s = (
                            commit_minutes * 60.0
                            if commit_minutes is not None and commit_minutes > 0
                            else 0.0
                        )
                        # Only latch an EXTENDED commit when the planned block is longer
                        # than the min_on the dwell ALREADY guarantees. A shorter/absent
                        # block length needs no separate latch — the min_on dwell is the
                        # documented "fall back to min_on" hold, and latching a second
                        # min_on-length commit on its own clock would only duplicate it.
                        if block_s > min_on_s:
                            self._water_commit_until[entity] = time.time() + block_s
                            logger.info(
                                "Water heater %s block-commit latched: holding ON for "
                                "%.0f min (planned block > min_on)",
                                entity,
                                block_s / 60.0,
                            )
                        else:
                            self._water_commit_until.pop(entity, None)
                    else:
                        # Floor met (or boost): no commit — let the plan/dwell govern OFF.
                        self._water_commit_until.pop(entity, None)
                else:
                    # Commanded OFF (block ended / plan OFF after commit expiry): clear
                    # the commitment so the min_off dwell governs the next ON.
                    self._water_commit_until.pop(entity, None)
            return ActionResult(
                action_type="water_temp",
                success=switch_ok,
                message=(
                    f"Water heater {desired_state} ({target}°C)"
                    if switch_ok
                    else f"Failed to switch {entity} {desired_state}"
                ),
                previous_value=current_state,
                new_value=desired_state,
                entity_id=entity,
                duration_ms=int((time.time() - start) * 1000),
                error_details=switch_error,
            )

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
        """Legacy wrapper: actuate the FIRST excess-PV sink (old single-slot API).

        Args:
            value: The value to set (on_value or off_value from config).

        Returns:
            ActionResult indicating success or failure.
        """
        from .config import ExcessPVSinkType

        excess_pv = self.config.excess_pv
        # The loader populates .sinks (with lossless legacy synthesis); directly
        # constructed configs (tests, old call sites) may only carry .custom_entity,
        # which IS a sink spec (ExcessPVSinkSpec aliases the same dataclass).
        sink_cfg = excess_pv.sinks[0] if excess_pv.sinks else excess_pv.custom_entity

        if excess_pv.sinks:
            # Loader-produced ladder: the rung's own flag is authoritative
            # (disabled rungs are never actuated — observe-first invariant).
            sink_active = sink_cfg.enabled
        else:
            # Directly-constructed legacy config: build #9 semantics.
            sink_active = excess_pv.sink == ExcessPVSinkType.CUSTOM_ENTITY or sink_cfg.enabled
        if not sink_active or not sink_cfg.entity:
            return ActionResult(
                action_type=f"sink:{sink_cfg.id}",
                success=True,
                message="Custom entity sink not configured",
                skipped=True,
                duration_ms=0,
            )
        return await self.set_sink(sink_cfg, self._values_match(value, sink_cfg.on_value))

    async def set_sink(self, cfg: ExcessPVSinkSpec, turn_on: bool) -> ActionResult:
        """Actuate one rung of the excess-PV sink ladder.

        Domain-dispatched: ``climate.*`` entities get hvac_mode/temperature service
        calls (with the anti-overcool comfort floor); switch/input_boolean entities
        are turned on/off; everything else gets ``on_value``/``off_value`` written
        via the matching domain service. Idempotent — a no-op when the entity is
        already in the desired state — so the executor's every-tick writes are
        self-healing rather than churn.
        """
        start = time.time()
        entity = cfg.entity

        if not entity:
            return ActionResult(
                action_type=f"sink:{cfg.id}",
                success=True,
                message=f"Sink {cfg.id} has no entity configured",
                skipped=True,
                duration_ms=int((time.time() - start) * 1000),
            )

        domain = entity.split(".", 1)[0] if "." in entity else "switch"

        # Climate entities (e.g. the villavagn AC as a surplus cooling sink) need
        # hvac_mode/temperature services, not a plain state write. Delegate.
        if domain == "climate":
            return await self._set_climate_sink(entity, turn_on, cfg, start)

        # Switches speak on/off natively — derive the value from the boolean so a
        # numeric on_value like "1"/"0" can't be mis-coerced (bool("0") is True).
        if domain in ("switch", "input_boolean"):
            value: Any = "on" if turn_on else "off"
            write_value: Any = turn_on
        else:
            value = cfg.on_value if turn_on else cfg.off_value
            write_value = value

        current = await self.ha.get_state_value(entity)

        if self._values_match(current, value):
            return ActionResult(
                action_type=f"sink:{cfg.id}",
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
                "[SHADOW] Would set sink %s (%s) to %s (current: %s)",
                cfg.id,
                entity,
                value,
                current,
            )
            return ActionResult(
                action_type=f"sink:{cfg.id}",
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
            success = await self._write_entity(entity, write_value, domain)
        except HACallError as e:
            success = False
            error_details = str(e)
            logger.error("Failed to set sink %s (%s): %s", cfg.id, entity, error_details)

        verified_value = None
        verification_success = None
        if success:
            verified_value, verification_success = await self._verify_action(entity, value)

        duration = int((time.time() - start) * 1000)

        return ActionResult(
            action_type=f"sink:{cfg.id}",
            success=success,
            message=(
                f"Changed {current} → {value}"
                if success
                else f"Failed: {error_details}"
                if error_details
                else f"Failed to set sink {cfg.id}"
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
        cfg: ExcessPVSinkSpec,
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
                action_type=f"sink:{cfg.id}",
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
            return _result(
                True, f"[SHADOW] Would change {current_mode} -> {desired_mode}", skipped=True
            )

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
                action_type=f"sink:{cfg.id}",
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
        """Price-conditioned export curtailment (C3).

        method="number" (legacy): clamp the feed-in NUMBER to 0 W below the
        threshold, restore it above (mode switch stays on).
        method="switch": below the threshold write clamp_limit_w (a known
        device-legal low value) — _set_max_export_power also enables the mode
        switch (F49) — and above the threshold turn the mode switch OFF, which
        is truly unlimited and never risks an out-of-range number write.
        """
        cc = self.config.export_curtailment
        if export_price_sek_kwh < cc.threshold_sek_per_kwh:
            if cc.method == "switch":
                logger.info(
                    "Export curtailment ACTIVE (switch method): effective export %.3f < %.3f "
                    "SEK/kWh -> limit %.0f W + mode switch ON",
                    export_price_sek_kwh,
                    cc.threshold_sek_per_kwh,
                    cc.clamp_limit_w,
                )
                return await self._set_max_export_power(cc.clamp_limit_w)
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

        if cc.method == "switch":
            # Restore = unlimited: mode switch OFF. No number write — the device
            # may reject high values (Sungrow: 10000 -> pymodbus isError).
            return await self._set_export_limit_switch(False)

        # Not curtailing: restore the feed-in limit if we know it (config value or captured).
        restore = (
            cc.restore_limit_w if cc.restore_limit_w > 0 else (self._restore_export_limit_w or 0.0)
        )
        if restore > 0:
            return await self._set_max_export_power(restore)
        return None

    async def _set_export_limit_switch(self, on: bool) -> ActionResult | None:
        """Set the export-limit MODE switch, idempotently (skip when already there)."""
        start = time.time()
        entity = self.config.inverter.grid_max_export_power_switch or self._resolve_entity_id(
            "export_power_limit_switch"
        )
        if not _is_entity_configured(entity) or entity is None:
            logger.debug("Skipping export-limit switch action: no switch entity available")
            return None

        current = await self.ha.get_state_value(entity)
        want = "on" if on else "off"
        if current is not None and str(current).lower() == want:
            return None  # already in the desired state — no write, no EEPROM churn

        if self.shadow_mode:
            logger.info("[SHADOW] Would set export-limit switch %s -> %s", entity, want)
            return None

        try:
            success = await self.ha.set_switch(entity, on)
        except HACallError as e:
            logger.warning("Failed to set export-limit switch %s -> %s: %s", entity, want, e)
            success = False
        logger.info("Set export-limit switch %s -> %s (success=%s)", entity, want, success)
        return ActionResult(
            action_type="export_limit_switch",
            success=bool(success),
            entity_id=entity,
            previous_value=current,
            new_value=want,
            duration_ms=int((time.time() - start) * 1000),
        )

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
        switch_entity = (
            self.config.inverter.grid_max_export_power_switch
            or self._resolve_entity_id("export_power_limit_switch")
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

    async def notify_unverified(self, target: str, detail: str, recovered: bool = False) -> None:
        """Tell a human that a write never reached the appliance — or that it has.

        Separate from notify_error: this is not an exception, it is the quieter and
        more dangerous case where every call SUCCEEDED and the device ignored us.
        """
        if not self.config.notifications.on_write_unverified:
            return
        if recovered:
            await self._send_notification(
                f"{target}: reagerar igen — {detail}",
                title="Darkstar: kontroll återställd",
            )
        else:
            await self._send_notification(
                f"{target} följer inte Darkstars kommando.\n{detail}",
                title="⚠️ Darkstar: skrivning gick inte igenom",
            )

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
