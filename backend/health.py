"""
Health Check System

Centralized health monitoring for Darkstar.
Validates HA connection, entity availability, config validity, and planner metrics via SQLite.
"""

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytz
import yaml

from backend.exceptions import PVForecastError

logger = logging.getLogger(__name__)

# REV F60: Forecast error tracking (like executor's recent_errors)
# Phase 8: Thread-safe access with lock
_forecast_errors: deque[dict[str, Any]] = deque(maxlen=10)
_forecast_status: str = "ok"  # "ok", "degraded", "error"
_forecast_lock: threading.Lock = threading.Lock()

# REV F65 Phase 5b: Load forecast status tracking
_load_forecast_status: str = "ok"  # "ok", "degraded"
_load_forecast_reason: str = ""  # "ml", "baseline", "demo", ""
_load_forecast_lock: threading.Lock = threading.Lock()


def record_forecast_error(error: Exception, context: dict[str, Any] | None = None) -> None:
    """Record a forecast error for health monitoring.

    Args:
        error: The exception that occurred
        context: Additional context about the forecast attempt
    """
    global _forecast_status

    error_entry: dict[str, Any] = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "type": type(error).__name__,
        "message": str(error),
        "context": context or {},
    }

    if isinstance(error, PVForecastError):
        error_entry["solar_arrays"] = getattr(error, "solar_arrays", 0)  # type: ignore[dict-item]
        error_entry["details"] = getattr(error, "details", {})  # type: ignore[dict-item]

    with _forecast_lock:
        _forecast_status = "error" if isinstance(error, PVForecastError) else "degraded"
        _forecast_errors.append(error_entry)

    logger.error("Forecast error recorded: %s", error_entry["message"])


def clear_forecast_errors() -> None:
    """Clear forecast errors (called after successful forecast)."""
    global _forecast_status
    with _forecast_lock:
        _forecast_errors.clear()
        _forecast_status = "ok"


def get_forecast_errors(limit: int = 5) -> list[dict[str, Any]]:
    """Get recent forecast errors (newest first)."""
    with _forecast_lock:
        return list(_forecast_errors)[-limit:]


def get_forecast_status() -> dict[str, Any]:
    """Get current forecast health status."""
    with _forecast_lock:
        return {
            "status": _forecast_status,
            "last_errors": list(_forecast_errors)[-5:],
            "error_count": len(_forecast_errors),
        }


def set_load_forecast_status(status: str, reason: str = "") -> None:
    """Set load forecast status for health monitoring.

    Args:
        status: "ok" or "degraded"
        reason: "ml" (ML models working), "baseline" (using baseline avg),
                "demo" (using demo data), "no_ml" (ML unavailable but data exists)
    """
    global _load_forecast_status, _load_forecast_reason
    with _load_forecast_lock:
        _load_forecast_status = status
        _load_forecast_reason = reason

    if status == "degraded":
        logger.warning(f"⚠️ Load forecast degraded: {reason}")


def get_load_forecast_status() -> dict[str, Any]:
    """Get current load forecast health status."""
    with _load_forecast_lock:
        return {
            "status": _load_forecast_status,
            "reason": _load_forecast_reason,
        }


def clear_load_forecast_status() -> None:
    """Clear load forecast degraded status (called after successful ML forecast)."""
    global _load_forecast_status, _load_forecast_reason
    with _load_forecast_lock:
        _load_forecast_status = "ok"
        _load_forecast_reason = ""


@dataclass
class HealthIssue:
    """A single health issue with guidance."""

    category: str  # "ha_connection", "entity", "config", "planner", "executor"
    severity: str  # "critical", "warning", "info"
    message: str  # User-friendly message
    guidance: str  # How to fix
    entity_id: str | None = None  # Specific entity involved (if applicable)
    code: str | None = None  # Machine-readable error code
    details: dict[str, Any] | None = None  # Structured diagnostic data
    retry_in_s: int | None = None  # Seconds until next planner retry
    config_blocking: bool = False  # True when the error requires a config change to resolve

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "guidance": self.guidance,
        }
        if self.entity_id is not None:
            d["entity_id"] = self.entity_id
        if self.code is not None:
            d["code"] = self.code
        if self.details is not None:
            d["details"] = self.details
        if self.retry_in_s is not None:
            d["retry_in_s"] = self.retry_in_s
        if self.config_blocking:
            d["config_blocking"] = True
        return d


@dataclass
class HealthStatus:
    """Overall system health status."""

    healthy: bool
    issues: list[HealthIssue] = field(default_factory=list[HealthIssue])
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now(pytz.UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_at": self.checked_at,
            "critical_count": len([i for i in self.issues if i.severity == "critical"]),
            "warning_count": len([i for i in self.issues if i.severity == "warning"]),
        }


class HealthChecker:
    """
    Comprehensive system health checker.

    Validates:
    - Home Assistant connection
    - Configured entity availability
    - Config file validity
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._secrets: dict[str, Any] = {}

    def check_planner(self) -> list[HealthIssue]:
        """Check planner service health from last error state."""
        issues: list[HealthIssue] = []

        try:
            from backend.services.planner_service import planner_service
            from planner.errors import fix_hints, is_config_blocking, user_message

            if planner_service.last_error_code is None:
                return issues

            code = planner_service.last_error_code
            severity = "critical" if is_config_blocking(code) else "warning"
            hints = fix_hints(code)

            issues.append(
                HealthIssue(
                    category="planner",
                    severity=severity,
                    message=user_message(code),
                    guidance=hints[0] if hints else "",
                    code=code.value,
                    details=planner_service.last_error_details,
                    retry_in_s=planner_service.retry_in_s,
                    config_blocking=is_config_blocking(code),
                )
            )
        except Exception as e:
            logger.debug("Could not check planner health: %s", e)

        return issues

    def check_plan_realism(self) -> list[HealthIssue]:
        """Surface the realism gap as a PHASE-IMBALANCE indicator, not lost money.

        The realism simulation computes what per-phase billing would cost over the
        net-node plan. Swedish settlement meters net all three phases momentarily
        (STAFS 2022:9 2 kap. 7 §, confirmed by Energimarknadsinspektionen), so that
        delta never reaches the invoice here — the gap's kWh (extra_import_kwh) are
        still worth surfacing, because energy crossing phases is fuse headroom being
        burnt on the heavy phase. History: until 2026-08-20 this warning recommended
        enabling phase_aware "for the economics", which steered this site into a
        term that double-priced ordinary import — the opposite of help. Thresholds
        unchanged (>= 2 SEK-equivalent and >= 20% of plan gross value) so the signal
        still only fires when the imbalance is material.
        """
        issues: list[HealthIssue] = []
        try:
            import json
            from typing import cast

            schedule_path = Path("data/schedule.json")
            if not schedule_path.exists():
                return issues
            with schedule_path.open() as f:
                schedule_raw: Any = json.load(f)
            if not isinstance(schedule_raw, dict):
                return issues
            schedule = cast("dict[str, Any]", schedule_raw)
            meta = cast("dict[str, Any]", schedule.get("meta") or {})
            s_index = cast("dict[str, Any]", meta.get("s_index") or {})
            realism = cast("dict[str, Any]", s_index.get("realism") or {})
            gap = float(realism.get("gap_sek", 0.0) or 0.0)
            slots = cast("list[Any]", schedule.get("schedule") or [])
            gross_value_sek = 0.0
            for slot_raw in slots:
                if isinstance(slot_raw, dict):
                    slot = cast("dict[str, Any]", slot_raw)
                    gross_value_sek += abs(float(slot.get("cost_sek", 0.0) or 0.0))
            if gap >= 2.0 and gap >= 0.2 * max(gross_value_sek, 1e-9):
                # Only quote a percentage against a meaningful denominator — against a
                # near-zero-value plan it renders as astronomical nonsense.
                pct_part = (
                    f" (~{100.0 * gap / gross_value_sek:.0f}% of plan value)"
                    if gross_value_sek >= 1.0
                    else " (plan value near zero)"
                )
                extra_kwh = float(realism.get("extra_import_kwh", 0.0) or 0.0)
                issues.append(
                    HealthIssue(
                        category="planner",
                        severity="info",
                        message=(
                            f"Phase imbalance: {extra_kwh:.1f} kWh crosses phases over "
                            f"the plan horizon (hypothetical {gap:.2f} SEK{pct_part})"
                        ),
                        guidance=(
                            "One phase imports while the others export. Swedish meters "
                            "net all phases momentarily (STAFS 2022:9), so this costs "
                            "nothing on the invoice — but it is real fuse headroom being "
                            "burnt on the chronically heavy phase. Review the phase "
                            "balance (move single-phase loads off the heavy phase), or "
                            "enable phase_aware fuse relief so the planner buys extra "
                            "discharge when a phase approaches the main fuse. Do NOT "
                            "expect an economic gain from either: this signal is about "
                            "amps, not money."
                        ),
                    )
                )
        except Exception as e:
            logger.debug("Could not check plan realism: %s", e)
        return issues

    async def check_all(self) -> HealthStatus:
        """Run all health checks and return combined status."""
        issues: list[HealthIssue] = []

        # Load config first (needed for other checks)
        config_issues = self.check_config_validity()
        issues.extend(config_issues)

        # If config is valid, proceed with other checks
        if not any(i.category == "config" and i.severity == "critical" for i in issues):
            issues.extend(await self.check_ha_connection())

            # Only check entities if HA is connected
            if not any(i.category == "ha_connection" for i in issues):
                issues.extend(await self.check_entities())

        # Check executor health
        issues.extend(self.check_executor())

        # Check recorder health (REV // Complete Cost Reality Fix)
        issues.extend(self.check_recorder())

        # Check observation coverage + zero-fabrication tripwire (2026-08-08)
        issues.extend(await self.check_observation_coverage())

        # Check forecast health (REV F60)
        issues.extend(self.check_forecast())

        # Check load forecast health (REV F65 Phase 5c)
        issues.extend(self.check_load_forecast())

        # Check planner health (error codes + retry policy)
        issues.extend(self.check_planner())

        # Check plan realism gap (phase-aware simulation vs single-net-node plan)
        issues.extend(self.check_plan_realism())

        # Determine overall health
        has_critical = any(i.severity == "critical" for i in issues)
        healthy = not has_critical

        return HealthStatus(healthy=healthy, issues=issues)

    def check_config_validity(self) -> list[HealthIssue]:
        """Validate config.yaml exists and has required structure."""
        issues: list[HealthIssue] = []

        # Load config
        try:
            with self.config_path.open(encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            issues.append(
                HealthIssue(
                    category="config",
                    severity="critical",
                    message="Configuration file not found",
                    guidance=(
                        f"Copy config.default.yaml to {self.config_path} "
                        "and configure your settings."
                    ),
                )
            )
            return issues
        except yaml.YAMLError as e:
            issues.append(
                HealthIssue(
                    category="config",
                    severity="critical",
                    message=f"Invalid YAML syntax in config file: {e}",
                    guidance=(
                        "Fix the YAML syntax error in config.yaml. "
                        "Check for incorrect indentation or special characters."
                    ),
                )
            )
            return issues

        # Load secrets
        try:
            with Path("secrets.yaml").open(encoding="utf-8") as f:
                self._secrets = yaml.safe_load(f) or {}
        except FileNotFoundError:
            issues.append(
                HealthIssue(
                    category="config",
                    severity="critical",
                    message="Secrets file not found",
                    guidance=(
                        "Create secrets.yaml with your Home Assistant URL and token. "
                        "See README for format."
                    ),
                )
            )
        except yaml.YAMLError as e:
            issues.append(
                HealthIssue(
                    category="config",
                    severity="critical",
                    message=f"Invalid YAML syntax in secrets file: {e}",
                    guidance="Fix the YAML syntax error in secrets.yaml.",
                )
            )

        # Validate required config sections
        issues.extend(self._validate_config_structure())

        return issues

    def _validate_config_structure(self) -> list[HealthIssue]:
        """Validate config has required sections and correct types."""
        issues: list[HealthIssue] = []

        if not self._config:
            return issues

        # Check input_sensors section exists
        if not self._config.get("input_sensors"):
            issues.append(
                HealthIssue(
                    category="config",
                    severity="warning",
                    message="No input_sensors configured",
                    guidance=(
                        "Add input_sensors section to config.yaml "
                        "to enable Home Assistant integration."
                    ),
                )
            )

        # Validate HA secrets
        if self._secrets:
            ha_config = self._secrets.get("home_assistant", {})
            if not ha_config.get("url"):
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="critical",
                        message="Home Assistant URL not configured",
                        guidance=(
                            "Add home_assistant.url to secrets.yaml "
                            "(e.g., http://homeassistant.local:8123)"
                        ),
                    )
                )
            if not ha_config.get("token"):
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="critical",
                        message="Home Assistant token not configured",
                        guidance=(
                            "Add home_assistant.token to secrets.yaml. "
                            "Generate a Long-Lived Access Token in HA."
                        ),
                    )
                )

        # REV LCL01: Validate system profile toggle consistency
        system_cfg = self._config.get("system", {})
        battery_cfg = self._config.get("battery", {})

        # Battery misconfiguration = critical (breaks MILP solver)
        if system_cfg.get("has_battery", True):
            capacity = battery_cfg.get("capacity_kwh", 0)
            try:
                capacity = float(capacity) if capacity else 0.0
            except (ValueError, TypeError):
                capacity = 0.0
            if capacity <= 0:
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="critical",
                        message="Battery enabled but capacity not configured",
                        guidance=(
                            "Set battery.capacity_kwh to your battery's capacity (e.g., 27.0), "
                            "or set system.has_battery to false."
                        ),
                    )
                )

        # Water heater misconfiguration = warning (feature disabled, not broken)
        if system_cfg.get("has_water_heater", True):
            water_heaters = self._config.get("water_heaters", [])
            has_power = any(
                h.get("enabled", True) and float(h.get("power_kw", 0) or 0) > 0
                for h in water_heaters
            )
            if not has_power:
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="warning",
                        message="Water heater enabled but power not configured",
                        guidance="Set water_heaters[].power_kw to your heater's power (e.g., 3.0), "
                        "or set system.has_water_heater to false.",
                    )
                )
            # An enabled heater with no control entity is planned-but-unactuatable: the
            # planner allocates it energy (and shapes the battery/export plan around that
            # phantom load) while the executor silently drops every command for it
            # (executor/config.py skips heaters without a target_entity). Surface it
            # loudly instead of failing silent.
            for idx, heater in enumerate(water_heaters):
                if not heater.get("enabled", True):
                    continue
                if not str(heater.get("target_entity", "") or "").strip():
                    heater_name = heater.get("name", f"Water Heater {idx + 1}")
                    issues.append(
                        HealthIssue(
                            category="config",
                            severity="critical",
                            message=(
                                f"Water heater '{heater_name}' has no control entity — "
                                f"its planned heating cannot be executed"
                            ),
                            guidance=(
                                f"Add 'target_entity' to water_heaters[{idx}] (the "
                                f"input_number/thermostat entity the executor should "
                                f"command). Until then the planner schedules this heater "
                                f"but nothing actuates it, so the plan's water allocation "
                                f"— and the battery/export economics built on it — are "
                                f"fiction. If the tank is intentionally uncontrolled, set "
                                f"enabled: false so the planner stops planning it."
                            ),
                        )
                    )

        # Solar misconfiguration = warning (PV forecasts will be zero)
        if system_cfg.get("has_solar", True):
            solar_arrays = system_cfg.get("solar_arrays", [])
            has_kwp = any(float(a.get("kwp", 0) or 0) > 0 for a in solar_arrays)
            if not has_kwp:
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="warning",
                        message="Solar enabled but panel size not configured",
                        guidance="Set system.solar_arrays[].kwp to your PV capacity (e.g., 10.0), "
                        "or set system.has_solar to false.",
                    )
                )

        # REV F61: Check for deprecated executor.ev_charger.penalty_levels
        executor_cfg = self._config.get("executor", {})
        ev_cfg = executor_cfg.get("ev_charger", {})
        if ev_cfg.get("penalty_levels"):
            issues.append(
                HealthIssue(
                    category="config",
                    severity="warning",
                    message="Deprecated setting: executor.ev_charger.penalty_levels",
                    guidance="This setting is deprecated and ignored. Use per-charger penalty levels in the EV Chargers section instead (accessible in Settings > Parameters).",
                )
            )

        # Task 10.3: Detect deprecated executor.ev_charger section (switch_entity, replan_*)
        # These fields have moved to per-device ev_chargers[].switch_entity etc.
        deprecated_executor_ev_fields = [
            k
            for k in ("switch_entity", "replan_on_plugin", "replan_on_unplug")
            if ev_cfg.get(k) not in (None, "", False)
        ]
        if deprecated_executor_ev_fields:
            issues.append(
                HealthIssue(
                    category="config",
                    severity="warning",
                    message=(
                        f"Deprecated setting(s) in executor.ev_charger: "
                        f"{', '.join(deprecated_executor_ev_fields)}"
                    ),
                    guidance=(
                        "These fields have moved to per-device settings in ev_chargers[]. "
                        "Run config migration (Settings > Advanced > Migrate Config) or manually "
                        "move them to each charger's switch_entity / replan_on_plugin / replan_on_unplug field."
                    ),
                )
            )

        return issues

    async def check_ha_connection(self) -> list[HealthIssue]:
        """Check if Home Assistant is reachable."""
        issues: list[HealthIssue] = []

        if not self._secrets:
            return issues  # Already reported in config check

        ha_config = self._secrets.get("home_assistant", {})
        url = ha_config.get("url", "").rstrip("/")
        token = ha_config.get("token", "")

        if not url or not token:
            return issues  # Already reported in config check

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{url}/api/",
                    headers={"Authorization": f"Bearer {token}"},
                )

            if response.status_code == 401:
                issues.append(
                    HealthIssue(
                        category="ha_connection",
                        severity="critical",
                        message="Home Assistant authentication failed",
                        guidance=(
                            "Your HA token is invalid or expired. "
                            "Generate a new Long-Lived Access Token in HA → Profile → Security."
                        ),
                    )
                )
            elif response.status_code != 200:
                issues.append(
                    HealthIssue(
                        category="ha_connection",
                        severity="critical",
                        message=f"Home Assistant returned error: HTTP {response.status_code}",
                        guidance="Check that your Home Assistant URL is correct and HA is running.",
                    )
                )

        except httpx.TimeoutException:
            issues.append(
                HealthIssue(
                    category="ha_connection",
                    severity="critical",
                    message="Home Assistant connection timed out",
                    guidance="Home Assistant is slow or unreachable. Check network connectivity.",
                )
            )
        except httpx.RequestError as e:
            issues.append(
                HealthIssue(
                    category="ha_connection",
                    severity="critical",
                    message=f"Cannot connect to Home Assistant: {e}",
                    guidance=f"Check that Home Assistant is running and reachable at {url}",
                )
            )
        except Exception as e:
            issues.append(
                HealthIssue(
                    category="ha_connection",
                    severity="critical",
                    message=f"Unexpected error connecting to HA: {e}",
                    guidance="Check your network and Home Assistant configuration.",
                )
            )

        return issues

    async def check_entities(self) -> list[HealthIssue]:
        """Check if configured entities exist in Home Assistant."""
        issues: list[HealthIssue] = []

        if not self._config or not self._secrets:
            return issues

        ha_config = self._secrets.get("home_assistant", {})
        url = ha_config.get("url", "").rstrip("/")
        token = ha_config.get("token", "")

        if not url or not token:
            return issues

        # Feature flags
        system_cfg = self._config.get("system", {})
        has_battery = system_cfg.get("has_battery", True)
        has_water_heater = system_cfg.get("has_water_heater", True)
        has_solar = system_cfg.get("has_solar", True)

        # Collect all entity IDs from config with feature context
        # (entity_id, config_key, required)
        entities_to_check: list[tuple[str, str, bool]] = []

        # Input sensors
        input_sensors = self._config.get("input_sensors", {})

        # Grid meter configuration
        grid_meter_type = system_cfg.get("grid_meter_type", "net")
        is_net_metering = grid_meter_type == "net"

        # Define which sensors are HARD requirements for core functionality
        # If False, a missing entity is a WARNING, not a CRITICAL error.
        # REV F65: Cumulative sensors are REQUIRED for forecasting/ML
        # NOTE: today_* sensors are DEPRECATED (v2.6.1-beta) - energy data comes from DB
        learning_cfg = self._config.get("learning", {})
        is_learning_enabled = learning_cfg.get("enable", False)

        sensor_requirements = {
            # Core energy sensors (CRITICAL)
            "battery_soc": has_battery,
            "load_power": True,
            "pv_power": has_solar,
            "grid_power": is_net_metering,
            "grid_import_power": not is_net_metering,
            "grid_export_power": not is_net_metering,
            # Cumulative sensors (REQUIRED for forecasting/ML - F65)
            "total_load_consumption": is_learning_enabled,
            # The recorder integrates pv_power over each slot when this cumulative
            # counter is absent (a Sungrow-only kWh counter omits an AC-coupled
            # Fronius), so pv_power satisfies the PV energy requirement on its own.
            "total_pv_production": is_learning_enabled and not bool(input_sensors.get("pv_power")),
            "total_grid_import": is_learning_enabled,
            "total_grid_export": is_learning_enabled,
            "total_battery_charge": is_learning_enabled,
            "total_battery_discharge": is_learning_enabled,
            # Features (WARNING if missing but enabled)
            # ARC15: water_power and water_heater_consumption removed - now in water_heaters[]
            "alarm_state": False,  # Optional
            "vacation_mode": False,  # Optional
        }

        # Sensors that should be completely skipped if not relevant for current meter type
        sensors_to_skip = []
        if is_net_metering:
            sensors_to_skip = ["grid_import_power", "grid_export_power"]
        else:
            sensors_to_skip = ["grid_power"]

        for key, entity_id in input_sensors.items():
            if key in sensors_to_skip:
                continue

            if entity_id and isinstance(entity_id, str):
                # Is this sensor tied to a hardware toggle?
                hardware_enabled = sensor_requirements.get(key, True)

                # Skip checking if hardware is disabled
                if hardware_enabled is False and key in ["battery_soc", "pv_power"]:
                    continue

                entities_to_check.append((entity_id, f"input_sensors.{key}", hardware_enabled))

        # Executor entities
        executor = self._config.get("executor", {})
        if executor:
            # Inverter entities - Require has_battery
            if has_battery:
                inverter = executor.get("inverter", {})
                for key in [
                    "work_mode_entity",
                    "grid_charging_entity",
                    "max_charging_current_entity",
                    "max_discharging_current_entity",
                ]:
                    entity_id = inverter.get(key)
                    if entity_id:
                        entities_to_check.append((entity_id, f"executor.inverter.{key}", True))

                # Check soc_target_entity (requires battery)
                soc_target = executor.get("soc_target_entity")
                if soc_target:
                    entities_to_check.append((soc_target, "executor.soc_target_entity", True))

            # ARC15: Water heater sensors now in water_heaters[] array
            # Per-heater checks are handled below

            # General toggle entities - Always check
            for key in ["automation_toggle_entity"]:
                entity_id = executor.get(key)
                if entity_id:
                    entities_to_check.append((entity_id, f"executor.{key}", True))

        # NEW: Check for MISSING required sensors that aren't even in input_sensors
        for req_key, is_required in sensor_requirements.items():
            if is_required and req_key not in input_sensors:
                # Core sensor is missing from config entirely
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="critical",
                        message=f"Missing required sensor: {req_key}",
                        guidance=(
                            f"Add '{req_key}' to input_sensors in config.yaml. "
                            f"This is required for {grid_meter_type} metering."
                        ),
                    )
                )

        # REV F65: Add forecasting-specific warnings when learning is enabled
        if is_learning_enabled:
            cumulative_sensors = [
                "total_load_consumption",
                "total_pv_production",
                "total_grid_import",
                "total_grid_export",
                "total_battery_charge",
                "total_battery_discharge",
            ]
            missing_cumulative = [
                s for s in cumulative_sensors if s not in input_sensors or not input_sensors.get(s)
            ]
            # The recorder integrates the pv_power sensor over each slot when the
            # total_pv_production cumulative counter is absent, so pv_power on its own
            # is a valid (and on split-inverter sites, more accurate) PV energy source.
            # Don't warn about the missing counter when pv_power is configured.
            if "total_pv_production" in missing_cumulative and input_sensors.get("pv_power"):
                missing_cumulative.remove("total_pv_production")
            if missing_cumulative:
                issues.append(
                    HealthIssue(
                        category="config",
                        severity="warning",
                        message="Forecasting may use inaccurate fallback data",
                        guidance=(
                            f"Learning/forecasting is enabled but missing energy sensors: "
                            f"{', '.join(missing_cumulative)}. "
                            f"Forecasting will fall back to lower-accuracy profiles. "
                            f"Add these sensors (or the matching power sensor) to input_sensors "
                            f"for accurate forecasts."
                        ),
                    )
                )

        # ARC15: Per-heater health checks for water_heaters[] array
        if has_water_heater:
            water_heaters = self._config.get("water_heaters", [])
            for idx, heater in enumerate(water_heaters):
                if heater.get("enabled", True):
                    heater_name = heater.get("name", f"Water Heater {idx + 1}")
                    if not heater.get("sensor"):
                        issues.append(
                            HealthIssue(
                                category="config",
                                severity="warning",
                                message=f"Water heater '{heater_name}' missing power sensor",
                                guidance=(
                                    f"Add 'sensor' to water_heaters[{idx}] in config.yaml "
                                    f"for power monitoring and load disaggregation."
                                ),
                            )
                        )

        # Check each entity concurrently using asyncio.gather
        headers = {"Authorization": f"Bearer {token}"}

        async def check_single_entity(entity_data: tuple[str, str, bool]) -> HealthIssue | None:
            """Check a single entity and return a HealthIssue if there's a problem."""
            entity_id, config_key, is_required = entity_data
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{url}/api/states/{entity_id}",
                        headers=headers,
                    )

                if response.status_code == 404:
                    # Downgrade severity if not a hard requirement
                    severity = "critical" if is_required else "warning"
                    return HealthIssue(
                        category="entity",
                        severity=severity,
                        message=f"Entity not found: {entity_id}",
                        guidance=(
                            f"Check that '{entity_id}' exists in Home Assistant. "
                            f"Update {config_key} in config.yaml if renamed."
                        ),
                        entity_id=entity_id,
                    )
                elif response.status_code == 200:
                    # Check for unavailable state
                    state_data = response.json()
                    state_value = state_data.get("state")
                    if state_value == "unavailable":
                        return HealthIssue(
                            category="entity",
                            severity="warning",
                            message=f"Entity unavailable: {entity_id}",
                            guidance=(
                                f"The entity '{entity_id}' exists but is "
                                "currently unavailable. Check your device/integration."
                            ),
                            entity_id=entity_id,
                        )
            except httpx.RequestError:
                # Connection issues already reported in check_ha_connection
                pass
            return None

        # Run all entity checks concurrently
        entity_results = await asyncio.gather(
            *[check_single_entity(entity_data) for entity_data in entities_to_check],
            return_exceptions=True,
        )

        # Collect results (filter out None and exceptions)
        for result in entity_results:
            if isinstance(result, HealthIssue):
                issues.append(result)
            elif isinstance(result, Exception):
                # Log unexpected errors but don't fail the entire health check
                logger.debug(f"Unexpected error checking entity: {result}")

        return issues

    def check_executor(self) -> list[HealthIssue]:
        """Check executor health status."""
        issues: list[HealthIssue] = []

        try:
            from backend.api.routers.executor import get_executor_health  # type: ignore[import]

            # type: ignore
            executor_health: dict[str, Any] = get_executor_health()  # type: ignore[call]

            if not executor_health.get("is_healthy"):  # type: ignore
                if executor_health.get("should_be_running") and not executor_health.get(  # type: ignore
                    "is_running"
                ):
                    issues.append(
                        HealthIssue(
                            category="executor",
                            severity="critical",
                            message="Executor should be running but is not active",
                            guidance=(
                                "The executor is enabled in config but not running. "
                                "Check executor logs or restart the service."
                            ),
                        )
                    )
                elif executor_health.get("has_error"):  # type: ignore
                    error_msg: str = str(executor_health.get("error", "Unknown error"))  # type: ignore
                    issues.append(
                        HealthIssue(
                            category="executor",
                            severity="warning",
                            message=f"Executor last run failed: {error_msg}",
                            guidance=(
                                "Check executor logs for details. "
                                "The error may be transient or indicate a configuration issue."
                            ),
                        )
                    )
        except Exception as e:
            logger.debug("Could not check executor health: %s", e)
            # Don't add an issue - executor health check is optional

        return issues

    def check_recorder(self) -> list[HealthIssue]:
        """Check recorder service health."""
        issues: list[HealthIssue] = []

        try:
            from backend.services.recorder_service import recorder_service

            status = recorder_service.status
            if not status.running:
                issues.append(
                    HealthIssue(
                        category="recorder",
                        severity="critical",
                        message="Recorder service is not running",
                        guidance="The observation recorder is inactive. This prevents 'Real' data from appearing in charts. Check server logs.",
                    )
                )
            elif status.last_error:
                issues.append(
                    HealthIssue(
                        category="recorder",
                        severity="warning",
                        message=f"Recorder encountered an error: {status.last_error}",
                        guidance="Check recorder logs for recent failures. Some observations may be missing.",
                    )
                )

            # Check if last recording was too long ago (e.g., > 30 mins)
            if status.last_record_at:
                from datetime import UTC, datetime

                delta = datetime.now(UTC) - status.last_record_at
                if delta.total_seconds() > 1800:  # 30 minutes
                    issues.append(
                        HealthIssue(
                            category="recorder",
                            severity="warning",
                            message=f"Recorder has not saved data in {int(delta.total_seconds() / 60)} minutes",
                            guidance="The recorder appears to be stalled. Check server logs.",
                        )
                    )
        except Exception as e:
            logger.debug("Could not check recorder health: %s", e)

        return issues

    async def check_observation_coverage(self) -> list[HealthIssue]:
        """Observation coverage + zero-fabrication tripwire (2026-08-08).

        A recorder collapse used to be INVISIBLE: the price mint pre-created a row for
        every slot, so a month of missing observations read as a bad FORECAST rather
        than as missing data. check_recorder() could not see it either --
        recorder_service.status.running stayed True throughout.
        """
        issues: list[HealthIssue] = []
        cov: dict[str, Any]

        try:
            from backend.learning.coverage import classify_coverage, observation_coverage
        except Exception as e:
            logger.warning("Observation coverage module unavailable: %s", e)
            return [
                HealthIssue(
                    category="observations",
                    severity="critical",
                    code="obs_coverage_unavailable",
                    message=f"Observation coverage module could not be loaded: {e}",
                    guidance=(
                        "The observation-integrity tripwire is blind. Forecast-accuracy "
                        "metrics cannot be trusted until this evaluates again."
                    ),
                )
            ]

        try:
            from backend.learning.store import OBS_FIX_APPLIED_KEY

            learning_cfg = self._config.get("learning") or {}
            db_path = learning_cfg.get("sqlite_path", "data/planner_learning.db")
            tz = pytz.timezone(self._config.get("timezone", "Europe/Stockholm"))

            fix_applied_at: str | None = None
            try:
                from backend.learning import get_learning_engine

                fix_applied_at = await get_learning_engine().store.get_system_state(
                    OBS_FIX_APPLIED_KEY
                )
            except Exception as e:
                logger.debug("Could not read %s: %s", OBS_FIX_APPLIED_KEY, e)

            # to_thread is mandatory: check_all is wrapped in asyncio.wait_for, which
            # cannot cancel a blocking sync call.
            cov = await asyncio.to_thread(observation_coverage, db_path, tz, 7, fix_applied_at)
        except Exception as e:
            logger.warning("Observation coverage check failed: %s", e)
            # A check that cannot evaluate must NOT return silence.
            cov = {"evaluable": False, "error": str(e), "rows_present": 0}

        for severity, code, message, guidance in classify_coverage(cov):
            issues.append(
                HealthIssue(
                    category="observations",
                    severity=severity,
                    code=code,
                    message=message,
                    guidance=guidance,
                    details=cov,
                )
            )
        return issues

    def check_forecast(self) -> list[HealthIssue]:
        """Check PV forecast health status."""
        issues: list[HealthIssue] = []

        forecast_info = get_forecast_status()
        status = forecast_info.get("status", "ok")

        if status == "error":
            # Get the most recent error
            recent_errors = forecast_info.get("last_errors", [])
            if recent_errors:
                latest = recent_errors[-1]
                error_msg = latest.get("message", "Unknown forecast error")

                issues.append(
                    HealthIssue(
                        category="forecast",
                        severity="critical",
                        message=f"PV Forecast Failed: {error_msg}",
                        guidance=(
                            "The PV forecast system is unable to generate accurate solar predictions. "
                            "Planning may be using outdated or invalid data. "
                            "Check your solar array configuration and Open-Meteo service availability."
                        ),
                    )
                )
        elif status == "degraded":
            issues.append(
                HealthIssue(
                    category="forecast",
                    severity="warning",
                    message="PV forecast experiencing issues",
                    guidance="Some forecast requests have failed but the system is still operational.",
                )
            )

        return issues

    def check_load_forecast(self) -> list[HealthIssue]:
        """Check Load forecast health status (REV F65 Phase 5c)."""
        issues: list[HealthIssue] = []

        load_info = get_load_forecast_status()
        status = load_info.get("status", "ok")
        reason = load_info.get("reason", "")

        if status == "degraded":
            if reason == "demo":
                issues.append(
                    HealthIssue(
                        category="forecast",
                        severity="warning",
                        message="Load forecast using demo data (0.5 kWh flat)",
                        guidance=(
                            "No historical load data available. The system is using a flat demo profile. "
                            "Configure 'total_load_consumption' sensor in input_sensors to enable accurate load forecasting."
                        ),
                    )
                )
            elif reason == "baseline":
                issues.append(
                    HealthIssue(
                        category="forecast",
                        severity="info",
                        message="Load forecast using baseline average (insufficient training data)",
                        guidance=(
                            "Not enough historical data to train ML models. The system is using baseline average (0.5 kWh/slot). "
                            "After 4+ days of data collection, statistical corrections will be applied. "
                            "After 14+ days, ML models will be trained for accurate predictions."
                        ),
                    )
                )
            elif reason == "no_ml":
                issues.append(
                    HealthIssue(
                        category="forecast",
                        severity="warning",
                        message="Load forecast ML models unavailable",
                        guidance=(
                            "Historical data exists but ML models are not trained. "
                            "Run the ML training pipeline to enable accurate load predictions. "
                            "Current forecast uses HA historical profile."
                        ),
                    )
                )

        return issues


async def get_health_status(config_path: str = "config.yaml") -> HealthStatus:
    """Convenience function to get current health status."""
    checker = HealthChecker(config_path)
    try:
        return await asyncio.wait_for(checker.check_all(), timeout=15.0)
    except TimeoutError:
        return HealthStatus(
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
