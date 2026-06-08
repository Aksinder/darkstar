import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException

from backend.loads.service import LoadDisaggregator

from .config import get_config

router = APIRouter(prefix="/api/loads", tags=["loads"])
logger = logging.getLogger("darkstar.api.loads")

# Simple singleton pattern for the disaggregator within the API
_disaggregator: LoadDisaggregator | None = None


def get_disaggregator(config: dict[str, Any] = Depends(get_config)) -> LoadDisaggregator:
    global _disaggregator
    if _disaggregator is None:
        _disaggregator = LoadDisaggregator(config)
    elif (
        len(_disaggregator.loads_registry) == 0
        and len(cast("list[Any]", config.get("deferrable_loads", []))) > 0
    ):
        # Re-initialize if the registry is empty but config now has loads (e.g. after first setup)
        _disaggregator = LoadDisaggregator(config)
    return _disaggregator


@router.get("/controllable")
async def get_controllable_loads(
    config: dict[str, Any] = Depends(get_config),
) -> dict[str, list[dict[str, Any]]]:
    """Enumerate all CONFIG-DEFINED controllable loads (deferrable + water + EV), including
    disabled ones, so the Load Priority GUI can assign each to a tier. Read-only; mirrors
    the arrays already exposed by GET /api/config."""

    def rows(items: Any, kind: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for x in cast("list[dict[str, Any]]", items or []):
            load_id = str(x.get("id", ""))
            if not load_id:
                continue
            out.append(
                {
                    "id": load_id,
                    "name": str(x.get("name") or load_id),
                    "type": kind,
                    "enabled": bool(x.get("enabled", True)),
                }
            )
        return out

    return {
        "loads": rows(config.get("deferrable_loads"), "deferrable")
        + rows(config.get("water_heaters"), "water_heater")
        + rows(config.get("ev_chargers"), "ev_charger")
    }


@router.get("/debug")
async def get_loads_debug(
    disaggregator: LoadDisaggregator = Depends(get_disaggregator),
) -> dict[str, Any]:
    """
    Returns current disaggregation breakdown and sensor health.
    """
    try:
        # Note: In a real production scenario, the recorder or a background service
        # would be updating this. For the debug endpoint, we'll trigger a fresh fetch.
        controllable_kw = await disaggregator.update_current_power()

        # We don't have total_load_kw here easily without fetching it from HA.
        # However, the registry and metrics are already populated if the system is running.

        loads_breakdown: list[dict[str, Any]] = []
        for load in disaggregator.list_active_loads():
            loads_breakdown.append(
                {
                    "id": load.id,
                    "name": load.name,
                    "power_kw": load.current_power_kw,
                    "healthy": load.is_healthy,
                    "type": load.type.value,
                    "sensor": load.sensor_key,
                    "disabled_reason": load.disabled_reason,
                }
            )

        return {
            "controllable_total_kw": controllable_kw,
            "loads": loads_breakdown,
            "quality_metrics": disaggregator.get_quality_metrics(),
        }
    except Exception as e:
        logger.error(f"Failed to fetch loads debug data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
