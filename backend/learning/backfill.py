import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytz
import yaml

from backend.learning import get_learning_engine

# Configure logging
logger = logging.getLogger(__name__)


class BackfillEngine:
    """
    Handles backfilling of missing observations from Home Assistant history and MariaDB.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.engine = get_learning_engine(config_path)
        self.store = self.engine.store
        self.ha_config = self._load_ha_config()
        self.timezone = pytz.timezone(self.config.get("timezone", "Europe/Stockholm"))
        self.learning_config = self.config.get("learning", {})

        # Load secrets for backfill fallback (HA)
        self.secrets = self._load_secrets()

    def _load_config(self, path: str) -> dict[str, Any]:
        try:
            with Path(path).open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _load_secrets(self) -> dict[str, Any]:
        try:
            with Path("secrets.yaml").open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _load_ha_config(self) -> dict[str, Any]:
        """Load HA config from secrets.yaml"""
        secrets = self._load_secrets()
        return secrets.get("home_assistant", {})

    def _make_ha_headers(self) -> dict[str, str]:
        token = self.ha_config.get("token")
        if not token:
            return {}
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _fetch_history(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> list[tuple[datetime, float]]:
        """Fetch history for a single entity from HA asynchronously."""
        url = self.ha_config.get("url")
        if not url or not entity_id:
            return []

        api_url = f"{url.rstrip('/')}/api/history/period/{start_time.isoformat()}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": end_time.isoformat(),
            "significant_changes_only": False,
            "minimal_response": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(api_url, headers=self._make_ha_headers(), params=params)
                response.raise_for_status()
                data = response.json()

            if not data or not data[0]:
                return []

            history: list[tuple[datetime, float]] = []
            state_item: dict[str, Any]
            for state_item in data[0]:
                try:
                    ts = datetime.fromisoformat(state_item["last_changed"])
                    val = float(state_item["state"])
                    history.append((ts, val))
                except (ValueError, TypeError, KeyError):
                    continue
            return history

        except Exception as e:
            logger.error(f"Failed to fetch history for {entity_id}: {e}")
            return []

    async def run(self) -> None:
        """Run the backfill process asynchronously."""
        # 2026-08-08. This engine has been UNREACHABLE for the entire life of the
        # zero-fabrication bug: get_last_observation_time (store.py) is
        # MAX(slot_start), and store_slot_prices minted observation rows ~48h into the
        # FUTURE, so `gap` below was always negative and the "Data is up to date."
        # branch always returned early. Stopping the fabrication makes MAX(slot_start)
        # the present again, which would wake this path for the first time in months.
        #
        # It is NOT safe to wake it silently: etl_cumulative_to_slots zero-fills absent
        # channels (engine.py: `deltas[deltas > max_kwh] = 0.0` and `slot_df[col] = 0.0`)
        # and the auto sensor map below has no battery/water/EV entry. On this site
        # total_pv_production was deliberately emptied on 2026-07-08, so it would write
        # pv_kwh=0.0 with load_kwh>0 -- rows that PASS ml/train.py's `load_kwh > 0.001`
        # filter and would feed the LIVE automatic retrain as genuine zero-PV daytime
        # samples. Default OFF; enabling it is a deliberate, reviewed act.
        if not self.learning_config.get("ha_backfill_enabled", False):
            logger.info(
                "BackfillEngine: disabled (learning.ha_backfill_enabled=false). This "
                "reproduces the engine's actual pre-2026-08-08 behaviour. Enable only "
                "after auditing etl_cumulative_to_slots zero-fill and the sensor map's "
                "battery/water/EV coverage."
            )
            return

        logger.info("Starting backfill process...")

        # 1. Sync from Home Assistant (Primary Source)
        try:
            # Check last observation time
            last_obs = await self.store.get_last_observation_time()
            now = datetime.now(self.timezone)

            # Default lookback if empty DB (e.g., 7 days)
            if not last_obs:
                logger.info("No existing observations. Backfilling last 7 days.")
                start_time = now - timedelta(days=7)
            else:
                # Check gap
                gap = now - last_obs
                if gap < timedelta(minutes=15):
                    logger.info("Data is up to date.")
                    return

                logger.info(f"Found data gap of {gap}. Starting backfill from {last_obs}.")
                start_time = last_obs

            # Cap backfill to 10 days to avoid overloading HA
            if (now - start_time) > timedelta(days=10):
                start_time = now - timedelta(days=10)
                logger.warning("Gap too large, capping backfill to last 10 days.")

            # 2. Identify sensors to fetch
            # REV // F26: Fallback to input_sensors if sensor_map is empty
            raw_map_source: dict[str, str] | None = self.learning_config.get("sensor_map")
            raw_map: dict[str, str] = raw_map_source or {}
            if not raw_map:
                logger.info("sensor_map is empty. Auto-detecting from config...")
                input_sensors = self.config.get("input_sensors", {})
                raw_map = {}
                # Map cumulative sensors (ARC15: water heater sensors now in water_heaters[])
                mapping = {
                    "total_grid_import": "import",
                    "total_grid_export": "export",
                    "total_pv_production": "pv",
                    "total_load_consumption": "load",
                    "battery_soc": "soc",
                }
                for config_key, canonical in mapping.items():
                    entity_id = input_sensors.get(config_key)
                    if entity_id:
                        raw_map[entity_id] = canonical

            if not raw_map:
                logger.warning(
                    "No sensors identified for backfill (sensor_map and input_sensors empty)."
                )
                return

            cumulative_data: dict[str, list[tuple[datetime, float]]] = {}
            count = 0
            for entity_id_str, canonical_str in raw_map.items():
                logger.info(f"Backfilling {canonical_str} ({entity_id_str})...")
                history = await self._fetch_history(entity_id_str, start_time, now)
                if history:
                    cumulative_data[entity_id_str] = history
                    count += len(history)

            if not cumulative_data:
                logger.warning("No history data found for any sensors.")
                return

            logger.info(f"Fetched {count} data points. Processing into slots...")

            # 3. ETL to slots (CPU-bound, wrap in to_thread for 100% production grade)
            import asyncio

            df = await asyncio.to_thread(self.engine.etl_cumulative_to_slots, cumulative_data)

            if df.empty:
                logger.warning("ETL produced empty DataFrame.")
                return

            logger.info(f"Generated {len(df)} slots. Storing to DB...")

            # 4. Store
            await self.engine.store_slot_observations(df)
            logger.info("Backfill complete.")

        except Exception as e:
            logger.error(f"Backfill failed during ETL/Storage: {e}")
