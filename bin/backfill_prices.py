import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytz
import yaml
from nordpool.elspot import Prices

sys.path.insert(0, str(Path.cwd()))
# Was `from learning import LearningEngine` -- there is no top-level `learning`
# package, so this script raised ImportError before it did anything. Combined with the
# un-awaited call below, it has never once stored a price.
from backend.learning import LearningEngine


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        raw_data: Any = yaml.safe_load(f)
    return cast("dict[str, Any]", raw_data) if isinstance(raw_data, dict) else {}


async def backfill_prices(days: int, config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    engine: Any = LearningEngine(config_path)  # type: ignore[assignment]

    nordpool_config: dict[str, Any] = config.get("nordpool", {}) or {}
    price_area: str = nordpool_config.get("price_area", "SE4")
    currency: str = nordpool_config.get("currency", "SEK")
    resolution_minutes: int = nordpool_config.get("resolution_minutes", 60)

    prices_client = Prices(currency)

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    print(f"Fetching prices from {start_date} to {end_date} for {price_area}...")

    # Nordpool library fetches by day. We iterate.
    current_date = start_date
    all_entries: list[dict[str, Any]] = []

    while current_date <= end_date:
        try:
            print(f"  Fetching {current_date}...")
            data: dict[str, Any] = prices_client.fetch(
                end_date=current_date, areas=[price_area], resolution=resolution_minutes
            )
            if data and data.get("areas") and data["areas"].get(price_area):
                values: list[Any] = data["areas"][price_area].get("values", [])
                all_entries.extend(values)
        except Exception as e:
            print(f"  Failed to fetch {current_date}: {e}")

        current_date += timedelta(days=1)

    if not all_entries:
        print("No data fetched.")
        return

    # Process and Store
    # Re-use logic from inputs.py _process_nordpool_data but simplified for storage
    pricing_config: dict[str, Any] = config.get("pricing", {}) or {}
    vat_percent: float = pricing_config.get("vat_percent", 25.0)
    grid_transfer_fee_sek: float = pricing_config.get("grid_transfer_fee_sek", 0.2456)
    energy_tax_sek: float = pricing_config.get("energy_tax_sek", 0.439)
    local_tz = pytz.timezone(config.get("timezone", "Europe/Stockholm"))

    records: list[dict[str, Any]] = []
    entry: dict[str, Any]
    for entry in all_entries:
        start_time = entry["start"].astimezone(local_tz)
        end_time = entry["end"].astimezone(local_tz)

        spot_price_sek_kwh = entry["value"] / 1000.0
        export_price_sek_kwh = spot_price_sek_kwh
        import_price_sek_kwh = (spot_price_sek_kwh + grid_transfer_fee_sek + energy_tax_sek) * (
            1 + vat_percent / 100.0
        )

        records.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "import_price_sek_kwh": import_price_sek_kwh,
                "export_price_sek_kwh": export_price_sek_kwh,
            }
        )

    # store_slot_prices is async. Without the await this built a coroutine, dropped it,
    # and printed "Done." -- a silent no-op that looked like a successful backfill.
    print(f"Storing {len(records)} price slots...")
    await engine.store_slot_prices(records)
    print(
        "Done. (Prices attach to EXISTING observation rows only; slots with no "
        "recorded observation are skipped by design.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Nordpool prices")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backfill")
    args = parser.parse_args()

    asyncio.run(backfill_prices(args.days))
