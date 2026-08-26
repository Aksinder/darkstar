"""The schedule writer must never leave a truncated file for the executor.

Live 2026-08-26 04:36: the executor read data/schedule.json mid-rewrite and got
"Expecting value: line 2398" — the writer used a bare truncating open("w").
The write now goes to a sibling .tmp and renames over the target.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import pytz

from planner.output.schedule import save_schedule_to_json

TZ = pytz.timezone("Europe/Stockholm")


def _df() -> pd.DataFrame:
    idx = pd.date_range("2026-08-26 10:00", periods=4, freq="15min", tz=TZ)
    return pd.DataFrame(
        {
            "start_time": idx,
            "end_time": idx + pd.Timedelta(minutes=15),
            "import_price_sek_kwh": [1.0, 1.1, 1.2, 1.3],
            "export_price_sek_kwh": [0.5] * 4,
            "pv_forecast_kwh": [0.2] * 4,
            "load_forecast_kwh": [0.3] * 4,
            "battery_charge_kw": [0.0] * 4,
            "battery_discharge_kw": [0.0] * 4,
            "grid_import_kwh": [0.1] * 4,
            "grid_export_kwh": [0.0] * 4,
            "water_heating_kw": [0.0] * 4,
            "projected_soc_percent": [50.0] * 4,
            "soc_target_percent": [50.0] * 4,
        },
        index=idx,
    )


async def _write(path: Path) -> None:
    await save_schedule_to_json(
        _df(),
        config={"timezone": "Europe/Stockholm"},
        now_slot=None,
        forecast_meta={},
        s_index_debug=None,
        window_responsibilities=[],
        planner_state={},
        output_path=str(path),
    )


class TestAtomicScheduleWrite:
    @pytest.mark.asyncio
    async def test_writes_valid_json(self, tmp_path):
        out = tmp_path / "schedule.json"
        await _write(out)
        data = json.loads(out.read_text())
        assert data["schedule"], "schedule must not be empty"
        assert not (tmp_path / "schedule.json.tmp").exists(), "temp file must be renamed away"

    @pytest.mark.asyncio
    async def test_crash_mid_write_leaves_previous_file_intact(self, tmp_path):
        out = tmp_path / "schedule.json"
        await _write(out)
        original = out.read_text()

        # A dump that dies halfway must not touch the published file.
        with patch("planner.output.schedule.json.dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                await _write(out)

        assert out.read_text() == original, "reader-visible file must be the previous version"
        assert json.loads(out.read_text())["schedule"], "and it must still parse"
