"""Regression tests for the 2026-10-11 audit's executor safety batch."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pandas as pd
import pytest

from executor.actions import HAClient
from executor.config import ExecutorConfig
from executor.controller import make_decision
from executor.engine import HEARTBEAT_ENTITY, ExecutorEngine
from executor.override import SlotPlan, SystemState


# --- DST -------------------------------------------------------------------


@pytest.mark.parametrize("utc", ["2026-10-25 00:20", "2026-10-25 01:20", "2026-03-29 00:50"])
def test_now_slot_rounding_survives_the_dst_nights(utc: str) -> None:
    """The planner's rounding recipe must not raise in the repeated/missing hour."""
    tz = "Europe/Stockholm"
    t = pd.Timestamp(utc, tz="UTC")
    floored = t.floor("15min").tz_convert(tz)
    ceiled = t.ceil("15min").tz_convert(tz)
    assert floored <= t.tz_convert(tz) <= ceiled
    assert floored.minute % 15 == 0 and ceiled.minute % 15 == 0


def test_pipeline_no_longer_floors_a_local_timestamp() -> None:
    src = Path("planner/pipeline.py").read_text(encoding="utf-8")
    assert 'Timestamp.now(tz=tz).floor' not in src
    assert '.tz_convert(tz).ceil("15min")' not in src


# --- reload guard ----------------------------------------------------------


def _engine(tmp_path: Path, body: str = "timezone: Europe/Stockholm\nexecutor:\n  enabled: true\n") -> ExecutorEngine:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body, encoding="utf-8")
    return ExecutorEngine(config_path=str(cfg))


def _touch(path: Path, body: str) -> None:
    import os

    path.write_text(body, encoding="utf-8")
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 5))


def test_unparsable_config_keeps_the_running_one(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.reload_config()
    assert eng.config.enabled is True
    _touch(tmp_path / "config.yaml", "executor:\n  enabled: true\n  enabled: [unclosed\n")
    asyncio.run(eng._reload_config_or_keep())
    assert eng.config.enabled is True, "a broken file must not swap in defaults"
    assert eng._config_reload_error is not None


def test_missing_config_does_not_raise_out_of_the_loop(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.reload_config()
    (tmp_path / "config.yaml").unlink()
    asyncio.run(eng._reload_config_or_keep())
    assert eng.config.enabled is True
    assert "FileNotFoundError" in (eng._config_reload_error or "")


def test_reload_recovers_and_clears_the_error(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.reload_config()
    _touch(tmp_path / "config.yaml", "{{{{")
    asyncio.run(eng._reload_config_or_keep())
    assert eng._config_reload_error is not None
    _touch(tmp_path / "config.yaml", "timezone: Europe/Stockholm\nexecutor:\n  enabled: false\n")
    asyncio.run(eng._reload_config_or_keep())
    assert eng._config_reload_error is None
    assert eng.config.enabled is False


# --- heartbeat -------------------------------------------------------------


def test_heartbeat_only_after_a_dispatch(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    eng = _engine(tmp_path)
    eng.ha_client = MagicMock()
    eng.ha_client.set_state = AsyncMock(return_value=True)
    now = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)

    eng._dispatched_this_tick = False
    asyncio.run(eng._publish_heartbeat(now))
    eng.ha_client.set_state.assert_not_called()

    eng._dispatched_this_tick = True
    asyncio.run(eng._publish_heartbeat(now))
    args = eng.ha_client.set_state.call_args.args
    assert args[0] == HEARTBEAT_ENTITY
    assert args[1].startswith("2026-10-11T12:00:00")


# --- unknown SoC -----------------------------------------------------------


def _export_slot() -> SlotPlan:
    return SlotPlan(export_kw=5.0, discharge_kw=5.0, soc_target=10, export_price_sek_kwh=3.0)


def test_export_runs_when_soc_is_known_and_high() -> None:
    cfg = ExecutorConfig()
    d = make_decision(_export_slot(), SystemState(current_soc_percent=80.0), None, cfg.controller, cfg.inverter)
    assert d.mode_intent == "export"


def test_export_is_blocked_when_soc_is_unknown() -> None:
    cfg = ExecutorConfig()
    state = SystemState(soc_known=False)  # value still the 50 % default
    d = make_decision(_export_slot(), state, None, cfg.controller, cfg.inverter)
    assert d.mode_intent == "self_consumption"


# --- 404 on a read ---------------------------------------------------------


def test_a_404_read_is_unreadable_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    client = HAClient("http://ha.invalid", "token")

    class _Resp:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        def raise_for_status(self) -> None:
            raise aiohttp.ClientResponseError(MagicMock(), (), status=404, message="Not Found")

    session = MagicMock()
    session.get = MagicMock(return_value=_Resp())
    monkeypatch.setattr(client, "_get_session", AsyncMock(return_value=session))

    assert asyncio.run(client.get_state("sensor.sungrow_inverter_state")) is None
    assert asyncio.run(client.get_state_value("sensor.sungrow_inverter_state")) is None
