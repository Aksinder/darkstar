"""Opt-in, per-event-loop shared Home Assistant httpx client."""

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core import ha_client, secrets
from backend.core.ha_client import (
    close_ha_http_client,
    enable_shared_ha_http_client,
    ha_http_client,
)


@pytest.mark.asyncio
async def test_unregistered_loop_gets_fresh_client_closed_after_use():
    """Short-lived loops (recorder/ml asyncio.run) behave exactly as before."""
    async with ha_http_client() as first:
        assert not first.is_closed
    assert first.is_closed
    async with ha_http_client() as second:
        assert second is not first
    assert ha_client._ha_http_clients == {}  # nothing cached for this loop


@pytest.mark.asyncio
async def test_registered_loop_reuses_one_open_client():
    enable_shared_ha_http_client()
    async with ha_http_client() as first:
        pass
    assert not first.is_closed  # shared: left open
    async with ha_http_client() as second:
        assert second is first
    await close_ha_http_client()
    assert first.is_closed
    # Closing opts the loop out again: back to per-call clients.
    async with ha_http_client() as third:
        assert third is not first
    assert third.is_closed


@pytest.mark.asyncio
async def test_closed_shared_client_is_replaced():
    enable_shared_ha_http_client()
    async with ha_http_client() as client:
        pass
    await client.aclose()  # closed behind the cache's back
    async with ha_http_client() as replacement:
        assert replacement is not client
        assert not replacement.is_closed
    await close_ha_http_client()


def test_registered_loops_never_share_and_dead_loops_are_pruned():
    """Server loop vs executor thread loop: distinct clients; closed loops dropped."""
    got: dict = {}

    async def long_lived(name: str, close: bool) -> None:
        enable_shared_ha_http_client()
        async with ha_http_client() as c:
            got[name] = c
        if close:
            await close_ha_http_client()

    t = threading.Thread(target=lambda: asyncio.run(long_lived("executor", close=False)))
    t.start()
    t.join()
    assert len(ha_client._ha_http_clients) == 1  # executor loop died without closing

    async def server() -> list:
        await long_lived("server", close=False)
        loops = list(ha_client._ha_http_clients)
        await close_ha_http_client()
        return loops

    loops = asyncio.run(server())
    assert got["executor"] is not got["server"]
    assert len(loops) == 1  # the dead executor loop was pruned on registration
    assert got["server"].is_closed


def test_no_running_loop_raises():
    with pytest.raises(RuntimeError):
        enable_shared_ha_http_client()


@pytest.mark.asyncio
async def test_power_history_passes_per_call_timeout():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [[]]
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    with (
        patch(
            "backend.core.ha_client.secrets.load_home_assistant_config",
            return_value={"url": "http://ha", "token": "t"},
        ),
        patch("backend.core.ha_client.httpx.AsyncClient", return_value=client),
    ):
        await ha_client.get_energy_from_power_history("sensor.p", end - timedelta(hours=1), end)
    assert client.get.call_args.kwargs["timeout"] == 12.0
    client.__aexit__.assert_called_once()  # unregistered loop: closed after the call


def test_secrets_cache_rereads_on_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "secrets.yaml"
    path.write_text("home_assistant:\n  url: http://a\n  token: x\n")
    assert secrets.load_home_assistant_config()["url"] == "http://a"

    with patch("backend.core.secrets.yaml.safe_load") as safe_load:
        cfg = secrets.load_home_assistant_config()
        safe_load.assert_not_called()
    cfg["url"] = "mutated"  # callers get a copy
    assert secrets.load_home_assistant_config()["url"] == "http://a"

    path.write_text("home_assistant:\n  url: http://bb\n  token: x\n")  # size changes
    assert secrets.load_home_assistant_config()["url"] == "http://bb"

    path.unlink()
    assert secrets.load_home_assistant_config() == {}
