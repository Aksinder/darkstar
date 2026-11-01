"""Lifecycle of the shared, per-event-loop Home Assistant httpx client."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.core import ha_client, secrets
from backend.core.ha_client import close_ha_http_client, get_ha_http_client


@pytest.mark.asyncio
async def test_same_loop_reuses_one_client():
    first = get_ha_http_client()
    assert isinstance(first, httpx.AsyncClient)
    assert get_ha_http_client() is first
    await close_ha_http_client()


@pytest.mark.asyncio
async def test_closed_client_is_replaced():
    client = get_ha_http_client()
    await client.aclose()  # closed behind the cache's back
    replacement = get_ha_http_client()
    assert replacement is not client
    assert not replacement.is_closed
    await close_ha_http_client()


@pytest.mark.asyncio
async def test_close_closes_and_forgets():
    client = get_ha_http_client()
    await close_ha_http_client()
    assert client.is_closed
    fresh = get_ha_http_client()
    assert fresh is not client and not fresh.is_closed
    await close_ha_http_client()
    # Closing twice (nothing cached) is a no-op.
    await close_ha_http_client()


def test_other_loop_gets_its_own_client_and_dead_loops_are_pruned():
    """A client bound to another (now closed) loop is never handed out again."""

    async def grab():
        client = get_ha_http_client()
        return client, asyncio.get_running_loop(), list(ha_client._ha_http_clients)

    client_a, _, _ = asyncio.run(grab())  # loop A, closed when asyncio.run returns
    client_b, loop_b, cached_loops = asyncio.run(grab())  # loop B
    assert client_a is not client_b
    # While loop B ran, loop A's dead entry had been pruned: only B is cached.
    assert cached_loops == [loop_b]


def test_executor_thread_loop_does_not_share_server_loop_client():
    """The executor runs its own loop in a thread; it must get a different client."""
    got: dict[str, httpx.AsyncClient] = {}

    async def grab(name: str) -> None:
        got[name] = get_ha_http_client()
        await asyncio.sleep(0)
        assert get_ha_http_client() is got[name]
        await close_ha_http_client()

    def thread_main() -> None:
        asyncio.run(grab("thread"))

    t = threading.Thread(target=thread_main)
    t.start()
    t.join()
    asyncio.run(grab("main"))
    assert got["thread"] is not got["main"]
    assert got["thread"].is_closed and got["main"].is_closed


def test_no_running_loop_raises():
    with pytest.raises(RuntimeError):
        get_ha_http_client()


@pytest.mark.asyncio
async def test_power_history_passes_per_call_timeout():
    from datetime import UTC, datetime, timedelta

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [[]]
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    with (
        patch(
            "backend.core.ha_client.secrets.load_home_assistant_config",
            return_value={"url": "http://ha", "token": "t"},
        ),
        patch("backend.core.ha_client.get_ha_http_client", return_value=client),
    ):
        await ha_client.get_energy_from_power_history("sensor.p", end - timedelta(hours=1), end)
    assert client.get.call_args.kwargs["timeout"] == 12.0


def test_secrets_cache_rereads_on_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "secrets.yaml"
    path.write_text("home_assistant:\n  url: http://a\n  token: x\n")
    assert secrets.load_home_assistant_config()["url"] == "http://a"

    # Cached: a second read does not touch the YAML parser.
    with patch("backend.core.secrets.yaml.safe_load") as safe_load:
        cfg = secrets.load_home_assistant_config()
        safe_load.assert_not_called()
    cfg["url"] = "mutated"  # callers get a copy
    assert secrets.load_home_assistant_config()["url"] == "http://a"

    path.write_text("home_assistant:\n  url: http://bb\n  token: x\n")  # size changes
    assert secrets.load_home_assistant_config()["url"] == "http://bb"

    path.unlink()
    assert secrets.load_home_assistant_config() == {}
