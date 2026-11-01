"""Config save must re-arm the HA websocket listener (port of upstream 885b296a).

Before this fix ``reload_ha_socket_client()`` had no caller, so an entity added or
changed in Settings was ignored by the websocket listener until a restart.
"""

from pathlib import Path as _Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import backend.ha_socket as ha_socket


def _bare_client(monitored: dict[str, str]) -> ha_socket.HAWebSocketClient:
    client = object.__new__(ha_socket.HAWebSocketClient)
    client.monitored_entities = dict(monitored)
    client.latest_values = {}
    client.ev_charger_configs = []
    client.water_heater_configs = []
    client.inversion_flags = {}
    return client


def test_reload_returns_only_newly_monitored_entities() -> None:
    client = _bare_client({"sensor.old": "soc"})
    with (
        patch.object(client, "_load_config"),
        patch.object(
            client,
            "_get_monitored_entities",
            return_value={"sensor.old": "soc", "sensor.new": "pv_power"},
        ),
    ):
        added = client.reload_monitored_entities()
    assert added == {"sensor.new"}
    assert client.monitored_entities["sensor.new"] == "pv_power"


@pytest.mark.asyncio
async def test_reload_and_seed_pushes_current_state_of_new_entities() -> None:
    client = _bare_client({"sensor.old": "soc"})
    seen: list[tuple[str, dict[str, Any]]] = []
    state = {"entity_id": "sensor.new", "state": "1.5"}
    with (
        patch.object(client, "_load_config"),
        patch.object(
            client,
            "_get_monitored_entities",
            return_value={"sensor.old": "soc", "sensor.new": "pv_power"},
        ),
        patch.object(client, "_handle_state_change", side_effect=lambda e, s: seen.append((e, s))),
        patch(
            "backend.core.ha_client.get_ha_entity_state", new=AsyncMock(return_value=state)
        ) as fetch,
    ):
        added = await client.reload_and_seed_entities()
    assert added == {"sensor.new"}
    fetch.assert_awaited_once_with("sensor.new")  # unchanged entities are not re-fetched
    assert seen == [("sensor.new", state)]


@pytest.mark.asyncio
async def test_config_save_reloads_ha_socket_entities(tmp_path, monkeypatch) -> None:
    from ruamel.yaml import YAML

    import backend.api.routers.config as config_router

    yaml_loader = YAML()
    base = {"timezone": "Europe/Stockholm", "system": {}, "battery": {}, "executor": {}}
    for name in ("config.yaml", "config.default.yaml"):
        with (tmp_path / name).open("w") as f:
            yaml_loader.dump(base, f)

    monkeypatch.setattr(
        config_router,
        "Path",
        lambda p: tmp_path / p if p in ["config.yaml", "config.default.yaml"] else _Path(p),
    )
    monkeypatch.setattr(config_router, "write_config", lambda *a, **k: True)
    monkeypatch.setattr(config_router, "get_executor_instance", lambda: None)
    monkeypatch.setattr(config_router, "_validate_config_for_save", lambda x: [])

    reload_mock = AsyncMock(return_value={"sensor.new"})
    monkeypatch.setattr(ha_socket, "reload_ha_socket_client_async", reload_mock)

    await config_router.save_config({"timezone": "Europe/Stockholm"})

    reload_mock.assert_awaited_once()
