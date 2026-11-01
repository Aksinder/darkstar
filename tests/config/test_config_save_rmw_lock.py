"""Config writers hold CONFIG_WRITE_LOCK across the WHOLE read-modify-write.

Only the write used to be locked: /api/config/save and Aurora Reflex read the
file outside the lock, so two saves that both read before either wrote left the
later one silently dropping the earlier one's change (upstream 7ee2b2e8 idea).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from ruamel.yaml import YAML

import backend.api.routers.config as config_router
from backend.config_migration import CONFIG_WRITE_LOCK

if TYPE_CHECKING:
    from pathlib import Path

BASE = (
    "timezone: Europe/Stockholm\n"
    "system:\n  inverter_profile: sungrow\n"
    "battery:\n  capacity_kwh: 16.0\n"
    "executor:\n  enabled: true\n  shadow_mode: false\n"
    "input_sensors:\n  battery_soc: sensor.soc\n"
    "ui:\n  theme: dark\n"
)


@pytest.mark.asyncio
async def test_two_concurrent_saves_both_land(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(BASE, encoding="utf-8")
    (tmp_path / "config.default.yaml").write_text(BASE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_router, "get_executor_instance", lambda: None)
    monkeypatch.setattr(config_router, "_validate_config_for_save", lambda x: [])
    import backend.ha_socket as ha_socket

    monkeypatch.setattr(ha_socket, "reload_ha_socket_client_async", AsyncMock(return_value=set()))

    # Widen the read->write window so an unlocked read-modify-write interleaves
    # every time: both saves would read the original file, and the second write
    # would drop the first one's key.
    real_merge = config_router.template_aware_merge

    def slow_merge(*a: Any, **k: Any) -> Any:
        time.sleep(0.2)
        return real_merge(*a, **k)

    monkeypatch.setattr(config_router, "template_aware_merge", slow_merge)

    r1, r2 = await asyncio.gather(
        config_router.save_config({"ui": {"theme": "light"}}),
        config_router.save_config({"executor": {"shadow_mode": True}}),
    )
    assert r1["status"] == r2["status"] == "success"
    with (tmp_path / "config.yaml").open(encoding="utf-8") as f:
        saved = YAML().load(f)
    assert saved["ui"]["theme"] == "light"
    assert saved["executor"]["shadow_mode"] is True


@pytest.mark.asyncio
async def test_reflex_reads_and_writes_under_the_lock(tmp_path: Path) -> None:
    from backend.learning.reflex import AuroraReflex

    cfg = tmp_path / "config.yaml"
    cfg.write_text(BASE, encoding="utf-8")
    reflex = object.__new__(AuroraReflex)
    reflex.config_path = cfg
    reflex.yaml = YAML()
    reflex.store = MagicMock()
    reflex.store.update_reflex_state = AsyncMock()

    held_during_read: list[bool] = []
    real_load = reflex.yaml.load

    def probing_load(stream: Any) -> Any:
        got: list[bool] = []

        def probe() -> None:
            ok = CONFIG_WRITE_LOCK.acquire(blocking=False)
            if ok:
                CONFIG_WRITE_LOCK.release()  # pragma: no cover - only on failure
            got.append(ok)

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        held_during_read.append(not got[0])
        return real_load(stream)

    reflex.yaml.load = probing_load  # type: ignore[method-assign]
    import backend.learning.reflex as reflex_mod

    orig_append = reflex_mod.append_strategy_event
    reflex_mod.append_strategy_event = lambda **k: None  # type: ignore[assignment]
    try:
        await reflex.update_config({"battery.capacity_kwh": 17.0}, dry_run=False)
    finally:
        reflex_mod.append_strategy_event = orig_append  # type: ignore[assignment]

    # The read, and the post-write verification read, both under the lock.
    assert held_during_read and all(held_during_read)
    assert "capacity_kwh: 17.0" in cfg.read_text(encoding="utf-8")
    reflex.store.update_reflex_state.assert_awaited_once_with("battery.capacity_kwh", 17.0)
