"""Regression tests for the 2026-10-11 audit, batch 2: API gate, config writers, EV SoC hold."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from backend.config_migration import ConfigUnreadableError, update_config
from backend.middleware.lan_gate import LanGate, gate_enabled, peer_allowed
from executor.actions import is_notify_service
from tests.executor.test_blind_wake import T0, FakeHA, _ctrl

GOOD = "timezone: Europe/Stockholm\nsystem:\n  inverter_profile: sungrow\nexecutor:\n  enabled: true\n  shadow_mode: false\nui:\n  theme: dark\n"


# --- update_config ---------------------------------------------------------


def test_update_config_changes_one_key_and_keeps_the_rest(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(GOOD, encoding="utf-8")
    update_config(cfg, lambda c: c["executor"].__setitem__("enabled", False))
    text = cfg.read_text(encoding="utf-8")
    assert "enabled: false" in text
    assert "timezone: Europe/Stockholm" in text and "theme: dark" in text


@pytest.mark.parametrize("body", ["", "[just, a, list]\n", "executor:\n  enabled: true\n  enabled: [oops\n"])
def test_update_config_never_writes_from_a_failed_read(tmp_path: Path, body: str) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigUnreadableError):
        update_config(cfg, lambda c: c.setdefault("executor", {}).__setitem__("enabled", False))
    assert cfg.read_text(encoding="utf-8") == body, "the file must be left exactly as it was"


def test_update_config_writes_through_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "persist" / "config.yaml"
    real.parent.mkdir()
    real.write_text(GOOD, encoding="utf-8")
    link = tmp_path / "config.yaml"
    link.symlink_to(real)
    update_config(link, lambda c: c["ui"].__setitem__("theme", "light"))
    assert link.is_symlink(), "the link itself must survive"
    assert "theme: light" in real.read_text(encoding="utf-8")


def test_concurrent_updates_do_not_lose_each_other(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(GOOD, encoding="utf-8")

    def _set(i: int) -> None:
        update_config(cfg, lambda c: c.setdefault("probe", {}).__setitem__(f"k{i}", i))

    threads = [threading.Thread(target=_set, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    text = cfg.read_text(encoding="utf-8")
    assert all(f"k{i}: {i}" in text for i in range(8))


# --- API handlers ----------------------------------------------------------


def test_toggle_refuses_an_unreadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from backend.api.routers import executor as router

    monkeypatch.chdir(tmp_path)
    broken = "executor:\n  enabled: true\n  enabled: [oops\n"
    (tmp_path / "config.yaml").write_text(broken, encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.toggle_executor(router.ToggleRequest(enabled=False)))
    assert exc.value.status_code == 409
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == broken


class _Req:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def json(self) -> Any:
        return self._payload


@pytest.mark.parametrize(
    "payload",
    [
        {"service": "script.open_garage"},
        {"service": "shell_command.anything"},
        {"service": "notify.x/../../y"},
        {"something_else": True},
        {"on_error": "yes"},
    ],
)
def test_notifications_reject_non_notify_services_and_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    from fastapi import HTTPException

    from backend.api.routers import executor as router

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(GOOD, encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.update_notifications(_Req(payload)))  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == GOOD


def test_notifications_accept_the_real_thing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.api.routers import executor as router

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(router, "get_executor_instance", lambda: None)
    (tmp_path / "config.yaml").write_text(GOOD, encoding="utf-8")
    asyncio.run(
        router.update_notifications(_Req({"service": "notify.notify_robert", "on_error": True}))  # type: ignore[arg-type]
    )
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "service: notify.notify_robert" in text and "on_error: true" in text


def test_is_notify_service() -> None:
    assert is_notify_service("notify.mobile_app_phone")
    assert is_notify_service("persistent_notification.create")
    for bad in ("script.x", "notify.", "notify", "homeassistant.restart", "notify.a b", None, 5):
        assert not is_notify_service(bad)


# --- LAN gate --------------------------------------------------------------


def test_peer_allowed() -> None:
    assert peer_allowed("172.30.32.2") and peer_allowed("172.30.32.1")
    assert peer_allowed("127.0.0.1") and peer_allowed("::1") and peer_allowed("::ffff:172.30.33.5")
    for bad in ("192.168.2.50", "10.0.0.4", "172.30.34.1", "8.8.8.8", "", None, "nonsense"):
        assert not peer_allowed(bad)


def test_gate_only_under_supervisor_and_respects_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    assert gate_enabled({}) is False
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    assert gate_enabled({}) is True and gate_enabled(None) is True
    assert gate_enabled({"api": {"allow_lan": True}}) is False
    assert gate_enabled({"api": {"allow_lan": "yes"}}) is True


def _call(gate: LanGate, peer: str) -> tuple[int | None, bool]:
    sent: list[dict[str, Any]] = []
    reached = False

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        nonlocal reached
        reached = True

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    gate.app = inner
    asyncio.run(gate({"type": "http", "client": (peer, 1234), "path": "/api/config"}, None, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return status, reached


def test_gate_blocks_lan_and_passes_home_assistant() -> None:
    gate = LanGate(None, enabled=True)
    assert _call(gate, "192.168.2.77") == (403, False)
    assert _call(gate, "172.30.32.2") == (None, True)
    assert _call(LanGate(None, enabled=False), "192.168.2.77") == (None, True)


# --- EV servo SoC hold -----------------------------------------------------


def _read(ctrl: Any, ha: FakeHA, now: float) -> Any:
    return asyncio.run(ctrl._read_charger(ha, ctrl.cfg.chargers[0], now, False))


def test_servo_keeps_the_last_soc_through_a_blackout() -> None:
    ctrl = _ctrl()
    seen = {"binary_sensor.tesla_plug": "on", "sensor.tesla_soc": "13", "sensor.tesla_power": "0"}
    assert _read(ctrl, FakeHA(seen), T0).soc_percent == 13.0

    blank = {k: "unknown" for k in seen}
    st = _read(ctrl, FakeHA(blank), T0 + 3600)
    assert st.soc_percent == 13.0, "the floors need a number to stay armed"
    assert "tesla" in ctrl._soc_held

    st = _read(ctrl, FakeHA({**seen, "sensor.tesla_soc": "41"}), T0 + 7200)
    assert st.soc_percent == 41.0 and "tesla" not in ctrl._soc_held


def test_servo_soc_hold_expires_after_a_day() -> None:
    ctrl = _ctrl()
    seen = {"binary_sensor.tesla_plug": "on", "sensor.tesla_soc": "13", "sensor.tesla_power": "0"}
    _read(ctrl, FakeHA(seen), T0)
    st = _read(ctrl, FakeHA({k: "unknown" for k in seen}), T0 + 25 * 3600)
    assert st.soc_percent is None


def test_servo_soc_hold_survives_a_restart(tmp_path: Path) -> None:
    from backend.core import ev_hold_store

    ev_hold_store.enable(tmp_path / "ev_holds.json")
    try:
        seen = {"binary_sensor.tesla_plug": "on", "sensor.tesla_soc": "13", "sensor.tesla_power": "0"}
        _read(_ctrl(), FakeHA(seen), T0)
        fresh = _ctrl()  # a new process: memory gone, disk not
        st = _read(fresh, FakeHA({k: "unknown" for k in seen}), T0 + 1800)
        assert st.soc_percent == 13.0
    finally:
        ev_hold_store.disable()
