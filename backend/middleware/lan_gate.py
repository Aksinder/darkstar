"""Refuse callers that are not Home Assistant when running as an add-on.

The add-on publishes port 5000 on the host, and nothing in the API authenticates:
every device on the LAN could force the battery to export, rewrite config.yaml or
make the add-on call HA services with its own token. Home Assistant's ingress
(the sidebar panel) already authenticates the user, and it reaches the add-on
from the Supervisor's address on the internal ``hassio`` Docker network.

So under the Supervisor only these peers get through:

* 172.30.32.0/23 — the hassio network: the ingress proxy (172.30.32.2) and
  Home Assistant Core itself (the gateway, 172.30.32.1), which keeps REST
  sensors and ``rest_command`` calls from HA working;
* loopback — health checks and tools inside the container.

Outside the Supervisor (docker-compose, local development) the gate is a no-op.
``api.allow_lan: true`` in config.yaml turns it off for people who knowingly
want the port open.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import Any, cast

logger = logging.getLogger("darkstar.lan_gate")

_ALLOWED = (
    ipaddress.ip_network("172.30.32.0/23"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
)

_BODY = (
    b"Darkstar: direct access to port 5000 is disabled. Open Darkstar from the "
    b"Home Assistant sidebar instead, or set api.allow_lan: true in config.yaml "
    b"and restart the add-on to allow LAN access.\n"
)


def peer_allowed(host: str | None) -> bool:
    """True if ``host`` is Home Assistant (hassio network) or loopback."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return any(addr in net for net in _ALLOWED if net.version == addr.version)


def gate_enabled(config: dict[str, Any] | None) -> bool:
    """The gate applies under the Supervisor unless the owner opted out."""
    if not os.environ.get("SUPERVISOR_TOKEN"):
        return False
    api_cfg: Any = (config or {}).get("api")
    if not isinstance(api_cfg, dict):
        return True
    return cast("dict[str, Any]", api_cfg).get("allow_lan") is not True


class LanGate:
    """Outermost ASGI wrapper, so Socket.IO is covered as well as the API."""

    def __init__(self, app: Any, *, enabled: bool) -> None:
        self.app = app
        self.enabled = enabled
        self._logged: set[str] = set()

    @property
    def other_asgi_app(self) -> Any:
        """The FastAPI app behind Socket.IO (tests and tools unwrap to it)."""
        return getattr(self.app, "other_asgi_app", self.app)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not self.enabled or scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        host = client[0] if client else None
        if peer_allowed(host):
            await self.app(scope, receive, send)
            return

        key = str(host)
        if key not in self._logged and len(self._logged) < 50:
            self._logged.add(key)
            logger.warning("Refused direct access from %s (%s)", host, scope.get("path"))

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": _BODY})
