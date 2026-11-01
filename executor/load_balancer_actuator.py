"""
The ONLY place the per-phase load balancer may touch a device — and only when
``load_balancing.actuate: true`` (default false = observe-only).

Deliberately narrow write paths:

* **EV current**: only the Easee ``easee.set_charger_dynamic_limit`` service
  (volatile/RAM, ``time_to_live: 0``). Never ``set_charger_max_limit``,
  ``set_circuit_*`` or any other persistent setting, and never the Easee
  "enabled" switch (a persistent write path): an Easee pause is dynamic limit 0.
* **Other chargers** (e.g. a Tesla via ``number.*_charge_current``): no amp
  writes at all. A balancer pause turns the charger's existing HA
  ``switch_entity`` off; a reduction above the floor is logged, not applied.
* **Shed loads**: only a ``loads[].entity`` that is an existing ``switch.*`` /
  ``input_boolean.*`` — explicit per-load opt-in. A load without one is
  observe-only even with actuate on.
* **Battery**: never actuated here; the S4 fuse_guard battery clamp owns it.

The balancer is a CAP on the ev_surplus servo, which keeps owning the charger:
this module remembers the cap it imposed, releases it (writes the servo's own
command back / switches back on) as soon as the balancer's target reaches the
servo's, and forgets it whenever the servo itself turns the charger off.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BalancedLoadConfig
    from .load_balancer import LoadBalancerStatus

logger = logging.getLogger(__name__)

_SWITCH_DOMAINS = ("switch.", "input_boolean.")


class LoadBalancerActuator:
    """Applies balancer decisions through the allowed write paths only."""

    def __init__(self) -> None:
        # charger id -> amps the balancer currently imposes (0 = paused by us).
        self._ev_cap: dict[str, int] = {}
        # charger id -> how the cap was applied ("easee" | "switch").
        self._ev_path: dict[str, str] = {}
        # charger id -> the servo command at the time the cap was written. When
        # the servo's command moves, the servo has rewritten the register itself
        # (its own write path), so our cap is no longer in force.
        self._ev_servo_at_cap: dict[str, float] = {}
        # shed load ids we switched off.
        self._shed_off: set[str] = set()

    def applied_setpoint_a(self, charger_id: str, servo_a: float | None) -> float | None:
        """What is really applied: the servo's command, lowered by our cap."""
        if servo_a is None or servo_a <= 0:
            return None
        cap = self._cap_in_force(charger_id, servo_a)
        if cap is None:
            return servo_a
        return None if cap <= 0 else min(float(cap), servo_a)

    def _cap_in_force(self, charger_id: str, servo_a: float) -> int | None:
        cap = self._ev_cap.get(charger_id)
        if cap is None:
            return None
        if self._ev_path.get(charger_id) == "easee" and self._ev_servo_at_cap.get(
            charger_id
        ) != float(servo_a):
            return None  # the servo rewrote the dynamic limit since we capped it
        return cap

    def _drop(self, charger_id: str) -> None:
        self._ev_cap.pop(charger_id, None)
        self._ev_path.pop(charger_id, None)
        self._ev_servo_at_cap.pop(charger_id, None)

    def forget(self) -> None:
        """Drop every cap (actuate switched off / balancer disabled)."""
        self._ev_cap.clear()
        self._ev_path.clear()
        self._ev_servo_at_cap.clear()
        self._shed_off.clear()

    async def apply(
        self,
        ha: Any,
        status: LoadBalancerStatus,
        *,
        chargers: dict[str, Any],
        servo_a: dict[str, float | None],
        loads: dict[str, BalancedLoadConfig],
        shadow: bool,
    ) -> list[str]:
        """Apply one tick's decisions; returns human-readable actions taken."""
        done: list[str] = []
        for out in status.ev_outputs:
            ccfg = chargers.get(out.charger_id)
            if ccfg is None:
                continue
            msg = await self._apply_ev(ha, ccfg, out.target_a, servo_a.get(out.charger_id), shadow)
            if msg:
                done.append(msg)
        for shed in status.shed_outputs:
            msg = await self._apply_shed(ha, loads.get(shed.load_id), shed, shadow)
            if msg:
                done.append(msg)
        return done

    async def _write_easee(self, ha: Any, device_id: str, amps: int, shadow: bool) -> None:
        if shadow:
            return
        await ha.call_service(
            "easee",
            "set_charger_dynamic_limit",
            None,
            {"device_id": device_id, "current": int(amps), "time_to_live": 0},
        )

    async def _write_switch(self, ha: Any, entity: str, on: bool, shadow: bool) -> None:
        if shadow:
            return
        domain = entity.split(".", 1)[0]
        await ha.call_service(domain, "turn_on" if on else "turn_off", entity)

    async def _apply_ev(
        self,
        ha: Any,
        ccfg: Any,
        target_a: int | None,
        servo: float | None,
        shadow: bool,
    ) -> str | None:
        cid = ccfg.id
        easee = getattr(ccfg, "easee_device_id", None)
        switch = getattr(ccfg, "switch_entity", None)
        tag = " (shadow, not sent)" if shadow else ""

        if servo is None or servo <= 0:
            # The servo has the charger off itself: nothing to cap, nothing to release.
            self._drop(cid)
            return None

        held = self._cap_in_force(cid, servo)
        if held is None:
            self._drop(cid)
        if target_a is not None and target_a >= servo:
            if held is None:
                return None
            # Release: hand the charger back to the servo's own command.
            path = self._ev_path.get(cid, "easee")
            self._drop(cid)
            if path == "easee" and easee:
                await self._write_easee(ha, easee, round(servo), shadow)
                return f"{cid}: released cap -> servo {round(servo)}A (Easee dynamic){tag}"
            if path == "switch" and switch:
                await self._write_switch(ha, switch, True, shadow)
                return f"{cid}: released pause -> {switch} on{tag}"
            return None

        want = 0 if target_a is None else int(target_a)
        if held == want:
            return None
        if easee:
            await self._write_easee(ha, easee, want, shadow)
            self._ev_cap[cid] = want
            self._ev_path[cid] = "easee"
            self._ev_servo_at_cap[cid] = float(servo)
            verb = "paused" if want == 0 else f"capped {round(servo)}->{want}A"
            return f"{cid}: {verb} (Easee dynamic limit){tag}"
        if want == 0 and switch and switch.startswith(_SWITCH_DOMAINS):
            await self._write_switch(ha, switch, False, shadow)
            self._ev_cap[cid] = 0
            self._ev_path[cid] = "switch"
            return f"{cid}: paused via {switch}{tag}"
        logger.info(
            "Load balancer: %s would go to %sA but has no allowed write path "
            "(only Easee dynamic current, or a switch for a pause) — not applied",
            cid,
            want,
        )
        return None

    async def _apply_shed(
        self, ha: Any, load: BalancedLoadConfig | None, shed: Any, shadow: bool
    ) -> str | None:
        if load is None:
            return None
        entity = getattr(load, "entity", None)
        if not entity or not str(entity).startswith(_SWITCH_DOMAINS):
            return None
        tag = " (shadow, not sent)" if shadow else ""
        if shed.actively_shedding and load.device_id not in self._shed_off:
            await self._write_switch(ha, entity, False, shadow)
            self._shed_off.add(load.device_id)
            return f"{load.device_id}: shed via {entity} off{tag}"
        if not shed.shed and load.device_id in self._shed_off:
            # Restore only what WE switched off.
            await self._write_switch(ha, entity, True, shadow)
            self._shed_off.discard(load.device_id)
            return f"{load.device_id}: restored via {entity} on{tag}"
        return None
