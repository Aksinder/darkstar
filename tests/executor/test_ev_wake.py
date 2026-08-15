"""Wake-on-actuation-failure: a sleeping Tesla answers switch.turn_on with HTTP 500.

Before this, a car that was home, plugged in and wanted by the servo simply never
started: every tick raised, the backoff grew, and nothing ever woke it. Pressing the
vendor's wake button once per failure window lets the ordinary backoff retry land on
an awake car.
"""

from __future__ import annotations

import pytest

from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import FakeHA, _cfg_dict, _states


class SleepyHA(FakeHA):
    """Raises on the Tesla switch (asleep) but accepts everything else."""

    def __init__(self, states, fail_entity="switch.tesla"):
        super().__init__(states)
        self.fail_entity = fail_entity

    async def call_service(self, domain, service, entity_id=None, data=None):
        if entity_id == self.fail_entity:
            self.calls.append((domain, service, entity_id, data))
            raise RuntimeError("500 Internal Server Error: vehicle unavailable")
        return await super().call_service(domain, service, entity_id, data)


def _wake_cfg(**over):
    raw = _cfg_dict(**over)
    for c in raw["ev_surplus"]["chargers"]:
        if c["id"] == "tesla":
            c["wake_entity"] = "button.white_betty_wake"
    return raw


def _surplus_states():
    # Enough export for BOTH cars at full tilt (the 1-phase FMB is served first and
    # takes 3.7 kW; the 3-phase Tesla needs 3.45 kW just to reach its 5 A minimum).
    return _states(**{"sensor.pv": "15000", "sensor.grid": "-12000", "sensor.soc": "98"})


def _controller(raw):
    cfg = parse_ev_surplus_config(raw)
    assert cfg is not None
    return EVSurplusController(cfg), cfg


def _wake_presses(ha):
    return [c for c in ha.calls if c[:3] == ("button", "press", "button.white_betty_wake")]


@pytest.mark.asyncio
async def test_parses_wake_entity():
    _, cfg = _controller(_wake_cfg())
    tesla = next(c for c in cfg.chargers if c.id == "tesla")
    assert tesla.wake_entity == "button.white_betty_wake"
    fmb = next(c for c in cfg.chargers if c.id == "easee_fmb")
    assert fmb.wake_entity is None


@pytest.mark.asyncio
async def test_failed_turn_on_presses_the_wake_button():
    ctrl, _ = _controller(_wake_cfg())
    ha = SleepyHA(_surplus_states())
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert len(_wake_presses(ha)) == 1


@pytest.mark.asyncio
async def test_wake_press_is_cooldown_limited():
    """Backoff spares the API; the wake press must not undo that."""
    ctrl, _ = _controller(_wake_cfg())
    ha = SleepyHA(_surplus_states())
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    # 200 s later the 120 s backoff has expired, but the 300 s wake cooldown has not.
    await ctrl.run(ha, now_ts=1200.0, shadow=False)
    assert len(_wake_presses(ha)) == 1
    # Past the cooldown, a still-failing car gets one more nudge.
    await ctrl.run(ha, now_ts=1600.0, shadow=False)
    assert len(_wake_presses(ha)) == 2


@pytest.mark.asyncio
async def test_no_wake_without_a_wake_entity():
    """Unconfigured chargers keep the old behaviour exactly."""
    ctrl, _ = _controller(_cfg_dict())
    ha = SleepyHA(_surplus_states())
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert not [c for c in ha.calls if c[0] == "button"]


@pytest.mark.asyncio
async def test_shadow_mode_never_presses():
    ctrl, _ = _controller(_wake_cfg())
    ha = SleepyHA(_surplus_states())
    await ctrl.run(ha, now_ts=1000.0, shadow=True)
    assert not _wake_presses(ha)


@pytest.mark.asyncio
async def test_failed_stop_does_not_wake():
    """A car we are trying to turn OFF is not drawing anyway — let it sleep."""
    ctrl, _ = _controller(_wake_cfg())
    # No surplus and no deadline pressure: the command is switch_on=False.
    ha = SleepyHA(_states(**{"sensor.tesla_soc": "95", "sensor.grid": "3000"}))
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert not _wake_presses(ha)
