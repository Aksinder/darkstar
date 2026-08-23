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


# ---------------------------------------------------------------------------
# The failure ledger, after 2026-08-23.
# ---------------------------------------------------------------------------


def _wake_cfg_with_override():
    raw = _wake_cfg()
    for c in raw["ev_surplus"]["chargers"]:
        if c["id"] == "tesla":
            c["override_entity"] = "input_select.tesla_mode"
    return raw


@pytest.mark.asyncio
async def test_a_departure_resets_the_backoff():
    """Two failures from a visit three hours earlier met the returning car with
    'fail #3, 480 s' — eight idle minutes with sun to spare. A car that leaves
    takes its failure history with it."""
    ctrl, _ = _controller(_wake_cfg())
    ha = SleepyHA(_surplus_states())
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    await ctrl.run(ha, now_ts=1200.0, shadow=False)
    assert ctrl._act_fail["tesla"][1] >= 1
    # The car drives off: unplugged and away.
    gone = SleepyHA(_surplus_states())
    gone.states["binary_sensor.tesla_plug"] = "off"
    await ctrl.run(gone, now_ts=1300.0, shadow=False)
    assert "tesla" not in ctrl._act_fail
    assert "tesla" not in ctrl._last_wake_ts


@pytest.mark.asyncio
async def test_a_protective_command_retries_every_tick():
    """Reducing is the safe direction and the cost of not reducing is immediate —
    the home battery or the grid feeds the car meanwhile. Stops and reductions
    do not earn the escalating backoff; starts and increases still do."""
    ctrl, _ = _controller(_wake_cfg_with_override())
    # A car we believe is ON at 14 A that a human just forced OFF: a STOP.
    ctrl._last_a["tesla"] = 14.0
    ctrl._last_switch["tesla"] = True
    ha = SleepyHA(_states(**{"input_select.tesla_mode": "force_off", "sensor.grid": "3000",
                             "sensor.tesla_power": "9000"}))
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert ctrl._act_fail["tesla"][1] == 1
    n_calls = len([c for c in ha.calls if c[2] == "switch.tesla"])
    # 70 s later — inside what WOULD be a 120 s start backoff — the stop is retried.
    await ctrl.run(ha, now_ts=1070.0, shadow=False)
    assert len([c for c in ha.calls if c[2] == "switch.tesla"]) > n_calls


@pytest.mark.asyncio
async def test_a_failed_stop_on_a_drawing_car_wakes_it():
    """The old rule skipped the wake on every failed stop ('the car isn't drawing
    anyway') — false for a car that started itself, which is exactly when a stop
    matters most."""
    ctrl, _ = _controller(_wake_cfg_with_override())
    ctrl._last_a["tesla"] = 14.0
    ctrl._last_switch["tesla"] = True
    ha = SleepyHA(_states(**{"input_select.tesla_mode": "force_off", "sensor.grid": "3000",
                             "sensor.tesla_power": "9000"}))
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert len(_wake_presses(ha)) == 1


@pytest.mark.asyncio
async def test_a_failed_stop_on_an_idle_car_does_not_wake(caplog):
    """And the old rule's actual case still holds: nothing is drawing, let it sleep."""
    ctrl, _ = _controller(_wake_cfg_with_override())
    ctrl._last_a["tesla"] = 14.0
    ctrl._last_switch["tesla"] = True
    ha = SleepyHA(_states(**{"input_select.tesla_mode": "force_off", "sensor.grid": "3000",
                             "sensor.tesla_power": "0"}))
    await ctrl.run(ha, now_ts=1000.0, shadow=False)
    assert not _wake_presses(ha)
