"""``applied`` must mean ISSUED, never merely INTENDED.

The defect this file locks out, in full: ``_actuate`` used to return None and the run
loop appended ``cmd.set_current_a`` to ``applied`` regardless of whether any service
call had been made. ``_actuate`` has six early returns, four of them silent. So the
summary line

    EV surplus: ... -> [('tesla', True, 16.0)]

could repeat unchanged for 41 consecutive ticks off a SINGLE write — and it did, on the
night of 2026-08-31. An HA automation (``darkstar_ev_fuse_watchdog``) clamped the car to
5 A at 02:20:00 on a false liveness verdict; ``should_write_current`` then compared the
servo's own stale intent against itself (|16.0 - 16.0| = 0 < min_step_a) and suppressed
every write for the next 41 minutes. Reading that log as evidence of delivery produced a
wrong root cause and cost an investigation an hour.

The tests below assert the two halves of the fix separately: every suppression path
names itself, and the fail-safe stop — the one path where the intent/delivery gap is
safety-relevant — says out loud when it sent nothing.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from executor.ev_surplus_runtime import ActuationResult, EVSurplusController
from tests.executor.test_ev_surplus_antihunt import FakeHA, _runtime_cfg


def _ctrl(**over):
    cfg = _runtime_cfg(**over)
    assert cfg is not None
    return EVSurplusController(cfg), cfg


class _Cmd:
    id = "tesla"
    switch_on = True
    set_current_a = 16.0
    raw_amps = 16.0
    reason = "on 16.0A (deadline)"
    fuse_limited = False


class TestEverySuppressionNamesItself:
    """Each of the four formerly-silent returns must be distinguishable."""

    def test_a_real_write_reports_the_amps(self):
        ctrl, cfg = _ctrl()
        ctrl._last_switch["tesla"] = True
        act = asyncio.run(ctrl._actuate(FakeHA({}), cfg.chargers[0], _Cmd(), 10_000.0, False))
        assert act.a_written == 16.0
        assert act.suppressed is None

    def test_guard_suppression_is_named_and_writes_nothing(self):
        # The exact 2026-08-31 latch: last_a already equals the wanted value.
        ctrl, cfg = _ctrl()
        ctrl._last_a["tesla"] = 16.0
        ctrl._last_ts["tesla"] = 10_000.0
        ctrl._last_switch["tesla"] = True
        ha = FakeHA({})
        act = asyncio.run(ctrl._actuate(ha, cfg.chargers[0], _Cmd(), 10_060.0, False))
        assert act.a_written is None, "a suppressed write must never report amps"
        assert act.suppressed == "guard"
        assert ha.calls == []

    def test_schmitt_suppression_is_named(self):
        ctrl, cfg = _ctrl()
        ctrl._last_a["tesla"] = 14.0
        ctrl._last_ts["tesla"] = 0.0
        ctrl._last_switch["tesla"] = True

        class Dither(_Cmd):
            set_current_a = 15.0
            raw_amps = 14.55

        act = asyncio.run(ctrl._actuate(FakeHA({}), cfg.chargers[0], Dither(), 10_000.0, False))
        assert act.a_written is None
        assert act.suppressed == "schmitt"

    def test_no_current_write_is_named(self):
        ctrl, cfg = _ctrl()
        ctrl._last_switch["tesla"] = False

        class Off(_Cmd):
            switch_on = False
            set_current_a = None
            raw_amps = None

        act = asyncio.run(ctrl._actuate(FakeHA({}), cfg.chargers[0], Off(), 10_000.0, False))
        assert act.a_written is None
        assert act.suppressed == "no_current_write"

    def test_shadow_never_reports_a_written(self):
        """Shadow keeps the whole decision path but sends nothing — reporting amps
        here would make an observe-only rollout read exactly like a live one."""
        ctrl, cfg = _ctrl()
        ctrl._last_switch["tesla"] = True
        ha = FakeHA({})
        act = asyncio.run(ctrl._actuate(ha, cfg.chargers[0], _Cmd(), 10_000.0, True))
        assert act.shadow is True
        assert act.a_written is None
        assert ha.calls == []


class TestFailSafeStopAnnouncesDelivery:
    """The fail-safe is where the intent/delivery gap is safety-relevant.

    A stop no-ops whenever _last_switch already says False — the same stale-intent
    state that let a foreign writer clamp the Tesla unnoticed. A fail-safe that quietly
    sent nothing must not read like one that stopped the car.
    """

    def test_it_shouts_when_nothing_was_sent(self, caplog):
        ctrl, _ = _ctrl()
        ctrl._last_switch["tesla"] = False  # servo already believes it is off
        ctrl._last_a["tesla"] = 0.0
        ctrl._last_ts["tesla"] = 10_000.0
        with caplog.at_level(logging.WARNING):
            asyncio.run(ctrl._failsafe_stop_all(FakeHA({}), 10_060.0, "sensors stale", False))
        assert "NOTHING WAS SENT" in caplog.text
        assert "stale" in caplog.text.lower()

    def test_shadow_mode_says_so_rather_than_claiming_a_stop(self, caplog):
        ctrl, _ = _ctrl()
        ctrl._last_switch["tesla"] = True
        with caplog.at_level(logging.WARNING):
            asyncio.run(ctrl._failsafe_stop_all(FakeHA({}), 10_000.0, "test", True))
        assert "shadow mode" in caplog.text
        assert "NOTHING WAS SENT" not in caplog.text, (
            "shadow is a deliberate no-send, not a stale-belief no-send — the two must "
            "not collapse into one message"
        )

    def test_a_real_stop_is_not_reported_as_a_failure(self, caplog):
        ctrl, _ = _ctrl()
        ctrl._last_switch["tesla"] = True
        ctrl._last_a["tesla"] = 16.0
        ctrl._last_ts["tesla"] = 0.0
        with caplog.at_level(logging.INFO):
            asyncio.run(ctrl._failsafe_stop_all(FakeHA({}), 10_000.0, "test", False))
        assert "NOTHING WAS SENT" not in caplog.text
        assert "stopped" in caplog.text

    def test_an_exception_still_does_not_claim_delivery(self, caplog):
        class Boom(FakeHA):
            async def call_service(self, *a, **k):
                raise RuntimeError("vehicle asleep")

        ctrl, _ = _ctrl()
        ctrl._last_switch["tesla"] = True
        with caplog.at_level(logging.INFO):
            asyncio.run(ctrl._failsafe_stop_all(Boom({}), 10_000.0, "test", False))
        assert "fail-safe stop failed" in caplog.text
        assert "stopped (switch=" not in caplog.text, (
            "the else-branch must not run when _actuate raised"
        )


class TestSummaryLineCannotLie:
    def test_applied_carries_written_and_suppressed(self):
        ctrl, _ = _ctrl()
        ha = FakeHA({
            "sensor.grid": "-9000", "sensor.batt": "0", "sensor.soc": "90",
            "sensor.price": "1.0", "binary_sensor.tesla_plug": "on",
            "sensor.tesla_power": "0",
        })
        result = asyncio.run(ctrl.run(ha, now_ts=10_000.0))
        for a in result.get("applied", []):
            assert "a_written" in a, "every applied row must state what was written"
            assert "suppressed" in a
            if a["a_written"] is None and a["on"]:
                assert a["suppressed"], "a non-write must always name its reason"

    def test_intent_is_retained_alongside_delivery(self):
        """`a` stays the intent — dropping it would lose the servo's decision, which
        is the other half of what makes the log readable."""
        act = ActuationResult(a_written=None, suppressed="guard")
        assert act.a_written is None and act.suppressed == "guard"
        assert act.switch_written is None and act.shadow is False


@pytest.mark.parametrize("field", ["a_written", "switch_written", "suppressed"])
def test_defaults_are_none_not_zero(field):
    """A 0.0 default would read as 'wrote 0 A' — the same lie in a new costume."""
    assert getattr(ActuationResult(), field) is None
