"""Manual override for a water heater: auto / force_on / force_off.

The companion to may_skip_day. A load allowed to sit out a whole expensive PERIOD may
stay cold for days, which is the intent — so there has to be a way to say "anyway".
It acts on the EXECUTOR, so it lands on the next tick instead of waiting for a replan,
which is what a human means by "heat it now".
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from executor.engine import ExecutorEngine


class FakeHA:
    def __init__(self, value):
        self.value = value
        self.selects: list[tuple[str, str]] = []

    async def get_state_value(self, entity):
        return self.value

    async def set_select_option(self, entity, option):
        self.selects.append((entity, option))
        return True


def _engine() -> ExecutorEngine:
    """A bare engine: only the override helper and its memo are exercised."""
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._override_since = {}
    # Expiry now also releases the SELECTOR back to auto, rate-limited per heater.
    eng._override_reset_ts = {}
    eng.ha_client = None
    return eng


def _device(**kw):
    base = {
        "id": "spa",
        "override_entity": "input_select.spa_override",
        "override_timeout_minutes": 0.0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


async def _read(value, device=None, engine=None):
    eng = engine or _engine()
    eng.ha_client = FakeHA(value)
    return await eng._heater_override(device or _device())


class TestReading:
    @pytest.mark.asyncio
    async def test_force_on_and_force_off_pass_through(self):
        assert await _read("force_on") == "force_on"
        assert await _read("force_off") == "force_off"

    @pytest.mark.asyncio
    async def test_auto_is_auto(self):
        assert await _read("auto") == "auto"

    @pytest.mark.asyncio
    async def test_case_and_padding_are_tolerated(self):
        assert await _read("  Force_On ") == "force_on"

    @pytest.mark.asyncio
    async def test_no_entity_configured_is_auto(self):
        eng = _engine()
        eng.ha_client = FakeHA("force_on")
        assert await eng._heater_override(_device(override_entity=None)) == "auto"


class TestDegradesToAutoNeverToAStuckForce:
    @pytest.mark.asyncio
    async def test_an_unknown_option_is_auto(self):
        assert await _read("vacation") == "auto"

    @pytest.mark.asyncio
    async def test_unavailable_is_auto(self):
        assert await _read("unavailable") == "auto"

    @pytest.mark.asyncio
    async def test_none_is_auto(self):
        assert await _read(None) == "auto"


class TestExpiry:
    """A forgotten force_on cannot run away — the appliance's own thermostat caps the
    temperature — but it CAN quietly buy at peak for days."""

    @pytest.mark.asyncio
    async def test_zero_timeout_never_expires(self):
        eng = _engine()
        dev = _device(override_timeout_minutes=0.0)
        assert await _read("force_on", dev, eng) == "force_on"
        eng._override_since["spa"] = (time.time() - 86400, "force_on")
        assert await _read("force_on", dev, eng) == "force_on"

    @pytest.mark.asyncio
    async def test_it_expires_after_the_timeout(self):
        eng = _engine()
        dev = _device(override_timeout_minutes=60.0)
        assert await _read("force_on", dev, eng) == "force_on"
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        assert await _read("force_on", dev, eng) == "auto"

    @pytest.mark.asyncio
    async def test_it_holds_before_the_timeout(self):
        eng = _engine()
        dev = _device(override_timeout_minutes=60.0)
        await _read("force_on", dev, eng)
        eng._override_since["spa"] = (time.time() - 600, "force_on")
        assert await _read("force_on", dev, eng) == "force_on"

    @pytest.mark.asyncio
    async def test_switching_selection_restarts_the_clock(self):
        """force_on -> force_off is a NEW instruction, not a continuation."""
        eng = _engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        assert await _read("force_off", dev, eng) == "force_off"

    @pytest.mark.asyncio
    async def test_returning_to_auto_clears_the_clock(self):
        eng = _engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        assert await _read("auto", dev, eng) == "auto"
        assert "spa" not in eng._override_since


class TestExpiryReleasesTheSelector:
    """The expiry has always worked internally — it just never reached the helper.

    Live 2026-09-05: input_select.darkstar_ev_easee_mode had read force_on since
    2026-09-04 03:33 and Darkstar logged "expired after 180 min — back to auto" on every
    60 s tick, for 33 hours. The UI said force_on, the system ran auto, and nothing but
    that log line distinguished them.

    Worse than cosmetic: the clock is in-memory and starts when the value is first SEEN,
    so every add-on restart restarts it. Restart more often than the timeout and a
    forgotten force never expires at all, while the log insists it is counting down.
    """

    def _live_engine(self, *, shadow=False):
        eng = _engine()
        eng.config = SimpleNamespace(shadow_mode=shadow)
        return eng

    @pytest.mark.asyncio
    async def test_it_writes_auto_back_to_the_helper(self):
        eng = self._live_engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        ha = FakeHA("force_on")
        eng.ha_client = ha
        assert await eng._heater_override(dev) == "auto"
        assert ha.selects == [("input_select.spa_override", "auto")]

    @pytest.mark.asyncio
    async def test_it_does_not_write_before_the_timeout(self):
        eng = self._live_engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 600, "force_on")
        ha = FakeHA("force_on")
        eng.ha_client = ha
        assert await eng._heater_override(dev) == "force_on"
        assert ha.selects == []

    @pytest.mark.asyncio
    async def test_shadow_mode_never_writes(self):
        """Observe-only must not touch a user's helper, however stale it looks."""
        eng = self._live_engine(shadow=True)
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        ha = FakeHA("force_on")
        eng.ha_client = ha
        assert await eng._heater_override(dev) == "auto"
        assert ha.selects == []

    @pytest.mark.asyncio
    async def test_the_write_is_rate_limited_not_once_only(self):
        """One write per interval, not one per tick — but never give up either.

        A write that silently failed would otherwise latch the override for good, which
        is the exact failure this exists to prevent. So it retries, slowly.
        """
        eng = self._live_engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")
        ha = FakeHA("force_on")
        eng.ha_client = ha

        for _ in range(5):  # five consecutive ticks
            assert await eng._heater_override(dev) == "auto"
        assert len(ha.selects) == 1, "must not write on every tick"

        # ...and once the retry window has passed, it tries again.
        eng._override_reset_ts["spa"] = time.time() - 601
        assert await eng._heater_override(dev) == "auto"
        assert len(ha.selects) == 2

    @pytest.mark.asyncio
    async def test_a_failed_write_still_reports_auto(self):
        """The internal answer must not depend on the helper accepting the write."""
        eng = self._live_engine()
        dev = _device(override_timeout_minutes=60.0)
        eng._override_since["spa"] = (time.time() - 3601, "force_on")

        class Failing(FakeHA):
            async def set_select_option(self, entity, option):
                raise RuntimeError("HA said no")

        eng.ha_client = Failing("force_on")
        assert await eng._heater_override(dev) == "auto"
