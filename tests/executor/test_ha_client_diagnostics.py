"""Two failures that were invisible: what timed out, and where the tick's time went.

Over eight days the executor logged 265 warnings reading, in full:

    HA API call failed (attempt 1/4): . Retrying in 1.0s...

str(asyncio.TimeoutError()) is the empty string, and a timeout carries no URL, so the
line named neither the entity nor the kind of failure. Meanwhile 10166 SLOW TICK
warnings reported a duration and nothing else — enough to know ticks were slow (median
7.1 s), useless for deciding whether the cost was HA round-trips or our own compute.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

from executor.actions import HAClient, HACallError, _retry_with_backoff


class TestTheWarningNamesTheFailure:
    @pytest.mark.asyncio
    async def test_a_timeout_is_no_longer_an_empty_string(self, caplog):
        async def boom():
            raise TimeoutError()

        with caplog.at_level(logging.WARNING), pytest.raises(HACallError):
            await _retry_with_backoff(
                boom, max_retries=1, base_delay=0.0, what="HA read sensor.solpaneler"
            )
        line = caplog.text
        assert "TimeoutError" in line
        assert "sensor.solpaneler" in line
        assert "(no detail)" in line

    @pytest.mark.asyncio
    async def test_an_error_with_text_keeps_its_text(self, caplog):
        async def boom():
            raise aiohttp.ClientConnectionError("connection reset")

        with caplog.at_level(logging.WARNING), pytest.raises(HACallError):
            await _retry_with_backoff(boom, max_retries=1, base_delay=0.0, what="HA read x")
        assert "connection reset" in caplog.text
        assert "(no detail)" not in caplog.text

    @pytest.mark.asyncio
    async def test_the_final_error_names_the_target_too(self):
        async def boom():
            raise TimeoutError()

        with pytest.raises(HACallError) as exc:
            await _retry_with_backoff(
                boom, max_retries=0, base_delay=0.0, what="HA switch.turn_on on switch.vvb"
            )
        assert "switch.vvb" in str(exc.value)

    @pytest.mark.asyncio
    async def test_the_default_description_still_works(self, caplog):
        """Callers that pass nothing must not start logging 'None failed'."""

        async def boom():
            raise TimeoutError()

        with caplog.at_level(logging.WARNING), pytest.raises(HACallError):
            await _retry_with_backoff(boom, max_retries=1, base_delay=0.0)
        assert "HA API call failed" in caplog.text


class TestReadsWaitLongerThanWrites:
    def test_the_read_deadline_is_longer(self):
        """A read that is merely slow is still a correct read; a stale write is not."""
        c = HAClient("http://supervisor/core", "tok")
        assert c.read_timeout.total > c.timeout.total

    def test_both_deadlines_are_configurable(self):
        c = HAClient("http://supervisor/core", "tok", timeout=3, read_timeout=30.0)
        assert c.timeout.total == 3
        assert c.read_timeout.total == 30.0


class TestHttpAccounting:
    def test_it_starts_at_zero(self):
        c = HAClient("http://supervisor/core", "tok")
        assert c.http_stats() == (0, 0.0)

    def test_reset_clears_a_previous_tick(self):
        c = HAClient("http://supervisor/core", "tok")
        c._http_calls, c._http_seconds = 61, 6.4
        c.http_stats_reset()
        assert c.http_stats() == (0, 0.0)

    @pytest.mark.asyncio
    async def test_a_failed_call_is_counted_too(self, monkeypatch):
        """Time spent waiting on a call that times out is exactly the time we most need
        accounted for — counting only successes would hide the problem."""
        c = HAClient("http://supervisor/core", "tok")

        class Boom:
            def get(self, *a, **kw):
                raise TimeoutError()

        async def fake_session():
            return Boom()

        monkeypatch.setattr(c, "_get_session", fake_session)
        assert await c.get_state("sensor.x") is None
        calls, seconds = c.http_stats()
        assert calls == 3, "three attempts, each one counted"
        assert seconds >= 0.0
