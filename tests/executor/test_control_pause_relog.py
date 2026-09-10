"""A pause can outlive the memory of it.

The "control paused - leaving manual" line fired once per pause episode. That is fine for
a pause measured in hours and useless for one measured in days: by the time anyone asks
why a tank is cold, the only evidence has scrolled out of the log.

2026-09-09: the villavagn tank had been control-paused long enough to run dry — 0%,
11.7 C — and the single line that would have explained it was three days old. Diagnosing
it took a code read; it should have taken a grep.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from executor.engine import _CONTROL_PAUSE_RELOG_S, ExecutorEngine


def _engine(paused_via="input_boolean.pausa"):
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng._control_pause_logged = {}
    eng.dispatcher = SimpleNamespace(
        control_pause_entity=AsyncMock(return_value=paused_via)
    )
    return eng


async def _call(eng):
    return await eng._device_control_paused(["input_boolean.pausa"], "k", "villavagn_tank", {})


class TestItKeepsSaying:
    @pytest.mark.asyncio
    async def test_it_logs_the_first_time(self, caplog):
        eng = _engine()
        with caplog.at_level("INFO"):
            assert await _call(eng) is True
        assert "control paused" in caplog.text

    @pytest.mark.asyncio
    async def test_it_does_not_log_on_every_tick(self, caplog):
        eng = _engine()
        await _call(eng)          # the first-time line, deliberately outside the window
        caplog.clear()            # ...and not counted: caplog spans the whole test
        with caplog.at_level("INFO"):
            for _ in range(20):
                await _call(eng)
        assert "control paused" not in caplog.text

    @pytest.mark.asyncio
    async def test_it_logs_again_once_the_window_passes(self, caplog):
        eng = _engine()
        await _call(eng)
        caplog.clear()
        eng._control_pause_logged["k"] = time.time() - _CONTROL_PAUSE_RELOG_S - 1
        with caplog.at_level("INFO"):
            await _call(eng)
        assert "control paused" in caplog.text
        assert "still paused after" in caplog.text, "the repeat says how long it has stood"

    @pytest.mark.asyncio
    async def test_the_window_is_findable_but_quiet(self):
        """Frequent enough to grep for, rare enough to ignore."""
        assert 600.0 <= _CONTROL_PAUSE_RELOG_S <= 6 * 3600.0


class TestTheEpisodeStillEnds:
    @pytest.mark.asyncio
    async def test_unpausing_clears_the_memo(self):
        eng = _engine()
        await _call(eng)
        assert "k" in eng._control_pause_logged
        eng.dispatcher.control_pause_entity = AsyncMock(return_value=None)
        assert await _call(eng) is False
        assert "k" not in eng._control_pause_logged

    @pytest.mark.asyncio
    async def test_a_new_episode_logs_immediately(self, caplog):
        """Not silenced by the previous episode's timestamp."""
        eng = _engine()
        await _call(eng)
        eng.dispatcher.control_pause_entity = AsyncMock(return_value=None)
        await _call(eng)
        eng.dispatcher.control_pause_entity = AsyncMock(return_value="input_boolean.pausa")
        caplog.clear()
        with caplog.at_level("INFO"):
            await _call(eng)
        assert "control paused" in caplog.text

    @pytest.mark.asyncio
    async def test_no_dispatcher_is_not_paused(self):
        eng = _engine()
        eng.dispatcher = None
        assert await _call(eng) is False
