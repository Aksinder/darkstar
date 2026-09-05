"""The tick must not fire on the interval boundary.

Firing on the boundary put every tick in a dead heat with everything else on this box
that is scheduled on whole minutes and whole hours — HA's hourly statistics compilation,
price-sensor refreshes, hourly template re-renders. 10162 of 10166 ticks began on second
:00, and the read timeouts clustered exactly there: 02:00 x26, 04:00 x18, 01:00 x12,
00:00 x8 over eight days, each one a batch of concurrent reads dying 5.000 s after the
tick started and succeeding on the retry 5 s later.

Nothing about the WORK changed; only its phase. So these tests pin two things: the offset
is applied, and the spacing between ticks is exactly what it was.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from executor.engine import ExecutorEngine

TZ = timezone.utc


def _engine(interval: int = 60, offset: float = 7.0) -> ExecutorEngine:
    eng = ExecutorEngine.__new__(ExecutorEngine)
    eng.config = SimpleNamespace(interval_seconds=interval, tick_offset_seconds=offset)
    return eng


class TestThePhaseOffset:
    def test_the_next_run_is_off_the_boundary(self):
        eng = _engine()
        nxt = eng._compute_next_run(datetime(2026, 9, 5, 1, 59, 55, tzinfo=TZ))
        assert nxt == datetime(2026, 9, 5, 2, 0, 7, tzinfo=TZ)

    def test_the_top_of_the_hour_is_specifically_avoided(self):
        """THE case: 02:00:00.010 is where four reads died together."""
        eng = _engine()
        nxt = eng._compute_next_run(datetime(2026, 9, 5, 1, 59, 59, 900000, tzinfo=TZ))
        assert (nxt.minute, nxt.second) != (0, 0)

    def test_spacing_is_unchanged(self):
        """Only the phase moves. Cadence is what everything downstream reasons about."""
        eng = _engine()
        now = datetime(2026, 9, 5, 3, 0, 0, tzinfo=TZ)
        seen = []
        for _ in range(10):
            now = eng._compute_next_run(now)
            seen.append(now)
        gaps = {(b - a).total_seconds() for a, b in zip(seen, seen[1:], strict=False)}
        assert gaps == {60.0}

    def test_it_never_returns_the_past(self):
        eng = _engine()
        for sec in range(0, 60):
            now = datetime(2026, 9, 5, 3, 10, sec, tzinfo=TZ)
            assert eng._compute_next_run(now) > now

    def test_the_slot_in_the_current_interval_is_not_skipped(self):
        """At :03 the :07 slot is still ahead — taking the next boundary would idle a
        whole interval and halve the control rate."""
        eng = _engine()
        now = datetime(2026, 9, 5, 3, 10, 3, tzinfo=TZ)
        assert eng._compute_next_run(now) == datetime(2026, 9, 5, 3, 10, 7, tzinfo=TZ)

    def test_zero_offset_restores_boundary_alignment(self):
        eng = _engine(offset=0.0)
        nxt = eng._compute_next_run(datetime(2026, 9, 5, 1, 59, 55, tzinfo=TZ))
        assert nxt == datetime(2026, 9, 5, 2, 0, 0, tzinfo=TZ)

    def test_an_offset_beyond_the_interval_cannot_stall_the_loop(self):
        """A mis-set offset must degrade to a phase shift, never to a stalled executor."""
        eng = _engine(interval=60, offset=185.0)
        now = datetime(2026, 9, 5, 3, 10, 0, tzinfo=TZ)
        nxt = eng._compute_next_run(now)
        assert now < nxt <= now + timedelta(seconds=60)

    def test_a_missing_offset_attribute_is_tolerated(self):
        """Older config objects (and half-built test engines) must still schedule."""
        eng = ExecutorEngine.__new__(ExecutorEngine)
        eng.config = SimpleNamespace(interval_seconds=60)
        now = datetime(2026, 9, 5, 3, 10, 30, tzinfo=TZ)
        assert eng._compute_next_run(now) == datetime(2026, 9, 5, 3, 11, 0, tzinfo=TZ)

    def test_a_five_minute_interval_keeps_its_cadence(self):
        eng = _engine(interval=300, offset=7.0)
        now = datetime(2026, 9, 5, 3, 2, 0, tzinfo=TZ)
        nxt = eng._compute_next_run(now)
        assert nxt == datetime(2026, 9, 5, 3, 5, 7, tzinfo=TZ)
        assert (eng._compute_next_run(nxt) - nxt).total_seconds() == 300.0
