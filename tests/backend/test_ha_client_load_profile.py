"""get_load_profile_from_ha must count meter intervals that cross local midnight.

Ported from upstream 04b978dd. The old bucketing walked `range(start_slot,
end_slot + 1)` within one day, so an interval from 23:50 to 00:05 had
start_slot=95 > end_slot=0 and contributed NOTHING — every night's last reading
was dropped — and a gap longer than a day was credited to a single day's slots.

Upstream's test module also covers a per-delta sanity guard and degraded-status
messaging that this fork does not carry; only the slot-distribution tests apply
here, run against the fork's own httpx call, plus direct tests of the pure
helper.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from backend.core.ha_client import _distribute_interval_energy, get_load_profile_from_ha

STOCKHOLM = pytz.timezone("Europe/Stockholm")

# Pinned "current time" for get_load_profile_from_ha. With a real clock the local
# time of day of each synthetic sample shifts per run, and so would the slots.
FIXED_NOW = datetime(2026, 9, 20, 12, 0, tzinfo=pytz.UTC)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


@pytest.fixture(autouse=True)
def _frozen_clock():
    with patch("backend.core.ha_client.datetime", _FrozenDatetime):
        yield


def _fresh_start() -> datetime:
    """A timestamp safely inside the trailing-7-day query window."""
    return FIXED_NOW - timedelta(days=7) + timedelta(minutes=20)


def _slot_of(ts: datetime) -> int:
    local = ts.astimezone(STOCKHOLM)
    return (local.hour * 60 + local.minute) // 15


def _state(value: str, ts: datetime, unit: str = "kWh") -> dict:
    return {
        "state": value,
        "last_changed": ts.isoformat(),
        "attributes": {"unit_of_measurement": unit},
    }


def _clean_lifetime_states(days: int = 6, daily_kwh: float = 10.0) -> list[dict]:
    """Plausible cumulative-meter history: steady small per-slot increments."""
    start = _fresh_start()
    per_step = daily_kwh / 96
    lifetime = 19600.0
    states = []
    for day in range(days):
        for step in range(96):
            t = start + timedelta(days=day, minutes=15 * step)
            lifetime += per_step
            states.append(_state(f"{lifetime:.4f}", t))
    return states


async def _run_get_load_profile(states: list[dict]) -> list[float]:
    response = MagicMock()
    response.json.return_value = [states]
    response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)

    cfg = {
        "timezone": "Europe/Stockholm",
        "input_sensors": {"total_load_consumption": "sensor.lifetime_kwh"},
    }
    with (
        patch("backend.core.ha_client.httpx.AsyncClient", return_value=client_cm),
        patch(
            "backend.core.secrets.load_home_assistant_config",
            return_value={"url": "http://homeassistant:8123", "token": "test_token"},
        ),
    ):
        return await get_load_profile_from_ha(cfg)


class TestDistributeIntervalEnergy:
    WINDOW = (FIXED_NOW - timedelta(days=7), FIXED_NOW)

    def _spread(self, start, end, kwh):
        sums = [0.0] * 96
        _distribute_interval_energy(sums, start, end, kwh, STOCKHOLM, *self.WINDOW)
        return sums

    def test_interval_inside_one_slot(self):
        t0 = STOCKHOLM.localize(datetime(2026, 9, 15, 10, 2))
        sums = self._spread(t0, t0 + timedelta(minutes=10), 1.0)
        assert sums[40] == pytest.approx(1.0) and sum(sums) == pytest.approx(1.0)

    def test_interval_crossing_midnight_wraps_to_slot_zero(self):
        t0 = STOCKHOLM.localize(datetime(2026, 9, 15, 23, 50))
        sums = self._spread(t0, t0 + timedelta(minutes=15), 3.0)
        assert sums[95] == pytest.approx(2.0)
        assert sums[0] == pytest.approx(1.0)
        assert sum(sums) == pytest.approx(3.0)

    def test_only_the_part_inside_the_window_is_recorded(self):
        """An interval that starts before the window keeps its rate; the part
        outside is not credited, so the 7-day denominator stays honest."""
        start = self.WINDOW[0] - timedelta(hours=1)
        sums = self._spread(start, self.WINDOW[0] + timedelta(hours=1), 8.0)
        assert sum(sums) == pytest.approx(4.0)

    @pytest.mark.parametrize("kwh,minutes", [(0.0, 15), (-1.0, 15), (1.0, 0), (1.0, -5)])
    def test_nothing_recorded_for_empty_or_backwards_intervals(self, kwh, minutes):
        t0 = STOCKHOLM.localize(datetime(2026, 9, 15, 10, 0))
        assert sum(self._spread(t0, t0 + timedelta(minutes=minutes), kwh)) == 0.0


class TestSlotDistribution:
    @pytest.mark.asyncio
    async def test_jump_in_interval_crossing_local_midnight_is_fully_counted(self):
        """An interval spanning local midnight must wrap into slots 95 and 0,
        not be silently dropped."""
        baseline_profile = await _run_get_load_profile(_clean_lifetime_states())

        states = _clean_lifetime_states()
        times = [datetime.fromisoformat(s["last_changed"]) for s in states]
        # First interval whose readings sit on either side of local midnight.
        idx = next(
            i
            for i in range(1, len(times))
            if times[i].astimezone(STOCKHOLM).date() != times[i - 1].astimezone(STOCKHOLM).date()
        )
        before = times[idx - 1].astimezone(STOCKHOLM)
        after = times[idx].astimezone(STOCKHOLM)
        assert (before.hour, before.minute) == (23, 50)
        assert (after.hour, after.minute) == (0, 5)

        jump = 30.0
        for i in range(idx, len(states)):
            states[i] = _state(f"{float(states[i]['state']) + jump:.4f}", times[i])
        profile = await _run_get_load_profile(states)

        assert sum(profile) - sum(baseline_profile) == pytest.approx(jump / 7, rel=1e-3)
        # 23:50-00:00 is 10 of the 15 minutes -> slot 95; 00:00-00:05 -> slot 0.
        assert profile[95] - baseline_profile[95] == pytest.approx(jump * 10 / 15 / 7, rel=1e-3)
        assert profile[0] - baseline_profile[0] == pytest.approx(jump * 5 / 15 / 7, rel=1e-3)
        for slot in range(1, 95):
            assert profile[slot] == pytest.approx(baseline_profile[slot], abs=1e-6)

    @pytest.mark.asyncio
    async def test_interval_across_dst_fall_back_uses_wall_clock_slots(self):
        """On the 25 h day (2026-10-25, 03:00 CEST -> 02:00 CET) the repeated
        02:00-03:00 hour lands in its wall-clock slots twice and no energy is lost."""
        now = datetime(2026, 10, 27, 12, 0, tzinfo=pytz.UTC)

        class _DstNow(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return now if tz is None else now.astimezone(tz)

        t0 = datetime(2026, 10, 24, 23, 50, tzinfo=pytz.UTC)  # 01:50 CEST
        t1 = datetime(2026, 10, 25, 2, 5, tzinfo=pytz.UTC)  # 03:05 CET
        assert _slot_of(t0) == 7 and _slot_of(t1) == 12
        minutes = (t1 - t0).total_seconds() / 60  # 135 real minutes
        energy = 13.5
        states = [_state("100.0", t0), _state(f"{100.0 + energy}", t1)]

        with patch("backend.core.ha_client.datetime", _DstNow):
            profile = await _run_get_load_profile(states)

        per_min = energy / minutes / 7
        assert sum(profile) == pytest.approx(energy / 7, rel=1e-6)
        assert profile[7] == pytest.approx(10 * per_min, rel=1e-6)
        for slot in range(8, 12):
            assert profile[slot] == pytest.approx(30 * per_min, rel=1e-6)
        assert profile[12] == pytest.approx(5 * per_min, rel=1e-6)

    @pytest.mark.asyncio
    async def test_long_gap_spanning_days_is_spread_evenly(self):
        """A 2-day gap spreads its energy evenly over every slot instead of
        piling it onto the first day's."""
        start = _fresh_start()
        states = [_state("100.0", start), _state("148.0", start + timedelta(days=2))]
        profile = await _run_get_load_profile(states)
        assert sum(profile) == pytest.approx(48.0 / 7, rel=1e-6)
        for slot in range(96):
            assert profile[slot] == pytest.approx(48.0 / 96 / 7, rel=1e-6)
