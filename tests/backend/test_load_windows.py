"""Calendar windows for ``mode: window`` cyclic loads (backend/core/load_windows.py).

Nothing read HA calendars before the room heaters. The traps are the usual ones in
this codebase — local time against UTC, DST days that are 23 or 25 hours long — plus
the calendar's own: all-day events with an exclusive end date, overlapping events,
and an API that can answer with something that is not a list at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
import pytz

from backend.core import load_windows as lw

TZ = pytz.timezone("Europe/Stockholm")


def _timed(start: str, end: str, summary: str = "Kim") -> dict[str, Any]:
    return {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}


def _all_day(start: str, end: str, summary: str = "Kim") -> dict[str, Any]:
    return {"summary": summary, "start": {"date": start}, "end": {"date": end}}


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


class TestParsing:
    def test_timed_event_keeps_its_offset(self):
        (win,) = lw.parse_calendar_events(
            [_timed("2026-11-13T15:00:00+01:00", "2026-11-20T08:00:00+01:00")], TZ
        )
        assert win == (_utc(2026, 11, 13, 14), _utc(2026, 11, 20, 7))
        assert win[0].tzinfo is not None and win[1].tzinfo is not None

    def test_utc_z_suffix_and_naive_local_are_both_understood(self):
        z, naive = lw.parse_calendar_events(
            [
                _timed("2026-11-13T14:00:00Z", "2026-11-13T15:00:00Z"),
                _timed("2026-11-14T15:00:00", "2026-11-14T16:00:00"),
            ],
            TZ,
        )
        assert z == (_utc(2026, 11, 13, 14), _utc(2026, 11, 13, 15))
        # A bare time can only be the site's own: 15:00 CET = 14:00 UTC.
        assert naive == (_utc(2026, 11, 14, 14), _utc(2026, 11, 14, 15))

    def test_all_day_event_runs_local_midnight_to_local_midnight(self):
        """end.date is EXCLUSIVE: a one-day event on the 13th ends as the 14th begins."""
        (win,) = lw.parse_calendar_events([_all_day("2026-11-13", "2026-11-14")], TZ)
        assert win == (_utc(2026, 11, 12, 23), _utc(2026, 11, 13, 23))

    def test_all_day_event_on_the_autumn_dst_day_is_25_hours(self):
        (win,) = lw.parse_calendar_events([_all_day("2026-10-25", "2026-10-26")], TZ)
        assert win == (_utc(2026, 10, 24, 22), _utc(2026, 10, 25, 23))
        assert win[1] - win[0] == timedelta(hours=25)

    def test_all_day_event_on_the_spring_dst_day_is_23_hours(self):
        (win,) = lw.parse_calendar_events([_all_day("2026-03-29", "2026-03-30")], TZ)
        assert win[1] - win[0] == timedelta(hours=23)

    def test_timed_event_across_the_dst_change_keeps_real_instants(self):
        """Friday 15:00 CEST to Friday 08:00 CET: 7 days plus the extra hour, less
        the 7 h between 15:00 and 08:00."""
        (win,) = lw.parse_calendar_events(
            [_timed("2026-10-23T15:00:00+02:00", "2026-10-30T08:00:00+01:00")], TZ
        )
        assert win[1] - win[0] == timedelta(days=6, hours=18)

    def test_overlapping_and_touching_events_merge(self):
        wins = lw.parse_calendar_events(
            [
                _timed("2026-11-13T10:00:00+01:00", "2026-11-13T12:00:00+01:00"),
                _timed("2026-11-13T11:00:00+01:00", "2026-11-13T13:00:00+01:00"),
                _timed("2026-11-13T13:00:00+01:00", "2026-11-13T14:00:00+01:00"),
                _timed("2026-11-13T18:00:00+01:00", "2026-11-13T19:00:00+01:00"),
                # wholly inside the first block: must not shorten it
                _timed("2026-11-13T10:30:00+01:00", "2026-11-13T10:45:00+01:00"),
            ],
            TZ,
        )
        assert wins == [
            (_utc(2026, 11, 13, 9), _utc(2026, 11, 13, 13)),
            (_utc(2026, 11, 13, 17), _utc(2026, 11, 13, 18)),
        ]

    def test_summary_match_is_case_insensitive_and_excludes_the_rest(self):
        payload = [
            _timed("2026-11-13T10:00:00+01:00", "2026-11-13T12:00:00+01:00", "Vecka med KIM"),
            _timed("2026-11-14T10:00:00+01:00", "2026-11-14T12:00:00+01:00", "Alex hemma"),
            {"start": {"dateTime": "2026-11-15T10:00:00+01:00"},
             "end": {"dateTime": "2026-11-15T12:00:00+01:00"}},  # no summary at all
        ]
        assert len(lw.parse_calendar_events(payload, TZ)) == 3
        assert lw.parse_calendar_events(payload, TZ, match="kim") == [
            (_utc(2026, 11, 13, 9), _utc(2026, 11, 13, 11))
        ]

    def test_malformed_events_are_skipped_one_by_one(self):
        good = _timed("2026-11-13T10:00:00+01:00", "2026-11-13T12:00:00+01:00")
        payload = [
            "nonsense",
            None,
            {"summary": "no times"},
            {"start": {"dateTime": "not a time"}, "end": {"dateTime": "2026-11-13T12:00:00Z"}},
            {"start": {"date": "2026-13-45"}, "end": {"date": "2026-11-14"}},
            _timed("2026-11-13T12:00:00+01:00", "2026-11-13T10:00:00+01:00"),  # ends first
            good,
        ]
        assert lw.parse_calendar_events(payload, TZ) == [
            (_utc(2026, 11, 13, 9), _utc(2026, 11, 13, 11))
        ]

    @pytest.mark.parametrize("payload", [None, {"message": "Entity not found"}, "oops", 7])
    def test_a_payload_that_is_not_a_list_is_an_error_not_an_empty_calendar(self, payload):
        with pytest.raises(ValueError):
            lw.parse_calendar_events(payload, TZ)

    def test_preheat_moves_every_start_earlier(self):
        wins = [(_utc(2026, 11, 13, 14), _utc(2026, 11, 13, 16))]
        assert lw.extend_for_preheat(wins, 45) == [
            (_utc(2026, 11, 13, 13, 15), _utc(2026, 11, 13, 16))
        ]
        assert lw.extend_for_preheat(wins, 0) == wins

    def test_iso_round_trip_and_damaged_input(self):
        wins = [(TZ.localize(datetime(2026, 11, 13, 15)), TZ.localize(datetime(2026, 11, 20, 8)))]
        assert lw.windows_from_iso(lw.windows_to_iso(wins)) == wins
        assert lw.windows_from_iso(None) is None
        assert lw.windows_from_iso([["2026-11-13T15:00:00", "2026-11-13T16:00:00"]]) is None
        assert lw.windows_from_iso([["x"]]) is None
        assert lw.windows_from_iso([]) == []


class _Response:
    def __init__(self, payload: Any, status: int = 200):
        self._payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> Any:
        return self._payload


class _Client:
    def __init__(self, response: _Response | Exception):
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def _fetch(response, **kwargs):
    client = _Client(response)
    with (
        patch("backend.core.ha_client.ha_http_client", return_value=client),
        patch(
            "backend.core.secrets.load_home_assistant_config",
            return_value={"url": "http://ha.local:8123/", "token": "t"},
        ),
    ):
        found = await lw.fetch_calendar_windows(
            "calendar.kim", _utc(2026, 11, 12, 10), _utc(2026, 11, 19, 10), TZ, **kwargs
        )
    return found, client


class TestFetch:
    @pytest.mark.asyncio
    async def test_asks_ha_for_the_range_in_aware_iso(self):
        found, client = await _fetch(_Response([]))
        assert found == []
        (url, params), = client.calls
        assert url == "http://ha.local:8123/api/calendars/calendar.kim"
        assert params == {
            "start": "2026-11-12T10:00:00+00:00",
            "end": "2026-11-19T10:00:00+00:00",
        }

    @pytest.mark.asyncio
    async def test_event_spanning_the_range_edges_is_not_truncated(self):
        """HA returns every event OVERLAPPING the range, with its real times."""
        found, _ = await _fetch(
            _Response([_timed("2026-11-06T15:00:00+01:00", "2026-11-27T08:00:00+01:00")])
        )
        assert found == [(_utc(2026, 11, 6, 14), _utc(2026, 11, 27, 7))]

    @pytest.mark.asyncio
    async def test_match_is_applied(self):
        found, _ = await _fetch(
            _Response([_timed("2026-11-13T15:00:00+01:00", "2026-11-13T16:00:00+01:00", "Alex")]),
            match="kim",
        )
        assert found == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [_Response([], status=500), _Response({"message": "nope"}), RuntimeError("timeout")],
    )
    async def test_failure_is_none_never_an_empty_calendar(self, response):
        found, _ = await _fetch(response)
        assert found is None

    @pytest.mark.asyncio
    async def test_an_outage_is_logged_once_not_once_per_retry(self, caplog):
        lw._reset_last_good_for_tests()
        with caplog.at_level("INFO", logger="darkstar.core.load_windows"):
            for _ in range(3):
                await _fetch(RuntimeError("timeout"))
            await _fetch(_Response([]))
            await _fetch(_Response([]))
        messages = [r.message for r in caplog.records]
        assert sum("could not be read" in m for m in messages) == 1
        assert sum("readable again" in m for m in messages) == 1

    @pytest.mark.asyncio
    async def test_no_ha_credentials_is_none(self):
        with patch("backend.core.secrets.load_home_assistant_config", return_value={}):
            assert (
                await lw.fetch_calendar_windows(
                    "calendar.kim", _utc(2026, 11, 12), _utc(2026, 11, 13), TZ
                )
                is None
            )


NOW = TZ.localize(datetime(2026, 11, 12, 10, 20))
EVENT = (_utc(2026, 11, 13, 14), _utc(2026, 11, 20, 7))


def _load(**over: Any) -> dict[str, Any]:
    base = {"id": "kim_heater", "mode": "window", "window_entities": ["calendar.kim"]}
    base.update(over)
    return base


def _state_reader(states: dict[str, Any]):
    async def get_state(entity: str):
        value = states.get(entity)
        if isinstance(value, Exception):
            raise value
        return None if value is None else {"state": value}

    return get_state


def _fetcher(answers: list[Any]):
    calls: list[dict[str, Any]] = []

    async def fetch(entity, start, end, tz, *, match=None):
        calls.append({"entity": entity, "start": start, "end": end, "match": match})
        return answers.pop(0) if len(answers) > 1 else answers[0]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


class TestPlanWindows:
    @pytest.fixture(autouse=True)
    def _clean(self):
        lw._reset_last_good_for_tests()
        yield
        lw._reset_last_good_for_tests()

    @pytest.mark.asyncio
    async def test_calendar_windows_over_the_lookahead_with_match_and_preheat(self):
        fetch = _fetcher([[EVENT]])
        found = await lw.plan_windows_for_load(
            _load(window_event_match="Kim", preheat_minutes=60),
            NOW, TZ, get_state=_state_reader({}), fetch=fetch,
        )
        assert found == [(EVENT[0] - timedelta(hours=1), EVENT[1])]
        (call,) = fetch.calls
        assert call["match"] == "Kim"
        assert call["start"] == NOW and call["end"] == NOW + lw.PLAN_LOOKAHEAD

    @pytest.mark.asyncio
    async def test_unreadable_calendar_reuses_the_last_good_answer_while_recent(self):
        fetch = _fetcher([[EVENT], None])
        args = (_load(), NOW, TZ)
        with patch("backend.core.load_windows.time.time", return_value=1000.0):
            assert await lw.plan_windows_for_load(
                *args, get_state=_state_reader({}), fetch=fetch
            ) == [EVENT]
        almost = 1000.0 + lw.CALENDAR_LAST_GOOD_MAX_AGE_S - 1
        with patch("backend.core.load_windows.time.time", return_value=almost):
            assert await lw.plan_windows_for_load(
                *args, get_state=_state_reader({}), fetch=fetch
            ) == [EVENT]

    @pytest.mark.asyncio
    async def test_unreadable_calendar_with_only_an_old_answer_books_nothing(self, caplog):
        fetch = _fetcher([[EVENT], None])
        args = (_load(), NOW, TZ)
        with patch("backend.core.load_windows.time.time", return_value=1000.0):
            await lw.plan_windows_for_load(*args, get_state=_state_reader({}), fetch=fetch)
        too_old = 1000.0 + lw.CALENDAR_LAST_GOOD_MAX_AGE_S + 1
        with (
            patch("backend.core.load_windows.time.time", return_value=too_old),
            caplog.at_level("WARNING", logger="darkstar.core.load_windows"),
        ):
            assert (
                await lw.plan_windows_for_load(*args, get_state=_state_reader({}), fetch=fetch)
                is None
            )
        assert sum("booking NO load" in r.message for r in caplog.records) == 1

    @pytest.mark.asyncio
    async def test_unreadable_calendar_never_read_before_books_nothing(self):
        assert (
            await lw.plan_windows_for_load(
                _load(), NOW, TZ, get_state=_state_reader({}), fetch=_fetcher([None])
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_last_good_answer_drops_windows_that_have_ended(self):
        gone = (_utc(2026, 11, 1, 14), _utc(2026, 11, 6, 7))
        fetch = _fetcher([[gone, EVENT], None])
        args = (_load(), NOW, TZ)
        with patch("backend.core.load_windows.time.time", return_value=1000.0):
            await lw.plan_windows_for_load(*args, get_state=_state_reader({}), fetch=fetch)
            assert await lw.plan_windows_for_load(
                *args, get_state=_state_reader({}), fetch=fetch
            ) == [EVENT]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entity", ["input_boolean.kim_here", "schedule.kim", "binary_sensor.x", "switch.y"]
    )
    async def test_a_non_calendar_gate_that_is_on_persists_over_the_horizon(self, entity):
        """No future schedule to read: its current state is assumed to last, from
        the start of the current hour so the in-progress slot is covered."""
        found = await lw.plan_windows_for_load(
            _load(window_entities=[entity], preheat_minutes=30),
            NOW, TZ, get_state=_state_reader({entity: "on"}), fetch=_fetcher([None]),
        )
        assert found == [(TZ.localize(datetime(2026, 11, 12, 10, 0)), NOW + lw.PLAN_LOOKAHEAD)]

    @pytest.mark.asyncio
    async def test_a_non_calendar_gate_that_is_off_gives_no_window(self):
        found = await lw.plan_windows_for_load(
            _load(window_entities=["input_boolean.kim_here"]),
            NOW, TZ, get_state=_state_reader({"input_boolean.kim_here": "off"}),
        )
        assert found == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["unavailable", "unknown", None, RuntimeError("down")])
    async def test_an_unreadable_gate_is_unknown_not_off_and_not_on(self, state):
        found = await lw.plan_windows_for_load(
            _load(window_entities=["input_boolean.kim_here"]),
            NOW, TZ, get_state=_state_reader({"input_boolean.kim_here": state}),
        )
        assert found is None

    @pytest.mark.asyncio
    async def test_any_of_several_entities_opens_the_window(self):
        found = await lw.plan_windows_for_load(
            _load(window_entities=["calendar.kim", "input_boolean.guest"]),
            NOW, TZ,
            get_state=_state_reader({"input_boolean.guest": "off"}),
            fetch=_fetcher([[EVENT]]),
        )
        assert found == [EVENT]

    @pytest.mark.asyncio
    async def test_no_window_entities_is_unknown(self):
        assert (
            await lw.plan_windows_for_load(
                _load(window_entities=[]), NOW, TZ, get_state=_state_reader({})
            )
            is None
        )


class TestMode:
    @pytest.mark.parametrize(
        ("raw", "mode"),
        [
            ({}, lw.MODE_DAILY),
            ({"mode": None}, lw.MODE_DAILY),
            ({"mode": "daily"}, lw.MODE_DAILY),
            ({"mode": " Window "}, lw.MODE_WINDOW),
            ({"mode": "windows"}, lw.MODE_INVALID),
        ],
    )
    def test_a_typo_is_invalid_not_the_default(self, raw, mode):
        assert lw.load_mode(raw) == mode
