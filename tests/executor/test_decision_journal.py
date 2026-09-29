"""The journal must be able to answer "which plan said so, and what did the one
before it say?" — the question nobody could answer about 2026-09-25 03:37.

That night the executor went from idle (holding 37 % for the morning peak) to
self-consumption and spent the battery on the cheapest hour of the day. The
add-on log holds about two hours and the plan lived in a database inside the
container, so by morning the evidence was gone.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
import pytz

from executor.decision_journal import DEFAULT_KEEP, DecisionJournal
from executor.engine import ExecutorEngine
from executor.override import SlotPlan

TZ = pytz.timezone("Europe/Stockholm")


def _entry(
    mode="idle",
    *,
    source="plan",
    planned_at="2026-09-25T03:15:02",
    target=38,
    slot_start="2026-09-25T03:30:00+02:00",
    ts="2026-09-25T03:36:07+02:00",
    **extra,
):
    return {
        "ts": ts,
        "mode": mode,
        "source": source,
        "reason": f"Plan: Hold/Idle | {mode}",
        "soc": 37.7,
        "slot_start": slot_start,
        "planned_at": planned_at,
        "slot": {"discharge_kw": 0.0, "soc_target": target},
        "ahead": [],
        **extra,
    }


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


class TestWhatGetsWritten:
    def test_the_incident_is_recorded_with_both_plans(self, tmp_path):
        path = tmp_path / "decision_journal.jsonl"
        j = DecisionJournal(path)
        assert j.observe(_entry("idle", planned_at="2026-09-25T03:15:02", target=38)) is False
        assert (
            j.observe(
                _entry(
                    "self_consumption",
                    planned_at="2026-09-25T03:30:04",
                    target=24,
                    ts="2026-09-25T03:37:07+02:00",
                )
            )
            is True
        )
        (line,) = _lines(path)
        assert (line["from"], line["to"], line["kind"]) == ("idle", "self_consumption", "mode")
        assert line["prev_planned_at"] == "2026-09-25T03:15:02"
        assert line["planned_at"] == "2026-09-25T03:30:04"
        assert line["prev_slot"]["soc_target"] == 38 and line["slot"]["soc_target"] == 24
        assert line["soc"] == 37.7

    def test_steady_state_writes_nothing(self, tmp_path):
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path)
        for minute in range(30):
            j.observe(_entry("idle", ts=f"2026-09-25T03:{minute:02d}:07+02:00"))
        assert not path.exists()

    def test_the_first_tick_after_a_start_is_remembered_not_reported(self, tmp_path):
        path = tmp_path / "j.jsonl"
        assert DecisionJournal(path).observe(_entry("charge")) is False
        assert not path.exists()

    def test_an_override_taking_over_is_a_change(self, tmp_path):
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path)
        j.observe(_entry("idle"))
        assert j.observe(_entry("idle", source="override")) is True
        assert _lines(path)[0]["prev_source"] == "plan"

    def test_a_replan_that_moves_the_hold_target_is_recorded(self, tmp_path):
        """Same mode, same slot, new plan, target down 14 points: a change of mind
        that has not reached the inverter yet."""
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path)
        j.observe(_entry("self_consumption", planned_at="A", target=38))
        assert j.observe(_entry("self_consumption", planned_at="B", target=24)) is True
        assert _lines(path)[0]["kind"] == "target"

    @pytest.mark.parametrize(
        "planned_at,target,slot_start",
        [
            ("A", 24, "2026-09-25T03:30:00+02:00"),  # same plan: the target did not move by replan
            ("B", 36, "2026-09-25T03:30:00+02:00"),  # new plan, 2 points: noise
            (
                "B",
                24,
                "2026-09-25T03:45:00+02:00",
            ),  # next slot: targets differ between slots anyway
        ],
    )
    def test_what_is_not_a_target_shift(self, tmp_path, planned_at, target, slot_start):
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path)
        j.observe(_entry("idle", planned_at="A", target=38))
        assert (
            j.observe(_entry("idle", planned_at=planned_at, target=target, slot_start=slot_start))
            is False
        )

    def test_optional_context_is_carried(self, tmp_path):
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path)
        j.observe(_entry("self_consumption"))
        j.observe(_entry("idle", ev_isolation=True, import_price=2.47))
        line = _lines(path)[0]
        assert line["ev_isolation"] is True and line["import_price"] == 2.47


class TestItCannotHurt:
    def test_disabled_where_the_directory_does_not_exist(self, tmp_path):
        j = DecisionJournal(tmp_path / "no_such_dir" / "j.jsonl")
        assert j.enabled is False
        j.observe(_entry("idle"))
        assert j.observe(_entry("charge")) is False
        assert not (tmp_path / "no_such_dir").exists()

    def test_no_path_at_all(self):
        j = DecisionJournal(None)
        j.observe(_entry("idle"))
        assert j.observe(_entry("charge")) is False

    def test_a_write_failure_is_swallowed(self, tmp_path):
        target = tmp_path / "j.jsonl"
        target.mkdir()  # a directory where the file should be
        j = DecisionJournal(target)
        j.observe(_entry("idle"))
        assert j.observe(_entry("charge")) is False

    def test_garbage_in_is_swallowed(self, tmp_path):
        j = DecisionJournal(tmp_path / "j.jsonl")
        assert j.observe(None) is False  # type: ignore[arg-type]
        assert j.observe({"mode": object()}) is False

    def test_the_file_stays_bounded(self, tmp_path):
        path = tmp_path / "j.jsonl"
        j = DecisionJournal(path, keep=20)
        for i in range(200):
            j.observe(_entry("idle" if i % 2 else "charge", ts=str(i)))
        lines = _lines(path)
        assert 20 <= len(lines) <= 20 + 50
        assert lines[-1]["ts"] == "199"  # newest kept
        assert int(lines[0]["ts"]) > 100  # oldest dropped

    def test_default_keep_is_days_not_hours(self):
        assert DEFAULT_KEEP >= 200


class TestEngineHook:
    def _engine(self, tmp_path, schedule):
        sched = tmp_path / "schedule.json"
        sched.write_text(json.dumps(schedule))
        engine = ExecutorEngine.__new__(ExecutorEngine)
        engine.config = SimpleNamespace(schedule_path=str(sched), timezone="Europe/Stockholm")
        engine._journal = DecisionJournal(tmp_path / "decision_journal.jsonl")
        return engine, sched

    @staticmethod
    def _schedule(planned_at, discharge_0345):
        return {
            "meta": {"planned_at": planned_at},
            "schedule": [
                {
                    "start_time": "2026-09-25T03:15:00+02:00",
                    "battery_discharge_kw": 0.0,
                    "soc_target_percent": 38,
                },
                {
                    "start_time": "2026-09-25T03:30:00+02:00",
                    "battery_discharge_kw": 0.0,
                    "soc_target_percent": 38,
                    "import_price_sek_kwh": 2.47,
                },
                {
                    "start_time": "2026-09-25T03:45:00+02:00",
                    "battery_discharge_kw": discharge_0345,
                    "soc_target_percent": 30,
                },
            ],
        }

    def test_a_flip_is_journalled_with_the_plan_that_caused_it(self, tmp_path):
        engine, sched = self._engine(tmp_path, self._schedule("2026-09-25T03:15:02", 0.0))
        state = SimpleNamespace(current_soc_percent=37.7)
        now = TZ.localize(datetime(2026, 9, 25, 3, 36, 7))
        hold = SimpleNamespace(mode_intent="idle", source="plan", reason="hold")
        engine._journal_decision(
            hold, SlotPlan(soc_target=38), "2026-09-25T03:30:00+02:00", state, now
        )

        sched.write_text(json.dumps(self._schedule("2026-09-25T03:30:04", 1.2)))
        go = SimpleNamespace(mode_intent="self_consumption", source="plan", reason="use it")
        engine._journal_decision(
            go,
            SlotPlan(discharge_kw=1.2, soc_target=24),
            "2026-09-25T03:30:00+02:00",
            state,
            now.replace(minute=37),
        )

        (line,) = _lines(tmp_path / "decision_journal.jsonl")
        assert (line["from"], line["to"]) == ("idle", "self_consumption")
        assert line["prev_planned_at"] == "2026-09-25T03:15:02"
        assert line["planned_at"] == "2026-09-25T03:30:04"
        assert line["slot"]["discharge_kw"] == 1.2
        # the past slot is gone, the current one and what follows are there
        assert [a["start"] for a in line["ahead"]] == ["03:30", "03:45"]
        assert line["ahead"][1]["discharge_kw"] == 1.2

    def test_a_missing_schedule_does_not_break_the_tick(self, tmp_path):
        engine, sched = self._engine(tmp_path, {"schedule": []})
        sched.unlink()
        now = TZ.localize(datetime(2026, 9, 25, 3, 36, 7))
        state = SimpleNamespace(current_soc_percent=50.0)
        for mode in ("idle", "charge"):
            engine._journal_decision(
                SimpleNamespace(mode_intent=mode, source="plan", reason=""),
                None,
                None,
                state,
                now,
            )
        (line,) = _lines(tmp_path / "decision_journal.jsonl")
        assert line["planned_at"] is None and line["slot"] is None and line["ahead"] == []

    def test_an_engine_without_a_journal_is_untouched(self):
        engine = ExecutorEngine.__new__(ExecutorEngine)
        engine._journal_decision(None, None, None, None, datetime.now(TZ))
