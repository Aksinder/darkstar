"""The daily EV-priority confirmation: ask, keep on tap, revert on silence."""

from __future__ import annotations

import pytest

from backend.services.ev_priority_attendant import KEEP_ACTION, EVPriorityAttendant


def _attendant(tmp_path, monkeypatch, select_state="tesla_first"):
    import backend.services.ev_priority_attendant as mod

    monkeypatch.setattr(mod, "_STATE_PATH", tmp_path / "ask.json")
    a = EVPriorityAttendant(
        {
            "timezone": "Europe/Stockholm",
            "executor": {
                "ev_surplus": {
                    "priority_entity": "input_select.darkstar_ev_priority",
                    "priority_attendant": {"ask_time": "20:00", "timeout_minutes": 60},
                }
            },
            "notifications": {"service": "notify.test"},
        }
    )
    a._state = a._load_state()
    calls: dict[str, list] = {"notify": [], "post": []}
    monkeypatch.setattr(
        a, "_notify", lambda t, m, actionable: calls["notify"].append((t, m, actionable)) or True
    )
    monkeypatch.setattr(
        a, "_post", lambda path, payload: calls["post"].append((path, payload)) or True
    )
    monkeypatch.setattr(a, "_get_select_state", lambda: select_state)
    return a, calls


# 2026-08-27 20:00:30 Europe/Stockholm as epoch (CEST = UTC+2)
ASK_TS = 1_787_853_630.0


class TestAttendant:
    def test_asks_at_ask_time_when_not_auto(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        assert calls["notify"], "must ask at 20:00 when select != auto"
        assert calls["notify"][0][2] is True, "the ask must be actionable"
        assert a._state.get("asked_at") == ASK_TS

    def test_auto_never_asks(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch, select_state="auto")
        a.tick_sync(ASK_TS)
        assert not calls["notify"]

    def test_midday_never_asks(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS - 6 * 3600)  # 14:00
        assert not calls["notify"]

    def test_tap_keeps_priority(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        # The tap only sets the cross-thread flag; the next tick folds it in.
        a.on_notification_action(KEEP_ACTION)
        assert a._kept_flag is True
        a.tick_sync(ASK_TS + 61 * 60)
        assert not calls["post"], "kept priority must not be reverted"

    def test_foreign_tap_is_ignored(self, tmp_path, monkeypatch):
        a, _ = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        a.on_notification_action("SOME_OTHER_ACTION")
        assert a._state.get("kept") is False

    def test_silence_reverts_to_auto(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        a.tick_sync(ASK_TS + 61 * 60)
        assert any(
            p[0].endswith("input_select/select_option") and p[1]["option"] == "auto"
            for p in calls["post"]
        ), f"must revert to auto, posts={calls['post']}"
        # And the revert is announced (non-actionable)
        assert any(not n[2] for n in calls["notify"][1:])

    def test_owner_change_during_window_wins(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        # Owner flips the select themselves mid-window
        monkeypatch.setattr(a, "_get_select_state", lambda: "fmb_first")
        a.tick_sync(ASK_TS + 61 * 60)
        assert not calls["post"], "an explicit owner change supersedes the ask"

    def test_asks_only_once_per_day(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        a.on_notification_action(KEEP_ACTION)
        a.tick_sync(ASK_TS + 61 * 60)  # resolves
        a.tick_sync(ASK_TS + 62 * 60)  # 21:02 — same day, no new ask
        asks = [n for n in calls["notify"] if n[2]]
        assert len(asks) == 1

    def test_state_survives_restart(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        a._save_state()
        # "Restart": a new attendant instance reads the same file
        b, calls_b = _attendant(tmp_path, monkeypatch)
        assert b._state.get("asked_at") == ASK_TS
        b.tick_sync(ASK_TS + 61 * 60)
        assert any(
            p[0].endswith("input_select/select_option") for p in calls_b["post"]
        ), "the pending ask must survive a restart and still revert"


class TestReviewFindings:
    def test_read_failure_in_window_retries_not_burns_day(self, tmp_path, monkeypatch):
        # HA unreachable at 20:00 sharp: the next tick inside the window must
        # still ask (the first draft stamped ask_day and went silent all day).
        a, calls = _attendant(tmp_path, monkeypatch)
        monkeypatch.setattr(a, "_get_select_state", lambda: None)
        a.tick_sync(ASK_TS)
        assert not calls["notify"] and not a._state.get("ask_day")
        monkeypatch.setattr(a, "_get_select_state", lambda: "tesla_first")
        a.tick_sync(ASK_TS + 120)
        assert calls["notify"], "recovered read inside the window must ask"

    def test_failed_revert_post_retries(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        # First revert attempt: POST fails — the pending ask must survive.
        monkeypatch.setattr(a, "_post", lambda p, pl: False)
        a.tick_sync(ASK_TS + 61 * 60)
        assert a._state.get("asked_at"), "failed POST must keep the ask pending"
        # HA back: the retry lands.
        ok_posts = []
        monkeypatch.setattr(a, "_post", lambda p, pl: ok_posts.append((p, pl)) or True)
        a.tick_sync(ASK_TS + 62 * 60)
        assert any(p[0].endswith("select_option") for p in ok_posts)

    def test_dst_spring_forward_day_still_asks(self, tmp_path, monkeypatch):
        # 2026-03-28 20:00:30 CET (the evening before spring-forward, UTC+1):
        # the next-minus-86400 arithmetic skipped this whole day.
        a, calls = _attendant(tmp_path, monkeypatch)
        import pytz
        from datetime import datetime
        tz = pytz.timezone("Europe/Stockholm")
        ts = tz.localize(datetime(2026, 3, 28, 20, 0, 30)).timestamp()
        a.tick_sync(ts)
        assert calls["notify"], "the day before DST must still ask at 20:00"

    def test_tap_flag_from_ws_thread_wins_last_instant(self, tmp_path, monkeypatch):
        a, calls = _attendant(tmp_path, monkeypatch)
        a.tick_sync(ASK_TS)
        # Simulate the tap landing between the fold and the revert POST:
        real_get = a._get_select_state
        def get_and_tap():
            a._kept_flag = True  # ws thread writes the flag mid-tick
            return real_get()
        monkeypatch.setattr(a, "_get_select_state", get_and_tap)
        a.tick_sync(ASK_TS + 61 * 60)
        assert not calls["post"], "a last-instant tap must block the revert"
