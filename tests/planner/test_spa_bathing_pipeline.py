"""Spa bathing times, planner side (planner/spa_bathing.py + the pipeline glue).

The invariants under test:

* no block / ``enabled: false`` => byte-identical plan and schedule output;
* shadow => the solver input is untouched, the legacy plan's temperature is simulated
  and published;
* an unknown calendar or temperature => ``fallback`` (legacy plan), never an invented
  window;
* every datetime is aware: DST nights, all-day events, in-progress windows.
"""

from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
import pytz

from backend.core import load_windows as lw
from backend.core.spa_thermal import simulate_temps_c
from planner import pipeline as pl
from planner.output.formatter import dataframe_to_json_response
from planner.solver.adapter import kepler_result_to_dataframe
from planner.solver.types import (
    BathingReport,
    KeplerInputSlot,
    KeplerResult,
    KeplerResultSlot,
    WaterHeaterInput,
)
from planner.spa_bathing import (
    BathingDecision,
    decide_bathing,
    ready_state_index,
    simulate_plan,
    summarize,
)

TZ = pytz.timezone("Europe/Stockholm")
Q = timedelta(minutes=15)
# Monday 2026-11-16 12:00 local (11:00 UTC).
NOW = TZ.localize(datetime(2026, 11, 16, 12, 0))

BATHING = {
    "enabled": True,
    "window_entities": ["calendar.spa"],
    "temp_sensor": "sensor.layzspa_temp_c",
}


def _cfg(**over: Any) -> dict[str, Any]:
    return {"id": "spa", "power_kw": 1.8, "bathing": {**BATHING, **over}}


def _heater(power_kw: float = 1.8, hid: str = "spa") -> WaterHeaterInput:
    return WaterHeaterInput(
        id=hid,
        power_kw=power_kw,
        min_kwh_per_day=3.0,
        max_hours_between_heating=24.0,
        min_spacing_hours=0.0,
    )


def _slots(n: int = 96, start: datetime = NOW, step: timedelta = Q) -> list[KeplerInputSlot]:
    return [
        KeplerInputSlot(
            start_time=start + i * step,
            end_time=start + (i + 1) * step,
            load_kwh=0.2,
            pv_kwh=0.0,
            import_price_sek_kwh=1.0,
            export_price_sek_kwh=0.5,
        )
        for i in range(n)
    ]


def _inputs(windows: Any = (), temp: Any = 30.0, ambient: Any = None) -> dict[str, Any]:
    return {
        "windows": None if windows is None else lw.windows_to_iso(list(windows)),
        "temp_c": temp,
        "ambient_c": ambient,
        "read_at": NOW.isoformat(),
    }


def _decide(
    cfg: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    *,
    heater: WaterHeaterInput | None = None,
    slots: list[Any] | None = None,
    now: datetime = NOW,
    mean: float | None = 4.0,
    tmp_path: Any = None,
) -> BathingDecision:
    return decide_bathing(
        heater_cfg=cfg if cfg is not None else _cfg(),
        inputs=inputs if inputs is not None else _inputs(),
        heater=heater or _heater(),
        slots=slots if slots is not None else _slots(),
        first_slot_remaining_h=None,
        now=now,
        mean_outdoor_temp_c=mean,
        learned_params_path=str(tmp_path / "params.json") if tmp_path else "/nonexistent/x.json",
    )


EVENING = (NOW + timedelta(hours=7), NOW + timedelta(hours=9))  # 19:00-21:00


# ---------------------------------------------------------------------------
# decide_bathing
# ---------------------------------------------------------------------------


class TestDecideModes:
    def test_no_block_is_disabled(self):
        d = _decide({"id": "spa", "power_kw": 1.8})
        assert d.mode == "disabled" and d.spec is None and d.spec_for_simulation is None

    def test_enabled_false_is_disabled(self):
        d = _decide(_cfg(enabled=False))
        assert d.mode == "disabled" and d.spec is None

    def test_no_inputs_is_disabled(self):
        d = decide_bathing(
            heater_cfg=_cfg(),
            inputs=None,
            heater=_heater(),
            slots=_slots(),
            first_slot_remaining_h=None,
            now=NOW,
            mean_outdoor_temp_c=None,
        )
        assert d.mode == "disabled"

    def test_shadow_is_the_default_and_hands_the_solver_nothing(self):
        d = _decide(inputs=_inputs([EVENING]))
        assert d.mode == "shadow"
        assert d.spec is None
        assert d.spec_for_simulation is not None
        assert len(d.spec_for_simulation.windows) == 1

    def test_shadow_false_is_solver_mode(self):
        d = _decide(_cfg(shadow=False), _inputs([EVENING]))
        assert d.mode == "solver"
        assert d.spec is not None and d.spec is d.spec_for_simulation

    def test_field_mapping(self):
        d = _decide(
            _cfg(
                shadow=False,
                ready_shortfall_sek_per_c=7.0,
                floor_shortfall_sek_per_c=3.0,
                hold_tolerance_c=0.5,
                overshoot_penalty_sek_per_c=12.0,
                ready_temp_c=37.0,
                eco_floor_c=27.0,
                temp_max_c=39.0,
            ),
            _inputs([EVENING], temp=31.5),
        )
        spec = d.spec
        assert spec is not None
        assert spec.initial_temp_c == 31.5
        assert spec.ready_shortfall_sek_per_c == 7.0
        assert spec.hold_shortfall_sek_per_c_h == 7.0
        assert spec.floor_shortfall_sek_per_c_h == 3.0
        assert spec.hold_tolerance_c == 0.5
        assert spec.overshoot_penalty_sek_per_c == 12.0
        assert (spec.ready_temp_c, spec.eco_floor_c, spec.temp_max_c) == (37.0, 27.0, 39.0)
        assert (spec.heat_capacity_kwh_per_c, spec.cooling_k_per_h) == (0.9, 0.021)

    def test_empty_window_list_is_not_a_fallback(self):
        """A readable calendar with no events means "no bath": hold the eco floor."""
        for shadow, mode in ((True, "shadow"), (False, "solver")):
            d = _decide(_cfg(shadow=shadow), _inputs([]))
            assert d.mode == mode
            assert d.windows == []
            assert d.spec_for_simulation is not None
            assert d.spec_for_simulation.windows == []


class TestFallback:
    @pytest.mark.parametrize(
        ("inputs", "reason"),
        [
            (_inputs(None), "calendar_unknown"),
            ({"temp_c": 30.0}, "calendar_unknown"),
            (_inputs([EVENING], temp=None), "temp_unreadable"),
            (_inputs([EVENING], temp="unavailable"), "temp_unreadable"),
            (_inputs([EVENING], temp=float("nan")), "temp_unreadable"),
            (_inputs([EVENING], temp=-3.0), "temp_out_of_range"),
            (_inputs([EVENING], temp=61.0), "temp_out_of_range"),
        ],
    )
    def test_unknown_inputs_fall_back(self, inputs, reason):
        d = _decide(_cfg(shadow=False), inputs)
        assert d.mode == "fallback"
        assert d.reason == reason
        assert d.spec is None and d.spec_for_simulation is None

    def test_damaged_window_list_is_unknown(self):
        bad = {**_inputs(), "windows": [["2026-11-16T19:00:00", "2026-11-16T21:00:00"]]}
        d = _decide(inputs=bad)  # naive ISO strings are "damaged", not local time
        assert d.mode == "fallback" and d.reason == "calendar_unknown"

    def test_fallback_order_calendar_before_temperature(self):
        d = _decide(inputs=_inputs(None, temp=None))
        assert d.reason == "calendar_unknown"

    def test_invalid_config(self):
        d = _decide(_cfg(temp_sensor=""), _inputs([EVENING]))
        assert d.mode == "fallback" and d.reason == "invalid_config:no_temp_sensor"

    def test_no_power(self):
        d = _decide(inputs=_inputs([EVENING]), heater=_heater(power_kw=0.0))
        assert d.mode == "fallback" and d.reason == "no_power"

    def test_naive_now_is_never_compared(self):
        d = _decide(now=NOW.replace(tzinfo=None))
        assert d.mode == "fallback" and d.reason == "naive_now"

    def test_naive_slots_are_never_paired_with_aware_windows(self):
        naive = _slots(start=datetime(2026, 11, 16, 12, 0))
        d = _decide(inputs=_inputs([EVENING]), slots=naive)
        assert d.mode == "fallback" and d.reason == "naive_slots"


class TestAmbientAndParams:
    def test_measured_ambient_wins(self):
        assert _decide(inputs=_inputs(ambient=2.5), mean=4.0).ambient_c == 2.5

    def test_implausible_measured_ambient_uses_the_forecast_mean(self):
        assert _decide(inputs=_inputs(ambient=80.0), mean=4.0).ambient_c == 4.0

    def test_no_ambient_at_all_uses_the_configured_default(self):
        d = _decide(_cfg(ambient_default_c=6.0), _inputs(ambient=None), mean=None)
        assert d.ambient_c == 6.0
        assert d.spec_for_simulation is not None and d.spec_for_simulation.ambient_c == 6.0

    def test_learned_params_are_used_when_fresh(self, tmp_path):
        import json

        (tmp_path / "params.json").write_text(
            json.dumps(
                {
                    "heat_capacity_kwh_per_c": 0.8,
                    "cooling_k_per_h": 0.03,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        d = _decide(tmp_path=tmp_path)
        assert d.params_source == "learned"
        assert d.spec_for_simulation is not None
        assert d.spec_for_simulation.heat_capacity_kwh_per_c == 0.8
        d2 = _decide(_cfg(learn=False), tmp_path=tmp_path)
        assert d2.params_source == "config"
        assert d2.spec_for_simulation is not None
        assert d2.spec_for_simulation.heat_capacity_kwh_per_c == 0.9

    def test_missing_learned_file_uses_config(self, tmp_path):
        assert _decide(tmp_path=tmp_path).params_source == "config"


class TestWindows:
    def test_ready_by_is_start_minus_margin(self):
        d = _decide(_cfg(ready_margin_minutes=45), _inputs([EVENING]))
        (w,) = d.spec_for_simulation.windows  # type: ignore[union-attr]
        assert w.start == EVENING[0] and w.end == EVENING[1]
        assert w.ready_by == EVENING[0] - timedelta(minutes=45)
        assert w.ready_by.tzinfo is not None

    def test_dst_spring_forward_night(self):
        """2026-03-29: 02:00 CET jumps to 03:00 CEST. A 03:00-05:00 bath is 01:00-03:00
        UTC, and 30 minutes before 03:00 CEST is 01:30 CET (00:30 UTC) — a wall-clock
        subtraction would land in the hour that does not exist."""
        (win,) = lw.parse_calendar_events(
            [
                {
                    "summary": "Bad",
                    "start": {"dateTime": "2026-03-29T03:00:00"},
                    "end": {"dateTime": "2026-03-29T05:00:00"},
                }
            ],
            TZ,
        )
        now = TZ.localize(datetime(2026, 3, 28, 18, 0))
        d = _decide(inputs=_inputs([win]), now=now, slots=_slots(start=now))
        (w,) = d.spec_for_simulation.windows  # type: ignore[union-attr]
        assert w.start == datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
        assert w.end == datetime(2026, 3, 29, 3, 0, tzinfo=UTC)
        assert w.ready_by == datetime(2026, 3, 29, 0, 30, tzinfo=UTC)
        local = w.ready_by.astimezone(TZ)
        assert (local.hour, local.minute, local.utcoffset()) == (1, 30, timedelta(hours=1))

    def test_all_day_event(self):
        """An all-day bath on the 17th runs local midnight to midnight; ready by 23:30
        the evening before."""
        (win,) = lw.parse_calendar_events(
            [{"summary": "Bad", "start": {"date": "2026-11-17"}, "end": {"date": "2026-11-18"}}],
            TZ,
        )
        d = _decide(inputs=_inputs([win]))
        (w,) = d.spec_for_simulation.windows  # type: ignore[union-attr]
        assert w.start == TZ.localize(datetime(2026, 11, 17, 0, 0))
        assert w.end - w.start == timedelta(hours=24)
        assert w.ready_by == TZ.localize(datetime(2026, 11, 16, 23, 30))

    def test_in_progress_window_is_kept_and_ended_window_dropped(self):
        ended = (NOW - timedelta(hours=3), NOW - timedelta(hours=1))
        running = (NOW - timedelta(minutes=30), NOW + timedelta(hours=1))
        d = _decide(inputs=_inputs([ended, running, EVENING]))
        assert d.windows == [running, EVENING]
        assert [w.start for w in d.spec_for_simulation.windows] == [  # type: ignore[union-attr]
            running[0],
            EVENING[0],
        ]

    def test_window_beyond_the_horizon_is_passed_on(self):
        """The solver applies the horizon rule (no target); the decision never drops a
        real window, so the sensors can still show next Saturday's bath."""
        far = (NOW + timedelta(days=5), NOW + timedelta(days=5, hours=2))
        d = _decide(inputs=_inputs([far]), slots=_slots(96))
        assert d.windows == [far]
        summary = summarize(d, [30.0] * 96, None, _slots(96), NOW)
        assert summary["next_bath"]["start"] == far[0].isoformat()
        assert summary["next_bath"]["predicted_temp_c"] is None
        assert summary["next_bath"]["shortfall_c"] is None
        assert summary["on_track"] is True


# ---------------------------------------------------------------------------
# simulate_plan / ready_state_index / summarize
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_matches_the_shared_recursion(self):
        d = _decide(inputs=_inputs([EVENING], temp=30.0, ambient=5.0))
        slots = _slots(8)
        kw = [0.0, 1.8, 1.8, 0.0, 0.0, 0.005, 1.8, 0.0]
        temps = simulate_plan(d, slots, kw, None)
        want = simulate_temps_c(30.0, 5.0, 0.021, 0.9, 1.8, [0.25] * 8, [0, 1, 1, 0, 0, 0, 1, 0])
        assert temps == pytest.approx(want)
        assert len(temps) == 8

    def test_partial_first_slot_uses_the_solvers_hours(self):
        d = _decide(inputs=_inputs([], temp=30.0, ambient=5.0))
        slots = _slots(3)
        temps = simulate_plan(d, slots, [1.8, 0.0, 0.0], 0.1)
        want = simulate_temps_c(30.0, 5.0, 0.021, 0.9, 1.8, [0.1, 0.25, 0.25], [1, 0, 0])
        assert temps == pytest.approx(want)

    def test_no_spec_simulates_nothing(self):
        d = _decide(inputs=_inputs(None))
        assert simulate_plan(d, _slots(4), [1.8] * 4, None) == []

    def test_length_mismatch_raises(self):
        d = _decide()
        with pytest.raises(ValueError):
            simulate_plan(d, _slots(4), [1.8] * 3, None)

    def test_ready_state_index(self):
        slots = _slots(8)  # 12:00..14:00
        assert ready_state_index(slots, NOW + timedelta(minutes=60), NOW) == 4
        assert ready_state_index(slots, NOW + timedelta(minutes=70), NOW) == 4
        assert ready_state_index(slots, NOW + timedelta(minutes=5), NOW) == 1  # inside slot 0
        assert ready_state_index(slots, NOW - timedelta(minutes=5), NOW) == 1  # already past
        assert ready_state_index(slots, NOW + timedelta(hours=3), NOW) is None  # beyond


SUMMARY_KEYS = {
    "heater_id",
    "mode",
    "reason",
    "planned_at",
    "measured_temp_c",
    "ambient_c",
    "ready_temp_c",
    "eco_floor_c",
    "params",
    "windows",
    "next_bath",
    "on_track",
}


class TestSummarize:
    def test_shadow_schema_and_prediction(self):
        d = _decide(inputs=_inputs([EVENING], temp=30.0, ambient=5.0))
        slots = _slots(96)
        kw = [1.8 if 16 <= i < 26 else 0.0 for i in range(96)]
        temps = simulate_plan(d, slots, kw, None)
        s = summarize(d, temps, None, slots, NOW)
        assert set(s) == SUMMARY_KEYS
        assert s["mode"] == "shadow" and s["reason"] is None
        assert s["params"] == {
            "heat_capacity_kwh_per_c": 0.9,
            "cooling_k_per_h": 0.021,
            "source": "config",
        }
        assert s["windows"] == [[EVENING[0].isoformat(), EVENING[1].isoformat()]]
        nb = s["next_bath"]
        assert set(nb) == {"start", "end", "ready_by", "predicted_temp_c", "shortfall_c"}
        # ready_by 18:30 = 26 quarters after 12:00 -> state 26 = temps[25]
        assert nb["predicted_temp_c"] == pytest.approx(round(temps[25], 2))
        assert nb["shortfall_c"] == pytest.approx(round(max(0.0, 38.0 - temps[25]), 2))
        assert s["on_track"] is (max(0.0, 38.0 - temps[25]) < 0.5)
        assert datetime.fromisoformat(s["planned_at"]).tzinfo is not None

    def test_solver_report_wins(self):
        d = _decide(_cfg(shadow=False), _inputs([EVENING]))
        report = BathingReport(
            heater_id="spa",
            window_ready_temp_c=[37.9],
            window_shortfall_c=[0.1],
            unavoidable_shortfall_c=[0.0],
        )
        s = summarize(d, [0.0] * 96, report, _slots(96), NOW)
        assert s["mode"] == "solver"
        assert s["next_bath"]["predicted_temp_c"] == 37.9
        assert s["next_bath"]["shortfall_c"] == 0.1
        assert s["on_track"] is True

    def test_off_track(self):
        d = _decide(_cfg(shadow=False), _inputs([EVENING]))
        report = BathingReport("spa", [35.0], [3.0], unavoidable_shortfall_c=[0.0])
        assert summarize(d, None, report, _slots(96), NOW)["on_track"] is False

    def test_fallback_has_no_prediction_and_null_on_track(self):
        d = _decide(inputs=_inputs([EVENING], temp=None))
        s = summarize(d, None, None, _slots(96), NOW)
        assert set(s) == SUMMARY_KEYS
        assert s["mode"] == "fallback" and s["reason"] == "temp_unreadable"
        assert s["on_track"] is None
        assert s["next_bath"]["predicted_temp_c"] is None
        assert s["windows"] == [[EVENING[0].isoformat(), EVENING[1].isoformat()]]

    def test_in_progress_window_is_not_the_next_bath(self):
        running = (NOW - timedelta(minutes=30), NOW + timedelta(hours=1))
        d = _decide(inputs=_inputs([running, EVENING]))
        s = summarize(d, [38.0] * 96, None, _slots(96), NOW)
        assert s["next_bath"]["start"] == EVENING[0].isoformat()
        assert len(s["windows"]) == 2

    def test_no_windows(self):
        d = _decide(inputs=_inputs([]))
        s = summarize(d, [30.0] * 96, None, _slots(96), NOW)
        assert s["next_bath"] is None and s["windows"] == [] and s["on_track"] is True


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------


class _KInput:
    def __init__(self, slots: list[KeplerInputSlot], rem: float | None = None):
        self.slots = slots
        self.first_slot_remaining_h = rem


def _states(**over: Any) -> list[dict[str, Any]]:
    return [{"id": "spa", "heated_today_kwh": 0.0, "bathing_inputs": _inputs([EVENING], **over)}]


class TestDecideGlue:
    def test_no_bathing_block_touches_nothing(self):
        h = _heater()
        out = pl._decide_spa_bathing([{"id": "spa"}], _states(), [h], _KInput(_slots()), NOW, 4.0)
        assert out == {} and h.bathing is None

    def test_disabled_block_is_not_recorded(self):
        h = _heater()
        out = pl._decide_spa_bathing(
            [_cfg(enabled=False)], _states(), [h], _KInput(_slots()), NOW, 4.0
        )
        assert out == {} and h.bathing is None

    def test_shadow_leaves_the_solver_input_unchanged(self):
        h = _heater()
        out = pl._decide_spa_bathing([_cfg()], _states(), [h], _KInput(_slots()), NOW, 4.0)
        assert out["spa"].mode == "shadow"
        assert h.bathing is None

    def test_solver_mode_attaches_the_spec(self):
        h, other = _heater(), _heater(hid="main_tank")
        out = pl._decide_spa_bathing(
            [_cfg(shadow=False), {"id": "main_tank"}],
            _states(),
            [other, h],
            _KInput(_slots()),
            NOW,
            4.0,
        )
        assert out["spa"].mode == "solver"
        assert h.bathing is out["spa"].spec and h.bathing is not None
        assert other.bathing is None
        assert list(out) == ["spa"]

    def test_vacation_included_tank_keeps_its_floor(self):
        h = _heater()
        out = pl._decide_spa_bathing(
            [_cfg(shadow=False)],
            _states(),
            [h],
            _KInput(_slots()),
            NOW,
            4.0,
            vacation_included_ids={"spa"},
        )
        assert out["spa"].mode == "fallback" and out["spa"].reason == "vacation"
        assert h.bathing is None

    def test_vacation_excluded_tank_still_gets_its_spec(self):
        """exclude_from_vacation tanks keep normal comfort heating, so bathing too."""
        h = _heater()
        out = pl._decide_spa_bathing(
            [_cfg(shadow=False)],
            _states(),
            [h],
            _KInput(_slots()),
            NOW,
            4.0,
            vacation_included_ids={"main_tank"},
        )
        assert out["spa"].mode == "solver" and h.bathing is not None

    def test_heater_absent_from_the_plan_is_skipped(self):
        out = pl._decide_spa_bathing([_cfg()], _states(), [], _KInput(_slots()), NOW, 4.0)
        assert out == {}

    def test_missing_inputs_are_disabled_not_invented(self):
        h = _heater()
        out = pl._decide_spa_bathing(
            [_cfg(shadow=False)], [{"id": "spa"}], [h], _KInput(_slots()), NOW, 4.0
        )
        assert out == {} and h.bathing is None

    def test_a_crash_leaves_the_heater_legacy(self, monkeypatch):
        def boom(**_: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(pl, "decide_bathing", boom)
        h = _heater()
        assert pl._decide_spa_bathing([_cfg()], _states(), [h], _KInput(_slots()), NOW, 4.0) == {}
        assert h.bathing is None

    def test_block_sits_after_vacation_rebuild_and_before_the_solve(self):
        """The spec must be attached AFTER every pre-solve consumer of the heater list:
        the vacation rebuild re-creates heaters from config (dropping a spec), and the
        shortfall gate / excess-PV flags read min_kwh_per_day."""
        src = inspect.getsource(pl.PlannerPipeline.generate_schedule)
        at = src.index("bathing_decisions = _decide_spa_bathing(")
        assert src.index("_apply_water_shortfall_gate(") < at
        assert src.index("_calculate_excess_pv_flags(") < at
        assert src.index("kepler_config.water_heaters = list(excluded_heaters) + al_heaters") < at
        assert src.index("scale_first_slot_to_remaining(") < at
        assert at < src.index("solver = KeplerSolver()")
        assert src.index("_apply_spa_bathing_to_plan(") > src.index("final_df = future_df.join")


def _result(n: int = 8, kw: list[float] | None = None, temps: list[float] | None = None):
    kw = kw or [0.0] * n
    slots = [
        KeplerResultSlot(
            start_time=NOW + i * Q,
            end_time=NOW + (i + 1) * Q,
            charge_kwh=0.0,
            discharge_kwh=0.0,
            grid_import_kwh=0.1,
            grid_export_kwh=0.0,
            soc_kwh=5.0,
            cost_sek=0.1,
            water_heat_kw=kw[i],
            water_heater_results={"spa": kw[i], "main_tank": 0.0},
            water_heater_temp_c={"spa": temps[i]} if temps else {},
        )
        for i in range(n)
    ]
    return KeplerResult(slots=slots, total_cost_sek=1.0, is_optimal=True, status_msg="Optimal")


def _frames(result: KeplerResult):
    result_df = kepler_result_to_dataframe(result, 10.0, 5.0)
    final_df = result_df.copy()
    return final_df, result_df


class TestApplyToPlan:
    def test_no_decisions_drops_the_temperature_column(self):
        result = _result()
        final_df, result_df = _frames(result)
        before = [c for c in final_df.columns if c != "water_heater_temp_c"]
        meta = pl._apply_spa_bathing_to_plan(final_df, result_df, result, None, {}, None, NOW)
        assert meta is None
        assert list(final_df.columns) == before
        assert "water_heater_bathing" not in final_df.columns

    def test_shadow_simulates_and_marks_every_slot(self):
        kw = [0.0, 1.8, 1.8, 1.8, 0.0, 0.0, 0.0, 0.0]
        result = _result(kw=kw)
        final_df, result_df = _frames(result)
        d = _decide(inputs=_inputs([EVENING], temp=30.0, ambient=5.0))
        meta = pl._apply_spa_bathing_to_plan(
            final_df, result_df, result, {"plan_source": "kepler"}, {"spa": d}, 0.2, NOW
        )
        want = simulate_temps_c(
            30.0, 5.0, 0.021, 0.9, 1.8, [0.2] + [0.25] * 7, [0, 1, 1, 1, 0, 0, 0, 0]
        )
        got = [row["spa"] for row in final_df["water_heater_temp_c"]]
        assert got == pytest.approx(want)
        assert all(row == {"spa": "shadow"} for row in final_df["water_heater_bathing"])
        assert meta is not None and meta["mode"] == "shadow"

    def test_solver_mode_keeps_the_solvers_temperatures(self):
        temps = [30.0 + i for i in range(8)]
        result = _result(temps=temps)
        result.bathing_reports = {"spa": BathingReport("spa", [None], [None])}
        final_df, result_df = _frames(result)
        d = _decide(_cfg(shadow=False), _inputs([EVENING]))
        meta = pl._apply_spa_bathing_to_plan(
            final_df, result_df, result, {"plan_source": "kepler"}, {"spa": d}, None, NOW
        )
        assert [row["spa"] for row in final_df["water_heater_temp_c"]] == temps
        assert all(row == {"spa": "solver"} for row in final_df["water_heater_bathing"])
        assert meta is not None and meta["mode"] == "solver"

    def test_solver_mode_without_a_report_is_reported_as_fallback(self):
        result = _result()
        final_df, result_df = _frames(result)
        d = _decide(_cfg(shadow=False), _inputs([EVENING]))
        meta = pl._apply_spa_bathing_to_plan(
            final_df, result_df, result, {"plan_source": "kepler"}, {"spa": d}, None, NOW
        )
        assert meta is not None
        assert (meta["mode"], meta["reason"]) == ("fallback", "solver_dropped_spec")
        assert meta["on_track"] is None
        assert "water_heater_temp_c" not in final_df.columns
        assert all(row == {"spa": "fallback"} for row in final_df["water_heater_bathing"])

    @pytest.mark.parametrize("shadow", [True, False])
    def test_plan_guard_fallback(self, shadow):
        result = _result()
        final_df, result_df = _frames(result)
        d = _decide(_cfg(shadow=shadow), _inputs([EVENING]))
        meta = pl._apply_spa_bathing_to_plan(
            final_df, result_df, result, {"plan_source": "fallback"}, {"spa": d}, None, NOW
        )
        assert meta is not None
        assert (meta["mode"], meta["reason"]) == ("fallback", "plan_guard_fallback")
        assert "water_heater_temp_c" not in final_df.columns

    def test_fallback_decision(self):
        result = _result()
        final_df, result_df = _frames(result)
        d = _decide(inputs=_inputs(None))
        meta = pl._apply_spa_bathing_to_plan(
            final_df, result_df, result, None, {"spa": d}, None, NOW
        )
        assert meta is not None and meta["reason"] == "calendar_unknown"
        assert all(row == {"spa": "fallback"} for row in final_df["water_heater_bathing"])


# ---------------------------------------------------------------------------
# schedule.json output
# ---------------------------------------------------------------------------


def _df_for_formatter(**cols: list[Any]) -> pd.DataFrame:
    n = 4
    idx = pd.DatetimeIndex([NOW + i * Q for i in range(n)])
    data: dict[str, Any] = {
        "start_time": list(idx),
        "end_time": [t + Q for t in idx],
        "battery_charge_kw": [0.0] * n,
        "water_heaters": [{"spa": 0.0}] * n,
        "water_heating_boost": [{}] * n,
    }
    data.update(cols)
    return pd.DataFrame(data, index=idx)


class TestFormatter:
    def test_without_bathing_columns_the_records_are_unchanged(self):
        plain = dataframe_to_json_response(_df_for_formatter(), now_override=NOW)
        with_empty = dataframe_to_json_response(
            _df_for_formatter(
                water_heater_temp_c=[{}, float("nan"), None, {}],
                water_heater_bathing=[{}, float("nan"), {}, None],
            ),
            now_override=NOW,
        )
        assert with_empty == plain
        assert all("water_heater_temp_c" not in r for r in plain)
        assert all("water_heater_bathing" not in r for r in plain)

    def test_values_are_normalised(self):
        recs = dataframe_to_json_response(
            _df_for_formatter(
                water_heater_temp_c=[
                    {"spa": 31.2345},
                    {"spa": float("nan")},
                    {"spa": float("inf"), "x": 30},
                    {"spa": "hot", "y": True},
                ],
                water_heater_bathing=[{"spa": "shadow"}, {"spa": ""}, float("nan"), {"spa": 1}],
            ),
            now_override=NOW,
        )
        assert recs[0]["water_heater_temp_c"] == {"spa": 31.2}
        assert "water_heater_temp_c" not in recs[1]
        assert recs[2]["water_heater_temp_c"] == {"x": 30.0}
        assert "water_heater_temp_c" not in recs[3]
        assert recs[0]["water_heater_bathing"] == {"spa": "shadow"}
        assert all("water_heater_bathing" not in r for r in recs[1:])
        assert all(
            math.isfinite(v) for r in recs for v in r.get("water_heater_temp_c", {}).values()
        )


class TestScheduleMeta:
    @pytest.mark.asyncio
    async def test_meta_spa_bathing_is_written(self, tmp_path):
        import json

        from planner.output.schedule import save_schedule_to_json

        d = _decide(inputs=_inputs([EVENING]))
        meta = summarize(d, [30.0] * 96, None, _slots(96), NOW)
        out = tmp_path / "schedule.json"
        await save_schedule_to_json(
            _df_for_formatter(),
            {},
            pd.Timestamp(NOW),
            {},
            {},
            [],
            {},
            output_path=str(out),
            extra_meta={"plan_source": "kepler", "spa_bathing": meta},
        )
        written = json.loads(out.read_text())["meta"]["spa_bathing"]
        assert set(written) == SUMMARY_KEYS
        assert written["mode"] == "shadow"


class TestByteIdentical:
    def test_no_bathing_schedule_equals_the_pre_feature_schedule(self):
        """A result run through the bathing glue with no decisions serialises to the
        same schedule.json records as a result that never had the new column."""
        import json

        result = _result(kw=[0.0, 1.8, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        final_df, result_df = _frames(result)
        legacy_df = result_df.drop(columns=["water_heater_temp_c"])
        meta = pl._apply_spa_bathing_to_plan(final_df, result_df, result, None, {}, None, NOW)
        assert meta is None
        got = json.dumps(dataframe_to_json_response(final_df, now_override=NOW), default=str)
        want = json.dumps(dataframe_to_json_response(legacy_df, now_override=NOW), default=str)
        assert got == want


def test_pipeline_built_spec_round_trips_through_a_solver_dump():
    """The spec the pipeline builds (pytz-aware windows, DST night included) survives
    kepler's dump serialisation (asdict + json default=str) and the replay script's
    rehydrate_config, instant for instant."""
    import importlib.util
    import json
    from dataclasses import asdict
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "replay_solver_dump.py"
    mod_spec = importlib.util.spec_from_file_location("replay_solver_dump", path)
    assert mod_spec is not None and mod_spec.loader is not None
    replay = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(replay)

    from planner.solver.types import KeplerConfig

    now = TZ.localize(datetime(2026, 3, 28, 18, 0))
    dst_bath = (TZ.localize(datetime(2026, 3, 29, 3, 0)), TZ.localize(datetime(2026, 3, 29, 5, 0)))
    d = _decide(_cfg(shadow=False), _inputs([dst_bath]), now=now, slots=_slots(start=now))
    heater = _heater()
    heater.bathing = d.spec
    cfg = KeplerConfig(
        capacity_kwh=10.0,
        min_soc_percent=10.0,
        max_soc_percent=100.0,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        wear_cost_sek_per_kwh=0.0,
        water_heaters=[heater],
    )
    dumped = json.loads(json.dumps(asdict(cfg), default=str))
    back = replay.rehydrate_config(dumped)
    assert back.water_heaters[0].bathing == d.spec
    w = back.water_heaters[0].bathing.windows[0]
    assert w.ready_by == datetime(2026, 3, 29, 0, 30, tzinfo=UTC)
    assert w.start.tzinfo is not None
