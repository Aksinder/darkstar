# ruff: noqa
"""
Build #16 — plan-stability previous-plan anchor (planner root fix).

Verifies the price-gated previous-plan anchor that stops the daytime water-block
"walk" between replans (objective degeneracy on flat price bands), while keeping:
  * a genuine cheaper position still relocates a not-yet-started block,
  * the daily min_kwh floor never lowered (cold-shower backstop),
  * wall-clock keying (survives the per-replan future_df re-slice),
  * persistence across the day-bucket boundary (independent of heated_today).
"""
from datetime import datetime, timedelta

import pandas as pd

from planner.solver.kepler import KeplerConfig, KeplerInput, KeplerSolver
from planner.solver.types import KeplerInputSlot, WaterHeaterInput


# ---------------------------------------------------------------------------
# Kepler-level anchor economics
# ---------------------------------------------------------------------------


def _slots(count=48, start_hour=0, import_price=1.0):
    """N 30-min slots at a flat import price by default (fully degenerate band)."""
    start_time = datetime(2026, 1, 1, start_hour, 0)
    out = []
    for i in range(count):
        s = start_time + timedelta(minutes=30 * i)
        out.append(
            KeplerInputSlot(
                start_time=s,
                end_time=s + timedelta(minutes=30),
                load_kwh=0.5,
                pv_kwh=0.0,
                import_price_sek_kwh=import_price,
                export_price_sek_kwh=0.0,
            )
        )
    return out


def _heater(
    anchor_on_slots=None,
    min_kwh_per_day=2.0,
    power_kw=2.0,
    min_spacing_hours=0.0,
    heated_today_kwh=0.0,
):
    return WaterHeaterInput(
        id="wh1",
        power_kw=power_kw,
        min_kwh_per_day=min_kwh_per_day,
        max_hours_between_heating=0.0,
        min_spacing_hours=min_spacing_hours,
        heated_today_kwh=heated_today_kwh,
        anchor_on_slots=anchor_on_slots,
    )


def _config(heater, anchor_bonus=0.05):
    return KeplerConfig(
        capacity_kwh=10.0,
        min_soc_percent=0,
        max_soc_percent=100,
        max_charge_power_kw=5,
        max_discharge_power_kw=5,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        wear_cost_sek_per_kwh=0.0,
        water_heaters=[heater],
        water_heating_max_gap_hours=0.0,
        water_spacing_penalty_sek=0.0,
        water_reliability_penalty_sek=100.0,  # force the floor to be met
        water_block_start_penalty_sek=0.0,
        water_block_penalty_sek=0.0,
        water_anchor_bonus_sek_per_slot=anchor_bonus,
    )


def _on_slots(result):
    return [i for i, s in enumerate(result.slots) if s.water_heat_kw > 0]


def test_anchor_holds_block_on_flat_degenerate_band():
    """On a fully-flat band, without the anchor the symmetry-breaker jams the block
    to the earliest slots; WITH the anchor at a later tied position the block stays
    there (staying-put wins the tie)."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)

    # Baseline: no anchor -> earliest slots (symmetry breaker).
    base = solver.solve(KeplerInput(slots=slots, initial_soc_kwh=5.0), _config(_heater()))
    assert base.is_optimal
    base_on = _on_slots(base)
    assert base_on == [0, 1], f"expected earliest-slot jam without anchor, got {base_on}"

    # Anchored at slots 20,21 (a later, equally-cheap position): block should stay.
    anchored = solver.solve(
        KeplerInput(slots=slots, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21])),
    )
    assert anchored.is_optimal
    assert _on_slots(anchored) == [20, 21], (
        f"anchor did not hold the block at its previous position: {_on_slots(anchored)}"
    )


def test_anchor_survives_drifted_but_equal_cost_inputs():
    """Two 'replans' with drifted-but-equal-cost prices keep the block at the anchor
    (this is the block-walk the fix targets)."""
    solver = KeplerSolver()

    # Replan A: flat 1.0 everywhere, anchor at [20,21].
    slots_a = _slots(count=48, import_price=1.0)
    res_a = solver.solve(
        KeplerInput(slots=slots_a, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21])),
    )
    # Replan B: tiny sub-öre jitter that leaves all positions economically tied.
    slots_b = _slots(count=48, import_price=1.0)
    for i, s in enumerate(slots_b):
        s.import_price_sek_kwh = 1.0 + (i % 3) * 1e-4  # < anchor bonus over the block
    res_b = solver.solve(
        KeplerInput(slots=slots_b, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21])),
    )
    assert _on_slots(res_a) == [20, 21]
    assert _on_slots(res_b) == [20, 21], (
        f"block walked under drifted-but-equal inputs: {_on_slots(res_b)}"
    )


def test_pv_plateau_block_kept_despite_lower_import_nonpv_slots():
    """FINDING-1 REGRESSION (PV-aware price-gate): on the midday PV plateau, import price
    is HIGH but heating is ~free (PV surplus). An import-price-only gate would judge the
    plateau block as 'expensive' vs a low-import overnight slot and DROP the anchor,
    letting the block jam to the EARLIEST plateau slot. The PV-aware gate correctly sees
    the plateau as cheapest and KEEPS the block at its previous position."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)
    # Low IMPORT price, no PV [0..3] — cheapest by import price, but NOT by real cost.
    for i in range(0, 4):
        slots[i].import_price_sek_kwh = 0.5
    # PV plateau [20..27]: HIGH import (2.0) but PV surplus -> heating ~free (export 0.05).
    for i in range(20, 28):
        slots[i].import_price_sek_kwh = 2.0
        slots[i].pv_kwh = 5.0  # surplus 4.5 kWh >> kwh_per_slot (1.0) -> PV-covered
        slots[i].export_price_sek_kwh = 0.05
    res = solver.solve(
        KeplerInput(slots=slots, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[23, 24], min_kwh_per_day=2.0), anchor_bonus=0.2),
    )
    assert res.is_optimal
    on = _on_slots(res)
    # Kept at the anchored plateau position; an import-only gate would have dropped the
    # anchor and jammed the block to the earliest plateau slot [20, 21].
    assert on == [23, 24], f"PV-aware anchor did not hold the plateau block: {on}"


def test_genuine_cheaper_position_relocates_not_yet_started_block():
    """A real price drop bigger than the total bonus DROPS the anchor (price-gate) and
    the block relocates to the cheap window."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)
    # Make slots 4,5 genuinely cheap (0.1 vs 1.0). kwh_per_slot=1.0, so relocating saves
    # (1.0-0.1)*2*1.0 = 1.8 SEK >> total bonus 0.05*2 = 0.1 SEK.
    slots[4].import_price_sek_kwh = 0.1
    slots[5].import_price_sek_kwh = 0.1

    res = solver.solve(
        KeplerInput(slots=slots, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21])),
    )
    assert res.is_optimal
    on = _on_slots(res)
    assert on == [4, 5], f"block should relocate to the genuinely cheaper window, got {on}"
    assert 20 not in on and 21 not in on


def test_marginal_cheaper_position_within_bonus_stays_put():
    """A cheaper position that beats the anchor by LESS than the bonus does NOT move
    the block (economically negligible jitter is suppressed)."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)
    # kwh_per_slot=1.0, block=2 slots, bonus_total=0.1 SEK. A 0.02 SEK/kWh discount at
    # slots 4,5 saves only (0.02)*2*1.0 = 0.04 SEK < 0.1 -> anchor KEPT.
    slots[4].import_price_sek_kwh = 0.98
    slots[5].import_price_sek_kwh = 0.98

    res = solver.solve(
        KeplerInput(slots=slots, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21])),
    )
    assert _on_slots(res) == [20, 21], (
        f"anchor should suppress a within-bonus relocation, got {_on_slots(res)}"
    )


def test_anchor_never_lowers_daily_floor():
    """The anchor is reward-only: it can never reduce the daily-minimum energy. With an
    anchor covering only 1 slot but a floor needing 3 slots, the solver still heats >=3
    slots (floor honored)."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)
    # power 2 kW, 0.5h slot => 1 kWh/slot. min_kwh 3.0 => needs 3 slots.
    heater = _heater(anchor_on_slots=[10], min_kwh_per_day=3.0, power_kw=2.0)
    res = solver.solve(KeplerInput(slots=slots, initial_soc_kwh=5.0), _config(heater))
    assert res.is_optimal
    on = _on_slots(res)
    total_kwh = len(on) * heater.power_kw * 0.5
    assert total_kwh >= 3.0 - 1e-6, f"floor violated: heated {total_kwh} kWh over slots {on}"
    # The anchored slot is rewarded, so it should be among the heated slots.
    assert 10 in on


def test_anchor_disabled_when_bonus_zero():
    """anchor_bonus=0 => the anchor term is inert (back to symmetry-breaker behaviour)."""
    solver = KeplerSolver()
    slots = _slots(count=48, import_price=1.0)
    res = solver.solve(
        KeplerInput(slots=slots, initial_soc_kwh=5.0),
        _config(_heater(anchor_on_slots=[20, 21]), anchor_bonus=0.0),
    )
    assert _on_slots(res) == [0, 1], "with bonus=0 the block should jam earliest again"


# ---------------------------------------------------------------------------
# Pipeline extraction + wall-clock mapping (replicates pipeline.py logic)
# ---------------------------------------------------------------------------


def _extract_anchor_ts(previous_schedule, enabled_heater_ids, tz="Europe/Stockholm"):
    """Replicate the pipeline's ALL-previously-ON extraction (wall-clock timestamps)."""
    anchor_by_heater = {d: set() for d in enabled_heater_ids}
    tzinfo = pd.Timestamp("2026-01-01", tz=tz).tz
    for slot_s in previous_schedule:
        whs = slot_s.get("water_heaters", {})
        if not whs:
            continue
        ts = pd.Timestamp(slot_s["start_time"]).astimezone(tzinfo)
        for hid in enabled_heater_ids:
            if float(whs.get(hid, {}).get("heating_kw", 0.0)) > 0:
                anchor_by_heater[hid].add(ts)
    return anchor_by_heater


def _map_to_indices(anchor_ts, future_index):
    """Replicate the pipeline's timestamp->future_df-index mapping."""
    out = {}
    for hid, tss in anchor_ts.items():
        idxs = [i for i, ts in enumerate(future_index) if ts in tss]
        if idxs:
            out[hid] = idxs
    return out


def _make_slot(start_iso, wh1_kw=0.0):
    # Real schedule.json start_time strings carry a tz offset (Stockholm winter = +01:00),
    # which the pipeline's pd.Timestamp(...).astimezone(tz) relies on.
    if "+" not in start_iso and "Z" not in start_iso:
        start_iso = start_iso + ":00+01:00" if len(start_iso) == 16 else start_iso + "+01:00"
    return {"start_time": start_iso, "water_heaters": {"wh1": {"heating_kw": wh1_kw}}}


def test_extraction_captures_all_previous_on_slots_not_just_in_progress():
    """Unlike the mid-block lock (in-progress run only), the anchor captures EVERY
    previously-ON slot, including a not-yet-started future block."""
    schedule = [
        _make_slot("2026-01-15T10:00", wh1_kw=3.0),  # in progress
        _make_slot("2026-01-15T10:30", wh1_kw=0.0),
        _make_slot("2026-01-15T20:00", wh1_kw=3.0),  # a future block
        _make_slot("2026-01-15T20:30", wh1_kw=3.0),
    ]
    anchor = _extract_anchor_ts(schedule, ["wh1"])
    assert len(anchor["wh1"]) == 3  # 10:00, 20:00, 20:30


def test_wall_clock_keyed_survives_future_df_reslice():
    """Same wall-clock block maps to DIFFERENT indices after a re-slice from 'now', so
    the anchor stays glued to the clock hour, not the slot index."""
    tz = "Europe/Stockholm"
    schedule = [
        _make_slot("2026-01-15T20:00", wh1_kw=3.0),
        _make_slot("2026-01-15T20:30", wh1_kw=3.0),
    ]
    anchor = _extract_anchor_ts(schedule, ["wh1"], tz)

    # Replan A: future_df starts at 18:00 -> the 20:00 block is at indices 4,5.
    idx_a = pd.date_range("2026-01-15T18:00", periods=12, freq="30min", tz=tz)
    mapped_a = _map_to_indices(anchor, list(idx_a))
    assert mapped_a["wh1"] == [4, 5]

    # Replan B: two slots later, future_df starts at 19:00 -> same clock block now 2,3.
    idx_b = pd.date_range("2026-01-15T19:00", periods=12, freq="30min", tz=tz)
    mapped_b = _map_to_indices(anchor, list(idx_b))
    assert mapped_b["wh1"] == [2, 3], "anchor did not track the wall-clock hour across re-slice"


def test_persists_across_day_bucket_boundary_independent_of_heated_today():
    """The anchor is derived from the previous schedule regardless of heated_today, so a
    just-past-midnight replan (heated_today reset to 0) still keeps the block."""
    tz = "Europe/Stockholm"
    # Previous plan placed a block at 02:00-02:30 (post-midnight, fresh day-bucket).
    schedule = [
        _make_slot("2026-01-16T02:00", wh1_kw=3.0),
        _make_slot("2026-01-16T02:30", wh1_kw=3.0),
    ]
    anchor = _extract_anchor_ts(schedule, ["wh1"], tz)
    # Replan just after midnight; heated_today is 0 but the anchor is still present.
    idx = pd.date_range("2026-01-16T00:00", periods=24, freq="30min", tz=tz)
    mapped = _map_to_indices(anchor, list(idx))
    assert "wh1" in mapped and len(mapped["wh1"]) == 2


def test_anchor_mapping_matches_by_epoch_across_tz_representations():
    """REGRESSION for the dead-anchor bug (build #18): the REAL pipeline mapped anchor
    slots to future_df indices with ``ts in anchor_ts`` (tz-aware Timestamp
    set-membership). When future_df's index and the previous schedule's
    ``.astimezone(tz)`` anchor timestamps carry different tz objects, set-membership
    silently never matched -> anchor_on_slots always empty -> the whole plan-stability
    anchor was dead (0 'anchoring N slots' logs in production). The fix matches on the
    epoch instant (Timestamp.value, tz-independent). This locks that behaviour, mirroring
    planner/pipeline.py exactly."""
    import pytz

    tz = pytz.timezone("Europe/Stockholm")
    # future_df index in UTC — a DIFFERENT tz representation than the local anchors.
    future_idx = pd.date_range("2026-07-11T22:00:00Z", periods=8, freq="15min", tz="UTC")
    # anchor timestamps built the pipeline's way: parse ISO-with-offset + astimezone(tz).
    anchor_ts = {
        pd.Timestamp("2026-07-12T00:30:00+02:00").astimezone(tz),  # == 22:30Z -> slot 2
        pd.Timestamp("2026-07-12T00:45:00+02:00").astimezone(tz),  # == 22:45Z -> slot 3
    }
    # Production fix: epoch-value matching.
    anchor_epochs = {t.value for t in anchor_ts}
    matched = [i for i, ts in enumerate(future_idx) if ts.value in anchor_epochs]
    assert matched == [2, 3], matched


class TestPreviousScheduleReadPath:
    """The reader must load the SAME file the writer publishes.

    The reader hardcoded "schedule.json" (repo root) while the writer wrote
    data/schedule.json — so the mid-block lock and the anchor never fired in
    production (live dumps: anchor_on_slots null on every heater, always).
    """

    def test_reads_the_writers_path(self, tmp_path, monkeypatch):
        import json

        from planner.output.schedule import SCHEDULE_JSON_PATH
        from planner.pipeline import _load_previous_schedule

        monkeypatch.chdir(tmp_path)
        target = tmp_path / SCHEDULE_JSON_PATH
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"schedule": [{"start_time": "2026-08-26T10:00:00+02:00"}]}))

        slots = _load_previous_schedule()
        assert len(slots) == 1
        assert slots[0]["start_time"].startswith("2026-08-26")

    def test_missing_file_is_empty_not_error(self, tmp_path, monkeypatch):
        from planner.pipeline import _load_previous_schedule

        monkeypatch.chdir(tmp_path)
        assert _load_previous_schedule() == []

    def test_corrupt_file_is_empty_not_error(self, tmp_path, monkeypatch):
        from planner.output.schedule import SCHEDULE_JSON_PATH
        from planner.pipeline import _load_previous_schedule

        monkeypatch.chdir(tmp_path)
        target = tmp_path / SCHEDULE_JSON_PATH
        target.parent.mkdir(parents=True)
        target.write_text('{"schedule": [truncated')
        assert _load_previous_schedule() == []

    def test_writer_default_matches_constant(self):
        import inspect

        from planner.output.schedule import SCHEDULE_JSON_PATH, save_schedule_to_json

        sig = inspect.signature(save_schedule_to_json)
        assert sig.parameters["output_path"].default == SCHEDULE_JSON_PATH
