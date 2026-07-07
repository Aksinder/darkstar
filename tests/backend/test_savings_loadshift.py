"""Tests for the load-shift savings streams (backend/learning/savings_loadshift.py)."""

import json
from datetime import datetime

import pytest
import pytz

from backend.learning.cycle_publisher import build_loadshift_sensors
from backend.learning.savings_loadshift import (
    WATER_BASELINE_DAILY_AVERAGE,
    ApplianceShiftSummary,
    CycleLedgerEntry,
    LoadshiftSummary,
    MeasuredRun,
    WaterShiftSummary,
    compute_appliance_shift,
    compute_water_shift,
    dedupe_ledger,
    enrich_cycle_ledger,
    load_cycle_ledger,
    marginal_price,
    water_unattributed_kwh,
)

TZ = pytz.timezone("Europe/Stockholm")
DAY = datetime(2026, 7, 6)


def _slot_start(hour, quarter=0, day=DAY):
    return TZ.localize(day.replace(hour=hour, minute=quarter * 15)).isoformat()


def _row(hour, quarter=0, imp=1.0, exp=0.0, p_imp=2.0, p_exp=0.5, day=DAY):
    return {
        "slot_start": _slot_start(hour, quarter, day),
        "import_kwh": imp,
        "export_kwh": exp,
        "import_price_sek_kwh": p_imp,
        "export_price_sek_kwh": p_exp,
    }


def _day_rows(price_by_hour, imp=1.0, exp=0.0, p_exp=0.5, day=DAY):
    """A full local day of 15-min observation rows with per-hour import prices."""
    return [
        _row(h, q, imp=imp, exp=exp, p_imp=price_by_hour(h), p_exp=p_exp, day=day)
        for h in range(24)
        for q in range(4)
    ]


def _prices(h):
    """Hours 2-5 cheap (0.2..0.35), the rest climb with the hour."""
    cheap = {2: 0.2, 3: 0.25, 4: 0.3, 5: 0.35}
    return cheap.get(h, 1.0 + h * 0.1)


class TestMarginalPrice:
    def test_net_import_uses_import_price(self):
        assert marginal_price(_row(12, imp=1.0, exp=0.0, p_imp=2.0, p_exp=0.1)) == 2.0

    def test_net_export_uses_export_price(self):
        assert marginal_price(_row(12, imp=0.0, exp=2.0, p_imp=2.0, p_exp=0.1)) == 0.1

    def test_tie_including_islanded_uses_import_price(self):
        assert marginal_price(_row(12, imp=0.0, exp=0.0, p_imp=2.0, p_exp=0.1)) == 2.0

    def test_missing_needed_price_is_none_not_zero(self):
        row = _row(12, imp=0.0, exp=2.0)
        row["export_price_sek_kwh"] = None
        assert marginal_price(row) is None

    def test_negative_export_price_passes_through_raw(self):
        assert marginal_price(_row(12, imp=0.0, exp=2.0, p_exp=-0.05)) == -0.05


def _dev(hour, kwh, quarter=0, tank="main_tank", day=DAY):
    return {"slot_start": _slot_start(hour, quarter, day), "device_id": tank, "kwh": kwh}


class TestWaterFourCheapest:
    def test_expensive_midday_import_heating_goes_negative(self):
        """Darkstar heating 2 kWh at an expensive import hour must LOSE against the
        old 4-cheapest-hours automation — unclamped negative credit."""
        rows = _day_rows(_prices)
        devs = [_dev(12, 1.0), _dev(12, 1.0, quarter=1)]
        s = compute_water_shift(rows, devs, tank_id="main_tank", element_power_kw=3.4)
        # actual: 2 kWh at hour-12 import price 2.2 = 4.4
        assert s.actual_cost_sek == pytest.approx(4.4)
        # baseline: 2 kWh fits in hour 2 (cap 0.85/slot * 4 slots = 3.4) at 0.2
        assert s.baseline_cost_sek == pytest.approx(0.4)
        assert s.credit_sek == pytest.approx(-4.0)
        assert s.coverage == 1.0
        assert s.n_days == 1

    def test_pv_boost_heating_credits_at_export_price(self):
        """Heating during net-export slots only forgoes export revenue — the
        marginal rule embeds the PV-boost credit with no separate formula."""
        rows = _day_rows(_prices)
        # Hour 12 is net-exporting with a 0.1 SEK/kWh export price.
        for r in rows:
            if r["slot_start"].startswith(_slot_start(12)[:13]):
                r["import_kwh"], r["export_kwh"], r["export_price_sek_kwh"] = 0.0, 3.0, 0.1
        devs = [_dev(12, 1.0), _dev(12, 1.0, quarter=1)]
        s = compute_water_shift(rows, devs, tank_id="main_tank", element_power_kw=3.4)
        assert s.actual_cost_sek == pytest.approx(0.2)  # 2 kWh * 0.1
        assert s.baseline_cost_sek == pytest.approx(0.4)
        assert s.credit_sek == pytest.approx(0.2)

    def test_element_power_cap_prevents_strawman_baseline(self):
        """5 kWh cannot all 'heat' in the single cheapest hour: 3.4 kWh caps out
        hour 2, the rest spills to hour 3 at its (higher) price."""
        rows = _day_rows(_prices)
        devs = [_dev(12, 1.25, quarter=q) for q in range(4)]  # 5 kWh
        s = compute_water_shift(rows, devs, tank_id="main_tank", element_power_kw=3.4)
        assert s.baseline_cost_sek == pytest.approx(3.4 * 0.2 + 1.6 * 0.25)

    def test_villavagn_element_cap_is_lower(self):
        """1.6 kW element: one cheap hour only holds 1.6 kWh."""
        rows = _day_rows(_prices)
        devs = [
            _dev(12, 0.8, quarter=0, tank="villavagn_tank"),
            _dev(12, 0.8, 1, "villavagn_tank"),
            _dev(12, 0.8, 2, "villavagn_tank"),
        ]  # 2.4 kWh
        s = compute_water_shift(rows, devs, tank_id="villavagn_tank", element_power_kw=1.6)
        assert s.baseline_cost_sek == pytest.approx(1.6 * 0.2 + 0.8 * 0.25)

    def test_only_matching_tank_rows_count(self):
        rows = _day_rows(_prices)
        devs = [_dev(12, 1.0), _dev(12, 1.0, quarter=1, tank="villavagn_tank")]
        s = compute_water_shift(rows, devs, tank_id="main_tank", element_power_kw=3.4)
        assert s.valued_kwh == pytest.approx(1.0)


class TestWaterDailyAverage:
    def test_baseline_is_day_average_import_price(self):
        rows = _day_rows(_prices)
        devs = [_dev(2, 1.0, tank="villavagn_tank")]  # heated at the cheap hour
        s = compute_water_shift(
            rows,
            devs,
            tank_id="villavagn_tank",
            element_power_kw=1.6,
            baseline=WATER_BASELINE_DAILY_AVERAGE,
        )
        avg = sum(_prices(h) for h in range(24)) / 24
        assert s.baseline_name == WATER_BASELINE_DAILY_AVERAGE
        # Summaries round to 4 decimals.
        assert s.baseline_cost_sek == pytest.approx(avg, abs=1e-3)
        assert s.actual_cost_sek == pytest.approx(0.2)
        assert s.credit_sek == pytest.approx(avg - 0.2, abs=1e-3)

    def test_unknown_baseline_fails_loud(self):
        with pytest.raises(ValueError):
            compute_water_shift([], [], tank_id="x", element_power_kw=3.4, baseline="thermostat")


class TestWaterCoverage:
    def test_unpriced_heating_slot_counts_as_unvalued(self):
        rows = _day_rows(_prices)
        # Hour 12's prices are missing entirely.
        for r in rows:
            if r["slot_start"].startswith(_slot_start(12)[:13]):
                r["import_price_sek_kwh"] = None
                r["export_price_sek_kwh"] = None
        devs = [_dev(12, 1.0), _dev(2, 1.0)]
        s = compute_water_shift(rows, devs, tank_id="main_tank", element_power_kw=3.4)
        assert s.valued_kwh == pytest.approx(1.0)
        assert s.unvalued_kwh == pytest.approx(1.0)
        assert s.coverage == pytest.approx(0.5)
        # Baseline only reconstructs the VALUED energy — no phantom credit.
        assert s.baseline_cost_sek == pytest.approx(1.0 * 0.2)

    def test_day_with_no_prices_is_fully_unvalued(self):
        rows = [
            {
                "slot_start": _slot_start(h, q),
                "import_kwh": 1.0,
                "export_kwh": 0.0,
                "import_price_sek_kwh": None,
                "export_price_sek_kwh": None,
            }
            for h in range(24)
            for q in range(4)
        ]
        s = compute_water_shift(rows, [_dev(12, 2.0)], tank_id="main_tank", element_power_kw=3.4)
        assert s.valued_kwh == 0.0
        assert s.unvalued_kwh == pytest.approx(2.0)
        assert s.credit_sek == 0.0

    def test_idle_tank_has_full_coverage_and_zero_credit(self):
        s = compute_water_shift(_day_rows(_prices), [], tank_id="main_tank", element_power_kw=3.4)
        assert s.credit_sek == 0.0
        assert s.coverage == 1.0
        assert s.n_days == 0


class TestWaterUnattributed:
    """Aggregate-vs-device cross-check: water energy with no (or a zero-wiped)
    device row is invisible to compute_water_shift (it only sees kwh>0 rows), so
    the shortfall must be COUNTED as unvalued instead of coverage staying 1.0
    while the tank credit silently shrinks."""

    def _obs(self, hour, water, quarter=0):
        return {"slot_start": _slot_start(hour, quarter), "water_kwh": water}

    def test_missing_device_row_counts_as_shortfall(self):
        obs = [self._obs(12, 0.8), self._obs(13, 0.4)]
        devs = [_dev(13, 0.4)]
        assert water_unattributed_kwh(obs, devs, ["main_tank"]) == pytest.approx(0.8)

    def test_zero_wiped_device_row_counts_as_shortfall(self):
        obs = [self._obs(12, 0.8)]
        devs = [_dev(12, 0.0)]
        assert water_unattributed_kwh(obs, devs, ["main_tank"]) == pytest.approx(0.8)

    def test_fully_attributed_window_has_no_shortfall(self):
        obs = [self._obs(12, 0.8)]
        devs = [_dev(12, 0.5), _dev(12, 0.3, tank="villavagn_tank")]
        assert water_unattributed_kwh(obs, devs, ["main_tank", "villavagn_tank"]) == 0.0

    def test_ev_device_rows_do_not_cover_water_energy(self):
        obs = [self._obs(12, 0.8)]
        devs = [{"slot_start": _slot_start(12), "device_id": "easee", "kwh": 0.8}]
        assert water_unattributed_kwh(obs, devs, ["main_tank"]) == pytest.approx(0.8)

    def test_shortfall_surfaces_as_zero_coverage_bucket(self):
        """The publisher exposes the shortfall as a synthetic zero-credit summary
        whose coverage drops below 1.0 — the honest signal for wiped slots."""
        s = WaterShiftSummary(
            tank_id="unattributed_water",
            baseline_name="uncovered",
            actual_cost_sek=0.0,
            baseline_cost_sek=0.0,
            credit_sek=0.0,
            valued_kwh=0.0,
            unvalued_kwh=0.8,
            n_days=0,
        )
        assert s.coverage == 0.0


# ---------------------------------------------------------------------------
# Appliances
# ---------------------------------------------------------------------------

T0 = TZ.localize(datetime(2026, 7, 6, 17, 0)).timestamp()
H = 3600.0


def _flat_slots(start_ts, hours, price_fn, imp=1.0, exp=0.0, p_exp=0.5):
    """15-min net-importing rows for ``hours`` from start_ts; price_fn(hour_index)."""
    rows = []
    for i in range(int(hours * 4)):
        ts = start_ts + i * 900.0
        rows.append(
            {
                "slot_start": datetime.fromtimestamp(ts, TZ).isoformat(),
                "import_kwh": imp,
                "export_kwh": exp,
                "import_price_sek_kwh": price_fn(int(i // 4)),
                "export_price_sek_kwh": p_exp,
            }
        )
    return rows


def _entry(armed, done, load_id="washer", held=True, kwh=None):
    return CycleLedgerEntry(
        load_id=load_id,
        armed_ts=armed,
        done_ts=done,
        held_by_us_ever=held,
        measured_kwh=kwh,
    )


class TestApplianceShift:
    def test_deferred_cycle_to_cheap_window_earns_credit(self):
        # Expensive for 2 h from arm, cheap after.
        rows = _flat_slots(T0, 8, lambda h: 2.0 if h < 2 else 0.5)
        entry = _entry(T0, T0 + 4 * H + 600)
        run = MeasuredRun(T0 + 3 * H, T0 + 4 * H, 1.0)  # 1 kWh, 1 h, in the cheap zone
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.baseline_cost_sek == pytest.approx(2.0)  # 1 kWh @ arm-window 2.0
        assert s.actual_cost_sek == pytest.approx(0.5)
        assert s.credit_sek == pytest.approx(1.5)
        assert s.n_valued_cycles == 1
        assert s.coverage == 1.0

    def test_deadline_pressure_run_costing_more_goes_negative(self):
        """Armed cheap, forced to run expensive (deadline/fail-open): negative."""
        rows = _flat_slots(T0, 8, lambda h: 0.5 if h < 2 else 2.0)
        entry = _entry(T0, T0 + 4 * H + 600)
        run = MeasuredRun(T0 + 3 * H, T0 + 4 * H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.credit_sek == pytest.approx(0.5 - 2.0)
        assert s.credit_sek < 0

    def test_never_deferred_cycle_scores_exactly_zero(self):
        """Run window == arm window: both sides use the marginal rule, so the
        credit is exactly 0 — not a phantom positive from an import-vs-marginal
        asymmetry."""
        rows = _flat_slots(T0, 8, lambda h: 2.0 if h < 1 else 0.5, imp=0.0, exp=2.0)
        entry = _entry(T0, T0 + H + 300, held=False)
        run = MeasuredRun(T0, T0 + H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.credit_sek == pytest.approx(0.0)

    def test_actual_valued_at_export_price_when_net_exporting(self):
        rows = _flat_slots(T0, 4, lambda h: 2.0)  # net-importing arm window
        # Run window net-exporting at 0.1 export price.
        for r in rows[8:12]:
            r["import_kwh"], r["export_kwh"], r["export_price_sek_kwh"] = 0.0, 2.0, 0.1
        entry = _entry(T0, T0 + 3 * H)
        run = MeasuredRun(T0 + 2 * H, T0 + 3 * H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.baseline_cost_sek == pytest.approx(2.0)
        assert s.actual_cost_sek == pytest.approx(0.1)

    def test_unmeasured_cycle_is_counted_not_credited(self):
        rows = _flat_slots(T0, 8, lambda h: 2.0)
        entry = _entry(T0, T0 + H)
        s = compute_appliance_shift([entry], [], rows, load_id="washer")
        assert s.credit_sek == 0.0
        assert s.unvalued_cycles == 1
        assert s.coverage == 0.0

    def test_window_not_fully_priced_makes_cycle_unvalued(self):
        rows = _flat_slots(T0, 8, lambda h: 2.0)
        rows[1]["import_price_sek_kwh"] = None  # hole inside the arm window
        entry = _entry(T0, T0 + 2 * H)
        run = MeasuredRun(T0 + H, T0 + 2 * H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.unvalued_cycles == 1
        assert s.n_valued_cycles == 0

    def test_credited_on_completion_day_only(self):
        """Cross-midnight cycles land in the window done_ts falls in — once."""
        rows = _flat_slots(T0, 10, lambda h: 1.0)
        midnight = T0 + 7 * H  # pretend window boundary
        entry = _entry(T0 + 5 * H, T0 + 8 * H)  # done AFTER the boundary
        run = MeasuredRun(T0 + 7 * H, T0 + 8 * H, 1.0)
        before = compute_appliance_shift(
            [entry], [run], rows, load_id="washer", window_end_ts=midnight
        )
        after = compute_appliance_shift(
            [entry], [run], rows, load_id="washer", window_start_ts=midnight
        )
        assert before.n_valued_cycles + before.unvalued_cycles == 0
        assert after.n_valued_cycles == 1

    def test_ledger_measured_kwh_takes_priority_over_run_energy(self):
        rows = _flat_slots(T0, 4, lambda h: 1.0)
        entry = _entry(T0, T0 + 2 * H, kwh=2.0)
        run = MeasuredRun(T0 + H, T0 + 2 * H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.baseline_cost_sek == pytest.approx(2.0)  # 2 kWh @ 1.0

    def test_other_load_ids_are_ignored(self):
        rows = _flat_slots(T0, 4, lambda h: 1.0)
        entry = _entry(T0, T0 + H, load_id="dishwasher")
        run = MeasuredRun(T0, T0 + H, 1.0)
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.n_valued_cycles + s.unvalued_cycles == 0

    def test_unheld_chain_merged_reload_cannot_fabricate_credit(self):
        """Kept-after-done chains can hand a back-to-back reload the PREVIOUS
        programme's armed_ts. A cycle Darkstar never held must score exactly 0
        by construction, even when its inherited arm anchor points at a
        pricier window than the actual run."""
        rows = _flat_slots(T0, 8, lambda h: 2.0 if h < 2 else 0.5)
        entry = _entry(T0, T0 + 3 * H + 300, held=False)  # inherited anchor at 2.0-zone
        run = MeasuredRun(T0 + 2 * H, T0 + 3 * H, 1.0)  # never-deferred reload at 0.5
        s = compute_appliance_shift([entry], [run], rows, load_id="washer")
        assert s.credit_sek == pytest.approx(0.0)
        assert s.baseline_cost_sek == pytest.approx(s.actual_cost_sek)
        assert s.actual_cost_sek == pytest.approx(0.5)
        assert s.n_valued_cycles == 1

    def test_old_cycle_priced_from_persisted_run_when_detection_aged_out(self):
        """A cycle older than the 14-day detection horizon has no live
        MeasuredRun; the enriched ledger fields must keep it priceable so the
        30d credit does not structurally decay."""
        rows = _flat_slots(T0, 8, lambda h: 2.0 if h < 2 else 0.5)
        entry = CycleLedgerEntry(
            load_id="washer",
            armed_ts=T0,
            done_ts=T0 + 4 * H + 600,
            held_by_us_ever=True,
            measured_kwh=1.0,
            run_start_ts=T0 + 3 * H,
            run_end_ts=T0 + 4 * H,
        )
        s = compute_appliance_shift([entry], [], rows, load_id="washer")
        assert s.n_valued_cycles == 1
        assert s.credit_sek == pytest.approx(1.5)

    def test_persisted_window_without_energy_stays_unvalued(self):
        """Partial enrichment must never invent energy (honesty contract)."""
        rows = _flat_slots(T0, 8, lambda h: 1.0)
        entry = CycleLedgerEntry(
            load_id="washer",
            armed_ts=T0,
            done_ts=T0 + 2 * H,
            held_by_us_ever=True,
            run_start_ts=T0 + H,
            run_end_ts=T0 + 2 * H,
        )
        s = compute_appliance_shift([entry], [], rows, load_id="washer")
        assert s.unvalued_cycles == 1
        assert s.n_valued_cycles == 0


class TestLedger:
    def test_dedupe_collapses_continuations_keeping_last_done(self):
        a = _entry(T0, T0 + H)
        b = _entry(T0, T0 + 2 * H)  # continuation of the same programme
        c = _entry(T0 + 5 * H, T0 + 6 * H)  # a new cycle
        out = dedupe_ledger([a, b, c])
        assert len(out) == 2
        assert out[0].done_ts == T0 + 2 * H
        assert out[1].armed_ts == T0 + 5 * H

    def test_continuation_counts_as_one_cycle_in_credit(self):
        rows = _flat_slots(T0, 8, lambda h: 1.0)
        a = _entry(T0, T0 + H)
        b = _entry(T0, T0 + 2 * H)
        run = MeasuredRun(T0, T0 + 2 * H, 1.0)
        s = compute_appliance_shift([a, b], [run], rows, load_id="washer")
        assert s.n_valued_cycles == 1

    def test_load_cycle_ledger_reads_jsonl_and_skips_corrupt_lines(self, tmp_path):
        p = tmp_path / "cycles.jsonl"
        p.write_text(
            '{"load_id": "washer", "armed_ts": 100.0, "done_ts": 200.0, '
            '"held_by_us_ever": true, "deadline_ts": null, "measured_kwh": null}\n'
            "not json at all\n"
            '{"load_id": "washer", "armed_ts": "nan-ish"}\n',
            encoding="utf-8",
        )
        entries = load_cycle_ledger(p)
        assert len(entries) == 1
        assert entries[0].load_id == "washer"
        assert entries[0].held_by_us_ever is True
        assert entries[0].deadline_ts is None

    def test_load_cycle_ledger_missing_file(self, tmp_path):
        assert load_cycle_ledger(tmp_path / "nope.jsonl") == []

    def test_load_cycle_ledger_parses_persisted_run_window(self, tmp_path):
        p = tmp_path / "cycles.jsonl"
        p.write_text(
            json.dumps(
                {
                    "load_id": "washer",
                    "armed_ts": 100.0,
                    "done_ts": 200.0,
                    "measured_kwh": 1.2,
                    "run_start_ts": 120.0,
                    "run_end_ts": 180.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (entry,) = load_cycle_ledger(p)
        assert entry.measured_kwh == pytest.approx(1.2)
        assert entry.run_start_ts == pytest.approx(120.0)
        assert entry.run_end_ts == pytest.approx(180.0)


class TestLedgerEnrichment:
    """enrich_cycle_ledger: persist the run join at the first publish after done
    so cycles stay priceable after the HA detection history ages out."""

    def _raw(self, **over):
        row = {
            "load_id": "washer",
            "armed_ts": T0,
            "done_ts": T0 + 2 * H,
            "held_by_us_ever": True,
            "deadline_ts": None,
            "measured_kwh": None,
        }
        row.update(over)
        return row

    def test_enrich_persists_energy_and_run_window(self, tmp_path):
        p = tmp_path / "cycles.jsonl"
        p.write_text(json.dumps(self._raw()) + "\nnot json at all\n", encoding="utf-8")
        runs = {"washer": [MeasuredRun(T0 + H, T0 + 2 * H, 1.2)]}
        (entry,) = enrich_cycle_ledger(p, runs)
        assert entry.measured_kwh == pytest.approx(1.2)
        assert entry.run_start_ts == pytest.approx(T0 + H)
        assert entry.run_end_ts == pytest.approx(T0 + 2 * H)
        # Persisted: a later tick with NO live runs still sees the join.
        (reloaded,) = load_cycle_ledger(p)
        assert reloaded.measured_kwh == pytest.approx(1.2)
        assert reloaded.run_end_ts == pytest.approx(T0 + 2 * H)
        # Corrupt lines survive the rewrite verbatim.
        assert "not json at all" in p.read_text(encoding="utf-8")

    def test_enrich_is_idempotent(self, tmp_path):
        p = tmp_path / "cycles.jsonl"
        p.write_text(json.dumps(self._raw()) + "\n", encoding="utf-8")
        enrich_cycle_ledger(p, {"washer": [MeasuredRun(T0 + H, T0 + 2 * H, 1.2)]})
        first = p.read_text(encoding="utf-8")
        # A different overlapping run later must NOT overwrite the filled join.
        enrich_cycle_ledger(p, {"washer": [MeasuredRun(T0, T0 + 2 * H, 9.9)]})
        assert p.read_text(encoding="utf-8") == first

    def test_enrich_without_match_leaves_file_untouched(self, tmp_path):
        p = tmp_path / "cycles.jsonl"
        original = json.dumps(self._raw()) + "\n"
        p.write_text(original, encoding="utf-8")
        (entry,) = enrich_cycle_ledger(p, {})
        assert entry.measured_kwh is None
        assert p.read_text(encoding="utf-8") == original

    def test_enrich_missing_file_returns_empty(self, tmp_path):
        assert enrich_cycle_ledger(tmp_path / "nope.jsonl", {}) == []


class TestSensors:
    def _summary(self):
        water = (
            WaterShiftSummary(
                tank_id="main_tank",
                baseline_name="four_cheapest_hours",
                actual_cost_sek=1.0,
                baseline_cost_sek=3.0,
                credit_sek=2.0,
                valued_kwh=4.0,
                unvalued_kwh=1.0,
                n_days=1,
            ),
        )
        apps = (
            ApplianceShiftSummary(
                load_id="washer",
                actual_cost_sek=0.5,
                baseline_cost_sek=0.2,
                credit_sek=-0.3,
                n_valued_cycles=1,
                unvalued_cycles=2,
            ),
        )
        return LoadshiftSummary(water=water, appliances=apps)

    def test_builds_both_sensors_with_breakdown(self):
        s = self._summary()
        sensors = build_loadshift_sensors(s, s)
        assert [x.object_id for x in sensors] == [
            "darkstar_loadshift_today",
            "darkstar_loadshift_30d",
        ]
        today = sensors[0]
        assert today.state == "1.7"  # 2.0 - 0.3, negatives unclamped in the total
        attrs = today.attributes
        assert attrs["main_tank_sek"] == 2.0
        assert "4 cheapest" in attrs["main_tank_baseline"]
        assert attrs["main_tank_coverage"] == 0.8
        assert attrs["washer_sek"] == -0.3
        assert attrs["washer_unvalued_cycles"] == 2
        assert "EV" in attrs["ev_note"]
        assert "darkstar_savings_" in attrs["battery_note"]

    def test_none_windows_are_skipped(self):
        assert build_loadshift_sensors(None, None) == []
