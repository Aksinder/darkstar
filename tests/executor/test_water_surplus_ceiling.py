"""The boost gate must measure its ceiling against the series it compares with.

should_boost_on_surplus prefers the EXPORT price — spare PV costs the revenue foregone,
not import — but its ceiling came from the idle-hold's percentile of the IMPORT window.
Export runs about a krona below import at this site, so the ceiling cleared nearly
everything. Measured against the real 48 h window on 2026-09-08 (103 slots, spot
0.21-1.87):

    P30 import ceiling = 2.01 SEK/kWh  ->  98.1% of hours passed
    P40 and above                      ->   100% passed

The highest export price in that window was 2.02. The gate could not refuse even the most
expensive hour of two days.

Because export = spot + premium + grid_benefit - fee is affine with slope 1 in spot, a
percentile of the EXPORT series admits almost exactly that percentage of hours — P20 ->
20.4%, P30 -> 30.1%, P40 -> 39.8% on the same series. So the configured number reads
directly as "boost in the cheapest N% of the window", which is the property these tests
pin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from executor.engine import ExecutorEngine
from executor.water_hold import price_percentile

# The LIVE 48 h window as it stood at 2026-09-08 22:21 — 103 slots, verbatim, so these
# tests reproduce the measurement rather than a stylised version of it. A condensed
# sample understates the defect badly: it thins out the expensive tail, which is exactly
# the region where the mis-denominated ceiling fails to bite.
# This site: import = spot * 1.25 + 0.910, export = spot + 0.15.
SPOT = [
    0.94801, 0.65441, 0.47814, 0.5325, 0.4624, 0.40747, 0.33591, 1.16215, 0.86227, 0.94394,
    0.82613, 1.12255, 1.05795, 1.00809, 0.99202, 0.89251, 0.89239, 0.89239, 0.89239, 0.89998,
    0.84208, 0.97651, 0.90924, 0.83672, 1.06119, 1.09734, 1.1182, 0.92977, 1.08797, 1.17197,
    1.26647, 1.24873, 1.36107, 1.4684, 1.61008, 1.65571, 1.7656, 1.82863, 1.72019, 1.87226,
    1.87449, 1.79595, 1.75545, 1.80521, 1.67423, 1.49026, 1.4414, 1.43727, 1.37223, 1.19417,
    1.09957, 1.19328, 1.11563, 1.0082, 0.88258, 1.04077, 0.95175, 0.85826, 0.78094, 0.87745,
    0.78083, 0.68678, 0.66704, 0.66782, 0.68578, 0.66224, 0.68567, 0.64394, 0.75372, 0.84855,
    0.89094, 0.72505, 0.95052, 1.23869, 1.34311, 1.16684, 1.42221, 1.47754, 1.57572, 1.51146,
    1.6006, 1.64991, 1.64935, 1.73659, 1.76482, 1.70424, 1.5263, 1.68316, 1.50599, 1.38651,
    1.30864, 1.45032, 1.37223, 1.31957, 1.11563, 0.89251, 0.72505, 0.48006, 0.33469, 0.70396,
    0.49445, 0.37452, 0.20639,
]
IMPORT_W = [s * 1.25 + 0.91 for s in SPOT]
EXPORT_W = [s + 0.15 for s in SPOT]


def _engine():
    return ExecutorEngine.__new__(ExecutorEngine)


def _device(**kw):
    base = {
        "id": "spa",
        "idle_hold_max_price_percentile": 30.0,
        "idle_hold_max_price_sek_per_kwh": None,
        "surplus_boost_max_price_percentile": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


CTX = {"price_window": IMPORT_W, "export_price_window": EXPORT_W}


class TestTheOldCeilingCouldNotRefuse:
    def test_the_import_percentile_sits_above_almost_every_export_price(self):
        """The absolute value depends on the sample; the fraction is the defect."""
        cap = price_percentile(IMPORT_W, 30.0)
        passing = sum(1 for e in EXPORT_W if e <= cap) / len(EXPORT_W)
        assert cap == pytest.approx(2.01, abs=0.02), "the measured P30 import ceiling"
        assert passing >= 0.97, f"a ceiling that gates nothing: {passing:.1%} passed"
        # ...and the correctly denominated one gates roughly what it says.
        right = price_percentile(EXPORT_W, 30.0)
        assert sum(1 for e in EXPORT_W if e <= right) / len(EXPORT_W) <= 0.35


class TestTheNumberMeansWhatItSays:
    @pytest.mark.parametrize("pct", [20.0, 30.0, 40.0, 50.0])
    def test_percentile_p_admits_about_p_percent_of_hours(self, pct):
        eng = _engine()
        cap = eng._heater_surplus_ceiling(_device(surplus_boost_max_price_percentile=pct), CTX)
        admitted = 100.0 * sum(1 for e in EXPORT_W if e <= cap) / len(EXPORT_W)
        assert abs(admitted - pct) <= 3.0, f"P{pct} admitted {admitted:.1f}%"

    def test_a_higher_percentile_is_never_stricter(self):
        eng = _engine()
        caps = [
            eng._heater_surplus_ceiling(_device(surplus_boost_max_price_percentile=p), CTX)
            for p in (20.0, 30.0, 40.0, 50.0, 80.0)
        ]
        assert caps == sorted(caps)

    def test_it_reads_the_export_series_not_the_import_one(self):
        eng = _engine()
        cap = eng._heater_surplus_ceiling(_device(surplus_boost_max_price_percentile=40.0), CTX)
        assert cap == price_percentile(EXPORT_W, 40.0)
        assert cap != price_percentile(IMPORT_W, 40.0)


class TestItStaysInertUntilChosen:
    def test_unset_keeps_the_old_import_ceiling_exactly(self):
        """A site that has not picked a surplus percentile must see today's behaviour."""
        eng = _engine()
        dev = _device()
        assert eng._heater_surplus_ceiling(dev, CTX) == eng._heater_price_ceiling(dev, CTX)

    def test_unset_and_no_idle_percentile_either_is_the_absolute_value(self):
        eng = _engine()
        dev = _device(idle_hold_max_price_percentile=None, idle_hold_max_price_sek_per_kwh=0.9)
        assert eng._heater_surplus_ceiling(dev, CTX) == 0.9


class TestItRefusesToRemoveTheCeiling:
    def test_an_empty_export_series_falls_back_to_the_absolute_value(self):
        eng = _engine()
        dev = _device(
            surplus_boost_max_price_percentile=40.0, idle_hold_max_price_sek_per_kwh=0.9
        )
        assert eng._heater_surplus_ceiling(dev, {"export_price_window": []}) == 0.9

    def test_a_missing_export_series_does_not_raise(self):
        eng = _engine()
        dev = _device(surplus_boost_max_price_percentile=40.0)
        assert eng._heater_surplus_ceiling(dev, {}) is None


class TestParsing:
    """Through the REAL loader. A config key that parses to nothing is exactly how a
    setting silently does nothing, which is the whole theme of this change."""

    def _load(self, tmp_path, extra):
        import yaml

        from executor.config import load_executor_config

        f = tmp_path / "config.yaml"
        f.write_text(
            yaml.dump(
                {
                    "executor": {"enabled": True},
                    "water_heaters": [
                        dict({"id": "spa", "target_entity": "input_number.spa"}, **extra)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {d.id: d for d in load_executor_config(str(f)).water_heater_devices}

    def test_the_key_parses(self, tmp_path):
        by_id = self._load(tmp_path, {"surplus_boost_max_price_percentile": 40})
        assert by_id["spa"].surplus_boost_max_price_percentile == 40.0

    def test_absent_is_none(self, tmp_path):
        by_id = self._load(tmp_path, {})
        assert by_id["spa"].surplus_boost_max_price_percentile is None

    def test_a_blank_value_is_none_not_zero(self, tmp_path):
        """A blank YAML value must not become a ceiling of 0 that blocks every boost."""
        by_id = self._load(tmp_path, {"surplus_boost_max_price_percentile": None})
        assert by_id["spa"].surplus_boost_max_price_percentile is None
