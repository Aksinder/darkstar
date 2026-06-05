"""Tests for the phase_model.json loaders used by the planner (fractions + phases)."""

import json

from backend.learning.phase_learning import load_device_phases, load_phase_fractions


def _write(tmp_path, model):
    (tmp_path / "phase_model.json").write_text(json.dumps(model), encoding="utf-8")
    return {"config_dir": str(tmp_path)}


class TestLoadPhaseFractions:
    def test_reads_fractions(self, tmp_path):
        cfg = _write(tmp_path, {"fractions": {"A": 0.5, "B": 0.3, "C": 0.2}})
        assert load_phase_fractions(cfg) == {"A": 0.5, "B": 0.3, "C": 0.2}

    def test_missing_file_returns_none(self, tmp_path):
        assert load_phase_fractions({"config_dir": str(tmp_path)}) is None

    def test_incomplete_fractions_returns_none(self, tmp_path):
        cfg = _write(tmp_path, {"fractions": {"A": 0.5, "B": 0.5}})  # no C
        assert load_phase_fractions(cfg) is None


class TestLoadDevicePhases:
    def test_filters_to_confident_single_phase(self, tmp_path):
        cfg = _write(
            tmp_path,
            {
                "devices": [
                    {"id": "easee", "phase": "A", "load_type": "single", "confidence": 0.9},
                    {"id": "vvb", "phase": None, "load_type": "three_phase", "confidence": 0.95},
                    {"id": "weak", "phase": "B", "load_type": "single", "confidence": 0.3},
                ]
            },
        )
        assert load_device_phases(cfg, min_confidence=0.6) == {"easee": "A"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_device_phases({"config_dir": str(tmp_path)}) == {}

    def test_no_devices_key_returns_empty(self, tmp_path):
        cfg = _write(tmp_path, {"fractions": {"A": 0.5, "B": 0.3, "C": 0.2}})
        assert load_device_phases(cfg) == {}
