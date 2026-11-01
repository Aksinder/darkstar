"""LearningEngine picks up config.yaml saves without a restart (port of upstream 172d5764).

ml/forward.py reads ``get_learning_engine().config``; before the fix that dict was
frozen at startup, so Settings changes never reached the forecast.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import backend.learning as learning_pkg
from backend.learning.engine import LearningEngine


def _write(path: Path, cfg: dict, mtime: float) -> None:
    path.write_text(yaml.safe_dump(cfg))
    os.utime(path, (mtime, mtime))


def _cfg(tmp_path: Path, kwp: float, inverted: bool = False) -> dict:
    return {
        "timezone": "Europe/Stockholm",
        "learning": {"sqlite_path": str(tmp_path / "learning.db")},
        "system": {"solar_array": {"kwp": kwp}},
        "input_sensors": {"grid_power_inverted": inverted},
    }


@pytest.fixture
def engine_factory():
    with patch("backend.learning.engine.LearningStore"):
        yield LearningEngine


def test_reload_picks_up_saved_config(tmp_path, engine_factory):
    cfg_path = tmp_path / "config.yaml"
    _write(cfg_path, _cfg(tmp_path, 5.0), 1_000_000)
    engine = engine_factory(str(cfg_path))
    assert engine.config["system"]["solar_array"]["kwp"] == 5.0

    assert engine.reload_config_if_changed() is False  # unchanged mtime -> no-op

    _write(cfg_path, _cfg(tmp_path, 9.5, inverted=True), 1_000_100)
    assert engine.reload_config_if_changed() is True
    assert engine.config["system"]["solar_array"]["kwp"] == 9.5
    assert engine.inversion_flags["grid"] is True


def test_broken_save_keeps_previous_config(tmp_path, engine_factory):
    cfg_path = tmp_path / "config.yaml"
    _write(cfg_path, _cfg(tmp_path, 5.0), 1_000_000)
    engine = engine_factory(str(cfg_path))

    cfg_path.write_text("system: [unclosed\n")
    os.utime(cfg_path, (1_000_200, 1_000_200))
    assert engine.reload_config_if_changed() is False
    assert engine.config["system"]["solar_array"]["kwp"] == 5.0


def test_singleton_accessor_returns_fresh_config(tmp_path, engine_factory, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write(cfg_path, _cfg(tmp_path, 5.0), 1_000_000)
    monkeypatch.setattr(learning_pkg, "_engine_instance", engine_factory(str(cfg_path)))

    _write(cfg_path, _cfg(tmp_path, 7.0), 1_000_300)
    assert learning_pkg.get_learning_engine().config["system"]["solar_array"]["kwp"] == 7.0
