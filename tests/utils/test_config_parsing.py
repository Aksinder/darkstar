"""strict_bool: config flags must not use Python truthiness ("false" is truthy)."""

import pytest

from utils.config_parsing import strict_bool


@pytest.mark.parametrize("v", [True, 1, 1.0, "true", "TRUE", " yes ", "on", "1"])
def test_true_values(v):
    assert strict_bool(v) is True


@pytest.mark.parametrize("v", [False, 0, 0.0, "false", "False", "no", "off", "0"])
def test_false_values(v):
    assert strict_bool(v, default=True) is False


@pytest.mark.parametrize("v", [None, "", "maybe", [], {}])
def test_unknown_falls_back_to_default(v):
    assert strict_bool(v) is False
    assert strict_bool(v, default=True) is True


def test_unseen_price_reserve_gate_uses_strict_parsing():
    """kepler.unseen_price_reserve.enabled: "false" must not switch the reserve on."""
    from pathlib import Path

    src = Path("planner/pipeline.py").read_text()
    assert 'strict_bool(upr_cfg.get("enabled", False))' in src
    assert strict_bool("false") is False
