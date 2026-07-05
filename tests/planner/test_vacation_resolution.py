"""Tests for the vacation source-of-truth (resolve_vacation_enabled)."""

from planner.pipeline import resolve_vacation_enabled


def test_wired_boolean_is_authoritative_both_ways():
    """The 2026-07-05 incident: config flag left true from a trip, boolean turned off
    at homecoming — vacation must END. Wired boolean wins in both directions."""
    assert resolve_vacation_enabled(True, False, entity_wired=True) is False
    assert resolve_vacation_enabled(False, True, entity_wired=True) is True
    assert resolve_vacation_enabled(True, True, entity_wired=True) is True
    assert resolve_vacation_enabled(False, False, entity_wired=True) is False


def test_unwired_falls_back_to_config_flag():
    assert resolve_vacation_enabled(True, False, entity_wired=False) is True
    assert resolve_vacation_enabled(False, False, entity_wired=False) is False
    # Legacy OR semantics preserved when unwired (ha_vacation defaults False anyway).
    assert resolve_vacation_enabled(False, True, entity_wired=False) is True
