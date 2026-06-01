"""Tests for deferrable-load executor control decisions."""

from executor.deferrable import (
    DeferrableLoadState,
    decide_deferrable_loads,
)

CFG = [
    {"id": "dishwasher", "enabled": True, "switch_entity": "switch.diskmaskin"},
    {"id": "washer", "enabled": True, "switch_entity": "switch.tvattstuga_plugg"},
]


def _d(decisions, lid):
    return next(d for d in decisions if d.id == lid)


def test_feature_disabled_touches_nothing():
    states = [DeferrableLoadState(id="dishwasher", pending=True, switch_on=True)]
    out = decide_deferrable_loads(CFG, states, {"dishwasher": False}, enabled=False)
    for d in out:
        assert d.write is False


def test_no_pending_run_leaves_power_on():
    states = [DeferrableLoadState(id="dishwasher", pending=False, switch_on=True)]
    out = decide_deferrable_loads(CFG, states, {}, enabled=True)
    d = _d(out, "dishwasher")
    assert d.switch_on is True
    assert d.write is False  # don't interfere with manual use


def test_pending_outside_window_holds_off():
    # Queued cycle, switch currently on, not planned to run now -> turn OFF (hold).
    states = [DeferrableLoadState(id="dishwasher", pending=True, switch_on=True)]
    out = decide_deferrable_loads(CFG, states, {"dishwasher": False}, enabled=True)
    d = _d(out, "dishwasher")
    assert d.switch_on is False
    assert d.write is True


def test_pending_in_window_powers_on():
    # Queued cycle, currently held off, planner says run now -> turn ON.
    states = [DeferrableLoadState(id="dishwasher", pending=True, switch_on=False)]
    out = decide_deferrable_loads(CFG, states, {"dishwasher": True}, enabled=True)
    d = _d(out, "dishwasher")
    assert d.switch_on is True
    assert d.write is True


def test_no_redundant_write_when_already_correct():
    # Already off and should stay off -> no write.
    states = [DeferrableLoadState(id="dishwasher", pending=True, switch_on=False)]
    out = decide_deferrable_loads(CFG, states, {"dishwasher": False}, enabled=True)
    d = _d(out, "dishwasher")
    assert d.switch_on is False
    assert d.write is False


def test_hold_cap_fails_open_to_on():
    # Held longer than the cap -> power on regardless of plan (fail-open).
    states = [DeferrableLoadState(id="dishwasher", pending=True, switch_on=False, held_minutes=999)]
    out = decide_deferrable_loads(
        CFG, states, {"dishwasher": False}, enabled=True, max_hold_minutes=720
    )
    d = _d(out, "dishwasher")
    assert d.switch_on is True
    assert d.write is True
    assert "cap" in d.reason


def test_load_without_switch_is_skipped():
    cfg = [{"id": "dishwasher", "enabled": True}]  # no switch_entity
    out = decide_deferrable_loads(cfg, [], {}, enabled=True)
    assert out == []


def test_disabled_load_skipped():
    cfg = [{"id": "washer", "enabled": False, "switch_entity": "switch.x"}]
    states = [DeferrableLoadState(id="washer", pending=True)]
    out = decide_deferrable_loads(cfg, states, {"washer": True}, enabled=True)
    assert out == []


def test_two_loads_independent_decisions():
    states = [
        DeferrableLoadState(id="dishwasher", pending=True, switch_on=True),
        DeferrableLoadState(id="washer", pending=True, switch_on=False),
    ]
    out = decide_deferrable_loads(CFG, states, {"dishwasher": False, "washer": True}, enabled=True)
    assert _d(out, "dishwasher").switch_on is False  # holding
    assert _d(out, "washer").switch_on is True  # running now
