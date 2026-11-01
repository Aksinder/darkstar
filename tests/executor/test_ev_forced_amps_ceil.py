"""Forced EV targets (plan floor / deadline guarantee) snap UP, never below the floor.

round() delivered up to half a step under the energy a guarantee was sized for
(upstream 5440027b). Surplus-only targets keep round(); fuse-capped ones stay DOWN.
"""

from executor.ev_surplus import EVSurplusConfig, EVSurplusInputs, compute_ev_surplus
from tests.executor.test_ev_surplus_priority import _fmb

CFG = EVSurplusConfig(enabled=True)


def _inputs(chargers, pv_w=0.0, battery_w=0.0, soc=50.0):
    return EVSurplusInputs(
        pv_w=pv_w, grid_w=0.0, battery_w=battery_w, battery_soc_percent=soc,
        import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=chargers,
    )


def test_plan_floor_between_steps_rounds_up():
    fmb = _fmb(deadline_hours=None, plan_floor_w=230.0 * 7.2)  # 7.2 A floor
    cmd = compute_ev_surplus(_inputs([fmb]), CFG)[0]
    assert cmd.switch_on and "plan" in cmd.reason
    assert cmd.set_current_a == 8.0  # round() would give 7 A, 0.2 A under the floor
    # The Schmitt quantizer sees the ceiled value, so a last write of 7 A cannot
    # suppress the raise as dither.
    assert cmd.raw_amps == 8.0


def test_plan_floor_on_grid_is_not_bumped_by_float_noise():
    fmb = _fmb(deadline_hours=None, plan_floor_w=230.0 * 7.0 + 1e-7)
    cmd = compute_ev_surplus(_inputs([fmb]), CFG)[0]
    assert cmd.set_current_a == 7.0


def test_deadline_floor_rounds_up():
    # 28 kWh, eff 0.9: soc 80 -> floor 86 in 1 h = 6% * 28 / 0.9 = 1866.7 W = 8.12 A.
    fmb = _fmb(soc_percent=80.0, deadline_hours=1.0)
    cmd = compute_ev_surplus(_inputs([fmb]), CFG)[0]
    assert cmd.switch_on and "deadline" in cmd.reason
    assert cmd.set_current_a == 9.0


def test_surplus_only_target_keeps_round():
    # 7.2 A of pure surplus, no floor: round() (the Schmitt band handles dither).
    fmb = _fmb(deadline_hours=None)
    cmd = compute_ev_surplus(_inputs([fmb], pv_w=5000.0, battery_w=230.0 * 7.2, soc=99.0), CFG)[0]
    assert cmd.switch_on and "surplus" in cmd.reason
    assert cmd.set_current_a == 7.0
    assert cmd.raw_amps == 7.2


def test_surplus_just_above_a_forced_floor_never_lands_below_it():
    # Floor 7.1 A, surplus 7.3 A: the target is surplus-led, but round(7.3) = 7 would
    # still undercut the floor — the floor's ceiling (8 A) wins.
    fmb = _fmb(deadline_hours=None, plan_floor_w=230.0 * 7.1)
    cmd = compute_ev_surplus(_inputs([fmb], pv_w=5000.0, battery_w=230.0 * 7.3, soc=99.0), CFG)[0]
    assert cmd.switch_on
    assert cmd.set_current_a == 8.0
