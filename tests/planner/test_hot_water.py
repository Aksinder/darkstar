"""Tests for the open-loop hot-water availability estimator."""

import pytest

from planner.hot_water import HotWaterEstimator, estimate_draw_kwh
from planner.thermal import WaterTankModel


def _tank():
    return WaterTankModel(volume_litres=200, t_cold_c=10, t_max_c=80, ua_w_per_k=2.0)


def test_starts_full():
    est = HotWaterEstimator(_tank())
    assert est.soc_percent() == pytest.approx(100.0)
    assert est.temperature_c() == pytest.approx(80.0)
    assert est.liters_in_tank() == pytest.approx(200.0)


def test_from_temperature_seed():
    tank = _tank()
    est = HotWaterEstimator.from_temperature(tank, 45.0)
    assert est.temperature_c() == pytest.approx(45.0)
    # 45 over [10,80] -> 50% SoC
    assert est.soc_percent() == pytest.approx(50.0, abs=0.5)


def test_standing_loss_lowers_state_when_idle():
    est = HotWaterEstimator.from_temperature(_tank(), 70.0)
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0)  # idle 1 h
    assert est.stored_kwh < before
    assert est.temperature_c() < 70.0


def test_heating_raises_state_and_caps_at_full():
    est = HotWaterEstimator.from_temperature(_tank(), 40.0)
    for _ in range(60):
        est.update(dt_minutes=10, heating_kw=3.0)  # 10 h of 3 kW
    assert est.soc_percent() == pytest.approx(100.0)  # capped at full
    assert est.temperature_c() == pytest.approx(80.0)


def test_auto_anchors_to_full_when_thermostat_satisfied():
    # Seed deliberately wrong-low; a real heat-up that then cuts off must re-pin
    # the estimate to full regardless of the accumulated drift.
    tank = _tank()
    est = HotWaterEstimator.from_temperature(tank, 30.0, full_anchor_after_min=8.0)
    # Heat for 20 min (sustained), then element switches off (thermostat satisfied).
    est.update(dt_minutes=20, heating_kw=3.0)
    assert est.soc_percent() < 100.0  # not full yet after 20 min from 30C
    est.update(dt_minutes=1, heating_kw=0.0)  # switch-off after sustained heat
    assert est.soc_percent() == pytest.approx(100.0)  # anchored to full


def test_brief_heating_blip_does_not_anchor_full():
    est = HotWaterEstimator.from_temperature(_tank(), 30.0, full_anchor_after_min=8.0)
    est.update(dt_minutes=2, heating_kw=3.0)  # only 2 min < anchor threshold
    est.update(dt_minutes=1, heating_kw=0.0)
    assert est.soc_percent() < 100.0  # must NOT jump to full on a blip


def test_manual_anchor_full():
    est = HotWaterEstimator.from_temperature(_tank(), 20.0)
    est.anchor_full()
    assert est.soc_percent() == pytest.approx(100.0)


def test_mixed_liters_exceeds_volume_and_floors_at_comfort():
    est = HotWaterEstimator.from_temperature(_tank(), 70.0)
    # At 70C with 40C comfort and 10C cold: 200 * (70-10)/(40-10) = 400 L.
    assert est.mixed_liters_at(40.0) == pytest.approx(400.0, abs=1.0)
    cold = HotWaterEstimator.from_temperature(_tank(), 35.0)
    assert cold.mixed_liters_at(40.0) == 0.0  # below comfort -> nothing usable


def test_estimate_draw_is_heating_minus_losses():
    tank = _tank()
    # 5 kWh heated in over 6 h at ~70C avg; losses = UA*dT*h.
    losses = tank.avg_loss_kw(70.0, 20.0) * 6.0  # 2 W/K * 50 K = 100 W -> 0.6 kWh
    draw = estimate_draw_kwh(tank, heating_energy_kwh=5.0, avg_temp_c=70.0, hours=6.0)
    assert draw == pytest.approx(5.0 - losses, abs=1e-6)
    assert losses == pytest.approx(0.6, abs=0.01)


def test_estimate_draw_clamped_at_zero():
    tank = _tank()
    # Less heating than losses (impossible draw) -> clamp to 0.
    assert estimate_draw_kwh(tank, heating_energy_kwh=0.1, avg_temp_c=70.0, hours=6.0) == 0.0


# -- learned draw (the FMB-style down-force) + persistence ------------------


def test_learned_draw_depletes_between_heating_runs():
    """A configured draw must visibly deplete the tank while idle (not just standing loss)."""
    est = HotWaterEstimator(_tank(), prior_draw_kw=1.0)  # ~1 kW average draw
    est.stored_kwh = est.tank.capacity_kwh()
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0)  # idle 1 h, no heating
    drop = before - est.stored_kwh
    # ~1 kWh from draw + a little standing loss; standing loss alone would be <0.1 kWh.
    assert drop > 0.9
    assert est.soc_percent() < 100.0


def test_full_anchor_learns_draw_rate():
    """Over a full->full window, the draw rate is learned from heating_in - losses."""
    tank = _tank()
    est = HotWaterEstimator(tank, prior_draw_kw=0.1, draw_learn_alpha=1.0, full_anchor_after_min=8.0)
    # Start full, then idle 10 h while ~2 kWh is tapped (we deplete via the prior, but the
    # learner uses the energy we put back in). Simulate: idle draws it down, then a heat-up.
    est.update(dt_minutes=600, heating_kw=0.0)  # 10 h idle (minutes_since_anchor=600)
    # Now heat for 1 h at 3 kW (3 kWh in) then cut off -> anchor + learn.
    est.update(dt_minutes=60, heating_kw=3.0)   # heating run = 60 min >= 8 -> eligible
    est.update(dt_minutes=5, heating_kw=0.0)    # cut-off -> _anchor_full_and_learn
    # Window ~11 h, 3 kWh in, losses small -> implied draw ~3/11 ~= 0.27 kW. alpha=1 -> adopt.
    assert est.learned_draw_kw > 0.15  # moved up from the 0.1 prior toward the observed rate
    assert est.soc_percent() == pytest.approx(100.0)  # re-pinned to full


def test_state_dict_roundtrips_through_apply_state():
    tank = _tank()
    a = HotWaterEstimator(tank, prior_draw_kw=0.1)
    a.stored_kwh = 5.0
    a.learned_draw_kw = 0.42
    a._energy_in_since_anchor_kwh = 1.3
    a._minutes_since_anchor = 123.0
    blob = a.state_dict()

    b = HotWaterEstimator(tank)  # starts full
    b.apply_state(blob)
    assert b.stored_kwh == pytest.approx(5.0)
    assert b.learned_draw_kw == pytest.approx(0.42)
    assert b._energy_in_since_anchor_kwh == pytest.approx(1.3)
    assert b._minutes_since_anchor == pytest.approx(123.0)


def test_apply_state_ignores_missing_keys_and_clamps():
    tank = _tank()
    est = HotWaterEstimator(tank)
    est.apply_state({"stored_kwh": 999.0})  # over capacity -> clamped; other keys untouched
    assert est.stored_kwh == pytest.approx(tank.capacity_kwh())
    assert est.learned_draw_kw == pytest.approx(est.prior_draw_kw)


# -- switch-aware draw gating (only count the learned draw down when the switch is OFF) --


def test_switch_off_counts_learned_draw_down():
    """Switch OFF + idle => the tank genuinely coasts: standing loss AND learned draw."""
    est = HotWaterEstimator.from_temperature(_tank(), 70.0, prior_draw_kw=1.0)
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0, switch_on=False)  # idle 1 h, switch cut
    # ~0.1 kWh standing loss + 1.0 kWh learned draw.
    assert before - est.stored_kwh > 0.9


def test_switch_on_idle_holds_no_learned_draw():
    """Switch ON + idle element => thermostat satisfied => hold. Only standing loss, no draw."""
    est = HotWaterEstimator.from_temperature(_tank(), 70.0, prior_draw_kw=1.0)
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=0.0, switch_on=True)  # idle 1 h, switch on/maintained
    drop = before - est.stored_kwh
    # Standing loss only (~0.1 kWh); the 1.0 kWh learned draw must NOT be applied.
    assert 0.0 < drop < 0.5


def test_switch_none_matches_legacy_off_behaviour():
    """Unknown switch state (default) stays on the safe side: draw depletes, same as OFF."""
    est_none = HotWaterEstimator.from_temperature(_tank(), 70.0, prior_draw_kw=1.0)
    est_off = HotWaterEstimator.from_temperature(_tank(), 70.0, prior_draw_kw=1.0)
    est_none.update(dt_minutes=60, heating_kw=0.0)  # switch_on defaults to None
    est_off.update(dt_minutes=60, heating_kw=0.0, switch_on=False)
    assert est_none.stored_kwh == pytest.approx(est_off.stored_kwh)


def test_switch_on_still_adds_energy_while_heating():
    """The switch flag only gates the idle draw; a drawing element still charges the tank."""
    est = HotWaterEstimator.from_temperature(_tank(), 40.0, prior_draw_kw=1.0)
    before = est.stored_kwh
    est.update(dt_minutes=60, heating_kw=3.0, switch_on=True)
    assert est.stored_kwh > before


class TestSaturation:
    """Powered but refusing energy = the thermostat has opened = nothing to fill.

    A LIVE measurement, unlike stored_kwh, which is a model estimate and can be days
    stale — main_tank sat at minutes_since_anchor 9014 (six days) on 2026-09-03 while
    physically saturated. The daily floor needs the live answer: on 2026-09-02 the house
    tank was commanded on through the evening peak drawing 0 W, putting 0.58 kWh in at
    2.42-2.44 SEK/kWh, because heated_today read 1.409 against a 6.00 floor it could not
    physically meet.
    """

    def _est(self):
        return HotWaterEstimator.from_temperature(_tank(), 60.0, prior_draw_kw=0.2)

    def test_powered_and_idle_becomes_saturated(self):
        est = self._est()
        assert not est.is_saturated
        est.update(dt_minutes=est.saturated_after_min, heating_kw=0.0, switch_on=True)
        assert est.is_saturated
        assert est.saturated_minutes == pytest.approx(est.saturated_after_min)

    def test_a_switch_that_is_off_is_never_saturated(self):
        """THE villavagn CASE, and the whole reason the switch state is required.

        That tank also reads ~0 W every fifteen minutes — but because an HA phase guard
        cuts its switch, not because it is full. Blocked, not satisfied. Calling it
        saturated would stop Darkstar heating a cold tank, and nobody would find out
        until the shower ran cold.
        """
        est = self._est()
        est.update(dt_minutes=600, heating_kw=0.0, switch_on=False)
        assert not est.is_saturated
        assert est.saturated_minutes == 0.0

    def test_an_unknown_switch_is_never_saturated(self):
        """An unwired or unreadable switch must not look like a satisfied thermostat."""
        est = self._est()
        est.update(dt_minutes=600, heating_kw=0.0, switch_on=None)
        assert not est.is_saturated

    def test_drawing_current_clears_it(self):
        """The tank accepted energy, so it was not full. The floor comes back by itself."""
        est = self._est()
        est.update(dt_minutes=600, heating_kw=0.0, switch_on=True)
        assert est.is_saturated
        est.update(dt_minutes=1, heating_kw=3.0, switch_on=True)
        assert not est.is_saturated
        assert est.saturated_minutes == 0.0

    def test_the_clock_needs_to_be_sustained(self):
        """One tick is not evidence. The element cycles faster than the sensor polls —
        reading a single instant is exactly how the 2026-09-03 diagnosis went wrong."""
        est = self._est()
        est.update(dt_minutes=est.saturated_after_min - 1, heating_kw=0.0, switch_on=True)
        assert not est.is_saturated

    def test_a_villavagn_style_trickle_still_counts_as_idle(self):
        """sensor.villavagn_vvb_power reads ~1.3 W when that tank is saturated, not 0.
        The threshold is heating_on_w, so a trickle must not read as heating."""
        est = self._est()
        est.update(dt_minutes=600, heating_kw=0.0013, switch_on=True)
        assert est.is_saturated

    def test_saturation_survives_a_restart(self):
        est = self._est()
        est.update(dt_minutes=600, heating_kw=0.0, switch_on=True)
        blob = est.state_dict()
        assert "saturated_min" in blob

        fresh = self._est()
        fresh.apply_state(blob)
        assert fresh.is_saturated

    def test_an_old_state_file_without_the_key_is_not_saturated(self):
        """Forward/back compat: a state written before this existed must not claim it."""
        est = self._est()
        est.apply_state({"stored_kwh": 5.0, "learned_draw_kw": 0.2})
        assert not est.is_saturated
