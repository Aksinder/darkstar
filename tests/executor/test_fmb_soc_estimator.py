"""Unit tests for the pure FMB SoC estimator logic."""

from __future__ import annotations

from executor.fmb_soc_estimator import (
    FmbSocConfig,
    FmbSocInputs,
    FmbSocState,
    initial_state,
    update_fmb_soc,
)

DAY = 86400.0
CAP = 28.0


def _cfg(**kw) -> FmbSocConfig:
    base = {
        "enabled": True,
        "capacity_kwh": CAP,
        "charge_efficiency": 1.0,  # tests reason in clean kWh; efficiency tested separately
        "prior_consumption_kwh_per_day": 5.6,
        "learn_alpha": 0.5,
        "full_idle_min_s": 300.0,
    }
    base.update(kw)
    return FmbSocConfig(**base)


def _inp(now, *, energy=None, power=0.0, plugged=True, enabled=True, limit=16.0, status=None):
    return FmbSocInputs(
        now_ts=now,
        lifetime_energy_kwh=energy,
        power_w=power,
        plugged=plugged,
        charger_enabled=enabled,
        dynamic_limit_a=limit,
        status=status,
    )


def test_initial_state_seeds_from_config():
    cfg = _cfg(initial_soc=50.0)
    st = initial_state(cfg)
    assert st.soc_pct == 50.0
    assert st.learned_rate_kwh_per_day == 5.6
    assert st.last_energy_kwh is None


def test_first_tick_only_baselines_no_drift():
    cfg = _cfg()
    st = initial_state(cfg)
    out, dbg = update_fmb_soc(st, _inp(1000.0, energy=100.0, power=0.0), cfg)
    assert dbg["reason"] == "init"
    assert out.soc_pct == 50.0  # no drift on the very first tick
    assert out.last_energy_kwh == 100.0
    assert out.last_ts == 1000.0


def test_charging_integrates_up():
    cfg = _cfg()
    st = FmbSocState(soc_pct=50.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6)
    # +2.8 kWh delivered = +10 % on a 28 kWh pack.
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=102.8, power=3000.0), cfg)
    assert dbg["charging"] is True
    assert abs(out.soc_pct - 60.0) < 1e-6
    assert abs(out.energy_since_anchor_kwh - 2.8) < 1e-6
    assert dbg["drift_pct"] == 0.0  # no decay while charging


def test_charge_efficiency_applied():
    cfg = _cfg(charge_efficiency=0.9)
    st = FmbSocState(soc_pct=50.0, last_energy_kwh=0.0, last_ts=0.0, learned_rate_kwh_per_day=5.6)
    out, _ = update_fmb_soc(st, _inp(60.0, energy=2.8, power=3000.0), cfg)
    # 2.8 kWh delivered x 0.9 = 2.52 kWh stored = +9 %.
    assert abs(out.soc_pct - 59.0) < 1e-6


def test_idle_drifts_down_at_learned_rate():
    cfg = _cfg()
    st = FmbSocState(
        soc_pct=80.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    # One full day idle at 5.6 kWh/day = 20 % drop.
    out, dbg = update_fmb_soc(st, _inp(DAY, energy=100.0, power=0.0), cfg)
    assert abs(out.soc_pct - 60.0) < 1e-6
    assert abs(dbg["drift_pct"] - 20.0) < 1e-6


def test_counter_reset_is_ignored():
    cfg = _cfg()
    st = FmbSocState(
        soc_pct=70.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    # Lifetime counter dropped (integration restart) → no phantom charge, no crash.
    out, _ = update_fmb_soc(st, _inp(60.0, energy=5.0, power=0.0), cfg)
    assert out.soc_pct <= 70.0
    assert out.last_energy_kwh == 5.0


def test_soc_clamped_0_100():
    cfg = _cfg()
    hi = FmbSocState(soc_pct=99.0, last_energy_kwh=0.0, last_ts=0.0, learned_rate_kwh_per_day=5.6)
    out, _ = update_fmb_soc(hi, _inp(60.0, energy=10.0, power=3000.0), cfg)
    assert out.soc_pct <= 100.0
    lo = FmbSocState(soc_pct=1.0, last_energy_kwh=0.0, last_ts=0.0, learned_rate_kwh_per_day=99.0)
    out2, _ = update_fmb_soc(lo, _inp(DAY, energy=0.0, power=0.0), cfg)
    assert out2.soc_pct >= 0.0


def test_taper_while_offering_latches_full_and_sets_100():
    cfg = _cfg(full_idle_min_s=300.0)
    st = FmbSocState(
        soc_pct=90.0, last_energy_kwh=10.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    # t=100: offering 16 A but ~0 W → taper timer starts (not yet full).
    s1, d1 = update_fmb_soc(st, _inp(100.0, energy=10.0, power=10.0, limit=16.0), cfg)
    assert d1["full"] is False
    # t=500: taper sustained >300 s → full latches, soc snaps to 100.
    s2, d2 = update_fmb_soc(s1, _inp(500.0, energy=10.0, power=10.0, limit=16.0), cfg)
    assert d2["full"] is True
    assert s2.soc_pct == 100.0
    assert s2.energy_since_anchor_kwh == 0.0
    assert s2.last_full_ts == 500.0


def test_status_completed_latches_full():
    cfg = _cfg()
    st = FmbSocState(
        soc_pct=88.0, last_energy_kwh=10.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=10.0, power=0.0, status="completed"), cfg)
    assert dbg["full"] is True
    assert out.soc_pct == 100.0


def test_learns_rate_from_full_to_full_cycle():
    """User's spec: the energy needed to refill since last full IS the consumption rate."""
    cfg = _cfg(learn_alpha=0.5, full_idle_min_s=0.0, min_anchor_energy_kwh=2.0)
    # Anchor: already full at t=0.
    st = FmbSocState(
        soc_pct=100.0,
        last_energy_kwh=0.0,
        last_ts=0.0,
        energy_since_anchor_kwh=0.0,
        last_full_ts=0.0,
        learned_rate_kwh_per_day=5.6,
        full_latched=True,
    )
    # One day later we plug in and deliver 10 kWh to refill to full → true use was 10 kWh/day.
    # First, leave full (drive) so the latch clears.
    st = FmbSocState(**{**st.__dict__, "full_latched": False, "soc_pct": 64.0})
    # Deliver 10 kWh over the charge (one tick for simplicity), then taper→full at t=1 day.
    charged, _ = update_fmb_soc(
        st, _inp(DAY - 10.0, energy=10.0, power=3000.0, limit=16.0), cfg
    )
    assert charged.energy_since_anchor_kwh > 9.9
    full, dbg = update_fmb_soc(
        charged, _inp(DAY, energy=10.0, power=5.0, limit=16.0, status="completed"), cfg
    )
    assert "learned" in dbg
    # implied ≈ 10 kWh/day; EMA(0.5) of (5.6, 10) ≈ 7.8.
    assert 7.5 < full.learned_rate_kwh_per_day < 8.1
    assert full.learned_rate_kwh_per_day > 5.6  # rate moved UP toward observed


def test_user_example_higher_than_expected_refill_speeds_drift():
    """Estimate drifted to ~80 %, but refill needed >20 % ⇒ next drift should be faster."""
    cfg = _cfg(learn_alpha=0.5, min_anchor_energy_kwh=2.0)
    # Last full 1 day ago; estimate now 80 (model thought 20 % consumed = 5.6 kWh).
    st = FmbSocState(
        soc_pct=80.0,
        last_energy_kwh=0.0,
        last_ts=0.0,
        energy_since_anchor_kwh=0.0,
        last_full_ts=-DAY,  # last full one day before t=0
        learned_rate_kwh_per_day=5.6,
        full_latched=False,
    )
    # Refill actually takes 7 kWh (25 %) → real consumption was 7 kWh/day, faster than 5.6.
    charged, _ = update_fmb_soc(st, _inp(10.0, energy=7.0, power=3000.0, limit=16.0), cfg)
    full, dbg = update_fmb_soc(
        charged, _inp(20.0, energy=7.0, power=5.0, limit=16.0, status="completed"), cfg
    )
    assert dbg["learned"]["implied_kwh_per_day"] > 5.6
    assert full.learned_rate_kwh_per_day > 5.6  # drift will now be faster, as requested


def test_no_learn_without_prior_full_anchor():
    cfg = _cfg(min_anchor_energy_kwh=2.0)
    st = FmbSocState(
        soc_pct=60.0,
        last_energy_kwh=0.0,
        last_ts=0.0,
        energy_since_anchor_kwh=20.0,
        last_full_ts=None,  # never had a full → cannot learn a rate yet
        learned_rate_kwh_per_day=5.6,
    )
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=0.0, power=5.0, status="completed"), cfg)
    assert "learned" not in dbg
    assert out.soc_pct == 100.0  # still anchors the level
    assert out.last_full_ts == 60.0


def test_full_latch_releases_on_unplug():
    cfg = _cfg()
    st = FmbSocState(
        soc_pct=100.0,
        last_energy_kwh=10.0,
        last_ts=0.0,
        last_full_ts=0.0,
        learned_rate_kwh_per_day=5.6,
        full_latched=True,
    )
    out, _ = update_fmb_soc(st, _inp(60.0, energy=10.0, power=0.0, plugged=False), cfg)
    assert out.full_latched is False


def test_seed_soc_applies_once_then_drifts():
    cfg = _cfg(seed_soc=40.0)
    st = FmbSocState(
        soc_pct=50.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    # Charger off / not offering current (idle, plugged but no surplus) → pure drift, no false-full.
    # Tick 1: seed snaps 50 -> 40.
    s1, d1 = update_fmb_soc(st, _inp(60.0, energy=100.0, power=0.0, enabled=False, limit=0.0), cfg)
    assert d1["seeded"] == 40.0
    assert abs(s1.soc_pct - 40.0) < 0.5  # 40 minus a tiny drift over 60 s
    assert s1.applied_seed == 40.0
    # Tick 2: same seed → no re-snap; it drifts down from 40, not reset to 40.
    s2, d2 = update_fmb_soc(
        s1, _inp(DAY + 60.0, energy=100.0, power=0.0, enabled=False, limit=0.0), cfg
    )
    assert "seeded" not in d2
    assert s2.soc_pct < 40.0  # drifted, not re-seeded


def test_seed_soc_change_reseeds():
    cfg = _cfg(seed_soc=30.0)
    st = FmbSocState(
        soc_pct=80.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6,
        applied_seed=40.0,  # a previous seed already applied
    )
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=100.0, power=0.0), cfg)
    assert dbg["seeded"] == 30.0
    assert abs(out.soc_pct - 30.0) < 0.5
    assert out.applied_seed == 30.0


def test_seed_soc_none_is_noop():
    cfg = _cfg(seed_soc=None)
    st = FmbSocState(
        soc_pct=55.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=100.0, power=0.0), cfg)
    assert "seeded" not in dbg
    assert out.applied_seed is None


def test_seed_on_first_tick():
    cfg = _cfg(seed_soc=40.0)
    st = initial_state(cfg)  # last_ts None → init tick
    out, dbg = update_fmb_soc(st, _inp(1000.0, energy=100.0, power=0.0), cfg)
    assert dbg["reason"] == "init"
    assert dbg["seeded"] == 40.0
    assert out.soc_pct == 40.0
    assert out.applied_seed == 40.0


def test_correction_first_observation_records_without_snapping():
    """A stale input_number at startup must NOT hijack the estimate."""
    cfg = _cfg()
    st = FmbSocState(
        soc_pct=50.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    # Pass an explicit stale 100 as the first-seen correction value.
    inp = FmbSocInputs(
        now_ts=60.0, lifetime_energy_kwh=100.0, power_w=0.0, plugged=True,
        charger_enabled=False, dynamic_limit_a=0.0, status=None, correction_value=100.0,
    )
    out, dbg = update_fmb_soc(st, inp, cfg)
    assert "corrected" not in dbg
    assert out.soc_pct < 51.0  # stayed near 50, did NOT jump to 100
    assert out.last_correction_value == 100.0  # but recorded for next-change detection


def test_correction_user_change_snaps():
    cfg = _cfg(correction_threshold=1.0)
    st = FmbSocState(
        soc_pct=50.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6,
        last_correction_value=100.0,  # already saw the stale 100
    )
    inp = FmbSocInputs(
        now_ts=60.0, lifetime_energy_kwh=100.0, power_w=0.0, plugged=True,
        charger_enabled=False, dynamic_limit_a=0.0, status=None, correction_value=40.0,
    )
    out, dbg = update_fmb_soc(st, inp, cfg)
    assert dbg["corrected"] == 40.0
    assert abs(out.soc_pct - 40.0) < 0.5
    assert out.last_correction_value == 40.0
    assert out.energy_since_anchor_kwh == 0.0  # anchor re-armed


def test_correction_below_threshold_ignored():
    cfg = _cfg(correction_threshold=1.0)
    st = FmbSocState(
        soc_pct=60.0, last_energy_kwh=100.0, last_ts=0.0, learned_rate_kwh_per_day=5.6,
        last_correction_value=40.0,
    )
    inp = FmbSocInputs(
        now_ts=60.0, lifetime_energy_kwh=100.0, power_w=0.0, plugged=True,
        charger_enabled=False, dynamic_limit_a=0.0, status=None, correction_value=40.5,
    )
    _out, dbg = update_fmb_soc(st, inp, cfg)
    assert "corrected" not in dbg  # 0.5 < threshold 1.0 → noise, ignored


def test_low_offer_does_not_false_full():
    """If we are NOT offering meaningful current, a 0 W reading must not be read as full."""
    cfg = _cfg(full_offered_min_a=5.0, full_idle_min_s=0.0)
    st = FmbSocState(
        soc_pct=70.0, last_energy_kwh=10.0, last_ts=0.0, learned_rate_kwh_per_day=5.6
    )
    out, dbg = update_fmb_soc(st, _inp(60.0, energy=10.0, power=0.0, limit=0.0), cfg)
    assert dbg["full"] is False
    assert out.soc_pct < 100.0
