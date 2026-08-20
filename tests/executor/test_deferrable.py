

class TestReclaimAfterManualCut:
    """Owner, 2026-08-20: "vi borde automatiskt få tillbaka i darkstar efter viss tid
    om vi manuellt avbrutit."

    Darkstar never re-energizes a human-cut plug on its own — right in the moment (the
    person may have opened the machine), wrong forever (the interrupted programme sits
    dead until someone remembers it). This hands ownership back after a delay; the
    ordinary window logic still decides WHEN to energize.
    """

    def _reclaim(self, **kw):
        from executor.deferrable import should_reclaim_after_manual_cut

        base = {
            "pending": True,
            "switch_on": False,
            "held_by_us": False,
            "manual_off_since": 1000.0,
            "now_ts": 1000.0 + 3600.0,
            "manual_cut_return_s": 3600.0,
        }
        base.update(kw)
        return should_reclaim_after_manual_cut(**base)

    def test_reclaims_once_the_timer_expires(self):
        assert self._reclaim() is True

    def test_holds_off_before_the_timer(self):
        assert self._reclaim(now_ts=1000.0 + 3599.0) is False

    def test_opt_in_only(self):
        """0 keeps the historical behaviour: a human-cut plug stays the human's."""
        assert self._reclaim(manual_cut_return_s=0.0) is False

    def test_never_reclaims_a_finished_cycle(self):
        """Nothing to resume — and reclaiming an idle appliance would let Darkstar
        switch on a machine nobody started."""
        assert self._reclaim(pending=False) is False

    def test_never_reclaims_a_hold_that_is_already_ours(self):
        assert self._reclaim(held_by_us=True) is False

    def test_a_powered_plug_is_not_a_cut(self):
        assert self._reclaim(switch_on=True) is False

    def test_no_stamp_means_no_clock(self):
        assert self._reclaim(manual_off_since=None) is False


class TestManualOffStamp:
    """The clock the reclaim reads, kept by the power state machine."""

    def _step(self, prev, power_w, switch_on, now_ts):
        from executor.deferrable import AppliancePowerConfig, update_appliance_power_state

        return update_appliance_power_state(
            prev, power_w, switch_on, now_ts, AppliancePowerConfig()
        )

    def test_stamps_when_a_pending_cycle_loses_power(self):
        from executor.deferrable import AppliancePowerState

        prev = AppliancePowerState(pending=True, switch_was_on=True)
        new, _ = self._step(prev, 0.0, False, 5000.0)
        assert new.manual_off_since == 5000.0

    def test_the_stamp_does_not_drift_while_it_stays_off(self):
        """The clock must measure the cut, not this tick."""
        from executor.deferrable import AppliancePowerState

        prev = AppliancePowerState(pending=True, manual_off_since=5000.0)
        new, _ = self._step(prev, 0.0, False, 9000.0)
        assert new.manual_off_since == 5000.0

    def test_re_powering_clears_it(self):
        from executor.deferrable import AppliancePowerState

        prev = AppliancePowerState(pending=True, manual_off_since=5000.0)
        new, _ = self._step(prev, 1500.0, True, 9000.0)
        assert new.manual_off_since is None

    def test_an_idle_appliance_carries_no_clock(self):
        from executor.deferrable import AppliancePowerState

        new, _ = self._step(AppliancePowerState(), 0.0, False, 5000.0)
        assert new.manual_off_since is None

    def test_it_survives_the_state_file_round_trip(self):
        from dataclasses import asdict

        from executor.deferrable import AppliancePowerState

        st = AppliancePowerState(pending=True, manual_off_since=5000.0)
        assert AppliancePowerState(**asdict(st)).manual_off_since == 5000.0

    def test_an_old_state_file_still_loads(self):
        """Forward compat: files written before this field existed."""
        from executor.deferrable import AppliancePowerState

        old = {"pending": True, "running": False, "start_ts": 1.0}
        assert AppliancePowerState(**old).manual_off_since is None
