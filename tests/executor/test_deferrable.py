

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

    def test_an_idle_plug_is_reclaimed_too(self):
        """Revised 2026-08-20 on the owner's correction: the RESTING state is plug-on,
        because start detection watches the appliance's own draw. A plug left off is
        not a machine kept safe — it is a detector switched off, so the next cycle can
        never be seen. The caller restores it to ON rather than resuming a cycle."""
        assert self._reclaim() is True

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

    def test_an_idle_appliance_carries_the_clock_too(self):
        """Widened 2026-08-20: an idle plug switched off is a start detector switched
        off, so it needs the same hand-back clock as an interrupted cycle."""
        from executor.deferrable import AppliancePowerState

        new, _ = self._step(AppliancePowerState(), 0.0, False, 5000.0)
        assert new.manual_off_since == 5000.0

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


class TestWhoTouchedThePlug:
    """HA stamps every state with the context that produced it: a service call carries
    the calling user's id, a device reporting itself carries None. Observed live
    2026-08-20 — switch.diskmaskin went off with user_id 613be4dd (Robert), while
    sensor.darkstar_dishwasher_state carried 97f1bc39 (Darkstar's own token).

    This turns "a human cut it" from an inference about our own memory into an
    observation about the world.
    """

    DARKSTAR = "97f1bc39f6184ab2a90da66a62f8b234"
    ROBERT = "613be4dd2bd54547bbe603f15be64363"

    def _who(self, **kw):
        from executor.deferrable import classify_plug_cut

        base = {
            "context_user_id": self.ROBERT,
            "darkstar_user_id": self.DARKSTAR,
            "held_by_us": False,
        }
        base.update(kw)
        return classify_plug_cut(**base)

    def test_a_named_person_is_a_human_cut(self):
        from executor.deferrable import CUT_BY_HUMAN

        assert self._who() == CUT_BY_HUMAN

    def test_our_own_token_is_us(self):
        from executor.deferrable import CUT_BY_US

        assert self._who(context_user_id=self.DARKSTAR, held_by_us=True) == CUT_BY_US

    def test_a_person_beats_our_memory_of_holding_it(self):
        """THE case this exists for: someone switches off a plug Darkstar already
        held. The state never changes — only the context does — so without reading it
        Darkstar keeps believing the hold is its own and re-energizes at the next
        window, overriding them."""
        from executor.deferrable import CUT_BY_HUMAN

        assert self._who(held_by_us=True) == CUT_BY_HUMAN

    def test_a_device_reporting_itself_decides_nothing(self):
        """None means nobody commanded it (a reboot, an integration refresh), so we
        fall back to what we remember."""
        from executor.deferrable import CUT_BY_US

        assert self._who(context_user_id=None, held_by_us=True) == CUT_BY_US

    def test_unknown_when_we_remember_nothing_and_nobody_is_named(self):
        from executor.deferrable import CUT_BY_UNKNOWN

        assert self._who(context_user_id=None, held_by_us=False) == CUT_BY_UNKNOWN

    def test_undiscovered_own_id_never_claims_a_cut_as_ours(self):
        """Before discovery we cannot prove a named cut was not us — so we must not
        say it was. Callers treat unknown as a human cut."""
        from executor.deferrable import CUT_BY_UNKNOWN

        assert self._who(darkstar_user_id=None, held_by_us=False) == CUT_BY_UNKNOWN
