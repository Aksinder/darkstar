

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


class TestRestoreDecision:
    """Owner, 2026-08-20: 240 min hand-back, "men skicka en actionable notifiering
    där jag får frågan om den fortsatt skall vara av eller om vi ska slå på igen".

    Asking is the honest default: restoring is the one part of the feature that
    energizes something a person deliberately switched off, so the choice goes back
    to them. Silence keeps the plug off — the safe outcome needs no tap.
    """

    def _decide(self, **kw):
        from executor.deferrable import restore_decision

        base = {
            "reclaim_due": True,
            "ask": True,
            "can_notify": True,
            "restore_asked_at": None,
            "now_ts": 100_000.0,
            "manual_cut_return_s": 14_400.0,
        }
        base.update(kw)
        return restore_decision(**base)

    def test_asks_instead_of_acting(self):
        assert self._decide() == "ask"

    def test_does_not_repeat_the_question_every_tick(self):
        assert self._decide(restore_asked_at=100_000.0 - 60.0) == "wait"

    def test_asks_again_one_interval_later(self):
        """A notification missed at 3am is not the end of it."""
        assert self._decide(restore_asked_at=100_000.0 - 14_400.0) == "ask"

    def test_no_notify_service_falls_back_to_restoring(self):
        """A promise of "hand it back after N minutes" that silently never fires
        would be worse than acting."""
        assert self._decide(can_notify=False) == "restore"

    def test_ask_disabled_restores_directly(self):
        assert self._decide(ask=False) == "restore"

    def test_nothing_due_nothing_happens(self):
        assert self._decide(reclaim_due=False) == "wait"


class TestWaitingIsNotFree:
    """Owner, 2026-08-22: the dishwasher armed at 13:27 was deferred to 01:30 the next
    morning over a 23:00 start that cost SEVEN ÖRE more. Formally optimal; not what
    anyone means by "run it when power is cheap".

    wait_cost_sek_per_hour prices the delay so the cheapest block only wins when it
    wins by more than the wait is worth.
    """

    # The real curve from that evening (hourly, SEK/kWh total): dear now, an evening
    # peak, then a long flat cheap night that keeps creeping down by a few öre.
    CURVE = [
        2.11, 2.05, 1.96, 1.64, 1.46,   # 15:00-19:00
        1.10, 1.00,                      # 22:00, 23:00
        0.98, 0.96, 0.95, 0.94, 0.94,   # 00:00-04:00
    ]
    ENERGY = 1.5  # kWh, ~2 h cycle

    def _slots(self):
        from executor.deferrable import WindowSlot

        return [WindowSlot(start_ts=i * 3600.0, import_price_sek_kwh=p)
                for i, p in enumerate(self.CURVE)]

    def _pick(self, wait_cost):
        from executor.deferrable import cheapest_window_start

        return cheapest_window_start(
            self._slots(), now_ts=0.0, duration_slots=2, deadline_ts=None,
            energy_kwh=self.ENERGY, wait_cost_sek_per_hour=wait_cost,
        )

    def test_off_by_default_keeps_chasing_the_last_ore(self):
        """The historical behaviour, pinned: the flat night's very cheapest block."""
        assert self._pick(0.0) == 10 * 3600.0  # 04:00-ish, the bottom of the curve

    def test_a_small_wait_price_takes_the_earlier_near_tie(self):
        """5 öre an hour is enough to stop paying hours for öre — it lands at the
        start of the cheap night instead of its deepest point."""
        picked = self._pick(0.05)
        assert picked is not None
        assert picked <= 7 * 3600.0

    def test_it_still_refuses_the_expensive_now(self):
        """The point is not impatience: avoiding the 2.11 peak is worth real money and
        must survive the wait price."""
        assert self._pick(0.05) != 0.0

    def test_a_large_wait_price_runs_immediately(self):
        """At 1 SEK/h the delay dwarfs any spread in this curve."""
        assert self._pick(1.0) == 0.0

    def test_no_energy_means_no_penalty(self):
        """Without kWh there is no way to turn a price curve into kronor; scoring a
        dimensionless sum against SEK/h would be nonsense, so it stays off."""
        from executor.deferrable import cheapest_window_start

        with_energy = cheapest_window_start(
            self._slots(), 0.0, 2, None, energy_kwh=0.0, wait_cost_sek_per_hour=1.0)
        assert with_energy == self._pick(0.0)

    def test_the_deadline_still_binds(self):
        from executor.deferrable import cheapest_window_start

        picked = cheapest_window_start(
            self._slots(), 0.0, 2, deadline_ts=4 * 3600.0,
            energy_kwh=self.ENERGY, wait_cost_sek_per_hour=0.05)
        assert picked is not None and picked + 2 * 3600.0 <= 4 * 3600.0

    def test_a_block_in_progress_is_not_credited_for_the_past(self):
        """Delay is clamped at zero: a slot that started before now must not earn a
        negative wait cost and win on that."""
        from executor.deferrable import WindowSlot, cheapest_window_start

        slots = [WindowSlot(start_ts=i * 3600.0, import_price_sek_kwh=p)
                 for i, p in enumerate([2.0, 1.0, 1.0])]
        # now is inside slot 0; slot 1-2 is genuinely cheaper and must win.
        assert cheapest_window_start(
            slots, now_ts=1800.0, duration_slots=2, deadline_ts=None,
            energy_kwh=1.0, wait_cost_sek_per_hour=0.05) == 3600.0

    def test_the_owners_exact_case(self):
        """23:00 vs 01:30 on the real numbers: 1.49 vs 1.42 SEK, 2.5 h apart.
        Break-even wait price is 0.07/2.5 = 2.8 öre per hour."""
        from executor.deferrable import WindowSlot, cheapest_window_start

        # Two candidates only, priced so the blocks cost exactly 1.49 and 1.42.
        slots = [
            WindowSlot(start_ts=0.0, import_price_sek_kwh=1.49 / 1.5),
            WindowSlot(start_ts=2.5 * 3600.0, import_price_sek_kwh=1.42 / 1.5),
        ]
        cheap_wins = cheapest_window_start(
            slots, 0.0, 1, None, energy_kwh=1.5, wait_cost_sek_per_hour=0.02)
        early_wins = cheapest_window_start(
            slots, 0.0, 1, None, energy_kwh=1.5, wait_cost_sek_per_hour=0.05)
        assert cheap_wins == 2.5 * 3600.0, "below break-even: still waits"
        assert early_wins == 0.0, "above break-even: runs earlier"
