"""The deadline guarantee buys early and cheap instead of late and dear.

Live driver (owner, 2026-08-31): the Tesla came home at 23:36 below its 40 %
weekday guarantee and ``_deadline_required_w`` — which defers grid until the
required average power crosses the charger minimum, i.e. until time nearly runs
out — crept along at 7 A straight through the morning ramp. It bought at 2.07 and
2.14 SEK/kWh at 06:30-06:45 while 1.0484 sat unused at 05:00.

Owner directive: "Om priset är lågt och det kommer att bli dyrare är det bättre
om vi laddar så mycket som möjligt så fort som möjligt."

The prices below are the REAL SE3 import curve from that night, so a regression
reproduces the actual incident rather than a synthetic one.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import pytz

from executor.ev_surplus import (
    EVSurplusConfig,
    EVSurplusInputs,
    _deadline_required_w,
    _now_is_in_cheapest_slots,
    compute_ev_surplus,
)
from tests.executor.test_ev_surplus_priority import _tesla

TZ = pytz.timezone("Europe/Stockholm")
CFG = EVSurplusConfig(enabled=True)

# 30->31 Aug 2026, SE3, all-in import price per 15-min slot — read back from
# sensor.total_price_per_kwh_sek. The cheapest quarter of the whole night is
# 05:00 (1.0484); the car was actually charged at 06:30-06:45 (2.07, 2.14).
NIGHT: list[tuple[str, float]] = [
    ("00:30", 1.2047), ("00:45", 1.2236), ("01:00", 1.2079), ("01:15", 1.2112),
    ("01:30", 1.2238), ("01:45", 1.2204), ("02:00", 1.2232), ("02:15", 1.2238),
    ("02:30", 1.2503), ("02:45", 1.2783), ("03:00", 1.1945), ("03:15", 1.1988),
    ("03:30", 1.2066), ("03:45", 1.2112), ("04:00", 1.1447), ("04:15", 1.1750),
    ("04:30", 1.2572), ("04:45", 1.4136), ("05:00", 1.0484), ("05:15", 1.2112),
    ("05:30", 1.4509), ("05:45", 1.7720), ("06:00", 1.2677), ("06:15", 1.6926),
    ("06:30", 2.0672), ("06:45", 2.1361),
]
DEADLINE = "07:30"  # recurring_deadline_time, mon-fri; 31 Aug 2026 is a Monday

# Tesla at 30 % toward a 40 % floor: 10 % of 60 kWh / 0.9 = 6.67 kWh at the plug.
# At 16 A x 3 x 230 V = 11 040 W that is ~0.60 h — under three quarters, so the
# selection has real work to do.
_TESLA_MAX_W = 16.0 * 3 * 230.0


def _ts(hhmm: str, *, second: int = 0) -> float:
    h, m = (int(x) for x in hhmm.split(":"))
    return TZ.localize(datetime(2026, 8, 31, h, m, second)).timestamp()


def _curve() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (_ts(hhmm), _ts(hhmm) + 900.0, price) for hhmm, price in NIGHT
    )


def _mid(hhmm: str) -> float:
    """Epoch a few minutes into the named slot — a realistic tick instant."""
    return _ts(hhmm, second=30) + 120.0


def _hours_left(now_ts: float) -> float:
    return (_ts(DEADLINE) - now_ts) / 3600.0


def _tesla_at(hhmm: str, **over):
    """The incident's Tesla, with deadline_hours consistent with the tick instant."""
    over.setdefault("soc_percent", 30.0)
    return _tesla(deadline_hours=_hours_left(_mid(hhmm)), **over)


def _required(hhmm: str, **over) -> float:
    now = _mid(hhmm)
    return _deadline_required_w(_tesla_at(hhmm, **over), CFG, now, _curve())


def _legacy(hhmm: str, **over) -> float:
    """Same charger, no curve — what the code did before this change."""
    return _deadline_required_w(_tesla_at(hhmm, **over), CFG)


class TestTheIncident:
    """The exact night that prompted the change."""

    def test_charges_flat_out_in_the_cheapest_quarter(self):
        # 05:00 at 1.0484 is the night's minimum — take everything there.
        assert _required("05:00") == pytest.approx(_TESLA_MAX_W)

    def test_charges_flat_out_in_the_next_cheapest(self):
        # 6.67 kWh needs ~0.60 h, so the three cheapest quarters are all required:
        # 05:00 (1.0484), 04:00 (1.1447), 04:15 (1.1750).
        assert _required("04:00") == pytest.approx(_TESLA_MAX_W)
        assert _required("04:15") == pytest.approx(_TESLA_MAX_W)

    @pytest.mark.parametrize("hhmm", ["00:30", "01:00", "03:00", "03:45"])
    def test_waits_while_cheaper_quarters_are_still_ahead(self, hhmm):
        # 03:00 (1.1945) is the fourth-cheapest of the night — a deliberate near
        # miss: three cheaper quarters cover the 0.60 h requirement, so even this
        # one must wait.
        assert _required(hhmm) == 0.0

    def test_the_peak_quarters_were_the_dearest_of_the_window(self):
        # Guards the fixture: if the curve is ever edited so 06:30/06:45 stop being
        # the expensive end, the simulation below stops meaning anything.
        assert max(NIGHT, key=lambda p: p[1])[0] == "06:45"

    def test_late_in_the_window_it_still_buys_what_is_left(self):
        # By 06:30 nothing cheaper remains before 07:30, so a car still short of
        # its floor SHOULD buy here. Declining would just miss the deadline — the
        # feature moves energy earlier, it never refuses to honour the guarantee.
        assert _required("06:30") > 0.0

    def test_the_energy_is_secured_before_the_ramp_begins(self):
        # Full power across 04:00, 04:15 and 05:00 is 3 x 0.25 h x 11.04 kW =
        # 8.28 kWh at the plug, over the 6.67 kWh needed — so by 05:30, where the
        # price first breaks 1.45, the gap is closed and nothing is bought.
        assert 3 * 0.25 * _TESLA_MAX_W / 1000.0 > (40.0 - 30.0) / 100.0 * 60.0 / 0.9


class TestTheNightSimulated:
    """Walk the real curve tick by tick and price the outcome both ways.

    This is the test that speaks to the owner's actual goal. Everything else
    checks a single decision; this one checks that the decisions compose into a
    cheaper night that still meets the guarantee.
    """

    # The simulated guarantee lands at 07:00, the end of the priced data, so both
    # policies get exactly the same runway. (The live deadline is 07:30; extending
    # the walk past the curve would hand the legacy ramp two quarters at invented
    # prices, which is precisely the comparison this test must not make.)
    SIM_DEADLINE = "07:00"

    @classmethod
    def _walk(cls, *, with_curve: bool, stop_before: str = "99:99") -> tuple[float, float]:
        """Returns (SEK spent, final SoC %) charging under the given policy."""
        curve = _curve()
        soc, cost = 30.0, 0.0
        for hhmm, price in NIGHT:
            if hhmm >= stop_before:
                break
            now = _mid(hhmm)
            c = _tesla(
                soc_percent=soc, deadline_hours=(_ts(cls.SIM_DEADLINE) - now) / 3600.0
            )
            w = (
                _deadline_required_w(c, CFG, now, curve)
                if with_curve
                else _deadline_required_w(c, CFG)
            )
            kwh = w / 1000.0 * 0.25  # one quarter at the commanded power
            cost += kwh * price
            soc += kwh * c.charge_efficiency / c.capacity_kwh * 100.0
        return cost, soc

    def test_both_policies_reach_the_guarantee(self):
        for with_curve in (True, False):
            _, soc = self._walk(with_curve=with_curve)
            assert soc >= 40.0 - 1e-6, f"guarantee missed (curve={with_curve}, soc={soc:.1f})"

    def test_front_loading_buys_at_a_lower_average_price(self):
        # The claim, stated so it cannot be satisfied by simply buying less energy.
        front_cost, front_soc = self._walk(with_curve=True)
        legacy_cost, legacy_soc = self._walk(with_curve=False)
        front = front_cost / ((front_soc - 30.0) / 100.0 * 60.0 / 0.9)
        legacy = legacy_cost / ((legacy_soc - 30.0) / 100.0 * 60.0 / 0.9)
        assert front < legacy, f"{front:.3f} SEK/kWh front-loaded vs {legacy:.3f} legacy"

    def test_front_loading_spends_nothing_in_the_morning_peak(self):
        # The three quarters that cost the most (1.69, 2.07, 2.14). Front-loading
        # must have finished long before them; the legacy ramp is still buying.
        peak = {"06:15", "06:30", "06:45"}
        before_peak = min(peak)
        front_all, _ = self._walk(with_curve=True)
        front_pre, _ = self._walk(with_curve=True, stop_before=before_peak)
        legacy_all, _ = self._walk(with_curve=False)
        legacy_pre, _ = self._walk(with_curve=False, stop_before=before_peak)
        assert front_all - front_pre == pytest.approx(0.0, abs=1e-9)
        assert legacy_all - legacy_pre > 0.0, "fixture broken: legacy should still be buying"

    def test_front_loading_finishes_before_the_morning_ramp(self):
        # The whole point: the guarantee is met while energy is still cheap.
        # 05:30 is where the price first breaks 1.45 and never comes back down.
        _, soc = self._walk(with_curve=True, stop_before="05:30")
        assert soc >= 40.0, f"still at {soc:.1f}% when the morning ramp starts"


class TestSafetyProperty:
    """Front-loading may only ever charge MORE than the legacy ramp, never less.

    This is the property that makes the feature safe to ship: a stale or wrong
    price curve can make charging costlier, but it cannot put the deadline at
    risk, because the urgency ramp underneath is untouched.
    """

    @pytest.mark.parametrize("hhmm", [h for h, _ in NIGHT])
    def test_never_below_legacy_at_any_hour(self, hhmm):
        assert _required(hhmm) >= _legacy(hhmm) - 1e-6

    @pytest.mark.parametrize("soc", [0.0, 15.0, 30.0, 39.0])
    def test_never_below_legacy_at_any_soc(self, soc):
        for hhmm, _ in NIGHT:
            assert _required(hhmm, soc_percent=soc) >= _legacy(hhmm, soc_percent=soc) - 1e-6

    def test_urgency_ramp_still_fires_when_the_curve_declines_the_slot(self):
        # A car still far below its floor with minutes to go: the slot is dear and
        # front-loading declines it, but the legacy guarantee must still force grid.
        w = _required("06:45", soc_percent=10.0)
        assert w > 0.0, "the deadline guarantee must survive an expensive final slot"


class TestCurveIsInert:
    """With no usable curve the function is byte-for-byte its old self."""

    @pytest.mark.parametrize("hhmm", [h for h, _ in NIGHT])
    def test_no_curve_matches_legacy(self, hhmm):
        now = _mid(hhmm)
        c = _tesla_at(hhmm)
        assert _deadline_required_w(c, CFG, now, ()) == _deadline_required_w(c, CFG)

    def test_zero_now_ts_matches_legacy(self):
        c = _tesla_at("05:00")
        assert _deadline_required_w(c, CFG, 0.0, _curve()) == _deadline_required_w(c, CFG)

    def test_now_outside_the_series_matches_legacy(self):
        # A curve that has gone stale (all slots yesterday) must not be consulted.
        stale = tuple((s - 86400.0, e - 86400.0, p) for s, e, p in _curve())
        now = _mid("05:00")
        c = _tesla_at("05:00")
        assert _deadline_required_w(c, CFG, now, stale) == _deadline_required_w(c, CFG)

    def test_soc_at_or_above_floor_never_charges(self):
        # No matter how cheap: the guarantee is a floor, not an invitation to fill.
        assert _required("05:00", soc_percent=40.0) == 0.0
        assert _required("05:00", soc_percent=95.0) == 0.0

    def test_floor_is_clamped_to_the_comfort_cap(self):
        # A floor above the cap must not size charging toward an unreachable SoC.
        assert _required("05:00", soc_percent=50.0, target_soc_percent=45.0,
                         floor_soc_percent=90.0) == 0.0


class TestSelection:
    """The slot picker itself."""

    def test_a_tie_favours_charging_now(self):
        # Past slots are clipped away, so the current slot is always the EARLIEST
        # survivor — and sorting breaks ties by start time. Equal price therefore
        # always resolves to "buy now", which is the safer half of the tie.
        flat = ((0.0, 900.0, 1.0), (900.0, 1800.0, 1.0))
        assert _now_is_in_cheapest_slots(1.9, 8000.0, 100.0, 1800.0, flat) is True

    def test_a_strictly_cheaper_later_slot_wins(self):
        curve = ((0.0, 900.0, 1.0), (900.0, 1800.0, 0.5))
        assert _now_is_in_cheapest_slots(1.9, 8000.0, 100.0, 1800.0, curve) is False

    def test_a_window_too_short_to_be_picky_charges_now(self):
        # Requirement exceeds everything on offer => every slot is needed.
        dear = ((0.0, 900.0, 99.0),)
        assert _now_is_in_cheapest_slots(50.0, 8000.0, 100.0, 900.0, dear) is True

    def test_a_slot_straddling_now_counts_only_its_remainder(self):
        # now sits 12 of 15 minutes into the cheap slot, leaving 0.05 h there. The
        # requirement needs 0.20 h, so the dearer following slot is needed too.
        curve = ((0.0, 900.0, 1.0), (900.0, 1800.0, 5.0))
        assert _now_is_in_cheapest_slots(2.208, 11040.0, 720.0, 1800.0, curve) is True
        # ...and from inside the dear slot, it is still needed (nothing cheaper left).
        assert _now_is_in_cheapest_slots(2.208, 11040.0, 1000.0, 1800.0, curve) is True

    @pytest.mark.parametrize(
        "args",
        [
            (6.0, 11040.0, 100.0, 1800.0, ()),          # no curve
            (6.0, 0.0, 100.0, 1800.0, ((0.0, 900.0, 1.0),)),   # no max power
            (0.0, 11040.0, 100.0, 1800.0, ((0.0, 900.0, 1.0),)),  # nothing needed
            (6.0, 11040.0, 1800.0, 900.0, ((0.0, 900.0, 1.0),)),  # deadline in the past
        ],
    )
    def test_undecidable_inputs_return_none(self, args):
        assert _now_is_in_cheapest_slots(*args) is None

    def test_unsorted_and_gapped_curves_are_handled(self):
        # The engine filters past slots out, so the curve may start mid-series and
        # need not be contiguous; order is never guaranteed by contract.
        curve = ((1800.0, 2700.0, 0.5), (0.0, 900.0, 9.0))
        assert _now_is_in_cheapest_slots(1.0, 11040.0, 2000.0, 2700.0, curve) is True


class TestWiredThrough:
    """The pure core is reached from compute_ev_surplus with a real inputs object."""

    def _run(self, hhmm: str, curve):
        now = _mid(hhmm)
        return compute_ev_surplus(
            EVSurplusInputs(
                pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
                import_price_sek=1.0, remaining_solar_kwh=0.0,
                chargers=[_tesla_at(hhmm)], now_ts=now, forward_prices=curve,
            ),
            CFG,
        )

    def test_cheap_slot_commands_full_power_with_zero_surplus(self):
        cmds = self._run("05:00", _curve())
        assert cmds[0].switch_on
        assert cmds[0].target_power_w == pytest.approx(_TESLA_MAX_W)

    def test_early_slot_with_cheaper_ones_ahead_stays_off(self):
        cmds = self._run("01:00", _curve())
        assert not cmds[0].switch_on

    def test_the_main_fuse_still_trumps_a_front_loaded_floor(self):
        """CRITICAL. Front-loading commands FULL power, so it makes two cars wanting
        max in the same cheap quarter far more likely than the old ramp ever did.
        The 25 A/phase guard must clamp them — a grid-backed guarantee may cost
        money, but it may never blow the main fuse."""
        from tests.executor.test_ev_surplus_priority import _fmb

        now = _mid("05:00")
        hours = _hours_left(now)
        tesla = _tesla(soc_percent=30.0, deadline_hours=hours, phase_map=("a", "b", "c"))
        fmb = _fmb(soc_percent=50.0, floor_soc_percent=86.0, deadline_hours=hours,
                   phase_map=("a",))
        cmds = compute_ev_surplus(
            EVSurplusInputs(
                pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
                import_price_sek=1.0, remaining_solar_kwh=0.0,
                phase_currents_a={"a": 5.0, "b": 5.0, "c": 5.0},
                chargers=[tesla, fmb], now_ts=now, forward_prices=_curve(),
            ),
            EVSurplusConfig(enabled=True, fuse_budget_a=23.0),
        )
        # Both want max (16 A each); phase A can only carry 23 - 5 = 18 A of new load.
        on_a = sum(
            c.set_current_a or 0.0
            for c in cmds
            if c.switch_on and c.id in {"tesla", "easee_fmb"}
        )
        assert on_a <= 18.0 + 1e-6, f"phase A over budget: {on_a:.1f} A"
        assert any(c.fuse_limited for c in cmds), "the clamp must be recorded"

    def test_default_inputs_carry_no_curve(self):
        # Every existing caller constructs EVSurplusInputs without the new fields;
        # they must keep the legacy ramp.
        i = EVSurplusInputs(
            pv_w=0.0, grid_w=0.0, battery_w=0.0, battery_soc_percent=50.0,
            import_price_sek=1.0, remaining_solar_kwh=0.0, chargers=[],
        )
        assert i.now_ts == 0.0
        assert i.forward_prices == ()
