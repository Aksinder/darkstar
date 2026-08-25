"""The shortfall-risk gate that replaced the unconditional daily hot-water floor.

Driving case, live 2026-08-25 19:28: switch.vvb went ON at an import price of
3.56 SEK/kWh with the tank at 69.7 C — 388 litres of 40-degree water, several
showers' worth — purely because the day bucket had credited 0.000 of its
6.00 kWh. The gate must book nothing in that state, and must still book the
full floor when the tank really is low.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
import pytz

from planner.water_shortfall import (
    ShortfallConfig,
    dynamic_floor_kwh,
    hours_to_cheap_window,
    margin_showers_now,
    shower_kwh,
    usable_kwh_above_comfort,
)

TZ = pytz.timezone("Europe/Stockholm")

# The live house tank.
VOLUME_L = 195.0
# 69.7 C measured that evening -> stored energy above the 10 C inlet.
STORED_AT_69C = VOLUME_L * (4186.0 / 3600.0) * (69.7 - 10.0) / 1000.0  # ~13.5 kWh


def _cfg(**kw):
    base = dict(enabled=True, comfort_c=40.0, t_cold_c=10.0, shower_litres=60.0,
                margin_showers=1.5, max_horizon_hours=16.0)
    base.update(kw)
    return ShortfallConfig(**base)


@dataclass
class _Slot:
    start_time: datetime
    end_time: datetime
    pv_kwh: float = 0.0
    import_price_sek_kwh: float = 2.0


def _slots(now, hours=20, pv_from_h=None, pv_kwh=1.5, price=2.0, cheap_from_h=None,
           cheap_price=0.5):
    out = []
    for i in range(hours * 4):
        st = now + timedelta(minutes=15 * i)
        h = i / 4.0
        p = cheap_price if (cheap_from_h is not None and h >= cheap_from_h) else price
        pv = pv_kwh if (pv_from_h is not None and h >= pv_from_h) else 0.0
        out.append(_Slot(st, st + timedelta(minutes=15), pv, p))
    return out


class TestShowerArithmetic:
    def test_one_shower_is_about_two_kwh(self):
        # 60 L lifted from 10 to 40 C.
        assert shower_kwh(_cfg()) == pytest.approx(2.093, abs=0.01)

    def test_owner_margin_is_three_kwh(self):
        assert 1.5 * shower_kwh(_cfg()) == pytest.approx(3.14, abs=0.02)

    def test_degenerate_config_is_zero_not_negative(self):
        assert shower_kwh(_cfg(comfort_c=10.0)) == 0.0
        assert shower_kwh(_cfg(shower_litres=0.0)) == 0.0


class TestMarginWindow:
    """Owner rule: 1.5 showers from 18:00 on a weekday, 14:00 at the weekend."""

    def _at(self, y, m, d, hh, mm=0):
        return TZ.localize(datetime(y, m, d, hh, mm))

    def test_weekday_evening_reserves(self):
        # Tuesday 2026-08-25 20:05 — the incident.
        assert margin_showers_now(self._at(2026, 8, 25, 20, 5), _cfg()) == 1.5

    def test_weekday_afternoon_does_not(self):
        assert margin_showers_now(self._at(2026, 8, 25, 17, 59), _cfg()) == 0.0

    def test_weekday_boundary_is_inclusive(self):
        assert margin_showers_now(self._at(2026, 8, 25, 18, 0), _cfg()) == 1.5

    def test_weekend_starts_earlier(self):
        sat = self._at(2026, 8, 29, 14, 0)   # Saturday
        sun = self._at(2026, 8, 30, 15, 0)   # Sunday
        assert margin_showers_now(sat, _cfg()) == 1.5
        assert margin_showers_now(sun, _cfg()) == 1.5

    def test_weekend_before_fourteen_does_not(self):
        assert margin_showers_now(self._at(2026, 8, 29, 13, 30), _cfg()) == 0.0

    def test_friday_evening_uses_the_weekday_hour(self):
        # Friday is a weekday: 15:00 must NOT reserve, 18:00 must.
        assert margin_showers_now(self._at(2026, 8, 28, 15, 0), _cfg()) == 0.0
        assert margin_showers_now(self._at(2026, 8, 28, 18, 0), _cfg()) == 1.5


class TestUsableEnergy:
    def test_energy_below_comfort_is_not_usable(self):
        # A tank sitting exactly at comfort holds energy but zero showers.
        at_comfort = VOLUME_L * (4186.0 / 3600.0) * (40.0 - 10.0) / 1000.0
        assert usable_kwh_above_comfort(at_comfort, VOLUME_L, _cfg()) == pytest.approx(0.0)

    def test_the_live_tank_had_plenty(self):
        usable = usable_kwh_above_comfort(STORED_AT_69C, VOLUME_L, _cfg())
        assert usable > 6.0  # ~6.7 kWh = three showers beyond comfort

    def test_unknown_state_is_none_not_zero(self):
        assert usable_kwh_above_comfort(None, VOLUME_L, _cfg()) is None
        assert usable_kwh_above_comfort(5.0, None, _cfg()) is None


class TestCheapWindow:
    def test_finds_tomorrow_sun(self):
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        h = hours_to_cheap_window(_slots(now, pv_from_h=11.0, pv_kwh=1.5),
                                  now, heater_kw=3.4, max_horizon_hours=16.0)
        assert h == pytest.approx(11.0, abs=0.3)

    def test_a_cheap_night_hour_counts_too(self):
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        h = hours_to_cheap_window(_slots(now, cheap_from_h=6.0), now,
                                  heater_kw=3.4, max_horizon_hours=16.0)
        assert h == pytest.approx(6.0, abs=0.3)

    def test_weak_pv_does_not_count(self):
        # PV below the heater's own draw cannot run it for free.
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        h = hours_to_cheap_window(_slots(now, pv_from_h=11.0, pv_kwh=0.2), now,
                                  heater_kw=3.4, max_horizon_hours=16.0)
        assert h is None

    def test_naive_aware_mismatch_refuses(self):
        now = datetime(2026, 8, 25, 20, 0)  # naive
        slots = _slots(TZ.localize(now), pv_from_h=2.0)
        assert hours_to_cheap_window(slots, now, heater_kw=3.4, max_horizon_hours=16.0) is None


class TestTheGate:
    def _call(self, **kw):
        base = dict(
            configured_min_kwh=6.0, heated_today_kwh=0.0,
            stored_kwh=STORED_AT_69C, volume_litres=VOLUME_L,
            learned_draw_kw=0.15, standby_loss_kwh=0.3,
            hours_to_cheap=11.0,
            now_local=TZ.localize(datetime(2026, 8, 25, 20, 5)),
            cfg=_cfg(),
        )
        base.update(kw)
        return dynamic_floor_kwh(**base)

    def test_the_incident_books_nothing(self):
        # need = 0.15*11 + 0.3 + 3.14 = 5.09 kWh; have = 6.7 usable -> no shortfall.
        floor, reason = self._call()
        assert floor == 0.0, reason

    def test_a_low_tank_still_books(self):
        floor, _ = self._call(stored_kwh=7.0)  # barely above comfort
        assert floor > 0.0

    def test_never_exceeds_the_configured_floor(self):
        floor, _ = self._call(stored_kwh=0.5, learned_draw_kw=3.0)
        assert floor <= 6.0

    def test_heated_today_is_added_back_so_the_solver_cancels_it(self):
        # Kepler's bucket-0 constraint is (min_kwh_per_day - heated_today); the
        # shortfall already reflects today's heating via the tank state, so the
        # add-back prevents deducting it twice.
        floor, _ = self._call(stored_kwh=0.5, learned_draw_kw=3.0, heated_today_kwh=2.0)
        assert floor == pytest.approx(6.0 + 2.0)

    def test_disabled_is_a_passthrough(self):
        floor, reason = self._call(cfg=_cfg(enabled=False))
        assert floor == 6.0 and "disabled" in reason

    @pytest.mark.parametrize("missing", [
        {"stored_kwh": None}, {"volume_litres": None},
        {"learned_draw_kw": None}, {"hours_to_cheap": None},
    ])
    def test_unknown_input_keeps_the_old_floor(self, missing):
        floor, _ = self._call(**missing)
        assert floor == 6.0, "blind must mean heat as before, never skip"

    def test_afternoon_reserves_less_than_evening(self):
        afternoon = TZ.localize(datetime(2026, 8, 25, 12, 0))
        low = dict(stored_kwh=8.0, learned_draw_kw=0.4)
        f_eve, _ = self._call(**low)
        f_noon, _ = self._call(now_local=afternoon, **low)
        assert f_noon <= f_eve, "the evening reserve must be the stricter one"

    def test_flat_prices_mean_waiting_buys_nothing(self):
        # No discount anywhere and no sun: waiting cannot save money, so the gate
        # must find no window and fall back to the configured floor.
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        assert hours_to_cheap_window(_slots(now), now, heater_kw=3.4,
                                     max_horizon_hours=16.0) is None

    def test_a_marginally_cheaper_hour_does_not_count(self):
        # 10 % off is not worth a cold shower risk; the default demands 25 %.
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        slots = _slots(now, price=2.0, cheap_from_h=6.0, cheap_price=1.85)
        assert hours_to_cheap_window(slots, now, heater_kw=3.4,
                                     max_horizon_hours=16.0) is None

    def test_beyond_the_horizon_is_not_a_window(self):
        now = TZ.localize(datetime(2026, 8, 25, 20, 0))
        slots = _slots(now, hours=20, pv_from_h=18.0, pv_kwh=1.5)
        assert hours_to_cheap_window(slots, now, heater_kw=3.4,
                                     max_horizon_hours=16.0) is None
