"""Unseen-price-day reserve: sizing (pure), window assembly and the Kepler credit."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from planner.solver.kepler import KeplerSolver
from planner.solver.types import KeplerConfig, KeplerInput, KeplerInputSlot
from planner.strategy.unseen_price_reserve import (
    UnseenSlot,
    _merge,
    build_unseen_slots,
    size_unseen_reserve,
)

ETA = 0.95


def _day(evening_price: float, cheap_price: float, pv_surplus: float = 0.0) -> list[UnseenSlot]:
    """96 quarters: 4 h expensive evening with 1 kWh/slot net load, rest cheap."""
    out = []
    for i in range(96):
        evening = 68 <= i < 84
        out.append(
            UnseenSlot(
                hours=0.25,
                import_price=evening_price if evening else cheap_price,
                export_price=0.3,
                net_load_kwh=1.0 if evening else 0.0,
                pv_surplus_kwh=pv_surplus if 40 <= i < 60 else 0.0,
            )
        )
    return out


def _size(slots, **kw):
    base = {
        "max_charge_kw": 8.0,
        "max_discharge_kw": 8.0,
        "charge_eff": ETA,
        "discharge_eff": ETA,
        "wear_cost_sek_per_kwh": 0.2,
        "threshold_sek_per_kwh": 0.10,
        "battery_value_sek_per_kwh": 0.0,
        "usable_kwh": 14.4,
        "max_segments": 6,
    }
    base.update(kw)
    return size_unseen_reserve(slots, **base)


def test_expensive_unseen_evening_with_pricey_own_refill_is_credited():
    # Own-day refill costs 2.0/0.9025+0.2 = 2.42 > evening 3.0? no: min(3.0, 2.42).
    r = _size(_day(evening_price=3.0, cheap_price=2.0))
    assert r.active
    total = sum(e for e, _ in r.segments)
    # 16 kWh net load delivered -> 16/0.95 stored, capped at usable 14.4.
    assert total == pytest.approx(14.4, abs=0.01)
    expected = ETA * (min(3.0, 2.0 / ETA**2 + 0.2) - 0.1 - 0.1)
    assert r.segments[0][1] == pytest.approx(expected, abs=1e-3)


def test_cheap_own_day_refill_makes_reserve_worthless():
    # The unseen day can refill itself at 0.1: carrying energy over is worth little.
    r = _size(_day(evening_price=3.0, cheap_price=0.1), battery_value_sek_per_kwh=0.2)
    assert not r.active
    assert r.reason == "below_threshold"


def test_pv_surplus_on_unseen_day_counts_as_own_refill():
    no_pv = _size(_day(evening_price=3.0, cheap_price=2.0))
    sunny = _size(_day(evening_price=3.0, cheap_price=2.0, pv_surplus=2.0))
    # PV at export 0.3 refills the evening cheaply: far less credit.
    assert sum(e * v for e, v in sunny.segments) < 0.5 * sum(e * v for e, v in no_pv.segments)


def test_credit_is_incremental_over_battery_value():
    a = _size(_day(evening_price=3.0, cheap_price=2.0))
    b = _size(_day(evening_price=3.0, cheap_price=2.0), battery_value_sek_per_kwh=0.5)
    assert b.segments[0][1] == pytest.approx(a.segments[0][1] - 0.5, abs=1e-3)


def test_no_net_load_means_no_reserve():
    slots = [UnseenSlot(0.25, 3.0, 0.3, 0.0, 0.0)] * 96
    assert _size(slots).reason == "no_net_load"


def test_merge_keeps_energy_and_concavity():
    raw = [(0.5, 2.0 - 0.05 * i) for i in range(30)]
    merged = _merge(raw, 4)
    assert len(merged) == 4
    assert sum(e for e, _ in merged) == pytest.approx(15.0, abs=1e-3)
    assert sum(e * v for e, v in merged) == pytest.approx(sum(e * v for e, v in raw), abs=1e-2)
    values = [v for _, v in merged]
    assert values == sorted(values, reverse=True)


def _forecast_df(start: pd.Timestamp, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min")
    return pd.DataFrame({"load_forecast_kwh": 0.5, "pv_forecast_kwh": 0.1}, index=idx)


def test_build_unseen_slots_requires_coverage():
    start = pd.Timestamp("2026-11-04 23:45", tz="Europe/Stockholm")
    df = _forecast_df(start, 200)
    full = {start + pd.Timedelta(minutes=15 * (i + 1)): (2.0, 0.5) for i in range(96)}
    slots, why = build_unseen_slots(full, df, start, start + pd.Timedelta(hours=24))
    assert why is None and len(slots) == 96
    assert slots[0].net_load_kwh == pytest.approx(0.4)

    half = dict(list(full.items())[:48])
    slots, why = build_unseen_slots(half, df, start, start + pd.Timedelta(hours=24))
    assert slots == [] and why == "insufficient_price_forecast"

    slots, why = build_unseen_slots(full, None, start, start + pd.Timedelta(hours=24))
    assert why == "insufficient_load_forecast"


def _horizon(T: int = 16):
    start = datetime(2026, 11, 4, 20, 0)
    return [
        KeplerInputSlot(
            start_time=start + timedelta(minutes=15 * i),
            end_time=start + timedelta(minutes=15 * (i + 1)),
            load_kwh=0.0,
            pv_kwh=0.0,
            import_price_sek_kwh=1.0,
            export_price_sek_kwh=1.0,  # selling now pays 1.0
        )
        for i in range(T)
    ]


def _cfg(**kw) -> KeplerConfig:
    base = {
        "capacity_kwh": 16.0,
        "max_charge_power_kw": 8.0,
        "max_discharge_power_kw": 8.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "min_soc_percent": 10.0,
        "max_soc_percent": 100.0,
        "wear_cost_sek_per_kwh": 0.0,
        "max_export_power_kw": 8.0,
        "enable_export": True,
    }
    base.update(kw)
    return KeplerConfig(**base)


def test_kepler_credit_holds_energy_worth_more_than_selling_now():
    inp = KeplerInput(slots=_horizon(), initial_soc_kwh=12.0)
    plain = KeplerSolver().solve(inp, _cfg())
    assert plain.is_optimal
    # Without the credit the pack is sold down to min_soc (1.6 kWh).
    assert plain.slots[-1].soc_kwh == pytest.approx(1.6, abs=0.05)

    held = KeplerSolver().solve(inp, _cfg(terminal_value_segments=[(4.0, 1.5), (4.0, 0.5)]))
    assert held.is_optimal
    # First 4 kWh above min_soc are worth 1.5 > 1.0 export -> kept; the next 4 at 0.5
    # are not -> sold.
    assert held.slots[-1].soc_kwh == pytest.approx(1.6 + 4.0, abs=0.05)
    # The credit is objective-only: reported plan cost is still pure energy cost.
    assert held.total_cost_sek == pytest.approx(-(12.0 - 5.6) * 1.0, abs=0.05)


def test_kepler_credit_never_buys_grid_that_does_not_pay():
    slots = [
        KeplerInputSlot(
            start_time=s.start_time,
            end_time=s.end_time,
            load_kwh=0.0,
            pv_kwh=0.0,
            import_price_sek_kwh=2.0,
            export_price_sek_kwh=0.5,
        )
        for s in _horizon()
    ]
    r = KeplerSolver().solve(
        KeplerInput(slots=slots, initial_soc_kwh=1.6),
        _cfg(terminal_value_segments=[(8.0, 1.5)]),
    )
    assert r.is_optimal
    assert sum(s.grid_import_kwh for s in r.slots) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.asyncio
async def test_pipeline_helper_sets_segments_and_degrades_quietly(monkeypatch):
    import planner.strategy.unseen_price_reserve as upr
    from planner.pipeline import _apply_unseen_price_reserve

    tz = "Europe/Stockholm"
    horizon_end = pd.Timestamp("2026-11-04 23:45", tz=tz)
    df = _forecast_df(horizon_end, 120)
    df.loc[df.index[70:86], "load_forecast_kwh"] = 1.5  # expensive evening load
    prices = {
        horizon_end + pd.Timedelta(minutes=15 * (i + 1)): (3.0 if 69 <= i < 85 else 2.0, 0.4)
        for i in range(96)
    }
    monkeypatch.setattr(upr, "load_unseen_prices", lambda *a, **k: prices)
    cfg = _cfg(charge_efficiency=ETA, discharge_efficiency=ETA, wear_cost_sek_per_kwh=0.2)
    fine = _horizon()
    await _apply_unseen_price_reserve(cfg, {"enabled": True}, {}, fine, df, horizon_end)
    assert cfg.terminal_value_segments
    assert sum(e for e, _ in cfg.terminal_value_segments) > 5.0

    monkeypatch.setattr(upr, "load_unseen_prices", lambda *a, **k: {})
    cfg2 = _cfg()
    await _apply_unseen_price_reserve(cfg2, {"enabled": True}, {}, fine, df, horizon_end)
    assert cfg2.terminal_value_segments == []


def test_load_unseen_prices_reads_latest_forecast_in_window(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from backend.learning.models import Base, PriceForecast
    from planner.strategy.unseen_price_reserve import load_unseen_prices

    db = tmp_path / "l.db"
    eng = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(eng)
    start = datetime(2026, 11, 5, 0, 0)
    with Session(eng) as s:
        for i in range(100):
            slot = (start + timedelta(minutes=15 * i)).isoformat()
            s.add(
                PriceForecast(
                    slot_start=slot,
                    issue_timestamp="2026-11-03T06:00:00",
                    days_ahead=2,
                    spot_p50=0.5,
                )
            )
            s.add(
                PriceForecast(
                    slot_start=slot,
                    issue_timestamp="2026-11-04T14:00:00",
                    days_ahead=1,
                    spot_p50=1.0,
                )
            )
        s.commit()
    eng.dispose()

    config = {
        "timezone": "Europe/Stockholm",
        "pricing": {"vat_percent": 25.0, "grid_transfer_fee_sek": 0.4, "energy_tax_sek": 0.4},
    }
    ws = pd.Timestamp("2026-11-04 23:45", tz="Europe/Stockholm")
    out = load_unseen_prices(config, str(db), ws, ws + pd.Timedelta(hours=24))
    assert len(out) == 96
    imp, exp = next(iter(out.values()))
    assert exp == pytest.approx(1.0)  # model output == effective export price
    assert imp == pytest.approx((1.0 + 0.4 + 0.4) * 1.25)  # latest issue wins
