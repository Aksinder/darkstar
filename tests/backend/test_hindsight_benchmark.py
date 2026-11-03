"""Nightly battery hindsight benchmark: pure scoring + service scheduling."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pytest
import pytz

from backend.hindsight_benchmark import (
    benchmark_day,
    rows_by_local_day,
    state_key,
    summarize,
)
from backend.services.benchmark_service import (
    BenchmarkService,
    benchmark_settings,
    next_run_after,
)
from planner.solver.types import KeplerConfig

TZ = pytz.timezone("Europe/Stockholm")
CAP = 16.0


def _config() -> KeplerConfig:
    return KeplerConfig(
        capacity_kwh=CAP,
        min_soc_percent=10.0,
        max_soc_percent=100.0,
        max_charge_power_kw=9.5,
        max_discharge_power_kw=9.5,
        charge_efficiency=0.975,
        discharge_efficiency=0.975,
        wear_cost_sek_per_kwh=0.2,
    )


def _price(hour: int) -> float:
    return 0.5 if hour < 6 else (3.0 if 17 <= hour < 21 else 1.2)


def _day_rows(
    day: date,
    *,
    battery: str = "idle",
    soc0: float = 50.0,
    flags: dict[int, dict[str, Any]] | None = None,
    drop: int | None = None,
) -> list[dict[str, Any]]:
    """Recorded quarters for one local day. battery: idle | arbitrage."""
    start = TZ.localize(datetime.combine(day, time()))
    end = TZ.localize(datetime.combine(day + timedelta(days=1), time()))
    rows = []
    soc = soc0
    t = start
    i = 0
    while t < end:
        local = t.astimezone(TZ)
        load = 0.4
        pv = 1.0 if 10 <= local.hour < 15 else 0.0
        ch = dis = 0.0
        if battery == "arbitrage":
            if local.hour < 2:
                ch = 1.0
            elif 17 <= local.hour < 19:
                dis = 0.4
        net = load - pv + ch - dis
        soc += (ch * 0.975 - dis / 0.975) / CAP * 100
        row = {
            "slot_start": t.isoformat(),
            "slot_end": (t + timedelta(minutes=15)).astimezone(TZ).isoformat(),
            "import_kwh": max(net, 0.0),
            "export_kwh": max(-net, 0.0),
            "pv_kwh": pv,
            "load_kwh": load,
            "water_kwh": 0.0,
            "ev_charging_kwh": 0.0,
            "batt_charge_kwh": ch,
            "batt_discharge_kwh": dis,
            "soc_end_percent": soc,
            "import_price_sek_kwh": _price(local.hour),
            "export_price_sek_kwh": _price(local.hour) * 0.5,
            "quality_flags": json.dumps((flags or {}).get(i)) if (flags or {}).get(i) else None,
        }
        if i != drop:
            rows.append(row)
        t = (t + timedelta(minutes=15)).astimezone(TZ)
        i += 1
    return rows


DAY = date(2026, 10, 20)
NOW = TZ.localize(datetime(2026, 10, 22, 3, 30))


def _by_day(day_rows: list[dict[str, Any]]) -> dict[date, list[dict[str, Any]]]:
    prev = _day_rows(DAY - timedelta(days=1))
    return rows_by_local_day(prev + day_rows, TZ)


def test_scores_day_and_hindsight_beats_actual():
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY, battery="arbitrage")), _config(), TZ, NOW)
    assert rec["status"] == "scored", rec
    assert rec["day_slots"] == 96
    # Hindsight is at least as good as reality and grid-only.
    assert rec["hindsight_cost_sek"] <= rec["actual_cost_sek"] + 1e-6
    assert rec["value_hindsight_sek"] >= rec["value_actual_sek"] - 1e-6
    # Actual cost reconstructs the metered bill plus wear.
    wear = 0.2 * 0.5 * (8 * 1.0 + 8 * 0.4)
    assert rec["actual_cost_sek"] == pytest.approx(rec["metered_cost_sek"] + wear, abs=1e-3)
    assert rec["warnings"] == []


def test_idle_battery_matches_grid_only():
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY)), _config(), TZ, NOW)
    assert rec["status"] == "scored"
    assert rec["actual_cost_sek"] == pytest.approx(rec["grid_only_cost_sek"], abs=1e-6)
    assert rec["stored_change_kwh"] == pytest.approx(0.0, abs=1e-6)


def test_cyclic_loads_are_part_of_demand():
    """The recorder takes the pool pump out of load_kwh; the meter still paid for it.
    Idle battery: base 0.3 + pump 0.1 = the 0.4 the meter imported, so actual must
    still equal grid-only with no phantom battery flow. A missing (legacy, unfilled)
    cyclic_kwh counts as 0 and is NOT an unverified slot."""
    rows = _day_rows(DAY)
    for r in rows:
        r["load_kwh"] = 0.3
        r["cyclic_kwh"] = 0.1
    rec = benchmark_day(DAY, _by_day(rows), _config(), TZ, NOW)
    assert rec["status"] == "scored" and rec["unverified_slots"] == 0
    assert rec["actual_cost_sek"] == pytest.approx(rec["grid_only_cost_sek"], abs=1e-6)
    assert rec["grid_only_cost_sek"] == pytest.approx(rec["metered_cost_sek"], abs=1e-6)
    assert rec["stored_change_kwh"] == pytest.approx(0.0, abs=1e-6)


def test_load_rescued_is_clean_but_grid_unmeasured_counts():
    rescued = {key: {"load_rescued": "energy_balance"} for key in range(96)}
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY, flags=rescued)), _config(), TZ, NOW)
    assert rec["status"] == "scored" and rec["unverified_slots"] == 0

    few = {i: {"grid_unmeasured": True} for i in range(3)}
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY, flags=few)), _config(), TZ, NOW)
    assert rec["status"] == "scored" and "unverified_slots" in rec["warnings"]

    many = {i: {"soc_stale": True} for i in range(10)}
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY, flags=many)), _config(), TZ, NOW)
    assert rec["status"] == "excluded" and rec["reason"] == "not_clean_recording"


def test_incomplete_and_unfinished_days_are_excluded():
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY, drop=40)), _config(), TZ, NOW)
    assert rec["reason"] == "incomplete_day"
    early = TZ.localize(datetime(2026, 10, 20, 23, 0))
    rec = benchmark_day(DAY, _by_day(_day_rows(DAY)), _config(), TZ, early)
    assert rec["reason"] == "not_finished"


def test_start_soc_back_derived_without_previous_day():
    rows = rows_by_local_day(_day_rows(DAY, battery="arbitrage"), TZ)
    rec = benchmark_day(DAY, rows, _config(), TZ, NOW)
    assert rec["status"] == "scored"
    assert "start_soc_back_derived" in rec["warnings"]
    assert rec["start_soc_kwh"] == pytest.approx(8.0, abs=0.01)


def test_dst_day_has_100_slots():
    dst_day = date(2026, 10, 25)  # Europe/Stockholm falls back: 25 h
    prev = _day_rows(dst_day - timedelta(days=1))
    rows = _day_rows(dst_day)
    assert len(rows) == 100
    rec = benchmark_day(
        dst_day, rows_by_local_day(prev + rows, TZ), _config(), TZ, NOW + timedelta(days=5)
    )
    assert rec["status"] == "scored" and rec["day_slots"] == 100


def test_summarize_totals():
    a = benchmark_day(DAY, _by_day(_day_rows(DAY, battery="arbitrage")), _config(), TZ, NOW)
    b = {"status": "excluded", "reason": "incomplete_day", "warnings": []}
    s = summarize([a, b])
    assert s["days_scored"] == 1 and s["days_total"] == 2
    assert s["by_reason"] == {"incomplete_day": 1}
    assert s["missed_sek"] == pytest.approx(a["missed_sek"], abs=1e-3)


# ---------------------------------------------------------------- service


def test_settings_and_next_run_are_local_and_tz_aware():
    enabled, run_at, window = benchmark_settings({})
    assert (enabled, run_at, window) == (True, time(3, 30), 14)
    assert benchmark_settings(
        {"learning": {"hindsight_benchmark": {"run_at": "04:15", "enabled": False}}}
    )[:2] == (False, time(4, 15))
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    nxt = next_run_after(now, time(3, 30), TZ)
    assert nxt.isoformat() == "2026-10-25T03:30:00+01:00"  # after the DST fall-back
    # 02:00 local (just after fall-back) -> today 03:30; 04:00 local -> tomorrow.
    nxt = next_run_after(datetime(2026, 10, 25, 1, 0, tzinfo=UTC), time(3, 30), TZ)
    assert nxt.isoformat() == "2026-10-25T03:30:00+01:00"
    nxt = next_run_after(datetime(2026, 10, 25, 3, 0, tzinfo=UTC), time(3, 30), TZ)
    assert nxt.isoformat() == "2026-10-26T03:30:00+01:00"


class _FakeStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.state: dict[str, str] = {}

    async def get_system_states(self, keys: list[str]) -> dict[str, str]:
        return {k: self.state[k] for k in keys if k in self.state}

    async def get_slot_observation_rows(self, start: datetime, end: datetime):
        return [r for r in self.rows if start <= datetime.fromisoformat(r["slot_start"]) < end]

    async def set_system_state(self, key: str, value: str) -> None:
        self.state[key] = value


def _service_config() -> dict[str, Any]:
    return {
        "timezone": "Europe/Stockholm",
        "battery": {
            "capacity_kwh": CAP,
            "min_soc_percent": 10,
            "max_soc_percent": 100,
            "max_charge_a": 100,
            "max_discharge_a": 100,
            "nominal_voltage_v": 95,
        },
        "learning": {"hindsight_benchmark": {"window_days": 3}},
    }


def test_service_scores_pending_days_once_and_waits_for_planner():
    rows = []
    for k in range(4, 0, -1):
        rows += _day_rows(date(2026, 10, 22) - timedelta(days=k), battery="arbitrage")
    store = _FakeStore(rows)
    busy = iter([True, False])
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    svc = BenchmarkService(
        config_loader=_service_config,
        now=lambda: NOW.astimezone(UTC),
        planner_busy=lambda: next(busy, False),
    )

    async def go() -> tuple[list[date], list[date]]:
        import backend.services.benchmark_service as mod

        orig = mod.asyncio.sleep
        mod.asyncio.sleep = fake_sleep  # type: ignore[assignment]
        try:
            first = await svc.run_once(store)  # type: ignore[arg-type]
            second = await svc.run_once(store)  # type: ignore[arg-type]
        finally:
            mod.asyncio.sleep = orig  # type: ignore[assignment]
        return first, second

    first, second = asyncio.run(go())
    assert first == [date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)]
    assert second == []
    assert sleeps, "waited for the busy planner"
    rec = json.loads(store.state[state_key(date(2026, 10, 20))])
    assert rec["status"] == "scored"


def test_service_fills_legacy_cyclic_from_device_store():
    """Rows without the cyclic_kwh column get the pump energy from slot_device_energy,
    so the scored grid-only cost matches the meter on an idle-battery day."""
    rows = []
    for k in range(4, 0, -1):
        rows += _day_rows(date(2026, 10, 22) - timedelta(days=k))
    dev = []
    for r in rows:
        r["load_kwh"] = 0.3  # recorder took the 0.1 pump out
        dev.append({"slot_start": r["slot_start"], "device_id": "poolpump", "kwh": 0.1})
    store = _FakeStore(rows)

    async def get_dev(start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        return dev

    store.get_device_energy_rows_between = get_dev  # type: ignore[attr-defined]
    cfg = _service_config()
    cfg["cyclic_loads"] = [{"id": "poolpump", "enabled": True}]
    svc = BenchmarkService(
        config_loader=lambda: cfg, now=lambda: NOW.astimezone(UTC), planner_busy=lambda: False
    )
    asyncio.run(svc.run_once(store))  # type: ignore[arg-type]
    rec = json.loads(store.state[state_key(date(2026, 10, 20))])
    assert rec["status"] == "scored"
    assert rec["grid_only_cost_sek"] == pytest.approx(rec["metered_cost_sek"], abs=1e-6)


def test_service_disabled_does_nothing():
    cfg = _service_config()
    cfg["learning"]["hindsight_benchmark"]["enabled"] = False
    svc = BenchmarkService(config_loader=lambda: cfg, now=lambda: NOW.astimezone(UTC))
    store = _FakeStore([])
    assert asyncio.run(svc.run_once(store)) == []  # type: ignore[arg-type]
