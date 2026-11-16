"""Spa bathing times — executor rules (executor/spa_bathing.py) and engine glue.

Pure tests for ``decide_bathing_rules`` and ``SpaBathingRuntime``, then engine tests
on a reference-site-shaped spa (20-40 C bridge helper) that check the precedence:
control pause > fuse shed > override > bathing rules > surplus-boost > idle-hold.
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from sqlalchemy import create_engine

from backend.core.spa_bathing_config import BathingConfig
from backend.core.spa_thermal import heat_rate_c_per_h, hours_to_reach_c
from backend.learning.models import Base
from executor.actions import HAClient
from executor.config import (
    ControllerConfig,
    ExecutorConfig,
    InverterConfig,
    NotificationConfig,
    WaterHeaterDeviceConfig,
    WaterHeaterGlobalConfig,
    load_executor_config,
)
from executor.engine import ExecutorEngine
from executor.spa_bathing import (
    RULE_CATCH_UP,
    RULE_ECO_FLOOR,
    RULE_IN_WINDOW,
    RULE_NONE,
    SpaBathingRuntime,
    decide_bathing_rules,
    window_active,
)

TZ = pytz.timezone("Europe/Stockholm")
NOW = TZ.localize(datetime(2026, 11, 16, 14, 0))
SPA_HELPER = "input_number.spa_darkstar_target_temp"
TEMP_SENSOR = "sensor.layzspa_temp_c"
CAL = "calendar.spa"

CFG = BathingConfig(
    enabled=True,
    shadow=False,
    window_entities=(CAL,),
    temp_sensor=TEMP_SENSOR,
    ready_temp_c=38.0,
    eco_floor_c=28.0,
    temp_max_c=40.0,
    heat_capacity_kwh_per_c=0.9,
    cooling_k_per_h=0.021,
    catch_up_margin_c=0.5,
    floor_write_temp_c=31.0,
    ambient_default_c=10.0,
    learn=False,
)


def _decide(**kw: Any):
    args: dict[str, Any] = {
        "cfg": CFG,
        "now": NOW,
        "windows": [],
        "measured_temp_c": 35.0,
        "planned_temp": 20,
        "off_temp": 20,
        "normal_temp": 38,
        "device_max_temp": 40,
        "power_kw": 1.8,
        "ambient_c": 10.0,
        "floor_latched": False,
        "heating_now": False,
    }
    args.update(kw)
    return decide_bathing_rules(**args)


# --------------------------------------------------------------------------------------
# Pure rules
# --------------------------------------------------------------------------------------


class TestInWindow:
    def test_in_window_replaces_off_with_ready(self):
        win = [(NOW - timedelta(minutes=30), NOW + timedelta(hours=1))]
        d, _ = _decide(windows=win)
        assert d.rule == RULE_IN_WINDOW
        assert d.action == "no_off"
        assert d.temp_c == 38

    def test_in_window_needs_no_temperature(self):
        win = [(NOW - timedelta(minutes=30), NOW + timedelta(hours=1))]
        d, _ = _decide(windows=win, measured_temp_c=None)
        assert d.rule == RULE_IN_WINDOW and d.temp_c == 38

    def test_in_window_plan_heat_is_left_alone(self):
        win = [(NOW - timedelta(minutes=30), NOW + timedelta(hours=1))]
        d, _ = _decide(windows=win, planned_temp=38)
        assert d.action == "none"

    def test_in_window_capped_by_device_max(self):
        win = [(NOW, NOW + timedelta(hours=1))]
        d, _ = _decide(windows=win, cfg=replace(CFG, ready_temp_c=40.0), device_max_temp=39)
        assert d.temp_c == 39

    def test_window_end_is_exclusive(self):
        win = [(NOW - timedelta(hours=1), NOW)]
        d, _ = _decide(windows=win)
        assert d.rule != RULE_IN_WINDOW


class TestCatchUp:
    @staticmethod
    def _needed_h(temp: float) -> float:
        reach = hours_to_reach_c(temp, 38.0, 10.0, 1.8, 0.9, 0.021)
        assert reach is not None
        return reach + 0.5 / heat_rate_c_per_h(1.8, 0.9)

    def test_fires_exactly_when_inequality_flips(self):
        temp = 33.0
        need = self._needed_h(temp)
        just_late = NOW + timedelta(hours=need - 0.01)
        just_ok = NOW + timedelta(hours=need + 0.01)
        d_late, _ = _decide(
            windows=[(just_late, just_late + timedelta(hours=2))], measured_temp_c=temp
        )
        d_ok, _ = _decide(windows=[(just_ok, just_ok + timedelta(hours=2))], measured_temp_c=temp)
        assert d_late.rule == RULE_CATCH_UP
        assert d_late.action == "heat" and d_late.temp_c == 38
        assert "efter schemat" in d_late.reason
        assert d_ok.action == "none"

    def test_time_left_counts_to_window_start_not_ready_by(self):
        # ready_margin_minutes must not make the rule fire earlier than the start.
        temp = 33.0
        need = self._needed_h(temp)
        start = NOW + timedelta(hours=need + 0.2)
        d, _ = _decide(
            windows=[(start, start + timedelta(hours=1))],
            measured_temp_c=temp,
            cfg=replace(CFG, ready_margin_minutes=60.0),
        )
        assert d.action == "none"

    def test_unreachable_target_heats_now(self):
        # Asymptote amb + P/(kC) below ready: never reachable -> heat now.
        cfg = replace(CFG, cooling_k_per_h=0.08, heat_capacity_kwh_per_c=0.9)
        start = NOW + timedelta(days=2)
        d, _ = _decide(
            cfg=cfg, windows=[(start, start + timedelta(hours=1))], power_kw=0.3, ambient_c=-20.0
        )
        assert d.rule == RULE_CATCH_UP and d.action == "heat"

    def test_already_at_ready_does_not_fire(self):
        start = NOW + timedelta(minutes=5)
        d, _ = _decide(windows=[(start, start + timedelta(hours=1))], measured_temp_c=38.0)
        assert d.action == "none"

    def test_plan_already_heats(self):
        start = NOW + timedelta(minutes=30)
        d, _ = _decide(windows=[(start, start + timedelta(hours=1))], planned_temp=38)
        assert d.action == "none"

    def test_missing_ambient_uses_default(self):
        start = NOW + timedelta(minutes=30)
        d, _ = _decide(windows=[(start, start + timedelta(hours=1))], ambient_c=None)
        assert d.rule == RULE_CATCH_UP


class TestEcoFloor:
    def test_latch_and_hysteresis(self):
        # Above the trigger: nothing.
        d, latched = _decide(measured_temp_c=27.5)
        assert d.action == "none" and latched is False
        # At floor - 1: latch, write 31 (above the bridge's 30 C fan_only threshold).
        d, latched = _decide(measured_temp_c=27.0)
        assert d.rule == RULE_ECO_FLOOR and d.temp_c == 31 and latched is True
        # Recovering but under floor + 1: keep heating.
        d, latched = _decide(measured_temp_c=28.9, floor_latched=True, heating_now=True)
        assert d.rule == RULE_ECO_FLOOR and d.temp_c == 31 and latched is True
        # At floor + 1 (29): release.
        d, latched = _decide(measured_temp_c=29.0, floor_latched=True, heating_now=True)
        assert d.action == "none" and latched is False

    def test_not_latched_while_already_heating(self):
        d, latched = _decide(measured_temp_c=26.0, heating_now=True)
        assert d.action == "none" and latched is False

    def test_unknown_heating_still_latches(self):
        d, latched = _decide(measured_temp_c=26.0, heating_now=None)
        assert d.rule == RULE_ECO_FLOOR and latched is True

    def test_plan_heat_leaves_latch_but_no_write(self):
        d, latched = _decide(measured_temp_c=26.0, planned_temp=38)
        assert d.action == "none" and latched is True

    def test_floor_works_without_windows(self):
        d, _ = _decide(measured_temp_c=26.0, windows=None)
        assert d.rule == RULE_ECO_FLOOR


class TestFailSafe:
    def test_unknown_windows_disable_window_and_catch_up(self):
        d, _ = _decide(windows=None, measured_temp_c=20.0 + 15.0)
        assert d.action == "none"

    def test_unknown_temperature_disables_catch_up_and_floor(self):
        start = NOW + timedelta(minutes=10)
        d, latched = _decide(windows=[(start, start + timedelta(hours=1))], measured_temp_c=None)
        assert d.action == "none" and latched is False
        d, latched = _decide(measured_temp_c=None, floor_latched=True)
        assert d.action == "none" and latched is True  # unchanged, but no write

    @pytest.mark.parametrize("bad", [float("nan"), -5.0, 75.0])
    def test_implausible_temperature_is_unknown(self, bad):
        d, _ = _decide(measured_temp_c=bad)
        assert d.action == "none"

    def test_naive_now_disables_window_rules(self):
        naive_now = datetime(2026, 11, 16, 14, 0)
        win = [(NOW - timedelta(minutes=5), NOW + timedelta(hours=1))]
        d, _ = _decide(now=naive_now, windows=win)
        assert d.rule != RULE_IN_WINDOW

    def test_naive_window_is_refused(self):
        naive = datetime(2026, 11, 16, 13, 0)
        d, _ = _decide(windows=[(naive, naive + timedelta(hours=3))])
        assert d.rule == RULE_NONE
        assert window_active([(naive, naive + timedelta(hours=3))], NOW) is None

    def test_empty_window_list_is_no_bath(self):
        d, _ = _decide(windows=[], measured_temp_c=35.0)
        assert d.action == "none"
        assert window_active([], NOW) is False

    def test_utc_and_local_windows_compare_correctly(self):
        # A window given in UTC must match the local "now" — no 1 h shift.
        start_utc = NOW.astimezone(UTC) - timedelta(minutes=1)
        d, _ = _decide(windows=[(start_utc, start_utc + timedelta(hours=1))])
        assert d.rule == RULE_IN_WINDOW

    def test_dst_window_times(self):
        # Spring-forward night: 02:00 local does not exist; a 01:30-03:30 window is
        # 1 h of real time. 'now' at 03:15 local (CEST) is inside it.
        start = TZ.localize(datetime(2027, 3, 28, 1, 30))
        end = TZ.localize(datetime(2027, 3, 28, 3, 30))
        now = TZ.localize(datetime(2027, 3, 28, 3, 15))
        assert (end - start) == timedelta(hours=1)
        d, _ = _decide(now=now, windows=[(start, end)])
        assert d.rule == RULE_IN_WINDOW


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------


class TestRuntime:
    @pytest.mark.asyncio
    async def test_windows_cached_and_filtered(self):
        past = (NOW - timedelta(hours=5), NOW - timedelta(hours=4))
        future = (NOW + timedelta(hours=3), NOW + timedelta(hours=5))
        fetch = AsyncMock(return_value=[past, future])
        rt = SpaBathingRuntime(fetch=fetch, learned=lambda: None)
        ts = NOW.timestamp()
        got = await rt.windows(CFG, ts, "Europe/Stockholm")
        assert got == [future]
        await rt.windows(CFG, ts + 60, "Europe/Stockholm")
        assert fetch.await_count == 1  # cached inside EVENT_REFRESH_S

    @pytest.mark.asyncio
    async def test_unreadable_calendar_is_unknown_then_stale_cache_expires(self):
        future = (NOW + timedelta(hours=3), NOW + timedelta(hours=5))
        fetch = AsyncMock(side_effect=[None])
        rt = SpaBathingRuntime(fetch=fetch, learned=lambda: None)
        assert await rt.windows(CFG, NOW.timestamp(), "Europe/Stockholm") is None

        fetch = AsyncMock(side_effect=[[future], None, None, None])
        rt = SpaBathingRuntime(fetch=fetch, learned=lambda: None)
        ts = NOW.timestamp()
        assert await rt.windows(CFG, ts, "Europe/Stockholm") == [future]
        # Refresh fails, the half-hour-old list is still followed...
        assert await rt.windows(CFG, ts + 400, "Europe/Stockholm") == [future]
        # ...but not after EVENT_MAX_AGE_S.
        assert await rt.windows(CFG, ts + 1900, "Europe/Stockholm") is None

    @pytest.mark.asyncio
    async def test_naive_calendar_answer_is_unknown(self):
        naive = datetime(2026, 11, 16, 18, 0)
        rt = SpaBathingRuntime(
            fetch=AsyncMock(return_value=[(naive, naive + timedelta(hours=1))]),
            learned=lambda: None,
        )
        assert await rt.windows(CFG, NOW.timestamp(), "Europe/Stockholm") is None

    @pytest.mark.asyncio
    async def test_no_window_entities_is_unknown(self):
        rt = SpaBathingRuntime(fetch=AsyncMock(), learned=lambda: None)
        cfg = replace(CFG, window_entities=())
        assert await rt.windows(cfg, NOW.timestamp(), "Europe/Stockholm") is None

    @pytest.mark.asyncio
    async def test_non_calendar_gate(self):
        rt = SpaBathingRuntime(fetch=AsyncMock(), learned=lambda: None)
        cfg = replace(CFG, window_entities=("input_boolean.bada",))
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value="on")
        got = await rt.windows(cfg, NOW.timestamp(), "Europe/Stockholm", ha)
        assert got is not None and window_active(got, NOW) is True
        ha.get_state_value = AsyncMock(return_value="off")
        assert await rt.windows(cfg, NOW.timestamp(), "Europe/Stockholm", ha) == []
        ha.get_state_value = AsyncMock(return_value="unavailable")
        assert await rt.windows(cfg, NOW.timestamp(), "Europe/Stockholm", ha) is None
        assert await rt.windows(cfg, NOW.timestamp(), "Europe/Stockholm") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("36.5", 36.5),
            ("unavailable", None),
            ("unknown", None),
            ("abc", None),
            ("-1", None),
            ("51", None),
            (None, None),
        ],
    )
    async def test_read_temp(self, raw, want):
        rt = SpaBathingRuntime(fetch=AsyncMock(), learned=lambda: None)
        ha = MagicMock()
        ha.get_state_value = AsyncMock(return_value=raw)
        assert await rt.read_temp_c(ha, TEMP_SENSOR) == want

    def test_learned_params_swap_in_only_when_learn(self):
        params = {"heat_capacity_kwh_per_c": 1.1, "cooling_k_per_h": 0.03}
        rt = SpaBathingRuntime(fetch=AsyncMock(), learned=lambda: params)
        assert rt.effective_config(CFG) is CFG  # learn=False
        eff = rt.effective_config(replace(CFG, learn=True))
        assert eff.heat_capacity_kwh_per_c == 1.1 and eff.cooling_k_per_h == 0.03
        rt_bad = SpaBathingRuntime(fetch=AsyncMock(), learned=lambda: {"x": 1.0})
        assert rt_bad.effective_config(replace(CFG, learn=True)).heat_capacity_kwh_per_c == 0.9


# --------------------------------------------------------------------------------------
# Config parser
# --------------------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, spa_extra: dict[str, Any]) -> str:
    spa: dict[str, Any] = {
        "id": "spa",
        "target_entity": SPA_HELPER,
        "power_kw": 1.8,
        "temp_off": 20,
        "temp_normal": 38,
    }
    spa.update(spa_extra)
    data = {
        "timezone": "Europe/Stockholm",
        "executor": {"enabled": True},
        "water_heaters": [{"id": "main_tank", "target_entity": "switch.vvb"}, spa],
    }
    p = tmp_path / "config.yaml"
    p.write_text(json.dumps(data), encoding="utf-8")  # JSON is valid YAML
    return str(p)


class TestConfig:
    def test_parser_populates_bathing(self, tmp_path):
        cfg = load_executor_config(
            _write_cfg(
                tmp_path,
                {
                    "bathing": {
                        "enabled": True,
                        "window_entities": [CAL],
                        "temp_sensor": TEMP_SENSOR,
                    }
                },
            )
        )
        spa = next(d for d in cfg.water_heater_devices if d.id == "spa")
        main = next(d for d in cfg.water_heater_devices if d.id == "main_tank")
        assert spa.bathing is not None
        assert spa.bathing.enabled is True
        assert spa.bathing.shadow is True  # safe default
        assert spa.bathing.window_entities == (CAL,)
        assert main.bathing is None

    def test_missing_block_is_none(self, tmp_path):
        cfg = load_executor_config(_write_cfg(tmp_path, {}))
        spa = next(d for d in cfg.water_heater_devices if d.id == "spa")
        assert spa.bathing is None

    def test_malformed_block_never_breaks_config(self, tmp_path):
        cfg = load_executor_config(_write_cfg(tmp_path, {"bathing": "yes please"}))
        spa = next(d for d in cfg.water_heater_devices if d.id == "spa")
        assert spa.bathing is None


# --------------------------------------------------------------------------------------
# Engine glue
# --------------------------------------------------------------------------------------


@pytest.fixture
def temp_schedule():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        p = f.name
    yield p
    with contextlib.suppress(OSError):
        Path(p).unlink()


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        p = f.name
    Base.metadata.create_all(create_engine(f"sqlite:///{p}"))
    yield p
    with contextlib.suppress(OSError):
        Path(p).unlink()


class Rig:
    """An engine with a reference-site-shaped spa and every live read under test control."""

    def __init__(self, eng: ExecutorEngine, ha: MagicMock, schedule: str) -> None:
        self.eng = eng
        self.ha = ha
        self.schedule = schedule
        self.states: dict[str, str] = {
            SPA_HELPER: "25",  # neither off (20) nor any rule value: every write shows
            TEMP_SENSOR: "35.0",
            "input_boolean.spa_pause": "off",
        }
        self.windows: list[tuple[datetime, datetime]] | None = []
        self.heater_w: float | None = 0.0
        self.override = "auto"
        self.ctx: dict[str, Any] = {
            "grid_w": 0.0,
            "battery_w": 0.0,
            "import_price": 1.0,
            "export_price": 0.5,
            "vacation": False,
            "phase_currents": None,
            "fuse_budget_a": None,
        }

    @property
    def spa(self) -> WaterHeaterDeviceConfig:
        return next(d for d in self.eng.config.water_heater_devices if d.id == "spa")

    def set_spa(self, **kw: Any) -> None:
        devs = self.eng.config.water_heater_devices
        idx = next(i for i, d in enumerate(devs) if d.id == "spa")
        devs[idx] = replace(devs[idx], **kw)

    def set_bathing(self, cfg: BathingConfig | None) -> None:
        self.set_spa(bathing=cfg)

    def write_slot(self, spa_kw: float = 0.0, bathing_mode: str | None = None) -> None:
        now = datetime.now(TZ)
        start = now - timedelta(minutes=5)
        end = start + timedelta(minutes=15)
        slot: dict[str, Any] = {
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "end_time_kepler": end.isoformat(),
            "battery_charge_kw": 0.0,
            "battery_discharge_kw": 0.0,
            "export_kwh": 0.0,
            "water_heating_kw": spa_kw,
            "water_heaters": {"main_tank": 0.0, "spa": spa_kw},
            "soc_target_percent": 50,
            "projected_soc_percent": 45,
        }
        if bathing_mode is not None:
            slot["water_heater_bathing"] = {"spa": bathing_mode}
        with Path(self.schedule).open("w", encoding="utf-8") as f:
            json.dump({"schedule": [slot], "meta": {"generated_at": now.isoformat()}}, f)

    def spa_writes(self) -> list[float]:
        return [
            c.args[1] for c in self.ha.set_input_number.call_args_list if c.args[0] == SPA_HELPER
        ]

    async def tick(self) -> list[float]:
        self.ha.set_input_number.reset_mock()
        await self.eng.run_once()
        return self.spa_writes()


@pytest.fixture
def rig(temp_schedule, temp_db):
    from executor.actions import ActionDispatcher

    with patch("executor.engine.load_executor_config") as mock_config:
        config = ExecutorConfig(
            enabled=True,
            schedule_path=temp_schedule,
            timezone="Europe/Stockholm",
            automation_toggle_entity="input_boolean.automation",
            inverter=InverterConfig(),
            controller=ControllerConfig(),
            notifications=NotificationConfig(),
            water_heater=WaterHeaterGlobalConfig(
                temp_normal=60, temp_off=40, temp_boost=70, temp_max=85
            ),
            water_heater_devices=[
                WaterHeaterDeviceConfig(
                    id="main_tank", name="VVB", target_entity="switch.vvb", power_kw=3.0
                ),
                WaterHeaterDeviceConfig(
                    id="spa",
                    name="Spa",
                    target_entity=SPA_HELPER,
                    power_kw=1.8,
                    temp_off=20,
                    temp_normal=38,
                    temp_boost=40,
                    temp_max=40,
                    idle_hold=True,
                    bathing=CFG,
                ),
            ],
        )
        mock_config.return_value = config
        with (
            patch("executor.engine.load_yaml") as mock_yaml,
            patch.object(ExecutorEngine, "_get_db_path", return_value=temp_db),
        ):
            mock_yaml.return_value = {"input_sensors": {}}
            eng = ExecutorEngine("config.yaml")
            ha = MagicMock(spec=HAClient)
            r = Rig(eng, ha, temp_schedule)

            async def get_state_value(entity_id: str) -> str | None:
                if entity_id in r.states:
                    return r.states[entity_id]
                if "input_boolean" in entity_id or "automation" in entity_id:
                    return "on"
                if "soc" in entity_id:
                    return "50"
                if entity_id == "switch.vvb":
                    return "off"
                return "0.0"

            ha.get_state_value = AsyncMock(side_effect=get_state_value)
            ha.get_state = AsyncMock(return_value=None)
            ha.set_select_option.return_value = True
            ha.set_switch.return_value = True
            ha.set_number.return_value = True
            ha.set_input_number = AsyncMock(return_value=True)
            ha.call_service = AsyncMock(return_value=True)
            eng.ha_client = ha
            eng.dispatcher = ActionDispatcher(ha, config, shadow_mode=False)
            eng._has_water_heater = True

            async def ctx(*_a: Any, **_k: Any) -> dict[str, Any]:
                d: defaultdict[str, Any] = defaultdict(lambda: None)
                d.update(r.ctx)
                return d

            async def heater_w(_device: Any) -> float | None:
                return r.heater_w

            async def override(_device: Any) -> str:
                return r.override

            async def fetch(*_a: Any, **_k: Any):
                return None if r.windows is None else list(r.windows)

            eng._water_energy_ctx = ctx  # type: ignore[method-assign]
            eng._heater_power_w = heater_w  # type: ignore[method-assign]
            eng._heater_override = override  # type: ignore[method-assign]
            eng.__dict__["_spa_bathing"] = SpaBathingRuntime(fetch=fetch, learned=lambda: None)
            yield r


def _live_window(minutes_ago: int = 10, hours: float = 1.0) -> tuple[datetime, datetime]:
    start = datetime.now(TZ) - timedelta(minutes=minutes_ago)
    return (start, start + timedelta(hours=hours))


class TestEngine:
    @pytest.mark.asyncio
    async def test_in_window_never_writes_off_even_at_high_price(self, rig):
        rig.windows = [_live_window()]
        rig.ctx["import_price"] = 9.0
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == [38.0]

    @pytest.mark.asyncio
    async def test_catch_up_heats_outside_plan(self, rig, caplog):
        start = datetime.now(TZ) + timedelta(minutes=30)
        rig.windows = [(start, start + timedelta(hours=2))]
        rig.states[TEMP_SENSOR] = "30.0"
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        with caplog.at_level(logging.WARNING, logger="executor.spa_bathing"):
            assert await rig.tick() == [38.0]
        assert any("efter schemat" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_eco_floor_writes_floor_value(self, rig):
        rig.states[TEMP_SENSOR] = "26.5"
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == [31.0]

    @pytest.mark.asyncio
    async def test_fuse_shed_beats_catch_up(self, rig):
        start = datetime.now(TZ) + timedelta(minutes=30)
        rig.windows = [(start, start + timedelta(hours=2))]
        rig.states[TEMP_SENSOR] = "30.0"
        rig.set_spa(fuse_shed=True)
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        with patch("executor.engine.should_shed_for_fuse", return_value=(True, "L2 over")):
            assert await rig.tick() == [20.0]

    @pytest.mark.asyncio
    async def test_fuse_shed_beats_in_window(self, rig):
        rig.windows = [_live_window()]
        rig.set_spa(fuse_shed=True)
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        with patch("executor.engine.should_shed_for_fuse", return_value=(True, "L2 over")):
            assert await rig.tick() == [20.0]

    @pytest.mark.asyncio
    async def test_force_off_beats_catch_up(self, rig):
        start = datetime.now(TZ) + timedelta(minutes=30)
        rig.windows = [(start, start + timedelta(hours=2))]
        rig.states[TEMP_SENSOR] = "30.0"
        rig.override = "force_off"
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == [20.0]

    @pytest.mark.asyncio
    async def test_force_off_beats_in_window(self, rig):
        rig.windows = [_live_window()]
        rig.override = "force_off"
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == [20.0]

    @pytest.mark.asyncio
    async def test_control_pause_beats_everything(self, rig):
        rig.windows = [_live_window()]
        rig.states[TEMP_SENSOR] = "25.0"
        rig.states["input_boolean.spa_pause"] = "on"
        rig.set_spa(control_pause_entities=["input_boolean.spa_pause"])
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == []

    @pytest.mark.asyncio
    async def test_shadow_logs_and_writes_nothing_extra(self, rig, caplog):
        rig.windows = [_live_window()]
        rig.set_bathing(replace(CFG, shadow=True))
        rig.write_slot(spa_kw=0.0, bathing_mode="shadow")
        with caplog.at_level(logging.INFO, logger="executor.spa_bathing"):
            writes = await rig.tick()
        # Legacy path: idle spa, cheap price -> idle-hold skips the off-write.
        assert writes == []
        assert any("[SHADOW spa-bathing]" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_global_shadow_mode_still_applies(self, rig):
        rig.windows = [_live_window()]
        rig.eng.dispatcher.shadow_mode = True
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == []  # decided 38, but the dispatcher never writes

    @pytest.mark.asyncio
    async def test_idle_hold_bypassed_for_solver_plan_outside_window(self, rig):
        start = datetime.now(TZ) + timedelta(hours=20)
        rig.windows = [(start, start + timedelta(hours=1))]
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == [20.0]  # the off-write lands

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["fallback", "shadow", None])
    async def test_idle_hold_kept_outside_a_solver_plan(self, rig, mode):
        start = datetime.now(TZ) + timedelta(hours=20)
        rig.windows = [(start, start + timedelta(hours=1))]
        rig.write_slot(spa_kw=0.0, bathing_mode=mode)
        assert await rig.tick() == []  # held, as today

    @pytest.mark.asyncio
    async def test_idle_hold_kept_when_windows_unknown(self, rig):
        rig.windows = None
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == []

    @pytest.mark.asyncio
    async def test_surplus_boost_blocked_near_temp_max(self, rig):
        rig.set_spa(surplus_boost=True)
        rig.ctx.update(grid_w=-3000.0, export_price=0.1, import_price=0.5)
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        rig.states[TEMP_SENSOR] = "35.0"
        assert await rig.tick() == [40.0]  # boost allowed well below the cap
        rig.states[TEMP_SENSOR] = "39.5"
        rig.states[SPA_HELPER] = "25"
        assert await rig.tick() == [20.0]  # at temp_max - 0.5: no boost, off lands

    @pytest.mark.asyncio
    async def test_failing_bathing_step_falls_back_to_legacy(self, rig):
        rig.windows = [_live_window()]
        rt = rig.eng.__dict__["_spa_bathing"]
        rt.windows = AsyncMock(side_effect=RuntimeError("boom"))
        rig.write_slot(spa_kw=0.0, bathing_mode="solver")
        assert await rig.tick() == []  # legacy idle-hold

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bathing", [None, replace(CFG, enabled=False)])
    @pytest.mark.parametrize(
        ("spa_kw", "heater_w", "price"),
        [(0.0, 0.0, 1.0), (0.0, 0.0, 9.0), (1.8, 0.0, 1.0), (0.0, 1800.0, 1.0)],
    )
    async def test_disabled_is_identical_to_legacy(self, rig, bathing, spa_kw, heater_w, price):
        """bathing missing/disabled => the same writes as a pre-bathing device."""
        rig.windows = [_live_window()]
        rig.states[TEMP_SENSOR] = "25.0"  # would trip every rule if they were live
        rig.heater_w = heater_w
        rig.ctx["import_price"] = price
        rig.write_slot(spa_kw=spa_kw, bathing_mode="solver")
        rig.set_bathing(bathing)
        with_block = await rig.tick()
        reads_with_block = [c.args[0] for c in rig.ha.get_state_value.call_args_list]

        # Reference: the same device with no bathing field at all.
        rig.states[SPA_HELPER] = "25"
        rig.set_bathing(None)
        rig.ha.get_state_value.reset_mock()
        reference = await rig.tick()
        assert with_block == reference
        assert TEMP_SENSOR not in reads_with_block
