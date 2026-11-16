"""Spa bathing config (backend/core/spa_bathing_config.py) and its template round trip.

The block lives at ``water_heaters[id=spa].bathing``. Defaults are applied in code:
the config-save merge is shallow per heater, so a user's block replaces any template
block wholesale and nothing may rely on the template for a default.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from backend.core.spa_bathing_config import (
    BathingConfig,
    any_bathing_enabled,
    bathing_config_problem,
    load_learned_params,
    parse_bathing_config,
)

ROOT = Path(__file__).resolve().parents[2]


class TestParse:
    def test_missing_block_is_none(self):
        assert parse_bathing_config(None) is None

    def test_non_mapping_is_treated_as_missing(self):
        assert parse_bathing_config("yes") is None  # type: ignore[arg-type]

    def test_empty_block_gets_every_default(self):
        assert parse_bathing_config({}) == BathingConfig()

    def test_shadow_defaults_true_when_enabled_without_saying(self):
        cfg = parse_bathing_config({"enabled": True})
        assert cfg is not None and cfg.enabled is True and cfg.shadow is True

    def test_explicit_shadow_false_is_kept(self):
        cfg = parse_bathing_config({"enabled": True, "shadow": False})
        assert cfg is not None and cfg.shadow is False

    def test_strings_and_entities(self):
        cfg = parse_bathing_config(
            {
                "enabled": "true",
                "shadow": "off",
                "window_entities": "calendar.spa",
                "window_event_match": "  Bad ",
                "temp_sensor": " sensor.layzspa_temp_c ",
            }
        )
        assert cfg is not None
        assert cfg.enabled is True and cfg.shadow is False
        assert cfg.window_entities == ("calendar.spa",)
        assert cfg.window_event_match == "Bad"
        assert cfg.temp_sensor == "sensor.layzspa_temp_c"

    def test_entity_list_drops_blanks(self):
        cfg = parse_bathing_config({"window_entities": ["calendar.spa", "", None]})
        assert cfg is not None and cfg.window_entities == ("calendar.spa",)

    def test_eco_floor_is_clamped_to_the_frost_guard(self):
        cfg = parse_bathing_config({"eco_floor_c": 5})
        assert cfg is not None and cfg.eco_floor_c == 20.0

    @pytest.mark.parametrize("ready", [27.0, 28.0, 41.0, float("nan"), "warm"])
    def test_ready_outside_floor_and_max_falls_back(self, ready):
        cfg = parse_bathing_config({"ready_temp_c": ready})
        assert cfg is not None
        assert cfg.eco_floor_c < cfg.ready_temp_c <= cfg.temp_max_c
        assert cfg.ready_temp_c == 38.0

    def test_ready_equal_to_max_is_valid(self):
        cfg = parse_bathing_config({"ready_temp_c": 40})
        assert cfg is not None and cfg.ready_temp_c == 40.0

    def test_max_below_floor_is_repaired(self):
        cfg = parse_bathing_config({"eco_floor_c": 30, "temp_max_c": 25})
        assert cfg is not None
        assert cfg.temp_max_c > cfg.eco_floor_c
        assert cfg.eco_floor_c < cfg.ready_temp_c <= cfg.temp_max_c

    @pytest.mark.parametrize("cap", [0, -1, "x", float("inf")])
    def test_capacity_must_be_positive(self, cap):
        cfg = parse_bathing_config({"heat_capacity_kwh_per_c": cap})
        assert cfg is not None and cfg.heat_capacity_kwh_per_c == 0.9

    @pytest.mark.parametrize(("k", "want"), [(0.0001, 0.005), (0.21, 0.08), (0.03, 0.03)])
    def test_cooling_k_is_clamped(self, k, want):
        cfg = parse_bathing_config({"cooling_k_per_h": k})
        assert cfg is not None and cfg.cooling_k_per_h == pytest.approx(want)

    def test_negative_penalties_fall_back(self):
        cfg = parse_bathing_config({"ready_shortfall_sek_per_c": -3, "ready_margin_minutes": -5})
        assert cfg is not None
        assert cfg.ready_shortfall_sek_per_c == 5.0
        assert cfg.ready_margin_minutes == 30.0

    def test_problem_reports_missing_inputs(self):
        assert bathing_config_problem(BathingConfig(enabled=True)) == "no_window_entities"
        assert (
            bathing_config_problem(BathingConfig(enabled=True, window_entities=("calendar.spa",)))
            == "no_temp_sensor"
        )
        assert (
            bathing_config_problem(
                BathingConfig(window_entities=("calendar.spa",), temp_sensor="sensor.t")
            )
            is None
        )

    def test_any_bathing_enabled(self):
        assert any_bathing_enabled({}) is False
        assert any_bathing_enabled({"water_heaters": [{"id": "spa"}]}) is False
        assert (
            any_bathing_enabled({"water_heaters": [{"id": "spa", "bathing": {"enabled": False}}]})
            is False
        )
        assert (
            any_bathing_enabled(
                {"water_heaters": ["junk", {"id": "spa", "bathing": {"enabled": True}}]}
            )
            is True
        )


class TestLearnedParams:
    def _write(self, tmp_path, **data):
        p = tmp_path / "spa_thermal_params.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_missing_file_is_none(self, tmp_path):
        assert load_learned_params(str(tmp_path / "nope.json")) is None

    def test_fresh_valid_file(self, tmp_path):
        path = self._write(
            tmp_path,
            heat_capacity_kwh_per_c=0.85,
            cooling_k_per_h=0.025,
            updated_at=datetime.now(UTC).isoformat(),
        )
        assert load_learned_params(path) == {
            "heat_capacity_kwh_per_c": 0.85,
            "cooling_k_per_h": 0.025,
        }

    def test_stale_file_is_ignored(self, tmp_path):
        path = self._write(
            tmp_path,
            heat_capacity_kwh_per_c=0.85,
            cooling_k_per_h=0.025,
            updated_at=(datetime.now(UTC) - timedelta(days=60)).isoformat(),
        )
        assert load_learned_params(path) is None

    def test_naive_timestamp_is_ignored(self, tmp_path):
        path = self._write(
            tmp_path,
            heat_capacity_kwh_per_c=0.85,
            cooling_k_per_h=0.025,
            updated_at=datetime.now().replace(tzinfo=None).isoformat(),
        )
        assert load_learned_params(path) is None

    @pytest.mark.parametrize(("cap", "k"), [(0.0, 0.02), (0.9, 0.5), ("x", 0.02)])
    def test_implausible_values_are_ignored(self, tmp_path, cap, k):
        path = self._write(
            tmp_path,
            heat_capacity_kwh_per_c=cap,
            cooling_k_per_h=k,
            updated_at=datetime.now(UTC).isoformat(),
        )
        assert load_learned_params(path) is None

    def test_garbage_file_is_none(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text("{not json")
        assert load_learned_params(str(p)) is None


class TestTemplate:
    def test_template_has_no_live_bathing_block(self):
        """The example is commented out: the shipped main_tank must not grow one."""
        template = yaml.safe_load((ROOT / "config.default.yaml").read_text())
        for wh in template["water_heaters"]:
            assert "bathing" not in wh
        assert any_bathing_enabled(template) is False

    def test_commented_example_lists_every_contract_key(self):
        text = (ROOT / "config.default.yaml").read_text()
        start = text.index("Example: Spa bathing times")
        block = text[start : text.index("# Unknown-load observability", start)]
        for name in BathingConfig.__dataclass_fields__:
            assert f"#     {name}:" in block, name

    def test_commented_example_parses_to_the_defaults(self):
        """Uncommenting the example yields exactly the code defaults."""
        text = (ROOT / "config.default.yaml").read_text()
        start = text.index("  #   bathing:")
        end = text.index("# Unknown-load observability", start)
        lines = [ln.replace("  #", "", 1) for ln in text[start:end].splitlines() if ln.strip()]
        raw = yaml.safe_load("\n".join(lines))["bathing"]
        assert parse_bathing_config(raw) == BathingConfig()

    def test_user_spa_entry_survives_the_config_save_merge(self):
        """template_aware_merge merges water_heaters by id with a SHALLOW update: the
        user's whole bathing block is kept as sent (no deep merge), and anything the
        user left out is defaulted by parse_bathing_config, not by the template."""
        from backend.config_migration import template_aware_merge

        template = yaml.safe_load((ROOT / "config.default.yaml").read_text())
        user_bathing = {
            "enabled": True,
            "window_entities": ["calendar.spa"],
            "temp_sensor": "sensor.layzspa_temp_c",
            "ready_temp_c": 37.5,
        }
        user = {
            "water_heaters": [
                {"id": "main_tank", "power_kw": 3.0},
                {"id": "spa", "name": "Spa", "power_kw": 1.8, "bathing": dict(user_bathing)},
            ]
        }
        template_aware_merge(template, user)
        spa = next(wh for wh in template["water_heaters"] if wh.get("id") == "spa")
        assert spa["bathing"] == user_bathing
        cfg = parse_bathing_config(spa["bathing"])
        assert cfg is not None
        assert cfg.enabled is True and cfg.shadow is True
        assert cfg.ready_temp_c == 37.5
        assert cfg.eco_floor_c == 28.0 and cfg.floor_write_temp_c == 31.0
        assert any_bathing_enabled(template) is True


# ---------------------------------------------------------------------------
# get_initial_state: bathing_inputs
# ---------------------------------------------------------------------------

SPA_TZ = "Europe/Stockholm"


def _initial_config(tmp_path, heaters):
    cfg = {
        "timezone": SPA_TZ,
        "system": {"battery": {"capacity_kwh": 10.0}, "has_ev_charger": False},
        "input_sensors": {"battery_soc": "sensor.batt_soc"},
        "learning": {"sqlite_path": str(tmp_path / "none.db")},
        "water_heaters": heaters,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


async def _initial_states(tmp_path, heaters, states, windows, heated=None):
    from unittest.mock import AsyncMock, patch

    from backend.core.ha_client import get_initial_state

    async def entity_state(entity_id):
        if entity_id == "sensor.batt_soc":
            return {"state": "75.0"}
        value = states.get(entity_id)
        if isinstance(value, Exception):
            raise value
        return None if value is None else {"state": value}

    calls: list[dict] = []

    async def plan_windows(raw, now, tz, *, get_state):
        calls.append(dict(raw))
        assert now.tzinfo is not None
        if isinstance(windows, Exception):
            raise windows
        return windows

    with (
        patch("backend.core.ha_client.get_ha_entity_state", side_effect=entity_state),
        patch("backend.core.ha_client.plan_windows_for_load", side_effect=plan_windows),
        patch("backend.core.secrets.load_home_assistant_config", return_value={}),
        patch(
            "backend.core.ha_client.get_water_heated_today_by_tank",
            AsyncMock(return_value=heated or {}),
        ),
        patch("backend.core.ha_client.get_water_absorption_stats", AsyncMock(return_value={})),
    ):
        res = await get_initial_state(_initial_config(tmp_path, heaters))
    return {s["id"]: s for s in res["water_heater_states"]}, calls


SPA = {
    "id": "spa",
    "power_kw": 1.8,
    "bathing": {
        "enabled": True,
        "window_entities": ["calendar.spa"],
        "window_event_match": "Bad",
        "temp_sensor": "sensor.layzspa_temp_c",
        "ambient_sensor": "sensor.ute_temp",
        "ready_margin_minutes": 45,
    },
}


class TestInitialState:
    @pytest.mark.asyncio
    async def test_inputs_are_attached(self, tmp_path):
        import pytz

        from backend.core.load_windows import windows_to_iso

        tz = pytz.timezone(SPA_TZ)
        win = (tz.localize(datetime(2026, 11, 16, 19)), tz.localize(datetime(2026, 11, 16, 21)))
        states, calls = await _initial_states(
            tmp_path,
            [SPA],
            {"sensor.layzspa_temp_c": "31.5", "sensor.ute_temp": "3.0"},
            [win],
            heated={"spa": 0.4},
        )
        bi = states["spa"]["bathing_inputs"]
        assert states["spa"]["heated_today_kwh"] == 0.4
        assert bi["windows"] == windows_to_iso([win])
        assert bi["temp_c"] == 31.5 and bi["ambient_c"] == 3.0
        assert datetime.fromisoformat(bi["read_at"]).tzinfo is not None
        # No preheat: the planner applies ready_margin itself.
        assert calls == [
            {
                "id": "spa",
                "window_entities": ["calendar.spa"],
                "window_event_match": "Bad",
                "preheat_minutes": 0,
            }
        ]

    @pytest.mark.asyncio
    async def test_failures_read_as_unknown_and_an_entry_is_appended(self, tmp_path):
        states, _ = await _initial_states(
            tmp_path,
            [SPA],
            {"sensor.layzspa_temp_c": "unavailable", "sensor.ute_temp": RuntimeError("x")},
            RuntimeError("calendar down"),
        )
        assert states["spa"]["heated_today_kwh"] == 0.0
        bi = states["spa"]["bathing_inputs"]
        assert bi["windows"] is None and bi["temp_c"] is None and bi["ambient_c"] is None

    @pytest.mark.asyncio
    async def test_unknown_calendar_is_none(self, tmp_path):
        states, _ = await _initial_states(tmp_path, [SPA], {"sensor.layzspa_temp_c": "30"}, None)
        assert states["spa"]["bathing_inputs"]["windows"] is None
        assert states["spa"]["bathing_inputs"]["temp_c"] == 30.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bathing", [None, {"enabled": False, "window_entities": ["calendar.spa"]}]
    )
    async def test_no_or_disabled_block_reads_nothing(self, tmp_path, bathing):
        heater: dict = {"id": "spa", "power_kw": 1.8}
        if bathing is not None:
            heater["bathing"] = bathing
        states, calls = await _initial_states(tmp_path, [heater], {}, [])
        assert calls == []
        assert "spa" not in states  # nothing heated, nothing paused: no entry, as before
