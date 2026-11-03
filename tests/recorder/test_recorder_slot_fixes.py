"""Recorder slot-energy fixes (2026-11-01).

A1  balance-rescued rows subtract cyclic loads from base load (as the normal path does).
A5  per-heater idle cutoff: standby draw is not heating, and stays in base load.
    An unmeasured heater is flagged in quality_flags instead of silently booked 0.
A+  the power-history integrator rejects non-finite samples.
B1  opt-in exact-slot grid/battery shadow: sign split, never touches energy columns.
"""

import json
import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from backend.core.ha_client import (
    integrate_power_history_kwh,
    integrate_power_history_split_kwh,
)
from backend.recorder import (
    RecorderStateStore,
    _exact_slot_power_shadow,  # pyright: ignore[reportPrivateUsage]
    record_observation_from_current_state,
)

TZ = pytz.timezone("Europe/Stockholm")
START = datetime(2026, 11, 1, 12, 0, tzinfo=pytz.UTC)
END = START + timedelta(minutes=15)


def _states(*samples: tuple[int, Any], unit: str = "W") -> list[dict[str, Any]]:
    """[(minute offset from START, state)] -> HA history states."""
    return [
        {
            "state": str(value),
            "attributes": {"unit_of_measurement": unit},
            "last_changed": (START + timedelta(minutes=minute)).isoformat(),
        }
        for minute, value in samples
    ]


# --------------------------------------------------------------------------- integrator


class TestIntegrator:
    def test_non_finite_samples_are_skipped(self):
        """nan/inf parse as floats; one would poison the slot integral."""
        states = _states((0, 1000), (5, "nan"), (10, "inf"))
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh is not None and math.isfinite(kwh)
        # 1 kW carried across the skipped samples for the whole quarter.
        assert kwh == pytest.approx(0.25)

    def test_only_non_finite_samples_is_none(self):
        assert integrate_power_history_kwh(_states((0, "nan"), (5, "-inf")), START, END) is None

    def test_idle_cutoff_zeroes_standby_only(self):
        """50 W standby for 5 min, 3 kW for 5 min, 100 W for 5 min; idle 100 W."""
        states = _states((0, 50), (5, 3000), (10, 100))
        assert integrate_power_history_kwh(states, START, END) == pytest.approx(
            (0.05 + 3.0 + 0.1) / 12
        )
        assert integrate_power_history_kwh(states, START, END, idle_kw=0.1) == pytest.approx(
            3.0 / 12
        )

    def test_sign_split_books_each_interval_to_its_side(self):
        """+2 kW for 5 min, -3 kW for 10 min: both flows, not their net."""
        states = _states((0, 2000), (5, -3000))
        pos, neg = integrate_power_history_split_kwh(states, START, END)  # type: ignore[misc]
        assert pos == pytest.approx(2.0 / 12)
        assert neg == pytest.approx(3.0 * 10 / 60)

    def test_sign_split_clamps_to_window_and_rejects_nan(self):
        states = _states((-30, 1000), (5, "nan"), (10, -1000), (20, 5000))
        pos, neg = integrate_power_history_split_kwh(states, START, END)  # type: ignore[misc]
        assert pos == pytest.approx(1.0 * 10 / 60)  # pre-window value held from START
        assert neg == pytest.approx(1.0 * 5 / 60)  # the sample after END is ignored

    def test_sign_split_none_without_numeric_samples(self):
        assert integrate_power_history_split_kwh(_states((0, "unavailable")), START, END) is None


# --------------------------------------------------------------------------- recorder


def _base_config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "timezone": "Europe/Stockholm",
        "learning": {"sqlite_path": ":memory:"},
        "input_sensors": {
            "pv_power": "sensor.pv_power",
            "load_power": "sensor.load_power",
            "grid_power": "sensor.grid_power",
            "battery_power": "sensor.battery_power",
            "battery_soc": "sensor.battery_soc",
            "total_pv_production": "sensor.total_pv_production",
            "total_load_consumption": "sensor.total_load_consumption",
        },
        "system": {
            "grid_meter_type": "net",
            "has_battery": True,
            "has_water_heater": True,
            "has_ev_charger": False,
            "grid": {"max_power_kw": 25.0},
        },
        "water_heaters": [],
        "ev_chargers": [],
    }
    config.update(extra)
    return config


async def _run(
    config: dict[str, Any],
    *,
    pv_delta: float,
    load_delta: float,
    grid_kw: float = 7.0,
    history: dict[str, list[dict[str, Any]]] | None = None,
    snapshots: dict[str, float | None] | None = None,
    batch: AsyncMock | None = None,
    batch_history: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """One observation. PV/load from counters; grid from a 7 kW snapshot (1.75 kWh).

    ``history`` maps entity -> HA states, integrated by the REAL integrator (with the
    idle cutoff the recorder passes); entities absent from it have no history.
    """
    history = history or {}
    snapshots = snapshots or {}
    with tempfile.TemporaryDirectory() as tmpdir:
        state_store = RecorderStateStore(Path(tmpdir) / "recorder_state.json")
        state_store.load()
        now = datetime.now(TZ)
        prev = (now - timedelta(minutes=15)).isoformat()
        state_store._state = {  # pyright: ignore[reportPrivateUsage]
            "pv_total": {"value": 100.0, "timestamp": prev},
            "load_total": {"value": 50.0, "timestamp": prev},
        }

        base_kw = {
            "sensor.pv_power": 5.0,
            "sensor.load_power": 7.0,
            "sensor.grid_power": grid_kw,
            "sensor.battery_power": 0.0,
        }

        async def mock_kw(entity: str) -> float | None:
            if entity in snapshots:
                return snapshots[entity]
            return base_kw.get(entity, 0.0)

        async def mock_float(entity: str) -> float | None:
            return 50.0 if entity == "sensor.battery_soc" else None

        async def mock_entity_state(entity: str) -> dict[str, Any] | None:
            return {
                "sensor.total_pv_production": {
                    "state": f"{100.0 + pv_delta}",
                    "attributes": {"unit_of_measurement": "kWh"},
                    "last_updated": now.isoformat(),
                },
                "sensor.total_load_consumption": {
                    "state": f"{50.0 + load_delta}",
                    "attributes": {"unit_of_measurement": "kWh"},
                    "last_updated": now.isoformat(),
                },
            }.get(entity)

        def moved(states, start):
            # Re-anchor the fixture's START-relative timestamps onto the real slot.
            shift = start - START
            return [
                {
                    **s,
                    "last_changed": (datetime.fromisoformat(s["last_changed"]) + shift).isoformat(),
                }
                for s in states
            ]

        async def mock_history(entity, start, end, idle_kw=None):
            states = history.get(entity)
            if states is None:
                return None
            return integrate_power_history_kwh(moved(states, start), start, end, idle_kw=idle_kw)

        if batch_history is not None:
            batch = AsyncMock(
                side_effect=lambda ids, start, end: {
                    e: moved(st, start) for e, st in batch_history.items() if e in ids
                }
            )

        mock_store = MagicMock()
        mock_store.get_system_state = AsyncMock(return_value=None)
        mock_store.set_system_state = AsyncMock()
        mock_store.store_slot_observations = AsyncMock()
        mock_store.close = AsyncMock()
        with (
            patch("backend.recorder.get_ha_sensor_kw_normalized", side_effect=mock_kw),
            patch("backend.recorder.get_ha_sensor_float", side_effect=mock_float),
            patch("backend.recorder.get_ha_entity_state", side_effect=mock_entity_state),
            patch("backend.recorder.get_energy_from_power_history", side_effect=mock_history),
            patch(
                "backend.recorder.get_power_history_batch",
                batch or AsyncMock(return_value=None),
            ),
            patch("backend.recorder.get_nordpool_data", return_value=None),
            patch("backend.recorder.LearningStore", return_value=mock_store),
        ):
            await record_observation_from_current_state(config=config, state_store=state_store)
        df = mock_store.store_slot_observations.call_args[0][0]
        return df.iloc[0].to_dict()


def _flags(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("quality_flags")
    return json.loads(raw) if isinstance(raw, str) and raw else {}


POOL = {"id": "poolpump", "enabled": True, "power_sensor": "sensor.pool_power"}


class TestRescueSubtractsCyclic:
    @pytest.mark.asyncio
    async def test_rescued_base_load_excludes_cyclic_energy(self):
        """Frozen load counter -> rescue. Balance 1.25 PV + 1.75 import = 3.0 kWh total;
        the pool pump ran 1.2 kW all quarter (0.3 kWh) -> base load 2.7, not 3.0."""
        config = _base_config(cyclic_loads=[POOL])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=0.0,
            history={"sensor.pool_power": _states((0, 1200))},
        )
        assert "load_rescued" in _flags(record)
        assert record["load_kwh"] == pytest.approx(2.7, abs=1e-6)

    @pytest.mark.asyncio
    async def test_rescue_and_normal_path_agree(self):
        """The same quarter measured (counter 3.0) or rescued gives the same base load."""
        config = _base_config(cyclic_loads=[POOL])
        hist = {"sensor.pool_power": _states((0, 1200))}
        measured = await _run(config, pv_delta=1.25, load_delta=3.0, history=hist)
        rescued = await _run(config, pv_delta=1.25, load_delta=0.0, history=hist)
        assert "load_rescued" not in _flags(measured)
        assert measured["load_kwh"] == pytest.approx(rescued["load_kwh"], abs=1e-6)

    @pytest.mark.asyncio
    async def test_subtracted_cyclic_energy_is_persisted_on_the_row(self):
        """Both paths record the cyclic energy they took out of load_kwh, so money
        consumers can add it back: base + cyclic == whole house (3.0)."""
        config = _base_config(cyclic_loads=[POOL])
        hist = {"sensor.pool_power": _states((0, 1200))}
        for load_delta in (3.0, 0.0):  # measured, rescued
            record = await _run(config, pv_delta=1.25, load_delta=load_delta, history=hist)
            assert record["cyclic_kwh"] == pytest.approx(0.3, abs=1e-6)
            assert record["load_kwh"] + record["cyclic_kwh"] == pytest.approx(3.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_cyclic_loads_records_zero(self):
        record = await _run(_base_config(), pv_delta=1.25, load_delta=3.0, history={})
        assert record["cyclic_kwh"] == 0.0


SPA = {"id": "spa", "enabled": True, "sensor": "sensor.spa_power"}


class TestHeaterIdleCutoff:
    @pytest.mark.asyncio
    async def test_standby_is_not_heating_and_stays_in_base_load(self):
        """Spa at 60 W circulation all quarter, idle_power_w 100: 0 heating credited,
        the 0.015 kWh stays in base load (counter 3.0 -> base 3.0)."""
        config = _base_config(water_heaters=[{**SPA, "idle_power_w": 100}])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            history={"sensor.spa_power": _states((0, 60))},
        )
        assert record["water_kwh"] == pytest.approx(0.0)
        assert record["water_heater_energy"] == {"spa": 0.0}
        assert record["load_kwh"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_heating_above_idle_still_counts(self):
        config = _base_config(water_heaters=[{**SPA, "idle_power_w": 100}])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            history={"sensor.spa_power": _states((0, 60), (5, 2400))},
        )
        assert record["water_heater_energy"]["spa"] == pytest.approx(2.4 * 10 / 60)
        assert record["load_kwh"] == pytest.approx(3.0 - 0.4)

    @pytest.mark.asyncio
    async def test_unset_idle_keeps_raw_integration(self):
        """No idle_power_w on the heater -> unchanged behaviour (raw power)."""
        config = _base_config(water_heaters=[SPA])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            history={"sensor.spa_power": _states((0, 60))},
        )
        assert record["water_heater_energy"]["spa"] == pytest.approx(0.015)
        assert record["load_kwh"] == pytest.approx(3.0 - 0.015)

    @pytest.mark.asyncio
    async def test_unavailable_heater_is_flagged(self):
        config = _base_config(water_heaters=[{**SPA, "idle_power_w": 100}])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            snapshots={"sensor.spa_power": None},
        )
        assert _flags(record)["water_unmeasured"] == {"spa": "unavailable"}
        assert record["water_kwh"] == 0.0

    @pytest.mark.asyncio
    async def test_snapshot_fallback_is_flagged_and_cut(self):
        config = _base_config(water_heaters=[{**SPA, "idle_power_w": 100}])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            snapshots={"sensor.spa_power": 0.06},
        )
        assert _flags(record)["water_unmeasured"] == {"spa": "snapshot"}
        assert record["water_kwh"] == 0.0

    @pytest.mark.asyncio
    async def test_measured_heater_adds_no_flag(self):
        config = _base_config(water_heaters=[{**SPA, "idle_power_w": 100}])
        record = await _run(
            config,
            pv_delta=1.25,
            load_delta=3.0,
            # PV measured too: a pv_power with no history is itself flagged
            # (pv_unmeasured), which is not what this test is about.
            history={"sensor.spa_power": _states((0, 60)), "sensor.pv_power": _states((0, 5000))},
        )
        assert "quality_flags" not in record or not _flags(record)


def _grid_batt_history():
    # Grid: import 4 kW 5 min, export 3 kW 10 min. Battery: discharge 2 kW 5 min,
    # charge 1.2 kW 10 min.
    return {
        "sensor.grid_power": _states((0, 4000), (5, -3000)),
        "sensor.battery_power": _states((0, 2000), (5, -1200)),
    }


class TestExactSlotShadow:
    @pytest.mark.asyncio
    async def test_sign_conventions_net_meter(self):
        batch = AsyncMock(return_value=_grid_batt_history())
        with patch("backend.recorder.get_power_history_batch", batch):
            out = await _exact_slot_power_shadow(
                {"grid_power": "sensor.grid_power", "battery_power": "sensor.battery_power"},
                "net",
                START,
                END,
            )
        assert batch.await_count == 1  # one request for all entities
        assert set(batch.await_args.args[0]) == {"sensor.grid_power", "sensor.battery_power"}
        assert out == {"import": 0.333, "export": 0.5, "discharge": 0.167, "charge": 0.2}

    @pytest.mark.asyncio
    async def test_inversion_flags_swap_sides(self):
        batch = AsyncMock(return_value=_grid_batt_history())
        with patch("backend.recorder.get_power_history_batch", batch):
            out = await _exact_slot_power_shadow(
                {
                    "grid_power": "sensor.grid_power",
                    "battery_power": "sensor.battery_power",
                    "grid_power_inverted": True,
                    "battery_power_inverted": True,
                },
                "net",
                START,
                END,
            )
        assert out == {"import": 0.5, "export": 0.333, "discharge": 0.2, "charge": 0.167}

    @pytest.mark.asyncio
    async def test_dual_meter_uses_positive_magnitudes(self):
        hist = {
            "sensor.imp": _states((0, 1000)),
            "sensor.exp": _states((0, 0), (10, 600)),
        }
        with patch("backend.recorder.get_power_history_batch", AsyncMock(return_value=hist)):
            out = await _exact_slot_power_shadow(
                {"grid_import_power": "sensor.imp", "grid_export_power": "sensor.exp"},
                "dual",
                START,
                END,
            )
        assert out == {"import": 0.25, "export": 0.05}

    @pytest.mark.asyncio
    async def test_shadow_off_is_identical_and_never_fetches(self):
        hist = {"sensor.pool_power": _states((0, 1200))}
        config = _base_config(cyclic_loads=[POOL])
        batch = AsyncMock(return_value=None)
        absent = await _run(config, pv_delta=1.25, load_delta=0.0, history=hist, batch=batch)
        off = await _run(
            {**config, "recorder": {"exact_slot_power_shadow": False}},
            pv_delta=1.25,
            load_delta=0.0,
            history=hist,
            batch=batch,
        )
        batch.assert_not_awaited()
        for row in (absent, off):
            row.pop("created_at")
        assert absent == off
        assert "shadow_exact" not in _flags(off)

    @pytest.mark.asyncio
    async def test_shadow_on_never_changes_energy_columns(self):
        config = _base_config()
        off = await _run(config, pv_delta=1.25, load_delta=3.0)
        on = await _run(
            {**config, "recorder": {"exact_slot_power_shadow": True}},
            pv_delta=1.25,
            load_delta=3.0,
            batch_history=_grid_batt_history(),
        )
        for col in (
            "pv_kwh",
            "load_kwh",
            "import_kwh",
            "export_kwh",
            "water_kwh",
            "ev_charging_kwh",
            "batt_charge_kwh",
            "batt_discharge_kwh",
        ):
            assert on[col] == off[col], col
        shadow = _flags(on)["shadow_exact"]
        assert shadow["import"] == 0.333 and shadow["export"] == 0.5
        # base from the exact flows: 1.25 + 0.333 - 0.5 + 0.167 - 0.2
        assert shadow["base"] == pytest.approx(1.05, abs=0.002)

    @pytest.mark.asyncio
    async def test_shadow_failure_does_not_cost_the_row(self):
        batch = AsyncMock(side_effect=RuntimeError("boom"))
        record = await _run(
            _base_config(recorder={"exact_slot_power_shadow": True}),
            pv_delta=1.25,
            load_delta=3.0,
            batch=batch,
        )
        assert record["load_kwh"] == pytest.approx(3.0)
        assert "shadow_exact" not in _flags(record)


class TestPvUnmeasuredFlag:
    """PV that did not come from pv_power history is tagged so forecast-vs-actual
    readers (planner intraday PV correction) don't read a dead feed as clouds."""

    @pytest.mark.asyncio
    async def test_counter_fallback_is_flagged(self):
        record = await _run(_base_config(), pv_delta=1.25, load_delta=3.0)
        assert _flags(record).get("pv_unmeasured") == "counter"

    @pytest.mark.asyncio
    async def test_snapshot_fallback_is_flagged(self):
        config = _base_config()
        config["input_sensors"]["total_pv_production"] = ""
        record = await _run(config, pv_delta=1.25, load_delta=3.0)
        assert _flags(record).get("pv_unmeasured") == "snapshot"

    @pytest.mark.asyncio
    async def test_history_integrated_pv_is_not_flagged(self):
        record = await _run(
            _base_config(),
            pv_delta=1.25,
            load_delta=3.0,
            history={"sensor.pv_power": _states((0, 5000))},
        )
        assert "pv_unmeasured" not in _flags(record)

    @pytest.mark.asyncio
    async def test_counter_only_site_is_not_flagged(self):
        """No pv_power configured: the counter IS the designed measurement."""
        config = _base_config()
        del config["input_sensors"]["pv_power"]
        record = await _run(config, pv_delta=1.25, load_delta=3.0)
        assert "pv_unmeasured" not in _flags(record)
