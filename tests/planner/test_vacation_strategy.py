"""Away-mode strategy overrides: sell what an empty house would otherwise reserve.

Owner directive 2026-08-27 ("vid vacation_mode borde vi sälja betydligt mer"):
vacation used to touch only the LOADS — water tanks, EVs, anti-legionella — while
every reserve sized for an occupied house stayed put:

* the safety floor keeps an unconditional minimum buffer (10 % of the pack at
  risk_appetite 3) plus a weather buffer, on top of the temporal deficit;
* ``battery_value`` credits energy left at the horizon end at 0.75x the cheapest
  forward IMPORT price — a buy-back that an away household never makes;
* the export floor (20 % live — the executor's guard logs 4-8 blocked exports a
  day at it) reserves capacity so a returning household is not met by an empty pack.

The overrides may only ever RELAX these. That direction is the safety property:
a mis-set key cannot make the planner more conservative and cannot strand the
house, and with every key absent the behaviour is bit-for-bit today's.
"""

from __future__ import annotations

import pandas as pd
import pytest
import pytz

from planner.strategy.s_index import calculate_safety_floor

TZ = "Europe/Stockholm"
BATTERY = {"capacity_kwh": 16.0, "min_soc_percent": 10.0}


def _df(load_kwh: float = 1.0, pv_kwh: float = 0.2, periods: int = 96) -> pd.DataFrame:
    idx = pd.date_range(
        "2026-08-27 00:00", periods=periods, freq="15min", tz=pytz.timezone(TZ)
    )
    return pd.DataFrame(
        {"load_forecast_kwh": [load_kwh] * periods, "pv_forecast_kwh": [pv_kwh] * periods},
        index=idx,
    )


class TestRiskAppetiteRelaxesTheFloor:
    """The override's whole point: a higher risk_appetite means a smaller floor.

    These call calculate_safety_floor WITHOUT an extended forecast, so the
    temporal-deficit term is zero and only the unconditional min_buffer decides.
    That is deliberate — the floor is max(deficit_reserve, min_buffer), so this
    isolates the lever the override actually moves, and it is precisely the
    regime an AWAY house sits in (a vacation-aware load forecast predicts almost
    no deficit). On a deficit-heavy forecast the deficit term dominates and the
    override changes little; the config comment says so.
    """

    def _floor(self, risk: int) -> float:
        floor, _ = calculate_safety_floor(
            _df(), BATTERY, {"risk_appetite": risk, "max_safety_buffer_percent": 40.0}, TZ
        )
        return floor

    def test_higher_risk_gives_a_strictly_lower_floor(self):
        f3, f4, f5 = self._floor(3), self._floor(4), self._floor(5)
        assert f4 < f3, f"risk 4 must reserve less than 3 (got {f4:.2f} vs {f3:.2f})"
        assert f5 < f4, f"risk 5 must reserve less than 4 (got {f5:.2f} vs {f4:.2f})"

    def test_the_freed_capacity_is_material(self):
        # 10 % of a 16 kWh pack is 1.6 kWh of unconditional buffer at risk 3;
        # the point of the feature is that an away house gets to sell it.
        freed = self._floor(3) - self._floor(5)
        assert freed >= 1.0, f"only {freed:.2f} kWh freed — not worth the feature"


class TestExportFloorGuardIsVacationAware:
    """The executor half. Review round 1 caught the planner-only version: the MILP
    planned export down to the lowered floor while the servo kept blocking at the
    configured one, so the away kWh were planned and never sold (live evidence:
    'Export intent blocked: SoC 20.0% at/below export floor 20%', 4-8 hits/day)."""

    def test_configured_floor_applies_when_home(self):
        from executor.controller import effective_export_floor_pct
        from executor.override import SystemState

        assert effective_export_floor_pct(20.0, SystemState()) == 20.0

    def test_vacation_floor_lowers_it(self):
        from executor.controller import effective_export_floor_pct
        from executor.override import SystemState

        st = SystemState(vacation_export_floor_percent=14.0)
        assert effective_export_floor_pct(20.0, st) == 14.0

    def test_vacation_floor_may_never_raise_it(self):
        from executor.controller import effective_export_floor_pct
        from executor.override import SystemState

        st = SystemState(vacation_export_floor_percent=30.0)
        assert effective_export_floor_pct(20.0, st) == 20.0, (
            "the guard protects the pack — an away value must never weaken it upward"
        )

    def test_homecoming_restores_on_the_next_tick(self):
        # The field is runtime state, re-read every tick: clearing it is all that
        # homecoming needs (no persisted config to go stale).
        from executor.controller import effective_export_floor_pct
        from executor.override import SystemState

        away = SystemState(vacation_export_floor_percent=14.0)
        home = SystemState()
        assert effective_export_floor_pct(20.0, away) == 14.0
        assert effective_export_floor_pct(20.0, home) == 20.0


class TestPipelineOverridesAreReal:
    """Exercise the pipeline's OWN expressions, not a copy of them.

    Review round 1: the first draft re-implemented the clamps as local test
    helpers, so inverting max()/min() in pipeline.py left every test green.
    These read the shipped source instead, so a drift breaks the test.
    """

    @staticmethod
    def _pipeline_src() -> str:
        from pathlib import Path

        import planner.pipeline as mod

        return Path(mod.__file__).read_text()

    def test_risk_override_is_relax_only_in_source(self):
        src = self._pipeline_src()
        assert "_eff_risk = max(_base_risk, min(5, int(_vac_risk)))" in src, (
            "the risk clamp must stay max(base, min(5, override)) — relax-only and "
            "capped; a bare min(5, override) would let a low value TIGHTEN the floor"
        )

    def test_risk_override_is_gated_on_vacation_in_source(self):
        src = self._pipeline_src()
        assert '_vac_risk = _vac_strategy.get("risk_appetite") if vacation_enabled else None' in src, (
            "the override must apply only while away"
        )

    def test_battery_value_needs_an_explicit_false_in_source(self):
        src = self._pipeline_src()
        assert '_vac_strategy.get("battery_value_enabled") is False' in src, (
            "an absent key must keep the credit; only an explicit false disables it"
        )

    def test_export_floor_is_lower_only_in_source(self):
        src = self._pipeline_src()
        assert "_new_floor = min(" in src and "if _new_floor < kepler_config.export_floor_soc_percent:" in src, (
            "the planner-side floor override must be lower-only"
        )


class TestVacationResolutionStillAuthoritative:
    """Hoisting the resolution above the strategy block must not change it."""

    def test_wired_entity_wins_both_directions(self):
        from planner.pipeline import resolve_vacation_enabled

        # Entity wired: HA is authoritative even against a stale config flag.
        assert resolve_vacation_enabled(True, False, True) is False
        assert resolve_vacation_enabled(False, True, True) is True
        # No entity: the config flag decides.
        assert resolve_vacation_enabled(True, False, False) is True
        assert resolve_vacation_enabled(False, False, False) is False


@pytest.mark.parametrize("risk", [3, 4, 5])
def test_floor_never_below_min_soc(risk: int):
    floor, _ = calculate_safety_floor(
        _df(), BATTERY, {"risk_appetite": risk, "max_safety_buffer_percent": 40.0}, TZ
    )
    assert floor >= BATTERY["capacity_kwh"] * BATTERY["min_soc_percent"] / 100.0 - 1e-6, (
        "relaxing the reserve must never dip under the battery's own min_soc"
    )
