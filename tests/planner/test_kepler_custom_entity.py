from datetime import datetime, timedelta

from planner.solver.kepler import KeplerSolver
from planner.solver.types import (
    ExcessPVSinkSpec,
    KeplerConfig,
    KeplerInput,
    KeplerInputSlot,
)


def _make_slots(
    n: int = 8,
    pv_kwh: float = 10.0,
    load_kwh: float = 1.0,
    export_price: float = 0.0,
    import_price: float = 1.0,
) -> list[KeplerInputSlot]:
    start = datetime(2025, 6, 1, 12, 0)
    return [
        KeplerInputSlot(
            start_time=start + timedelta(minutes=15 * i),
            end_time=start + timedelta(minutes=15 * (i + 1)),
            load_kwh=load_kwh,
            pv_kwh=pv_kwh,
            import_price_sek_kwh=import_price,
            export_price_sek_kwh=export_price,
        )
        for i in range(n)
    ]


class TestCustomEntitySolverVariable:
    def test_solver_prefers_export_when_reward_is_low(self):
        """With high SoC target penalty, solver won't discharge for export.
        Entity activation then directly competes with export for surplus PV."""
        capacity = 10.0
        initial_soc = capacity * 0.97

        # PV=3.0, load=1.0 → 2.0 kWh surplus per slot
        # Without entity: export 2.0 kWh @ 5.0 = 10.0 SEK/slot
        # With entity: export 1.5 kWh @ 5.0 + reward 0.5*2.0*0.25 = 7.5 + 0.25 = 7.75 SEK
        # Solver should NOT activate entity
        slots = _make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=5.0)

        config = KeplerConfig(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=0.5,
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=2.0,
        )
        input_data = KeplerInput(slots=slots, initial_soc_kwh=initial_soc)

        result = KeplerSolver().solve(input_data, config)
        assert result.is_optimal

        for s in result.slots:
            assert not s.custom_entity_active, (
                "Custom entity should NOT activate — it costs 0.5 kWh export revenue "
                "(2.5 SEK) for only 0.25 SEK reward."
            )

    def test_solver_activates_entity_when_reward_exceeds_export(self):
        """Same scenario but reward >> export price. Solver activates entity."""
        capacity = 10.0
        initial_soc = capacity * 0.97

        slots = _make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=0.1)

        config = KeplerConfig(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=2.0,
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=2.0,
        )
        input_data = KeplerInput(slots=slots, initial_soc_kwh=initial_soc)

        result = KeplerSolver().solve(input_data, config)
        assert result.is_optimal

        active_slots = [s for s in result.slots if s.custom_entity_active]
        assert len(active_slots) > 0, (
            "Custom entity should activate when reward (2.0) >> export price (0.1)"
        )

    def test_custom_entity_power_kw_sizes_reward(self):
        capacity = 10.0
        initial_soc = capacity * 0.97

        slots = _make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=0.5)

        base_kwargs = dict(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=1.0,
            excess_pv_soc_threshold_percent=95.0,
        )

        config_high = KeplerConfig(
            **base_kwargs,
            excess_pv_custom_entity_power_kw=5.0,
        )
        config_low = KeplerConfig(
            **base_kwargs,
            excess_pv_custom_entity_power_kw=0.1,
        )

        input_data = KeplerInput(slots=slots, initial_soc_kwh=initial_soc)

        result_high = KeplerSolver().solve(input_data, config_high)
        result_low = KeplerSolver().solve(input_data, config_low)

        assert result_high.is_optimal
        assert result_low.is_optimal

        high_active = sum(1 for s in result_high.slots if s.custom_entity_active)
        low_active = sum(1 for s in result_low.slots if s.custom_entity_active)
        assert high_active >= low_active, (
            f"Higher power_kw should produce equal or more activation slots. "
            f"Got high_power active={high_active}, low_power active={low_active}"
        )


class TestCustomEntityPriceCeiling:
    """The price ceiling restricts the sink to low/minus-price surplus hours.

    This is the "low or minus price" trigger for the villavagn-AC cooling sink: even
    with a strong activation reward, the sink stays off when grid export still pays
    above the ceiling, and fires once export drops at/below it.
    """

    def _base(self, export_price: float, **over):
        capacity = 10.0
        initial_soc = capacity * 0.97
        kwargs = dict(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=2.0,  # strong reward — only price gate should hold it off
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=2.0,
        )
        kwargs.update(over)
        slots = _make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=export_price)
        return KeplerInput(slots=slots, initial_soc_kwh=initial_soc), KeplerConfig(**kwargs)

    def test_blocked_when_export_above_ceiling(self):
        # Export pays 0.5 > ceiling 0.2 → never soak locally, sell instead.
        inp, cfg = self._base(export_price=0.5, excess_pv_price_ceiling_sek_per_kwh=0.2)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert not any(s.custom_entity_active for s in result.slots)

    def test_active_when_export_at_or_below_ceiling(self):
        # Export pays 0.1 <= ceiling 0.2 → soak surplus locally.
        inp, cfg = self._base(export_price=0.1, excess_pv_price_ceiling_sek_per_kwh=0.2)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert any(s.custom_entity_active for s in result.slots)

    def test_none_ceiling_is_legacy_unrestricted(self):
        # No ceiling → high reward activates even at a high export price (old behaviour).
        inp, cfg = self._base(export_price=0.5, excess_pv_price_ceiling_sek_per_kwh=None)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert any(s.custom_entity_active for s in result.slots)


class TestCustomEntityIndependentEnable:
    """custom_entity can run independently of the primary `sink` selector.

    With sink set to water_heater_boost, the custom_entity sink (e.g. the villavagn AC)
    still activates when excess_pv_custom_entity_enabled=True — so both surplus uses
    coexist instead of being mutually exclusive.
    """

    def _cfg(self, **over):
        capacity = 10.0
        initial_soc = capacity * 0.97
        kwargs = dict(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            # Primary sink is the VVB boost; the AC sink rides alongside via the flag.
            excess_pv_sink="water_heater_boost",
            excess_pv_reward_sek_per_kwh=2.0,
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=2.0,
        )
        kwargs.update(over)
        return KeplerInput(
            slots=_make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=0.1),
            initial_soc_kwh=initial_soc,
        ), KeplerConfig(**kwargs)

    def test_custom_entity_runs_with_boost_sink_when_enabled(self):
        inp, cfg = self._cfg(excess_pv_custom_entity_enabled=True)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert any(s.custom_entity_active for s in result.slots)

    def test_custom_entity_off_with_boost_sink_when_not_enabled(self):
        # Default: flag off and sink != custom_entity → custom entity stays off.
        inp, cfg = self._cfg(excess_pv_custom_entity_enabled=False)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert not any(s.custom_entity_active for s in result.slots)


class TestSinkLadder:
    """Prioritized excess-PV sink ladder (build #10).

    Ordered rungs share the flag + SoC-gate scaffolding of the legacy single
    custom_entity slot; priority is SOFT (reward * (1 - i*eps) per kWh), so
    scarce surplus fills earlier rungs first and abundant surplus runs them all.
    """

    def _cfg(self, sinks, *, n=8, export_price=0.1, import_price=10.0, pv=1.5, load=1.0, **over):
        capacity = 10.0
        initial_soc = capacity * 0.97
        kwargs = {
            "capacity_kwh": capacity,
            "max_charge_power_kw": 5.0,
            "max_discharge_power_kw": 5.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "min_soc_percent": 0.0,
            "max_soc_percent": 100.0,
            "wear_cost_sek_per_kwh": 0.01,
            "enable_export": True,
            "max_export_power_kw": 10.0,
            "target_soc_kwh": initial_soc,
            "target_soc_penalty_sek": 1000.0,
            "excess_pv_slots": [True] * n,
            "excess_pv_sink": "disabled",
            "excess_pv_reward_sek_per_kwh": 2.0,
            "excess_pv_soc_threshold_percent": 95.0,
            "excess_pv_sinks": sinks,
        }
        kwargs.update(over)
        slots = _make_slots(
            n=n, pv_kwh=pv, load_kwh=load, export_price=export_price, import_price=import_price
        )
        return KeplerInput(slots=slots, initial_soc_kwh=initial_soc), KeplerConfig(**kwargs)

    def test_scarce_surplus_fills_first_rung_only(self):
        # Surplus is 0.5 kWh/slot (2 kW); each rung draws 2 kW. One rung fits the
        # surplus, two would need 10 SEK/kWh grid import for a ~2 SEK/kWh reward.
        # The soft priority epsilon must route the surplus to rung 0.
        sinks = [
            ExcessPVSinkSpec(id="rung0", power_kw=2.0, enabled=True),
            ExcessPVSinkSpec(id="rung1", power_kw=2.0, enabled=True),
        ]
        inp, cfg = self._cfg(sinks)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert all(s.sink_states["rung0"] for s in result.slots), (
            "rung 0 (highest priority) must soak the scarce surplus"
        )
        assert not any(s.sink_states["rung1"] for s in result.slots), (
            "rung 1 must NOT run — surplus only covers one rung and importing at "
            "10 SEK/kWh for a ~2 SEK/kWh reward is a loss"
        )

    def test_abundant_surplus_runs_all_rungs(self):
        # Surplus 2.5 kWh/slot (10 kW) covers both 2 kW rungs comfortably.
        sinks = [
            ExcessPVSinkSpec(id="rung0", power_kw=2.0, enabled=True),
            ExcessPVSinkSpec(id="rung1", power_kw=2.0, enabled=True),
        ]
        inp, cfg = self._cfg(sinks, pv=3.5, load=1.0)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert all(s.sink_states["rung0"] for s in result.slots)
        assert all(s.sink_states["rung1"] for s in result.slots)

    def test_per_sink_price_ceiling(self):
        # Export pays 0.1: rung 0's ceiling (0.05) blocks it, rung 1's (0.2)
        # permits it — the gate is per-rung, not global.
        sinks = [
            ExcessPVSinkSpec(
                id="rung0", power_kw=2.0, price_ceiling_sek_per_kwh=0.05, enabled=True
            ),
            ExcessPVSinkSpec(id="rung1", power_kw=2.0, price_ceiling_sek_per_kwh=0.2, enabled=True),
        ]
        inp, cfg = self._cfg(sinks, export_price=0.1)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert not any(s.sink_states["rung0"] for s in result.slots)
        assert any(s.sink_states["rung1"] for s in result.slots)

    def test_disabled_rung_never_runs(self):
        sinks = [
            ExcessPVSinkSpec(id="rung0", power_kw=2.0, enabled=True),
            ExcessPVSinkSpec(id="observer", power_kw=2.0, enabled=False),
        ]
        inp, cfg = self._cfg(sinks, pv=3.5, load=1.0)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        assert all("observer" not in s.sink_states for s in result.slots), (
            "a disabled rung gets no solver variable and no reported state"
        )

    def test_first_sink_mirrors_custom_entity_active(self):
        sinks = [
            ExcessPVSinkSpec(id="rung0", power_kw=2.0, enabled=True),
            ExcessPVSinkSpec(id="rung1", power_kw=2.0, enabled=True),
        ]
        inp, cfg = self._cfg(sinks)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        for s in result.slots:
            assert s.custom_entity_active == s.sink_states["rung0"]

    def test_all_disabled_ladder_suppresses_legacy_fallback(self):
        # Observe-first rollout: a ladder whose rungs are ALL disabled must win the
        # dual-read over the legacy custom_entity block. The executor skips disabled
        # rungs and actuates nothing, so the planner must not resurrect a phantom
        # "custom_entity" rung — the surplus is exported instead.
        sinks = [ExcessPVSinkSpec(id="observer", power_kw=1.0, enabled=False)]
        inp, cfg = self._cfg(
            sinks,
            pv=3.0,
            load=1.0,
            excess_pv_sink="custom_entity",
            excess_pv_custom_entity_enabled=True,
            excess_pv_custom_entity_power_kw=2.0,
        )
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        for s in result.slots:
            assert not s.custom_entity_active
            assert not s.sink_states.get("custom_entity")
            assert "observer" not in s.sink_states, (
                "a disabled rung gets no solver variable and no reported state"
            )

    def test_legacy_scalar_config_synthesizes_single_rung(self):
        # No excess_pv_sinks list: the legacy scalar fields must synthesize a
        # single "custom_entity" rung so old configs behave byte-identically.
        capacity = 10.0
        initial_soc = capacity * 0.97
        cfg = KeplerConfig(
            capacity_kwh=capacity,
            max_charge_power_kw=5.0,
            max_discharge_power_kw=5.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=initial_soc,
            target_soc_penalty_sek=1000.0,
            excess_pv_slots=[True] * 8,
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=2.0,
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=2.0,
        )
        inp = KeplerInput(
            slots=_make_slots(n=8, pv_kwh=3.0, load_kwh=1.0, export_price=0.1),
            initial_soc_kwh=initial_soc,
        )
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        active = [s for s in result.slots if s.custom_entity_active]
        assert active, "legacy scalar path must still activate"
        for s in result.slots:
            assert s.sink_states.get("custom_entity") == s.custom_entity_active


class TestSinkNotHourlyBlocked:
    """Sink binaries stay per-slot even when water_hourly_blocks is on (build #9
    invariant). Hour-tying the sinks held them OFF until the next hour boundary
    whenever the SoC gate opened mid-hour."""

    def test_sink_activates_on_first_gated_slot_not_hour_boundary(self):
        # 8 quarter-slots from 12:00 (two wall-clock hours). Battery starts at 90%
        # of 16 kWh; max charge 3 kW = 0.75 kWh/slot, so start-of-slot SoC crosses
        # the 95% gate (15.2 kWh) at slot 2 — mid-hour. Build #9 behavior: the sink
        # fires from slot 2. An hour-tied sink group would hold it off until slot 4.
        capacity = 16.0
        n = 8
        start = datetime(2025, 6, 1, 12, 0)
        slots = [
            KeplerInputSlot(
                start_time=start + timedelta(minutes=15 * i),
                end_time=start + timedelta(minutes=15 * (i + 1)),
                load_kwh=0.25,
                pv_kwh=2.0,  # big surplus every slot
                import_price_sek_kwh=1.0,
                export_price_sek_kwh=0.0,  # below ceiling -> price_ok everywhere
            )
            for i in range(n)
        ]
        cfg = KeplerConfig(
            capacity_kwh=capacity,
            max_charge_power_kw=3.0,
            max_discharge_power_kw=3.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            min_soc_percent=0.0,
            max_soc_percent=100.0,
            wear_cost_sek_per_kwh=0.01,
            enable_export=True,
            max_export_power_kw=10.0,
            target_soc_kwh=capacity,
            target_soc_penalty_sek=0.0,
            excess_pv_slots=[True] * n,
            # LIVE legacy config shape: scalar custom_entity fields, no sinks list.
            excess_pv_sink="custom_entity",
            excess_pv_reward_sek_per_kwh=0.5,
            excess_pv_soc_threshold_percent=95.0,
            excess_pv_custom_entity_power_kw=1.0,
            excess_pv_price_ceiling_sek_per_kwh=0.2,
            water_hourly_blocks=True,
        )
        inp = KeplerInput(slots=slots, initial_soc_kwh=14.4)
        result = KeplerSolver().solve(inp, cfg)
        assert result.is_optimal
        plan = [bool(s.custom_entity_active) for s in result.slots]
        assert plan == [False, False, True, True, True, True, True, True], (
            f"sink must activate on the first SoC-gated slot (mid-hour), not the "
            f"next hour boundary; got {plan}"
        )
