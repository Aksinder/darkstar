"""
Task 3.5: Pipeline water heater tests.

Covers:
- Per-device mid-block detection from previous schedule (new format)
- Fallback when previous schedule has old format (no water_heaters dict)
- Per-device today's energy tracking via build_water_heater_inputs
"""

import pandas as pd

from planner.pipeline import partition_vacation_water_heaters
from planner.solver.adapter import build_water_heater_inputs


class TestVacationPartition:
    """Vacation mode splits tanks into excluded (keep heating) vs included (comfort off)."""

    def test_default_all_included(self):
        heaters = [{"id": "main"}, {"id": "villavagn"}]
        excluded, included = partition_vacation_water_heaters(heaters)
        assert excluded == []
        assert [h["id"] for h in included] == ["main", "villavagn"]

    def test_excluded_tank_kept_separate(self):
        heaters = [
            {"id": "main", "exclude_from_vacation": False},
            {"id": "villavagn", "exclude_from_vacation": True},
        ]
        excluded, included = partition_vacation_water_heaters(heaters)
        assert [h["id"] for h in excluded] == ["villavagn"]
        assert [h["id"] for h in included] == ["main"]

    def test_disabled_tanks_drop_from_both(self):
        heaters = [
            {"id": "main", "enabled": True},
            {"id": "off", "enabled": False, "exclude_from_vacation": True},
            {"id": "off2", "enabled": False},
        ]
        excluded, included = partition_vacation_water_heaters(heaters)
        assert excluded == []
        assert [h["id"] for h in included] == ["main"]

# ---------------------------------------------------------------------------
# Helpers replicating the pipeline's mid-block detection logic
# ---------------------------------------------------------------------------


def _detect_forced_slots(
    previous_schedule: list[dict],
    enabled_heater_ids: list[str],
    now_slot: pd.Timestamp,
    tz: str = "Europe/Stockholm",
) -> dict[str, set[pd.Timestamp]]:
    """
    Replicate the pipeline's per-device mid-block detection logic (task 3.1).

    Returns:
        heater_id → set of timestamps to force ON
    """
    force_water_by_heater: dict[str, set] = {d: set() for d in enabled_heater_ids}
    if not previous_schedule or not enabled_heater_ids:
        return force_water_by_heater

    now_iso = now_slot.isoformat()
    current_idx = -1
    for i, s in enumerate(previous_schedule):
        if str(s["start_time"]).startswith(now_iso[:16]):
            current_idx = i
            break

    if current_idx < 0:
        return force_water_by_heater

    curr = previous_schedule[current_idx]
    curr_water_heaters = curr.get("water_heaters", {})

    for heater_id in enabled_heater_ids:
        if isinstance(curr_water_heaters, dict):
            heater_data = curr_water_heaters.get(heater_id, {})
            currently_heating = float(heater_data.get("heating_kw", 0.0)) > 0
        else:
            currently_heating = (
                len(enabled_heater_ids) == 1 and float(curr.get("water_heating_kw", 0.0)) > 0
            )

        if currently_heating:
            for j in range(current_idx, len(previous_schedule)):
                slot_s = previous_schedule[j]
                slot_water_heaters = slot_s.get("water_heaters", {})
                if isinstance(slot_water_heaters, dict):
                    slot_heater = slot_water_heaters.get(heater_id, {})
                    slot_heating = float(slot_heater.get("heating_kw", 0.0)) > 0
                else:
                    slot_heating = float(slot_s.get("water_heating_kw", 0.0)) > 0
                if slot_heating:
                    ts = pd.Timestamp(slot_s["start_time"]).tz_localize(tz)
                    force_water_by_heater[heater_id].add(ts)
                else:
                    break

    return force_water_by_heater


# ---------------------------------------------------------------------------
# Mid-block detection tests
# ---------------------------------------------------------------------------


class TestPerDeviceMidBlockDetection:
    """Per-device mid-block detection from previous schedule."""

    def _make_slot(self, start_iso: str, wh1_kw: float = 0.0, wh2_kw: float = 0.0) -> dict:
        """Build a schedule slot in new per-device format."""
        return {
            "start_time": start_iso,
            "water_heaters": {
                "wh1": {"heating_kw": wh1_kw},
                "wh2": {"heating_kw": wh2_kw},
            },
            "water_heating_kw": wh1_kw + wh2_kw,
        }

    def test_active_heater_forces_remaining_planned_slots(self):
        """When wh1 is active at current slot, remaining consecutive wh1 slots are forced."""
        # Use local Stockholm time to match schedule start_time strings
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")  # 10:00 Stockholm
        schedule = [
            self._make_slot("2026-01-15T10:00", wh1_kw=3.0, wh2_kw=0.0),
            self._make_slot("2026-01-15T10:30", wh1_kw=3.0, wh2_kw=0.0),
            self._make_slot("2026-01-15T11:00", wh1_kw=0.0, wh2_kw=0.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1", "wh2"], now)

        # wh1 should have 2 forced slots (10:00 and 10:30)
        assert len(forced["wh1"]) == 2
        # wh2 is not heating → no forced slots
        assert len(forced["wh2"]) == 0

    def test_inactive_heater_gets_no_forced_slots(self):
        """Heater not heating at current slot gets no forced slots."""
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")
        schedule = [
            self._make_slot("2026-01-15T10:00", wh1_kw=0.0, wh2_kw=2.0),
            self._make_slot("2026-01-15T10:30", wh1_kw=0.0, wh2_kw=2.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1", "wh2"], now)

        assert len(forced["wh1"]) == 0
        assert len(forced["wh2"]) == 2

    def test_mid_block_stops_at_gap_in_schedule(self):
        """Forced slots stop as soon as a slot is not planned to heat."""
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")
        schedule = [
            self._make_slot("2026-01-15T10:00", wh1_kw=3.0),
            self._make_slot("2026-01-15T10:30", wh1_kw=0.0),  # gap
            self._make_slot("2026-01-15T11:00", wh1_kw=3.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1"], now)

        # Only the first slot (gap stops locking)
        assert len(forced["wh1"]) == 1

    def test_now_slot_not_in_schedule_no_forced_slots(self):
        """When current time doesn't match any slot, no forced slots produced."""
        now = pd.Timestamp("2026-01-15T12:00:00+01:00")  # Different from 10:00 slot
        schedule = [
            self._make_slot("2026-01-15T10:00", wh1_kw=3.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1"], now)

        assert len(forced["wh1"]) == 0


class TestOldFormatFallback:
    """Fallback when previous schedule has old format (no water_heaters dict)."""

    def _make_old_slot(self, start_iso: str, water_heating_kw: float = 0.0) -> dict:
        """Build a schedule slot in old aggregate-only format."""
        return {
            "start_time": start_iso,
            "water_heating_kw": water_heating_kw,
            # No "water_heaters" key
        }

    def test_single_heater_old_format_no_forced_slots(self):
        """With old format (no water_heaters dict), mid-block detection produces no forced slots.

        The pipeline code uses curr.get("water_heaters", {}) which returns {} (a dict),
        so the per-device dict branch is taken and no heater_id is found → no forced slots.
        This is safe behavior: can't attribute aggregate heating to a specific device.
        """
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")
        schedule = [
            self._make_old_slot("2026-01-15T10:00", water_heating_kw=3.0),
            self._make_old_slot("2026-01-15T10:30", water_heating_kw=3.0),
            self._make_old_slot("2026-01-15T11:00", water_heating_kw=0.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1"], now)

        # Old format: no per-device water_heaters dict → no attribution → no forced slots
        assert len(forced["wh1"]) == 0

    def test_multiple_heaters_old_format_no_forced_slots(self):
        """With multiple heaters and old format, no heater gets forced (ambiguous)."""
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")
        schedule = [
            self._make_old_slot("2026-01-15T10:00", water_heating_kw=3.0),
        ]
        forced = _detect_forced_slots(schedule, ["wh1", "wh2"], now)

        # Can't attribute aggregate to specific device → no forced slots
        assert len(forced["wh1"]) == 0
        assert len(forced["wh2"]) == 0

    def test_empty_schedule_no_forced_slots(self):
        """Empty previous schedule produces no forced slots."""
        now = pd.Timestamp("2026-01-15T10:00:00+01:00")
        forced = _detect_forced_slots([], ["wh1"], now)

        assert len(forced["wh1"]) == 0


# ---------------------------------------------------------------------------
# Per-device today's energy tracking
# ---------------------------------------------------------------------------


class TestPerDeviceTodaysEnergyTracking:
    """Per-device heated_today_kwh tracking via build_water_heater_inputs."""

    def test_per_device_heated_today_kwh_applied(self):
        """Each heater gets its own heated_today_kwh from water_heater_states."""
        heaters = [
            {"id": "wh1", "enabled": True, "power_kw": 3.0, "min_kwh_per_day": 6.0},
            {"id": "wh2", "enabled": True, "power_kw": 2.0, "min_kwh_per_day": 4.0},
        ]
        states = [
            {"id": "wh1", "heated_today_kwh": 2.5, "force_on_slots": None},
            {"id": "wh2", "heated_today_kwh": 1.0, "force_on_slots": None},
        ]

        result = build_water_heater_inputs(heaters, water_heater_states=states)

        assert len(result) == 2
        wh1 = next(r for r in result if r.id == "wh1")
        wh2 = next(r for r in result if r.id == "wh2")
        assert wh1.heated_today_kwh == 2.5
        assert wh2.heated_today_kwh == 1.0

    def test_missing_state_defaults_to_zero(self):
        """Heater with no matching state gets heated_today_kwh=0.0."""
        heaters = [
            {"id": "wh1", "enabled": True, "power_kw": 3.0, "min_kwh_per_day": 6.0},
        ]

        result = build_water_heater_inputs(heaters, water_heater_states=[])

        assert result[0].heated_today_kwh == 0.0

    def test_force_on_slots_set_from_states(self):
        """force_on_slots from mid-block detection is passed through via states."""
        heaters = [
            {"id": "wh1", "enabled": True, "power_kw": 3.0, "min_kwh_per_day": 6.0},
        ]
        states = [
            {"id": "wh1", "heated_today_kwh": 0.0, "force_on_slots": [0, 1, 2]},
        ]

        result = build_water_heater_inputs(heaters, water_heater_states=states)

        assert result[0].force_on_slots == [0, 1, 2]

    def test_no_states_produces_default_inputs(self):
        """Without states, heaters still get WaterHeaterInput with defaults."""
        heaters = [
            {"id": "wh1", "enabled": True, "power_kw": 3.0, "min_kwh_per_day": 6.0},
        ]

        result = build_water_heater_inputs(heaters)

        assert len(result) == 1
        assert result[0].heated_today_kwh == 0.0
        assert result[0].force_on_slots is None
