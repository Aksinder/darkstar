"""Phase-1 phase-aware observability: surface the planner's forward per-phase imbalance cost.

The net-node MILP is phase-blind; the realism sim re-prices the optimal plan against the
measured per-phase load split and reports the hidden cost. build_realism_sensors turns that into
a single HA sensor so the structural loss is visible instead of buried in the schedule meta.
"""

from backend.learning.cycle_publisher import build_realism_sensors


def test_empty_or_missing_realism_publishes_nothing():
    assert build_realism_sensors(None) == []
    assert build_realism_sensors({}) == []


def test_builds_imbalance_cost_sensor():
    sensors = build_realism_sensors(
        {
            "gap_sek": 2.63,
            "extra_import_kwh": 2.12,
            "phase_flagged_slots": 56,
            "idle_exposed_slots": 91,
        }
    )
    assert len(sensors) == 1
    s = sensors[0]
    assert s.object_id == "darkstar_phase_imbalance_cost"
    assert s.entity_id == "sensor.darkstar_phase_imbalance_cost"
    assert s.state == "2.63"
    assert s.unit == "SEK"
    assert s.device_class == "monetary"
    assert s.attributes["extra_import_kwh"] == 2.12
    assert s.attributes["phase_flagged_slots"] == 56
    assert s.attributes["idle_exposed_slots"] == 91

    payload = s.to_payload()
    assert payload["state"] == "2.63"
    assert payload["attributes"]["unit_of_measurement"] == "SEK"
    assert payload["attributes"]["device_class"] == "monetary"


def test_handles_partial_realism_dict():
    # Only gap_sek present -> still builds, missing fields default to 0.
    sensors = build_realism_sensors({"gap_sek": 1.0})
    assert len(sensors) == 1
    assert sensors[0].state == "1.0"
    assert sensors[0].attributes["extra_import_kwh"] == 0.0
    assert sensors[0].attributes["phase_flagged_slots"] == 0
