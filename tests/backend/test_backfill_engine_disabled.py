"""BackfillEngine must stay frozen unless explicitly enabled (2026-08-08).

This engine was UNREACHABLE for the entire life of the zero-fabrication bug:
get_last_observation_time is MAX(slot_start), and store_slot_prices minted rows ~48h
into the future, so the gap was always negative and run() always returned early.

Stopping the fabrication makes MAX(slot_start) the present again -- which would wake
this path for the first time in months. It zero-fills absent channels and its auto
sensor map has no battery/water/EV entry, so on this site (total_pv_production
deliberately emptied) it would write pv_kwh=0.0 rows that PASS ml/train.py's
`load_kwh > 0.001` filter straight into the live automatic retrain.
"""

import logging

import pytest

from backend.learning.backfill import BackfillEngine

_STARTED = "Starting backfill process"
_FROZEN = "ha_backfill_enabled=false"


def _engine(learning_config: dict) -> BackfillEngine:
    """Build without __init__ -- it loads config and the whole learning engine.

    Deliberately leaves `store` unset, so reaching the work path is observable.
    """
    eng = BackfillEngine.__new__(BackfillEngine)
    eng.learning_config = learning_config
    return eng


@pytest.mark.asyncio
async def test_disabled_by_default(caplog):
    """No key present -> frozen. This reproduces the real pre-2026-08-08 behaviour."""
    with caplog.at_level(logging.INFO):
        await _engine({}).run()
    assert _FROZEN in caplog.text
    assert _STARTED not in caplog.text


@pytest.mark.asyncio
async def test_explicitly_disabled(caplog):
    with caplog.at_level(logging.INFO):
        await _engine({"ha_backfill_enabled": False}).run()
    assert _STARTED not in caplog.text


@pytest.mark.asyncio
async def test_enabling_actually_lets_it_run(caplog):
    """Guard against the gate being unconditional -- that would be a silent no-op."""
    with caplog.at_level(logging.INFO):
        await _engine({"ha_backfill_enabled": True}).run()
    assert _STARTED in caplog.text
    assert _FROZEN not in caplog.text
