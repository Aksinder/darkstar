"""The servo must not be blind to its own actuation.

Live failure, 2026-08-15: the servo commanded the Tesla ON at 8 A (12:07:04); the grid
went from -5404 W to -151 W as the car took 5.2 kW; sensor.white_betty_charger_power
stayed at 0.0 the whole time because the Tesla integration polls on a multi-minute
cadence. The control law is "target = measured draw + headroom", so 0 W measured plus
151 W of headroom gave a 151 W target — below the car's 3450 W three-phase minimum —
and the servo switched off the very load it had just created (12:08:05), then locked
itself out for 15 minutes on min_off_s.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_ev_surplus_priority import FakeHA, _cfg_dict, _states


class TimedHA(FakeHA):
    """FakeHA whose get_state carries a per-entity last_updated."""

    def __init__(self, states, stamps):
        super().__init__(states)
        self.stamps = stamps  # entity -> ISO 8601

    async def get_state(self, entity):
        base = await super().get_state(entity)
        if base is not None and entity in self.stamps:
            base = {**base, "last_updated": self.stamps[entity]}
        return base


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


# The real timeline of the incident, in one time base: sensor stamps are ISO (as HA
# reports them) and cmd/now are epochs (as the runtime keeps them), so they MUST be
# derived from the same instants or the staleness comparison is meaningless.
BEFORE = "2026-08-15T10:05:00+00:00"  # 12:05 local — predates the command
AFTER = "2026-08-15T10:09:00+00:00"   # 12:09 local — the sensor caught up
CMD_TS = _epoch("2026-08-15T10:07:04+00:00")
STALE_NOW = _epoch("2026-08-15T10:07:29+00:00")   # mid-ramp, sensor still behind
FRESH_NOW = _epoch("2026-08-15T10:09:30+00:00")   # after the sensor reported


def _ctrl(**over):
    cfg = parse_ev_surplus_config(_cfg_dict(**over))
    assert cfg is not None
    return EVSurplusController(cfg), cfg


def _tesla_cfg(cfg):
    return next(c for c in cfg.chargers if c.id == "tesla")


async def _read(ctrl, ha, cfg, now_ts):
    return await ctrl._read_charger(ha, _tesla_cfg(cfg), now_ts, vacation=False)


@pytest.mark.asyncio
async def test_stale_zero_is_replaced_by_the_commanded_draw():
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS  # we wrote at t=1000
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    # 8 A x 3 phases x 230 V
    assert st.current_power_w == pytest.approx(8.0 * 3 * 230.0)


@pytest.mark.asyncio
async def test_measurement_wins_once_the_sensor_catches_up():
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "5200"}), {"sensor.tesla_power": AFTER})
    st = await _read(ctrl, ha, cfg, now_ts=FRESH_NOW)
    assert st.current_power_w == pytest.approx(5200.0)


@pytest.mark.asyncio
async def test_a_fresh_zero_is_believed():
    """The car declined the charge: a CURRENT reading of 0 must not be overridden."""
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": AFTER})
    st = await _read(ctrl, ha, cfg, now_ts=FRESH_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_no_substitution_before_we_have_commanded_anything():
    ctrl, cfg = _ctrl()
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_a_commanded_stop_substitutes_zero_not_the_old_draw():
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 0.0  # stopped
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "5200"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_opt_out_restores_the_raw_reading():
    raw = _cfg_dict()
    for c in raw["ev_surplus"]["chargers"]:
        c["trust_commanded_draw"] = False
    ctrl, cfg = _ctrl(**raw["ev_surplus"])
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_the_car_stays_on_through_its_own_ramp_up():
    """End-to-end reproduction: export collapses because the car ate it."""
    ctrl, _cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_switch["tesla"] = True
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(
        # grid -151 W: the 5.4 kW of export is now inside the car. Battery full.
        _states(**{
            "sensor.grid": "-151", "sensor.batt": "0", "sensor.soc": "100",
            "sensor.tesla_power": "0", "binary_sensor.easee_plug": "off",
        }),
        {"sensor.tesla_power": BEFORE},
    )
    await ctrl.run(ha, now_ts=STALE_NOW, shadow=True)
    assert ctrl._last_a["tesla"] > 0.0, "servo switched off the load it just created"


# ---------------------------------------------------------------------------
# Reality over memory (2026-08-23): a car that started ITSELF.
#
# 13:46:41 the Tesla was plugged in and began charging at 11 kW on its own. The
# servo had switched it off at 11:46, so its memory said OFF, and _is_on prefers
# the commanded state — an 11 kW load was modelled as "off", the servo decided to
# START it, switch.turn_on on an already-charging car failed, and in its model
# there was never a running car to REDUCE. 42 minutes of battery + grid into the
# car. A clear, provably fresh draw against a remembered OFF means the world moved
# without us: adopt it.
# ---------------------------------------------------------------------------

LATER = "2026-08-15T10:12:00+00:00"  # well after the command


@pytest.mark.asyncio
async def test_a_self_started_car_is_adopted_as_on():
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 0.0          # we stopped it earlier...
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    # ...and a reading from AFTER that stop shows 11 kW: it started itself.
    ha = TimedHA(_states(**{"sensor.tesla_power": "11000"}), {"sensor.tesla_power": LATER})
    st = await _read(ctrl, ha, cfg, now_ts=_epoch(LATER) + 30)
    assert st.commanded_on is True
    assert st.current_power_w == 11000.0
    # The amp memory is synced to what it draws, so the next command is a
    # REDUCTION from 11 kW — not a start.
    assert ctrl._last_a["tesla"] == pytest.approx(11000.0 / (3 * 230.0))
    assert ctrl._last_switch["tesla"] is True


@pytest.mark.asyncio
async def test_a_stale_reading_is_still_lag_not_a_self_start():
    """The pre-existing rule, unchanged: a reading that predates our stop is the
    sensor lagging us, and is substituted with 0 — never adopted as a start."""
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 0.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "11000"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.commanded_on is False
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_no_timestamp_means_no_proof_means_no_adoption():
    """Without last_updated we cannot tell lag from a self-start; memory stands."""
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 0.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "11000"}), {})
    st = await _read(ctrl, ha, cfg, now_ts=_epoch(LATER))
    assert st.commanded_on is False


@pytest.mark.asyncio
async def test_a_trickle_is_not_a_self_start():
    """Below half the minimum on-power it is noise or a battery heater, not a charge."""
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 0.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "300"}), {"sensor.tesla_power": LATER})
    st = await _read(ctrl, ha, cfg, now_ts=_epoch(LATER))
    assert st.commanded_on is False


# --- T4: the substitution is bounded in wall-clock time ------------------------------
#
# Without a bound it never ends. These sensors poll and write on CHANGE, so when the car
# declines the value does not move, last_updated never advances past our write, and the
# fiction stands indefinitely. Measured on sensor.white_betty_charger_power: 0.0 held
# with a frozen last_updated for 3.6 h while the integration reported every 600 s.

EXPIRED_NOW = _epoch("2026-08-15T10:24:00+00:00")  # CMD_TS + 1016 s, past the 900 s ttl


@pytest.mark.asyncio
async def test_the_incident_survives_the_bound():
    """The case the mechanism exists for must still be covered.

    ttl >= R + T, where R is the charger's REPORT interval (Tesla 600 s, a hard
    coordinator grid) and T the 60 s servo tick. At the decisive 12:08:05 tick only 61 s
    have elapsed, against a 900 s ttl — 839 s of margin, structural rather than lucky.
    """
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    st = await _read(ctrl, ha, cfg, now_ts=_epoch("2026-08-15T10:08:05+00:00"))
    assert st.current_power_w == pytest.approx(8.0 * 3 * 230.0)


@pytest.mark.asyncio
async def test_substitution_expires_and_the_raw_reading_returns():
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    # Inside the window the fiction stands...
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.current_power_w == pytest.approx(8.0 * 3 * 230.0)
    # ...past it the measurement returns, however unwelcome.
    st = await _read(ctrl, ha, cfg, now_ts=EXPIRED_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_repeated_writes_do_not_extend_the_episode():
    """The ttl is anchored to the FIRST unconfirmed command, not the latest.

    _last_cmd_ts is restamped by every write, so anchoring there would let a servo that
    keeps writing every tick hold the fiction open forever — precisely the failure the
    bound exists to close.
    """
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    await _read(ctrl, ha, cfg, now_ts=STALE_NOW)  # opens the episode
    # The servo writes again, 10 minutes in. The anchor must NOT move.
    ctrl._last_cmd_ts["tesla"] = EXPIRED_NOW - 60.0
    st = await _read(ctrl, ha, cfg, now_ts=EXPIRED_NOW)
    assert st.current_power_w == 0.0


@pytest.mark.asyncio
async def test_a_stop_then_start_gets_a_fresh_substitution_window():
    """The silent trap: an anchor cleared only on sensor confirmation never clears.

    A stop zeroes the commanded amps while the sensor's value is ALREADY 0.0, so
    last_updated never advances past the write. Without an explicit clear on the stop
    path, the NEXT start begins already expired and gets no protection at all — the
    2026-08-15 incident restored by the fix meant to prevent it.
    """
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    await _read(ctrl, ha, cfg, now_ts=EXPIRED_NOW)  # episode expires
    assert "tesla" in ctrl._subst_since

    # The stop path clears the anchor (mirrors what _actuate does on switch_on=False).
    ctrl._subst_since.pop("tesla", None)
    ctrl._subst_warned.discard("tesla")

    # A new start, well after the old episode would have expired, is protected again.
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = EXPIRED_NOW
    st = await _read(ctrl, ha, cfg, now_ts=EXPIRED_NOW + 30.0)
    assert st.current_power_w == pytest.approx(8.0 * 3 * 230.0)


@pytest.mark.asyncio
async def test_catching_up_closes_the_episode():
    """A confirmed reading ends the episode, so the next command starts a fresh ttl."""
    ctrl, cfg = _ctrl()
    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    stale = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    await _read(ctrl, stale, cfg, now_ts=STALE_NOW)
    assert "tesla" in ctrl._subst_since

    fresh = TimedHA(_states(**{"sensor.tesla_power": "5200"}), {"sensor.tesla_power": AFTER})
    st = await _read(ctrl, fresh, cfg, now_ts=FRESH_NOW)
    assert st.current_power_w == pytest.approx(5200.0)
    assert "tesla" not in ctrl._subst_since


@pytest.mark.asyncio
async def test_ttl_is_per_charger_and_configurable():
    """No single global value exists: the Tesla needs >= 660 s (600 s poll grid + 60 s
    tick) while the Easee's ceiling is its 180 s min_off_s. The interval is empty, so
    the bound has to be per-charger."""
    raw = _cfg_dict()
    for c in raw["ev_surplus"]["chargers"]:
        if c["id"] == "tesla":
            c["commanded_draw_ttl_s"] = 120.0
    ctrl, cfg = _ctrl(**raw["ev_surplus"])
    assert _tesla_cfg(cfg).commanded_draw_ttl_s == pytest.approx(120.0)

    ctrl._last_a["tesla"] = 8.0
    ctrl._last_cmd_ts["tesla"] = CMD_TS
    ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": BEFORE})
    # 25 s in: inside the shortened ttl.
    st = await _read(ctrl, ha, cfg, now_ts=STALE_NOW)
    assert st.current_power_w == pytest.approx(8.0 * 3 * 230.0)
    # 200 s in: past it, though the 900 s default would still have been substituting.
    st = await _read(ctrl, ha, cfg, now_ts=CMD_TS + 200.0)
    assert st.current_power_w == 0.0
