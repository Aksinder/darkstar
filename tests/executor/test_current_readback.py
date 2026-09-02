"""T1: read the charger's current limit back, and yield to a foreign clamp.

Darkstar writes number.white_betty_charge_current but never read it back, so _last_a was
what it INTENDED, not what the device holds. Reading it back is only safe under two gates
and one direction rule, and each of them exists for a specific, measured reason:

- FRESHNESS (dev_ts > our last write): the Easee answers its own write in ~2 s, so an
  ungated readback races the command it is supposed to verify.
- KEY (never create _last_a from a readback): the entity is a LIMIT REGISTER, not a
  charging indicator — number.white_betty_charge_current reads 16 with the cable
  unplugged and the switch off. Creating the key hands commanded_on a definite value on
  a process that has never actuated, while _last_switch stays absent — and the stop gate
  treats an absent key as "changed", so that is the one path to a real turn_off.
- DIRECTION (downward only, never <= 0): commanded_on is exactly `_last_a > 0.0`, and
  _last_a feeds the fuse clamp as commanded_current_a, where the own-draw term must
  never over-state.

The foreign clamp itself is a live HA automation, darkstar_ev_laddvakt_vid_dod_add_on_
sakring, which writes exactly 5 A (Tesla) and 6 A (Easee) when it thinks the add-on has
died. Darkstar yields the RAISE to it and keeps the switch and the right to REDUCE.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from executor.ev_surplus_runtime import EVSurplusController, parse_ev_surplus_config
from tests.executor.test_commanded_draw import TimedHA, _epoch
from tests.executor.test_ev_surplus_priority import _cfg_dict, _states

READBACK = "number.tesla_amps"  # the tesla's current_entity in the test config
MIN_A = 5.0  # its min_current_a, matching production

CMD = _epoch("2026-09-02T05:00:00+00:00")
BEFORE = "2026-09-02T04:59:00+00:00"  # readback predates our write => stale
AFTER = "2026-09-02T05:00:30+00:00"  # readback moved after our write => fresh
NOW = _epoch("2026-09-02T05:01:00+00:00")


def _ctrl():
    cfg = parse_ev_surplus_config(_cfg_dict())
    assert cfg is not None
    return EVSurplusController(cfg), cfg


def _tesla(cfg):
    return next(c for c in cfg.chargers if c.id == "tesla")


async def _read(ctrl, cfg, *, device_a, stamp=AFTER, last_a=16.0, cmd_ts=CMD):
    """One _read_charger tick with the readback entity present at device_a."""
    if last_a is not None:
        ctrl._last_a["tesla"] = last_a
    if cmd_ts is not None:
        ctrl._last_cmd_ts["tesla"] = cmd_ts
    ha = TimedHA(
        _states(**{READBACK: str(device_a), "sensor.tesla_power": "0"}),
        {READBACK: stamp, "sensor.tesla_power": stamp},
    )
    return await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_readback_matching_intent_changes_nothing(self):
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=16.0, last_a=16.0)
        assert ctrl._last_a["tesla"] == pytest.approx(16.0)
        assert "tesla" not in ctrl._clamp_a

    @pytest.mark.asyncio
    async def test_a_lower_readback_syncs_downward(self):
        """The device holds less than we asked for: believe the device."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=10.0, last_a=16.0)
        assert ctrl._last_a["tesla"] == pytest.approx(10.0)
        assert "tesla" not in ctrl._clamp_a  # 10 > the 5 A floor: not a clamp

    @pytest.mark.asyncio
    async def test_a_higher_readback_never_raises_last_a(self):
        """_last_a feeds the fuse clamp's own-draw term, which must never over-state."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=16.0, last_a=8.0)
        assert ctrl._last_a["tesla"] == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_a_stale_readback_is_ignored(self):
        """A reading that predates our write may just be echoing the old value."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=5.0, stamp=BEFORE, last_a=16.0)
        assert ctrl._last_a["tesla"] == pytest.approx(16.0)
        assert "tesla" not in ctrl._clamp_a

    @pytest.mark.asyncio
    async def test_readback_never_creates_the_key(self):
        """The register reads 16 with the cable unplugged. Creating _last_a from it would
        tell the system the car is commanded ON while _last_switch is still absent — and
        the stop gate reads an absent key as changed."""
        ctrl, cfg = _ctrl()
        ctrl._last_cmd_ts["tesla"] = CMD
        ha = TimedHA(
            _states(**{READBACK: "16", "sensor.tesla_power": "0"}),
            {READBACK: AFTER, "sensor.tesla_power": AFTER},
        )
        st = await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert "tesla" not in ctrl._last_a
        assert st.commanded_on is not True

    @pytest.mark.asyncio
    async def test_a_zero_readback_never_enters_last_a(self):
        """commanded_on IS `_last_a > 0.0`, so a zero is the single value that could
        flip it and manufacture a stop out of a readback."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=0.0, last_a=16.0)
        assert ctrl._last_a["tesla"] == pytest.approx(16.0)

    @pytest.mark.asyncio
    async def test_an_absent_readback_entity_changes_nothing(self):
        ctrl, cfg = _ctrl()
        ctrl._last_a["tesla"] = 16.0
        ctrl._last_cmd_ts["tesla"] = CMD
        ha = TimedHA(_states(**{"sensor.tesla_power": "0"}), {"sensor.tesla_power": AFTER})
        await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert ctrl._last_a["tesla"] == pytest.approx(16.0)
        assert "tesla" not in ctrl._clamp_a

    @pytest.mark.asyncio
    async def test_unavailable_readback_changes_nothing(self):
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a="unavailable", last_a=16.0)
        assert ctrl._last_a["tesla"] == pytest.approx(16.0)
        assert "tesla" not in ctrl._clamp_a


class TestForeignClamp:
    @pytest.mark.asyncio
    async def test_a_drop_to_the_floor_we_did_not_command_is_a_clamp(self):
        """The live watchdog writes exactly 5 A for the Tesla."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=5.0, last_a=16.0)
        assert ctrl._clamp_a["tesla"] == pytest.approx(5.0)
        assert "tesla" in ctrl._clamp_since
        # ...and the downward sync still happened.
        assert ctrl._last_a["tesla"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_our_own_floor_command_is_not_a_clamp(self):
        """THE discriminator. Darkstar commanded ('tesla', True, 5.0) 42 times on
        2026-08-31, and 5 A is the Tesla's own minimum — so the value alone cannot tell
        a foreign clamp from our own legitimate floor. `prev > min_current_a` can."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=MIN_A, last_a=MIN_A)
        assert "tesla" not in ctrl._clamp_a

    @pytest.mark.asyncio
    async def test_the_clamp_releases_when_the_limit_rises(self):
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=5.0, last_a=16.0)
        assert "tesla" in ctrl._clamp_a
        await _read(ctrl, cfg, device_a=12.0, last_a=5.0)
        assert "tesla" not in ctrl._clamp_a
        assert "tesla" not in ctrl._clamp_since

    @pytest.mark.asyncio
    async def test_a_departure_clears_the_clamp(self):
        """A car that left and came back gets a clean slate."""
        ctrl, cfg = _ctrl()
        await _read(ctrl, cfg, device_a=5.0, last_a=16.0)
        assert "tesla" in ctrl._clamp_a
        ha = TimedHA(
            _states(**{READBACK: "5", "binary_sensor.tesla_plug": "off"}),
            {READBACK: AFTER},
        )
        await ctrl._read_charger(ha, _tesla(cfg), NOW, vacation=False)
        assert "tesla" not in ctrl._clamp_a


class TestYieldOnlyBlocksRaises:
    """Darkstar yields the RAISE, never the load: stops and fuse reductions always go."""

    @staticmethod
    def _cmd(amps, *, fuse_limited=False, switch_on=True):
        from executor.ev_surplus import ChargerCommand

        return ChargerCommand(
            "tesla", switch_on=switch_on, set_current_a=amps,
            target_power_w=amps * 3 * 230.0, reason="test",
            fuse_limited=fuse_limited,
        )

    @pytest.mark.asyncio
    async def test_a_raise_above_the_clamp_is_suppressed(self):
        ctrl, cfg = _ctrl()
        ctrl._clamp_a["tesla"] = 5.0
        ctrl._last_a["tesla"] = 5.0
        ha = TimedHA(_states(), {})
        res = await ctrl._actuate(ha, _tesla(cfg), self._cmd(16.0), NOW, shadow=False)
        assert res.suppressed == "foreign_clamp"

    @pytest.mark.asyncio
    async def test_a_fuse_reduction_is_never_suppressed(self):
        """Yielding the DOWNWARD direction would turn a courtesy into a safety
        regression on a shared fuse."""
        ctrl, cfg = _ctrl()
        ctrl._clamp_a["tesla"] = 5.0
        ctrl._last_a["tesla"] = 16.0
        ha = TimedHA(_states(), {})
        res = await ctrl._actuate(
            ha, _tesla(cfg), self._cmd(6.0, fuse_limited=True), NOW, shadow=False
        )
        assert res.suppressed != "foreign_clamp"
