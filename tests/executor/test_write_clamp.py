"""Number writes are clamped to the entity's own min/max, and a refused write
says why.

HA rejects a number.set_value outside the entity range (HTTP 400), so a request
like "max charge power 0" against a Sungrow number with min 10 failed every tick,
and the "failed 3 ticks in a row" notice could not say why: the dispatcher threw
the HA error away. Hand port of the idea in upstream d0b3f956 WITHOUT its step
snap, which is anchored at min and shifts every Sungrow setpoint (9500 -> 9510).

Safety rule under test throughout: missing or garbage attributes mean NO clamp,
never a different value.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from executor.actions import (
    ActionDispatcher,
    HACallError,
    HAClient,
    _clamp_to_entity_range,
)
from executor.config import ExcessPVSinkSpec, ExecutorConfig
from executor.controller import ControllerDecision
from executor.profiles import EntityDefinition, InverterProfile, ModeAction

SUNGROW = {"min": 10, "max": 9900, "step": 100}  # live number.battery_max_*_power
CHARGE = "number.battery_max_charge_power"


@pytest.fixture(autouse=True)
def _fast_readback(monkeypatch):
    import executor.actions as actions_mod

    monkeypatch.setattr(actions_mod, "VERIFY_MAX_WAIT_MS", 3)
    monkeypatch.setattr(actions_mod, "VERIFY_POLL_INTERVAL_MS", 1)


# --------------------------------------------------------------------------- #
# Pure clamp
# --------------------------------------------------------------------------- #
class TestClampToEntityRange:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 10), (0.0, 10), (5, 10), (-50, 10), (10000, 9900), (9900.5, 9900), (1e9, 9900)],
    )
    def test_out_of_range_is_clamped(self, value, expected):
        assert _clamp_to_entity_range(value, SUNGROW) == expected

    @pytest.mark.parametrize("value", [10, 100, 110, 3000, 3010, 9500, 9510, 9900, 55.5])
    def test_in_range_is_untouched_no_step_snap(self, value):
        """No step rounding at all: 9500 stays 9500 (upstream's min-anchored snap
        made it 9510), and an off-grid value is left for HA/the device to judge."""
        out = _clamp_to_entity_range(value, SUNGROW)
        assert out is value

    def test_forced_power_10w_is_not_snapped_to_zero(self):
        out = _clamp_to_entity_range(10.0, {"min": 0, "max": 9900, "step": 100})
        assert out == 10.0

    @pytest.mark.parametrize(
        "attrs",
        [
            None,
            {},
            "min=10",
            ["min", 10],
            {"min": None, "max": None},
            {"min": "unknown", "max": "unavailable"},
            {"min": "abc", "max": object()},
            {"min": float("nan"), "max": float("inf")},
            {"min": True, "max": False},
            {"min": 9900, "max": 10},  # inverted: garbage, ignore both
            {"friendly_name": "x", "step": 100},
        ],
    )
    @pytest.mark.parametrize("value", [0, 9500, 10000, -5])
    def test_missing_or_garbage_attributes_mean_no_clamp(self, attrs, value):
        assert _clamp_to_entity_range(value, attrs) is value

    def test_each_bound_is_judged_on_its_own(self):
        assert _clamp_to_entity_range(0, {"min": 10, "max": "junk"}) == 10
        assert _clamp_to_entity_range(99999, {"min": 10, "max": "junk"}) == 99999
        assert _clamp_to_entity_range(99999, {"max": 9900}) == 9900
        assert _clamp_to_entity_range(-1, {"max": 9900}) == -1

    def test_numeric_string_attributes_are_accepted(self):
        assert _clamp_to_entity_range(0, {"min": "10", "max": "9900.0"}) == 10

    @pytest.mark.parametrize("value", [True, False, None, "auto", float("nan"), float("inf")])
    def test_non_numeric_value_is_never_touched(self, value):
        assert _clamp_to_entity_range(value, SUNGROW) is value

    def test_fractional_bound_stays_a_float(self):
        assert _clamp_to_entity_range(0, {"min": 0.5, "max": 5}) == 0.5


# --------------------------------------------------------------------------- #
# A real HAClient over a fake HTTP session
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status: int = 200, payload: Any = None, text: str = ""):
        self.status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    async def json(self) -> Any:
        return self._payload

    async def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                MagicMock(), (), status=self.status, message="Bad Request"
            )


class _Session:
    """States by entity id; a service call stores the written value as the state.
    ``reject`` maps entity id -> (status, body) for a refused write."""

    def __init__(self, states: dict[str, dict[str, Any]], reject=None):
        self.states = states
        self.reject = reject or {}
        self.gets: list[str] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **_kw: Any) -> _Resp:
        entity = url.rsplit("/", 1)[1]
        self.gets.append(entity)
        if entity not in self.states:
            return _Resp(404)
        return _Resp(200, {"entity_id": entity, **self.states[entity]})

    def post(self, url: str, json: dict[str, Any] | None = None, **_kw: Any) -> _Resp:
        json = json or {}
        entity = json.get("entity_id", "")
        self.posts.append((url.split("/api/services/")[1], dict(json)))
        if entity in self.reject:
            status, body = self.reject[entity]
            return _Resp(status, text=body)
        if "value" in json:
            self.states[entity]["state"] = str(json["value"])
        return _Resp(200)


def _client(states, reject=None) -> tuple[HAClient, _Session]:
    client = HAClient("http://ha.invalid", "token")
    session = _Session(states, reject)
    client._get_session = AsyncMock(return_value=session)  # type: ignore[method-assign]
    return client, session


def _profile() -> InverterProfile:
    profile = MagicMock(spec=InverterProfile)
    profile.entities = {
        "max_charge_power": EntityDefinition(
            default_entity=CHARGE,
            domain="number",
            category="battery",
            description="max charge",
            required=True,
        )
    }
    return profile


def _dispatcher(client: HAClient) -> ActionDispatcher:
    cfg = ExecutorConfig()
    return ActionDispatcher(client, cfg, profile=_profile())


DECISION = ControllerDecision(
    mode_intent="idle", charge_value=0.0, discharge_value=0.0,
    soc_target=10, water_temp=40, control_unit="W",
)


async def _run_action(disp: ActionDispatcher, value: Any):
    return await disp._execute_action(
        ModeAction(entity="max_charge_power", value=value), DECISION, "idle"
    )


def _writes(session: _Session) -> list[float]:
    return [p[1]["value"] for p in session.posts if "value" in p[1]]


class TestProfilePathClamp:
    @pytest.mark.asyncio
    async def test_below_min_is_written_at_min_and_verified(self, caplog):
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        disp = _dispatcher(client)
        with caplog.at_level(logging.WARNING, logger="executor.actions"):
            result = await _run_action(disp, 0)

        assert _writes(session) == [10.0]
        assert result.success is True
        assert result.new_value == 10
        assert result.verification_success is True
        clamp_logs = [r.getMessage() for r in caplog.records if "Clamped" in r.getMessage()]
        assert len(clamp_logs) == 1
        assert CHARGE in clamp_logs[0] and "0 → 10" in clamp_logs[0]

    @pytest.mark.asyncio
    async def test_above_max_is_written_at_max(self):
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        result = await _run_action(_dispatcher(client), 12000)
        assert _writes(session) == [9900.0]
        assert result.new_value == 9900

    @pytest.mark.asyncio
    async def test_9500_stays_9500(self, caplog):
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        with caplog.at_level(logging.DEBUG, logger="executor.actions"):
            result = await _run_action(_dispatcher(client), 9500)
        assert _writes(session) == [9500.0]
        assert result.new_value == 9500
        assert not [r for r in caplog.records if "Clamped" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_already_at_the_limit_is_a_skip_not_a_rewrite_every_tick(self, caplog):
        """Clamp runs BEFORE the skip check: 0 requested, entity already at min 10."""
        client, session = _client({CHARGE: {"state": "10", "attributes": dict(SUNGROW)}})
        disp = _dispatcher(client)
        with caplog.at_level(logging.DEBUG, logger="executor.actions"):
            for _ in range(3):
                result = await _run_action(disp, 0)
                assert result.skipped is True and result.success is True
        assert session.posts == []
        # WARNING once per entity per hour, DEBUG after that.
        clamps = [r for r in caplog.records if "Clamped" in r.getMessage()]
        assert [r.levelno for r in clamps] == [logging.WARNING, logging.DEBUG, logging.DEBUG]

    @pytest.mark.asyncio
    async def test_no_extra_ha_read_for_the_attributes(self):
        """One read before the write (state + attributes) and one read-back. The
        clamp itself never talks to HA."""
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        await _run_action(_dispatcher(client), 0)
        assert session.gets == [CHARGE, CHARGE]

    @pytest.mark.parametrize(
        "attributes",
        [
            None,
            {},
            {"min": "unknown", "max": None},
            {"min": 9900, "max": 10},
            {"min": "x", "max": float("nan")},
        ],
    )
    @pytest.mark.parametrize("value", [0, 9500, 12000])
    @pytest.mark.asyncio
    async def test_missing_or_garbage_attributes_write_the_value_untouched(
        self, attributes, value
    ):
        state: dict[str, Any] = {"state": "5000"}
        if attributes is not None:
            state["attributes"] = attributes
        client, session = _client({CHARGE: state})
        result = await _run_action(_dispatcher(client), value)
        assert _writes(session) == [float(value)]
        assert result.new_value == value

    @pytest.mark.asyncio
    async def test_stale_attributes_are_not_used(self):
        client, _ = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        disp = _dispatcher(client)
        await client.get_state(CHARGE)
        ts, attrs = client._number_attributes[CHARGE]
        client._number_attributes[CHARGE] = (ts - 3600.0, attrs)
        assert client.cached_attributes(CHARGE) is None
        assert await disp._clamp_to_entity(CHARGE, 0) == 0

    @pytest.mark.asyncio
    async def test_a_read_without_attributes_forgets_the_old_ones(self):
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        await client.get_state(CHARGE)
        assert client.cached_attributes(CHARGE) == SUNGROW
        session.states[CHARGE] = {"state": "5000"}
        await client.get_state(CHARGE)
        assert client.cached_attributes(CHARGE) is None

    @pytest.mark.asyncio
    async def test_non_number_entities_are_not_cached(self):
        client, _ = _client({"select.ems_mode": {"state": "x", "attributes": {"min": 1}}})
        await client.get_state("select.ems_mode")
        assert client.cached_attributes("select.ems_mode") is None

    @pytest.mark.parametrize("ha", [MagicMock(), AsyncMock(), MagicMock(spec=HAClient), object()])
    @pytest.mark.asyncio
    async def test_a_client_without_usable_attributes_means_no_clamp(self, ha):
        disp = ActionDispatcher(ha, ExecutorConfig())
        assert await disp._clamp_to_entity(CHARGE, 0) == 0
        assert await disp._clamp_to_entity(CHARGE, 99999) == 99999


class TestOtherWritePathsClamp:
    @pytest.mark.asyncio
    async def test_write_entity_clamps_generically(self):
        client, session = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        disp = _dispatcher(client)
        await client.get_state_value(CHARGE)
        assert await disp._write_entity(CHARGE, 0, "number") is True
        assert _writes(session) == [10.0]

    @pytest.mark.asyncio
    async def test_water_temp_helper_is_clamped_and_verified(self):
        entity = "input_number.vvb_target"
        client, session = _client(
            {entity: {"state": "50", "attributes": {"min": 20, "max": 65, "step": 1}}}
        )
        disp = ActionDispatcher(client, ExecutorConfig())
        result = await disp.set_water_temp(70, entity)
        assert _writes(session) == [65.0]
        assert result.success is True
        assert result.new_value == 65
        assert result.verification_success is True
        # Next tick: already at the clamped target -> skip, no rewrite.
        result = await disp.set_water_temp(70, entity)
        assert result.skipped is True
        assert len(session.posts) == 1

    @pytest.mark.asyncio
    async def test_water_temp_in_range_is_untouched(self):
        entity = "input_number.vvb_target"
        client, session = _client(
            {entity: {"state": "40", "attributes": {"min": 20, "max": 85, "step": 5}}}
        )
        result = await ActionDispatcher(client, ExecutorConfig()).set_water_temp(62, entity)
        assert _writes(session) == [62.0]
        assert result.new_value == 62

    @pytest.mark.asyncio
    async def test_max_export_power_is_clamped_and_verified(self):
        entity = "number.export_power_limit"
        client, session = _client(
            {entity: {"state": "400", "attributes": {"min": 0, "max": 10000, "step": 100}}}
        )
        cfg = ExecutorConfig()
        cfg.inverter.grid_max_export_power = entity
        result = await ActionDispatcher(client, cfg)._set_max_export_power(12000.0)
        assert result is not None
        assert _writes(session) == [10000.0]
        assert result.new_value == 10000
        assert result.verification_success is True

    @pytest.mark.asyncio
    async def test_number_sink_is_clamped(self):
        entity = "number.sink_power"
        client, session = _client(
            {entity: {"state": "0", "attributes": {"min": 0, "max": 2000}}}
        )
        sink = ExcessPVSinkSpec(
            id="sink", entity=entity, power_kw=1.0, enabled=True, on_value=3000, off_value=0
        )
        result = await ActionDispatcher(client, ExecutorConfig()).set_sink(sink, True)
        assert _writes(session) == [2000.0]
        assert result.success is True
        assert result.verification_success is True


class TestLoudClamp:
    """A max-side (or material) clamp used to be an HTTP 400 and a "failed 3 ticks"
    notice. It must stay audible: one error notification per episode, while the
    write itself succeeds."""

    @pytest.mark.asyncio
    async def test_max_side_clamp_notifies_once_and_still_succeeds(self):
        client, session = _client(
            {CHARGE: {"state": "100", "attributes": {"min": 10, "max": 5000}}}
        )
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        for _ in range(4):
            result = await _run_action(disp, 9500)
            assert result.success is True
        assert _writes(session) == [5000.0]
        assert disp.notify_error.await_count == 1
        msg = disp.notify_error.await_args.args[0]
        assert CHARGE in msg
        assert "9500" in msg and "5000" in msg
        assert "min 10" in msg and "max 5000" in msg

    @pytest.mark.asyncio
    async def test_a_request_back_in_range_ends_the_episode(self):
        client, _ = _client({CHARGE: {"state": "100", "attributes": {"min": 10, "max": 5000}}})
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        await _run_action(disp, 9500)
        await _run_action(disp, 9500)
        assert disp.notify_error.await_count == 1
        await _run_action(disp, 4000)
        await _run_action(disp, 9500)
        assert disp.notify_error.await_count == 2

    @pytest.mark.asyncio
    async def test_device_that_ignores_the_write_still_gets_one_notice(self):
        """Every tick writes (the state never moves), so every tick also runs the
        defensive clamp inside _write_entity — that must not re-arm the notice."""
        client, session = _client(
            {CHARGE: {"state": "100", "attributes": {"min": 10, "max": 5000}}}
        )
        real_post = session.post

        def _post(url, json=None, **kw):
            resp = real_post(url, json=json, **kw)
            session.states[CHARGE]["state"] = "100"
            return resp

        session.post = _post  # type: ignore[method-assign]
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        for _ in range(3):
            await _run_action(disp, 9500)
        assert _writes(session) == [5000.0, 5000.0, 5000.0]
        assert disp.notify_error.await_count == 1

    @pytest.mark.asyncio
    async def test_min_side_nudge_stays_quiet(self):
        client, _ = _client({CHARGE: {"state": "5000", "attributes": dict(SUNGROW)}})
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        for _ in range(3):
            await _run_action(disp, 0)
        disp.notify_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_material_min_side_clamp_is_loud(self):
        """OFF temperature 40 against a helper with min 45: the tank is told 45."""
        entity = "input_number.vvb_target"
        client, session = _client(
            {entity: {"state": "60", "attributes": {"min": 45, "max": 70, "step": 1}}}
        )
        disp = ActionDispatcher(client, ExecutorConfig())
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        for _ in range(3):
            await disp.set_water_temp(40, entity)
        assert _writes(session) == [45.0]
        assert disp.notify_error.await_count == 1
        assert "requested 40" in disp.notify_error.await_args.args[0]

    @pytest.mark.asyncio
    async def test_min_side_without_a_known_max_stays_quiet(self):
        client, _ = _client({CHARGE: {"state": "5000", "attributes": {"min": 10}}})
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        await _run_action(disp, 0)
        disp.notify_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_notification_does_not_fail_the_write(self):
        client, session = _client(
            {CHARGE: {"state": "100", "attributes": {"min": 10, "max": 5000}}}
        )
        disp = _dispatcher(client)
        disp.notify_error = AsyncMock(side_effect=RuntimeError("notify down"))  # type: ignore[method-assign]
        result = await _run_action(disp, 9500)
        assert result.success is True
        assert _writes(session) == [5000.0]


class TestWaterTempClampDetails:
    @pytest.mark.asyncio
    async def test_fractional_helper_bound_is_not_rewritten_every_tick(self):
        entity = "input_number.vvb_target"
        client, session = _client(
            {entity: {"state": "50", "attributes": {"min": 20, "max": 62.5, "step": 0.5}}}
        )
        disp = ActionDispatcher(client, ExecutorConfig())
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        first = await disp.set_water_temp(70, entity)
        assert first.new_value == 62.5 and first.verification_success is True
        for _ in range(3):
            assert (await disp.set_water_temp(70, entity)).skipped is True
        assert _writes(session) == [62.5]

    @pytest.mark.asyncio
    async def test_off_temperature_clamped_up_is_still_a_stop(self):
        entity = "input_number.vvb_target"
        client, _ = _client(
            {entity: {"state": "60", "attributes": {"min": 45, "max": 70, "step": 1}}}
        )
        disp = ActionDispatcher(client, ExecutorConfig())
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        disp._maybe_notify = AsyncMock()  # type: ignore[method-assign]
        off = disp.config.water_heater.temp_off
        assert off < 45
        await disp.set_water_temp(off, entity)
        assert disp._maybe_notify.await_args.args[0] == "water_heat_stop"

    @pytest.mark.asyncio
    async def test_heat_temperature_clamped_down_is_still_a_start(self):
        entity = "input_number.vvb_target"
        client, _ = _client(
            {entity: {"state": "30", "attributes": {"min": 20, "max": 38, "step": 1}}}
        )
        disp = ActionDispatcher(client, ExecutorConfig())
        disp.notify_error = AsyncMock()  # type: ignore[method-assign]
        disp._maybe_notify = AsyncMock()  # type: ignore[method-assign]
        await disp.set_water_temp(70, entity)
        assert disp._maybe_notify.await_args.args[0] == "water_heat_start"


# --------------------------------------------------------------------------- #
# HA's own error text reaches the result and the notification
# --------------------------------------------------------------------------- #
HA_BODY = '{"message":"Value 9950.0 for number.battery_max_charge_power is outside valid range 10 - 9900"}'


class TestHaErrorText:
    @pytest.mark.asyncio
    async def test_call_service_keeps_the_response_body(self):
        client, _ = _client({}, reject={CHARGE: (400, HA_BODY)})
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert exc.value.status_code == 400
        assert "outside valid range 10 - 9900" in str(exc.value)

    @pytest.mark.asyncio
    async def test_call_service_without_a_body_falls_back_to_the_http_reason(self):
        client, _ = _client({}, reject={CHARGE: (400, "")})
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert exc.value.response_body == "Bad Request"

    @pytest.mark.asyncio
    async def test_a_long_body_is_truncated(self):
        client, _ = _client({}, reject={CHARGE: (400, "x" * 5000)})
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert len(exc.value.response_body or "") == 300

    @pytest.mark.asyncio
    async def test_the_token_is_redacted_from_the_body(self):
        client, _ = _client({}, reject={CHARGE: (400, "boom Bearer token rest")})
        assert client.token == "token"
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert exc.value.response_body == "boom Bearer [redacted] rest"
        assert "token" not in str(exc.value).replace("[redacted]", "")

    @pytest.mark.asyncio
    async def test_a_stalled_error_body_is_abandoned_quickly(self, monkeypatch):
        import time as _time

        import executor.actions as actions_mod

        monkeypatch.setattr(actions_mod, "HA_ERROR_BODY_TIMEOUT_S", 0.05)
        client, session = _client({})

        class _Stalled(_Resp):
            async def text(self) -> str:
                await asyncio.sleep(30)
                return "never"

        session.post = lambda url, json=None, **kw: _Stalled(400)  # type: ignore[method-assign]
        started = _time.monotonic()
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert _time.monotonic() - started < 2.0
        assert exc.value.status_code == 400
        assert exc.value.response_body == "Bad Request"  # falls back to the reason

    @pytest.mark.asyncio
    async def test_retry_count_is_unchanged_and_the_body_survives_exhaustion(self, monkeypatch):
        """5xx is still retried max_retries times; the final error now also carries
        what HA answered."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client, session = _client({}, reject={CHARGE: (500, "modbus: device not responding")})
        with pytest.raises(HACallError) as exc:
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0}, max_retries=2)
        assert len(session.posts) == 3
        assert "failed after 3 attempts" in str(exc.value)
        assert "modbus: device not responding" in str(exc.value)
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_4xx_is_still_not_retried(self):
        client, session = _client({}, reject={CHARGE: (400, HA_BODY)})
        with pytest.raises(HACallError):
            await client.call_service("number", "set_value", CHARGE, {"value": 1.0})
        assert len(session.posts) == 1

    @pytest.mark.asyncio
    async def test_profile_write_failure_carries_the_reason(self):
        # No usable attributes -> no clamp -> HA refuses the write and says why.
        client, session = _client(
            {CHARGE: {"state": "5000", "attributes": {}}}, reject={CHARGE: (400, HA_BODY)}
        )
        result = await _run_action(_dispatcher(client), 9950)

        assert result.success is False
        assert result.skipped is False
        assert result.verification_success is None
        assert "outside valid range 10 - 9900" in (result.error_details or "")
        assert result.message.startswith("Failed to set max_charge_power: ")
        assert "outside valid range 10 - 9900" in result.message
        assert len(session.posts) == 1  # one attempt, as before

    @pytest.mark.asyncio
    async def test_write_entity_default_still_swallows_and_returns_false(self):
        client, _ = _client({}, reject={CHARGE: (400, HA_BODY)})
        disp = _dispatcher(client)
        assert await disp._write_entity(CHARGE, 5, "number") is False
        with pytest.raises(HACallError):
            await disp._write_entity(CHARGE, 5, "number", raise_errors=True)

    @pytest.mark.asyncio
    async def test_config_error_is_never_raised(self):
        """A non-numeric value for a number entity stays a quiet False even with
        raise_errors (it is a config error, not an HA answer)."""
        client, _ = _client({})
        disp = _dispatcher(client)
        assert await disp._write_entity(CHARGE, "on", "number", raise_errors=True) is False

    @pytest.mark.asyncio
    async def test_sink_write_failure_carries_the_reason(self):
        entity = "number.sink_power"
        client, _ = _client(
            {entity: {"state": "0"}}, reject={entity: (400, '{"message":"entity is read only"}')}
        )
        sink = ExcessPVSinkSpec(
            id="sink", entity=entity, power_kw=1.0, enabled=True, on_value=500, off_value=0
        )
        result = await ActionDispatcher(client, ExecutorConfig()).set_sink(sink, True)
        assert result.success is False
        assert "entity is read only" in (result.error_details or "")
        assert "entity is read only" in result.message

    @pytest.mark.asyncio
    async def test_the_three_fail_notification_says_why(self):
        from executor.engine import ACTION_FAILURE_NOTIFY_STREAK, ExecutorEngine

        client, _ = _client(
            {CHARGE: {"state": "5000", "attributes": {}}}, reject={CHARGE: (400, HA_BODY)}
        )
        disp = _dispatcher(client)
        eng = object.__new__(ExecutorEngine)
        eng.dispatcher = MagicMock()
        eng.dispatcher.notify_error = AsyncMock()
        for _ in range(ACTION_FAILURE_NOTIFY_STREAK):
            await eng._track_command_failures([await _run_action(disp, 9950)])

        assert eng.dispatcher.notify_error.await_count == 1
        msg = eng.dispatcher.notify_error.await_args.args[0]
        assert "max_charge_power" in msg
        assert "outside valid range 10 - 9900" in msg
