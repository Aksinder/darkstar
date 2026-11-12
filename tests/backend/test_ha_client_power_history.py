"""Unit tests for get_energy_from_power_history in ha_client."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytz

from backend.core.ha_client import get_energy_from_power_history


def _make_state(state_val: str, unit: str = "kW") -> dict:
    return {"state": state_val, "attributes": {"unit_of_measurement": unit}}


def _make_ha_config(url: str = "http://ha.local", token: str = "tok") -> dict:
    return {"url": url, "token": token}


START = datetime(2024, 1, 1, 10, 0, tzinfo=pytz.UTC)
END = START + timedelta(minutes=15)  # 0.25 hours


class TestGetEnergyFromPowerHistory:
    """Tests for get_energy_from_power_history."""

    @pytest.mark.asyncio
    async def test_normal_data_15_points_averaging_5kw(self):
        """15 points averaging 5 kW over 0.25h → 1.25 kWh."""
        states = [_make_state("5.0") for _ in range(15)]
        response_data = [states]

        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = response_data

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result == pytest.approx(1.25, abs=0.001)

    @pytest.mark.asyncio
    async def test_sparse_data_3_points(self):
        """3 points averaging 8 kW over 0.25h → 2.0 kWh."""
        states = [_make_state("8.0") for _ in range(3)]

        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = [states]

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result == pytest.approx(2.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        """Empty data list returns None."""
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = []

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result is None

    @pytest.mark.asyncio
    async def test_http_timeout_returns_none(self):
        """HTTP timeout exception returns None."""
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        """Connection error returns None."""
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result is None

    @pytest.mark.asyncio
    async def test_w_to_kw_normalization(self):
        """States in W are normalized to kW. 3000 W = 3 kW → 0.75 kWh over 0.25h."""
        states = [_make_state("3000.0", unit="W") for _ in range(5)]

        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = [states]

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result == pytest.approx(0.75, abs=0.001)

    @pytest.mark.asyncio
    async def test_mixed_unavailable_states_filtered(self):
        """unavailable/unknown states filtered; only numeric values averaged."""
        states = [
            _make_state("unavailable"),
            _make_state("4.0"),
            _make_state("unknown"),
            _make_state("6.0"),
            _make_state("unavailable"),
        ]

        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = [states]

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        # mean([4.0, 6.0]) = 5.0 kW x 0.25h = 1.25 kWh
        assert result == pytest.approx(1.25, abs=0.001)

    @pytest.mark.asyncio
    async def test_all_unavailable_returns_none(self):
        """All-unavailable state list returns None."""
        states = [_make_state("unavailable") for _ in range(5)]

        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = [states]

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_ha_config_returns_none(self):
        """Missing url/token returns None without HTTP call."""
        with patch(
            "backend.core.ha_client.secrets.load_home_assistant_config",
            return_value={},
        ):
            result = await get_energy_from_power_history("sensor.ev_power", START, END)

        assert result is None


class TestTimeWeightedIntegration:
    """The rewrite that made cycling loads meterable (2026-08-25).

    State-change history is event-driven: a 5-minute burst contributes dozens of
    samples, a long idle stretch one. The old unweighted mean was dominated by
    whichever mode CHANGED most, not by time — the house VVB's 3.5 kW bursts
    credited 0.000 kWh. Each sample must cover exactly the span to the next one.
    """

    def _ts(self, minutes: float) -> str:
        return (START + timedelta(minutes=minutes)).isoformat()

    def _state(self, val: str, minutes: float, unit: str = "kW") -> dict:
        return {
            "state": val,
            "last_changed": self._ts(minutes),
            "attributes": {"unit_of_measurement": unit},
        }

    def test_the_live_burst_case(self):
        from backend.core.ha_client import integrate_power_history_kwh

        # The 20:46-20:51 incident shape: idle, then 3.5 kW for 5 min, then idle.
        states = [
            self._state("0.0", 0.0),
            self._state("3.5", 1.0),   # burst starts 1 min in
            self._state("0.0", 6.0),   # ends 5 min later
        ]
        kwh = integrate_power_history_kwh(states, START, END)
        # 3.5 kW x 5/60 h = 0.2917 kWh. The old mean gave (0+3.5+0)/3 x 0.25 = 0.29
        # only by numerical coincidence of this tiny example; with the real dozens
        # of idle samples it collapsed toward 0.
        assert kwh == pytest.approx(3.5 * 5 / 60, abs=0.001)

    def test_many_idle_samples_do_not_dilute_the_burst(self):
        from backend.core.ha_client import integrate_power_history_kwh

        # 20 idle chatter samples around one 5-minute 3.5 kW burst. The unweighted
        # mean would give (3.5/21) * 0.25 = 0.042 kWh; time-weighting must not care
        # how often the idle side chatters.
        states = [self._state("0.0", m) for m in [0, 0.2, 0.4, 0.6, 0.8]]
        states += [self._state("3.5", 1.0)]
        states += [self._state("0.0", 6.0 + m) for m in [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]]
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh == pytest.approx(3.5 * 5 / 60, abs=0.001)

    def test_start_state_before_window_is_clamped(self):
        from backend.core.ha_client import integrate_power_history_kwh

        # HA includes the state valid AT window start with an earlier timestamp.
        states = [self._state("2.0", -30.0), self._state("0.0", 7.5)]
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh == pytest.approx(2.0 * 7.5 / 60, abs=0.001)

    def test_last_value_holds_across_an_unavailable_gap(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [
            self._state("3.5", 0.0),
            {"state": "unavailable", "last_changed": self._ts(5.0), "attributes": {}},
            self._state("0.0", 10.0),
        ]
        # The unavailable sample is skipped; 3.5 kW carries 0->10 min.
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh == pytest.approx(3.5 * 10 / 60, abs=0.001)

    def test_watts_are_normalized_with_timestamps(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [self._state("3500", 0.0, unit="W"), self._state("0", 7.5, unit="W")]
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh == pytest.approx(3.5 * 7.5 / 60, abs=0.001)

    def test_bare_values_fall_back_to_the_legacy_mean(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [{"state": "4.0", "attributes": {"unit_of_measurement": "kW"}}] * 3
        assert integrate_power_history_kwh(states, START, END) == pytest.approx(1.0)

    def test_no_numeric_samples_is_none(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [{"state": "unavailable", "attributes": {}}]
        assert integrate_power_history_kwh(states, START, END) is None


class TestPowerHistoryUnitResolution:
    """The unit must survive HA's minimal history response (attributes on the first
    state only) even when that first state is unavailable -- else W is read as kW."""

    def _ts(self, minutes: float) -> str:
        return (START + timedelta(minutes=minutes)).isoformat()

    def _state(self, val: str, minutes: float, unit: str | None = None) -> dict:
        state: dict = {"state": val, "last_changed": self._ts(minutes)}
        if unit is not None:
            state["attributes"] = {"unit_of_measurement": unit}
        return state

    def test_unavailable_first_w_series_is_not_read_as_kw(self):
        from backend.core.ha_client import integrate_power_history_kwh

        # Minimal response: only the first state has attributes, and it is unavailable.
        states = [
            self._state("unavailable", -5.0, unit="W"),
            self._state("1.0", 0.0),
            self._state("1.0", 7.5),
        ]
        kwh = integrate_power_history_kwh(states, START, END)
        # 1 W idle for 15 min = 0.00025 kWh (was 0.25 kWh of phantom heating).
        assert kwh == pytest.approx(0.00025, abs=1e-7)

    def test_unit_on_a_later_state_reaches_earlier_samples(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [self._state("2000", 0.0), self._state("2000", 7.5, unit="W")]
        assert integrate_power_history_kwh(states, START, END) == pytest.approx(0.5)

    def test_mixed_units_are_applied_per_sample(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [
            self._state("2000", 0.0, unit="W"),
            self._state("4", 5.0, unit="kW"),
            self._state("3", 10.0),  # no unit: inherits the nearest earlier one (kW)
        ]
        kwh = integrate_power_history_kwh(states, START, END)
        assert kwh == pytest.approx((2.0 + 4.0 + 3.0) * 5 / 60)

    @pytest.mark.parametrize(
        ("unit", "factor"),
        [("w", 1e-3), ("W", 1e-3), ("kw", 1.0), ("KW", 1.0), ("MW", 1e3), ("mW", 1e-6)],
    )
    def test_units_are_case_insensitive_except_milli_vs_mega(self, unit, factor):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [self._state("4", 0.0, unit=unit)]
        assert integrate_power_history_kwh(states, START, END) == pytest.approx(4 * factor / 4)

    def test_no_unit_anywhere_uses_the_fallback_unit(self):
        from backend.core.ha_client import integrate_power_history_kwh

        states = [self._state("2000", 0.0)]
        kwh = integrate_power_history_kwh(states, START, END, fallback_unit="W")
        assert kwh == pytest.approx(0.5)

    def test_no_unit_and_no_fallback_assumes_kw_and_warns_once(self, caplog):
        from backend.core import ha_client

        ha_client._power_unit_warned.discard("sensor.unitless")
        states = [self._state("2", 0.0), self._state("2", 5.0)]
        with caplog.at_level("WARNING", logger=ha_client.logger.name):
            for _ in range(2):
                kwh = ha_client.integrate_power_history_kwh(
                    states, START, END, entity_id="sensor.unitless"
                )
        assert kwh == pytest.approx(0.5)
        assert sum("sensor.unitless" in r.getMessage() for r in caplog.records) == 1

    def test_kw_series_is_unchanged(self):
        from backend.core.ha_client import integrate_power_history_kwh

        full = [self._state("3.5", 0.0, unit="kW"), self._state("0.0", 7.5, unit="kW")]
        minimal = [self._state("3.5", 0.0, unit="kW"), self._state("0.0", 7.5)]
        expected = 3.5 * 7.5 / 60
        assert integrate_power_history_kwh(full, START, END) == pytest.approx(expected)
        assert integrate_power_history_kwh(minimal, START, END) == pytest.approx(expected)

    def test_split_integral_honours_the_unit_of_an_unavailable_first_state(self):
        from backend.core.ha_client import integrate_power_history_split_kwh

        states = [
            self._state("unknown", -1.0, unit="W"),
            self._state("4000", 0.0),
            self._state("-2000", 7.5),
        ]
        pos, neg = integrate_power_history_split_kwh(states, START, END)
        assert pos == pytest.approx(0.5)
        assert neg == pytest.approx(0.25)

    @staticmethod
    def _client(history: list, current_state: object) -> AsyncMock:
        def _get(url: str, **_kwargs: object) -> MagicMock:
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = current_state if "/api/states/" in url else history
            return response

        client = AsyncMock()
        client.get = AsyncMock(side_effect=_get)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    @pytest.mark.asyncio
    async def test_fetch_falls_back_to_the_current_state_unit(self):
        history = [[self._state("2000", 0.0)]]
        current = {"state": "unavailable", "attributes": {"unit_of_measurement": "W"}}
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            client = self._client(history, current)
            mock_client_cls.return_value = client
            result = await get_energy_from_power_history("sensor.vvb_power", START, END)

        assert result == pytest.approx(0.5)
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_fetch_does_not_look_up_the_state_when_history_has_a_unit(self):
        history = [[self._state("unavailable", -5.0, unit="W"), self._state("2000", 0.0)]]
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            client = self._client(history, None)
            mock_client_cls.return_value = client
            result = await get_energy_from_power_history("sensor.vvb_power", START, END)

        assert result == pytest.approx(0.5)
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_batch_stamps_the_current_unit_into_a_unitless_series(self):
        from backend.core.ha_client import (
            get_power_history_batch,
            integrate_power_history_split_kwh,
        )

        series = [{**self._state("4000", 0.0), "entity_id": "sensor.grid_power"}]
        current = {"state": "4000", "attributes": {"unit_of_measurement": "W"}}
        with (
            patch(
                "backend.core.ha_client.secrets.load_home_assistant_config",
                return_value=_make_ha_config(),
            ),
            patch("backend.core.ha_client.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client_cls.return_value = self._client([series], current)
            result = await get_power_history_batch(["sensor.grid_power"], START, END)

        assert result is not None
        states = result["sensor.grid_power"]
        assert "attributes" not in series[0]  # the fetched state is not mutated
        assert integrate_power_history_split_kwh(states, START, END) == pytest.approx((1.0, 0.0))
