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
