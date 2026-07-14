"""Tests for GET /api/forecast/bias — the Phase 1b validation instrument.

After the DNI/DHI physics fix + clean PV-residual retrain, morning PV signed
bias should sit near 0 and aurora's window MAE should beat the naive baseline.
This endpoint is how that gets measured on the live box (HTTP-only access), so
the aggregation semantics are pinned here: bias = forecast - actual (positive =
over-forecast), hours taken from the LOCAL-offset ISO slot strings, per-day
window rows carry actual production so cloudy mornings are identifiable.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.routers.forecast import _aggregate_bias_rows


def _row(slot: str, version: str, pv_f, pv_a, load_f=0.2, load_a=0.25):
    return (slot, version, pv_f, load_f, pv_a, load_a)


class TestAggregateBiasRows:
    def test_signed_bias_positive_means_over_forecast(self):
        rows = [
            _row("2026-07-09T06:00:00+02:00", "aurora", pv_f=1.0, pv_a=0.5),
            _row("2026-07-09T07:00:00+02:00", "aurora", pv_f=1.2, pv_a=0.7),
        ]
        out = _aggregate_bias_rows(rows, 4, 11)
        win = out["aurora"]["window_pv"]
        assert win["n"] == 2
        assert win["bias"] == pytest.approx(0.5)  # consistently over-forecast
        assert win["mae"] == pytest.approx(0.5)

    def test_under_forecast_is_negative(self):
        rows = [_row("2026-07-09T08:00:00+02:00", "aurora", pv_f=0.4, pv_a=1.0)]
        out = _aggregate_bias_rows(rows, 4, 11)
        assert out["aurora"]["window_pv"]["bias"] == pytest.approx(-0.6)

    def test_window_uses_local_hour_from_iso_string(self):
        rows = [
            # 03:xx local — OUTSIDE the 4-11 window even though it might be
            # inside in UTC; the string's own offset hour is what counts.
            _row("2026-07-09T03:45:00+02:00", "aurora", pv_f=0.1, pv_a=0.0),
            _row("2026-07-09T04:00:00+02:00", "aurora", pv_f=0.5, pv_a=0.5),
            _row("2026-07-09T11:45:00+02:00", "aurora", pv_f=1.0, pv_a=1.0),
            _row("2026-07-09T12:00:00+02:00", "aurora", pv_f=2.0, pv_a=2.0),
        ]
        out = _aggregate_bias_rows(rows, 4, 11)
        assert out["aurora"]["window_pv"]["n"] == 2  # 04:00 and 11:45 only
        assert out["aurora"]["overall"]["pv"]["n"] == 4

    def test_per_day_window_rows_expose_cloudy_mornings(self):
        rows = [
            # Clear day: high actual, unbiased.
            _row("2026-07-09T08:00:00+02:00", "aurora", pv_f=2.0, pv_a=2.0),
            # Cloudy day: low actual, strongly over-forecast — the failure
            # mode validation step (d) must catch.
            _row("2026-07-10T08:00:00+02:00", "aurora", pv_f=1.5, pv_a=0.2),
        ]
        out = _aggregate_bias_rows(rows, 4, 11)
        by_day = {d["date"]: d for d in out["aurora"]["window_pv_by_day"]}
        assert by_day["2026-07-09"]["bias"] == pytest.approx(0.0)
        assert by_day["2026-07-09"]["actual_pv_kwh"] == pytest.approx(2.0)
        cloudy = by_day["2026-07-10"]
        assert cloudy["actual_pv_kwh"] == pytest.approx(0.2)  # low = cloudy
        assert cloudy["bias"] == pytest.approx(1.3)  # over-forecast, visible

    def test_versions_kept_separate_and_none_values_skipped(self):
        rows = [
            _row("2026-07-09T08:00:00+02:00", "aurora", pv_f=1.0, pv_a=0.8),
            _row("2026-07-09T08:00:00+02:00", "baseline_7_day_avg", pv_f=0.5, pv_a=0.8),
            # Missing actual — must not count toward pv stats.
            _row("2026-07-09T08:15:00+02:00", "aurora", pv_f=1.0, pv_a=None),
        ]
        out = _aggregate_bias_rows(rows, 4, 11)
        assert out["aurora"]["window_pv"]["n"] == 1
        assert out["baseline_7_day_avg"]["window_pv"]["n"] == 1
        assert out["baseline_7_day_avg"]["window_pv"]["bias"] == pytest.approx(-0.3)

    def test_by_hour_breakdown(self):
        rows = [
            _row("2026-07-09T06:00:00+02:00", "aurora", pv_f=0.5, pv_a=1.0),
            _row("2026-07-10T06:15:00+02:00", "aurora", pv_f=0.7, pv_a=1.2),
            _row("2026-07-09T13:00:00+02:00", "aurora", pv_f=3.0, pv_a=3.0),
        ]
        out = _aggregate_bias_rows(rows, 4, 11)
        hours = {h["hour"]: h for h in out["aurora"]["by_hour_pv"]}
        assert hours[6]["n"] == 2
        assert hours[6]["bias"] == pytest.approx(-0.5)  # morning under-forecast
        assert hours[13]["bias"] == pytest.approx(0.0)

    def test_empty_rows(self):
        assert _aggregate_bias_rows([], 4, 11) == {}


@pytest.fixture
def client():
    from backend.main import create_app

    app = create_app()
    fastapi_app = app.other_asgi_app if hasattr(app, "other_asgi_app") else app
    with (
        patch("backend.main.LearningStore", return_value=MagicMock(close=AsyncMock())),
        TestClient(fastapi_app) as client,
    ):
        yield client


def _engine_with_rows(rows):
    """Mock engine whose session returns the given joined rows."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    store = MagicMock()
    store.AsyncSession.return_value = session_ctx

    import pytz

    engine = MagicMock()
    engine.store = store
    engine.timezone = pytz.UTC
    return engine


def test_bias_endpoint_end_to_end(client):
    rows = [
        _row("2026-07-09T06:00:00+00:00", "aurora", pv_f=1.0, pv_a=0.5),
        _row("2026-07-09T06:00:00+00:00", "baseline_7_day_avg", pv_f=0.4, pv_a=0.5),
    ]
    with patch(
        "backend.api.routers.forecast.get_learning_engine",
        return_value=_engine_with_rows(rows),
    ):
        resp = client.get("/api/forecast/bias?days=7")
    assert resp.status_code == 200
    data = resp.json()

    assert data["bias_sign"].startswith("positive = over-forecast")
    assert data["versions"]["aurora"]["window_pv"]["bias"] == pytest.approx(0.5)
    comp = data["comparison"]
    assert comp["window_mae_pv_aurora"] == pytest.approx(0.5)
    assert comp["window_mae_pv_baseline"] == pytest.approx(0.1)
    assert comp["aurora_beats_baseline_window_pv"] is False  # baseline better here


def test_bias_endpoint_rejects_inverted_window(client):
    with patch(
        "backend.api.routers.forecast.get_learning_engine",
        return_value=_engine_with_rows([]),
    ):
        resp = client.get("/api/forecast/bias?hour_start=11&hour_end=4")
    assert resp.status_code == 400
