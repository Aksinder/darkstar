"""Price-forecast anchoring: inference lags from known prices, leak-free training lags.

Ported from upstream 0e1330e5 (+ the known-price lag part of 894d6596), adapted:
the fork derives the knowable horizon from the issue timestamp instead of storing
a ``known_prices_until`` column, so training and inference share one rule.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.learning.models import Base, PriceForecast, SlotObservation
from ml.price_features import (
    _get_price_lags,
    build_price_features_batch,
    price_knowable_until,
)
from ml.price_forecast import generate_price_forecasts
from ml.price_train import _add_price_lag_features, _build_training_dataset

TZ = pytz.timezone("Europe/Stockholm")


def local(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return TZ.localize(datetime(y, m, d, h, mi))


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "learning.db")
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield path, session
    session.close()
    engine.dispose()


def _observe(session, ts: datetime, price: float) -> None:
    session.add(
        SlotObservation(
            slot_start=ts.isoformat(),
            slot_end=(ts + timedelta(minutes=15)).isoformat(),
            export_price_sek_kwh=price,
        )
    )
    session.commit()


# --- price_knowable_until -------------------------------------------------------


def test_knowable_until_is_end_of_local_issue_day():
    assert price_knowable_until(local(2026, 4, 10, 6), TZ) == local(2026, 4, 11)
    assert price_knowable_until(local(2026, 4, 10, 23, 45), "Europe/Stockholm") == local(
        2026, 4, 11
    )
    # A UTC issue time just after local midnight belongs to the NEW local day.
    issue_utc = local(2026, 4, 11, 0, 30).astimezone(pytz.UTC)
    assert price_knowable_until(issue_utc, TZ) == local(2026, 4, 12)


def test_knowable_until_on_dst_day_is_local_midnight():
    # 2026-10-25 is 25 h long in Stockholm; the bound is still the local midnight.
    bound = price_knowable_until(local(2026, 10, 25, 6), TZ)
    assert bound == local(2026, 10, 26)
    assert bound.utcoffset() == timedelta(hours=1)


# --- inference lags ---------------------------------------------------------------


def test_lag_falls_back_to_published_price_when_not_yet_observed(db):
    """D+1 12:00 issued at D 06:00: its lag_1d (D 12:00) is unobserved but published."""
    _, session = db
    slot = local(2026, 4, 11, 12)
    published = {slot - timedelta(days=1): 0.77}

    lags = _get_price_lags(slot, session, published)
    assert lags["price_lag_1d"] == pytest.approx(0.77)

    # Without published prices this was always NaN in production (the midnight jump).
    assert pd.isna(_get_price_lags(slot, session)["price_lag_1d"])


def test_observation_wins_over_published_price(db):
    _, session = db
    slot = local(2026, 4, 11, 12)
    _observe(session, slot - timedelta(days=1), 0.42)
    _observe(session, slot - timedelta(days=7), 0.30)

    lags = _get_price_lags(slot, session, {slot - timedelta(days=1): 9.99})
    assert lags["price_lag_1d"] == pytest.approx(0.42)
    assert lags["price_lag_7d"] == pytest.approx(0.30)


def test_trailing_average_mixes_observed_and_published_slots(db):
    _, session = db
    slot = local(2026, 4, 11, 12)
    lag_1d = slot - timedelta(days=1)
    window = [lag_1d - timedelta(minutes=15 * i) for i in range(93)]  # 23 h + lag_1d
    # Older half observed at 1.0, newer half only published at 3.0.
    for ts in window[47:]:
        _observe(session, ts, 1.0)
    published = dict.fromkeys(window[:47], 3.0)

    lags = _get_price_lags(slot, session, published)
    expected = (46 * 1.0 + 47 * 3.0) / 93
    assert lags["price_lag_24h_avg"] == pytest.approx(expected)


def test_batch_lookup_with_pandas_timestamps_hits_published_keys(db):
    """build_price_features_batch iterates pandas Timestamps; the published map is
    keyed by plain datetimes -- they must still match."""
    _, session = db
    start = local(2026, 4, 11, 12)
    df = build_price_features_batch(
        start_time=start,
        end_time=start + timedelta(minutes=30),
        days_ahead=1,
        db_session=session,
        known_spot={
            start - timedelta(days=1): 0.5,
            start - timedelta(days=1) + timedelta(minutes=15): 0.6,
        },
    )
    assert list(df["price_lag_1d"])[:2] == pytest.approx([0.5, 0.6])


@pytest.mark.asyncio
async def test_generate_price_forecasts_passes_session_and_caps_published_prices(db):
    """Inference opens a lag session (it never did -> all lags NaN) and only uses
    published prices up to the same knowable bound training masks with."""
    db_path, _ = db
    bound = price_knowable_until(datetime.now(TZ), TZ)
    inside = bound - timedelta(minutes=15)
    beyond = bound + timedelta(minutes=15)  # e.g. a published D+1 price after 13:00
    seen: list[dict] = []

    def build_features(start_time, end_time, days_ahead, weather_df, **kwargs):
        seen.append(kwargs)
        return pd.DataFrame()

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("ml.price_forecast._load_known_spot", return_value={inside: 1.0, beyond: 2.0}),
        patch(
            "ml.price_forecast.get_regional_weather",
            return_value={"c": pd.DataFrame({"temp_c": [5.0]})},
        ),
        patch("ml.price_forecast.compute_regional_wind_index", return_value=pd.Series([1.0])),
        patch("ml.price_forecast.build_price_features_batch", side_effect=build_features),
    ):
        await generate_price_forecasts(
            {"timezone": "Europe/Stockholm", "nordpool": {"price_area": "SE3"}},
            db_path=db_path,
        )

    assert len(seen) == 7
    for kwargs in seen:
        assert kwargs["db_session"] is not None
        assert kwargs["known_spot"] == {inside: 1.0}


# --- training lags ----------------------------------------------------------------


def test_training_masks_lags_not_knowable_at_issue_time(db):
    """Same DB, two horizons: D+1 keeps lag_1d (D's price was published), D+5's
    lag_1d (D+4's actual price) is masked although the observation exists."""
    db_path, session = db
    issue = local(2026, 4, 10, 6)
    row_a = local(2026, 4, 11, 12)  # D+1
    row_b = local(2026, 4, 15, 12)  # D+5
    for ts in (
        row_a - timedelta(days=1),  # D 12:00 -- after issue time, but published
        row_a - timedelta(days=7),
        row_b - timedelta(days=1),  # D+4 -- not knowable
        row_b - timedelta(days=7),  # D-2 -- knowable
    ):
        _observe(session, ts, 0.5)

    df = pd.DataFrame(
        {
            "slot_start": pd.to_datetime([row_a, row_b], utc=True),
            "issue_timestamp": pd.to_datetime([issue, issue], utc=True),
            "days_ahead": [1, 5],
        }
    )
    out = _add_price_lag_features(df, db_path, timezone="Europe/Stockholm")

    assert out.iloc[0]["price_lag_1d"] == pytest.approx(0.5)
    assert out.iloc[0]["price_lag_7d"] == pytest.approx(0.5)
    assert out.iloc[0]["price_lag_24h_avg"] == pytest.approx(0.5)
    assert pd.isna(out.iloc[1]["price_lag_1d"])
    assert pd.isna(out.iloc[1]["price_lag_24h_avg"])
    assert out.iloc[1]["price_lag_7d"] == pytest.approx(0.5)


def test_training_dataset_uses_local_time_for_lags_and_calendar(db):
    """Timestamps come back from SQLite as strings and were parsed to UTC: the UTC
    isoformat never matched the local observation keys (every lag NaN) and the hour
    feature was the UTC hour while inference uses the local one."""
    db_path, session = db
    slot = local(2026, 4, 11, 12)
    _observe(session, slot, 0.9)  # the target
    _observe(session, slot - timedelta(days=1), 0.5)
    session.add(
        PriceForecast(
            slot_start=slot.isoformat(),
            issue_timestamp=local(2026, 4, 10, 6).isoformat(),
            days_ahead=1,
            wind_index=1.0,
        )
    )
    session.commit()

    df = _build_training_dataset(db_path, timezone="Europe/Stockholm")

    assert df is not None and len(df) == 1
    row = df.iloc[0]
    assert row["hour"] == 12  # local, not 10 (UTC)
    assert row["price_lag_1d"] == pytest.approx(0.5)
    assert row["export_price_sek_kwh"] == pytest.approx(0.9)


def test_training_and_inference_agree_on_lag_availability(db):
    """For the same issue time and slot, training yields a lag exactly when
    inference can have one."""
    db_path, session = db
    issue = local(2026, 4, 10, 6)
    bound = price_knowable_until(issue, TZ)
    empty_engine = create_engine("sqlite://")
    Base.metadata.create_all(empty_engine)
    empty_session = sessionmaker(bind=empty_engine)()
    for days_ahead in (1, 2):
        slot = local(2026, 4, 10, 12) + timedelta(days=days_ahead)
        source = slot - timedelta(days=1)
        _observe(session, source, 0.5)
        df = pd.DataFrame(
            {
                "slot_start": pd.to_datetime([slot], utc=True),
                "issue_timestamp": pd.to_datetime([issue], utc=True),
                "days_ahead": [days_ahead],
            }
        )
        trained = _add_price_lag_features(df, db_path, timezone="Europe/Stockholm")
        # Inference at `issue`: source is in the future (no observation), so only a
        # published price within the bound can supply it.
        published = {source: 0.5} if source < bound else {}
        inferred = _get_price_lags(slot, empty_session, published)
        assert pd.isna(trained.iloc[0]["price_lag_1d"]) == pd.isna(inferred["price_lag_1d"])


@pytest.mark.parametrize(
    ("day", "expected"),
    [((2026, 4, 11), 96), ((2026, 3, 29), 92), ((2026, 10, 25), 100)],
)
def test_batch_covers_exactly_one_local_day(day, expected):
    """[start, end) half-open: the next day's 00:00 slot is not part of this day."""
    start = local(*day)
    nxt = datetime(*day) + timedelta(days=1)
    end = local(nxt.year, nxt.month, nxt.day)
    df = build_price_features_batch(start_time=start, end_time=end, days_ahead=1)
    assert len(df) == expected
    assert pd.Timestamp(df.index[-1]) < end
