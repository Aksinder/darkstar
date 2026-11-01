"""Post-publication price forecast (D+2..D+7 with published D+1 lags), no schema change.

The knowable bound is derived from issue_timestamp alone: a row issued at/after
POST_PUBLICATION_HOUR local time (since POST_PUBLICATION_RULE_SINCE) treats D+1 as
published. Inference and training share that rule; inference that cannot supply
D+1 at such an issue time does not produce the affected (D+2) rows.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.learning.models import Base, PriceForecast, SlotObservation
from backend.services.scheduler_service import SchedulerService
from ml import price_forecast as PF
from ml.price_features import (
    POST_PUBLICATION_HOUR,
    _get_price_lags,
    is_post_publication_issue,
    price_knowable_until,
)
from ml.price_train import _add_price_lag_features, _build_training_dataset

TZ = pytz.timezone("Europe/Stockholm")


def local(y, m, d, h=0, mi=0):
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


def _observe(session, ts, price):
    session.add(
        SlotObservation(
            slot_start=ts.isoformat(),
            slot_end=(ts + timedelta(minutes=15)).isoformat(),
            export_price_sek_kwh=price,
        )
    )
    session.commit()


def _quarters(start, end):
    out, cur = [], start.astimezone(pytz.UTC)
    while cur < end:
        out.append(cur.astimezone(TZ))
        cur += timedelta(minutes=15)
    return out


# --- the shared rule ------------------------------------------------------------------


def test_bound_extends_to_end_of_d1_from_post_publication_hour():
    assert price_knowable_until(local(2026, 11, 10, 6), TZ) == local(2026, 11, 11)
    assert price_knowable_until(local(2026, 11, 10, POST_PUBLICATION_HOUR - 1, 59), TZ) == local(
        2026, 11, 11
    )
    assert price_knowable_until(local(2026, 11, 10, POST_PUBLICATION_HOUR), TZ) == local(
        2026, 11, 12
    )
    assert price_knowable_until(local(2026, 11, 10, 23, 45), TZ) == local(2026, 11, 12)
    # Same instant given in UTC.
    utc = local(2026, 11, 10, 15).astimezone(pytz.UTC)
    assert price_knowable_until(utc, "Europe/Stockholm") == local(2026, 11, 12)


def test_rows_issued_before_the_rule_keep_the_old_bound():
    """Old manual runs after 14:00 were inferred with the end-of-day cap."""
    assert not is_post_publication_issue(local(2026, 10, 20, 15), TZ)
    assert price_knowable_until(local(2026, 10, 20, 15), TZ) == local(2026, 10, 21)


def test_bound_on_dst_eve_ends_at_local_midnight_after_the_25h_day():
    bound = price_knowable_until(local(2027, 10, 30, 15), TZ)  # D+1 = 2027-10-31 (25 h)
    assert bound == local(2027, 11, 1)
    assert bound - local(2027, 10, 31) == timedelta(hours=25)


def test_naive_issue_timestamp_is_local_wall_time_not_container_time():
    assert price_knowable_until(datetime(2026, 11, 10, 15), TZ) == local(2026, 11, 12)


# --- train/serve consistency --------------------------------------------------------------


@pytest.mark.parametrize("issue_hour", [6, 13, 15])
def test_training_and_inference_agree_for_d2_lags(db, issue_hour):
    """A D+2 row's lag_1d sits in D+1: known (both sides) only for a post-publication
    issue. Its 24h window straddles D/D+1, so before publication both sides average
    only the knowable part (prices vary per hour to prove the same subset is used)."""
    db_path, session = db
    issue = local(2026, 11, 10, issue_hour)
    slot = local(2026, 11, 12, 12)

    def price(ts):
        return 0.1 + ts.hour / 100

    for ts in _quarters(local(2026, 11, 10), local(2026, 11, 12)):
        _observe(session, ts, price(ts))  # observed after the fact (training's view)

    df = pd.DataFrame(
        {
            "slot_start": pd.to_datetime([slot], utc=True),
            "issue_timestamp": pd.to_datetime([issue], utc=True),
            "days_ahead": [2],
        }
    )
    trained = _add_price_lag_features(df, db_path, timezone="Europe/Stockholm").iloc[0]

    # Inference at `issue`: nothing in the window observed yet, published up to the bound.
    bound = price_knowable_until(issue, TZ)
    published = {
        ts: price(ts) for ts in _quarters(local(2026, 11, 10), local(2026, 11, 12)) if ts < bound
    }
    empty = create_engine("sqlite://")
    Base.metadata.create_all(empty)
    inferred = _get_price_lags(slot, sessionmaker(bind=empty)(), published)

    post = issue_hour >= POST_PUBLICATION_HOUR
    assert pd.isna(trained["price_lag_1d"]) == pd.isna(inferred["price_lag_1d"]) == (not post)
    if post:
        assert trained["price_lag_1d"] == pytest.approx(inferred["price_lag_1d"])
    assert trained["price_lag_24h_avg"] == pytest.approx(inferred["price_lag_24h_avg"])


def test_training_dataset_reads_bound_from_stored_issue_timestamp(db):
    db_path, session = db
    slot = local(2026, 11, 12, 12)
    _observe(session, slot, 0.9)
    _observe(session, slot - timedelta(days=1), 0.5)
    for issued, slot_offset in ((local(2026, 11, 10, 15), 0), (local(2026, 11, 10, 6), 15)):
        session.add(
            PriceForecast(
                slot_start=(slot + timedelta(minutes=slot_offset)).isoformat(),
                issue_timestamp=issued.isoformat(),
                days_ahead=2,
            )
        )
    _observe(session, slot + timedelta(minutes=15), 0.9)
    _observe(session, slot - timedelta(days=1) + timedelta(minutes=15), 0.5)
    session.commit()

    df = _build_training_dataset(db_path, timezone="Europe/Stockholm").sort_values("slot_start")
    assert df.iloc[0]["price_lag_1d"] == pytest.approx(0.5)  # issued 15:00 -> D+1 known
    assert pd.isna(df.iloc[1]["price_lag_1d"])  # issued 06:00 -> not knowable


# --- generator ------------------------------------------------------------------------------


class _At(datetime):
    at: datetime

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz) if tz else cls.at.replace(tzinfo=None)


async def _generate(db_path, now, known, days_ahead_range=None):
    seen: list = []

    def build(start_time, end_time, days_ahead, weather_df, **kwargs):
        seen.append((days_ahead, kwargs["known_spot"]))
        return pd.DataFrame()

    clock = type("_Clock", (_At,), {"at": now})
    with (
        patch.object(PF, "datetime", clock),
        patch("pathlib.Path.exists", return_value=False),
        patch("ml.price_forecast._load_known_spot", return_value=known),
        patch(
            "ml.price_forecast.get_regional_weather",
            return_value={"c": pd.DataFrame({"temp_c": [5.0]})},
        ),
        patch("ml.price_forecast.compute_regional_wind_index", return_value=pd.Series([1.0])),
        patch("ml.price_forecast.build_price_features_batch", side_effect=build),
    ):
        await PF.generate_price_forecasts(
            {"timezone": "Europe/Stockholm", "nordpool": {"price_area": "SE3"}},
            db_path=db_path,
            days_ahead_range=days_ahead_range,
        )
    return seen


@pytest.mark.asyncio
async def test_post_publication_run_uses_published_d1_as_lags(db):
    db_path, _ = db
    d1 = _quarters(local(2026, 11, 11), local(2026, 11, 12))
    beyond = local(2026, 11, 12, 0)  # never knowable
    known = {**dict.fromkeys(d1, 0.4), beyond: 9.9}

    seen = await _generate(db_path, local(2026, 11, 10, 15), known, range(2, 8))

    assert [h for h, _ in seen] == [2, 3, 4, 5, 6, 7]
    assert all(len(k) == 96 and beyond not in k for _, k in seen)


@pytest.mark.asyncio
async def test_late_publication_drops_d2_rows_instead_of_inconsistent_lags(db):
    db_path, _ = db
    seen = await _generate(db_path, local(2026, 11, 10, 15), {})
    assert [h for h, _ in seen] == [1, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_post_publication_run_on_dst_eve_needs_all_100_quarters(db):
    db_path, _ = db
    d1 = _quarters(local(2027, 10, 31), local(2027, 11, 1))
    assert len(d1) == 100
    full = await _generate(db_path, local(2027, 10, 30, 15), dict.fromkeys(d1, 0.4), range(2, 4))
    assert [h for h, _ in full] == [2, 3]
    partial = dict.fromkeys(d1[:96], 0.4)  # a 96-slot assumption would call this complete
    short = await _generate(db_path, local(2027, 10, 30, 15), partial, range(2, 4))
    assert [h for h, _ in short] == [3]


# --- scheduler ------------------------------------------------------------------------------


@pytest.fixture
def sched(db):
    db_path, _ = db
    svc = SchedulerService()
    engine = MagicMock()
    engine.db_path = db_path
    engine.config = {"timezone": "Europe/Stockholm"}
    with (
        patch.object(svc, "_load_global_config", return_value={"timezone": "Europe/Stockholm"}),
        patch("backend.learning.get_learning_engine", return_value=engine),
    ):
        yield svc, db_path


def _published_tomorrow(now_local):
    start = TZ.localize(datetime.combine(now_local.date() + timedelta(days=1), datetime.min.time()))
    return dict.fromkeys(_quarters(start, start + timedelta(hours=26)), 0.4)


@pytest.mark.asyncio
async def test_scheduler_runs_once_after_publication(sched):
    svc, _ = sched
    now = local(2026, 11, 10, 15)
    gen = AsyncMock(return_value=[])
    with (
        patch(
            "backend.core.prices.get_known_spot_by_slot",
            AsyncMock(return_value=_published_tomorrow(now)),
        ),
        patch("ml.price_forecast.generate_price_forecasts", gen),
    ):
        await svc._check_post_publication_forecast({}, now=now.astimezone(UTC))
        await svc._check_post_publication_forecast(
            {}, now=(now + timedelta(hours=1)).astimezone(UTC)
        )
    gen.assert_awaited_once()
    assert gen.await_args.kwargs["days_ahead_range"] == range(2, 8)
    assert svc._status.current_task == "idle"


@pytest.mark.asyncio
async def test_scheduler_waits_before_hour_and_while_unpublished(sched):
    svc, _ = sched
    gen = AsyncMock(return_value=[])
    known = AsyncMock(return_value={})
    with (
        patch("backend.core.prices.get_known_spot_by_slot", known),
        patch("ml.price_forecast.generate_price_forecasts", gen),
    ):
        await svc._check_post_publication_forecast({}, now=local(2026, 11, 10, 13).astimezone(UTC))
        known.assert_not_awaited()
        await svc._check_post_publication_forecast({}, now=local(2026, 11, 10, 14).astimezone(UTC))
        known.assert_awaited_once()
        # Within the poll interval: no new check.
        await svc._check_post_publication_forecast(
            {}, now=local(2026, 11, 10, 14, 5).astimezone(UTC)
        )
        assert known.await_count == 1
    gen.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_disabled_by_flag(sched):
    svc, _ = sched
    gen = AsyncMock(return_value=[])
    with patch("ml.price_forecast.generate_price_forecasts", gen):
        await svc._check_post_publication_forecast(
            {"post_publication_run": False}, now=local(2026, 11, 10, 15).astimezone(UTC)
        )
    gen.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_restart_does_not_repeat_todays_run(sched, db):
    svc, _ = sched
    _, session = db
    session.add(
        PriceForecast(
            slot_start=local(2026, 11, 12, 0).isoformat(),
            issue_timestamp=local(2026, 11, 10, 14, 20).isoformat(),
            days_ahead=2,
        )
    )
    session.commit()
    gen = AsyncMock(return_value=[])
    with (
        patch(
            "backend.core.prices.get_known_spot_by_slot",
            AsyncMock(return_value=_published_tomorrow(local(2026, 11, 10))),
        ),
        patch("ml.price_forecast.generate_price_forecasts", gen),
    ):
        await svc._check_post_publication_forecast({}, now=local(2026, 11, 10, 16).astimezone(UTC))
    gen.assert_not_awaited()
