"""D+1 forecast fallback picks tomorrow's rows by local date, latest issue, any horizon.

Regression: the fallback read only ``days_ahead=1`` rows (limit 96) and dropped slots
dated <= today. Between 00:00 and the 06:00 forecast run the newest days_ahead=1
rows are for TODAY, so the planner's horizon collapsed to today exactly when night
charging is decided.
"""

from datetime import datetime, timedelta

import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.learning.models import Base, PriceForecast
from ml import price_forecast as PF

TZ = pytz.timezone("Europe/Stockholm")
CONFIG = {"timezone": "Europe/Stockholm", "nordpool": {"price_area": "SE3"}}


def local(y, m, d, h=0, mi=0):
    return TZ.localize(datetime(y, m, d, h, mi))


def _frozen(now_local: datetime):
    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_local.astimezone(tz) if tz else now_local.replace(tzinfo=None)

    return _Now


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "learning.db")
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    yield path, sessionmaker(bind=engine)
    engine.dispose()


def _day_slots(day_start: datetime) -> list[datetime]:
    """Every quarter of the local day, walked in UTC (92/96/100 on DST days)."""
    nxt = (day_start + timedelta(hours=26)).date()
    end = TZ.localize(datetime.combine(nxt, datetime.min.time()))
    out, cur = [], day_start.astimezone(pytz.UTC)
    while cur < end:
        out.append(cur.astimezone(TZ))
        cur += timedelta(minutes=15)
    return out


def _seed(session_factory, day_start, issued: datetime, days_ahead: int, p50, slots=None):
    session = session_factory()
    for slot in slots or _day_slots(day_start):
        session.add(
            PriceForecast(
                slot_start=slot.isoformat(),
                issue_timestamp=issued.isoformat(),
                days_ahead=days_ahead,
                spot_p10=None if p50 is None else p50 - 0.1,
                spot_p50=p50,
                spot_p90=None if p50 is None else p50 + 0.1,
            )
        )
    session.commit()
    session.close()


async def _fallback(monkeypatch, db_path, now_local):
    monkeypatch.setattr(PF, "datetime", _frozen(now_local))
    return await PF.get_d1_price_forecast_fallback(CONFIG, db_path=db_path)


@pytest.mark.asyncio
async def test_night_gap_uses_yesterdays_d2_rows(db, monkeypatch):
    """03:00 on day T, before the daily run: only days_ahead=1 rows for T exist, plus
    yesterday's days_ahead=2 rows for T+1 -- those must fill tomorrow."""
    db_path, sf = db
    _seed(sf, local(2026, 11, 10), local(2026, 11, 9, 6), 1, 0.30)  # today: stale
    _seed(sf, local(2026, 11, 11), local(2026, 11, 9, 6), 2, 0.50)  # 06:00 D+2
    _seed(sf, local(2026, 11, 11), local(2026, 11, 9, 15), 2, 0.55)  # post-publication D+2

    rows = await _fallback(monkeypatch, db_path, local(2026, 11, 10, 3))

    assert rows is not None and len(rows) == 96
    assert {datetime.fromisoformat(r["slot_start"]).date() for r in rows} == {
        local(2026, 11, 11).date()
    }
    # Latest issue wins: the post-publication run.
    assert {r["spot_p50"] for r in rows} == {0.55}
    assert {r["days_ahead"] for r in rows} == {2}
    assert "import_p50" in rows[0] and "export_p50" in rows[0]


@pytest.mark.asyncio
async def test_latest_issue_wins_across_horizons(db, monkeypatch):
    """After the daily run its days_ahead=1 rows are newer than yesterday's D+2."""
    db_path, sf = db
    _seed(sf, local(2026, 11, 11), local(2026, 11, 9, 15), 2, 0.55)
    _seed(sf, local(2026, 11, 11), local(2026, 11, 10, 6), 1, 0.70)

    rows = await _fallback(monkeypatch, db_path, local(2026, 11, 10, 9))

    assert len(rows) == 96
    assert {r["spot_p50"] for r in rows} == {0.70}


@pytest.mark.asyncio
async def test_weather_only_rows_never_shadow_a_valid_forecast(db, monkeypatch):
    db_path, sf = db
    _seed(sf, local(2026, 11, 11), local(2026, 11, 9, 15), 2, 0.55)
    _seed(sf, local(2026, 11, 11), local(2026, 11, 10, 6), 1, None)  # weather-only, newer

    rows = await _fallback(monkeypatch, db_path, local(2026, 11, 10, 9))
    assert len(rows) == 96 and {r["spot_p50"] for r in rows} == {0.55}


@pytest.mark.asyncio
async def test_only_weather_rows_or_only_stale_rows_give_none(db, monkeypatch):
    db_path, sf = db
    _seed(sf, local(2026, 11, 11), local(2026, 11, 10, 6), 1, None)
    _seed(sf, local(2026, 11, 10), local(2026, 11, 9, 6), 1, 0.4)  # today
    _seed(sf, local(2026, 11, 9), local(2026, 11, 8, 6), 1, 0.4)  # yesterday
    assert await _fallback(monkeypatch, db_path, local(2026, 11, 10, 9)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("now", "tomorrow", "expected"),
    [
        (local(2026, 10, 24, 2), local(2026, 10, 25), 100),  # 25-hour day
        (local(2026, 3, 28, 2), local(2026, 3, 29), 92),  # 23-hour day
    ],
)
async def test_dst_tomorrow_selected_by_local_date_not_row_limit(
    db, monkeypatch, now, tomorrow, expected
):
    db_path, sf = db
    _seed(sf, tomorrow, now - timedelta(hours=11), 2, 0.5)
    # The following day's rows (same issue) must not leak in.
    _seed(sf, tomorrow + timedelta(hours=26), now - timedelta(hours=11), 3, 0.9)

    rows = await _fallback(monkeypatch, db_path, now)

    assert rows is not None and len(rows) == expected
    assert len({r["slot_start"] for r in rows}) == expected
    assert {r["spot_p50"] for r in rows} == {0.5}


@pytest.mark.asyncio
async def test_issue_times_compared_as_instants_across_offsets(db, monkeypatch):
    """'...T02:30+01:00' sorts BEFORE '...T02:45+02:00' as a string but is LATER."""
    db_path, sf = db
    slot = [local(2026, 10, 26, 12)]
    later = TZ.localize(datetime(2026, 10, 25, 2, 30), is_dst=False)  # 01:30Z
    earlier = TZ.localize(datetime(2026, 10, 25, 2, 45), is_dst=True)  # 00:45Z
    _seed(sf, None, later, 2, 0.8, slots=slot)
    _seed(sf, None, earlier, 2, 0.1, slots=slot)

    rows = await _fallback(monkeypatch, db_path, local(2026, 10, 25, 10))
    assert [r["spot_p50"] for r in rows] == [0.8]
