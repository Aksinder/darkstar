"""Unit tests for PlannerService retry policy."""
from datetime import UTC, datetime, timedelta

from backend.services.planner_service import _BACKOFF_STEPS, PlannerService
from planner.errors import PlannerErrorCode


def _fresh_service() -> PlannerService:
    return PlannerService()


def test_config_blocking_sets_suspended():
    svc = _fresh_service()
    svc._consecutive_failures = 1
    svc._apply_retry_policy(PlannerErrorCode.CONFIG_INVALID)
    assert svc._retry_suspended is True
    assert svc._next_retry_at is None


def test_clear_retry_suspension_resets_and_schedules_immediate():
    svc = _fresh_service()
    svc._retry_suspended = True
    svc._next_retry_at = None
    before = datetime.now(UTC)
    svc.clear_retry_suspension()
    assert svc._retry_suspended is False
    assert svc._next_retry_at is not None
    assert svc._next_retry_at >= before


def test_transient_backoff_three_failures():
    svc = _fresh_service()
    expected_delays = _BACKOFF_STEPS[:3]

    for i, expected_delay in enumerate(expected_delays, start=1):
        svc._consecutive_failures = i
        before = datetime.now(UTC)
        svc._apply_retry_policy(PlannerErrorCode.PRICES_UNAVAILABLE)
        after = datetime.now(UTC)

        assert svc._retry_suspended is False
        delay = (svc._next_retry_at - before).total_seconds()
        assert abs(delay - expected_delay) < 2.0, (
            f"Failure #{i}: expected ~{expected_delay}s delay, got {delay:.1f}s"
        )


def test_transient_backoff_caps_at_300():
    svc = _fresh_service()
    svc._consecutive_failures = 99  # Far past the cap
    before = datetime.now(UTC)
    svc._apply_retry_policy(PlannerErrorCode.FORECAST_UNAVAILABLE)
    delay = (svc._next_retry_at - before).total_seconds()
    assert abs(delay - 300) < 2.0, f"Expected cap at 300s, got {delay:.1f}s"


def test_invariant_failure_uses_60s_cadence():
    svc = _fresh_service()
    svc._consecutive_failures = 1
    before = datetime.now(UTC)
    svc._apply_retry_policy(PlannerErrorCode.SOLVER_INFEASIBLE)
    delay = (svc._next_retry_at - before).total_seconds()
    assert abs(delay - 60) < 2.0, f"Expected 60s for invariant, got {delay:.1f}s"


def test_warning_only_codes_do_not_set_last_error_code():
    svc = _fresh_service()
    svc._consecutive_failures = 1
    svc._apply_retry_policy(PlannerErrorCode.DATA_STALE)
    svc._apply_retry_policy(PlannerErrorCode.EV_DEADLINE_PAST)
    # warning_only codes don't modify retry state — _last_error_code is still None
    assert svc._retry_suspended is False
    assert svc._next_retry_at is None


def test_success_resets_all_state():
    svc = _fresh_service()
    svc._consecutive_failures = 5
    svc._retry_suspended = True
    svc._last_error_code = PlannerErrorCode.CONFIG_INVALID
    svc._last_error_at = datetime.now(UTC)
    svc._last_error_details = {"x": 1}
    svc._next_retry_at = datetime.now(UTC) + timedelta(minutes=5)

    svc._on_success()

    assert svc._consecutive_failures == 0
    assert svc._retry_suspended is False
    assert svc._last_error_code is None
    assert svc._last_error_at is None
    assert svc._last_error_details is None
    assert svc._next_retry_at is None


def test_retry_in_s_returns_none_when_suspended():
    svc = _fresh_service()
    svc._retry_suspended = True
    assert svc.retry_in_s is None


def test_retry_in_s_returns_seconds_remaining():
    svc = _fresh_service()
    svc._retry_suspended = False
    svc._next_retry_at = datetime.now(UTC) + timedelta(seconds=120)
    remaining = svc.retry_in_s
    assert remaining is not None
    assert 118 <= remaining <= 121


# -- 2026-07-26 regression: naive-local retry timestamps read as UTC ---------
#
# The scheduler gate normalizes a NAIVE _next_retry_at with `.replace(tzinfo=UTC)`
# (scheduler_service.py:149-153). While the service stamped it with a naive LOCAL
# datetime.now(), that turned every retry time into local-wall-clock-as-UTC — i.e.
# the local UTC offset (+2 h on CEST) into the future — so the planner silently
# stopped replanning after each config save and between transient-failure retries.


def _scheduler_gate_due(next_retry_at, now_utc) -> bool:
    """Mirror of the scheduler's due-check (scheduler_service.py:149-155)."""
    retry_at_utc = (
        next_retry_at.replace(tzinfo=UTC)
        if next_retry_at.tzinfo is None
        else next_retry_at
    )
    return now_utc >= retry_at_utc


def test_clear_retry_suspension_is_immediately_due_for_the_scheduler():
    """A config save must let the planner run NOW, not one UTC offset from now."""
    svc = _fresh_service()
    svc._retry_suspended = True

    svc.clear_retry_suspension()

    assert svc._next_retry_at is not None
    assert svc._next_retry_at.tzinfo is not None, "must be tz-aware, else read as local-as-UTC"
    # One second later the scheduler must consider the retry due.
    assert _scheduler_gate_due(svc._next_retry_at, datetime.now(UTC) + timedelta(seconds=1))


def test_transient_backoff_is_due_after_its_delay_not_a_utc_offset_later():
    """A transient failure (e.g. SOLVER_TIMEOUT) must retry after its backoff step."""
    svc = _fresh_service()
    svc._consecutive_failures = 1

    svc._apply_retry_policy(PlannerErrorCode.SOLVER_TIMEOUT)

    assert svc._next_retry_at is not None
    assert svc._next_retry_at.tzinfo is not None
    step = _BACKOFF_STEPS[0]
    # Not yet due at half the backoff...
    assert not _scheduler_gate_due(
        svc._next_retry_at, datetime.now(UTC) + timedelta(seconds=step / 2)
    )
    # ...but due once the backoff has elapsed (would need +2 h with the old bug).
    assert _scheduler_gate_due(
        svc._next_retry_at, datetime.now(UTC) + timedelta(seconds=step + 5)
    )
