"""Ratchet: no NEW naive ``datetime.now()`` / ``datetime.utcnow()`` in executor/ or planner/.

The add-on container runs on LOCAL time (Europe/Stockholm), so a bare
``datetime.now()`` is local wall-clock while several consumers read naive values as
UTC. Every such pair has silently mis-scheduled by the UTC offset ("it just stopped",
no error): the dead VVB plan-stability anchor, the planner that stopped replanning
~2 h after each config save. ``utcnow()`` is the mirror trap (naive UTC) and is
deprecated besides.

Ruff's DTZ rules would do this, but the repo-wide ruff config is shared and the
existing occurrences are many; this guard instead counts them per file and fails
only when a file gains one. Fix a call (``datetime.now(UTC)`` / ``datetime.now(tz)``)
and lower its number here; never raise a number to get a change through — use an
aware datetime instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED_DIRS = ("executor", "planner")

# file (relative to repo root) -> naive now()/utcnow() calls present when the guard
# was introduced. Only ever LOWER these.
ALLOWED: dict[str, int] = {
    # meta.planned_at: naive local, display + executor journal only (the executor's
    # plan-age cutoff deliberately uses the file mtime instead).
    "planner/output/schedule.py": 1,
    # .timestamp() of a naive local now is still the correct epoch: harmless.
    "planner/pipeline.py": 1,
    # vacation updated_at: display string.
    "planner/vacation_state.py": 1,
}


def _naive_now_calls(source: str) -> list[int]:
    """Line numbers of datetime.now() without a tz, and of datetime.utcnow()."""
    tree = ast.parse(source)
    class_names: set[str] = set()  # names bound to the datetime.datetime class
    module_names: set[str] = set()  # names bound to the datetime module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            for alias in node.names:
                if alias.name == "datetime":
                    class_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "datetime":
                    module_names.add(alias.asname or alias.name)

    def is_datetime_class(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in class_names
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "datetime"
            and isinstance(expr.value, ast.Name)
            and expr.value.id in module_names
        )

    lines: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not is_datetime_class(node.func.value):
            continue
        if node.func.attr == "utcnow":
            lines.append(node.lineno)
        elif node.func.attr == "now":
            tz_args = list(node.args) + [k.value for k in node.keywords if k.arg == "tz"]
            if not tz_args or all(isinstance(a, ast.Constant) and a.value is None for a in tz_args):
                lines.append(node.lineno)
    return sorted(lines)


def _scan() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for d in GUARDED_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            hits = _naive_now_calls(path.read_text(encoding="utf-8"))
            if hits:
                found[path.relative_to(ROOT).as_posix()] = hits
    return found


def test_no_new_naive_datetime_now() -> None:
    regressions = {f: lines for f, lines in _scan().items() if len(lines) > ALLOWED.get(f, 0)}
    assert not regressions, (
        "New naive datetime.now()/utcnow() (container runs LOCAL time; use "
        "datetime.now(UTC) or datetime.now(tz)): "
        + "; ".join(
            f"{f} lines {lines} (allowed {ALLOWED.get(f, 0)})" for f, lines in regressions.items()
        )
    )


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("from datetime import datetime\ndatetime.now()", [2]),
        ("from datetime import datetime\ndatetime.now(None)", [2]),
        ("from datetime import datetime\ndatetime.now(tz=None)", [2]),
        ("from datetime import datetime\ndatetime.utcnow()", [2]),
        ("from datetime import datetime as dt\ndt.now()", [2]),
        ("import datetime\ndatetime.datetime.now()", [2]),
        ("import datetime as d\nd.datetime.utcnow()", [2]),
        ("from datetime import UTC, datetime\ndatetime.now(UTC)", []),
        ("from datetime import datetime\ndatetime.now(tz)", []),
        ("from datetime import datetime\ndatetime.now(tz=tz)", []),
        ("import pandas as pd\npd.Timestamp.now()", []),
        ("from datetime import date\ndate.today()", []),
    ],
)
def test_detector(src: str, expected: list[int]) -> None:
    assert _naive_now_calls(src) == expected
