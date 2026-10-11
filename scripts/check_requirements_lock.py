#!/usr/bin/env python3
"""Fail when a requirements*.txt changed without its lock being regenerated.

Offline and stdlib-only on purpose: it compares the files in the checkout and
never asks PyPI anything, so a new upstream release cannot fail it. Checks:

  * every line of an input file is an exact ``name==version`` pin;
  * the lock's direct entries (the ones uv annotates ``via -r <input>``) are the
    same set of packages at the same versions - catches a bumped, added or
    removed pin;
  * every lock entry carries at least one ``--hash``;
  * a package present in two locks has the same version in both;
  * the ruff-only CI job installs the ruff version of requirements-dev.txt.

Completeness of the transitive set is enforced by the install itself: under
``pip --require-hashes`` a missing or stale entry is a hard error.

Fix a failure with ``scripts/lock_requirements.sh``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [
    ("requirements-build.txt", "requirements-build.lock"),
    ("requirements.txt", "requirements.lock"),
    ("requirements-dev.txt", "requirements-dev.lock"),
]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s;\\]+)")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_input(path: Path, errors: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        m = PIN.match(line)
        if not m:
            errors.append(f"{path.name}:{lineno}: not an exact 'name==version' pin: {raw.strip()}")
            continue
        pins[norm(m[1])] = m[3]
    return pins


def read_lock(path: Path, source: str) -> tuple[dict[str, str], set[str], set[str]]:
    """Return (all pins, names that are direct deps of `source`, names without a hash)."""
    pins: dict[str, str] = {}
    direct: set[str] = set()
    hashed: set[str] = set()
    current: str | None = None
    for raw in path.read_text().splitlines():
        m = PIN.match(raw)
        if m:
            current = norm(m[1])
            pins[current] = m[3]
            continue
        if current is None:
            continue
        stripped = raw.strip()
        if stripped.startswith("--hash=sha256:"):
            hashed.add(current)
        elif stripped.startswith("#") and re.search(rf"-r {re.escape(source)}\s*$", stripped):
            direct.add(current)
    return pins, direct, set(pins) - hashed


def main() -> int:
    errors: list[str] = []
    locks: dict[str, dict[str, str]] = {}

    for src, lock in PAIRS:
        src_path, lock_path = ROOT / src, ROOT / lock
        if not lock_path.exists():
            errors.append(f"{lock} is missing")
            continue
        wanted = read_input(src_path, errors)
        pins, direct, unhashed = read_lock(lock_path, src)
        locks[lock] = pins

        for name, version in sorted(wanted.items()):
            if name not in pins:
                errors.append(f"{src} pins {name}=={version} but {lock} has no entry for it")
            elif pins[name] != version:
                errors.append(f"{src} pins {name}=={version} but {lock} has {name}=={pins[name]}")
        for name in sorted(direct - set(wanted)):
            errors.append(f"{lock} lists {name} as a direct dependency but {src} no longer has it")
        for name in sorted(set(wanted) & set(pins) - direct):
            errors.append(f"{lock} does not record {name} as coming from {src}")
        for name in sorted(unhashed):
            errors.append(f"{lock}: {name} has no --hash")

    names = list(locks)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for name in sorted(set(locks[a]) & set(locks[b])):
                if locks[a][name] != locks[b][name]:
                    errors.append(
                        f"{name} is {locks[a][name]} in {a} but {locks[b][name]} in {b}"
                    )

    ruff = read_input(ROOT / "requirements-dev.txt", []).get("ruff")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for found in re.findall(r"pip install [^\n]*\bruff(?:==(\S+))?", ci):
        if found != ruff:
            errors.append(
                f"ci.yml installs ruff=={found or '<unpinned>'}, requirements-dev.txt has {ruff}"
            )

    if errors:
        print("Dependency locks are out of date:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("\nRegenerate with: scripts/lock_requirements.sh", file=sys.stderr)
        return 1
    total = sum(len(p) for p in locks.values())
    print(f"Locks match their inputs ({total} hashed entries in {len(locks)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
