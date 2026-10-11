#!/usr/bin/env bash
# Regenerate the hashed lock files from the requirements*.txt inputs.
#
#   scripts/lock_requirements.sh                        # after editing a pin
#   scripts/lock_requirements.sh --upgrade-package anyio   # move one transitive dep
#   scripts/lock_requirements.sh --upgrade              # move every transitive dep
#
# Without --upgrade*, uv keeps every version already in the existing lock and
# only changes what the edited pin forces, so a routine bump gives a small diff.
# The locks are "universal": one file valid for CPython 3.12 on linux/amd64,
# linux/aarch64 (the images) and macOS (dev), with the hashes of every wheel and
# sdist of each pinned version.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(--universal --generate-hashes --python-version 3.12 --quiet
        --custom-compile-command "scripts/lock_requirements.sh")

# Build backend for the one sdist-only dependency (suncalc). Installed first, so
# that build runs against a hashed setuptools instead of whatever is newest.
uv pip compile requirements-build.txt "${COMMON[@]}" "$@" -o requirements-build.lock
uv pip compile requirements.txt "${COMMON[@]}" "$@" -o requirements.lock
# Dev tools are layered on the runtime lock so shared packages agree.
uv pip compile requirements-dev.txt -c requirements.lock "${COMMON[@]}" "$@" -o requirements-dev.lock

python3 scripts/check_requirements_lock.py
