# Darkstar Energy Manager - Home Assistant Add-on Dockerfile
# Built for multi-arch (amd64, arm64)
# Build context is repo root

# Base image pinned by digest (multi-arch index: amd64 + arm64), so a rebuild of
# the tag cannot change Python, pip or Debian under an unattended build.
# This is python 3.12.15 / Debian trixie / pip 25.0.1, resolved 2026-10-11.
# The pin also freezes base-image security updates: refresh it deliberately with
#   docker buildx imagetools inspect python:3.12-slim   (take the top-level Digest)
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Utilities + TLS for the NodeSource repo
    curl \
    jq \
    bash \
    ca-certificates \
    gnupg \
    # GLPK solver for PuLP (Kepler optimizer)
    glpk-utils \
    libglpk-dev \
    # OpenMP runtime for highspy (HiGHS primary solver); its manylinux wheel
    # links libgomp.so.1, absent from python:3.12-slim -> import fails -> CBC fallback
    libgomp1 \
    # Build dependencies for any packages that need compilation
    gcc \
    g++ \
    make \
    # Node.js 22.x from NodeSource (frontend engines require >=20.19; node:sqlite needs 22)
    # NOT pinned: the setup script and the nodejs 22.x package it selects float, as
    # do the apt packages above. Node only builds the frontend bundle, whose npm
    # inputs are locked by pnpm-lock.yaml; pinning Node itself means a different
    # install method (official tarball + checksum), which is a separate change.
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pnpm and build frontend. Pin pnpm@9 to match lockfileVersion 9.0
# (pnpm 10 changes frozen-lockfile/approve-builds behaviour and breaks the build).
# pnpm@9 is a major-version range, not an exact pin: it floats within 9.x.
COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN npm install -g pnpm@9 && cd frontend && pnpm install --frozen-lockfile

# Install Python dependencies from the hashed locks: every wheel/sdist is checked
# against requirements.lock, and anything not listed there is an error.
# Two steps because suncalc ships no wheel: the first installs the hashed build
# backend (setuptools), the second builds suncalc against it (--no-build-isolation)
# instead of letting pip download an unpinned setuptools.
# To change a dependency see the header of requirements.txt.
COPY requirements-build.lock requirements.lock ./
RUN pip3 install --no-cache-dir --require-hashes -r requirements-build.lock \
    && pip3 install --no-cache-dir --require-hashes --no-build-isolation -r requirements.lock

# Copy application code (except frontend)
COPY backend/ ./backend/
COPY planner/ ./planner/
COPY executor/ ./executor/
COPY bin/ ./bin/
COPY profiles/ ./profiles/
COPY utils/ ./utils/
COPY ml/*.py ./ml/
# Models are generated at runtime (Clean Slate strategy)
RUN mkdir -p ml/models
COPY ml/models/defaults/ ./ml/models/defaults/

# Copy frontend source and build (AFTER copying backend to avoid overwriting static assets)
COPY frontend/ ./frontend/
RUN cd frontend && pnpm build

# Copy configuration templates
COPY config.default.yaml ./
COPY secrets.example.yaml ./
COPY VERSION ./
COPY ml/regions.json ./ml/regions.json

# Copy Alembic migration files for automatic database schema updates
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Copy HA Add-on entrypoint (note: path from repo root context)
COPY darkstar-dev/run.sh /run.sh
RUN chmod +x /run.sh

# Environment

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["/run.sh"]
