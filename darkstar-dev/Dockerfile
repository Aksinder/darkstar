# Darkstar Energy Manager - Home Assistant Add-on Dockerfile
# Built for multi-arch (amd64, arm64)
# Build context is repo root

FROM python:3.12-slim

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
    # Build dependencies for any packages that need compilation
    gcc \
    g++ \
    make \
    # Node.js 22.x from NodeSource (frontend engines require >=20.19; node:sqlite needs 22)
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pnpm and build frontend. Pin pnpm@9 to match lockfileVersion 9.0
# (pnpm 10 changes frozen-lockfile/approve-builds behaviour and breaks the build).
COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN npm install -g pnpm@9 && cd frontend && pnpm install --frozen-lockfile

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

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
