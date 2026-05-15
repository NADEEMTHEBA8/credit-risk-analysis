# ============================================================================
# Credit Risk Pipeline — Runtime Image
# ============================================================================
# Two-stage build:
#   - builder: installs Python deps (cached layer, only rebuilds on req change)
#   - runtime: copies installed packages + code into a slim final image
# ============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps for psycopg2 and lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Runtime-only system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy project code
COPY src/ ./src/
COPY sql/ ./sql/
COPY dags/ ./dags/
COPY tests/ ./tests/

# Default command runs the pipeline. Override with `docker run ... pytest tests/`
CMD ["python", "-m", "src.pipeline"]
