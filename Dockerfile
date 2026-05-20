# Credit Risk Pipeline — runtime image

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries for psycopg2 and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        libgomp1 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/

CMD ["python", "-m", "src.main"]
