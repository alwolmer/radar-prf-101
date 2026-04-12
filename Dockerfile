FROM maven:3.9-eclipse-temurin-17 AS spark-jars

WORKDIR /tmp/spark-jars
COPY .env.docker ./
COPY scripts/fetch_spark_jars.sh ./scripts/fetch_spark_jars.sh
RUN bash ./scripts/fetch_spark_jars.sh .env.docker /opt/spark-jars

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/src" \
    SPARK_JARS_DIR=/opt/spark-jars \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

# Install only the runtime packages needed inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
        git \
        procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=spark-jars /opt/spark-jars /opt/spark-jars

# Keep the virtual environment outside /app so the bind mount does not hide it.
WORKDIR /tmp/build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-groups --no-install-project \
    && rm -rf /root/.cache/uv

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
