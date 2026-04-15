FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /tmp/build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-groups --no-install-project \
    && rm -rf /root/.cache/uv

WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts", "--allowed-hosts", "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"]
