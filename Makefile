.PHONY: install lint linter test pre-commit-install pre-commit-run dvc-status dvc-checkout dvc-repro dvc-repro-bronze dvc-repro-silver prf-source2bronze prf-bronze2silver extract-urls extract-data

install:
	UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups

lint:
	UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
	UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests

linter:
	UV_CACHE_DIR=/tmp/uv-cache uv run ruff check --fix src tests
	UV_CACHE_DIR=/tmp/uv-cache uv run ruff format src tests

test:
	UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v

pre-commit-install:
	UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit install

pre-commit-run:
	UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --all-files

dvc-status:
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc status

dvc-checkout:
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc checkout

dvc-repro:
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc repro

dvc-repro-bronze:
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc repro prf_source2bronze

dvc-repro-silver:
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc repro prf_bronze2silver

prf-source2bronze:
	DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=data UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.etl.bronze.prf_source2bronze

prf-bronze2silver:
	DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=data UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.etl.silver.prf_bronze2silver
