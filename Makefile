.PHONY: install lint linter test pre-commit-install pre-commit-run dvc-status dvc-checkout extract-urls extract-data

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

extract-urls:
	UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.util.extract_urls

extract-data:
	UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.util.extract_data
	DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache uv run python -m dvc add data/bronze
