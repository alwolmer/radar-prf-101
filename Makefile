DOCKER_COMPOSE ?= docker compose
SERVICE ?= spark-env
DOCKER_EXEC = $(DOCKER_COMPOSE) exec -T $(SERVICE)
UV_ENV = UV_CACHE_DIR=/tmp/uv-cache
DVC_ENV = DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache
LOCAL_DATALAKE_ENV = DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=/app/data

.PHONY: docker-build docker-up docker-ensure-running install lint linter test pre-commit-install pre-commit-run dvc-status dvc-checkout dvc-repro dvc-repro-bronze dvc-repro-silver bronze silver prf-source2bronze dnit-source2bronze prf-bronze2silver dnit-bronze2silver

docker-build:
	$(DOCKER_COMPOSE) build $(SERVICE)

docker-up:
	$(DOCKER_COMPOSE) up -d $(SERVICE)

docker-ensure-running:
	@running_container="$$( $(DOCKER_COMPOSE) ps --status running -q $(SERVICE) )"; \
	if [ -n "$$running_container" ]; then \
		echo "$(SERVICE) is already running"; \
	else \
		$(DOCKER_COMPOSE) up -d $(SERVICE); \
	fi

install: docker-up
	$(DOCKER_EXEC) env $(UV_ENV) uv sync --frozen --all-groups --no-install-project

lint: docker-ensure-running
	$(DOCKER_EXEC) ruff check src tests
	$(DOCKER_EXEC) ruff format --check src tests

linter: docker-ensure-running
	$(DOCKER_EXEC) ruff check --fix src tests
	$(DOCKER_EXEC) ruff format src tests

test: docker-ensure-running
	$(DOCKER_EXEC) pytest tests/ -v

pre-commit-install: docker-ensure-running
	$(DOCKER_EXEC) pre-commit install

pre-commit-run: docker-ensure-running
	$(DOCKER_EXEC) pre-commit run --all-files

dvc-status: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc status

dvc-checkout: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc checkout

dvc-repro: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro

dvc-repro-bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro prf_source2bronze dnit_source2bronze

dvc-repro-silver: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro prf_bronze2silver dnit_bronze2silver

bronze: prf-source2bronze dnit-source2bronze

silver: prf-bronze2silver dnit-bronze2silver

prf-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.prf_source2bronze

dnit-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.dnit_source2bronze

prf-bronze2silver: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.prf_bronze2silver

dnit-bronze2silver: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.dnit_bronze2silver
