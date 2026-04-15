DOCKER_COMPOSE ?= docker compose
SERVICE ?= spark-env
MLFLOW_SERVICE ?= mlflow
DOCKER_EXEC = $(DOCKER_COMPOSE) exec -T $(SERVICE)
UV_ENV = UV_CACHE_DIR=/tmp/uv-cache
DVC_ENV = DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache
LOCAL_DATALAKE_ENV = DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=/app/data
ML_ENV = DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=/app/data MLFLOW_TRACKING_URI=http://mlflow:5000

.PHONY: docker-build docker-up docker-ensure-running mlflow-ensure-running install lint linter test pre-commit-install pre-commit-run dvc-status dvc-checkout dvc-repro dvc-repro-bronze dvc-repro-silver dvc-repro-gold bronze silver gold prf-source2bronze dnit-source2bronze ibge-municipios-source2bronze ibge-rgi-source2bronze prf-bronze2silver dnit-bronze2silver ibge-bronze2silver br101-rgi-weekly-panel-silver2gold mlflow-build mlflow-up activity-group-featurize activity-group-train

docker-build:
	$(DOCKER_COMPOSE) build $(SERVICE)

docker-up:
	$(DOCKER_COMPOSE) up -d $(SERVICE)

mlflow-build:
	$(DOCKER_COMPOSE) build mlflow

mlflow-up:
	$(DOCKER_COMPOSE) up -d mlflow

docker-ensure-running:
	@running_container="$$( $(DOCKER_COMPOSE) ps --status running -q $(SERVICE) )"; \
	if [ -n "$$running_container" ]; then \
		echo "$(SERVICE) is already running"; \
	else \
		$(DOCKER_COMPOSE) up -d $(SERVICE); \
	fi

mlflow-ensure-running:
	@running_container="$$( $(DOCKER_COMPOSE) ps --status running -q $(MLFLOW_SERVICE) )"; \
	if [ -n "$$running_container" ]; then \
		echo "$(MLFLOW_SERVICE) is already running"; \
	else \
		$(DOCKER_COMPOSE) up -d $(MLFLOW_SERVICE); \
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
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro prf_source2bronze dnit_source2bronze ibge_municipios_src2bronze ibge_rgi_src2bronze

dvc-repro-silver: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro prf_bronze2silver dnit_bronze2silver ibge_bronze2silver

dvc-repro-gold: docker-ensure-running
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro br101_rgi_weekly_panel_silver2gold

bronze: prf-source2bronze dnit-source2bronze ibge-municipios-source2bronze ibge-rgi-source2bronze

silver: prf-bronze2silver dnit-bronze2silver ibge-bronze2silver

gold: br101-rgi-weekly-panel-silver2gold

prf-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.prf_source2bronze

dnit-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.dnit_source2bronze

ibge-municipios-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.ibge_municipios_src2bronze

ibge-rgi-source2bronze: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.bronze.ibge_rgi_src2bronze

prf-bronze2silver: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.prf_bronze2silver

dnit-bronze2silver: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.dnit_bronze2silver

ibge-bronze2silver: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.ibge_bronze2silver

br101-rgi-weekly-panel-silver2gold: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.gold.br101_rgi_weekly_panel_silver2gold

activity-group-featurize: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.ml.activity_group_regression featurize

activity-group-train: docker-ensure-running mlflow-ensure-running
	$(DOCKER_EXEC) env $(ML_ENV) python -m src.ml.activity_group_regression train
