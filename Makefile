DOCKER_COMPOSE ?= docker compose
SERVICE ?= spark-env
MLFLOW_SERVICE ?= mlflow
VIZ_SERVICE ?= viz
DOCKER_EXEC = $(DOCKER_COMPOSE) exec -T $(SERVICE)
DVC_ENV = DVC_SITE_CACHE_DIR=/tmp/dvc UV_CACHE_DIR=/tmp/uv-cache
LOCAL_DATALAKE_ENV = DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=/app/data
ML_ENV = DATALAKE_BACKEND=local DATALAKE_LOCAL_ROOT=/app/data MLFLOW_TRACKING_URI=http://mlflow:5000

.PHONY: docker-build docker-pull docker-publish docker-up docker-ensure-running mlflow-build mlflow-pull mlflow-publish mlflow-up mlflow-ensure-running viz viz-build viz-up viz-logs install lint linter test pre-commit-install pre-commit-install-local pre-commit-run dvc-status dvc-checkout dvc-repro dvc-repro-bronze dvc-repro-silver dvc-repro-gold bronze silver gold prf-source2bronze dnit-source2bronze ibge-municipios-source2bronze ibge-rgi-source2bronze prf-bronze2silver dnit-bronze2silver ibge-bronze2silver br101-sc-municipio-silver2gold openmeteo-json2gold openmeteo-api2gold activity-group-featurize activity-group-train municipio-day-featurize municipio-day-featurize-weather municipio-day-featurize-no-weather municipio-day-train municipio-day-train-weather municipio-day-train-no-weather

docker-build:
	$(DOCKER_COMPOSE) build $(SERVICE)

docker-pull:
	$(DOCKER_COMPOSE) pull $(SERVICE)

docker-publish: docker-build
	$(DOCKER_COMPOSE) push $(SERVICE)

docker-up: docker-pull
	$(DOCKER_COMPOSE) up -d $(SERVICE)

mlflow-build:
	$(DOCKER_COMPOSE) build mlflow

mlflow-pull:
	$(DOCKER_COMPOSE) pull $(MLFLOW_SERVICE)

mlflow-publish: mlflow-build
	$(DOCKER_COMPOSE) push $(MLFLOW_SERVICE)

mlflow-up: mlflow-pull
	$(DOCKER_COMPOSE) up -d $(MLFLOW_SERVICE)

viz: viz-up

viz-build:
	$(DOCKER_COMPOSE) build $(VIZ_SERVICE)

viz-up:
	$(DOCKER_COMPOSE) up -d --build $(VIZ_SERVICE)

viz-logs:
	$(DOCKER_COMPOSE) logs -f $(VIZ_SERVICE)

docker-ensure-running:
	@running_container="$$( $(DOCKER_COMPOSE) ps --status running -q $(SERVICE) )"; \
	if [ -n "$$running_container" ]; then \
		echo "$(SERVICE) is already running"; \
	else \
		$(MAKE) docker-up; \
	fi

mlflow-ensure-running:
	@running_container="$$( $(DOCKER_COMPOSE) ps --status running -q $(MLFLOW_SERVICE) )"; \
	if [ -n "$$running_container" ]; then \
		echo "$(MLFLOW_SERVICE) is already running"; \
	else \
		$(MAKE) mlflow-up; \
	fi

install: docker-up mlflow-up

lint: docker-ensure-running
	$(DOCKER_EXEC) ruff check src tests
	$(DOCKER_EXEC) ruff format --check src tests

linter: docker-ensure-running
	$(DOCKER_EXEC) ruff check --fix src tests
	$(DOCKER_EXEC) ruff format src tests

test: docker-ensure-running
	$(DOCKER_EXEC) pytest tests/ -v

pre-commit-install-container: docker-ensure-running
	$(DOCKER_EXEC) pre-commit install

pre-commit-install-local:
	uv run pre-commit install

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
	$(DOCKER_EXEC) env $(DVC_ENV) python -m dvc repro br101_sc_municipio_silver2gold

bronze: prf-source2bronze dnit-source2bronze ibge-municipios-source2bronze ibge-rgi-source2bronze

silver: prf-bronze2silver dnit-bronze2silver ibge-bronze2silver

gold: br101-sc-municipio-silver2gold

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

br101-sc-municipio-silver2gold: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.gold.br101_sc_municipio_silver2gold

openmeteo-json2gold: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.openmeteo_src2gold --source-json-dir data/bronze/openmeteo-weather

openmeteo-api2gold: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.etl.silver.openmeteo_src2gold --start-date-from-existing-max --end-date-today

activity-group-featurize: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.ml.activity_group_regression featurize

activity-group-train: docker-ensure-running mlflow-ensure-running
	$(DOCKER_EXEC) env $(ML_ENV) python -m src.ml.activity_group_regression train

municipio-day-featurize: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) python -m src.ml.municipio_day_regression featurize

municipio-day-featurize-weather: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) ML_MUNICIPIO_DAY_INCLUDE_WEATHER_FEATURES=true ML_MUNICIPIO_DAY_FEATURE_OUTPUT_SUBPATH=gold/ml/municipio_day_features_weather python -m src.ml.municipio_day_regression featurize

municipio-day-featurize-no-weather: docker-ensure-running
	$(DOCKER_EXEC) env $(LOCAL_DATALAKE_ENV) ML_MUNICIPIO_DAY_INCLUDE_WEATHER_FEATURES=false ML_MUNICIPIO_DAY_FEATURE_OUTPUT_SUBPATH=gold/ml/municipio_day_features_no_weather python -m src.ml.municipio_day_regression featurize

municipio-day-train: docker-ensure-running mlflow-ensure-running
	$(DOCKER_EXEC) env $(ML_ENV) python -m src.ml.municipio_day_regression train

municipio-day-train-weather: docker-ensure-running mlflow-ensure-running
	$(DOCKER_EXEC) env $(ML_ENV) ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH=gold/ml/municipio_day_features_weather ML_MUNICIPIO_DAY_EXPERIMENT_OUTPUT_SUBPATH=gold/ml/municipio_day_regression_weather MLFLOW_RUN_NAME=municipio_day__variant=with_weather python -m src.ml.municipio_day_regression train

municipio-day-train-no-weather: docker-ensure-running mlflow-ensure-running
	$(DOCKER_EXEC) env $(ML_ENV) ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH=gold/ml/municipio_day_features_no_weather ML_MUNICIPIO_DAY_EXPERIMENT_OUTPUT_SUBPATH=gold/ml/municipio_day_regression_no_weather MLFLOW_RUN_NAME=municipio_day__variant=no_weather python -m src.ml.municipio_day_regression train
