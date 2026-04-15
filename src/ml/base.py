from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

DEFAULT_ML_LOG_LEVEL = "INFO"


def _configure_logging() -> None:
    level_name = os.environ.get("ML_LOG_LEVEL", DEFAULT_ML_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    root_logger.setLevel(level)


def _run_phase(logger: logging.Logger, phase_name: str, operation: Any) -> Any:
    phase_start = time.perf_counter()
    logger.info("Starting %s", phase_name)
    try:
        result = operation()
    except Exception:
        logger.exception(
            "Failed %s after %.2fs", phase_name, time.perf_counter() - phase_start
        )
        raise
    logger.info("Finished %s in %.2fs", phase_name, time.perf_counter() - phase_start)
    return result


def _normalize_mlflow_params(values: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _normalize_mlflow_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        normalized[str(key)] = numeric
    return normalized


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        detail = ""
        if "google.protobuf" in str(exc) or "protobuf" in str(exc).lower():
            detail = (
                " The installed protobuf runtime is likely incompatible with the "
                "pinned MLflow version; sync the environment with the repo lockfile."
            )
        raise ImportError(
            "MLflow could not be imported. Install a compatible MLflow runtime before "
            f"running regression experiments.{detail}"
        ) from exc
    return mlflow


class BaseFeaturizationRun(ABC):
    """Reusable feature-engineering base class with the same lifecycle shape as ETL jobs."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        job_name: str | None = None,
    ) -> None:
        _configure_logging()
        self.config = dict(config or {})
        self.job_name = job_name or self.__class__.__name__
        self.logger = logging.getLogger(self.job_name)
        self.validate_config()

    def validate_config(self) -> None:
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")

    @abstractmethod
    def extract(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    def run(self) -> Any:
        run_start = time.perf_counter()
        self.logger.info("Starting featurization run %s", self.job_name)
        try:
            raw_data = _run_phase(self.logger, "extract phase", self.extract)
            feature_data = _run_phase(
                self.logger,
                "transform phase",
                lambda: self.transform(raw_data),
            )
            result = _run_phase(
                self.logger,
                "load phase",
                lambda: self.load(feature_data),
            )
            self.logger.info(
                "Finished featurization run %s in %.2fs",
                self.job_name,
                time.perf_counter() - run_start,
            )
            return result
        except Exception:
            self.logger.exception("Featurization run %s failed", self.job_name)
            raise
        finally:
            _run_phase(self.logger, "cleanup phase", self.cleanup)


class BaseMLflowRegressionExperiment(ABC):
    """Generic regression lifecycle with MLflow tracking baked into the run contract."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        job_name: str | None = None,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        _configure_logging()
        self.config = dict(config or {})
        self.job_name = job_name or self.__class__.__name__
        self.logger = logging.getLogger(self.job_name)
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "file:./mlruns"
        )
        self.experiment_name = experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", self.job_name
        )
        self.run_name = run_name
        self.validate_config()

    def validate_config(self) -> None:
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")

    @abstractmethod
    def extract(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def featurize(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def train(self, feature_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, feature_data: Any, training_output: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def persist(
        self,
        feature_data: Any,
        training_output: Any,
        evaluation_output: Any,
        *,
        run_id: str,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    def mlflow_tags(self) -> Mapping[str, Any]:
        return {}

    def mlflow_params(
        self,
        feature_data: Any,
        training_output: Any,
        evaluation_output: Any,
    ) -> Mapping[str, Any]:
        return {}

    def mlflow_metrics(
        self,
        feature_data: Any,
        training_output: Any,
        evaluation_output: Any,
    ) -> Mapping[str, Any]:
        return {}

    def log_mlflow_artifacts(
        self,
        feature_data: Any,
        training_output: Any,
        evaluation_output: Any,
    ) -> None:
        return None

    def run(self) -> dict[str, Any]:
        run_start = time.perf_counter()
        mlflow = _import_mlflow()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.logger.info("Starting regression experiment %s", self.job_name)

        try:
            with mlflow.start_run(run_name=self.run_name) as active_run:
                run_id = active_run.info.run_id
                experiment_id = getattr(active_run.info, "experiment_id", None)
                tags = {
                    "job_name": self.job_name,
                    "experiment_type": "regression",
                    **dict(self.mlflow_tags()),
                }
                if tags:
                    mlflow.set_tags(_normalize_mlflow_params(tags))

                raw_data = _run_phase(self.logger, "extract phase", self.extract)
                feature_data = _run_phase(
                    self.logger,
                    "featurize phase",
                    lambda: self.featurize(raw_data),
                )
                training_output = _run_phase(
                    self.logger,
                    "train phase",
                    lambda: self.train(feature_data),
                )
                evaluation_output = _run_phase(
                    self.logger,
                    "evaluate phase",
                    lambda: self.evaluate(feature_data, training_output),
                )

                params = _normalize_mlflow_params(
                    self.mlflow_params(feature_data, training_output, evaluation_output)
                )
                if params:
                    mlflow.log_params(params)

                metrics = _normalize_mlflow_metrics(
                    self.mlflow_metrics(
                        feature_data, training_output, evaluation_output
                    )
                )
                if metrics:
                    mlflow.log_metrics(metrics)

                _run_phase(
                    self.logger,
                    "mlflow artifact logging phase",
                    lambda: self.log_mlflow_artifacts(
                        feature_data,
                        training_output,
                        evaluation_output,
                    ),
                )
                persisted_output = _run_phase(
                    self.logger,
                    "persist phase",
                    lambda: self.persist(
                        feature_data,
                        training_output,
                        evaluation_output,
                        run_id=run_id,
                    ),
                )
                mlflow.log_metric(
                    "experiment_runtime_seconds",
                    time.perf_counter() - run_start,
                )
                self.logger.info(
                    "Finished regression experiment %s in %.2fs",
                    self.job_name,
                    time.perf_counter() - run_start,
                )
                return {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "tracking_uri": self.tracking_uri,
                    "experiment_name": self.experiment_name,
                    "persisted_output": persisted_output,
                    "feature_data": feature_data,
                    "training_output": training_output,
                    "evaluation_output": evaluation_output,
                }
        except Exception:
            self.logger.exception("Regression experiment %s failed", self.job_name)
            raise
        finally:
            _run_phase(self.logger, "cleanup phase", self.cleanup)


class BaseModelServingEndpoint(ABC):
    """Abstract serving endpoint contract for loading a model and producing predictions."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        endpoint_name: str | None = None,
    ) -> None:
        _configure_logging()
        self.config = dict(config or {})
        self.endpoint_name = endpoint_name or self.__class__.__name__
        self.logger = logging.getLogger(self.endpoint_name)
        self.validate_config()

    def validate_config(self) -> None:
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")

    @abstractmethod
    def deploy(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> Any:
        raise NotImplementedError

    def preprocess(self, payload: Any) -> Any:
        return payload

    @abstractmethod
    def predict(self, model: Any, payload: Any) -> Any:
        raise NotImplementedError

    def postprocess(self, predictions: Any) -> Any:
        return predictions

    @abstractmethod
    def healthcheck(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    def invoke(self, payload: Any) -> Any:
        model = self.load_model()
        prepared = self.preprocess(payload)
        predictions = self.predict(model, prepared)
        return self.postprocess(predictions)
