from __future__ import annotations

from types import SimpleNamespace

from src.ml.base import BaseFeaturizationRun, BaseMLflowRegressionExperiment


class _FakeRunContext:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="run-123")

    def __enter__(self) -> _FakeRunContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.tags = {}
        self.params = {}
        self.metrics = {}
        self.tracking_uri = None
        self.experiment_name = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, run_name: str | None = None) -> _FakeRunContext:
        self.run_name = run_name
        return _FakeRunContext()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value


class _ToyFeaturizer(BaseFeaturizationRun):
    def extract(self) -> dict[str, int]:
        return {"base": 2}

    def transform(self, data: dict[str, int]) -> dict[str, int]:
        return {"value": data["base"] + 3}

    def load(self, data: dict[str, int]) -> dict[str, int]:
        return {"saved": data["value"] * 2}

    def cleanup(self) -> None:
        return None


class _ToyExperiment(BaseMLflowRegressionExperiment):
    def __init__(self) -> None:
        self.persisted = None
        super().__init__(
            tracking_uri="http://mlflow.test",
            experiment_name="toy-exp",
            run_name="toy-run",
        )

    def extract(self) -> dict[str, int]:
        return {"raw": 5}

    def featurize(self, data: dict[str, int]) -> dict[str, int]:
        return {"feature": data["raw"] + 1}

    def train(self, feature_data: dict[str, int]) -> dict[str, int]:
        return {"model_score": feature_data["feature"] * 2}

    def evaluate(
        self,
        feature_data: dict[str, int],
        training_output: dict[str, int],
    ) -> dict[str, float]:
        return {
            "rmse": float(feature_data["feature"]),
            "score": float(training_output["model_score"]),
        }

    def persist(
        self,
        feature_data: dict[str, int],
        training_output: dict[str, int],
        evaluation_output: dict[str, float],
        *,
        run_id: str,
    ) -> dict[str, object]:
        self.persisted = {
            "feature_data": feature_data,
            "training_output": training_output,
            "evaluation_output": evaluation_output,
            "run_id": run_id,
        }
        return self.persisted

    def cleanup(self) -> None:
        return None

    def mlflow_tags(self) -> dict[str, str]:
        return {"stage": "test"}

    def mlflow_params(
        self,
        feature_data: dict[str, int],
        training_output: dict[str, int],
        evaluation_output: dict[str, float],
    ) -> dict[str, object]:
        _ = evaluation_output
        return {
            "feature": feature_data["feature"],
            "model_score": training_output["model_score"],
        }

    def mlflow_metrics(
        self,
        feature_data: dict[str, int],
        training_output: dict[str, int],
        evaluation_output: dict[str, float],
    ) -> dict[str, float]:
        _ = feature_data
        _ = training_output
        return {"rmse": evaluation_output["rmse"]}


def test_base_featurization_run_executes_lifecycle() -> None:
    result = _ToyFeaturizer().run()
    assert result == {"saved": 10}


def test_base_mlflow_regression_experiment_executes_lifecycle(monkeypatch) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("src.ml.base._import_mlflow", lambda: fake_mlflow)

    result = _ToyExperiment().run()

    assert result["run_id"] == "run-123"
    assert result["persisted_output"]["run_id"] == "run-123"
    assert fake_mlflow.tracking_uri == "http://mlflow.test"
    assert fake_mlflow.experiment_name == "toy-exp"
    assert fake_mlflow.tags["stage"] == "test"
    assert fake_mlflow.params["feature"] == "6"
    assert fake_mlflow.metrics["rmse"] == 6.0
