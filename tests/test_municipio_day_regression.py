from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.ml.municipio_day_regression import (
    BASE_SEQUENCE_CHANNELS,
    N_SEQUENCE_CHANNELS,
    MunicipioDayExperimentConfig,
    MunicipioDayRegressionExperiment,
    _assemble_per_municipio_arrays,
    _default_experiment_run_name,
)


class _FakeRocket:
    def transform(self, X_seq: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                X_seq[:, 0, :].mean(axis=1),
                X_seq[:, 0, -1],
                X_seq[:, 1, :].mean(axis=1),
            ]
        ).astype(np.float32)


def _build_panel() -> pd.DataFrame:
    dates = list(pd.date_range("2023-01-01", periods=5, freq="D"))
    dates += list(pd.date_range("2024-01-01", periods=2, freq="D"))
    dates += list(pd.date_range("2025-01-01", periods=2, freq="D"))

    rows = []
    for mun_idx, mun_id in enumerate(["4201", "4202"]):
        for date_idx, accident_date in enumerate(dates):
            year = int(accident_date.year)
            split = "train"
            if year == 2024:
                split = "validation"
            elif year == 2025:
                split = "test"
            rows.append(
                {
                    "codigo_municipio": mun_id,
                    "accident_date": accident_date.date(),
                    "year": year,
                    "accident_count": float(date_idx + mun_idx),
                    "split": split,
                    "non_working_day_weight": 0.0,
                    "days_off_ahead": 0.0,
                    "days_off_before": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _add_weather_columns(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.copy()
    enriched["target_max_temp_c"] = 25.0
    enriched["target_min_temp_c"] = 18.0
    enriched["target_max_wind_speed_kmh"] = 30.0
    enriched["target_sunrise_seconds"] = 6.0 * 3600.0
    enriched["target_sunset_seconds"] = 18.0 * 3600.0
    enriched["target_precipitation_mm"] = 1.0
    enriched["target_precipitation_hours"] = 2.0
    return enriched


def test_assemble_per_municipio_arrays_keeps_split_data_separate() -> None:
    registry = {"municipio_codes": ["4201", "4202"]}
    panel = _build_panel()

    train = _assemble_per_municipio_arrays(panel, registry, 2, "train")
    validation = _assemble_per_municipio_arrays(panel, registry, 2, "validation")

    assert set(train) == {"4201", "4202"}
    assert train["4201"]["X_seq"].shape == (3, BASE_SEQUENCE_CHANNELS, 2)
    assert train["4201"]["y_scaled"].shape == (3,)
    assert train["4201"]["y_raw"].tolist() == [2.0, 3.0, 4.0]
    assert train["4201"]["n"] == 3

    assert validation["4202"]["X_seq"].shape == (2, BASE_SEQUENCE_CHANNELS, 2)
    assert validation["4202"]["y_raw"].tolist() == [6.0, 7.0]
    assert validation["4202"]["y_mean"] == 3.0


def test_assemble_per_municipio_arrays_can_include_weather_channels() -> None:
    registry = {
        "municipio_codes": ["4201", "4202"],
        "include_weather_features": True,
        "n_sequence_channels": N_SEQUENCE_CHANNELS,
    }
    panel = _add_weather_columns(_build_panel())

    train = _assemble_per_municipio_arrays(panel, registry, 2, "train")

    assert train["4201"]["X_seq"].shape == (3, N_SEQUENCE_CHANNELS, 2)
    assert np.isfinite(train["4201"]["X_seq"]).all()


def test_train_and_evaluate_fit_one_ridge_per_municipio(tmp_path: Path) -> None:
    registry = {
        "municipio_codes": ["4201", "4202"],
        "n_municipios": 2,
        "n_sequence_channels": BASE_SEQUENCE_CHANNELS,
        "lookback_days": 2,
    }
    panel = _build_panel()
    per_mun_seqs = {
        split: _assemble_per_municipio_arrays(panel, registry, 2, split)
        for split in ("train", "validation", "test")
    }
    feature_data = {
        "rocket": _FakeRocket(),
        "registry": registry,
        "per_mun_seqs": per_mun_seqs,
        "split_counts": {
            split: sum(mun_data["n"] for mun_data in split_data.values())
            for split, split_data in per_mun_seqs.items()
        },
    }
    config = MunicipioDayExperimentConfig(
        project_root=tmp_path,
        ridge_alphas=[0.1, 1.0],
    )
    experiment = MunicipioDayRegressionExperiment(config=config)

    training_output = experiment.train(feature_data)
    metrics = experiment.evaluate(feature_data, training_output)

    assert set(training_output["models"]) == {"4201", "4202"}
    assert all(isinstance(model, Ridge) for model in training_output["models"].values())
    assert set(training_output["alphas"]) == {"4201", "4202"}
    assert training_output["alpha_selection_split"] == "validation"
    assert set(training_output["alpha_selection_scores"]) == {"4201", "4202"}
    assert training_output["n_rocket_features"] == 3
    assert set(metrics["train"]["per_municipio"]) == {"4201", "4202"}
    assert metrics["validation"]["n"] == 4
    assert "macro_r2" in metrics["validation"]
    assert metrics["validation"]["macro_r2"] == np.mean(
        [
            mun_metrics["r2"]
            for mun_metrics in metrics["validation"]["per_municipio"].values()
        ]
    )


def test_experiment_config_ignores_removed_svd_field(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("svd_components: 1000\nn_kernels: 32\n", encoding="utf-8")
    monkeypatch.setenv("ML_MUNICIPIO_DAY_SVD_COMPONENTS", "2000")

    config = MunicipioDayExperimentConfig.from_yaml(config_path, project_root=tmp_path)

    assert config.n_kernels == 32
    assert not hasattr(config, "svd_components")


def test_experiment_uses_parseable_default_run_name(tmp_path: Path) -> None:
    config = MunicipioDayExperimentConfig(
        project_root=tmp_path,
        n_kernels=64,
        random_seed=7,
    )

    experiment = MunicipioDayRegressionExperiment(config=config)

    assert experiment.run_name == _default_experiment_run_name(config)
    assert experiment.run_name == (
        "municipio_day"
        "__model=shared_minirocket_per_mun_ridge"
        "__alpha_select=validation"
        "__kernels=64"
        "__seed=7"
    )
