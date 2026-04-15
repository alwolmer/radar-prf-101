from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etl.silver import notebook4_pipeline as nb4
from src.ml.activity_group_regression import (
    ActivityGroupExperimentConfig,
    ActivityGroupFeaturizationConfig,
    ActivityGroupFeaturizationRun,
    ActivityGroupModelConfig,
    ActivityGroupRegressionExperiment,
)


def _build_canonical_accidents() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weekly_counts = {
        "11001": [0, 1, 0, 1, 1, 1, 1, 1],
        "11002": [1, 1, 1, 2, 2, 2, 2, 2],
        "11003": [2, 2, 3, 3, 3, 3, 4, 4],
    }
    names = {"11001": "RGI A", "11002": "RGI B", "11003": "RGI C"}
    accident_id = 1
    for week_idx, week_start in enumerate(
        pd.date_range("2024-01-01", periods=8, freq="W-MON")
    ):
        for rgi_id, counts in weekly_counts.items():
            for occurrence in range(counts[week_idx]):
                rows.append(
                    {
                        "id": accident_id,
                        "timestamp": week_start + pd.Timedelta(days=occurrence),
                        "year": int(week_start.year),
                        "uf": "SC",
                        "codigo_rgi": rgi_id,
                        "nome_rgi": names[rgi_id],
                    }
                )
                accident_id += 1
    return pd.DataFrame(rows)


def _build_road_sections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id": "rgi_1",
                "codigo_rgi": "11001",
                "nome_rgi": "RGI A",
                "sg_uf": "SC",
                "cd_rgint": "1001",
                "nm_rgint": "RGINT 1",
                "road_length_m": 1000.0,
            },
            {
                "section_id": "rgi_2",
                "codigo_rgi": "11002",
                "nome_rgi": "RGI B",
                "sg_uf": "SC",
                "cd_rgint": "1002",
                "nm_rgint": "RGINT 2",
                "road_length_m": 2000.0,
            },
            {
                "section_id": "rgi_3",
                "codigo_rgi": "11003",
                "nome_rgi": "RGI C",
                "sg_uf": "SC",
                "cd_rgint": "1003",
                "nm_rgint": "RGINT 3",
                "road_length_m": 3000.0,
            },
        ]
    )


def test_activity_group_featurization_run_materializes_group_bundles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    gold_root = data_root / "gold" / "br101_rgi_weekly_panel"
    (gold_root / "canonical_accidents_by_rgi_section").mkdir(
        parents=True, exist_ok=True
    )
    (gold_root / "road_sections_by_rgi").mkdir(parents=True, exist_ok=True)
    _build_canonical_accidents().to_parquet(
        gold_root / "canonical_accidents_by_rgi_section" / "part.parquet",
        index=False,
    )
    _build_road_sections().to_parquet(
        gold_root / "road_sections_by_rgi" / "part.parquet",
        index=False,
    )

    monkeypatch.setenv("DATALAKE_BACKEND", "local")
    monkeypatch.setenv("DATALAKE_LOCAL_ROOT", str(data_root))

    sequence_config = nb4.Notebook4Config(
        project_root=tmp_path,
        notebook3_silver_dir=tmp_path / "unused" / "silver",
        notebook3_gold_dir=tmp_path / "unused" / "gold",
        silver_output_dir=tmp_path / "data" / "silver" / "out",
        gold_output_dir=tmp_path / "data" / "gold" / "out",
        train_start_year=2024,
        train_start_week=1,
        train_end_year=2024,
        train_end_week=4,
        validation_start_year=2024,
        validation_start_week=5,
        validation_end_year=2024,
        validation_end_week=6,
        test_start_year=2024,
        test_start_week=7,
        test_end_year=2024,
        test_end_week=8,
        lookback_weeks=2,
        latency_gap_weeks=1,
        forecast_horizon_weeks=1,
        rnn_train_stride_weeks=1,
        rnn_eval_stride_weeks=1,
        low_activity_share=0.34,
        high_activity_share=0.33,
    )
    config = ActivityGroupFeaturizationConfig(
        project_root=tmp_path,
        sequence_config=sequence_config,
        output_subpath="gold/ml/activity_group_test_features",
        recent_history_weeks=2,
        seasonal_lag_weeks=(1, 2),
    )

    result = ActivityGroupFeaturizationRun(config=config).run()

    output_dir = data_root / "gold" / "ml" / "activity_group_test_features"
    assert Path(result["output_uri"]) == output_dir
    assert result["weekly_rgi_panel_with_groups"]["group_name"].nunique() == 3
    assert set(result["group_sequence_manifest"]["split"]) == {
        "train",
        "validation",
        "test",
    }
    assert (
        output_dir / "group_sequence_bundles" / "low_activity" / "train.npz"
    ).exists()
    assert (output_dir / "weekly_rgi_panel_with_groups.parquet").exists()
    assert (output_dir / "dynamic_exclusion_diagnostics.parquet").exists()
    assert (output_dir / "group_rgi_vocabulary.parquet").exists()


def test_activity_group_experiment_config_loads_model_yaml_overrides(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config" / "ml" / "activity_group"
    model_root = config_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    (config_root / "featurization.yaml").write_text(
        "\n".join(
            [
                "gold_input_subpath: gold/br101_rgi_weekly_panel",
                "output_subpath: gold/ml/activity_group_features",
                "bridge_weight: 0.5",
                "recent_history_weeks: 10",
                "seasonal_lag_weeks:",
                "  - 52",
                "sequence_defaults:",
                "  lookback_weeks: 12",
            ]
        ),
        encoding="utf-8",
    )
    (config_root / "experiment.yaml").write_text(
        "\n".join(
            [
                "tracking_uri: http://mlflow:5000",
                "experiment_name: activity-group-test",
                "model_config_dir: config/ml/activity_group/models",
                "architectures:",
                "  - gru",
                "  - lstm",
            ]
        ),
        encoding="utf-8",
    )
    (model_root / "gru.yaml").write_text(
        "\n".join(
            [
                "architecture: gru",
                "recurrent_units: 16",
                "dense_units_first: 48",
                "dense_units_second: 24",
                "random_seed: 77",
                "max_epochs: 12",
                "batch_size: 8",
                "early_stopping_patience: 3",
                "rgi_representation_mode: one_hot",
            ]
        ),
        encoding="utf-8",
    )
    (model_root / "lstm.yaml").write_text(
        "\n".join(
            [
                "architecture: lstm",
                "recurrent_units: 20",
                "dense_units_first: 40",
                "dense_units_second: 20",
                "random_seed: 88",
                "max_epochs: 15",
                "batch_size: 4",
                "early_stopping_patience: 2",
                "rgi_representation_mode: embedding",
                "rgi_embedding_dim: 6",
            ]
        ),
        encoding="utf-8",
    )

    config = ActivityGroupExperimentConfig.from_project_root(tmp_path)

    assert config.architectures == ("gru", "lstm")
    assert config.model_configs is not None
    assert config.model_configs["gru"].recurrent_units == 16
    assert config.model_configs["gru"].batch_size == 8
    assert config.model_configs["gru"].rgi_representation_mode == "one_hot"
    assert config.model_configs["lstm"].dense_units_second == 20
    assert config.model_configs["lstm"].random_seed == 88
    assert config.model_configs["lstm"].rgi_embedding_dim == 6


def test_activity_group_experiment_persists_group_comparison_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATALAKE_BACKEND", "local")
    monkeypatch.setenv("DATALAKE_LOCAL_ROOT", str(data_root))

    sequence_config = nb4.Notebook4Config(
        project_root=tmp_path,
        notebook3_silver_dir=tmp_path / "unused" / "silver",
        notebook3_gold_dir=tmp_path / "unused" / "gold",
        silver_output_dir=tmp_path / "data" / "silver" / "out",
        gold_output_dir=tmp_path / "data" / "gold" / "out",
        train_start_year=2024,
        train_start_week=1,
        train_end_year=2024,
        train_end_week=4,
        validation_start_year=2024,
        validation_start_week=5,
        validation_end_year=2024,
        validation_end_week=6,
        test_start_year=2024,
        test_start_week=7,
        test_end_year=2024,
        test_end_week=8,
        lookback_weeks=2,
        latency_gap_weeks=1,
        forecast_horizon_weeks=1,
        rnn_train_stride_weeks=1,
        rnn_eval_stride_weeks=1,
        low_activity_share=0.34,
        high_activity_share=0.33,
    )
    feature_config = ActivityGroupFeaturizationConfig(
        project_root=tmp_path,
        sequence_config=sequence_config,
        output_subpath="gold/ml/activity_group_features",
        recent_history_weeks=2,
        seasonal_lag_weeks=(1, 2),
    )
    model_configs = {
        "gru": ActivityGroupModelConfig(
            architecture="gru",
            recurrent_units=16,
            dense_units_first=32,
            dense_units_second=16,
            random_seed=7,
            max_epochs=5,
            batch_size=8,
            early_stopping_patience=2,
        ),
        "lstm": ActivityGroupModelConfig(
            architecture="lstm",
            recurrent_units=16,
            dense_units_first=32,
            dense_units_second=16,
            random_seed=7,
            max_epochs=5,
            batch_size=8,
            early_stopping_patience=2,
        ),
    }
    experiment_config = ActivityGroupExperimentConfig(
        project_root=tmp_path,
        tracking_uri="http://mlflow:5000",
        experiment_name="activity-group-test",
        architectures=("gru", "lstm"),
        show_progress=False,
        model_config_dir=tmp_path,
        model_configs=model_configs,
    )
    experiment = ActivityGroupRegressionExperiment(
        experiment_config=experiment_config,
        feature_config=feature_config,
    )

    split_metrics = pd.DataFrame(
        [
            {"split": "train", "rmse": 1.0, "r2": 0.8, "n_predictions": 10},
            {"split": "validation", "rmse": 1.2, "r2": 0.7, "n_predictions": 5},
            {"split": "test", "rmse": 1.4, "r2": 0.6, "n_predictions": 4},
        ]
    )
    comparison_dataset_key = "low_activity|history=2|horizon=1|latency=1|lags=1-2|rgis=3"
    output_dir = experiment._finalize_experiment_output(
        feature_data={
            "output_uri": str(data_root / "gold" / "ml" / "activity_group_features"),
            "group_sequence_manifest": pd.DataFrame(
                [{"group_name": "low_activity", "split": "train"}]
            ),
        },
        group_results=[
            {
                "group_name": "low_activity",
                "run_id": "parent-run",
                "experiment_id": "1",
                "comparison_dataset_key": comparison_dataset_key,
                "artifact_dir": tmp_path / "artifacts" / "comparison",
            }
        ],
        variant_results=[
            {
                "group_name": "low_activity",
                "architecture": "gru",
                "run_id": "child-run",
                "experiment_id": "1",
                "parent_run_id": "parent-run",
                "parent_experiment_id": "1",
                "comparison_dataset_key": comparison_dataset_key,
                "split_metrics": split_metrics,
                "artifact_dir": tmp_path / "artifacts" / "gru",
            }
        ],
    )

    group_manifest = pd.read_parquet(output_dir / "group_run_manifest.parquet")
    run_manifest = pd.read_parquet(output_dir / "run_manifest.parquet")

    assert group_manifest.loc[0, "group_name"] == "low_activity"
    assert group_manifest.loc[0, "run_id"] == "parent-run"
    assert group_manifest.loc[0, "comparison_dataset_key"] == comparison_dataset_key
    assert run_manifest.loc[0, "run_id"] == "child-run"
    assert run_manifest.loc[0, "parent_run_id"] == "parent-run"
    assert run_manifest.loc[0, "comparison_dataset_key"] == comparison_dataset_key


def test_activity_group_comparison_output_materializes_plots_and_test_window_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATALAKE_BACKEND", "local")
    monkeypatch.setenv("DATALAKE_LOCAL_ROOT", str(data_root))

    sequence_config = nb4.Notebook4Config(
        project_root=tmp_path,
        notebook3_silver_dir=tmp_path / "unused" / "silver",
        notebook3_gold_dir=tmp_path / "unused" / "gold",
        silver_output_dir=tmp_path / "data" / "silver" / "out",
        gold_output_dir=tmp_path / "data" / "gold" / "out",
        train_start_year=2024,
        train_start_week=1,
        train_end_year=2024,
        train_end_week=4,
        validation_start_year=2024,
        validation_start_week=5,
        validation_end_year=2024,
        validation_end_week=6,
        test_start_year=2024,
        test_start_week=7,
        test_end_year=2024,
        test_end_week=14,
        lookback_weeks=2,
        latency_gap_weeks=1,
        forecast_horizon_weeks=1,
        rnn_train_stride_weeks=1,
        rnn_eval_stride_weeks=1,
        low_activity_share=0.34,
        high_activity_share=0.33,
    )
    feature_config = ActivityGroupFeaturizationConfig(
        project_root=tmp_path,
        sequence_config=sequence_config,
        output_subpath="gold/ml/activity_group_features",
        recent_history_weeks=2,
        seasonal_lag_weeks=(1, 2),
    )
    experiment_config = ActivityGroupExperimentConfig(
        project_root=tmp_path,
        tracking_uri="http://mlflow:5000",
        experiment_name="activity-group-test",
        architectures=("gru", "lstm"),
        show_progress=False,
        model_config_dir=tmp_path,
        model_configs={
            "gru": ActivityGroupModelConfig(
                architecture="gru",
                recurrent_units=16,
                dense_units_first=32,
                dense_units_second=16,
                random_seed=7,
                max_epochs=5,
                batch_size=8,
                early_stopping_patience=2,
            ),
            "lstm": ActivityGroupModelConfig(
                architecture="lstm",
                recurrent_units=16,
                dense_units_first=32,
                dense_units_second=16,
                random_seed=7,
                max_epochs=5,
                batch_size=8,
                early_stopping_patience=2,
            ),
        },
    )
    experiment = ActivityGroupRegressionExperiment(
        experiment_config=experiment_config,
        feature_config=feature_config,
    )

    weeks = pd.date_range("2024-02-12", periods=8, freq="W-MON")

    def build_forecasts(architecture: str, offset: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for rgi_id, rgi_name, base_actual in [
            ("11001", "RGI A", 1),
            ("11002", "RGI B", 3),
        ]:
            for step, week_start in enumerate(weeks, start=1):
                rows.append(
                    {
                        "model": f"activity_group_{architecture}",
                        "architecture": architecture,
                        "group_name": "low_activity",
                        "split": "test",
                        "rgi_id": rgi_id,
                        "rgi_name": rgi_name,
                        "uf": "SC",
                        "week_start": week_start,
                        "week_end": week_start + pd.Timedelta(days=6),
                        "year_week": f"2024-{step:02d}",
                        "forecast_batch_start": week_start,
                        "cutoff_week_start": week_start - pd.Timedelta(days=7),
                        "horizon_step": 1,
                        "prediction": base_actual + step + offset,
                        "actual": base_actual + step,
                        "is_available": True,
                        "metadata": "synthetic=1",
                    }
                )
        return pd.DataFrame(rows)

    comparison_output = experiment._materialize_group_comparison_output(
        group_name="low_activity",
        group_variant_results=[
            {
                "architecture": "gru",
                "run_id": "run-gru",
                "split_metrics": pd.DataFrame(
                    [
                        {
                            "split": "train",
                            "rmse": 0.9,
                            "r2": 0.7,
                            "n_predictions": 16,
                        },
                        {
                            "split": "validation",
                            "rmse": 1.1,
                            "r2": 0.6,
                            "n_predictions": 8,
                        },
                        {
                            "split": "test",
                            "rmse": 1.4,
                            "r2": 0.5,
                            "n_predictions": 16,
                        },
                    ]
                ),
                "forecasts": build_forecasts("gru", 0),
            },
            {
                "architecture": "lstm",
                "run_id": "run-lstm",
                "split_metrics": pd.DataFrame(
                    [
                        {
                            "split": "train",
                            "rmse": 0.8,
                            "r2": 0.75,
                            "n_predictions": 16,
                        },
                        {
                            "split": "validation",
                            "rmse": 1.0,
                            "r2": 0.65,
                            "n_predictions": 8,
                        },
                        {
                            "split": "test",
                            "rmse": 1.2,
                            "r2": 0.55,
                            "n_predictions": 16,
                        },
                    ]
                ),
                "forecasts": build_forecasts("lstm", 1),
            },
        ],
        run_id="parent-run",
    )

    assert (comparison_output / "architecture_split_metrics.parquet").exists()
    assert (comparison_output / "test_window_forecasts.parquet").exists()
    assert (comparison_output / "plots" / "architecture_comparison.png").exists()
    assert (comparison_output / "plots" / "test_window_rgi_comparison.png").exists()

    test_window = pd.read_parquet(comparison_output / "test_window_forecasts.parquet")
    assert set(test_window["architecture"]) == {"gru", "lstm"}
    assert test_window["week_start"].nunique() == 8
    assert test_window["rgi_id"].nunique() == 2
