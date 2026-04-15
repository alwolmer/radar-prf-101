from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.etl.silver import notebook4_pipeline as nb4
from src.ml.activity_group_regression import (
    ActivityGroupExperimentConfig,
    ActivityGroupFeaturizationConfig,
    ActivityGroupFeaturizationRun,
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
