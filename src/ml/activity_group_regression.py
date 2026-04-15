"""Activity-group forecasting workflow built on top of notebook 3/4 helpers.

This module intentionally reuses the exploratory implementation in
`src.etl.silver.notebook4_pipeline` where it is still useful for evaluation and
grouping logic, but the ML-facing contracts live here.

Pipeline summary:
1. Read only gold-layer artefacts from `gold/br101_rgi_weekly_panel`.
2. Rebuild the dense `RGI x week` panel used for forecasting.
3. Apply config-driven exclusion rules, then derive train-only activity groups.
4. Build group training bundles with:
   - short recent-history windows
   - seasonal lag features aligned to each forecast step
   - RGI identity features
   - future holiday features
5. Train one model per `(activity_group, architecture)` pair.
6. Persist gold-layer feature artefacts plus experiment outputs and a feature-store
   style view of the final design matrix per architecture.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import yaml

from src.etl.datalake import DatalakeAdapter
from src.etl.silver import notebook3_pipeline as nb3
from src.etl.silver import notebook4_pipeline as nb4
from src.ml.base import (
    BaseFeaturizationRun,
    BaseMLflowRegressionExperiment,
    _import_mlflow,
    _normalize_mlflow_metrics,
    _normalize_mlflow_params,
    _run_phase,
)

DEFAULT_FEATURE_INPUT_SUBPATH = "gold/br101_rgi_weekly_panel"
DEFAULT_FEATURE_OUTPUT_SUBPATH = "gold/ml/activity_group_features"
DEFAULT_EXPERIMENT_OUTPUT_SUBPATH = "gold/ml/activity_group_regression"
DEFAULT_EXPERIMENT_NAME = "radar-prf-101-activity-group-rnn"
DEFAULT_CONFIG_ROOT = Path("config/ml/activity_group")
DEFAULT_FEATURIZATION_CONFIG_PATH = DEFAULT_CONFIG_ROOT / "featurization.yaml"
DEFAULT_EXPERIMENT_CONFIG_PATH = DEFAULT_CONFIG_ROOT / "experiment.yaml"
DEFAULT_MODEL_CONFIG_DIR = DEFAULT_CONFIG_ROOT / "models"
DEFAULT_SEASONAL_LAG_WEEKS = (52, 104, 156)
DEFAULT_FEATURE_STORE_MAX_ROWS = 20000
DEFAULT_DYNAMIC_EXCLUSIONS = (
    {
        "name": "exclude_sp_and_rs",
        "conditions": [
            {"column": "uf", "values": ["SP", "RS"]},
        ],
    },
)
SEQUENCE_META_COLUMNS = [
    "group_name",
    "rgi_id",
    "rgi_name",
    "uf",
    "rgi_index",
    "forecast_batch_start",
    "cutoff_week_start",
    "target_week_starts",
    "target_week_ends",
    "target_year_weeks",
]


def _resolve_browser_tracking_uri(tracking_uri: str) -> str:
    split = urlsplit(tracking_uri)
    if split.scheme not in {"http", "https"}:
        return tracking_uri

    browser_host = (
        os.environ.get("MLFLOW_BROWSER_HOST", "localhost").strip() or "localhost"
    )
    browser_port = os.environ.get("MLFLOW_BROWSER_PORT", "").strip()
    hostname = split.hostname or ""
    if hostname not in {"mlflow", "0.0.0.0"}:
        return tracking_uri

    target_port = browser_port or (str(split.port) if split.port is not None else "")
    netloc = browser_host if not target_port else f"{browser_host}:{target_port}"
    return urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))


def _print_training_summary(result: Mapping[str, Any]) -> None:
    persisted_output = Path(str(result["persisted_output"]))
    print(f"Persisted output: {persisted_output}")

    feature_store_root = persisted_output / "feature_store"
    if feature_store_root.exists():
        print(f"Feature store: {feature_store_root}")
    runs_root = persisted_output / "runs"
    if runs_root.exists():
        print(f"Variant artifacts: {runs_root}")

    tracking_uri = str(result.get("tracking_uri", ""))
    browser_tracking_uri = _resolve_browser_tracking_uri(tracking_uri)
    variant_results = list(result.get("variant_results", []))
    experiment_id = next(
        (
            item.get("experiment_id")
            for item in variant_results
            if item.get("experiment_id")
        ),
        None,
    )
    if (
        browser_tracking_uri.startswith(("http://", "https://"))
        and experiment_id is not None
    ):
        print(f"Experiment UI: {browser_tracking_uri}/#/experiments/{experiment_id}")
        for item in variant_results:
            run_id = item.get("run_id")
            if not run_id:
                continue
            print(
                "Run UI "
                f"[{item['group_name']}/{item['architecture']}]: "
                f"{browser_tracking_uri}/#/experiments/{item['experiment_id']}/runs/{run_id}"
            )


def _env_with_fallback(*names: str, default: str) -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return raw
    return default


def _csv_env(
    primary_name: str,
    legacy_name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw_value = os.environ.get(primary_name) or os.environ.get(legacy_name)
    if not raw_value:
        return default
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _bool_env(primary_name: str, legacy_name: str, default: bool) -> bool:
    raw = os.environ.get(primary_name)
    if raw is None:
        raw = os.environ.get(legacy_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_root() -> Path:
    return nb3.resolve_project_root()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML config at {path} must deserialize to a mapping.")
    return dict(raw)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )


def _empty_manual_exclusions() -> pd.DataFrame:
    return pd.DataFrame(columns=["uf", "year", "exclude_from_modeling"])


def _split_bounds(
    config: nb4.Notebook4Config,
    split: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return config.train_start, config.train_end
    if split == "validation":
        return config.validation_start, config.validation_end
    if split == "test":
        return config.test_start, config.test_end
    raise ValueError(f"Unsupported split {split!r}.")


def _is_aligned_batch_start(
    batch_start: pd.Timestamp,
    split_start: pd.Timestamp,
    stride_weeks: int,
) -> bool:
    week_delta = int((batch_start - split_start).days // 7)
    return week_delta % stride_weeks == 0


def _normalize_overall_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}

    metrics: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        architecture = str(row.architecture)
        split = str(row.split)
        metrics[f"rmse__{split}__{architecture}"] = float(row.rmse)
        metrics[f"macro_rmse__{split}__{architecture}"] = float(row.macro_rmse)
        metrics[f"coverage__{split}__{architecture}"] = float(row.forecast_row_coverage)
    return metrics


def _build_notebook3_compatible_config(
    config: ActivityGroupFeaturizationConfig,
) -> nb3.Notebook3Config:
    sequence_config = config.sequence_config
    start_iso = sequence_config.train_start.isocalendar()
    end_iso = sequence_config.test_end.isocalendar()
    return nb3.Notebook3Config(
        project_root=config.project_root,
        silver_input_dir=config.project_root / "data" / "silver" / "eda_runbook",
        silver_output_dir=config.project_root
        / "data"
        / "silver"
        / "notebook3_accident_count",
        gold_output_dir=config.project_root
        / "data"
        / "gold"
        / "notebook3_accident_count",
        iso_start_year=int(start_iso.year),
        iso_start_week=int(start_iso.week),
        iso_end_year=int(end_iso.year),
        iso_end_week=int(end_iso.week),
        train_start_year=sequence_config.train_start_year,
        train_start_week=sequence_config.train_start_week,
        train_end_year=sequence_config.train_end_year,
        train_end_week=sequence_config.train_end_week,
        validation_start_year=sequence_config.validation_start_year,
        validation_start_week=sequence_config.validation_start_week,
        validation_end_year=sequence_config.validation_end_year,
        validation_end_week=sequence_config.validation_end_week,
        test_start_year=sequence_config.test_start_year,
        test_start_week=sequence_config.test_start_week,
        test_end_year=sequence_config.test_end_year,
        test_end_week=sequence_config.test_end_week,
        lookback_weeks=config.recent_history_weeks,
        latency_gap_weeks=sequence_config.latency_gap_weeks,
        forecast_horizon_weeks=sequence_config.forecast_horizon_weeks,
        rnn_train_stride_weeks=sequence_config.rnn_train_stride_weeks,
        rnn_eval_stride_weeks=sequence_config.rnn_eval_stride_weeks,
        random_seed=sequence_config.random_seed,
        bridge_weight=config.bridge_weight,
    )


def _build_rgi_static_features(road_sections_by_rgi: pd.DataFrame) -> pd.DataFrame:
    section_frame = road_sections_by_rgi.copy()
    section_frame["codigo_rgi"] = nb3.normalize_code(section_frame["codigo_rgi"])
    section_frame["cd_rgint"] = nb3.normalize_code(section_frame["cd_rgint"])
    section_frame["sg_uf"] = (
        section_frame["sg_uf"].astype("string").str.upper().str.strip()
    )

    grouped = (
        section_frame.groupby(
            ["codigo_rgi", "nome_rgi", "sg_uf", "cd_rgint", "nm_rgint"],
            as_index=False,
            dropna=False,
        )["road_length_m"]
        .sum()
        .rename(
            columns={
                "codigo_rgi": "rgi_id",
                "nome_rgi": "rgi_name",
                "sg_uf": "uf",
                "cd_rgint": "rgint_id",
                "nm_rgint": "rgint_name",
                "road_length_m": "br101_length_m_in_rgi",
            }
        )
    )
    grouped["br101_length_km_in_rgi"] = grouped["br101_length_m_in_rgi"] / 1000.0
    grouped["rgi_name"] = grouped["rgi_name"].astype("string")
    grouped["uf"] = grouped["uf"].astype("string")
    grouped["rgint_name"] = grouped["rgint_name"].astype("string")
    return grouped.sort_values(["uf", "rgi_id"]).reset_index(drop=True)


def _normalize_value_set(values: tuple[str, ...]) -> set[str]:
    return {str(value).strip().upper() for value in values}


def _series_as_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.upper().str.strip()


@dataclass(frozen=True)
class DynamicExclusionCondition:
    column: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class DynamicExclusionRule:
    name: str
    conditions: tuple[DynamicExclusionCondition, ...]


def _parse_dynamic_exclusion_rules(raw_rules: Any) -> tuple[DynamicExclusionRule, ...]:
    rules_source = raw_rules if raw_rules is not None else DEFAULT_DYNAMIC_EXCLUSIONS
    parsed_rules: list[DynamicExclusionRule] = []
    for rule_idx, raw_rule in enumerate(rules_source):
        if not isinstance(raw_rule, dict):
            raise TypeError("Each dynamic exclusion rule must be a mapping.")
        raw_conditions = raw_rule.get("conditions", [])
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise ValueError("Each dynamic exclusion rule must include conditions.")
        conditions = tuple(
            DynamicExclusionCondition(
                column=str(condition["column"]),
                values=tuple(str(value) for value in condition["values"]),
            )
            for condition in raw_conditions
        )
        parsed_rules.append(
            DynamicExclusionRule(
                name=str(raw_rule.get("name", f"rule_{rule_idx + 1}")),
                conditions=conditions,
            )
        )
    return tuple(parsed_rules)


def _apply_dynamic_exclusion_rules(
    frame: pd.DataFrame,
    rules: tuple[DynamicExclusionRule, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rules:
        return frame.copy(), pd.DataFrame(columns=["rule_name", "rows_excluded"])

    excluded_mask = pd.Series(False, index=frame.index)
    diagnostics: list[dict[str, Any]] = []
    for rule in rules:
        rule_mask = pd.Series(True, index=frame.index)
        for condition in rule.conditions:
            if condition.column not in frame.columns:
                raise KeyError(
                    f"Dynamic exclusion column {condition.column!r} was not found in the frame."
                )
            series = frame[condition.column]
            if pd.api.types.is_numeric_dtype(series):
                normalized_values = (
                    pd.to_numeric(pd.Series(list(condition.values)), errors="coerce")
                    .dropna()
                    .astype(float)
                    .tolist()
                )
                condition_mask = pd.to_numeric(series, errors="coerce").isin(
                    normalized_values
                )
            else:
                condition_mask = _series_as_string(series).isin(
                    _normalize_value_set(condition.values)
                )
            rule_mask &= condition_mask.fillna(False)
        excluded_mask |= rule_mask
        diagnostics.append(
            {
                "rule_name": rule.name,
                "rows_excluded": int(rule_mask.sum()),
            }
        )

    retained = frame.loc[~excluded_mask].copy()
    return retained, pd.DataFrame(diagnostics)


@dataclass(frozen=True)
class ActivityGroupFeaturizationConfig:
    """Config for transforming canonical accident assignments into model features."""

    project_root: Path
    sequence_config: nb4.Notebook4Config
    gold_input_subpath: str = DEFAULT_FEATURE_INPUT_SUBPATH
    output_subpath: str = DEFAULT_FEATURE_OUTPUT_SUBPATH
    bridge_weight: float = 0.5
    recent_history_weeks: int = 12
    seasonal_lag_weeks: tuple[int, ...] = DEFAULT_SEASONAL_LAG_WEEKS
    dynamic_exclusion_rules: tuple[DynamicExclusionRule, ...] = field(
        default_factory=lambda: _parse_dynamic_exclusion_rules(None)
    )
    feature_store_max_rows: int = DEFAULT_FEATURE_STORE_MAX_ROWS
    config_path: Path | None = None

    @classmethod
    def from_project_root(
        cls,
        project_root: Path | None = None,
    ) -> ActivityGroupFeaturizationConfig:
        root = project_root or _resolve_project_root()
        config_path = root / DEFAULT_FEATURIZATION_CONFIG_PATH
        yaml_config = _read_yaml_mapping(config_path)
        sequence_defaults = nb4.Notebook4Config.from_project_root(root)
        sequence_overrides = {
            key: value
            for key, value in yaml_config.get("sequence_defaults", {}).items()
            if value is not None
        }
        sequence_config = replace(sequence_defaults, **sequence_overrides)
        return cls(
            project_root=root,
            sequence_config=sequence_config,
            gold_input_subpath=_env_with_fallback(
                "ML_ACTIVITY_GROUP_FEATURE_INPUT_SUBPATH",
                "ML_NOTEBOOK4_FEATURE_INPUT_SUBPATH",
                default=str(
                    yaml_config.get("gold_input_subpath", DEFAULT_FEATURE_INPUT_SUBPATH)
                ),
            ),
            output_subpath=_env_with_fallback(
                "ML_ACTIVITY_GROUP_FEATURE_OUTPUT_SUBPATH",
                "ML_NOTEBOOK4_FEATURE_OUTPUT_SUBPATH",
                default=str(
                    yaml_config.get("output_subpath", DEFAULT_FEATURE_OUTPUT_SUBPATH)
                ),
            ),
            bridge_weight=float(
                _env_with_fallback(
                    "ML_ACTIVITY_GROUP_BRIDGE_WEIGHT",
                    "ML_NOTEBOOK4_BRIDGE_WEIGHT",
                    default=str(yaml_config.get("bridge_weight", 0.5)),
                )
            ),
            recent_history_weeks=int(yaml_config.get("recent_history_weeks", 12)),
            seasonal_lag_weeks=tuple(
                int(value)
                for value in yaml_config.get(
                    "seasonal_lag_weeks",
                    list(DEFAULT_SEASONAL_LAG_WEEKS),
                )
            ),
            dynamic_exclusion_rules=_parse_dynamic_exclusion_rules(
                yaml_config.get("dynamic_exclusion_rules")
            ),
            feature_store_max_rows=int(
                yaml_config.get(
                    "feature_store_max_rows", DEFAULT_FEATURE_STORE_MAX_ROWS
                )
            ),
            config_path=config_path,
        )


@dataclass(frozen=True)
class ActivityGroupModelConfig:
    """Architecture-specific training overrides loaded from one YAML per model."""

    architecture: str
    recurrent_units: int
    dense_units_first: int
    dense_units_second: int
    random_seed: int
    max_epochs: int
    batch_size: int
    early_stopping_patience: int
    rgi_representation_mode: str = "embedding"
    rgi_embedding_dim: int = 8
    config_path: Path | None = None

    @classmethod
    def from_yaml(
        cls,
        *,
        architecture: str,
        path: Path,
        base_config: nb4.Notebook4Config,
    ) -> ActivityGroupModelConfig:
        if not path.exists():
            raise FileNotFoundError(f"Missing model hyperparameter config at {path}.")
        yaml_config = _read_yaml_mapping(path)
        return cls(
            architecture=str(yaml_config.get("architecture", architecture)),
            recurrent_units=int(
                yaml_config.get("recurrent_units", base_config.recurrent_units)
            ),
            dense_units_first=int(
                yaml_config.get("dense_units_first", base_config.dense_units_first)
            ),
            dense_units_second=int(
                yaml_config.get("dense_units_second", base_config.dense_units_second)
            ),
            random_seed=int(yaml_config.get("random_seed", base_config.random_seed)),
            max_epochs=int(yaml_config.get("max_epochs", base_config.max_epochs)),
            batch_size=int(yaml_config.get("batch_size", base_config.batch_size)),
            early_stopping_patience=int(
                yaml_config.get(
                    "early_stopping_patience",
                    base_config.early_stopping_patience,
                )
            ),
            rgi_representation_mode=str(
                yaml_config.get("rgi_representation_mode", "embedding")
            ),
            rgi_embedding_dim=int(yaml_config.get("rgi_embedding_dim", 8)),
            config_path=path,
        )

    def apply_to(self, base_config: nb4.Notebook4Config) -> nb4.Notebook4Config:
        return replace(
            base_config,
            recurrent_units=self.recurrent_units,
            dense_units_first=self.dense_units_first,
            dense_units_second=self.dense_units_second,
            random_seed=self.random_seed,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping_patience=self.early_stopping_patience,
        )


@dataclass(frozen=True)
class ActivityGroupExperimentConfig:
    """Top-level experiment config: MLflow wiring plus enabled architectures."""

    project_root: Path
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    output_subpath: str = DEFAULT_EXPERIMENT_OUTPUT_SUBPATH
    architectures: tuple[str, ...] = ("gru", "lstm")
    show_progress: bool = True
    model_config_dir: Path = DEFAULT_MODEL_CONFIG_DIR
    config_path: Path | None = None
    model_configs: dict[str, ActivityGroupModelConfig] | None = None

    @classmethod
    def from_project_root(
        cls,
        project_root: Path | None = None,
    ) -> ActivityGroupExperimentConfig:
        root = project_root or _resolve_project_root()
        config_path = root / DEFAULT_EXPERIMENT_CONFIG_PATH
        yaml_config = _read_yaml_mapping(config_path)
        architecture_values = yaml_config.get("architectures", ("gru", "lstm"))
        if not isinstance(architecture_values, list | tuple):
            raise TypeError(
                "architectures must be a list in the activity-group experiment YAML."
            )
        architectures = tuple(str(item) for item in architecture_values)
        model_config_dir = root / Path(
            yaml_config.get("model_config_dir", DEFAULT_MODEL_CONFIG_DIR)
        )
        base_sequence_config = ActivityGroupFeaturizationConfig.from_project_root(
            root
        ).sequence_config
        model_configs = {
            architecture: ActivityGroupModelConfig.from_yaml(
                architecture=architecture,
                path=model_config_dir / f"{architecture}.yaml",
                base_config=base_sequence_config,
            )
            for architecture in architectures
        }
        return cls(
            project_root=root,
            tracking_uri=os.environ.get(
                "MLFLOW_TRACKING_URI",
                str(yaml_config.get("tracking_uri", "http://mlflow:5000")),
            ),
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME",
                str(yaml_config.get("experiment_name", DEFAULT_EXPERIMENT_NAME)),
            ),
            run_name=os.environ.get("MLFLOW_RUN_NAME") or None,
            output_subpath=_env_with_fallback(
                "ML_ACTIVITY_GROUP_EXPERIMENT_OUTPUT_SUBPATH",
                "ML_NOTEBOOK4_EXPERIMENT_OUTPUT_SUBPATH",
                default=str(
                    yaml_config.get("output_subpath", DEFAULT_EXPERIMENT_OUTPUT_SUBPATH)
                ),
            ),
            architectures=_csv_env(
                "ML_ACTIVITY_GROUP_ARCHITECTURES",
                "ML_NOTEBOOK4_ARCHITECTURES",
                architectures,
            ),
            show_progress=_bool_env(
                "ML_ACTIVITY_GROUP_SHOW_PROGRESS",
                "ML_NOTEBOOK4_SHOW_PROGRESS",
                bool(yaml_config.get("show_progress", True)),
            ),
            model_config_dir=model_config_dir,
            config_path=config_path,
            model_configs=model_configs,
        )


def _build_group_rgi_vocabularies(
    panel_with_groups: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    vocabularies: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for group_name in nb4.GROUP_ORDER:
        group_frame = (
            panel_with_groups.loc[
                panel_with_groups["group_name"].eq(group_name),
                ["rgi_id", "rgi_name", "uf"],
            ]
            .drop_duplicates()
            .sort_values(["uf", "rgi_id"])
        )
        if group_frame.empty:
            continue
        vocabularies[group_name] = {
            str(rgi_id): idx
            for idx, rgi_id in enumerate(group_frame["rgi_id"].astype(str).tolist())
        }
        rows.extend(
            [
                {
                    "group_name": group_name,
                    "rgi_id": str(row.rgi_id),
                    "rgi_name": str(row.rgi_name),
                    "uf": str(row.uf),
                    "rgi_index": vocabularies[group_name][str(row.rgi_id)],
                }
                for row in group_frame.itertuples(index=False)
            ]
        )
    return vocabularies, pd.DataFrame(rows)


def _build_seasonal_context(
    scaled_counts: np.ndarray,
    *,
    target_start_idx: int,
    batch_end_idx: int,
    seasonal_lag_weeks: tuple[int, ...],
) -> np.ndarray:
    rows: list[list[float]] = []
    for target_idx in range(target_start_idx, batch_end_idx + 1):
        seasonal_values = []
        for lag in seasonal_lag_weeks:
            lag_idx = target_idx - lag
            seasonal_values.append(
                float(scaled_counts[lag_idx]) if lag_idx >= 0 else 0.0
            )
        rows.append(seasonal_values)
    return np.asarray(rows, dtype=np.float32)


def _build_activity_sequence_dataset(
    rgi_panel: pd.DataFrame,
    config: ActivityGroupFeaturizationConfig,
    split: str,
    *,
    target_scale: float,
    stride_weeks: int,
    rgi_index: int,
) -> dict[str, Any]:
    split_start, split_end = _split_bounds(config.sequence_config, split)
    group = rgi_panel.sort_values("week_start").reset_index(drop=True)
    counts = group["accident_count"].to_numpy(dtype=np.float32)
    weeks = group["week_start"].tolist()
    week_ends = group["week_end"].tolist()
    year_weeks = group["year_week"].tolist()
    holiday_values = group[nb3.HOLIDAY_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaled_counts = counts / target_scale
    scaled_length = float(group["br101_length_km_in_rgi"].iloc[0])
    scale_feature = float(np.log1p(target_scale))

    recent_history_rows: list[np.ndarray] = []
    seasonal_context_rows: list[np.ndarray] = []
    future_holiday_rows: list[np.ndarray] = []
    road_length_rows: list[list[float]] = []
    scale_feature_rows: list[list[float]] = []
    rgi_index_rows: list[list[int]] = []
    target_rows: list[np.ndarray] = []
    actual_target_rows: list[np.ndarray] = []
    target_scale_rows: list[list[float]] = []
    meta_rows: list[dict[str, Any]] = []

    for target_start_idx, batch_start in enumerate(weeks):
        batch_end_idx = (
            target_start_idx + config.sequence_config.forecast_horizon_weeks - 1
        )
        cutoff_idx = target_start_idx - config.sequence_config.latency_gap_weeks
        recent_history_start_idx = cutoff_idx - config.recent_history_weeks + 1

        if batch_start < split_start or batch_end_idx >= len(weeks):
            continue
        if (
            weeks[batch_end_idx] > split_end
            or cutoff_idx < 0
            or recent_history_start_idx < 0
        ):
            continue
        if not _is_aligned_batch_start(batch_start, split_start, stride_weeks):
            continue

        recent_history_rows.append(
            scaled_counts[recent_history_start_idx : cutoff_idx + 1].reshape(-1, 1)
        )
        seasonal_context_rows.append(
            _build_seasonal_context(
                scaled_counts,
                target_start_idx=target_start_idx,
                batch_end_idx=batch_end_idx,
                seasonal_lag_weeks=config.seasonal_lag_weeks,
            )
        )
        future_holiday_rows.append(holiday_values[target_start_idx : batch_end_idx + 1])
        road_length_rows.append([scaled_length])
        scale_feature_rows.append([scale_feature])
        rgi_index_rows.append([rgi_index])
        target_rows.append(scaled_counts[target_start_idx : batch_end_idx + 1])
        actual_target_rows.append(counts[target_start_idx : batch_end_idx + 1])
        target_scale_rows.append([target_scale])
        meta_rows.append(
            {
                "group_name": str(group["group_name"].iloc[0]),
                "rgi_id": str(group["rgi_id"].iloc[0]),
                "rgi_name": str(group["rgi_name"].iloc[0]),
                "uf": str(group["uf"].iloc[0]),
                "rgi_index": int(rgi_index),
                "forecast_batch_start": batch_start,
                "cutoff_week_start": weeks[cutoff_idx],
                "target_week_starts": weeks[target_start_idx : batch_end_idx + 1],
                "target_week_ends": week_ends[target_start_idx : batch_end_idx + 1],
                "target_year_weeks": year_weeks[target_start_idx : batch_end_idx + 1],
            }
        )

    if not recent_history_rows:
        return {
            "split": split,
            "group_name": str(group["group_name"].iloc[0]),
            "rgi_id": str(group["rgi_id"].iloc[0]),
            "inputs": {
                "recent_history": np.empty(
                    (0, config.recent_history_weeks, 1), dtype=np.float32
                ),
                "seasonal_context": np.empty(
                    (
                        0,
                        config.sequence_config.forecast_horizon_weeks,
                        len(config.seasonal_lag_weeks),
                    ),
                    dtype=np.float32,
                ),
                "future_holiday": np.empty(
                    (
                        0,
                        config.sequence_config.forecast_horizon_weeks,
                        len(nb3.HOLIDAY_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                "road_length": np.empty((0, 1), dtype=np.float32),
                "scale_feature": np.empty((0, 1), dtype=np.float32),
                "rgi_index": np.empty((0, 1), dtype=np.int32),
            },
            "targets": np.empty(
                (0, config.sequence_config.forecast_horizon_weeks), dtype=np.float32
            ),
            "actual_targets": np.empty(
                (0, config.sequence_config.forecast_horizon_weeks), dtype=np.float32
            ),
            "target_scale": np.empty((0, 1), dtype=np.float32),
            "meta": pd.DataFrame(meta_rows, columns=SEQUENCE_META_COLUMNS),
        }

    return {
        "split": split,
        "group_name": str(group["group_name"].iloc[0]),
        "rgi_id": str(group["rgi_id"].iloc[0]),
        "inputs": {
            "recent_history": np.stack(recent_history_rows).astype(np.float32),
            "seasonal_context": np.stack(seasonal_context_rows).astype(np.float32),
            "future_holiday": np.stack(future_holiday_rows).astype(np.float32),
            "road_length": np.asarray(road_length_rows, dtype=np.float32),
            "scale_feature": np.asarray(scale_feature_rows, dtype=np.float32),
            "rgi_index": np.asarray(rgi_index_rows, dtype=np.int32),
        },
        "targets": np.stack(target_rows).astype(np.float32),
        "actual_targets": np.stack(actual_target_rows).astype(np.float32),
        "target_scale": np.asarray(target_scale_rows, dtype=np.float32),
        "meta": pd.DataFrame(meta_rows),
    }


def _empty_group_sequence_dataset(
    config: ActivityGroupFeaturizationConfig,
    *,
    split: str,
    group_name: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "group_name": group_name,
        "inputs": {
            "recent_history": np.empty(
                (0, config.recent_history_weeks, 1), dtype=np.float32
            ),
            "seasonal_context": np.empty(
                (
                    0,
                    config.sequence_config.forecast_horizon_weeks,
                    len(config.seasonal_lag_weeks),
                ),
                dtype=np.float32,
            ),
            "future_holiday": np.empty(
                (
                    0,
                    config.sequence_config.forecast_horizon_weeks,
                    len(nb3.HOLIDAY_FEATURE_COLUMNS),
                ),
                dtype=np.float32,
            ),
            "road_length": np.empty((0, 1), dtype=np.float32),
            "scale_feature": np.empty((0, 1), dtype=np.float32),
            "rgi_index": np.empty((0, 1), dtype=np.int32),
        },
        "targets": np.empty(
            (0, config.sequence_config.forecast_horizon_weeks), dtype=np.float32
        ),
        "actual_targets": np.empty(
            (0, config.sequence_config.forecast_horizon_weeks), dtype=np.float32
        ),
        "target_scale": np.empty((0, 1), dtype=np.float32),
        "meta": pd.DataFrame(columns=SEQUENCE_META_COLUMNS),
    }


def _combine_sequence_bundles(
    bundles: list[dict[str, Any]],
    config: ActivityGroupFeaturizationConfig,
    *,
    split: str,
    group_name: str,
) -> dict[str, Any]:
    non_empty = [bundle for bundle in bundles if len(bundle["meta"]) > 0]
    if not non_empty:
        return _empty_group_sequence_dataset(config, split=split, group_name=group_name)

    return {
        "split": split,
        "group_name": group_name,
        "inputs": {
            "recent_history": np.concatenate(
                [bundle["inputs"]["recent_history"] for bundle in non_empty], axis=0
            ).astype(np.float32),
            "seasonal_context": np.concatenate(
                [bundle["inputs"]["seasonal_context"] for bundle in non_empty], axis=0
            ).astype(np.float32),
            "future_holiday": np.concatenate(
                [bundle["inputs"]["future_holiday"] for bundle in non_empty], axis=0
            ).astype(np.float32),
            "road_length": np.concatenate(
                [bundle["inputs"]["road_length"] for bundle in non_empty], axis=0
            ).astype(np.float32),
            "scale_feature": np.concatenate(
                [bundle["inputs"]["scale_feature"] for bundle in non_empty], axis=0
            ).astype(np.float32),
            "rgi_index": np.concatenate(
                [bundle["inputs"]["rgi_index"] for bundle in non_empty], axis=0
            ).astype(np.int32),
        },
        "targets": np.concatenate(
            [bundle["targets"] for bundle in non_empty], axis=0
        ).astype(np.float32),
        "actual_targets": np.concatenate(
            [bundle["actual_targets"] for bundle in non_empty], axis=0
        ).astype(np.float32),
        "target_scale": np.concatenate(
            [bundle["target_scale"] for bundle in non_empty], axis=0
        ).astype(np.float32),
        "meta": pd.concat(
            [bundle["meta"] for bundle in non_empty], ignore_index=True
        ).reset_index(drop=True),
    }


def _prepare_activity_group_datasets(
    panel_with_groups: pd.DataFrame,
    config: ActivityGroupFeaturizationConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame]:
    group_vocabularies, vocabulary_frame = _build_group_rgi_vocabularies(
        panel_with_groups
    )

    local_datasets: dict[str, dict[str, Any]] = {}
    for rgi_id, rgi_panel in panel_with_groups.groupby("rgi_id"):
        rgi_panel = rgi_panel.sort_values("week_start").reset_index(drop=True)
        group_name = str(rgi_panel["group_name"].iloc[0])
        rgi_index = group_vocabularies[group_name][str(rgi_id)]
        target_scale = nb4.build_local_target_scale(rgi_panel)
        local_datasets[str(rgi_id)] = {
            "meta": {
                "rgi_id": str(rgi_id),
                "rgi_name": str(rgi_panel["rgi_name"].iloc[0]),
                "uf": str(rgi_panel["uf"].iloc[0]),
                "group_name": group_name,
                "rgi_index": int(rgi_index),
                "target_scale": target_scale,
            },
            "train": _build_activity_sequence_dataset(
                rgi_panel,
                config,
                split="train",
                target_scale=target_scale,
                stride_weeks=config.sequence_config.rnn_train_stride_weeks,
                rgi_index=rgi_index,
            ),
            "validation": _build_activity_sequence_dataset(
                rgi_panel,
                config,
                split="validation",
                target_scale=target_scale,
                stride_weeks=config.sequence_config.rnn_eval_stride_weeks,
                rgi_index=rgi_index,
            ),
            "test": _build_activity_sequence_dataset(
                rgi_panel,
                config,
                split="test",
                target_scale=target_scale,
                stride_weeks=config.sequence_config.rnn_eval_stride_weeks,
                rgi_index=rgi_index,
            ),
        }

    group_datasets: dict[str, dict[str, Any]] = {}
    for group_name in nb4.GROUP_ORDER:
        datasets_in_group = [
            dataset
            for dataset in local_datasets.values()
            if str(dataset["meta"]["group_name"]) == group_name
        ]
        if not datasets_in_group:
            continue
        group_datasets[group_name] = {
            "meta": {
                "group_name": group_name,
                "n_rgis": int(len(datasets_in_group)),
                "rgi_ids": sorted(
                    str(dataset["meta"]["rgi_id"]) for dataset in datasets_in_group
                ),
            },
            "train": _combine_sequence_bundles(
                [dataset["train"] for dataset in datasets_in_group],
                config,
                split="train",
                group_name=group_name,
            ),
            "validation": _combine_sequence_bundles(
                [dataset["validation"] for dataset in datasets_in_group],
                config,
                split="validation",
                group_name=group_name,
            ),
            "test": _combine_sequence_bundles(
                [dataset["test"] for dataset in datasets_in_group],
                config,
                split="test",
                group_name=group_name,
            ),
        }
    return local_datasets, group_datasets, vocabulary_frame


def _build_group_sequence_manifest(
    group_datasets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, dataset in group_datasets.items():
        for split in ("train", "validation", "test"):
            bundle = dataset[split]
            rows.append(
                {
                    "group_name": group_name,
                    "split": split,
                    "n_rgis": int(dataset["meta"]["n_rgis"]),
                    "n_samples": int(len(bundle["meta"])),
                    "min_week_start": bundle["meta"]["forecast_batch_start"].min()
                    if not bundle["meta"].empty
                    else pd.NaT,
                    "max_week_start": bundle["meta"]["forecast_batch_start"].max()
                    if not bundle["meta"].empty
                    else pd.NaT,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["group_name", "split"]).reset_index(drop=True)
    )


def _build_local_sequence_manifest(
    local_datasets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rgi_id, dataset in local_datasets.items():
        meta = dataset["meta"]
        for split in ("train", "validation", "test"):
            bundle = dataset[split]
            rows.append(
                {
                    "rgi_id": rgi_id,
                    "rgi_name": meta["rgi_name"],
                    "uf": meta["uf"],
                    "group_name": meta["group_name"],
                    "rgi_index": meta["rgi_index"],
                    "split": split,
                    "n_samples": int(len(bundle["meta"])),
                    "min_week_start": bundle["meta"]["forecast_batch_start"].min()
                    if not bundle["meta"].empty
                    else pd.NaT,
                    "max_week_start": bundle["meta"]["forecast_batch_start"].max()
                    if not bundle["meta"].empty
                    else pd.NaT,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["group_name", "rgi_id", "split"])
        .reset_index(drop=True)
    )


def _save_bundle(bundle: dict[str, Any], bundle_root: Path, split: str) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_root / f"{split}.npz",
        recent_history=bundle["inputs"]["recent_history"],
        seasonal_context=bundle["inputs"]["seasonal_context"],
        future_holiday=bundle["inputs"]["future_holiday"],
        road_length=bundle["inputs"]["road_length"],
        scale_feature=bundle["inputs"]["scale_feature"],
        rgi_index=bundle["inputs"]["rgi_index"],
        targets=bundle["targets"],
        actual_targets=bundle["actual_targets"],
        target_scale=bundle["target_scale"],
    )
    bundle["meta"].to_parquet(bundle_root / f"{split}_meta.parquet", index=False)


def _save_keras_model(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)


def _prepare_model_inputs(
    bundle: dict[str, Any],
    model_config: ActivityGroupModelConfig,
    *,
    n_rgis: int,
) -> dict[str, np.ndarray]:
    base_inputs = {
        "recent_history": bundle["inputs"]["recent_history"],
        "seasonal_context": bundle["inputs"]["seasonal_context"],
        "future_holiday": bundle["inputs"]["future_holiday"],
        "road_length": bundle["inputs"]["road_length"],
        "scale_feature": bundle["inputs"]["scale_feature"],
    }
    rgi_index = bundle["inputs"]["rgi_index"].reshape(-1)
    if model_config.rgi_representation_mode == "one_hot":
        one_hot = np.eye(n_rgis, dtype=np.float32)[rgi_index]
        return {**base_inputs, "rgi_one_hot": one_hot}
    if model_config.rgi_representation_mode == "embedding":
        return {
            **base_inputs,
            "rgi_index": bundle["inputs"]["rgi_index"].astype(np.int32),
        }
    raise ValueError("rgi_representation_mode must be either 'one_hot' or 'embedding'.")


def _build_activity_group_model(
    sequence_config: nb4.Notebook4Config,
    feature_config: ActivityGroupFeaturizationConfig,
    model_config: ActivityGroupModelConfig,
    *,
    architecture: str,
    n_rgis: int,
) -> Any:
    nb4.require_tensorflow()
    keras = nb4.keras
    if keras is None:
        raise ModuleNotFoundError(nb4.tensorflow_status_message())

    recent_history_input = keras.Input(
        shape=(feature_config.recent_history_weeks, 1),
        name="recent_history",
    )
    seasonal_context_input = keras.Input(
        shape=(
            sequence_config.forecast_horizon_weeks,
            len(feature_config.seasonal_lag_weeks),
        ),
        name="seasonal_context",
    )
    future_holiday_input = keras.Input(
        shape=(
            sequence_config.forecast_horizon_weeks,
            len(nb3.HOLIDAY_FEATURE_COLUMNS),
        ),
        name="future_holiday",
    )
    road_length_input = keras.Input(shape=(1,), name="road_length")
    scale_feature_input = keras.Input(shape=(1,), name="scale_feature")

    recurrent_layer = keras.layers.GRU if architecture == "gru" else keras.layers.LSTM
    recent_history_encoded = recurrent_layer(
        model_config.recurrent_units,
        name=f"recent_history_{architecture}",
    )(recent_history_input)
    seasonal_context_flat = keras.layers.Flatten(name="seasonal_context_flat")(
        seasonal_context_input
    )
    future_holiday_flat = keras.layers.Flatten(name="future_holiday_flat")(
        future_holiday_input
    )

    model_inputs: dict[str, Any] = {
        "recent_history": recent_history_input,
        "seasonal_context": seasonal_context_input,
        "future_holiday": future_holiday_input,
        "road_length": road_length_input,
        "scale_feature": scale_feature_input,
    }
    representation_features: list[Any] = []
    if model_config.rgi_representation_mode == "one_hot":
        rgi_identity_input = keras.Input(shape=(n_rgis,), name="rgi_one_hot")
        model_inputs["rgi_one_hot"] = rgi_identity_input
        representation_features.append(rgi_identity_input)
    else:
        rgi_identity_input = keras.Input(shape=(1,), dtype="int32", name="rgi_index")
        model_inputs["rgi_index"] = rgi_identity_input
        representation_features.append(
            keras.layers.Flatten(name="rgi_embedding_flat")(
                keras.layers.Embedding(
                    input_dim=n_rgis,
                    output_dim=model_config.rgi_embedding_dim,
                    name="rgi_embedding",
                )(rgi_identity_input)
            )
        )

    x = keras.layers.Concatenate(name="model_features")(
        [
            recent_history_encoded,
            seasonal_context_flat,
            future_holiday_flat,
            road_length_input,
            scale_feature_input,
            *representation_features,
        ]
    )
    x = keras.layers.Dense(
        model_config.dense_units_first,
        activation="relu",
        name="dense_1",
    )(x)
    x = keras.layers.Dense(
        model_config.dense_units_second,
        activation="relu",
        name="dense_2",
    )(x)
    output = keras.layers.Dense(
        sequence_config.forecast_horizon_weeks,
        name="forecast",
    )(x)

    model = keras.Model(
        inputs=model_inputs,
        outputs=output,
        name=f"activity_group_{architecture}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def _train_activity_group_model(
    train_bundle: dict[str, Any],
    validation_bundle: dict[str, Any],
    *,
    sequence_config: nb4.Notebook4Config,
    feature_config: ActivityGroupFeaturizationConfig,
    model_config: ActivityGroupModelConfig,
    architecture: str,
    n_rgis: int,
    show_progress: bool,
) -> dict[str, Any]:
    nb4.require_tensorflow()
    tf = nb4.tf
    keras = nb4.keras
    if tf is None or keras is None:
        raise ModuleNotFoundError(nb4.tensorflow_status_message())
    if train_bundle["targets"].size == 0:
        raise ValueError(
            f"No train samples were found for group_name={train_bundle['group_name']!r}."
        )
    if validation_bundle["targets"].size == 0:
        raise ValueError(
            "No validation samples were found for "
            f"group_name={validation_bundle['group_name']!r}."
        )

    tf.keras.utils.set_random_seed(model_config.random_seed)
    model = _build_activity_group_model(
        sequence_config,
        feature_config,
        model_config,
        architecture=architecture,
        n_rgis=n_rgis,
    )

    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=model_config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    fit_verbose = 1 if show_progress else 0
    if show_progress and nb4.TqdmCallback is not None:
        callbacks.append(
            nb4.TqdmCallback(
                verbose=0,
                desc=f"{train_bundle['group_name']} {architecture.upper()} training",
                leave=True,
            )
        )
        fit_verbose = 0

    timer_start = perf_counter()
    history = model.fit(
        x=_prepare_model_inputs(train_bundle, model_config, n_rgis=n_rgis),
        y=train_bundle["targets"],
        validation_data=(
            _prepare_model_inputs(validation_bundle, model_config, n_rgis=n_rgis),
            validation_bundle["targets"],
        ),
        epochs=model_config.max_epochs,
        batch_size=model_config.batch_size,
        verbose=fit_verbose,
        callbacks=callbacks,
        shuffle=True,
    )
    runtime_seconds = perf_counter() - timer_start

    history_dict = {key: list(value) for key, value in history.history.items()}
    best_epoch = (
        int(np.argmin(history.history["val_loss"]) + 1)
        if history.history.get("val_loss")
        else 0
    )
    best_val_loss = (
        float(np.min(history.history["val_loss"]))
        if history.history.get("val_loss")
        else float("nan")
    )
    return {
        "group_name": train_bundle["group_name"],
        "architecture": architecture,
        "model": model,
        "history": history_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "runtime_seconds": runtime_seconds,
        "n_train_samples": int(len(train_bundle["meta"])),
        "n_validation_samples": int(len(validation_bundle["meta"])),
        "n_train_rgis": int(train_bundle["meta"]["rgi_id"].nunique()),
        "n_validation_rgis": int(validation_bundle["meta"]["rgi_id"].nunique()),
        "hyperparameters": asdict(model_config),
    }


def _generate_activity_group_forecasts(
    model: Any,
    bundle: dict[str, Any],
    *,
    model_config: ActivityGroupModelConfig,
    architecture: str,
    n_rgis: int,
) -> pd.DataFrame:
    nb4.require_tensorflow()
    if bundle["targets"].size == 0:
        return nb4.empty_forecast_frame()
    predictions_scaled = model.predict(
        _prepare_model_inputs(bundle, model_config, n_rgis=n_rgis),
        verbose=0,
    )
    return nb4.bundle_predictions_to_forecast_frame(
        bundle,
        predictions_scaled,
        architecture=architecture,
    )


def _build_feature_store_frame(
    group_datasets: dict[str, dict[str, Any]],
    model_config: ActivityGroupModelConfig,
    *,
    n_total_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, dataset in group_datasets.items():
        n_group_rgis = int(dataset["meta"]["n_rgis"])
        for split in ("train", "validation", "test"):
            bundle = dataset[split]
            if bundle["meta"].empty:
                continue
            prepared_inputs = _prepare_model_inputs(
                bundle,
                model_config,
                n_rgis=n_group_rgis,
            )
            meta_records = bundle["meta"].to_dict(orient="records")
            for row_idx, meta in enumerate(meta_records):
                row = {
                    "split": split,
                    "group_name": group_name,
                    "rgi_id": meta["rgi_id"],
                    "rgi_name": meta["rgi_name"],
                    "uf": meta["uf"],
                    "forecast_batch_start": meta["forecast_batch_start"],
                    "cutoff_week_start": meta["cutoff_week_start"],
                    "road_length": float(bundle["inputs"]["road_length"][row_idx][0]),
                    "scale_feature": float(
                        bundle["inputs"]["scale_feature"][row_idx][0]
                    ),
                    "rgi_index": int(bundle["inputs"]["rgi_index"][row_idx][0]),
                }
                recent_history = bundle["inputs"]["recent_history"][row_idx].reshape(-1)
                for feature_idx, value in enumerate(recent_history[::-1], start=1):
                    row[f"recent_history_t_minus_{feature_idx}"] = float(value)

                seasonal_context = bundle["inputs"]["seasonal_context"][row_idx]
                for step_idx in range(seasonal_context.shape[0]):
                    for lag_idx in range(seasonal_context.shape[1]):
                        row[f"seasonal_lag_{lag_idx + 1}_step_{step_idx + 1}"] = float(
                            seasonal_context[step_idx, lag_idx]
                        )

                future_holiday = bundle["inputs"]["future_holiday"][row_idx]
                for step_idx in range(future_holiday.shape[0]):
                    for feature_idx, feature_name in enumerate(
                        nb3.HOLIDAY_FEATURE_COLUMNS
                    ):
                        row[f"{feature_name}_step_{step_idx + 1}"] = float(
                            future_holiday[step_idx, feature_idx]
                        )

                if model_config.rgi_representation_mode == "one_hot":
                    one_hot = prepared_inputs["rgi_one_hot"][row_idx]
                    for one_hot_idx, value in enumerate(one_hot, start=1):
                        row[f"rgi_one_hot_{one_hot_idx:02d}"] = float(value)

                targets = bundle["targets"][row_idx]
                actual_targets = bundle["actual_targets"][row_idx]
                for step_idx, target_value in enumerate(targets, start=1):
                    row[f"target_scaled_step_{step_idx}"] = float(target_value)
                for step_idx, actual_value in enumerate(actual_targets, start=1):
                    row[f"actual_target_step_{step_idx}"] = float(actual_value)

                rows.append(row)
                if len(rows) >= n_total_rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _build_feature_registry(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_frame.columns:
        role = "metadata"
        if column.startswith("recent_history_"):
            role = "recent_history_feature"
        elif column.startswith("seasonal_lag_"):
            role = "seasonal_feature"
        elif column.startswith(tuple(nb3.HOLIDAY_FEATURE_COLUMNS)):
            role = "future_holiday_feature"
        elif column.startswith("rgi_one_hot_") or column == "rgi_index":
            role = "rgi_identity_feature"
        elif column in {"road_length", "scale_feature"}:
            role = "static_feature"
        elif column.startswith("target_") or column.startswith("actual_target_"):
            role = "target"

        rows.append(
            {
                "feature_name": column,
                "role": role,
                "dtype": str(feature_frame[column].dtype),
                "included_in_model": role.endswith("_feature"),
            }
        )
    return pd.DataFrame(rows)


def _build_feature_profile(
    feature_frame: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    profiled_columns = registry.loc[
        registry["included_in_model"], "feature_name"
    ].tolist()
    rows: list[dict[str, Any]] = []
    for column in profiled_columns:
        series = feature_frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "feature_name": column,
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan,
                "min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "max": float(numeric.max()) if numeric.notna().any() else np.nan,
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _persist_feature_store_views(
    output_dir: Path,
    feature_data: dict[str, Any],
    experiment_config: ActivityGroupExperimentConfig,
    feature_config: ActivityGroupFeaturizationConfig,
) -> None:
    feature_store_root = output_dir / "feature_store"
    for architecture, model_config in experiment_config.model_configs.items():
        architecture_root = feature_store_root / architecture
        architecture_root.mkdir(parents=True, exist_ok=True)
        feature_frame = _build_feature_store_frame(
            feature_data["group_datasets"],
            model_config,
            n_total_rows=feature_config.feature_store_max_rows,
        )
        registry = _build_feature_registry(feature_frame)
        profile = _build_feature_profile(feature_frame, registry)
        feature_frame.to_parquet(
            architecture_root / "feature_matrix.parquet", index=False
        )
        registry.to_parquet(architecture_root / "feature_registry.parquet", index=False)
        profile.to_parquet(architecture_root / "feature_profile.parquet", index=False)
        feature_data["group_rgi_vocabulary"].to_parquet(
            architecture_root / "group_rgi_vocabulary.parquet",
            index=False,
        )


def _summarize_split_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    available = forecasts.loc[
        forecasts["prediction"].notna() & forecasts["actual"].notna()
    ].copy()
    if available.empty:
        return pd.DataFrame(columns=["split", "rmse", "r2", "n_predictions"])

    rows: list[dict[str, Any]] = []
    split_order = {"train": 0, "validation": 1, "test": 2}
    for split, frame in available.groupby("split", dropna=False):
        actual = frame["actual"].to_numpy(dtype=float)
        prediction = frame["prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "split": str(split),
                "rmse": nb4._rmse(actual, prediction),
                "r2": nb4._r2(actual, prediction),
                "n_predictions": int(len(frame)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["_order"] = summary["split"].map(split_order).fillna(99)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _write_model_summary(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    destination.write_text("\n".join(summary_lines), encoding="utf-8")


def _write_training_curve_plot(
    history: Mapping[str, list[float]], destination: Path
) -> None:
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, max((len(values) for values in history.values()), default=0) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(epochs), history.get("loss", []), label="train", linewidth=2)
    axes[0].plot(
        list(epochs), history.get("val_loss", []), label="validation", linewidth=2
    )
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()

    axes[1].plot(list(epochs), history.get("rmse", []), label="train", linewidth=2)
    axes[1].plot(
        list(epochs), history.get("val_rmse", []), label="validation", linewidth=2
    )
    axes[1].set_title("RMSE by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_split_timeline_plot(
    forecasts: pd.DataFrame,
    split_metrics: pd.DataFrame,
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    split_order = ["train", "validation", "test"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for axis, split in zip(axes, split_order, strict=True):
        frame = forecasts.loc[forecasts["split"].eq(split)].copy()
        metric_row = split_metrics.loc[split_metrics["split"].eq(split)]
        axis.set_title(f"{split.title()} Forecasts")
        if frame.empty:
            axis.text(0.5, 0.5, "No predictions", ha="center", va="center")
            axis.set_axis_off()
            continue

        timeline = (
            frame.groupby("week_start", as_index=False)[["actual", "prediction"]]
            .mean(numeric_only=True)
            .sort_values("week_start")
        )
        axis.plot(
            timeline["week_start"],
            timeline["actual"],
            label="actual",
            color="#1f1f1f",
            linewidth=2,
        )
        axis.plot(
            timeline["week_start"],
            timeline["prediction"],
            label="prediction",
            color="#1d6fd6",
            linewidth=2,
        )
        if not metric_row.empty:
            row = metric_row.iloc[0]
            axis.text(
                0.01,
                0.95,
                f"RMSE={row['rmse']:.3f}  R2={row['r2']:.3f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
        axis.legend(loc="upper right")
        axis.set_ylabel("Accident count")

    axes[-1].set_xlabel("Week start")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_prediction_parity_plot(forecasts: pd.DataFrame, destination: Path) -> None:
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    available = forecasts.loc[
        forecasts["prediction"].notna() & forecasts["actual"].notna()
    ].copy()
    fig, axis = plt.subplots(figsize=(7, 7))
    if available.empty:
        axis.text(0.5, 0.5, "No predictions", ha="center", va="center")
        axis.set_axis_off()
    else:
        color_map = {
            "train": "#26547c",
            "validation": "#f4a259",
            "test": "#2a9d8f",
        }
        for split, frame in available.groupby("split", dropna=False):
            axis.scatter(
                frame["actual"],
                frame["prediction"],
                s=20,
                alpha=0.45,
                label=str(split),
                color=color_map.get(str(split), "#666666"),
            )
        max_value = float(
            max(available["actual"].max(), available["prediction"].max(), 1.0)
        )
        axis.plot([0, max_value], [0, max_value], linestyle="--", color="#222222")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Actual accidents")
        axis.set_ylabel("Predicted accidents")
        axis.set_title("Prediction Parity by Split")
        axis.legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


class ActivityGroupFeaturizationRun(BaseFeaturizationRun):
    """Build gold-layer activity-group feature artefacts from canonical BR-101 inputs."""

    def __init__(
        self,
        *,
        config: ActivityGroupFeaturizationConfig | None = None,
        datalake_config: dict[str, Any] | None = None,
    ) -> None:
        self.featurization_config = (
            config or ActivityGroupFeaturizationConfig.from_project_root()
        )
        self.notebook3_config = _build_notebook3_compatible_config(
            self.featurization_config
        )
        self.datalake = DatalakeAdapter.from_env(
            project_root=self.featurization_config.project_root,
            config=datalake_config,
        )
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)
        super().__init__(
            config=asdict(self.featurization_config),
            job_name="activity_group_featurization",
        )

    def validate_config(self) -> None:
        super().validate_config()
        if not self.featurization_config.gold_input_subpath:
            raise ValueError("gold_input_subpath must not be empty")
        if not self.featurization_config.output_subpath.startswith("gold/"):
            raise ValueError(
                "Activity-group featurization must persist under the gold layer."
            )

    def extract(self) -> dict[str, pd.DataFrame]:
        gold_root = self.datalake.stage_directory(
            self.featurization_config.gold_input_subpath,
            self._staging_dir / "gold_inputs",
        )
        return {
            "canonical_accidents_by_rgi_section": pd.read_parquet(
                gold_root / "canonical_accidents_by_rgi_section"
            ),
            "road_sections_by_rgi": pd.read_parquet(gold_root / "road_sections_by_rgi"),
        }

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        rgi_static_features = _build_rgi_static_features(data["road_sections_by_rgi"])

        canonical_like = data["canonical_accidents_by_rgi_section"].rename(
            columns={"codigo_rgi": "CD_RGI", "nome_rgi": "NM_RGI"}
        )
        canonical_like, exclusion_diagnostics = _apply_dynamic_exclusion_rules(
            canonical_like,
            self.featurization_config.dynamic_exclusion_rules,
        )

        modeling_accidents, preparation_summary = nb3.prepare_modeling_accidents(
            canonical_like,
            _empty_manual_exclusions(),
            rgi_static_features,
            self.notebook3_config,
        )
        weekly_calendar = nb3.build_weekly_calendar(self.notebook3_config)
        weekly_rgi_target = nb3.aggregate_weekly_targets(
            modeling_accidents, weekly_calendar
        )
        weekly_holiday_features = nb3.build_weekly_holiday_features(
            rgi_static_features,
            weekly_calendar,
            self.notebook3_config,
        )
        weekly_rgi_panel = nb3.build_dense_weekly_panel(
            rgi_static_features,
            weekly_calendar,
            weekly_rgi_target,
            weekly_holiday_features,
        )
        weekly_rgi_panel = nb4.assign_split_labels(
            weekly_rgi_panel,
            self.featurization_config.sequence_config,
        )
        rgi_activity_summary = nb4.assign_activity_groups(
            nb4.build_rgi_activity_summary(weekly_rgi_panel),
            self.featurization_config.sequence_config,
        )
        weekly_rgi_panel_with_groups = nb4.attach_activity_groups(
            weekly_rgi_panel,
            rgi_activity_summary,
        )
        group_diagnostics = nb4.build_group_diagnostics(
            weekly_rgi_panel_with_groups,
            rgi_activity_summary,
        )
        local_datasets, group_datasets, group_rgi_vocabulary = (
            _prepare_activity_group_datasets(
                weekly_rgi_panel_with_groups,
                self.featurization_config,
            )
        )

        return {
            "preparation_summary": preparation_summary,
            "dynamic_exclusion_diagnostics": exclusion_diagnostics,
            "rgi_static_features": rgi_static_features,
            "modeling_accidents": modeling_accidents,
            "weekly_calendar": weekly_calendar,
            "weekly_rgi_target": weekly_rgi_target,
            "weekly_holiday_features": weekly_holiday_features,
            "weekly_rgi_panel": weekly_rgi_panel,
            "weekly_rgi_panel_with_groups": weekly_rgi_panel_with_groups,
            "rgi_activity_summary": rgi_activity_summary,
            "group_diagnostics": group_diagnostics,
            "local_datasets": local_datasets,
            "group_datasets": group_datasets,
            "local_sequence_manifest": _build_local_sequence_manifest(local_datasets),
            "group_sequence_manifest": _build_group_sequence_manifest(group_datasets),
            "group_rgi_vocabulary": group_rgi_vocabulary,
        }

    def load(self, data: dict[str, Any]) -> dict[str, Any]:
        staging_output_dir = self._staging_dir / "activity_group_features"
        staging_output_dir.mkdir(parents=True, exist_ok=True)

        parquet_outputs = {
            "preparation_summary.parquet": data["preparation_summary"],
            "dynamic_exclusion_diagnostics.parquet": data[
                "dynamic_exclusion_diagnostics"
            ],
            "rgi_static_features.parquet": data["rgi_static_features"],
            "modeling_accidents.parquet": data["modeling_accidents"],
            "weekly_calendar.parquet": data["weekly_calendar"],
            "weekly_rgi_target.parquet": data["weekly_rgi_target"],
            "weekly_holiday_features.parquet": data["weekly_holiday_features"],
            "weekly_rgi_panel.parquet": data["weekly_rgi_panel"],
            "weekly_rgi_panel_with_groups.parquet": data[
                "weekly_rgi_panel_with_groups"
            ],
            "rgi_activity_summary.parquet": data["rgi_activity_summary"],
            "group_diagnostics.parquet": data["group_diagnostics"],
            "local_sequence_manifest.parquet": data["local_sequence_manifest"],
            "group_sequence_manifest.parquet": data["group_sequence_manifest"],
            "group_rgi_vocabulary.parquet": data["group_rgi_vocabulary"],
        }
        for relative_path, frame in parquet_outputs.items():
            frame.to_parquet(staging_output_dir / relative_path, index=False)

        bundles_root = staging_output_dir / "group_sequence_bundles"
        for group_name, group_dataset in data["group_datasets"].items():
            group_root = bundles_root / group_name
            for split in ("train", "validation", "test"):
                _save_bundle(group_dataset[split], group_root, split)

        output_uri = self.datalake.persist_directory(
            staging_output_dir,
            self.featurization_config.output_subpath,
        )
        return {**data, "output_uri": output_uri}

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


class ActivityGroupRegressionExperiment(BaseMLflowRegressionExperiment):
    """Train and evaluate one activity-group model per configured architecture."""

    def __init__(
        self,
        *,
        experiment_config: ActivityGroupExperimentConfig | None = None,
        feature_config: ActivityGroupFeaturizationConfig | None = None,
        datalake_config: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_config = (
            experiment_config or ActivityGroupExperimentConfig.from_project_root()
        )
        self.feature_run = ActivityGroupFeaturizationRun(
            config=feature_config,
            datalake_config=datalake_config,
        )
        self.sequence_config = self.feature_run.featurization_config.sequence_config
        self.datalake = DatalakeAdapter.from_env(
            project_root=self.experiment_config.project_root,
            config=datalake_config,
        )
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)
        self._materialized_output_dir: Path | None = None
        super().__init__(
            config=asdict(self.experiment_config),
            job_name="activity_group_regression",
            tracking_uri=self.experiment_config.tracking_uri,
            experiment_name=self.experiment_config.experiment_name,
            run_name=self.experiment_config.run_name,
        )

    def validate_config(self) -> None:
        super().validate_config()
        unsupported = set(self.experiment_config.architectures) - {"gru", "lstm"}
        if unsupported:
            raise ValueError(
                f"Unsupported architectures requested: {sorted(unsupported)}"
            )
        if not self.experiment_config.architectures:
            raise ValueError("At least one architecture must be configured")
        if self.experiment_config.model_configs is None:
            raise ValueError(
                "model_configs must be loaded for activity-group experiments"
            )

    def _variant_run_name(self, group_name: str, architecture: str) -> str:
        if self.experiment_config.run_name:
            return f"{self.experiment_config.run_name}-{group_name}-{architecture}"
        return f"{group_name}-{architecture}"

    def _variant_relative_dir(self, group_name: str, architecture: str) -> Path:
        return Path("runs") / group_name / architecture

    def _variant_output_dir(self, group_name: str, architecture: str) -> Path:
        output_dir = (
            self._staging_dir
            / "activity_group_regression"
            / self._variant_relative_dir(group_name, architecture)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _materialize_variant_output(
        self,
        *,
        group_name: str,
        architecture: str,
        model_config: ActivityGroupModelConfig,
        training_run: dict[str, Any],
        forecasts: pd.DataFrame,
        split_metrics: pd.DataFrame,
        group_dataset: dict[str, Any],
        group_rgi_vocabulary: pd.DataFrame,
        run_id: str | None,
    ) -> Path:
        output_dir = self._variant_output_dir(group_name, architecture)

        _save_keras_model(training_run["model"], output_dir / "model.keras")
        _write_model_summary(training_run["model"], output_dir / "model_summary.txt")

        pd.DataFrame(
            {
                "epoch": range(
                    1,
                    max(
                        (len(values) for values in training_run["history"].values()),
                        default=0,
                    )
                    + 1,
                ),
                **training_run["history"],
            }
        ).to_parquet(output_dir / "training_history.parquet", index=False)
        forecasts.to_parquet(output_dir / "forecasts.parquet", index=False)
        split_metrics.to_parquet(output_dir / "split_metrics.parquet", index=False)

        plots_dir = output_dir / "plots"
        _write_training_curve_plot(
            training_run["history"],
            plots_dir / "training_curves.png",
        )
        _write_split_timeline_plot(
            forecasts,
            split_metrics,
            plots_dir / "forecast_timeline.png",
        )
        _write_prediction_parity_plot(
            forecasts,
            plots_dir / "prediction_parity.png",
        )

        feature_store_root = output_dir / "feature_store"
        feature_store_root.mkdir(parents=True, exist_ok=True)
        feature_frame = _build_feature_store_frame(
            {group_name: group_dataset},
            model_config,
            n_total_rows=self.feature_run.featurization_config.feature_store_max_rows,
        )
        registry = _build_feature_registry(feature_frame)
        profile = _build_feature_profile(feature_frame, registry)
        feature_frame.to_parquet(
            feature_store_root / "feature_matrix.parquet", index=False
        )
        registry.to_parquet(
            feature_store_root / "feature_registry.parquet", index=False
        )
        profile.to_parquet(feature_store_root / "feature_profile.parquet", index=False)
        group_rgi_vocabulary.to_parquet(
            feature_store_root / "group_rgi_vocabulary.parquet",
            index=False,
        )

        _write_json(
            output_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "experiment_name": self.experiment_config.experiment_name,
                "group_name": group_name,
                "architecture": architecture,
                "tracking_uri": self.experiment_config.tracking_uri,
                "feature_output_subpath": self.feature_run.featurization_config.output_subpath,
                "model_output_subpath": str(
                    Path(self.experiment_config.output_subpath)
                    / self._variant_relative_dir(group_name, architecture)
                ),
                "hyperparameters": {
                    key: value
                    for key, value in asdict(model_config).items()
                    if key != "config_path"
                },
            },
        )

        config_dir = output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        if self.feature_run.featurization_config.config_path is not None:
            (config_dir / "featurization.yaml").write_text(
                self.feature_run.featurization_config.config_path.read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
        if self.experiment_config.config_path is not None:
            (config_dir / "experiment.yaml").write_text(
                self.experiment_config.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if model_config.config_path is not None:
            (config_dir / f"{architecture}.yaml").write_text(
                model_config.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        return output_dir

    def _finalize_experiment_output(
        self,
        *,
        feature_data: dict[str, Any],
        variant_results: list[dict[str, Any]],
    ) -> Path:
        output_dir = self._staging_dir / "activity_group_regression"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_manifest = pd.DataFrame(
            [
                {
                    "group_name": result["group_name"],
                    "architecture": result["architecture"],
                    "run_id": result["run_id"],
                    "experiment_id": result["experiment_id"],
                    "relative_artifact_dir": str(
                        self._variant_relative_dir(
                            result["group_name"], result["architecture"]
                        )
                    ),
                    "n_predictions_train": int(
                        result["split_metrics"]
                        .loc[lambda frame: frame["split"].eq("train"), "n_predictions"]
                        .sum()
                    ),
                    "n_predictions_validation": int(
                        result["split_metrics"]
                        .loc[
                            lambda frame: frame["split"].eq("validation"),
                            "n_predictions",
                        ]
                        .sum()
                    ),
                    "n_predictions_test": int(
                        result["split_metrics"]
                        .loc[lambda frame: frame["split"].eq("test"), "n_predictions"]
                        .sum()
                    ),
                }
                for result in variant_results
            ]
        )
        split_metric_summary = pd.concat(
            [
                result["split_metrics"].assign(
                    group_name=result["group_name"],
                    architecture=result["architecture"],
                )
                for result in variant_results
            ],
            ignore_index=True,
        )
        run_manifest.to_parquet(output_dir / "run_manifest.parquet", index=False)
        split_metric_summary.to_parquet(
            output_dir / "variant_split_metrics.parquet", index=False
        )
        feature_data["group_sequence_manifest"].to_parquet(
            output_dir / "group_sequence_manifest.parquet", index=False
        )
        _write_json(
            output_dir / "experiment_manifest.json",
            {
                "experiment_name": self.experiment_config.experiment_name,
                "tracking_uri": self.experiment_config.tracking_uri,
                "feature_output_uri": feature_data["output_uri"],
                "variant_count": len(variant_results),
            },
        )
        return output_dir

    def run(self) -> dict[str, Any]:
        run_start = perf_counter()
        mlflow = _import_mlflow()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.logger.info("Starting regression experiment %s", self.job_name)

        try:
            raw_data = _run_phase(self.logger, "extract phase", self.extract)
            feature_data = _run_phase(
                self.logger,
                "featurize phase",
                lambda: self.featurize(raw_data),
            )

            variant_results: list[dict[str, Any]] = []
            group_datasets = feature_data["group_datasets"]
            for group_name in nb4.GROUP_ORDER:
                dataset = group_datasets.get(group_name)
                if dataset is None:
                    continue
                n_group_rgis = int(dataset["meta"]["n_rgis"])
                group_rgi_vocabulary = (
                    feature_data["group_rgi_vocabulary"]
                    .loc[
                        feature_data["group_rgi_vocabulary"]["group_name"].eq(
                            group_name
                        )
                    ]
                    .reset_index(drop=True)
                )

                for architecture in self.experiment_config.architectures:
                    model_config = self.experiment_config.model_configs[architecture]
                    training_run = _run_phase(
                        self.logger,
                        f"train phase [{group_name}/{architecture}]",
                        lambda group_dataset=dataset, current_model_config=model_config: (
                            _train_activity_group_model(
                                group_dataset["train"],
                                group_dataset["validation"],
                                sequence_config=current_model_config.apply_to(
                                    self.sequence_config
                                ),
                                feature_config=self.feature_run.featurization_config,
                                model_config=current_model_config,
                                architecture=architecture,
                                n_rgis=n_group_rgis,
                                show_progress=self.experiment_config.show_progress,
                            )
                        ),
                    )

                    forecasts = nb4.combine_forecasts(
                        _generate_activity_group_forecasts(
                            training_run["model"],
                            dataset["train"],
                            model_config=model_config,
                            architecture=architecture,
                            n_rgis=n_group_rgis,
                        ),
                        _generate_activity_group_forecasts(
                            training_run["model"],
                            dataset["validation"],
                            model_config=model_config,
                            architecture=architecture,
                            n_rgis=n_group_rgis,
                        ),
                        _generate_activity_group_forecasts(
                            training_run["model"],
                            dataset["test"],
                            model_config=model_config,
                            architecture=architecture,
                            n_rgis=n_group_rgis,
                        ),
                    )
                    split_metrics = _summarize_split_metrics(forecasts)

                    with contextlib.redirect_stdout(io.StringIO()):
                        with mlflow.start_run(
                            run_name=self._variant_run_name(group_name, architecture)
                        ) as active_run:
                            run_id = active_run.info.run_id
                            experiment_id = getattr(
                                active_run.info, "experiment_id", None
                            )
                            mlflow.set_tags(
                                _normalize_mlflow_params(
                                    {
                                        "job_name": self.job_name,
                                        "experiment_type": "regression_variant",
                                        "model_family": "activity_group_rnn",
                                        "group_name": group_name,
                                        "architecture": architecture,
                                    }
                                )
                            )
                            mlflow.log_params(
                                _normalize_mlflow_params(
                                    {
                                        "gold_input_subpath": self.feature_run.featurization_config.gold_input_subpath,
                                        "feature_output_subpath": self.feature_run.featurization_config.output_subpath,
                                        "recent_history_weeks": self.feature_run.featurization_config.recent_history_weeks,
                                        "forecast_horizon_weeks": self.sequence_config.forecast_horizon_weeks,
                                        "latency_gap_weeks": self.sequence_config.latency_gap_weeks,
                                        "seasonal_lag_weeks": ",".join(
                                            str(value)
                                            for value in self.feature_run.featurization_config.seasonal_lag_weeks
                                        ),
                                        "n_group_rgis": n_group_rgis,
                                        **{
                                            key: value
                                            for key, value in asdict(model_config).items()
                                            if key != "config_path"
                                        },
                                    }
                                )
                            )
                            mlflow.log_metrics(
                                _normalize_mlflow_metrics(
                                    {
                                        f"rmse_{row.split}": row.rmse
                                        for row in split_metrics.itertuples(index=False)
                                    }
                                    | {
                                        f"r2_{row.split}": row.r2
                                        for row in split_metrics.itertuples(index=False)
                                    }
                                )
                            )

                            variant_output_dir = self._materialize_variant_output(
                                group_name=group_name,
                                architecture=architecture,
                                model_config=model_config,
                                training_run=training_run,
                                forecasts=forecasts,
                                split_metrics=split_metrics,
                                group_dataset=dataset,
                                group_rgi_vocabulary=group_rgi_vocabulary,
                                run_id=run_id,
                            )
                            mlflow.log_artifacts(str(variant_output_dir))

                    variant_results.append(
                        {
                            "group_name": group_name,
                            "architecture": architecture,
                            "run_id": run_id,
                            "experiment_id": experiment_id,
                            "split_metrics": split_metrics,
                            "artifact_dir": variant_output_dir,
                        }
                    )

            if not variant_results:
                raise ValueError(
                    "No training runs were produced from the activity-group datasets."
                )

            persisted_root = _run_phase(
                self.logger,
                "persist phase",
                lambda: self.datalake.persist_directory(
                    self._finalize_experiment_output(
                        feature_data=feature_data,
                        variant_results=variant_results,
                    ),
                    self.experiment_config.output_subpath,
                ),
            )
            self.logger.info(
                "Finished regression experiment %s in %.2fs",
                self.job_name,
                perf_counter() - run_start,
            )
            return {
                "tracking_uri": self.tracking_uri,
                "experiment_name": self.experiment_name,
                "persisted_output": persisted_root,
                "feature_data": feature_data,
                "variant_results": variant_results,
            }
        except Exception:
            self.logger.exception("Regression experiment %s failed", self.job_name)
            raise
        finally:
            _run_phase(self.logger, "cleanup phase", self.cleanup)

    def extract(self) -> dict[str, Any]:
        return {
            "feature_output_subpath": self.feature_run.featurization_config.output_subpath,
        }

    def featurize(self, data: dict[str, Any]) -> dict[str, Any]:
        _ = data
        return self.feature_run.run()

    def train(self, feature_data: dict[str, Any]) -> dict[str, Any]:
        training_runs: list[dict[str, Any]] = []
        group_datasets = feature_data["group_datasets"]

        for group_name in nb4.GROUP_ORDER:
            dataset = group_datasets.get(group_name)
            if dataset is None:
                continue
            n_group_rgis = int(dataset["meta"]["n_rgis"])
            for architecture in self.experiment_config.architectures:
                model_config = self.experiment_config.model_configs[architecture]
                training_runs.append(
                    _train_activity_group_model(
                        dataset["train"],
                        dataset["validation"],
                        sequence_config=model_config.apply_to(self.sequence_config),
                        feature_config=self.feature_run.featurization_config,
                        model_config=model_config,
                        architecture=architecture,
                        n_rgis=n_group_rgis,
                        show_progress=self.experiment_config.show_progress,
                    )
                )

        if not training_runs:
            raise ValueError(
                "No training runs were produced from the activity-group datasets."
            )

        return {"training_runs": training_runs}

    def evaluate(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
    ) -> dict[str, Any]:
        forecast_frames: list[pd.DataFrame] = []
        for run in training_output["training_runs"]:
            group_name = str(run["group_name"])
            group_dataset = feature_data["group_datasets"][group_name]
            n_group_rgis = int(group_dataset["meta"]["n_rgis"])
            model_config = self.experiment_config.model_configs[
                str(run["architecture"])
            ]
            for split in ("validation", "test"):
                forecast_frames.append(
                    _generate_activity_group_forecasts(
                        run["model"],
                        group_dataset[split],
                        model_config=model_config,
                        architecture=str(run["architecture"]),
                        n_rgis=n_group_rgis,
                    )
                )

        forecasts = nb4.combine_forecasts(*forecast_frames)
        (
            benchmark_metrics,
            group_level_metrics,
            rgi_level_metrics,
            horizon_metrics,
        ) = nb4.evaluate_forecasts(
            feature_data["weekly_rgi_panel_with_groups"],
            forecasts,
            self.sequence_config,
        )
        return {
            "forecasts": forecasts,
            "benchmark_metrics": benchmark_metrics,
            "group_level_metrics": group_level_metrics,
            "rgi_level_metrics": rgi_level_metrics,
            "horizon_metrics": horizon_metrics,
            "training_histories": nb4.flatten_training_histories(
                training_output["training_runs"]
            ),
            "model_run_summary": nb4.build_model_run_summary(
                training_output["training_runs"]
            ),
            "rgi_metric_report": nb4.build_rgi_metric_report(rgi_level_metrics),
        }

    def mlflow_tags(self) -> dict[str, Any]:
        return {
            "model_family": "activity_group_rnn",
            "feature_source": "gold/br101_rgi_weekly_panel",
            "feature_output_subpath": self.feature_run.featurization_config.output_subpath,
        }

    def mlflow_params(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
    ) -> dict[str, Any]:
        _ = training_output
        _ = evaluation_output
        return {
            "architectures": ",".join(self.experiment_config.architectures),
            "gold_input_subpath": self.feature_run.featurization_config.gold_input_subpath,
            "feature_output_subpath": self.feature_run.featurization_config.output_subpath,
            "recent_history_weeks": self.feature_run.featurization_config.recent_history_weeks,
            "forecast_horizon_weeks": self.sequence_config.forecast_horizon_weeks,
            "latency_gap_weeks": self.sequence_config.latency_gap_weeks,
            "seasonal_lag_weeks": ",".join(
                str(value)
                for value in self.feature_run.featurization_config.seasonal_lag_weeks
            ),
            "n_rgis": int(
                feature_data["weekly_rgi_panel_with_groups"]["rgi_id"].nunique()
            ),
            "n_groups": int(
                feature_data["weekly_rgi_panel_with_groups"]["group_name"].nunique()
            ),
            **{
                f"{architecture}__{key}": value
                for architecture, model_config in self.experiment_config.model_configs.items()
                for key, value in asdict(model_config).items()
                if key != "config_path"
            },
        }

    def mlflow_metrics(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
    ) -> dict[str, Any]:
        _ = feature_data
        metrics = _normalize_overall_metrics(evaluation_output["benchmark_metrics"])
        for run in training_output["training_runs"]:
            group_name = str(run["group_name"])
            architecture = str(run["architecture"])
            metrics[f"best_val_loss__{group_name}__{architecture}"] = float(
                run["best_val_loss"]
            )
            metrics[f"runtime_seconds__{group_name}__{architecture}"] = float(
                run["runtime_seconds"]
            )
        return metrics

    def _materialize_output(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> Path:
        if self._materialized_output_dir is not None:
            return self._materialized_output_dir

        output_dir = self._staging_dir / "activity_group_regression"
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_outputs = {
            "benchmark_metrics.parquet": evaluation_output["benchmark_metrics"],
            "group_level_metrics.parquet": evaluation_output["group_level_metrics"],
            "rgi_level_metrics.parquet": evaluation_output["rgi_level_metrics"],
            "horizon_metrics.parquet": evaluation_output["horizon_metrics"],
            "rgi_metric_report.parquet": evaluation_output["rgi_metric_report"],
            "forecasts.parquet": evaluation_output["forecasts"],
            "training_histories.parquet": evaluation_output["training_histories"],
            "model_run_summary.parquet": evaluation_output["model_run_summary"],
            "group_sequence_manifest.parquet": feature_data["group_sequence_manifest"],
        }
        for relative_path, frame in parquet_outputs.items():
            frame.to_parquet(output_dir / relative_path, index=False)

        models_dir = output_dir / "models"
        for run in training_output["training_runs"]:
            model_name = f"{run['group_name']}__{run['architecture']}.keras"
            _save_keras_model(run["model"], models_dir / model_name)

        _persist_feature_store_views(
            output_dir,
            feature_data,
            self.experiment_config,
            self.feature_run.featurization_config,
        )

        _write_json(
            output_dir / "experiment_manifest.json",
            {
                "run_id": run_id,
                "tracking_uri": self.experiment_config.tracking_uri,
                "experiment_name": self.experiment_config.experiment_name,
                "architectures": list(self.experiment_config.architectures),
                "feature_output_uri": feature_data["output_uri"],
                "output_subpath": self.experiment_config.output_subpath,
                "featurization_config_path": self.feature_run.featurization_config.config_path,
                "experiment_config_path": self.experiment_config.config_path,
                "model_config_paths": {
                    architecture: model_config.config_path
                    for architecture, model_config in self.experiment_config.model_configs.items()
                },
            },
        )

        config_dir = output_dir / "config"
        if self.feature_run.featurization_config.config_path is not None:
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "featurization.yaml").write_text(
                self.feature_run.featurization_config.config_path.read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
        if self.experiment_config.config_path is not None:
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "experiment.yaml").write_text(
                self.experiment_config.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        model_config_dir = config_dir / "models"
        for architecture, model_config in self.experiment_config.model_configs.items():
            model_config_dir.mkdir(parents=True, exist_ok=True)
            (model_config_dir / f"{architecture}.yaml").write_text(
                model_config.config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        self._materialized_output_dir = output_dir
        return output_dir

    def log_mlflow_artifacts(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
    ) -> None:
        mlflow = _import_mlflow()
        output_dir = self._materialize_output(
            feature_data,
            training_output,
            evaluation_output,
        )
        mlflow.log_artifacts(str(output_dir))

    def persist(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
        *,
        run_id: str,
    ) -> str:
        output_dir = self._materialize_output(
            feature_data,
            training_output,
            evaluation_output,
            run_id=run_id,
        )
        return self.datalake.persist_directory(
            output_dir,
            self.experiment_config.output_subpath,
        )

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Activity-group featurization and regression training entrypoints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "featurize", help="Build activity-group features from gold artefacts."
    )
    subparsers.add_parser(
        "train", help="Train activity-group models with MLflow tracking."
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "featurize":
        featurizer = ActivityGroupFeaturizationRun()
        result = featurizer.run()
        print(result["output_uri"])
        return

    if args.command == "train":
        experiment = ActivityGroupRegressionExperiment()
        result = experiment.run()
        _print_training_summary(result)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
