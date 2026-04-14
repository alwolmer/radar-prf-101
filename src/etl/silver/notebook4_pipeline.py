from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.keras import TqdmCallback
except Exception:  # pragma: no cover - optional integration
    TqdmCallback = None

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    tf = None
    keras = None

from src.etl.silver.notebook3_pipeline import (
    HOLIDAY_FEATURE_COLUMNS,
    iso_week_start,
    resolve_project_root,
    validate_required_columns,
)
from src.etl.silver.notebook3_pipeline import (
    assign_split_labels as notebook3_assign_split_labels,
)

GROUP_ORDER = ["low_activity", "mid_activity", "high_activity"]

FORECAST_COLUMNS = [
    "model",
    "architecture",
    "group_name",
    "split",
    "rgi_id",
    "rgi_name",
    "uf",
    "week_start",
    "week_end",
    "year_week",
    "forecast_batch_start",
    "cutoff_week_start",
    "horizon_step",
    "prediction",
    "actual",
    "is_available",
    "metadata",
]

SEQUENCE_META_COLUMNS = [
    "group_name",
    "rgi_id",
    "rgi_name",
    "uf",
    "forecast_batch_start",
    "cutoff_week_start",
    "target_week_starts",
    "target_week_ends",
    "target_year_weeks",
]


@dataclass(frozen=True)
class Notebook4Config:
    project_root: Path
    notebook3_silver_dir: Path
    notebook3_gold_dir: Path
    silver_output_dir: Path
    gold_output_dir: Path
    train_start_year: int = 2017
    train_start_week: int = 1
    train_end_year: int = 2023
    train_end_week: int = 52
    validation_start_year: int = 2024
    validation_start_week: int = 1
    validation_end_year: int = 2024
    validation_end_week: int = 52
    test_start_year: int = 2025
    test_start_week: int = 1
    test_end_year: int = 2025
    test_end_week: int = 52
    lookback_weeks: int = 52
    latency_gap_weeks: int = 4
    forecast_horizon_weeks: int = 8
    rnn_train_stride_weeks: int = 1
    rnn_eval_stride_weeks: int = 4
    low_activity_share: float = 0.20
    high_activity_share: float = 0.20
    recurrent_units: int = 32
    dense_units_first: int = 64
    dense_units_second: int = 32
    random_seed: int = 101
    max_epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 5

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> Notebook4Config:
        root = project_root or resolve_project_root()
        return cls(
            project_root=root,
            notebook3_silver_dir=root / "data" / "silver" / "notebook3_accident_count",
            notebook3_gold_dir=root / "data" / "gold" / "notebook3_accident_count",
            silver_output_dir=root
            / "data"
            / "silver"
            / "notebook4_local_rnn_accident_count",
            gold_output_dir=root
            / "data"
            / "gold"
            / "notebook4_local_rnn_accident_count",
        )

    @property
    def train_start(self) -> pd.Timestamp:
        return iso_week_start(self.train_start_year, self.train_start_week)

    @property
    def train_end(self) -> pd.Timestamp:
        return iso_week_start(self.train_end_year, self.train_end_week)

    @property
    def validation_start(self) -> pd.Timestamp:
        return iso_week_start(self.validation_start_year, self.validation_start_week)

    @property
    def validation_end(self) -> pd.Timestamp:
        return iso_week_start(self.validation_end_year, self.validation_end_week)

    @property
    def test_start(self) -> pd.Timestamp:
        return iso_week_start(self.test_start_year, self.test_start_week)

    @property
    def test_end(self) -> pd.Timestamp:
        return iso_week_start(self.test_end_year, self.test_end_week)

    def ensure_output_dirs(self) -> None:
        self.silver_output_dir.mkdir(parents=True, exist_ok=True)
        self.gold_output_dir.mkdir(parents=True, exist_ok=True)


def tensorflow_status_message() -> str:
    if tf is not None:
        return f"TensorFlow runtime available: {tf.__version__}"
    return (
        "TensorFlow is not installed in the current environment. "
        "Install it before running the Notebook 4 training cells."
    )


def require_tensorflow() -> None:
    if tf is None or keras is None:
        raise ModuleNotFoundError(tensorflow_status_message())


def _import_plotting() -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def _required_panel_columns() -> list[str]:
    return [
        "rgi_id",
        "rgi_name",
        "uf",
        "week_start",
        "week_end",
        "year_week",
        "accident_count",
        "br101_length_km_in_rgi",
        *HOLIDAY_FEATURE_COLUMNS,
    ]


def load_notebook4_inputs(config: Notebook4Config) -> dict[str, pd.DataFrame]:
    config.ensure_output_dirs()
    panel_path = config.notebook3_silver_dir / "weekly_rgi_panel.parquet"
    benchmark_path = config.notebook3_gold_dir / "benchmark_metrics.parquet"
    static_path = config.notebook3_silver_dir / "rgi_static_features.parquet"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing notebook 3 weekly panel artifact at {panel_path}."
        )

    weekly_rgi_panel = pd.read_parquet(panel_path)
    validate_required_columns(
        weekly_rgi_panel, _required_panel_columns(), "weekly_rgi_panel"
    )

    panel = weekly_rgi_panel.copy()
    panel["rgi_id"] = panel["rgi_id"].astype("string")
    panel["rgi_name"] = panel["rgi_name"].astype("string")
    panel["uf"] = panel["uf"].astype("string").str.upper().str.strip()
    panel["week_start"] = pd.to_datetime(panel["week_start"], errors="coerce")
    panel["week_end"] = pd.to_datetime(panel["week_end"], errors="coerce")
    panel["year_week"] = panel["year_week"].astype("string")
    panel["accident_count"] = (
        pd.to_numeric(panel["accident_count"], errors="coerce").fillna(0).astype(int)
    )
    panel["br101_length_km_in_rgi"] = pd.to_numeric(
        panel["br101_length_km_in_rgi"], errors="coerce"
    ).astype(float)
    for column in HOLIDAY_FEATURE_COLUMNS:
        panel[column] = (
            pd.to_numeric(panel[column], errors="coerce").fillna(0.0).astype(float)
        )

    panel_with_splits = notebook3_assign_split_labels(panel, config)
    panel_with_splits = panel_with_splits.sort_values(
        ["rgi_id", "week_start"]
    ).reset_index(drop=True)

    notebook3_benchmark_metrics = (
        pd.read_parquet(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    )
    rgi_static_features = (
        pd.read_parquet(static_path) if static_path.exists() else pd.DataFrame()
    )
    return {
        "weekly_rgi_panel": panel_with_splits,
        "notebook3_benchmark_metrics": notebook3_benchmark_metrics,
        "rgi_static_features": rgi_static_features,
    }


def assign_split_labels(panel: pd.DataFrame, config: Notebook4Config) -> pd.DataFrame:
    return notebook3_assign_split_labels(panel, config)


def build_rgi_activity_summary(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    train_panel = panel_with_splits.loc[panel_with_splits["split"].eq("train")].copy()
    if train_panel.empty:
        raise ValueError("No train rows were found in the weekly panel.")

    def mean_positive(series: pd.Series) -> float:
        positive = series.loc[series.gt(0)]
        return float(positive.mean()) if not positive.empty else 0.0

    summary = (
        train_panel.groupby(["rgi_id", "rgi_name", "uf"], as_index=False)
        .agg(
            train_total_accidents=("accident_count", "sum"),
            train_mean_weekly_accidents=("accident_count", "mean"),
            train_median_weekly_accidents=("accident_count", "median"),
            train_active_week_share=("accident_count", lambda s: float(s.gt(0).mean())),
            train_zero_week_share=("accident_count", lambda s: float(s.eq(0).mean())),
            train_mean_positive_weekly_accidents=("accident_count", mean_positive),
            br101_length_km_in_rgi=("br101_length_km_in_rgi", "first"),
        )
        .sort_values(
            [
                "train_mean_weekly_accidents",
                "train_active_week_share",
                "train_mean_positive_weekly_accidents",
                "rgi_id",
            ],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    summary["activity_rank"] = np.arange(1, len(summary) + 1, dtype=int)
    return summary


def assign_activity_groups(
    activity_summary: pd.DataFrame,
    config: Notebook4Config,
) -> pd.DataFrame:
    summary = (
        activity_summary.copy()
        .sort_values(
            [
                "train_mean_weekly_accidents",
                "train_active_week_share",
                "train_mean_positive_weekly_accidents",
                "rgi_id",
            ],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    n_rgis = len(summary)
    if n_rgis < 3:
        raise ValueError(
            "Notebook 4 requires at least three RGIs to form activity groups."
        )

    low_count = max(1, int(np.floor(n_rgis * config.low_activity_share)))
    high_count = max(1, int(np.floor(n_rgis * config.high_activity_share)))
    remaining = n_rgis - low_count - high_count
    if remaining < 1:
        high_count = max(1, high_count - (1 - remaining))
        remaining = n_rgis - low_count - high_count
    if remaining < 1:
        low_count = max(1, low_count - (1 - remaining))

    summary["group_name"] = "mid_activity"
    summary.loc[summary.index < low_count, "group_name"] = "low_activity"
    summary.loc[summary.index >= n_rgis - high_count, "group_name"] = "high_activity"
    summary["group_order"] = summary["group_name"].map(
        {name: idx for idx, name in enumerate(GROUP_ORDER)}
    )
    return summary


def attach_activity_groups(
    panel_with_splits: pd.DataFrame,
    activity_summary_with_groups: pd.DataFrame,
) -> pd.DataFrame:
    merged = panel_with_splits.merge(
        activity_summary_with_groups[
            [
                "rgi_id",
                "group_name",
                "group_order",
                "activity_rank",
                "train_total_accidents",
                "train_mean_weekly_accidents",
                "train_median_weekly_accidents",
                "train_active_week_share",
                "train_zero_week_share",
                "train_mean_positive_weekly_accidents",
            ]
        ],
        on="rgi_id",
        how="left",
    )
    if merged["group_name"].isna().any():
        missing = (
            merged.loc[merged["group_name"].isna(), "rgi_id"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Missing activity-group assignments for RGIs: {missing[:5]}")
    return merged.sort_values(["group_order", "rgi_id", "week_start"]).reset_index(
        drop=True
    )


def build_group_diagnostics(
    panel_with_groups: pd.DataFrame,
    activity_summary_with_groups: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        summary_group = activity_summary_with_groups.loc[
            activity_summary_with_groups["group_name"].eq(group_name)
        ].copy()
        if summary_group.empty:
            continue
        panel_group = panel_with_groups.loc[
            panel_with_groups["group_name"].eq(group_name)
        ].copy()
        train_panel_group = panel_group.loc[panel_group["split"].eq("train")].copy()

        diagnostics.extend(
            [
                {
                    "diagnostic_group": "membership",
                    "group_name": group_name,
                    "diagnostic_name": "n_rgis",
                    "split": "overall",
                    "value": float(summary_group["rgi_id"].nunique()),
                },
                {
                    "diagnostic_group": "train_activity",
                    "group_name": group_name,
                    "diagnostic_name": "mean_weekly_accidents_mean",
                    "split": "train",
                    "value": float(summary_group["train_mean_weekly_accidents"].mean()),
                },
                {
                    "diagnostic_group": "train_activity",
                    "group_name": group_name,
                    "diagnostic_name": "zero_week_share_mean",
                    "split": "train",
                    "value": float(summary_group["train_zero_week_share"].mean()),
                },
                {
                    "diagnostic_group": "train_panel",
                    "group_name": group_name,
                    "diagnostic_name": "rows",
                    "split": "train",
                    "value": float(len(train_panel_group)),
                },
            ]
        )
    return pd.DataFrame(diagnostics)


def plot_group_diagnostics(
    panel_with_groups: pd.DataFrame,
    activity_summary_with_groups: pd.DataFrame,
) -> None:
    plt, sns = _import_plotting()
    ordered_groups = [
        name
        for name in GROUP_ORDER
        if name in activity_summary_with_groups["group_name"].astype(str).tolist()
    ]
    if not ordered_groups:
        return

    plt.figure(figsize=(14, 4))
    sns.histplot(
        data=activity_summary_with_groups,
        x="train_mean_weekly_accidents",
        hue="group_name",
        hue_order=ordered_groups,
        bins=20,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.title("Train mean weekly accidents by activity group")
    plt.xlabel("Train mean weekly accidents")
    plt.ylabel("RGI count")
    plt.tight_layout()
    plt.show()

    aggregate_by_group = (
        panel_with_groups.groupby(["group_name", "week_start"], as_index=False)[
            "accident_count"
        ]
        .sum()
        .sort_values(["group_name", "week_start"])
    )
    plt.figure(figsize=(16, 5))
    sns.lineplot(
        data=aggregate_by_group,
        x="week_start",
        y="accident_count",
        hue="group_name",
        hue_order=ordered_groups,
    )
    plt.title("Aggregate weekly accident counts by activity group")
    plt.xlabel("Week start")
    plt.ylabel("Accident count")
    plt.tight_layout()
    plt.show()


def _split_bounds(
    config: Notebook4Config, split: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return config.train_start, config.train_end
    if split == "validation":
        return config.validation_start, config.validation_end
    if split == "test":
        return config.test_start, config.test_end
    raise ValueError(f"Unsupported split {split!r}.")


def build_local_target_scale(rgi_panel: pd.DataFrame) -> float:
    train_panel = rgi_panel.loc[rgi_panel["split"].eq("train")].copy()
    mean_count = float(train_panel["accident_count"].mean())
    return float(max(1.0, 1.0 + mean_count))


def _is_aligned_batch_start(
    batch_start: pd.Timestamp,
    split_start: pd.Timestamp,
    stride_weeks: int,
) -> bool:
    week_delta = int((batch_start - split_start).days // 7)
    return week_delta % stride_weeks == 0


def build_local_sequence_dataset(
    rgi_panel: pd.DataFrame,
    config: Notebook4Config,
    split: str,
    *,
    target_scale: float,
    stride_weeks: int,
) -> dict[str, Any]:
    split_start, split_end = _split_bounds(config, split)
    group = rgi_panel.sort_values("week_start").reset_index(drop=True)
    counts = group["accident_count"].to_numpy(dtype=np.float32)
    weeks = group["week_start"].tolist()
    week_ends = group["week_end"].tolist()
    year_weeks = group["year_week"].tolist()
    holiday_values = group[HOLIDAY_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaled_counts = counts / target_scale
    scaled_length = float(group["br101_length_km_in_rgi"].iloc[0])
    scale_feature = float(np.log1p(target_scale))

    history_rows: list[np.ndarray] = []
    future_holiday_rows: list[np.ndarray] = []
    road_length_rows: list[list[float]] = []
    scale_feature_rows: list[list[float]] = []
    target_rows: list[np.ndarray] = []
    actual_target_rows: list[np.ndarray] = []
    target_scale_rows: list[list[float]] = []
    meta_rows: list[dict[str, Any]] = []

    for target_start_idx, batch_start in enumerate(weeks):
        batch_end_idx = target_start_idx + config.forecast_horizon_weeks - 1
        cutoff_idx = target_start_idx - config.latency_gap_weeks
        history_start_idx = cutoff_idx - config.lookback_weeks + 1

        if batch_start < split_start or batch_end_idx >= len(weeks):
            continue
        if weeks[batch_end_idx] > split_end or cutoff_idx < 0 or history_start_idx < 0:
            continue
        if not _is_aligned_batch_start(batch_start, split_start, stride_weeks):
            continue

        history_rows.append(
            scaled_counts[history_start_idx : cutoff_idx + 1].reshape(-1, 1)
        )
        future_holiday_rows.append(holiday_values[target_start_idx : batch_end_idx + 1])
        road_length_rows.append([scaled_length])
        scale_feature_rows.append([scale_feature])
        target_rows.append(scaled_counts[target_start_idx : batch_end_idx + 1])
        actual_target_rows.append(counts[target_start_idx : batch_end_idx + 1])
        target_scale_rows.append([target_scale])
        meta_rows.append(
            {
                "group_name": str(group["group_name"].iloc[0]),
                "rgi_id": str(group["rgi_id"].iloc[0]),
                "rgi_name": str(group["rgi_name"].iloc[0]),
                "uf": str(group["uf"].iloc[0]),
                "forecast_batch_start": batch_start,
                "cutoff_week_start": weeks[cutoff_idx],
                "target_week_starts": weeks[target_start_idx : batch_end_idx + 1],
                "target_week_ends": week_ends[target_start_idx : batch_end_idx + 1],
                "target_year_weeks": year_weeks[target_start_idx : batch_end_idx + 1],
            }
        )

    if not history_rows:
        return {
            "split": split,
            "group_name": str(group["group_name"].iloc[0]),
            "rgi_id": str(group["rgi_id"].iloc[0]),
            "inputs": {
                "history": np.empty((0, config.lookback_weeks, 1), dtype=np.float32),
                "future_holiday": np.empty(
                    (0, config.forecast_horizon_weeks, len(HOLIDAY_FEATURE_COLUMNS)),
                    dtype=np.float32,
                ),
                "road_length": np.empty((0, 1), dtype=np.float32),
                "scale_feature": np.empty((0, 1), dtype=np.float32),
            },
            "targets": np.empty((0, config.forecast_horizon_weeks), dtype=np.float32),
            "actual_targets": np.empty(
                (0, config.forecast_horizon_weeks), dtype=np.float32
            ),
            "target_scale": np.empty((0, 1), dtype=np.float32),
            "meta": pd.DataFrame(meta_rows, columns=SEQUENCE_META_COLUMNS),
        }

    return {
        "split": split,
        "group_name": str(group["group_name"].iloc[0]),
        "rgi_id": str(group["rgi_id"].iloc[0]),
        "inputs": {
            "history": np.stack(history_rows).astype(np.float32),
            "future_holiday": np.stack(future_holiday_rows).astype(np.float32),
            "road_length": np.asarray(road_length_rows, dtype=np.float32),
            "scale_feature": np.asarray(scale_feature_rows, dtype=np.float32),
        },
        "targets": np.stack(target_rows).astype(np.float32),
        "actual_targets": np.stack(actual_target_rows).astype(np.float32),
        "target_scale": np.asarray(target_scale_rows, dtype=np.float32),
        "meta": pd.DataFrame(meta_rows),
    }


def _empty_group_sequence_dataset(
    config: Notebook4Config,
    *,
    split: str,
    group_name: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "group_name": group_name,
        "inputs": {
            "history": np.empty((0, config.lookback_weeks, 1), dtype=np.float32),
            "future_holiday": np.empty(
                (0, config.forecast_horizon_weeks, len(HOLIDAY_FEATURE_COLUMNS)),
                dtype=np.float32,
            ),
            "road_length": np.empty((0, 1), dtype=np.float32),
            "scale_feature": np.empty((0, 1), dtype=np.float32),
        },
        "targets": np.empty((0, config.forecast_horizon_weeks), dtype=np.float32),
        "actual_targets": np.empty(
            (0, config.forecast_horizon_weeks), dtype=np.float32
        ),
        "target_scale": np.empty((0, 1), dtype=np.float32),
        "meta": pd.DataFrame(columns=SEQUENCE_META_COLUMNS),
    }


def combine_sequence_bundles(
    bundles: list[dict[str, Any]],
    config: Notebook4Config,
    *,
    split: str,
    group_name: str,
) -> dict[str, Any]:
    non_empty = [bundle for bundle in bundles if len(bundle["meta"]) > 0]
    if not non_empty:
        return _empty_group_sequence_dataset(
            config,
            split=split,
            group_name=group_name,
        )

    return {
        "split": split,
        "group_name": group_name,
        "inputs": {
            "history": np.concatenate(
                [bundle["inputs"]["history"] for bundle in non_empty], axis=0
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


def prepare_local_rnn_datasets(
    panel_with_groups: pd.DataFrame,
    config: Notebook4Config,
) -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for rgi_id, rgi_panel in panel_with_groups.groupby("rgi_id"):
        rgi_panel = rgi_panel.sort_values("week_start").reset_index(drop=True)
        target_scale = build_local_target_scale(rgi_panel)
        datasets[str(rgi_id)] = {
            "meta": {
                "rgi_id": str(rgi_id),
                "rgi_name": str(rgi_panel["rgi_name"].iloc[0]),
                "uf": str(rgi_panel["uf"].iloc[0]),
                "group_name": str(rgi_panel["group_name"].iloc[0]),
                "target_scale": target_scale,
            },
            "train": build_local_sequence_dataset(
                rgi_panel,
                config,
                split="train",
                target_scale=target_scale,
                stride_weeks=config.rnn_train_stride_weeks,
            ),
            "validation": build_local_sequence_dataset(
                rgi_panel,
                config,
                split="validation",
                target_scale=target_scale,
                stride_weeks=config.rnn_eval_stride_weeks,
            ),
            "test": build_local_sequence_dataset(
                rgi_panel,
                config,
                split="test",
                target_scale=target_scale,
                stride_weeks=config.rnn_eval_stride_weeks,
            ),
        }
    return datasets


def prepare_group_rnn_datasets(
    local_datasets: dict[str, dict[str, Any]],
    config: Notebook4Config,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in GROUP_ORDER}
    for dataset in local_datasets.values():
        group_name = str(dataset["meta"]["group_name"])
        grouped.setdefault(group_name, []).append(dataset)

    group_datasets: dict[str, dict[str, Any]] = {}
    for group_name in GROUP_ORDER:
        datasets_in_group = grouped.get(group_name, [])
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
            "train": combine_sequence_bundles(
                [dataset["train"] for dataset in datasets_in_group],
                config,
                split="train",
                group_name=group_name,
            ),
            "validation": combine_sequence_bundles(
                [dataset["validation"] for dataset in datasets_in_group],
                config,
                split="validation",
                group_name=group_name,
            ),
            "test": combine_sequence_bundles(
                [dataset["test"] for dataset in datasets_in_group],
                config,
                split="test",
                group_name=group_name,
            ),
        }
    return group_datasets


def build_sequence_manifest(local_datasets: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rgi_id, dataset in local_datasets.items():
        meta = dataset["meta"]
        for split in ["train", "validation", "test"]:
            bundle = dataset[split]
            rows.append(
                {
                    "rgi_id": rgi_id,
                    "rgi_name": meta["rgi_name"],
                    "uf": meta["uf"],
                    "group_name": meta["group_name"],
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


def build_group_sequence_manifest(
    group_datasets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, dataset in group_datasets.items():
        for split in ["train", "validation", "test"]:
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


def build_rnn_model(
    config: Notebook4Config,
    *,
    architecture: str,
) -> Any:
    require_tensorflow()
    if architecture not in {"gru", "lstm"}:
        raise ValueError(f"Unsupported architecture {architecture!r}.")

    history_input = keras.Input(shape=(config.lookback_weeks, 1), name="history")
    future_holiday_input = keras.Input(
        shape=(config.forecast_horizon_weeks, len(HOLIDAY_FEATURE_COLUMNS)),
        name="future_holiday",
    )
    road_length_input = keras.Input(shape=(1,), name="road_length")
    scale_feature_input = keras.Input(shape=(1,), name="scale_feature")

    recurrent_layer = keras.layers.GRU if architecture == "gru" else keras.layers.LSTM
    history_encoded = recurrent_layer(
        config.recurrent_units, name=f"history_{architecture}"
    )(history_input)
    future_holiday_flat = keras.layers.Flatten(name="future_holiday_flat")(
        future_holiday_input
    )
    x = keras.layers.Concatenate(name="model_features")(
        [history_encoded, future_holiday_flat, road_length_input, scale_feature_input]
    )
    x = keras.layers.Dense(config.dense_units_first, activation="relu", name="dense_1")(
        x
    )
    x = keras.layers.Dense(
        config.dense_units_second, activation="relu", name="dense_2"
    )(x)
    output = keras.layers.Dense(config.forecast_horizon_weeks, name="forecast")(x)

    model = keras.Model(
        inputs={
            "history": history_input,
            "future_holiday": future_holiday_input,
            "road_length": road_length_input,
            "scale_feature": scale_feature_input,
        },
        outputs=output,
        name=f"notebook4_activity_group_{architecture}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_group_model(
    train_bundle: dict[str, Any],
    validation_bundle: dict[str, Any],
    config: Notebook4Config,
    *,
    architecture: str,
    show_progress: bool = True,
) -> dict[str, Any]:
    require_tensorflow()
    if train_bundle["targets"].size == 0:
        raise ValueError(
            f"No train samples were found for group_name={train_bundle['group_name']!r}."
        )
    if validation_bundle["targets"].size == 0:
        raise ValueError(
            "No validation samples were found for "
            f"group_name={validation_bundle['group_name']!r}."
        )

    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_rnn_model(config, architecture=architecture)
    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    fit_verbose = 1 if show_progress else 0
    if show_progress and TqdmCallback is not None:
        callbacks.append(
            TqdmCallback(
                verbose=0,
                desc=(f"{train_bundle['group_name']} {architecture.upper()} training"),
                leave=True,
            )
        )
        fit_verbose = 0

    timer_start = perf_counter()
    history = model.fit(
        x=train_bundle["inputs"],
        y=train_bundle["targets"],
        validation_data=(validation_bundle["inputs"], validation_bundle["targets"]),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
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
    }


def bundle_predictions_to_forecast_frame(
    bundle: dict[str, Any],
    predictions_scaled: np.ndarray,
    *,
    architecture: str,
) -> pd.DataFrame:
    if bundle["targets"].size == 0:
        return empty_forecast_frame()

    # Accident counts are discrete, so forecasts are clipped at zero and rounded to integers.
    predictions = np.rint(
        np.maximum(predictions_scaled, 0.0) * bundle["target_scale"]
    ).astype(np.int32)
    rows: list[dict[str, Any]] = []
    for meta, prediction_row, actual_row, scale_row in zip(
        bundle["meta"].to_dict(orient="records"),
        predictions,
        bundle["actual_targets"],
        bundle["target_scale"],
        strict=True,
    ):
        for step, (week_start, week_end, year_week, prediction, actual) in enumerate(
            zip(
                meta["target_week_starts"],
                meta["target_week_ends"],
                meta["target_year_weeks"],
                prediction_row,
                actual_row,
                strict=True,
            ),
            start=1,
        ):
            rows.append(
                {
                    "model": f"activity_group_{architecture}",
                    "architecture": architecture,
                    "group_name": meta["group_name"],
                    "split": bundle["split"],
                    "rgi_id": meta["rgi_id"],
                    "rgi_name": meta["rgi_name"],
                    "uf": meta["uf"],
                    "week_start": week_start,
                    "week_end": week_end,
                    "year_week": year_week,
                    "forecast_batch_start": meta["forecast_batch_start"],
                    "cutoff_week_start": meta["cutoff_week_start"],
                    "horizon_step": step,
                    "prediction": int(prediction),
                    "actual": int(actual),
                    "is_available": True,
                    "metadata": f"target_scale={float(scale_row[0]):.4f}; rounded_to_int=1",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["architecture", "rgi_id", "forecast_batch_start", "horizon_step"])
        .reset_index(drop=True)
    )


def generate_rnn_forecasts(
    model: Any,
    bundle: dict[str, Any],
    *,
    architecture: str,
) -> pd.DataFrame:
    require_tensorflow()
    if bundle["targets"].size == 0:
        return empty_forecast_frame()
    predictions_scaled = model.predict(bundle["inputs"], verbose=0)
    return bundle_predictions_to_forecast_frame(
        bundle, predictions_scaled, architecture=architecture
    )


def empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FORECAST_COLUMNS)


def combine_forecasts(*forecast_frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [
        frame for frame in forecast_frames if frame is not None and not frame.empty
    ]
    if not non_empty:
        return empty_forecast_frame()
    return (
        pd.concat(non_empty, ignore_index=True)
        .sort_values(
            [
                "architecture",
                "split",
                "group_name",
                "rgi_id",
                "forecast_batch_start",
                "horizon_step",
            ]
        )
        .reset_index(drop=True)
    )


def _rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def _r2(actual: np.ndarray, prediction: np.ndarray) -> float:
    if actual.size < 2:
        return float("nan")
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((actual - prediction) ** 2))
    return 1.0 - (ss_res / ss_tot)


def _summarize_metric_frame(frame: pd.DataFrame) -> pd.Series:
    actual = frame["actual"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    return pd.Series(
        {
            "rmse": _rmse(actual, prediction),
            "r2": _r2(actual, prediction),
            "n_predictions": int(len(frame)),
        }
    )


def _expected_rows_per_rgi(config: Notebook4Config) -> dict[str, int]:
    split_lengths = {}
    for split, start, end in [
        ("validation", config.validation_start, config.validation_end),
        ("test", config.test_start, config.test_end),
    ]:
        starts = pd.date_range(
            start=start, end=end, freq=f"{config.rnn_eval_stride_weeks}W-MON"
        )
        valid_starts = [
            ts
            for ts in starts
            if ts + pd.Timedelta(weeks=config.forecast_horizon_weeks - 1) <= end
        ]
        split_lengths[split] = len(valid_starts) * config.forecast_horizon_weeks
    return split_lengths


def evaluate_forecasts(
    panel_with_groups: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: Notebook4Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = forecasts.loc[
        forecasts["prediction"].notna() & forecasts["actual"].notna()
    ].copy()
    if available.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    expected_per_rgi = _expected_rows_per_rgi(config)
    total_rgis = int(panel_with_groups["rgi_id"].nunique())
    group_rgi_counts = (
        panel_with_groups[["group_name", "rgi_id"]]
        .drop_duplicates()
        .groupby("group_name", as_index=False)
        .size()
        .rename(columns={"size": "n_group_rgis"})
    )

    rgi_level_metrics = (
        available.groupby(
            [
                "model",
                "architecture",
                "group_name",
                "split",
                "rgi_id",
                "rgi_name",
                "uf",
            ],
            as_index=False,
        )
        .apply(_summarize_metric_frame, include_groups=False)
        .reset_index(drop=True)
    )
    rgi_level_metrics["expected_rows"] = (
        rgi_level_metrics["split"].map(expected_per_rgi).astype(int)
    )
    rgi_level_metrics["forecast_row_coverage"] = (
        rgi_level_metrics["n_predictions"] / rgi_level_metrics["expected_rows"]
    )

    group_level_metrics = (
        available.groupby(
            ["model", "architecture", "group_name", "split"], as_index=False
        )
        .apply(_summarize_metric_frame, include_groups=False)
        .reset_index(drop=True)
    )
    group_macro = (
        rgi_level_metrics.groupby(
            ["model", "architecture", "group_name", "split"], as_index=False
        )["rmse"]
        .mean()
        .rename(columns={"rmse": "macro_rmse"})
    )
    group_prediction_counts = (
        available.groupby(
            ["model", "architecture", "group_name", "split"], as_index=False
        )["rgi_id"]
        .nunique()
        .rename(columns={"rgi_id": "n_rgis_with_predictions"})
    )
    group_level_metrics = (
        group_level_metrics.merge(
            group_macro,
            on=["model", "architecture", "group_name", "split"],
            how="left",
        )
        .merge(
            group_rgi_counts,
            on="group_name",
            how="left",
        )
        .merge(
            group_prediction_counts,
            on=["model", "architecture", "group_name", "split"],
            how="left",
        )
    )
    group_level_metrics["expected_rows"] = group_level_metrics["split"].map(
        expected_per_rgi
    ).astype(int) * group_level_metrics["n_group_rgis"].fillna(0).astype(int)
    group_level_metrics["forecast_row_coverage"] = (
        group_level_metrics["n_predictions"] / group_level_metrics["expected_rows"]
    )

    overall_metrics = (
        available.groupby(["model", "architecture", "split"], as_index=False)
        .apply(_summarize_metric_frame, include_groups=False)
        .reset_index(drop=True)
    )
    overall_macro = (
        rgi_level_metrics.groupby(["model", "architecture", "split"], as_index=False)[
            "rmse"
        ]
        .mean()
        .rename(columns={"rmse": "macro_rmse"})
    )
    overall_prediction_counts = available.groupby(
        ["model", "architecture", "split"], as_index=False
    ).agg(
        n_rgis_with_predictions=("rgi_id", "nunique"),
        n_groups_with_predictions=("group_name", "nunique"),
    )
    overall_metrics = overall_metrics.merge(
        overall_macro,
        on=["model", "architecture", "split"],
        how="left",
    ).merge(
        overall_prediction_counts,
        on=["model", "architecture", "split"],
        how="left",
    )
    overall_metrics["expected_rows"] = (
        overall_metrics["split"].map(expected_per_rgi).astype(int) * total_rgis
    )
    overall_metrics["forecast_row_coverage"] = (
        overall_metrics["n_predictions"] / overall_metrics["expected_rows"]
    )

    horizon_metrics = (
        available.groupby(
            ["model", "architecture", "group_name", "split", "horizon_step"],
            as_index=False,
        )
        .apply(_summarize_metric_frame, include_groups=False)
        .reset_index(drop=True)
    )

    return (
        overall_metrics.sort_values(["split", "rmse"]).reset_index(drop=True),
        group_level_metrics.sort_values(["split", "group_name", "rmse"]).reset_index(
            drop=True
        ),
        rgi_level_metrics.sort_values(["split", "architecture", "rmse"]).reset_index(
            drop=True
        ),
        horizon_metrics.sort_values(
            ["split", "group_name", "horizon_step"]
        ).reset_index(drop=True),
    )


def flatten_training_histories(training_runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in training_runs:
        history = run["history"]
        epochs = max((len(values) for values in history.values()), default=0)
        for epoch_idx in range(epochs):
            row = {
                "group_name": run["group_name"],
                "architecture": run["architecture"],
                "epoch": epoch_idx + 1,
            }
            for metric_name, values in history.items():
                row[metric_name] = (
                    values[epoch_idx] if epoch_idx < len(values) else np.nan
                )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["group_name", "architecture", "epoch"])
        .reset_index(drop=True)
    )


def build_model_run_summary(training_runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "group_name": run["group_name"],
            "architecture": run["architecture"],
            "best_epoch": run["best_epoch"],
            "best_val_loss": run["best_val_loss"],
            "runtime_seconds": run["runtime_seconds"],
            "n_train_samples": run["n_train_samples"],
            "n_validation_samples": run["n_validation_samples"],
            "n_train_rgis": run["n_train_rgis"],
            "n_validation_rgis": run["n_validation_rgis"],
        }
        for run in training_runs
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["group_name", "architecture"])
        .reset_index(drop=True)
    )


def build_rgi_metric_report(rgi_level_metrics: pd.DataFrame) -> pd.DataFrame:
    if rgi_level_metrics.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "group_name",
                "model",
                "architecture",
                "rgi_id",
                "rgi_name",
                "uf",
                "rmse",
                "r2",
                "n_predictions",
                "forecast_row_coverage",
            ]
        )

    columns = [
        "split",
        "group_name",
        "model",
        "architecture",
        "rgi_id",
        "rgi_name",
        "uf",
        "rmse",
        "r2",
        "n_predictions",
        "forecast_row_coverage",
    ]
    report = rgi_level_metrics.loc[:, columns].copy()
    return report.sort_values(
        ["split", "group_name", "architecture", "rgi_id"]
    ).reset_index(drop=True)


def summarize_horizon_rmse(
    forecasts: pd.DataFrame,
    *,
    split: str = "test",
    architecture: str | None = None,
    group_name: str | None = None,
    rgi_id: str | None = None,
) -> pd.DataFrame:
    filtered = forecasts.loc[forecasts["split"].eq(split)].copy()
    if architecture is not None:
        filtered = filtered.loc[filtered["architecture"].eq(architecture)].copy()
    if group_name is not None:
        filtered = filtered.loc[filtered["group_name"].eq(group_name)].copy()
    if rgi_id is not None:
        filtered = filtered.loc[filtered["rgi_id"].astype(str).eq(str(rgi_id))].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["horizon_step", "rmse", "n_predictions"])

    return (
        filtered.groupby("horizon_step", as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "rmse": _rmse(
                        frame["actual"].to_numpy(dtype=float),
                        frame["prediction"].to_numpy(dtype=float),
                    ),
                    "n_predictions": int(len(frame)),
                }
            ),
            include_groups=False,
        )
        .sort_values("horizon_step")
        .reset_index(drop=True)
    )


def plot_test_predictions_for_rgi(
    panel_with_groups: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    rgi_id: str,
    split: str = "test",
    figsize: tuple[int, int] = (16, 6),
) -> pd.DataFrame:
    plt, _ = _import_plotting()
    actuals = (
        panel_with_groups.loc[
            panel_with_groups["split"].eq(split)
            & panel_with_groups["rgi_id"].astype(str).eq(str(rgi_id))
        ]
        .sort_values("week_start")
        .copy()
    )
    if actuals.empty:
        raise ValueError(
            f"No panel rows found for rgi_id={rgi_id!r} and split={split!r}."
        )

    prediction_frame = forecasts.loc[
        forecasts["split"].eq(split) & forecasts["rgi_id"].astype(str).eq(str(rgi_id))
    ].copy()
    if prediction_frame.empty:
        raise ValueError(
            f"No forecast rows found for rgi_id={rgi_id!r} and split={split!r}."
        )

    models = prediction_frame["model"].dropna().astype(str).unique().tolist()
    rgi_name = str(actuals["rgi_name"].iloc[0])
    uf = str(actuals["uf"].iloc[0])

    plt.figure(figsize=figsize)
    plt.plot(
        actuals["week_start"],
        actuals["accident_count"],
        label="actual",
        color="black",
        linewidth=2.0,
    )
    for current_model in models:
        model_frame = (
            prediction_frame.loc[prediction_frame["model"].eq(current_model)]
            .sort_values(["week_start", "horizon_step"])
            .groupby("week_start", as_index=False)
            .first()
        )
        plt.plot(
            model_frame["week_start"],
            model_frame["prediction"],
            label=current_model,
            linewidth=1.6,
        )

    plt.title(
        f"{split.title()} actual vs activity-group-model predictions for "
        f"{rgi_name} ({uf}) [{rgi_id}]"
    )
    plt.xlabel("Week start")
    plt.ylabel("Accident count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    summaries: list[pd.DataFrame] = []
    for architecture in sorted(
        prediction_frame["architecture"].dropna().astype(str).unique().tolist()
    ):
        summary = summarize_horizon_rmse(
            prediction_frame,
            split=split,
            architecture=architecture,
            rgi_id=str(rgi_id),
        )
        summary.insert(0, "architecture", architecture)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
