from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm

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


FORECAST_COLUMNS = [
    "model",
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

HOLIDAY_FEATURE_COLUMNS = [
    "prev_week_non_work_day_share",
    "prev_week_longest_non_work_streak",
    "curr_week_non_work_day_share",
    "curr_week_longest_non_work_streak",
    "next_week_non_work_day_share",
    "next_week_longest_non_work_streak",
]


@dataclass(frozen=True)
class Notebook3Config:
    project_root: Path
    silver_input_dir: Path
    silver_output_dir: Path
    gold_output_dir: Path
    iso_start_year: int = 2017
    iso_start_week: int = 1
    iso_end_year: int = 2025
    iso_end_week: int = 52
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
    seasonal_period: int = 52
    moving_average_window: int = 4
    lookback_weeks: int = 52
    latency_gap_weeks: int = 4
    forecast_horizon_weeks: int = 4
    bridge_weight: float = 0.5
    rnn_train_stride_weeks: int = 1
    rnn_eval_stride_weeks: int = 4
    random_seed: int = 101
    sarima_maxiter: int = 50

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> Notebook3Config:
        root = project_root or resolve_project_root()
        return cls(
            project_root=root,
            silver_input_dir=root / "data" / "silver" / "eda_runbook",
            silver_output_dir=root / "data" / "silver" / "notebook3_accident_count",
            gold_output_dir=root / "data" / "gold" / "notebook3_accident_count",
        )

    @property
    def outer_window_start(self) -> pd.Timestamp:
        return iso_week_start(self.iso_start_year, self.iso_start_week)

    @property
    def outer_window_end(self) -> pd.Timestamp:
        return iso_week_start(self.iso_end_year, self.iso_end_week)

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


def resolve_project_root() -> Path:
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[3])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd.parent])

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError("Could not resolve the project root.")


def iso_week_start(year: int, week: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisocalendar(year, week, 1))


def year_week_label(week_start: pd.Timestamp) -> str:
    iso = week_start.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def normalize_code(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string")


def normalize_week_start(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    return ts.dt.to_period("W-SUN").dt.start_time


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str], frame_name: str
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def tensorflow_status_message() -> str:
    if tf is not None:
        return f"TensorFlow runtime available: {tf.__version__}"
    return (
        "TensorFlow is not installed in the current environment. "
        "Install it before running notebook cells 14-15."
    )


def require_tensorflow() -> None:
    if tf is None or keras is None:
        raise ModuleNotFoundError(tensorflow_status_message())


def build_rgi_static_features(config: Notebook3Config) -> pd.DataFrame:
    road_path = config.silver_input_dir / "road_sections_by_rgi.parquet"
    rgis_path = config.silver_input_dir / "rgis_in_scope.parquet"

    if not road_path.exists():
        raise FileNotFoundError(
            "Missing road-length artifact at "
            f"{road_path}. Notebook 3 requires BR-101 road length by RGI."
        )
    if not rgis_path.exists():
        raise FileNotFoundError(f"Missing retained-RGI artifact at {rgis_path}.")

    road_sections = pd.read_parquet(
        road_path,
        columns=[
            "CD_RGI",
            "NM_RGI",
            "CD_RGINT",
            "NM_RGINT",
            "SIGLA_UF",
            "road_length_m",
        ],
    )
    validate_required_columns(
        road_sections,
        ["CD_RGI", "NM_RGI", "SIGLA_UF", "road_length_m"],
        "road_sections_by_rgi",
    )

    rgis_in_scope = pd.read_parquet(
        rgis_path,
        columns=["CD_RGI", "NM_RGI", "CD_RGINT", "NM_RGINT", "SIGLA_UF"],
    )

    grouped = (
        road_sections.groupby(
            ["CD_RGI", "NM_RGI", "CD_RGINT", "NM_RGINT", "SIGLA_UF"],
            as_index=False,
            dropna=False,
        )["road_length_m"]
        .sum()
        .rename(columns={"road_length_m": "br101_length_m_in_rgi"})
    )
    grouped["br101_length_km_in_rgi"] = grouped["br101_length_m_in_rgi"] / 1000.0
    grouped = grouped.rename(
        columns={
            "CD_RGI": "rgi_id",
            "NM_RGI": "rgi_name",
            "CD_RGINT": "rgint_id",
            "NM_RGINT": "rgint_name",
            "SIGLA_UF": "uf",
        }
    )
    grouped["rgi_id"] = normalize_code(grouped["rgi_id"])
    grouped["rgint_id"] = normalize_code(grouped["rgint_id"])
    grouped["uf"] = grouped["uf"].astype("string").str.upper().str.strip()
    grouped["rgi_name"] = grouped["rgi_name"].astype("string")
    grouped["rgint_name"] = grouped["rgint_name"].astype("string")

    rgis_in_scope = rgis_in_scope.rename(
        columns={
            "CD_RGI": "rgi_id",
            "NM_RGI": "rgi_name_scope",
            "CD_RGINT": "rgint_id_scope",
            "NM_RGINT": "rgint_name_scope",
            "SIGLA_UF": "uf_scope",
        }
    )
    rgis_in_scope["rgi_id"] = normalize_code(rgis_in_scope["rgi_id"])
    merged = grouped.merge(rgis_in_scope, on="rgi_id", how="left")
    merged["rgi_name"] = merged["rgi_name"].fillna(merged["rgi_name_scope"])
    merged["rgint_id"] = merged["rgint_id"].fillna(merged["rgint_id_scope"])
    merged["rgint_name"] = merged["rgint_name"].fillna(merged["rgint_name_scope"])
    merged["uf"] = merged["uf"].fillna(merged["uf_scope"])
    merged = merged[
        [
            "rgi_id",
            "rgi_name",
            "uf",
            "rgint_id",
            "rgint_name",
            "br101_length_m_in_rgi",
            "br101_length_km_in_rgi",
        ]
    ].drop_duplicates()

    if merged.empty:
        raise ValueError("The aggregated BR-101 road-length table is empty.")

    return merged.sort_values(["uf", "rgi_name", "rgi_id"]).reset_index(drop=True)


def load_notebook3_inputs(config: Notebook3Config) -> dict[str, pd.DataFrame]:
    config.ensure_output_dirs()
    canonical_path = config.silver_input_dir / "canonical_accidents.parquet"
    exclusions_path = config.silver_input_dir / "manual_exclusions.parquet"

    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Missing canonical accidents artifact at {canonical_path}."
        )
    if not exclusions_path.exists():
        raise FileNotFoundError(
            f"Missing manual exclusions artifact at {exclusions_path}."
        )

    canonical_accidents = pd.read_parquet(canonical_path)
    manual_exclusions = pd.read_parquet(exclusions_path)
    rgi_static_features = build_rgi_static_features(config)

    validate_required_columns(
        canonical_accidents,
        ["id", "timestamp", "year", "uf", "CD_RGI", "NM_RGI"],
        "canonical_accidents",
    )
    validate_required_columns(
        manual_exclusions,
        ["uf", "year", "exclude_from_modeling"],
        "manual_exclusions",
    )

    return {
        "canonical_accidents": canonical_accidents,
        "manual_exclusions": manual_exclusions,
        "rgi_static_features": rgi_static_features,
    }


def apply_manual_exclusions(
    df: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    flag_column: str,
) -> pd.DataFrame:
    active = exclusions.loc[
        exclusions[flag_column].fillna(False), ["uf", "year", flag_column]
    ].copy()
    active[flag_column] = True
    merged = df.merge(active, on=["uf", "year"], how="left")
    exclusion_mask = merged[flag_column].fillna(False).to_numpy(dtype=bool)
    return merged.loc[~exclusion_mask].drop(columns=[flag_column], errors="ignore")


def prepare_modeling_accidents(
    canonical_accidents: pd.DataFrame,
    manual_exclusions: pd.DataFrame,
    rgi_static_features: pd.DataFrame,
    config: Notebook3Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = canonical_accidents.copy()
    analysis_df["timestamp"] = pd.to_datetime(analysis_df["timestamp"], errors="coerce")
    analysis_df = analysis_df.loc[analysis_df["timestamp"].notna()].copy()
    analysis_df["accident_uf"] = (
        analysis_df["uf"].astype("string").str.upper().str.strip()
    )
    analysis_df["uf"] = analysis_df["accident_uf"]
    analysis_df["year"] = pd.to_numeric(analysis_df["year"], errors="coerce").astype(
        "Int64"
    )
    analysis_df = analysis_df.dropna(subset=["year"]).copy()
    analysis_df["year"] = analysis_df["year"].astype(int)
    analysis_df = apply_manual_exclusions(
        analysis_df,
        manual_exclusions,
        flag_column="exclude_from_modeling",
    )

    analysis_df["rgi_id"] = normalize_code(analysis_df["CD_RGI"])
    analysis_df["rgi_name"] = analysis_df["NM_RGI"].astype("string")
    analysis_df["week_start"] = normalize_week_start(analysis_df["timestamp"])
    analysis_df = analysis_df.loc[analysis_df["week_start"].notna()].copy()
    analysis_df = analysis_df.loc[
        analysis_df["week_start"].between(
            config.outer_window_start, config.outer_window_end
        )
    ].copy()
    analysis_df = analysis_df.loc[analysis_df["rgi_id"].notna()].copy()
    analysis_df = analysis_df.merge(
        rgi_static_features[["rgi_id", "rgi_name", "uf", "br101_length_km_in_rgi"]],
        on="rgi_id",
        how="inner",
        suffixes=("_accident", ""),
    )
    analysis_df["rgi_name"] = analysis_df["rgi_name"].fillna(
        analysis_df["rgi_name_accident"]
    )
    analysis_df = analysis_df.drop(columns=["rgi_name_accident"], errors="ignore")

    summary = pd.DataFrame(
        [
            {"step": "canonical_accidents_loaded", "rows": len(canonical_accidents)},
            {
                "step": "rows_with_valid_timestamp",
                "rows": int(canonical_accidents["timestamp"].notna().sum()),
            },
            {
                "step": "rows_after_manual_exclusions",
                "rows": len(
                    apply_manual_exclusions(
                        canonical_accidents.assign(
                            timestamp=pd.to_datetime(
                                canonical_accidents["timestamp"], errors="coerce"
                            ),
                            uf=canonical_accidents["uf"]
                            .astype("string")
                            .str.upper()
                            .str.strip(),
                            year=pd.to_numeric(
                                canonical_accidents["year"], errors="coerce"
                            )
                            .fillna(-1)
                            .astype(int),
                        ).loc[lambda frame: frame["timestamp"].notna()],
                        manual_exclusions,
                        flag_column="exclude_from_modeling",
                    )
                ),
            },
            {"step": "rows_after_outer_window_filter", "rows": len(analysis_df)},
            {"step": "retained_rgis", "rows": int(analysis_df["rgi_id"].nunique())},
        ]
    )

    return (
        analysis_df[
            [
                "id",
                "timestamp",
                "year",
                "accident_uf",
                "uf",
                "rgi_id",
                "rgi_name",
                "week_start",
                "br101_length_km_in_rgi",
            ]
        ].copy(),
        summary,
    )


def build_weekly_calendar(config: Notebook3Config) -> pd.DataFrame:
    week_starts = pd.date_range(
        start=config.outer_window_start,
        end=config.outer_window_end,
        freq="W-MON",
    )
    calendar = pd.DataFrame({"week_start": week_starts})
    calendar["week_end"] = calendar["week_start"] + pd.Timedelta(days=6)
    iso = calendar["week_start"].dt.isocalendar()
    calendar["iso_year"] = iso["year"].astype(int)
    calendar["iso_week"] = iso["week"].astype(int)
    calendar["year_week"] = [
        f"{year}-W{week:02d}"
        for year, week in zip(calendar["iso_year"], calendar["iso_week"], strict=True)
    ]
    calendar["week_index"] = np.arange(len(calendar), dtype=int)
    return calendar


def aggregate_weekly_targets(
    accidents: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        accidents.groupby(["rgi_id", "week_start"], as_index=False)
        .agg(accident_count=("id", "size"))
        .sort_values(["rgi_id", "week_start"])
    )
    grouped = grouped.merge(
        weekly_calendar[
            ["week_start", "week_end", "year_week", "iso_year", "iso_week"]
        ],
        on="week_start",
        how="left",
    )
    grouped["accident_count"] = grouped["accident_count"].astype(int)
    return grouped.reset_index(drop=True)


def _week_start_from_date(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series) - pd.to_timedelta(
        pd.to_datetime(date_series).dt.weekday, unit="D"
    )


def _longest_non_work_streak(scores: pd.Series) -> float:
    best = 0.0
    current = 0.0
    for score in scores.to_numpy(dtype=float):
        if score > 0:
            current += float(score)
        else:
            best = max(best, current)
            current = 0.0
    return max(best, current)


def build_weekly_holiday_features(
    rgi_static_features: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    config: Notebook3Config,
) -> pd.DataFrame:
    retained_ufs = sorted(
        rgi_static_features["uf"].dropna().astype(str).unique().tolist()
    )
    date_start = weekly_calendar["week_start"].min() - pd.Timedelta(weeks=1)
    date_end = weekly_calendar["week_end"].max() + pd.Timedelta(weeks=1)
    all_dates = pd.date_range(start=date_start, end=date_end, freq="D")

    base = pd.MultiIndex.from_product(
        [retained_ufs, all_dates], names=["uf", "date"]
    ).to_frame(index=False)
    base["year"] = base["date"].dt.year
    base["is_weekend"] = base["date"].dt.weekday >= 5

    holiday_cache: dict[tuple[str, int], holidays.HolidayBase] = {}

    def is_holiday(row: pd.Series) -> bool:
        key = (str(row["uf"]), int(row["year"]))
        if key not in holiday_cache:
            holiday_cache[key] = holidays.Brazil(years=[key[1]], subdiv=key[0])
        return row["date"].date() in holiday_cache[key]

    base["is_holiday"] = base.apply(is_holiday, axis=1)
    base["is_non_work_core"] = base["is_weekend"] | base["is_holiday"]
    base = base.sort_values(["uf", "date"]).reset_index(drop=True)
    base["prev_is_non_work_core"] = base.groupby("uf")["is_non_work_core"].shift(
        1, fill_value=False
    )
    base["next_is_non_work_core"] = base.groupby("uf")["is_non_work_core"].shift(
        -1, fill_value=False
    )
    base["is_bridge_day"] = (
        ~base["is_non_work_core"]
        & base["prev_is_non_work_core"]
        & base["next_is_non_work_core"]
    )
    base["non_work_score"] = np.select(
        [base["is_non_work_core"], base["is_bridge_day"]],
        [1.0, config.bridge_weight],
        default=0.0,
    )
    base["week_start"] = _week_start_from_date(base["date"])

    weekly = (
        base.groupby(["uf", "week_start"], as_index=False)
        .agg(
            non_work_day_share_in_week=("non_work_score", "mean"),
            longest_non_work_streak_in_week=(
                "non_work_score",
                _longest_non_work_streak,
            ),
        )
        .sort_values(["uf", "week_start"])
    )
    weekly = weekly.merge(
        weekly_calendar[["week_start", "week_end", "year_week"]],
        on="week_start",
        how="left",
    )
    margin_calendar = pd.DataFrame(
        {
            "week_start": pd.date_range(
                date_start.normalize(), date_end.normalize(), freq="W-MON"
            )
        }
    )
    margin_calendar["week_end"] = margin_calendar["week_start"] + pd.Timedelta(days=6)
    margin_calendar["year_week"] = margin_calendar["week_start"].map(year_week_label)

    weekly = (
        pd.MultiIndex.from_product(
            [retained_ufs, margin_calendar["week_start"]], names=["uf", "week_start"]
        )
        .to_frame(index=False)
        .merge(margin_calendar, on="week_start", how="left")
        .merge(weekly, on=["uf", "week_start", "week_end", "year_week"], how="left")
        .sort_values(["uf", "week_start"])
        .reset_index(drop=True)
    )
    weekly["non_work_day_share_in_week"] = weekly["non_work_day_share_in_week"].fillna(
        0.0
    )
    weekly["longest_non_work_streak_in_week"] = weekly[
        "longest_non_work_streak_in_week"
    ].fillna(0.0)

    weekly["prev_week_non_work_day_share"] = weekly.groupby("uf")[
        "non_work_day_share_in_week"
    ].shift(1)
    weekly["prev_week_longest_non_work_streak"] = weekly.groupby("uf")[
        "longest_non_work_streak_in_week"
    ].shift(1)
    weekly["curr_week_non_work_day_share"] = weekly["non_work_day_share_in_week"]
    weekly["curr_week_longest_non_work_streak"] = weekly[
        "longest_non_work_streak_in_week"
    ]
    weekly["next_week_non_work_day_share"] = weekly.groupby("uf")[
        "non_work_day_share_in_week"
    ].shift(-1)
    weekly["next_week_longest_non_work_streak"] = weekly.groupby("uf")[
        "longest_non_work_streak_in_week"
    ].shift(-1)

    main_week_starts = set(weekly_calendar["week_start"].tolist())
    weekly = weekly.loc[weekly["week_start"].isin(main_week_starts)].copy()
    for column in HOLIDAY_FEATURE_COLUMNS:
        weekly[column] = weekly[column].fillna(0.0)

    return weekly[
        ["uf", "week_start", "week_end", "year_week", *HOLIDAY_FEATURE_COLUMNS]
    ].reset_index(drop=True)


def build_dense_weekly_panel(
    rgi_static_features: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    weekly_targets: pd.DataFrame,
    weekly_holiday_features: pd.DataFrame,
) -> pd.DataFrame:
    panel = rgi_static_features.merge(weekly_calendar, how="cross")
    panel = panel.merge(
        weekly_targets[["rgi_id", "week_start", "accident_count"]],
        on=["rgi_id", "week_start"],
        how="left",
    )
    panel = panel.merge(
        weekly_holiday_features,
        on=["uf", "week_start", "week_end", "year_week"],
        how="left",
    )
    panel["accident_count"] = panel["accident_count"].fillna(0).astype(int)
    for column in HOLIDAY_FEATURE_COLUMNS:
        panel[column] = panel[column].fillna(0.0)

    duplicated = panel.duplicated(subset=["rgi_id", "week_start"]).any()
    if duplicated:
        raise ValueError("The weekly panel is not unique at rgi_id x week_start.")

    return panel.sort_values(["rgi_id", "week_start"]).reset_index(drop=True)


def assign_split_labels(panel: pd.DataFrame, config: Notebook3Config) -> pd.DataFrame:
    labeled = panel.copy()
    labeled["split"] = np.select(
        [
            labeled["week_start"].between(config.train_start, config.train_end),
            labeled["week_start"].between(
                config.validation_start, config.validation_end
            ),
            labeled["week_start"].between(config.test_start, config.test_end),
        ],
        ["train", "validation", "test"],
        default="outside_scope",
    )
    labeled["is_train"] = labeled["split"].eq("train")
    labeled["is_validation"] = labeled["split"].eq("validation")
    labeled["is_test"] = labeled["split"].eq("test")
    return labeled


def build_split_manifest(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    return panel_with_splits[
        [
            "rgi_id",
            "rgi_name",
            "uf",
            "week_start",
            "week_end",
            "year_week",
            "split",
            "is_train",
            "is_validation",
            "is_test",
        ]
    ].copy()


def build_panel_diagnostics(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    diagnostics: list[dict[str, Any]] = []
    overall_panel = panel_with_splits.copy()
    overall_per_rgi = overall_panel.groupby(
        ["rgi_id", "rgi_name", "uf"], as_index=False
    ).agg(
        total_accidents=("accident_count", "sum"),
        mean_weekly_accidents=("accident_count", "mean"),
        median_weekly_accidents=("accident_count", "median"),
        zero_share=("accident_count", lambda s: float(s.eq(0).mean())),
    )

    diagnostics.extend(
        [
            {
                "diagnostic_group": "panel",
                "diagnostic_name": "panel_rows",
                "split": "overall",
                "value": float(len(panel_with_splits)),
            },
            {
                "diagnostic_group": "panel",
                "diagnostic_name": "retained_rgis",
                "split": "overall",
                "value": float(panel_with_splits["rgi_id"].nunique()),
            },
            {
                "diagnostic_group": "panel",
                "diagnostic_name": "total_weeks",
                "split": "overall",
                "value": float(panel_with_splits["week_start"].nunique()),
            },
            {
                "diagnostic_group": "overall_dense_target",
                "diagnostic_name": "min",
                "split": "overall",
                "value": float(overall_panel["accident_count"].min()),
            },
            {
                "diagnostic_group": "overall_dense_target",
                "diagnostic_name": "max",
                "split": "overall",
                "value": float(overall_panel["accident_count"].max()),
            },
            {
                "diagnostic_group": "overall_dense_target",
                "diagnostic_name": "std",
                "split": "overall",
                "value": float(overall_panel["accident_count"].std(ddof=0)),
            },
            {
                "diagnostic_group": "overall_dense_target",
                "diagnostic_name": "mean",
                "split": "overall",
                "value": float(overall_panel["accident_count"].mean()),
            },
            {
                "diagnostic_group": "overall_dense_target",
                "diagnostic_name": "median",
                "split": "overall",
                "value": float(overall_panel["accident_count"].median()),
            },
            {
                "diagnostic_group": "overall_rgi_total_accidents",
                "diagnostic_name": "min",
                "split": "overall",
                "value": float(overall_per_rgi["total_accidents"].min()),
            },
            {
                "diagnostic_group": "overall_rgi_total_accidents",
                "diagnostic_name": "max",
                "split": "overall",
                "value": float(overall_per_rgi["total_accidents"].max()),
            },
            {
                "diagnostic_group": "overall_rgi_total_accidents",
                "diagnostic_name": "std",
                "split": "overall",
                "value": float(overall_per_rgi["total_accidents"].std(ddof=0)),
            },
            {
                "diagnostic_group": "overall_rgi_total_accidents",
                "diagnostic_name": "mean",
                "split": "overall",
                "value": float(overall_per_rgi["total_accidents"].mean()),
            },
            {
                "diagnostic_group": "overall_rgi_total_accidents",
                "diagnostic_name": "median",
                "split": "overall",
                "value": float(overall_per_rgi["total_accidents"].median()),
            },
            {
                "diagnostic_group": "overall_rgi_mean_weekly_accidents",
                "diagnostic_name": "min",
                "split": "overall",
                "value": float(overall_per_rgi["mean_weekly_accidents"].min()),
            },
            {
                "diagnostic_group": "overall_rgi_mean_weekly_accidents",
                "diagnostic_name": "max",
                "split": "overall",
                "value": float(overall_per_rgi["mean_weekly_accidents"].max()),
            },
            {
                "diagnostic_group": "overall_rgi_mean_weekly_accidents",
                "diagnostic_name": "std",
                "split": "overall",
                "value": float(overall_per_rgi["mean_weekly_accidents"].std(ddof=0)),
            },
            {
                "diagnostic_group": "overall_rgi_mean_weekly_accidents",
                "diagnostic_name": "mean",
                "split": "overall",
                "value": float(overall_per_rgi["mean_weekly_accidents"].mean()),
            },
            {
                "diagnostic_group": "overall_rgi_mean_weekly_accidents",
                "diagnostic_name": "median",
                "split": "overall",
                "value": float(overall_per_rgi["mean_weekly_accidents"].median()),
            },
        ]
    )

    for split, split_df in panel_with_splits.groupby("split"):
        if split == "outside_scope":
            continue
        diagnostics.extend(
            [
                {
                    "diagnostic_group": "panel",
                    "diagnostic_name": "rows",
                    "split": split,
                    "value": float(len(split_df)),
                },
                {
                    "diagnostic_group": "panel",
                    "diagnostic_name": "weeks",
                    "split": split,
                    "value": float(split_df["week_start"].nunique()),
                },
                {
                    "diagnostic_group": "target",
                    "diagnostic_name": "mean_weekly_accidents",
                    "split": split,
                    "value": float(split_df["accident_count"].mean()),
                },
                {
                    "diagnostic_group": "target",
                    "diagnostic_name": "variance_weekly_accidents",
                    "split": split,
                    "value": float(split_df["accident_count"].var(ddof=0)),
                },
                {
                    "diagnostic_group": "target",
                    "diagnostic_name": "zero_share",
                    "split": split,
                    "value": float(split_df["accident_count"].eq(0).mean()),
                },
            ]
        )

        per_rgi = split_df.groupby(["rgi_id", "rgi_name", "uf"], as_index=False).agg(
            total_accidents=("accident_count", "sum"),
            mean_weekly_accidents=("accident_count", "mean"),
            zero_share=("accident_count", lambda s: float(s.eq(0).mean())),
        )
        for column in ["total_accidents", "mean_weekly_accidents", "zero_share"]:
            diagnostics.extend(
                [
                    {
                        "diagnostic_group": f"{column}_distribution",
                        "diagnostic_name": "mean",
                        "split": split,
                        "value": float(per_rgi[column].mean()),
                    },
                    {
                        "diagnostic_group": f"{column}_distribution",
                        "diagnostic_name": "median",
                        "split": split,
                        "value": float(per_rgi[column].median()),
                    },
                    {
                        "diagnostic_group": f"{column}_distribution",
                        "diagnostic_name": "p10",
                        "split": split,
                        "value": float(per_rgi[column].quantile(0.10)),
                    },
                    {
                        "diagnostic_group": f"{column}_distribution",
                        "diagnostic_name": "p90",
                        "split": split,
                        "value": float(per_rgi[column].quantile(0.90)),
                    },
                ]
            )

        top_rgis = per_rgi.nlargest(10, "total_accidents").assign(
            rank=lambda frame: np.arange(1, len(frame) + 1)
        )
        bottom_rgis = per_rgi.nsmallest(10, "total_accidents").assign(
            rank=lambda frame: np.arange(1, len(frame) + 1)
        )
        for frame, name in [
            (top_rgis, "top_total_accidents"),
            (bottom_rgis, "bottom_total_accidents"),
        ]:
            for row in frame.itertuples(index=False):
                diagnostics.append(
                    {
                        "diagnostic_group": "rgi_rank",
                        "diagnostic_name": name,
                        "split": split,
                        "value": float(row.total_accidents),
                        "rgi_id": row.rgi_id,
                        "rgi_name": row.rgi_name,
                        "uf": row.uf,
                        "rank": int(row.rank),
                    }
                )

    return pd.DataFrame(diagnostics)


def summarize_horizon_rmse(
    forecasts: pd.DataFrame,
    *,
    split: str = "test",
    model: str | None = None,
    rgi_id: str | None = None,
) -> pd.DataFrame:
    filtered = forecasts.loc[forecasts["split"].eq(split)].copy()
    if model is not None:
        filtered = filtered.loc[filtered["model"].eq(model)].copy()
    if rgi_id is not None:
        filtered = filtered.loc[filtered["rgi_id"].astype(str).eq(str(rgi_id))].copy()

    filtered = filtered.loc[
        filtered["prediction"].notna() & filtered["actual"].notna()
    ].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=["horizon_step", "horizon_label", "rmse", "n_predictions"]
        )

    summary = (
        filtered.groupby("horizon_step", as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "rmse": float(
                        np.sqrt(np.mean((frame["actual"] - frame["prediction"]) ** 2))
                    ),
                    "n_predictions": int(len(frame)),
                }
            ),
            include_groups=False,
        )
        .sort_values("horizon_step")
        .reset_index(drop=True)
    )
    summary["horizon_label"] = summary["horizon_step"].map(
        lambda value: f"q{int(value)}"
    )
    return summary[["horizon_step", "horizon_label", "rmse", "n_predictions"]]


def plot_test_predictions_for_rgi(
    panel_with_splits: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    rgi_id: str,
    model: str | None = None,
    split: str = "test",
    figsize: tuple[int, int] = (16, 6),
) -> pd.DataFrame:
    actuals = (
        panel_with_splits.loc[
            panel_with_splits["split"].eq(split)
            & panel_with_splits["rgi_id"].astype(str).eq(str(rgi_id))
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
    if model is not None:
        prediction_frame = prediction_frame.loc[
            prediction_frame["model"].eq(model)
        ].copy()

    if prediction_frame.empty:
        raise ValueError(
            f"No forecast rows found for rgi_id={rgi_id!r}, split={split!r}, model={model!r}."
        )

    prediction_frame = prediction_frame.sort_values(
        ["model", "week_start", "horizon_step"]
    ).copy()
    rgi_name = str(actuals["rgi_name"].iloc[0])
    uf = str(actuals["uf"].iloc[0])
    models = prediction_frame["model"].dropna().astype(str).unique().tolist()

    plt.figure(figsize=figsize)
    plt.plot(
        actuals["week_start"],
        actuals["accident_count"],
        label="actual",
        color="black",
        linewidth=2.0,
    )

    palette = sns.color_palette("tab10", n_colors=max(len(models), 1))
    for color, current_model in zip(palette, models, strict=False):
        model_frame = (
            prediction_frame.loc[prediction_frame["model"].eq(current_model)]
            .drop_duplicates(subset=["week_start"])
            .sort_values("week_start")
        )
        plt.plot(
            model_frame["week_start"],
            model_frame["prediction"],
            label=f"{current_model} prediction",
            color=color,
            linewidth=1.8,
            alpha=0.9,
        )
        plt.scatter(
            model_frame["week_start"],
            model_frame["prediction"],
            color=color,
            s=18,
            alpha=0.75,
        )

    plt.title(f"{split.title()} actual vs predictions for {rgi_name} ({uf}) [{rgi_id}]")
    plt.xlabel("Week start")
    plt.ylabel("Accident count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    horizon_frames: list[pd.DataFrame] = []
    for current_model in models:
        horizon_summary = summarize_horizon_rmse(
            prediction_frame,
            split=split,
            model=current_model,
            rgi_id=str(rgi_id),
        )
        horizon_summary.insert(0, "model", current_model)
        horizon_frames.append(horizon_summary)

    return (
        pd.concat(horizon_frames, ignore_index=True)
        if horizon_frames
        else pd.DataFrame(
            columns=["model", "horizon_step", "horizon_label", "rmse", "n_predictions"]
        )
    )


def plot_panel_diagnostics(panel_with_splits: pd.DataFrame) -> None:
    per_rgi = (
        panel_with_splits.groupby(["rgi_id", "rgi_name"], as_index=False)[
            "accident_count"
        ]
        .sum()
        .sort_values(["accident_count", "rgi_name"], ascending=[False, True])
    )
    plt.figure(figsize=(14, 4))
    sns.histplot(per_rgi["accident_count"], bins=20)
    plt.title("Total accidents per RGI")
    plt.xlabel("Total accidents")
    plt.ylabel("RGI count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    sns.histplot(
        panel_with_splits.groupby("rgi_id", as_index=False)["accident_count"].mean()[
            "accident_count"
        ],
        bins=20,
    )
    plt.title("Mean weekly accidents per RGI")
    plt.xlabel("Mean weekly accident count")
    plt.ylabel("RGI count")
    plt.tight_layout()
    plt.show()

    zero_share = (
        panel_with_splits.groupby("rgi_id")["accident_count"]
        .apply(lambda s: float(s.eq(0).mean()))
        .reset_index(name="zero_share")
    )
    plt.figure(figsize=(14, 4))
    sns.histplot(zero_share["zero_share"], bins=20)
    plt.title("Per-RGI zero share")
    plt.xlabel("Zero-count share")
    plt.ylabel("RGI count")
    plt.tight_layout()
    plt.show()

    aggregate_series = (
        panel_with_splits.groupby("week_start", as_index=False)["accident_count"]
        .sum()
        .sort_values("week_start")
    )
    plt.figure(figsize=(16, 4))
    sns.lineplot(data=aggregate_series, x="week_start", y="accident_count")
    plt.title("Aggregate weekly accident count across all RGIs")
    plt.xlabel("Week start")
    plt.ylabel("Accident count")
    plt.tight_layout()
    plt.show()

    heatmap_source = panel_with_splits.merge(
        per_rgi[["rgi_id", "rgi_name"]],
        on="rgi_id",
        how="left",
        suffixes=("", "_ordered"),
    )
    ordered_rgis = per_rgi["rgi_id"].tolist()
    heatmap = (
        heatmap_source.assign(
            rgi_order=pd.Categorical(
                heatmap_source["rgi_id"], categories=ordered_rgis, ordered=True
            )
        )
        .pivot(index="rgi_order", columns="year_week", values="accident_count")
        .fillna(0.0)
    )
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        np.log1p(heatmap), cmap="mako", cbar_kws={"label": "log1p(accident_count)"}
    )
    plt.title("Weekly accident counts across the full BR-101 RGI universe")
    plt.xlabel("ISO week")
    plt.ylabel("RGIs ordered by total accidents")
    plt.tight_layout()
    plt.show()


def empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FORECAST_COLUMNS)


def _split_bounds(
    config: Notebook3Config, split: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return config.train_start, config.train_end
    if split == "validation":
        return config.validation_start, config.validation_end
    if split == "test":
        return config.test_start, config.test_end
    raise ValueError(f"Unsupported split {split!r}.")


def iter_forecast_batches(config: Notebook3Config, split: str) -> list[dict[str, Any]]:
    split_start, split_end = _split_bounds(config, split)
    batches: list[dict[str, Any]] = []
    batch_start = split_start
    while (
        batch_start + pd.Timedelta(weeks=config.forecast_horizon_weeks - 1) <= split_end
    ):
        target_weeks = [
            batch_start + pd.Timedelta(weeks=offset)
            for offset in range(config.forecast_horizon_weeks)
        ]
        cutoff_week_start = batch_start - pd.Timedelta(weeks=config.latency_gap_weeks)
        batches.append(
            {
                "split": split,
                "forecast_batch_start": batch_start,
                "cutoff_week_start": cutoff_week_start,
                "target_weeks": target_weeks,
            }
        )
        batch_start += pd.Timedelta(weeks=config.forecast_horizon_weeks)
    return batches


def generate_baseline_forecasts(
    panel_with_splits: pd.DataFrame,
    config: Notebook3Config,
    split: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    batches = iter_forecast_batches(config, split)

    for _, group in panel_with_splits.groupby("rgi_id"):
        group = group.sort_values("week_start").reset_index(drop=True)
        series = group.set_index("week_start")["accident_count"]
        week_meta = group.set_index("week_start")[["week_end", "year_week"]]
        rgi_id = str(group["rgi_id"].iloc[0])
        rgi_name = str(group["rgi_name"].iloc[0])
        uf = str(group["uf"].iloc[0])

        for batch in batches:
            cutoff = batch["cutoff_week_start"]
            history = series.loc[series.index <= cutoff]
            moving_average_prediction = (
                np.nan
                if history.empty
                else float(history.tail(config.moving_average_window).mean())
            )

            for step, target_week in enumerate(batch["target_weeks"], start=1):
                actual = float(series.get(target_week, np.nan))
                week_end = week_meta.loc[target_week, "week_end"]
                year_week = week_meta.loc[target_week, "year_week"]

                seasonal_source_week = target_week - pd.Timedelta(
                    weeks=config.seasonal_period
                )
                seasonal_prediction = np.nan
                if (
                    seasonal_source_week in series.index
                    and seasonal_source_week <= cutoff
                ):
                    seasonal_prediction = float(series.loc[seasonal_source_week])

                records.append(
                    {
                        "model": "seasonal_naive",
                        "split": split,
                        "rgi_id": rgi_id,
                        "rgi_name": rgi_name,
                        "uf": uf,
                        "week_start": target_week,
                        "week_end": week_end,
                        "year_week": year_week,
                        "forecast_batch_start": batch["forecast_batch_start"],
                        "cutoff_week_start": cutoff,
                        "horizon_step": step,
                        "prediction": seasonal_prediction,
                        "actual": actual,
                        "is_available": not np.isnan(seasonal_prediction),
                        "metadata": f"seasonal_source_week={year_week_label(seasonal_source_week)}",
                    }
                )
                records.append(
                    {
                        "model": "moving_average_4w",
                        "split": split,
                        "rgi_id": rgi_id,
                        "rgi_name": rgi_name,
                        "uf": uf,
                        "week_start": target_week,
                        "week_end": week_end,
                        "year_week": year_week,
                        "forecast_batch_start": batch["forecast_batch_start"],
                        "cutoff_week_start": cutoff,
                        "horizon_step": step,
                        "prediction": moving_average_prediction,
                        "actual": actual,
                        "is_available": not np.isnan(moving_average_prediction),
                        "metadata": f"history_rows={int(len(history.tail(config.moving_average_window)))}",
                    }
                )

    if not records:
        return empty_forecast_frame()

    return (
        pd.DataFrame(records)
        .sort_values(["model", "rgi_id", "week_start"])
        .reset_index(drop=True)
    )


def build_sarima_candidate_grid(
    config: Notebook3Config,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    grid: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for p, d, q, seasonal_p, seasonal_d, seasonal_q in product([0, 1], repeat=6):
        if (p, d, q, seasonal_p, seasonal_d, seasonal_q) == (0, 0, 0, 0, 0, 0):
            continue
        grid.append(
            ((p, d, q), (seasonal_p, seasonal_d, seasonal_q, config.seasonal_period))
        )
    return grid


def sarima_spec_label(
    order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]
) -> str:
    return f"{order} x {seasonal_order}"


def _fit_sarima_history(
    history: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    maxiter: int,
) -> Any:
    model = SARIMAX(
        history.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=maxiter)


def generate_sarima_forecasts_for_rgi(
    rgi_panel: pd.DataFrame,
    config: Notebook3Config,
    split: str,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = (
        rgi_panel.set_index("week_start")["accident_count"].sort_index().asfreq("W-MON")
    )
    week_meta = (
        rgi_panel.set_index("week_start")[["week_end", "year_week"]]
        .sort_index()
        .asfreq("W-MON")
    )
    rgi_id = str(rgi_panel["rgi_id"].iloc[0])
    rgi_name = str(rgi_panel["rgi_name"].iloc[0])
    uf = str(rgi_panel["uf"].iloc[0])

    records: list[dict[str, Any]] = []
    timing_start = perf_counter()

    try:
        for batch in iter_forecast_batches(config, split):
            cutoff = batch["cutoff_week_start"]
            history = series.loc[series.index <= cutoff]
            if history.empty:
                raise ValueError("No history available at the forecast cutoff.")
            model_result = _fit_sarima_history(
                history, order, seasonal_order, config.sarima_maxiter
            )
            steps = config.latency_gap_weeks + config.forecast_horizon_weeks - 1
            forecast = model_result.forecast(steps=steps)
            target_forecast = forecast.iloc[
                config.latency_gap_weeks - 1 : config.latency_gap_weeks
                - 1
                + config.forecast_horizon_weeks
            ]

            for step, (target_week, prediction) in enumerate(
                zip(
                    batch["target_weeks"],
                    target_forecast.to_numpy(dtype=float),
                    strict=True,
                ),
                start=1,
            ):
                actual = float(series.get(target_week, np.nan))
                records.append(
                    {
                        "model": "sarima",
                        "split": split,
                        "rgi_id": rgi_id,
                        "rgi_name": rgi_name,
                        "uf": uf,
                        "week_start": target_week,
                        "week_end": week_meta.loc[target_week, "week_end"],
                        "year_week": week_meta.loc[target_week, "year_week"],
                        "forecast_batch_start": batch["forecast_batch_start"],
                        "cutoff_week_start": cutoff,
                        "horizon_step": step,
                        "prediction": float(max(prediction, 0.0)),
                        "actual": actual,
                        "is_available": True,
                        "metadata": sarima_spec_label(order, seasonal_order),
                    }
                )
    except Exception as exc:
        return empty_forecast_frame(), {
            "rgi_id": rgi_id,
            "rgi_name": rgi_name,
            "uf": uf,
            "split": split,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": perf_counter() - timing_start,
            "n_predictions": 0,
        }

    forecasts = pd.DataFrame(records).sort_values("week_start").reset_index(drop=True)
    return forecasts, {
        "rgi_id": rgi_id,
        "rgi_name": rgi_name,
        "uf": uf,
        "split": split,
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "status": "success",
        "failure_reason": "",
        "runtime_seconds": perf_counter() - timing_start,
        "n_predictions": int(forecasts["prediction"].notna().sum()),
    }


def rmse_from_frame(forecasts: pd.DataFrame) -> float:
    available = forecasts.loc[
        forecasts["prediction"].notna() & forecasts["actual"].notna()
    ]
    if available.empty:
        return float("nan")
    return float(np.sqrt(np.mean((available["actual"] - available["prediction"]) ** 2)))


def select_best_sarima_specs(
    panel_with_splits: pd.DataFrame,
    config: Notebook3Config,
    candidate_grid: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]
    | None = None,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[int, ...]]]]:
    candidates = candidate_grid or build_sarima_candidate_grid(config)
    diagnostics: list[dict[str, Any]] = []
    best_specs: dict[str, dict[str, tuple[int, ...]]] = {}
    rgi_groups = list(panel_with_splits.groupby("rgi_id"))

    with tqdm(
        total=len(rgi_groups) * len(candidates),
        desc="SARIMA validation search",
        disable=not show_progress,
    ) as progress_bar:
        for _, rgi_panel in rgi_groups:
            best_rmse = float("inf")
            best_spec: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = (
                None
            )
            current_rgi_id = str(rgi_panel["rgi_id"].iloc[0])

            for order, seasonal_order in candidates:
                validation_forecasts, info = generate_sarima_forecasts_for_rgi(
                    rgi_panel,
                    config,
                    split="validation",
                    order=order,
                    seasonal_order=seasonal_order,
                )
                validation_rmse = rmse_from_frame(validation_forecasts)
                diagnostics.append({**info, "validation_rmse": validation_rmse})
                progress_bar.update(1)
                progress_bar.set_postfix(
                    rgi_id=current_rgi_id,
                    best_rmse=f"{best_rmse:.3f}" if np.isfinite(best_rmse) else "n/a",
                )
                if np.isnan(validation_rmse):
                    continue
                if validation_rmse < best_rmse:
                    best_rmse = validation_rmse
                    best_spec = (order, seasonal_order)
                    progress_bar.set_postfix(
                        rgi_id=current_rgi_id, best_rmse=f"{best_rmse:.3f}"
                    )

            if best_spec is not None:
                best_specs[current_rgi_id] = {
                    "order": best_spec[0],
                    "seasonal_order": best_spec[1],
                }

    diagnostics_df = pd.DataFrame(diagnostics)
    if not diagnostics_df.empty:
        diagnostics_df["is_selected"] = diagnostics_df.apply(
            lambda row: (
                row["rgi_id"] in best_specs
                and row["order"] == str(best_specs[row["rgi_id"]]["order"])
                and row["seasonal_order"]
                == str(best_specs[row["rgi_id"]]["seasonal_order"])
            ),
            axis=1,
        )

    return diagnostics_df, best_specs


def generate_sarima_forecasts(
    panel_with_splits: pd.DataFrame,
    config: Notebook3Config,
    best_specs: dict[str, dict[str, tuple[int, ...]]],
    split: str,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    eligible_groups = [
        (_, rgi_panel)
        for _, rgi_panel in panel_with_splits.groupby("rgi_id")
        if str(rgi_panel["rgi_id"].iloc[0]) in best_specs
    ]
    for _, rgi_panel in tqdm(
        eligible_groups,
        desc=f"SARIMA {split} forecasts",
        disable=not show_progress,
    ):
        rgi_id = str(rgi_panel["rgi_id"].iloc[0])
        spec = best_specs[rgi_id]
        forecast_df, _ = generate_sarima_forecasts_for_rgi(
            rgi_panel,
            config,
            split=split,
            order=tuple(spec["order"]),
            seasonal_order=tuple(spec["seasonal_order"]),
        )
        if not forecast_df.empty:
            frames.append(forecast_df)

    if not frames:
        return empty_forecast_frame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["rgi_id", "week_start"])
        .reset_index(drop=True)
    )


def build_entity_encoders(panel_with_splits: pd.DataFrame) -> dict[str, dict[str, int]]:
    rgi_ids = sorted(panel_with_splits["rgi_id"].dropna().astype(str).unique().tolist())
    ufs = sorted(panel_with_splits["uf"].dropna().astype(str).unique().tolist())
    return {
        "rgi_id": {value: index for index, value in enumerate(rgi_ids)},
        "uf": {value: index for index, value in enumerate(ufs)},
    }


def fit_static_scaler(panel_with_splits: pd.DataFrame) -> dict[str, float]:
    train_values = panel_with_splits.loc[
        panel_with_splits["split"].eq("train"), "br101_length_km_in_rgi"
    ].astype(float)
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    return {"mean": mean, "std": std if std > 0 else 1.0}


def _is_aligned_batch_start(
    batch_start: pd.Timestamp,
    split_start: pd.Timestamp,
    stride_weeks: int,
) -> bool:
    week_delta = int((batch_start - split_start).days // 7)
    return week_delta % stride_weeks == 0


def build_sequence_dataset(
    panel_with_splits: pd.DataFrame,
    config: Notebook3Config,
    split: str,
    encoders: dict[str, dict[str, int]],
    static_scaler: dict[str, float],
    *,
    stride_weeks: int,
) -> dict[str, Any]:
    split_start, split_end = _split_bounds(config, split)
    history_rows: list[np.ndarray] = []
    future_holiday_rows: list[np.ndarray] = []
    road_length_rows: list[list[float]] = []
    rgi_rows: list[int] = []
    uf_rows: list[int] = []
    target_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for _, group in panel_with_splits.groupby("rgi_id"):
        group = group.sort_values("week_start").reset_index(drop=True)
        counts = group["accident_count"].to_numpy(dtype=np.float32)
        weeks = group["week_start"].tolist()
        week_ends = group["week_end"].tolist()
        year_weeks = group["year_week"].tolist()
        holiday_values = group[HOLIDAY_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        rgi_id = str(group["rgi_id"].iloc[0])
        rgi_name = str(group["rgi_name"].iloc[0])
        uf = str(group["uf"].iloc[0])
        scaled_length = (
            float(group["br101_length_km_in_rgi"].iloc[0]) - static_scaler["mean"]
        ) / static_scaler["std"]

        for target_start_idx, batch_start in enumerate(weeks):
            batch_end_idx = target_start_idx + config.forecast_horizon_weeks - 1
            cutoff_idx = target_start_idx - config.latency_gap_weeks
            history_start_idx = cutoff_idx - config.lookback_weeks + 1

            if (
                batch_start < split_start
                or weeks[min(batch_end_idx, len(weeks) - 1)] > split_end
            ):
                continue
            if batch_end_idx >= len(weeks) or cutoff_idx < 0 or history_start_idx < 0:
                continue
            if not _is_aligned_batch_start(batch_start, split_start, stride_weeks):
                continue

            history_rows.append(
                counts[history_start_idx : cutoff_idx + 1].reshape(-1, 1)
            )
            future_holiday_rows.append(
                holiday_values[target_start_idx : batch_end_idx + 1]
            )
            road_length_rows.append([scaled_length])
            rgi_rows.append(encoders["rgi_id"][rgi_id])
            uf_rows.append(encoders["uf"][uf])
            target_rows.append(counts[target_start_idx : batch_end_idx + 1])
            meta_rows.append(
                {
                    "rgi_id": rgi_id,
                    "rgi_name": rgi_name,
                    "uf": uf,
                    "forecast_batch_start": batch_start,
                    "cutoff_week_start": weeks[cutoff_idx],
                    "target_week_starts": weeks[target_start_idx : batch_end_idx + 1],
                    "target_week_ends": week_ends[target_start_idx : batch_end_idx + 1],
                    "target_year_weeks": year_weeks[
                        target_start_idx : batch_end_idx + 1
                    ],
                }
            )

    if not history_rows:
        return {
            "inputs": {
                "history": np.empty((0, config.lookback_weeks, 1), dtype=np.float32),
                "future_holiday": np.empty(
                    (0, config.forecast_horizon_weeks, len(HOLIDAY_FEATURE_COLUMNS)),
                    dtype=np.float32,
                ),
                "road_length": np.empty((0, 1), dtype=np.float32),
                "rgi_id": np.empty((0,), dtype=np.int32),
                "uf_id": np.empty((0,), dtype=np.int32),
            },
            "targets": np.empty((0, config.forecast_horizon_weeks), dtype=np.float32),
            "meta": pd.DataFrame(meta_rows),
            "split": split,
        }

    return {
        "inputs": {
            "history": np.stack(history_rows).astype(np.float32),
            "future_holiday": np.stack(future_holiday_rows).astype(np.float32),
            "road_length": np.asarray(road_length_rows, dtype=np.float32),
            "rgi_id": np.asarray(rgi_rows, dtype=np.int32),
            "uf_id": np.asarray(uf_rows, dtype=np.int32),
        },
        "targets": np.stack(target_rows).astype(np.float32),
        "meta": pd.DataFrame(meta_rows),
        "split": split,
    }


def prepare_rnn_datasets(
    panel_with_splits: pd.DataFrame, config: Notebook3Config
) -> dict[str, Any]:
    encoders = build_entity_encoders(panel_with_splits)
    static_scaler = fit_static_scaler(panel_with_splits)
    train_bundle = build_sequence_dataset(
        panel_with_splits,
        config,
        split="train",
        encoders=encoders,
        static_scaler=static_scaler,
        stride_weeks=config.rnn_train_stride_weeks,
    )
    validation_bundle = build_sequence_dataset(
        panel_with_splits,
        config,
        split="validation",
        encoders=encoders,
        static_scaler=static_scaler,
        stride_weeks=config.rnn_eval_stride_weeks,
    )
    test_bundle = build_sequence_dataset(
        panel_with_splits,
        config,
        split="test",
        encoders=encoders,
        static_scaler=static_scaler,
        stride_weeks=config.rnn_eval_stride_weeks,
    )
    return {
        "train": train_bundle,
        "validation": validation_bundle,
        "test": test_bundle,
        "encoders": encoders,
        "static_scaler": static_scaler,
    }


def build_rnn_model(config: Notebook3Config, *, n_rgis: int, n_ufs: int) -> Any:
    require_tensorflow()

    history_input = keras.Input(shape=(config.lookback_weeks, 1), name="history")
    future_holiday_input = keras.Input(
        shape=(config.forecast_horizon_weeks, len(HOLIDAY_FEATURE_COLUMNS)),
        name="future_holiday",
    )
    road_length_input = keras.Input(shape=(1,), name="road_length")
    rgi_input = keras.Input(shape=(), dtype="int32", name="rgi_id")
    uf_input = keras.Input(shape=(), dtype="int32", name="uf_id")

    history_encoded = keras.layers.GRU(32, name="history_gru")(history_input)
    future_holiday_flat = keras.layers.Flatten(name="future_holiday_flat")(
        future_holiday_input
    )
    rgi_embedding = keras.layers.Flatten(name="rgi_embedding_flat")(
        keras.layers.Embedding(
            input_dim=max(n_rgis, 1),
            output_dim=min(16, max(4, n_rgis // 4 + 1)),
            name="rgi_embedding",
        )(rgi_input)
    )
    uf_embedding = keras.layers.Flatten(name="uf_embedding_flat")(
        keras.layers.Embedding(
            input_dim=max(n_ufs, 1),
            output_dim=min(8, max(2, n_ufs)),
            name="uf_embedding",
        )(uf_input)
    )

    x = keras.layers.Concatenate(name="model_features")(
        [
            history_encoded,
            future_holiday_flat,
            road_length_input,
            rgi_embedding,
            uf_embedding,
        ]
    )
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    output = keras.layers.Dense(config.forecast_horizon_weeks, name="forecast")(x)

    model = keras.Model(
        inputs={
            "history": history_input,
            "future_holiday": future_holiday_input,
            "road_length": road_length_input,
            "rgi_id": rgi_input,
            "uf_id": uf_input,
        },
        outputs=output,
        name="notebook3_accident_count_rnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_rnn_model(
    train_bundle: dict[str, Any],
    validation_bundle: dict[str, Any],
    config: Notebook3Config,
    *,
    n_rgis: int,
    n_ufs: int,
    epochs: int = 50,
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict[str, Any]:
    require_tensorflow()
    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_rnn_model(config, n_rgis=n_rgis, n_ufs=n_ufs)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    fit_verbose = 1 if show_progress else 0
    if show_progress and TqdmCallback is not None:
        callbacks.append(
            TqdmCallback(
                verbose=0,
                desc="RNN training",
                leave=True,
            )
        )
        fit_verbose = 0
    history = model.fit(
        x=train_bundle["inputs"],
        y=train_bundle["targets"],
        validation_data=(validation_bundle["inputs"], validation_bundle["targets"]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=fit_verbose,
        callbacks=callbacks,
    )
    return {"model": model, "history": history.history}


def generate_rnn_forecasts(
    model: Any,
    bundle: dict[str, Any],
    split: str,
) -> pd.DataFrame:
    require_tensorflow()
    if bundle["targets"].size == 0:
        return empty_forecast_frame()

    predictions = np.maximum(model.predict(bundle["inputs"], verbose=0), 0.0)
    rows: list[dict[str, Any]] = []
    for meta, prediction_row, actual_row in zip(
        bundle["meta"].to_dict(orient="records"),
        predictions,
        bundle["targets"],
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
                    "model": "rnn_global",
                    "split": split,
                    "rgi_id": meta["rgi_id"],
                    "rgi_name": meta["rgi_name"],
                    "uf": meta["uf"],
                    "week_start": week_start,
                    "week_end": week_end,
                    "year_week": year_week,
                    "forecast_batch_start": meta["forecast_batch_start"],
                    "cutoff_week_start": meta["cutoff_week_start"],
                    "horizon_step": step,
                    "prediction": float(prediction),
                    "actual": float(actual),
                    "is_available": True,
                    "metadata": "tensorflow_gru",
                }
            )

    return (
        pd.DataFrame(rows).sort_values(["rgi_id", "week_start"]).reset_index(drop=True)
    )


def combine_forecasts(*forecast_frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [
        frame for frame in forecast_frames if frame is not None and not frame.empty
    ]
    if not non_empty:
        return empty_forecast_frame()
    combined = pd.concat(non_empty, ignore_index=True)
    return combined.sort_values(["model", "split", "rgi_id", "week_start"]).reset_index(
        drop=True
    )


def evaluate_forecasts(
    panel_with_splits: pd.DataFrame,
    combined_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_rows = (
        panel_with_splits.loc[panel_with_splits["split"].isin(["validation", "test"])]
        .groupby(["split", "rgi_id"], as_index=False)
        .size()
        .rename(columns={"size": "expected_rows"})
    )
    overall_expected = (
        panel_with_splits.loc[panel_with_splits["split"].isin(["validation", "test"])]
        .groupby("split")
        .size()
        .to_dict()
    )

    if combined_forecasts.empty:
        return (
            pd.DataFrame(
                columns=[
                    "model",
                    "split",
                    "rmse",
                    "macro_rgi_rmse",
                    "forecast_row_coverage",
                    "n_predictions",
                    "n_target_rows",
                    "n_rgis_with_predictions",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "model",
                    "split",
                    "rgi_id",
                    "rgi_name",
                    "uf",
                    "rmse",
                    "n_predictions",
                    "expected_rows",
                    "forecast_row_coverage",
                ]
            ),
        )

    available = combined_forecasts.loc[
        combined_forecasts["prediction"].notna() & combined_forecasts["actual"].notna()
    ].copy()
    rgi_metrics = available.groupby(
        ["model", "split", "rgi_id", "rgi_name", "uf"], as_index=False
    ).agg(
        rmse=("prediction", lambda s: np.nan),
        n_predictions=("prediction", "size"),
    )
    if not rgi_metrics.empty:
        rmse_values = (
            available.groupby(["model", "split", "rgi_id"], as_index=False)
            .apply(
                lambda frame: float(
                    np.sqrt(np.mean((frame["actual"] - frame["prediction"]) ** 2))
                ),
                include_groups=False,
            )
            .rename(columns={None: "rmse"})
        )
        rgi_metrics = rgi_metrics.drop(columns=["rmse"]).merge(
            rmse_values,
            on=["model", "split", "rgi_id"],
            how="left",
        )

    rgi_metrics = rgi_metrics.merge(expected_rows, on=["split", "rgi_id"], how="left")
    rgi_metrics["forecast_row_coverage"] = (
        rgi_metrics["n_predictions"] / rgi_metrics["expected_rows"]
    )

    overall_metrics = available.groupby(["model", "split"], as_index=False).apply(
        lambda frame: pd.Series(
            {
                "rmse": float(
                    np.sqrt(np.mean((frame["actual"] - frame["prediction"]) ** 2))
                ),
                "n_predictions": int(len(frame)),
                "n_rgis_with_predictions": int(frame["rgi_id"].nunique()),
            }
        ),
        include_groups=False,
    )
    macro_rmse = (
        rgi_metrics.groupby(["model", "split"], as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "macro_rgi_rmse"})
    )
    overall_metrics = overall_metrics.merge(
        macro_rmse, on=["model", "split"], how="left"
    )
    overall_metrics["n_target_rows"] = (
        overall_metrics["split"].map(overall_expected).astype(int)
    )
    overall_metrics["forecast_row_coverage"] = (
        overall_metrics["n_predictions"] / overall_metrics["n_target_rows"]
    )

    return overall_metrics.sort_values(["split", "rmse"]), rgi_metrics.sort_values(
        ["split", "model", "rmse"]
    )
