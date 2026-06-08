"""Municipio-level daily accident count forecasting via Mini-ROCKET + Ridge.

Two-phase pipeline
------------------
featurize
    Reads the zero-padded ``canonical_accidents_by_municipio_day`` gold panel
    (SC only, all BR-101 municipios) and produces an enriched panel with
    precomputed exogenous features: holiday-proximity signals, target-date
    cyclical calendar encodings, weather features, and split assignment.  The
    result is a compact parquet suitable for reuse across training runs.

train
    Reads the enriched panel, builds multivariate lookback sequences
    (accident_count + calendar + holiday + weather channels) on the fly, applies a fitted
    MiniRocketMultivariate transform, trains one RidgeCV model per in-scope
    municipio, evaluates on validation and test splits, and logs everything to
    MLflow.

Usage
-----
    python -m src.ml.municipio_day_regression featurize
    python -m src.ml.municipio_day_regression train
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import holidays
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.etl.datalake import DatalakeAdapter
from src.ml.base import (
    BaseFeaturizationRun,
    BaseMLflowRegressionExperiment,
    _configure_logging,
    _import_mlflow,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Channel counts in the multivariate lookback sequence fed to MiniRocket.
#: Base channel layout:
#:   0  accident_count          — z-scored per municipio using training statistics
#:   1  sin(2π · dow / 7)       — weekly position
#:   2  cos(2π · dow / 7)
#:   3  sin(2π · dom / 31)      — within-month position
#:   4  cos(2π · dom / 31)
#:   5  sin(2π · doy / 365.25)  — annual position
#:   6  cos(2π · doy / 365.25)
#:   7  non_working_day_weight  — 1.0 holiday/weekend, 0.5 bridge, 0.0 workday
#:   8  days_off_ahead          — weighted forward non-working stretch
#:   9  days_off_before         — weighted backward non-working stretch
#: Optional weather channels:
#:   10  target_max_temp_c       — z-scored per municipio using training weather
#:   11  target_min_temp_c       — z-scored per municipio using training weather
#:   12  target_max_wind_speed_kmh
#:   13  target_sunrise_seconds
#:   14  target_sunset_seconds
#:   15  target_precipitation_mm
#:   16  target_precipitation_hours
BASE_SEQUENCE_CHANNELS = 10
WEATHER_SEQUENCE_CHANNELS = 7
N_SEQUENCE_CHANNELS = BASE_SEQUENCE_CHANNELS + WEATHER_SEQUENCE_CHANNELS

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_N_KERNELS = 10_000
DEFAULT_RANDOM_SEED = 101
DEFAULT_RIDGE_ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
DEFAULT_HOLIDAY_MAX_LOOKAHEAD = 30

DEFAULT_CONFIG_ROOT = Path("config/ml/municipio_day")
DEFAULT_FEATURIZATION_CONFIG_PATH = DEFAULT_CONFIG_ROOT / "featurization.yaml"
DEFAULT_EXPERIMENT_CONFIG_PATH = DEFAULT_CONFIG_ROOT / "experiment.yaml"

DEFAULT_GOLD_INPUT_SUBPATH = "gold/br101_sc_municipio_panel"
DEFAULT_FEATURE_OUTPUT_SUBPATH = "gold/ml/municipio_day_features"
DEFAULT_FEATURE_INPUT_SUBPATH = "gold/ml/municipio_day_features"
DEFAULT_EXPERIMENT_OUTPUT_SUBPATH = "gold/ml/municipio_day_regression"
DEFAULT_FORECAST_OUTPUT_SUBPATH = "gold/ml/municipio_day_forecast"
DEFAULT_EXPERIMENT_NAME = "radar-prf-101-municipio-day-ridge"
DEFAULT_REGISTERED_MODEL_NAME = "radar-prf-101-municipio-day"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_WEATHER_INPUT_SUBPATH = "daily_weather"
WEATHER_FEATURE_COLUMNS = [
    "target_max_temp_c",
    "target_min_temp_c",
    "target_max_wind_speed_kmh",
    "target_sunrise_seconds",
    "target_sunset_seconds",
    "target_precipitation_mm",
    "target_precipitation_hours",
]

NUMPY_COMPAT_ALIASES = (
    ("trapz", "trapezoid"),  # np.trapezoid since 2.0
    ("in1d", "isin"),  # removed in 2.0
    ("cumproduct", "cumprod"),  # removed in 2.0
    ("product", "prod"),  # removed in 2.0
    ("sometrue", "any"),  # removed in 2.0
    ("alltrue", "all"),  # removed in 2.0
    ("row_stack", "vstack"),  # removed in 2.0
)


def _patch_numpy_compat_aliases() -> None:
    """Restore NumPy aliases still referenced by numba/aeon at runtime."""
    for old_name, new_name in NUMPY_COMPAT_ALIASES:
        if not hasattr(np, old_name):
            setattr(np, old_name, getattr(np, new_name))  # type: ignore[attr-defined]


try:
    import mlflow.pyfunc as _mlflow_pyfunc

    _MLFLOW_PYTHON_MODEL_BASE = _mlflow_pyfunc.PythonModel
except Exception:  # pragma: no cover - handled when MLflow is actually used
    _MLFLOW_PYTHON_MODEL_BASE = object


class MunicipioDayPyfuncModel(_MLFLOW_PYTHON_MODEL_BASE):
    """MLflow pyfunc wrapper for the Mini-ROCKET + per-municipio Ridge bundle."""

    def load_context(self, context: Any) -> None:
        self.rocket = joblib.load(context.artifacts["rocket"])
        self.models = joblib.load(context.artifacts["models"])
        registry_path = Path(context.artifacts["registry"])
        self.registry = json.loads(registry_path.read_text(encoding="utf-8"))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        _patch_numpy_compat_aliases()
        frame = pd.DataFrame(model_input)
        required = {"codigo_municipio", "sequence", "y_mean", "y_std"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

        sequences = np.stack(
            [
                np.asarray(sequence, dtype=np.float32)
                for sequence in frame["sequence"].tolist()
            ],
            axis=0,
        )
        X_rocket = self.rocket.transform(sequences).astype(np.float32)

        rows: list[dict[str, Any]] = []
        for i, input_row in frame.reset_index(drop=True).iterrows():
            mun_id = str(input_row["codigo_municipio"])
            model = self.models.get(mun_id)
            if model is None:
                raise ValueError(f"No model available for codigo_municipio={mun_id}")
            pred_scaled = float(model.predict(X_rocket[i : i + 1])[0])
            y_mean = float(input_row["y_mean"])
            y_std = float(input_row["y_std"])
            prediction_raw = pred_scaled * y_std + y_mean
            rows.append(
                {
                    "codigo_municipio": mun_id,
                    "prediction_raw": prediction_raw,
                    "predicted_accident_count": max(0.0, prediction_raw),
                    "prediction_scaled": pred_scaled,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Holiday / non-working-day helpers
# ---------------------------------------------------------------------------


def _build_sc_calendar(years: list[int]) -> frozenset[date]:
    """Return the set of SC + national Brazilian public holidays for *years*."""
    cal = holidays.Brazil(state="SC", years=years)
    return frozenset(cal.keys())


def _is_raw_non_working(d: date, cal: frozenset[date]) -> bool:
    """True when *d* is a weekend or an official public holiday."""
    return d.weekday() >= 5 or d in cal


def _day_weight(d: date, cal: frozenset[date]) -> float:
    """Non-working weight for date *d*.

    Returns
    -------
    1.0
        Weekend or public holiday.
    0.5
        Bridge day — a regular weekday that lies between two raw non-working
        days (one on each immediate neighbour side).
    0.0
        Ordinary working day.
    """
    if _is_raw_non_working(d, cal):
        return 1.0
    prev_nw = _is_raw_non_working(d - timedelta(days=1), cal)
    next_nw = _is_raw_non_working(d + timedelta(days=1), cal)
    if prev_nw and next_nw:
        return 0.5
    return 0.0


def _days_off_forward(d: date, cal: frozenset[date], max_look: int) -> float:
    """Sum of day weights for the contiguous non-working stretch starting at d+1.

    Accumulates 1.0 per non-working day and 0.5 per bridge day.  Stops at the
    first ordinary working day (weight == 0.0).
    """
    total = 0.0
    for k in range(1, max_look + 1):
        w = _day_weight(d + timedelta(days=k), cal)
        if w == 0.0:
            break
        total += w
    return total


def _days_off_backward(d: date, cal: frozenset[date], max_look: int) -> float:
    """Sum of day weights for the contiguous non-working stretch ending at d-1."""
    total = 0.0
    for k in range(1, max_look + 1):
        w = _day_weight(d - timedelta(days=k), cal)
        if w == 0.0:
            break
        total += w
    return total


# ---------------------------------------------------------------------------
# Cyclical encoding helpers
# ---------------------------------------------------------------------------


def _cyclical_pair(x: float, period: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * x / period
    return math.sin(angle), math.cos(angle)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _seconds_since_midnight(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    timestamps = pd.to_datetime(series, errors="coerce")
    return (
        timestamps.dt.hour.astype("float64") * 3600.0
        + timestamps.dt.minute.astype("float64") * 60.0
        + timestamps.dt.second.astype("float64")
    )


def _prepare_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    prepared = weather.copy()
    if pd.api.types.is_datetime64_any_dtype(prepared["weather_date"]):
        prepared["accident_date"] = prepared["weather_date"].dt.date
    else:
        prepared["accident_date"] = pd.to_datetime(
            prepared["weather_date"], errors="coerce"
        ).dt.date

    prepared["target_max_temp_c"] = pd.to_numeric(
        prepared.get("temperature_2m_max"), errors="coerce"
    )
    prepared["target_min_temp_c"] = pd.to_numeric(
        prepared.get("temperature_2m_min"), errors="coerce"
    )
    prepared["target_max_wind_speed_kmh"] = pd.to_numeric(
        prepared.get("wind_speed_10m_max"), errors="coerce"
    )
    prepared["target_sunrise_seconds"] = _seconds_since_midnight(
        prepared.get("sunrise")
    )
    prepared["target_sunset_seconds"] = _seconds_since_midnight(prepared.get("sunset"))
    prepared["target_precipitation_mm"] = pd.to_numeric(
        prepared.get("precipitation_sum"), errors="coerce"
    )
    prepared["target_precipitation_hours"] = pd.to_numeric(
        prepared.get("precipitation_hours"), errors="coerce"
    )

    return prepared[
        ["codigo_municipio", "accident_date", *WEATHER_FEATURE_COLUMNS]
    ].drop_duplicates(["codigo_municipio", "accident_date"], keep="last")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MunicipioDayFeaturizationConfig:
    gold_input_subpath: str = DEFAULT_GOLD_INPUT_SUBPATH
    include_weather_features: bool = True
    weather_input_subpath: str = DEFAULT_WEATHER_INPUT_SUBPATH
    output_subpath: str = DEFAULT_FEATURE_OUTPUT_SUBPATH
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    holiday_country: str = "BR"
    holiday_state: str = "SC"
    holiday_max_lookahead: int = DEFAULT_HOLIDAY_MAX_LOOKAHEAD
    train_start_year: int = 2017
    train_end_year: int = 2023
    validation_start_year: int = 2024
    validation_end_year: int = 2024
    test_start_year: int = 2025
    test_end_year: int = 2026
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    @classmethod
    def from_yaml(
        cls, path: Path, project_root: Path = PROJECT_ROOT
    ) -> MunicipioDayFeaturizationConfig:
        raw: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        # Env overrides
        raw["gold_input_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH",
            raw.get("gold_input_subpath", DEFAULT_GOLD_INPUT_SUBPATH),
        )
        raw["output_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_FEATURE_OUTPUT_SUBPATH",
            raw.get("output_subpath", DEFAULT_FEATURE_OUTPUT_SUBPATH),
        )
        raw["weather_input_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_WEATHER_INPUT_SUBPATH",
            raw.get("weather_input_subpath", DEFAULT_WEATHER_INPUT_SUBPATH),
        )
        raw["include_weather_features"] = _env_bool(
            "ML_MUNICIPIO_DAY_INCLUDE_WEATHER_FEATURES",
            bool(raw.get("include_weather_features", True)),
        )
        if "lookback_days" in raw:
            raw["lookback_days"] = int(
                os.environ.get("ML_MUNICIPIO_DAY_LOOKBACK_DAYS", raw["lookback_days"])
            )
        raw.pop("project_root", None)
        return cls(
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
            project_root=project_root,
        )


@dataclass
class MunicipioDayExperimentConfig:
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    output_subpath: str = DEFAULT_EXPERIMENT_OUTPUT_SUBPATH
    feature_input_subpath: str = DEFAULT_FEATURE_INPUT_SUBPATH
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME
    champion_alias: str = DEFAULT_CHAMPION_ALIAS
    register_model: bool = True
    n_kernels: int = DEFAULT_N_KERNELS
    random_seed: int = DEFAULT_RANDOM_SEED
    ridge_alphas: list[float] = field(
        default_factory=lambda: list(DEFAULT_RIDGE_ALPHAS)
    )
    alpha_selection_split: str = "validation"
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    @classmethod
    def from_yaml(
        cls, path: Path, project_root: Path = PROJECT_ROOT
    ) -> MunicipioDayExperimentConfig:
        raw: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        raw["tracking_uri"] = os.environ.get(
            "MLFLOW_TRACKING_URI", raw.get("tracking_uri", "http://mlflow:5000")
        )
        raw["experiment_name"] = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            raw.get("experiment_name", DEFAULT_EXPERIMENT_NAME),
        )
        raw["run_name"] = os.environ.get("MLFLOW_RUN_NAME", raw.get("run_name"))
        raw["output_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_EXPERIMENT_OUTPUT_SUBPATH",
            raw.get("output_subpath", DEFAULT_EXPERIMENT_OUTPUT_SUBPATH),
        )
        raw["feature_input_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH",
            raw.get("feature_input_subpath", DEFAULT_FEATURE_INPUT_SUBPATH),
        )
        raw["registered_model_name"] = os.environ.get(
            "ML_MUNICIPIO_DAY_REGISTERED_MODEL_NAME",
            raw.get("registered_model_name", DEFAULT_REGISTERED_MODEL_NAME),
        )
        raw["champion_alias"] = os.environ.get(
            "ML_MUNICIPIO_DAY_CHAMPION_ALIAS",
            raw.get("champion_alias", DEFAULT_CHAMPION_ALIAS),
        )
        raw["register_model"] = _env_bool(
            "ML_MUNICIPIO_DAY_REGISTER_MODEL",
            bool(raw.get("register_model", True)),
        )
        raw["alpha_selection_split"] = os.environ.get(
            "ML_MUNICIPIO_DAY_ALPHA_SELECTION_SPLIT",
            raw.get("alpha_selection_split", "validation"),
        )
        raw.pop("project_root", None)
        return cls(
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
            project_root=project_root,
        )


def _default_experiment_run_name(cfg: MunicipioDayExperimentConfig) -> str:
    return (
        "municipio_day"
        "__model=shared_minirocket_per_mun_ridge"
        f"__alpha_select={cfg.alpha_selection_split}"
        f"__kernels={cfg.n_kernels}"
        f"__seed={cfg.random_seed}"
    )


@dataclass
class MunicipioDayForecastConfig:
    tracking_uri: str = "http://mlflow:5000"
    model_uri: str = f"models:/{DEFAULT_REGISTERED_MODEL_NAME}@{DEFAULT_CHAMPION_ALIAS}"
    feature_input_subpath: str = DEFAULT_FEATURE_INPUT_SUBPATH
    output_subpath: str = DEFAULT_FORECAST_OUTPUT_SUBPATH
    horizon_days: int = 30
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    @classmethod
    def from_yaml(
        cls, path: Path, project_root: Path = PROJECT_ROOT
    ) -> MunicipioDayForecastConfig:
        raw: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        raw["tracking_uri"] = os.environ.get(
            "MLFLOW_TRACKING_URI", raw.get("tracking_uri", "http://mlflow:5000")
        )
        registered_model_name = os.environ.get(
            "ML_MUNICIPIO_DAY_REGISTERED_MODEL_NAME",
            raw.get("registered_model_name", DEFAULT_REGISTERED_MODEL_NAME),
        )
        champion_alias = os.environ.get(
            "ML_MUNICIPIO_DAY_CHAMPION_ALIAS",
            raw.get("champion_alias", DEFAULT_CHAMPION_ALIAS),
        )
        raw["model_uri"] = os.environ.get(
            "ML_MUNICIPIO_DAY_MODEL_URI",
            raw.get("model_uri", f"models:/{registered_model_name}@{champion_alias}"),
        )
        raw["feature_input_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH",
            raw.get("feature_input_subpath", DEFAULT_FEATURE_INPUT_SUBPATH),
        )
        raw["output_subpath"] = os.environ.get(
            "ML_MUNICIPIO_DAY_FORECAST_OUTPUT_SUBPATH",
            raw.get("output_subpath", DEFAULT_FORECAST_OUTPUT_SUBPATH),
        )
        raw["horizon_days"] = int(
            os.environ.get(
                "ML_MUNICIPIO_DAY_FORECAST_HORIZON_DAYS",
                raw.get("horizon_days", 30),
            )
        )
        raw.pop("project_root", None)
        return cls(
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
            project_root=project_root,
        )


# ---------------------------------------------------------------------------
# Low-level sequence construction (vectorised over dates, looped over municipios)
# ---------------------------------------------------------------------------


def _precompute_calendar_channel_arrays(all_dates: list[date]) -> np.ndarray:
    """Build a (6, T) float32 array of cyclical calendar values for *all_dates*.

    Channels (sequence indices 1-6, index 0 being accident_count):
        0  sin(2π · dow / 7)
        1  cos(2π · dow / 7)
        2  sin(2π · dom / 31)
        3  cos(2π · dom / 31)
        4  sin(2π · doy / 365.25)
        5  cos(2π · doy / 365.25)
    """
    T = len(all_dates)
    out = np.empty((6, T), dtype=np.float32)
    for i, d in enumerate(all_dates):
        dow = d.weekday()
        dom = d.day
        doy = d.timetuple().tm_yday
        out[0, i], out[1, i] = _cyclical_pair(dow, 7.0)
        out[2, i], out[3, i] = _cyclical_pair(dom, 31.0)
        out[4, i], out[5, i] = _cyclical_pair(doy, 365.25)
    return out


def _precompute_exo_channel_arrays(
    all_dates: list[date],
    panel: pd.DataFrame,
) -> np.ndarray:
    """Build a (3, T) float32 array of exogenous channel values for *all_dates*.

    Channels (sequence indices 7-9):
        0  non_working_day_weight  (1.0 holiday/weekend, 0.5 bridge, 0.0 workday)
        1  days_off_ahead          (weighted forward non-working stretch)
        2  days_off_before         (weighted backward non-working stretch)

    Values are date-specific (identical across municipios) and are read from
    the enriched panel stored in the datalake.
    """
    date_exo = (
        panel[
            [
                "accident_date",
                "non_working_day_weight",
                "days_off_ahead",
                "days_off_before",
            ]
        ]
        .drop_duplicates("accident_date")
        .set_index("accident_date")
    )
    T = len(all_dates)
    out = np.zeros((3, T), dtype=np.float32)
    for i, d in enumerate(all_dates):
        if d in date_exo.index:
            row = date_exo.loc[d]
            out[0, i] = row["non_working_day_weight"]
            out[1, i] = row["days_off_ahead"]
            out[2, i] = row["days_off_before"]
    return out


def _precompute_weather_channel_arrays(
    all_dates: list[date],
    mun_panel: pd.DataFrame,
    train_years: set[int],
) -> np.ndarray:
    """Build normalized weather channel values for one municipio.

    Channels are aligned to ``all_dates`` and z-scored using only the training
    years for that municipio. Missing values are imputed with the training mean
    for the same weather feature before normalization.
    """
    T = len(all_dates)
    out = np.zeros((len(WEATHER_FEATURE_COLUMNS), T), dtype=np.float32)
    if not set(WEATHER_FEATURE_COLUMNS).issubset(mun_panel.columns):
        return out

    weather_frame = mun_panel[["accident_date", *WEATHER_FEATURE_COLUMNS]].set_index(
        "accident_date"
    )
    train_mask = np.array([d.year in train_years for d in all_dates], dtype=bool)

    for channel_index, column_name in enumerate(WEATHER_FEATURE_COLUMNS):
        values = (
            weather_frame[column_name]
            .reindex(all_dates)
            .astype(float)
            .to_numpy(dtype=np.float32)
        )
        train_values = values[train_mask]
        train_values = train_values[np.isfinite(train_values)]
        if len(train_values) == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(train_values.mean())
            std = float(train_values.std())
            if std < 1e-8:
                std = 1.0
        clean_values = np.where(np.isfinite(values), values, mean).astype(np.float32)
        out[channel_index] = (clean_values - mean) / std

    return out


def _sliding_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a (T-window, window) array of consecutive windows of *arr* (copy)."""
    # sliding_window_view returns T-window+1 rows; drop the last one because
    # window[T-window] covers arr[T-window:T] and its target arr[T] is out of bounds.
    return np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)[
        :-1
    ].copy()


def _build_municipio_sequences(
    ac_series: pd.Series,
    all_dates: list[date],
    cal_channels: np.ndarray,
    exo_channels: np.ndarray,
    weather_channels: np.ndarray | None,
    lookback_days: int,
    train_years: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequence data for one municipio.

    Parameters
    ----------
    ac_series:
        accident_count indexed by date; missing dates treated as 0.
    all_dates:
        Sorted list of every date in the full panel date range.
    cal_channels:
        (6, T) calendar channel array from :func:`_precompute_calendar_channel_arrays`.
    exo_channels:
        (3, T) exogenous channel array from :func:`_precompute_exo_channel_arrays`.
    lookback_days:
        Length of each lookback window L.
    train_years:
        Set of years whose dates provide the normalization statistics for
        accident_count (z-score, computed only from training data).

    Returns
    -------
    X_seq : (N, n_channels, L)  float32
        Multivariate sequences where N = T - L.  Channel 0 is z-scored.
    y_raw : (N,)  float32
        Raw accident counts at the target dates (not z-scored).
    target_date_indices : (N,)  int
        Positions in *all_dates* of each target date.
    ac_mean : float
        Per-municipio mean accident count over training years (for inverse scaling).
    ac_std : float
        Per-municipio std accident count over training years (for inverse scaling).
    """
    T = len(all_dates)
    L = lookback_days

    # Build accident count array aligned to all_dates
    ac_array = ac_series.reindex(all_dates, fill_value=0.0).values.astype(np.float32)

    # Z-score per municipio using training period statistics
    train_mask = np.array([d.year in train_years for d in all_dates], dtype=bool)
    train_ac = ac_array[train_mask]
    ac_mean = float(train_ac.mean()) if train_mask.any() else 0.0
    ac_std = float(train_ac.std()) if train_mask.any() else 1.0
    if ac_std < 1e-8:
        ac_std = 1.0
    ac_norm = (ac_array - ac_mean) / ac_std

    # Sliding windows: (T-L, L) — window i covers dates [i, i+L-1], target is i+L
    ac_windows = _sliding_windows(ac_norm, L)  # (T-L, L)
    cal_windows = np.stack(
        [_sliding_windows(cal_channels[ch], L) for ch in range(6)],
        axis=1,
    )  # (T-L, 6, L)
    exo_windows = np.stack(
        [_sliding_windows(exo_channels[ch], L) for ch in range(3)],
        axis=1,
    )  # (T-L, 3, L)
    channel_blocks = [ac_windows[:, np.newaxis, :], cal_windows, exo_windows]
    if weather_channels is not None:
        weather_windows = np.stack(
            [
                _sliding_windows(weather_channels[ch], L)
                for ch in range(len(WEATHER_FEATURE_COLUMNS))
            ],
            axis=1,
        )  # (T-L, 7, L)
        channel_blocks.append(weather_windows)

    # Stack into (T-L, n_channels, L)
    X_seq = np.concatenate(
        channel_blocks,
        axis=1,
    )

    target_date_indices = np.arange(L, T, dtype=np.int32)  # (T-L,)

    return X_seq, ac_array[L:], target_date_indices, ac_mean, ac_std


# ---------------------------------------------------------------------------
# Featurization phase
# ---------------------------------------------------------------------------


class MunicipioDayFeaturizer(BaseFeaturizationRun):
    """Enrich the gold day panel with exogenous features and persist it.

    Produces
    --------
    panel_with_features.parquet
        One row per (codigo_municipio, accident_date).  Includes the original
        columns plus: split, non_working_day_weight, days_off_ahead,
        days_off_before, and daily weather features.
    feature_registry.json
        Configuration snapshot and ordered municipio code list used by the
        train phase.
    """

    def __init__(
        self,
        config: MunicipioDayFeaturizationConfig | None = None,
    ) -> None:
        self.feat_config = config or MunicipioDayFeaturizationConfig()
        super().__init__(
            config=asdict(self.feat_config),
            job_name="municipio_day_featurizer",
        )
        self.datalake = DatalakeAdapter.from_env(
            project_root=self.feat_config.project_root
        )
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging = Path(self._tmp.name)

    def extract(self) -> dict[str, pd.DataFrame]:
        cfg = self.feat_config
        panel_path = self.datalake.stage_directory(
            f"{cfg.gold_input_subpath}/canonical_accidents_by_municipio_day",
            self._staging / "panel",
        )
        scope_path = self.datalake.stage_directory(
            f"{cfg.gold_input_subpath}/municipios_in_scope",
            self._staging / "scope",
        )
        panel = pd.read_parquet(str(panel_path))
        scope = pd.read_parquet(str(scope_path))
        weather: pd.DataFrame | None = None
        if cfg.include_weather_features:
            weather_path = self.datalake.stage_directory(
                f"{cfg.gold_input_subpath}/{cfg.weather_input_subpath}",
                self._staging / "weather",
            )
            weather = pd.read_parquet(str(weather_path))
        self.logger.info(
            "Loaded panel: %d rows, %d municipios in scope, %d weather rows",
            len(panel),
            len(scope),
            len(weather) if weather is not None else 0,
        )
        return {"panel": panel, "scope": scope, "weather": weather}

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        cfg = self.feat_config
        panel: pd.DataFrame = data["panel"].copy()
        scope: pd.DataFrame = data["scope"]
        weather: pd.DataFrame | None = data["weather"]

        # Ordered municipio list (deterministic, lexicographic by code)
        municipio_codes: list[str] = sorted(
            scope["codigo_municipio"].dropna().unique().tolist()
        )

        # Convert accident_date to Python date if needed
        if pd.api.types.is_datetime64_any_dtype(panel["accident_date"]):
            panel["accident_date"] = panel["accident_date"].dt.date
        elif not isinstance(panel["accident_date"].iloc[0], date):
            panel["accident_date"] = pd.to_datetime(panel["accident_date"]).dt.date

        # Holiday calendar covering all years present in the panel
        all_years = sorted(panel["year"].dropna().unique().astype(int).tolist())
        # Extend by one year each side for the forward/backward lookahead
        cal = _build_sc_calendar(list(range(min(all_years) - 1, max(all_years) + 2)))
        max_look = cfg.holiday_max_lookahead

        # Compute exogenous features row-by-row (fast enough for ~100k rows)
        non_working_weights: list[float] = []
        days_off_ahead_vals: list[float] = []
        days_off_before_vals: list[float] = []

        for d in panel["accident_date"]:
            non_working_weights.append(_day_weight(d, cal))
            days_off_ahead_vals.append(_days_off_forward(d, cal, max_look))
            days_off_before_vals.append(_days_off_backward(d, cal, max_look))

        panel["non_working_day_weight"] = non_working_weights
        panel["days_off_ahead"] = days_off_ahead_vals
        panel["days_off_before"] = days_off_before_vals

        weather_coverage: float | None = None
        if cfg.include_weather_features:
            if weather is None:
                raise ValueError(
                    "Weather features are enabled but no weather data was loaded"
                )
            weather_features = _prepare_weather_features(weather)
            panel = panel.merge(
                weather_features,
                on=["codigo_municipio", "accident_date"],
                how="left",
                validate="many_to_one",
            )
            weather_coverage = float(
                panel[WEATHER_FEATURE_COLUMNS].notna().all(axis=1).mean()
            )
            self.logger.info(
                "Joined weather features: %.2f%% complete rows",
                weather_coverage * 100.0,
            )
        else:
            self.logger.info("Weather features disabled for this featurization run")

        # Assign split
        train_years = set(range(cfg.train_start_year, cfg.train_end_year + 1))
        val_years = set(range(cfg.validation_start_year, cfg.validation_end_year + 1))
        test_years = set(range(cfg.test_start_year, cfg.test_end_year + 1))

        def _assign_split(yr: int) -> str:
            if yr in train_years:
                return "train"
            if yr in val_years:
                return "validation"
            if yr in test_years:
                return "test"
            return "out_of_range"

        panel["split"] = panel["year"].apply(_assign_split)

        # Keep only the columns needed downstream
        keep_cols = [
            "codigo_municipio",
            "nome_municipio",
            "sg_uf",
            "cd_rgint",
            "nm_rgint",
            "accident_date",
            "year",
            "month",
            "day",
            "accident_count",
            "split",
            "non_working_day_weight",
            "days_off_ahead",
            "days_off_before",
        ]
        if cfg.include_weather_features:
            keep_cols.extend(WEATHER_FEATURE_COLUMNS)
        panel = panel[[c for c in keep_cols if c in panel.columns]]
        panel = panel.sort_values(["codigo_municipio", "accident_date"]).reset_index(
            drop=True
        )

        registry = {
            "municipio_codes": municipio_codes,
            "n_municipios": len(municipio_codes),
            "n_sequence_channels": (
                N_SEQUENCE_CHANNELS
                if cfg.include_weather_features
                else BASE_SEQUENCE_CHANNELS
            ),
            "include_weather_features": cfg.include_weather_features,
            "model_variant": (
                "municipio_day_with_weather"
                if cfg.include_weather_features
                else "municipio_day_without_weather"
            ),
            "sequence_channel_descriptions": {
                "0": "accident_count (z-scored per municipio, training stats)",
                "1": "sin(2pi * day_of_week / 7)",
                "2": "cos(2pi * day_of_week / 7)",
                "3": "sin(2pi * day_of_month / 31)",
                "4": "cos(2pi * day_of_month / 31)",
                "5": "sin(2pi * day_of_year / 365.25)",
                "6": "cos(2pi * day_of_year / 365.25)",
                "7": "non_working_day_weight (1.0=holiday/weekend, 0.5=bridge, 0.0=working)",
                "8": "days_off_ahead (weighted forward non-working stretch)",
                "9": "days_off_before (weighted backward non-working stretch)",
            },
            "weather_input_subpath": cfg.weather_input_subpath,
            "weather_feature_columns": (
                WEATHER_FEATURE_COLUMNS if cfg.include_weather_features else []
            ),
            "weather_complete_row_share": weather_coverage,
            "holiday_country": cfg.holiday_country,
            "holiday_state": cfg.holiday_state,
            "holiday_max_lookahead": cfg.holiday_max_lookahead,
            "lookback_days": cfg.lookback_days,
            "split_config": {
                "train": [cfg.train_start_year, cfg.train_end_year],
                "validation": [cfg.validation_start_year, cfg.validation_end_year],
                "test": [cfg.test_start_year, cfg.test_end_year],
            },
            "target_column": "accident_count",
        }
        if cfg.include_weather_features:
            registry["sequence_channel_descriptions"].update(
                {
                    "10": "target_max_temp_c (z-scored per municipio, training weather stats)",
                    "11": "target_min_temp_c (z-scored per municipio, training weather stats)",
                    "12": "target_max_wind_speed_kmh (z-scored per municipio, training weather stats)",
                    "13": "target_sunrise_seconds (z-scored per municipio, training weather stats)",
                    "14": "target_sunset_seconds (z-scored per municipio, training weather stats)",
                    "15": "target_precipitation_mm (z-scored per municipio, training weather stats)",
                    "16": "target_precipitation_hours (z-scored per municipio, training weather stats)",
                }
            )

        split_counts = panel["split"].value_counts().to_dict()
        self.logger.info("Split row counts: %s", split_counts)
        return {"panel": panel, "registry": registry}

    def load(self, data: dict[str, Any]) -> str:
        cfg = self.feat_config
        output_staging = self._staging / "output"
        output_staging.mkdir(parents=True, exist_ok=True)

        panel: pd.DataFrame = data["panel"]
        registry: dict[str, Any] = data["registry"]

        panel_local = output_staging / "panel_with_features.parquet"
        registry_local = output_staging / "feature_registry.json"

        panel.to_parquet(str(panel_local), index=False)
        registry_local.write_text(
            json.dumps(registry, indent=2, default=str),
            encoding="utf-8",
        )

        panel_dest = self.datalake.persist_file(
            panel_local,
            f"{cfg.output_subpath}/panel_with_features.parquet",
        )
        registry_dest = self.datalake.persist_file(
            registry_local,
            f"{cfg.output_subpath}/feature_registry.json",
        )
        self.logger.info(
            "Featurization output persisted: panel=%s registry=%s",
            panel_dest,
            registry_dest,
        )
        return panel_dest

    def cleanup(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Training / regression experiment
# ---------------------------------------------------------------------------


def _load_feature_data(
    datalake: DatalakeAdapter,
    subpath: str,
    staging_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel_path = datalake.stage_file(
        f"{subpath}/panel_with_features.parquet",
        staging_dir / "features",
    )
    registry_path = datalake.stage_file(
        f"{subpath}/feature_registry.json",
        staging_dir / "features",
    )
    panel = pd.read_parquet(str(panel_path))
    registry: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    return panel, registry


def _assemble_split_arrays(
    panel: pd.DataFrame,
    registry: dict[str, Any],
    lookback_days: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return combined arrays for *split*.

    Exogenous features are included inside the multivariate sequences and fed
    directly to MiniRocketMultivariate — no separate scalar post-processing.

    Drops samples whose lookback window would reach before the panel start.
    """
    municipio_codes: list[str] = registry["municipio_codes"]
    n_mun = len(municipio_codes)
    mun_index = {c: i for i, c in enumerate(municipio_codes)}
    n_sequence_channels = int(
        registry.get("n_sequence_channels", BASE_SEQUENCE_CHANNELS)
    )

    per_municipio = _assemble_per_municipio_arrays(
        panel, registry, lookback_days, split
    )

    if not per_municipio:
        L = lookback_days
        return (
            np.empty((0, n_sequence_channels, L), dtype=np.float32),
            np.empty((0, n_mun), dtype=np.float32),
            np.empty(0, dtype=np.float32),  # y_scaled
            np.empty(0, dtype=np.float32),  # y_raw
            np.empty(0, dtype=np.float32),  # y_mean
            np.empty(0, dtype=np.float32),  # y_std
        )

    seq_list: list[np.ndarray] = []
    oh_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    y_raw_list: list[np.ndarray] = []
    y_mean_list: list[np.ndarray] = []
    y_std_list: list[np.ndarray] = []

    for mun_id in municipio_codes:
        mun_data = per_municipio.get(mun_id)
        if mun_data is None:
            continue

        n_split = len(mun_data["y_scaled"])
        oh = np.zeros((n_split, n_mun), dtype=np.float32)
        oh[:, mun_index[mun_id]] = 1.0

        seq_list.append(mun_data["X_seq"])
        oh_list.append(oh)
        y_list.append(mun_data["y_scaled"])
        y_raw_list.append(mun_data["y_raw"])
        y_mean_list.append(np.full(n_split, mun_data["y_mean"], dtype=np.float32))
        y_std_list.append(np.full(n_split, mun_data["y_std"], dtype=np.float32))

    return (
        np.concatenate(seq_list, axis=0),
        np.concatenate(oh_list, axis=0),
        np.concatenate(y_list, axis=0),  # y_scaled (used for training)
        np.concatenate(y_raw_list, axis=0),  # y_raw (used for evaluation)
        np.concatenate(y_mean_list, axis=0),  # per-sample municipio mean
        np.concatenate(y_std_list, axis=0),  # per-sample municipio std
    )


def _assemble_per_municipio_arrays(
    panel: pd.DataFrame,
    registry: dict[str, Any],
    lookback_days: int,
    split: str,
) -> dict[str, dict[str, Any]]:
    """Return per-municipio sequence arrays for *split*.

    The returned mapping is keyed by ``codigo_municipio``.  Each value contains
    ``X_seq``, ``y_scaled``, ``y_raw``, ``y_mean``, ``y_std``, and ``n``.
    """
    municipio_codes: list[str] = registry["municipio_codes"]
    include_weather_features = bool(registry.get("include_weather_features", False))

    if pd.api.types.is_datetime64_any_dtype(panel["accident_date"]):
        panel = panel.copy()
        panel["accident_date"] = panel["accident_date"].dt.date
    elif not isinstance(panel["accident_date"].iloc[0], date):
        panel = panel.copy()
        panel["accident_date"] = pd.to_datetime(panel["accident_date"]).dt.date

    all_dates: list[date] = sorted(panel["accident_date"].unique().tolist())
    train_years: set[int] = set(
        panel.loc[panel["split"] == "train", "year"].unique().astype(int).tolist()
    )

    cal_channels = _precompute_calendar_channel_arrays(all_dates)
    exo_channels = _precompute_exo_channel_arrays(all_dates, panel)

    per_municipio: dict[str, dict[str, Any]] = {}

    for mun_id in municipio_codes:
        mun_panel = panel[panel["codigo_municipio"] == mun_id].set_index(
            "accident_date"
        )
        ac_series = mun_panel["accident_count"].astype(float)
        weather_channels = (
            _precompute_weather_channel_arrays(
                all_dates,
                mun_panel.reset_index(),
                train_years,
            )
            if include_weather_features
            else None
        )

        X_seq, y_raw_all, target_idx, ac_mean, ac_std = _build_municipio_sequences(
            ac_series,
            all_dates,
            cal_channels,
            exo_channels,
            weather_channels,
            lookback_days,
            train_years,
        )
        y_scaled_all = (y_raw_all - ac_mean) / ac_std

        # Filter to the requested split
        target_dates_all = [all_dates[i] for i in target_idx]
        split_mask = np.array(
            [
                mun_panel["split"].get(d, "out_of_range") == split
                for d in target_dates_all
            ],
            dtype=bool,
        )
        if not split_mask.any():
            continue

        X_seq_split = X_seq[split_mask]
        y_split = y_scaled_all[split_mask].astype(np.float32)
        y_raw_split = y_raw_all[split_mask]
        n_split = int(split_mask.sum())

        per_municipio[mun_id] = {
            "X_seq": X_seq_split.astype(np.float32, copy=False),
            "y_scaled": y_split,
            "y_raw": y_raw_split.astype(np.float32, copy=False),
            "y_mean": float(ac_mean),
            "y_std": float(ac_std),
            "n": n_split,
        }

    return per_municipio


class MunicipioDayRegressionExperiment(BaseMLflowRegressionExperiment):
    """Shared Mini-ROCKET + per-municipio RidgeCV experiment for SC accidents."""

    def __init__(
        self,
        config: MunicipioDayExperimentConfig | None = None,
    ) -> None:
        self.exp_config = config or MunicipioDayExperimentConfig()
        super().__init__(
            config=asdict(self.exp_config),
            job_name="municipio_day_regression",
            tracking_uri=self.exp_config.tracking_uri,
            experiment_name=self.exp_config.experiment_name,
            run_name=self.exp_config.run_name
            or _default_experiment_run_name(self.exp_config),
        )
        self.datalake = DatalakeAdapter.from_env(
            project_root=self.exp_config.project_root
        )
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging = Path(self._tmp.name)

    # -- BaseMLflowRegressionExperiment lifecycle --

    def extract(self) -> dict[str, Any]:
        panel, registry = _load_feature_data(
            self.datalake,
            self.exp_config.feature_input_subpath,
            self._staging,
        )
        self.logger.info(
            "Loaded enriched panel: %d rows, %d municipios",
            len(panel),
            registry["n_municipios"],
        )
        return {"panel": panel, "registry": registry}

    def featurize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build per-municipio sequences and fit the shared MiniRocket."""
        _patch_numpy_compat_aliases()
        try:
            from aeon.transformations.collection.convolution_based import (
                MiniRocket,
            )
        except ImportError as exc:
            raise ImportError(
                "aeon is required for the Mini-ROCKET featurization step. "
                "Install it with: uv add aeon"
            ) from exc

        cfg = self.exp_config
        panel: pd.DataFrame = data["panel"]
        registry: dict[str, Any] = data["registry"]
        lookback_days: int = registry["lookback_days"]

        per_mun_seqs: dict[str, dict[str, dict[str, Any]]] = {}
        split_counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            self.logger.info(
                "Building %s sequences (lookback=%d days)…", split, lookback_days
            )
            per_mun_seqs[split] = _assemble_per_municipio_arrays(
                panel, registry, lookback_days, split
            )
            split_counts[split] = int(
                sum(mun_data["n"] for mun_data in per_mun_seqs[split].values())
            )
            self.logger.info(
                "Built %d %s samples across %d municipios",
                split_counts[split],
                split,
                len(per_mun_seqs[split]),
            )

        train_sequences = [
            mun_data["X_seq"]
            for mun_data in per_mun_seqs["train"].values()
            if len(mun_data["y_scaled"]) > 0
        ]
        if not train_sequences:
            raise ValueError("No training sequences available for MiniRocket fit")
        X_seq_train = np.concatenate(train_sequences, axis=0)

        self.logger.info(
            "Fitting MiniRocket (n_kernels=%d, seed=%d)…",
            cfg.n_kernels,
            cfg.random_seed,
        )
        rocket = MiniRocket(
            n_kernels=cfg.n_kernels,
            random_state=cfg.random_seed,
        )
        rocket.fit(X_seq_train)

        return {
            "per_mun_seqs": per_mun_seqs,
            "rocket": rocket,
            "registry": registry,
            "split_counts": split_counts,
        }

    def train(self, feature_data: dict[str, Any]) -> dict[str, Any]:
        cfg = self.exp_config
        rocket = feature_data["rocket"]
        per_mun_train = feature_data["per_mun_seqs"]["train"]
        alpha_selection_split = cfg.alpha_selection_split
        per_mun_alpha_selection = feature_data["per_mun_seqs"].get(
            alpha_selection_split, {}
        )
        municipio_codes: list[str] = feature_data["registry"]["municipio_codes"]

        models: dict[str, Ridge | RidgeCV] = {}
        alphas: dict[str, float] = {}
        train_sample_counts: dict[str, int] = {}
        alpha_selection_scores: dict[str, float | None] = {}
        n_rocket_features: int | None = None

        for mun_id in municipio_codes:
            train_data = per_mun_train.get(mun_id)
            if train_data is None or len(train_data["y_scaled"]) == 0:
                self.logger.warning(
                    "No training data for municipio=%s; skipping", mun_id
                )
                continue

            X_rocket = rocket.transform(train_data["X_seq"]).astype(np.float32)
            if n_rocket_features is None:
                n_rocket_features = int(X_rocket.shape[1])

            selection_data = per_mun_alpha_selection.get(mun_id)
            if selection_data is not None and len(selection_data["y_scaled"]) > 0:
                X_select = rocket.transform(selection_data["X_seq"]).astype(np.float32)
                y_select = selection_data["y_scaled"]
                best_score = float("inf")
                best_alpha = float(cfg.ridge_alphas[0])
                best_model: Ridge | None = None
                for alpha in cfg.ridge_alphas:
                    candidate = Ridge(alpha=float(alpha))
                    candidate.fit(X_rocket, train_data["y_scaled"])
                    y_select_pred = candidate.predict(X_select)
                    score = float(mean_squared_error(y_select, y_select_pred))
                    if score < best_score:
                        best_score = score
                        best_alpha = float(alpha)
                        best_model = candidate
                if best_model is None:
                    raise ValueError(f"No Ridge model fit for municipio={mun_id}")
                ridge = best_model
                alpha = best_alpha
                alpha_selection_scores[mun_id] = best_score
            else:
                self.logger.warning(
                    "No %s data for municipio=%s; falling back to RidgeCV LOOCV",
                    alpha_selection_split,
                    mun_id,
                )
                ridge = RidgeCV(
                    alphas=cfg.ridge_alphas,
                    scoring="neg_mean_squared_error",
                    cv=None,
                )
                ridge.fit(X_rocket, train_data["y_scaled"])
                alpha = float(ridge.alpha_)
                alpha_selection_scores[mun_id] = None

            models[mun_id] = ridge
            alphas[mun_id] = alpha
            train_sample_counts[mun_id] = int(len(train_data["y_scaled"]))
            self.logger.info(
                "municipio=%s  alpha=%.4g  n_train=%d  alpha_selection=%s",
                mun_id,
                alpha,
                train_sample_counts[mun_id],
                alpha_selection_split
                if alpha_selection_scores[mun_id] is not None
                else "loocv",
            )

        if not models:
            raise ValueError("No per-municipio Ridge models were trained")

        return {
            "models": models,
            "alphas": alphas,
            "alpha_selection_scores": alpha_selection_scores,
            "alpha_selection_split": alpha_selection_split,
            "train_sample_counts": train_sample_counts,
            "n_rocket_features": n_rocket_features,
        }

    def evaluate(
        self, feature_data: dict[str, Any], training_output: dict[str, Any]
    ) -> dict[str, Any]:
        rocket = feature_data["rocket"]
        models: dict[str, Ridge | RidgeCV] = training_output["models"]
        per_mun_seqs = feature_data["per_mun_seqs"]
        metrics: dict[str, Any] = {}

        for split in ("train", "validation", "test"):
            y_true_all: list[np.ndarray] = []
            y_pred_all: list[np.ndarray] = []
            per_municipio: dict[str, dict[str, float]] = {}

            for mun_id, model in models.items():
                split_data = per_mun_seqs[split].get(mun_id)
                if split_data is None or len(split_data["y_raw"]) == 0:
                    continue

                X_rocket = rocket.transform(split_data["X_seq"]).astype(np.float32)
                y_pred_scaled = model.predict(X_rocket)
                y_pred = y_pred_scaled * split_data["y_std"] + split_data["y_mean"]
                y_raw = split_data["y_raw"]

                mae_m = float(mean_absolute_error(y_raw, y_pred))
                rmse_m = float(mean_squared_error(y_raw, y_pred) ** 0.5)
                r2_m = float(r2_score(y_raw, y_pred))
                per_municipio[mun_id] = {
                    "mae": mae_m,
                    "rmse": rmse_m,
                    "r2": r2_m,
                    "n": float(len(y_raw)),
                }

                y_true_all.append(y_raw)
                y_pred_all.append(y_pred)

            if not y_true_all:
                self.logger.warning("Skipping evaluation for empty split=%s", split)
                continue

            y_raw = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
            mae = float(mean_absolute_error(y_raw, y_pred))
            rmse = float(mean_squared_error(y_raw, y_pred) ** 0.5)
            r2 = float(r2_score(y_raw, y_pred))
            macro_mae = float(np.mean([m["mae"] for m in per_municipio.values()]))
            macro_rmse = float(np.mean([m["rmse"] for m in per_municipio.values()]))
            macro_r2 = float(np.mean([m["r2"] for m in per_municipio.values()]))
            metrics[split] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "macro_mae": macro_mae,
                "macro_rmse": macro_rmse,
                "macro_r2": macro_r2,
                "n": int(len(y_raw)),
                "per_municipio": per_municipio,
            }
            self.logger.info(
                "Split=%-12s  MAE=%.4f  RMSE=%.4f  R²=%.4f  macro_R²=%.4f",
                split,
                mae,
                rmse,
                r2,
                macro_r2,
            )

        return metrics

    def persist(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
        *,
        run_id: str,
    ) -> str:
        cfg = self.exp_config
        artifact_staging = self._staging / "artifacts"
        artifact_staging.mkdir(parents=True, exist_ok=True)

        rocket_local = artifact_staging / "minirocket_transformer.joblib"
        ridge_local = artifact_staging / "ridge_model.joblib"
        eval_local = artifact_staging / "evaluation_metrics.parquet"
        per_mun_eval_local = artifact_staging / "per_municipio_metrics.parquet"
        registry_local = artifact_staging / "feature_registry.json"

        joblib.dump(feature_data["rocket"], str(rocket_local))
        joblib.dump(training_output["models"], str(ridge_local))

        eval_rows = [
            {
                "split": split,
                "mae": split_metrics["mae"],
                "rmse": split_metrics["rmse"],
                "r2": split_metrics["r2"],
                "macro_mae": split_metrics["macro_mae"],
                "macro_rmse": split_metrics["macro_rmse"],
                "macro_r2": split_metrics["macro_r2"],
                "n": split_metrics["n"],
            }
            for split, split_metrics in evaluation_output.items()
        ]
        if eval_rows:
            pd.DataFrame(eval_rows).to_parquet(str(eval_local), index=False)

        per_mun_eval_rows = []
        for split, split_metrics in evaluation_output.items():
            for mun_id, mun_metrics in split_metrics.get("per_municipio", {}).items():
                per_mun_eval_rows.append(
                    {
                        "split": split,
                        "codigo_municipio": mun_id,
                        "mae": mun_metrics["mae"],
                        "rmse": mun_metrics["rmse"],
                        "r2": mun_metrics["r2"],
                        "n": int(mun_metrics["n"]),
                    }
                )
        if per_mun_eval_rows:
            pd.DataFrame(per_mun_eval_rows).to_parquet(
                str(per_mun_eval_local), index=False
            )

        registry_local.write_text(
            json.dumps(feature_data["registry"], indent=2, default=str),
            encoding="utf-8",
        )

        mlflow = _import_mlflow()
        mlflow.log_artifacts(str(artifact_staging))
        registered_version: str | None = None
        if cfg.register_model:
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MunicipioDayPyfuncModel(),
                artifacts={
                    "rocket": str(rocket_local),
                    "models": str(ridge_local),
                    "registry": str(registry_local),
                },
                registered_model_name=cfg.registered_model_name,
                metadata={
                    "model_variant": str(feature_data["registry"].get("model_variant")),
                    "feature_input_subpath": cfg.feature_input_subpath,
                    "run_id": run_id,
                },
            )
            client = mlflow.tracking.MlflowClient()
            versions = [
                version
                for version in client.search_model_versions(
                    f"name = '{cfg.registered_model_name}'"
                )
                if getattr(version, "run_id", None) == run_id
            ]
            if versions:
                latest = max(versions, key=lambda version: int(version.version))
                registered_version = str(latest.version)
            else:
                registered_version = str(
                    getattr(model_info, "registered_model_version", "") or ""
                )
            if registered_version:
                client.set_registered_model_alias(
                    cfg.registered_model_name,
                    cfg.champion_alias,
                    registered_version,
                )
                self.logger.info(
                    "Registered MLflow model %s version %s as alias %s",
                    cfg.registered_model_name,
                    registered_version,
                    cfg.champion_alias,
                )
            else:
                self.logger.warning(
                    "Model registration finished but no version was resolved for %s",
                    cfg.registered_model_name,
                )

        subpath = cfg.output_subpath
        for local_file in artifact_staging.iterdir():
            dest = self.datalake.persist_file(
                local_file, f"{subpath}/{local_file.name}"
            )
            self.logger.debug("Persisted artifact: %s", dest)

        self.logger.info("Experiment artifacts persisted under %s", subpath)
        return self.datalake.uri_for(subpath)

    def mlflow_params(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
    ) -> dict[str, Any]:
        reg = feature_data["registry"]
        return {
            "lookback_days": reg["lookback_days"],
            "n_kernels": self.exp_config.n_kernels,
            "random_seed": self.exp_config.random_seed,
            "ridge_alphas": str(self.exp_config.ridge_alphas),
            "ridge_alpha_chosen": json.dumps(training_output["alphas"]),
            "alpha_selection_split": training_output["alpha_selection_split"],
            "n_municipios": reg["n_municipios"],
            "n_sequence_channels": reg["n_sequence_channels"],
            "model_variant": reg.get("model_variant"),
            "include_weather_features": reg.get("include_weather_features", False),
            "n_features_total": training_output["n_rocket_features"],
            "holiday_country": reg["holiday_country"],
            "holiday_state": reg["holiday_state"],
            "holiday_max_lookahead": reg["holiday_max_lookahead"],
            "holidays_version": holidays.__version__,
            "weather_input_subpath": reg.get("weather_input_subpath"),
            "weather_feature_columns": json.dumps(
                reg.get("weather_feature_columns", [])
            ),
            "weather_complete_row_share": reg.get("weather_complete_row_share"),
            "split_config": json.dumps(reg["split_config"]),
            "n_train_samples": feature_data["split_counts"]["train"],
            "n_val_samples": feature_data["split_counts"]["validation"],
            "n_test_samples": feature_data["split_counts"]["test"],
            "n_models": len(training_output["models"]),
        }

    def mlflow_metrics(
        self,
        feature_data: dict[str, Any],
        training_output: dict[str, Any],
        evaluation_output: dict[str, Any],
    ) -> dict[str, Any]:
        flat: dict[str, float] = {}
        for split, m in evaluation_output.items():
            flat[f"mae__{split}"] = m["mae"]
            flat[f"rmse__{split}"] = m["rmse"]
            flat[f"r2__{split}"] = m["r2"]
            flat[f"macro_mae__{split}"] = m["macro_mae"]
            flat[f"macro_rmse__{split}"] = m["macro_rmse"]
            flat[f"macro_r2__{split}"] = m["macro_r2"]
            for mun_id, mun_metrics in m.get("per_municipio", {}).items():
                flat[f"mae__{split}__{mun_id}"] = mun_metrics["mae"]
                flat[f"rmse__{split}__{mun_id}"] = mun_metrics["rmse"]
                flat[f"r2__{split}__{mun_id}"] = mun_metrics["r2"]
        return flat

    def cleanup(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Forecast phase
# ---------------------------------------------------------------------------


def _coerce_accident_dates(panel: pd.DataFrame) -> pd.DataFrame:
    coerced = panel.copy()
    if pd.api.types.is_datetime64_any_dtype(coerced["accident_date"]):
        coerced["accident_date"] = coerced["accident_date"].dt.date
    elif len(coerced) and not isinstance(coerced["accident_date"].iloc[0], date):
        coerced["accident_date"] = pd.to_datetime(coerced["accident_date"]).dt.date
    coerced["codigo_municipio"] = coerced["codigo_municipio"].astype(str)
    return coerced


def _build_future_feature_panel(
    panel: pd.DataFrame,
    registry: dict[str, Any],
    horizon_days: int,
) -> tuple[pd.DataFrame, list[date], date]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    observed = _coerce_accident_dates(panel)
    source_cutoff = max(observed["accident_date"])
    future_dates = [
        source_cutoff + timedelta(days=offset) for offset in range(1, horizon_days + 1)
    ]

    static_columns = [
        "codigo_municipio",
        "nome_municipio",
        "sg_uf",
        "cd_rgint",
        "nm_rgint",
    ]
    static = (
        observed[[c for c in static_columns if c in observed.columns]]
        .drop_duplicates("codigo_municipio", keep="last")
        .set_index("codigo_municipio")
    )
    municipio_codes: list[str] = registry["municipio_codes"]
    max_look = int(registry.get("holiday_max_lookahead", DEFAULT_HOLIDAY_MAX_LOOKAHEAD))
    years = sorted(
        {
            *(d.year for d in observed["accident_date"].unique().tolist()),
            *(d.year for d in future_dates),
        }
    )
    cal = _build_sc_calendar(list(range(min(years) - 1, max(years) + 2)))

    rows: list[dict[str, Any]] = []
    include_weather = bool(registry.get("include_weather_features", False))
    for mun_id in municipio_codes:
        base = static.loc[mun_id].to_dict() if mun_id in static.index else {}
        for d in future_dates:
            row: dict[str, Any] = {
                "codigo_municipio": mun_id,
                **base,
                "accident_date": d,
                "year": d.year,
                "month": d.month,
                "day": d.day,
                "accident_count": 0.0,
                "split": "forecast",
                "non_working_day_weight": _day_weight(d, cal),
                "days_off_ahead": _days_off_forward(d, cal, max_look),
                "days_off_before": _days_off_backward(d, cal, max_look),
            }
            if include_weather:
                for column in WEATHER_FEATURE_COLUMNS:
                    row[column] = np.nan
            rows.append(row)

    future = pd.DataFrame(rows)
    combined = pd.concat([observed, future], ignore_index=True, sort=False)
    combined = combined.sort_values(["codigo_municipio", "accident_date"]).reset_index(
        drop=True
    )
    return combined, future_dates, source_cutoff


def _build_prediction_rows_for_date(
    panel: pd.DataFrame,
    registry: dict[str, Any],
    target_date: date,
) -> list[dict[str, Any]]:
    lookback_days = int(registry["lookback_days"])
    municipio_codes: list[str] = registry["municipio_codes"]
    include_weather = bool(registry.get("include_weather_features", False))

    all_dates: list[date] = sorted(panel["accident_date"].unique().tolist())
    target_pos = all_dates.index(target_date)
    if target_pos < lookback_days:
        raise ValueError(
            f"Not enough history to forecast {target_date}: "
            f"need {lookback_days} days, found {target_pos}"
        )
    sequence_index = target_pos - lookback_days
    train_years: set[int] = set(
        panel.loc[panel["split"] == "train", "year"].unique().astype(int).tolist()
    )
    cal_channels = _precompute_calendar_channel_arrays(all_dates)
    exo_channels = _precompute_exo_channel_arrays(all_dates, panel)

    rows: list[dict[str, Any]] = []
    for mun_id in municipio_codes:
        mun_panel = panel[panel["codigo_municipio"] == mun_id].set_index(
            "accident_date"
        )
        ac_series = mun_panel["accident_count"].astype(float)
        weather_channels = (
            _precompute_weather_channel_arrays(
                all_dates,
                mun_panel.reset_index(),
                train_years,
            )
            if include_weather
            else None
        )
        X_seq, _, target_idx, ac_mean, ac_std = _build_municipio_sequences(
            ac_series,
            all_dates,
            cal_channels,
            exo_channels,
            weather_channels,
            lookback_days,
            train_years,
        )
        if int(target_idx[sequence_index]) != target_pos:
            raise RuntimeError("Forecast sequence index alignment failed")
        rows.append(
            {
                "codigo_municipio": mun_id,
                "sequence": X_seq[sequence_index],
                "y_mean": ac_mean,
                "y_std": ac_std,
            }
        )
    return rows


class MunicipioDayForecaster:
    def __init__(self, config: MunicipioDayForecastConfig | None = None) -> None:
        _configure_logging()
        self.forecast_config = config or MunicipioDayForecastConfig()
        self.logger = logging.getLogger("municipio_day_forecaster")
        self.datalake = DatalakeAdapter.from_env(
            project_root=self.forecast_config.project_root
        )
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging = Path(self._tmp.name)

    def run(self) -> str:
        cfg = self.forecast_config
        panel, registry = _load_feature_data(
            self.datalake,
            cfg.feature_input_subpath,
            self._staging,
        )
        forecast_panel, future_dates, source_cutoff = _build_future_feature_panel(
            panel,
            registry,
            cfg.horizon_days,
        )

        mlflow = _import_mlflow()
        mlflow.set_tracking_uri(cfg.tracking_uri)
        model = mlflow.pyfunc.load_model(cfg.model_uri)
        _patch_numpy_compat_aliases()
        generated_at = datetime.now(UTC).isoformat()

        predictions: list[pd.DataFrame] = []
        for horizon_day, target_date in enumerate(future_dates, start=1):
            payload = pd.DataFrame(
                _build_prediction_rows_for_date(forecast_panel, registry, target_date)
            )
            predicted = model.predict(payload)
            predicted = pd.DataFrame(predicted)
            predicted["accident_date"] = target_date
            predicted["forecast_horizon_day"] = horizon_day
            predictions.append(predicted)

            value_by_mun = predicted.set_index("codigo_municipio")[
                "predicted_accident_count"
            ].to_dict()
            date_mask = forecast_panel["accident_date"] == target_date
            forecast_panel.loc[date_mask, "accident_count"] = forecast_panel.loc[
                date_mask, "codigo_municipio"
            ].map(value_by_mun)
            self.logger.info(
                "Forecasted %s (%d/%d)",
                target_date,
                horizon_day,
                len(future_dates),
            )

        forecast = pd.concat(predictions, ignore_index=True)
        static = forecast_panel[
            [
                c
                for c in [
                    "codigo_municipio",
                    "nome_municipio",
                    "sg_uf",
                    "cd_rgint",
                    "nm_rgint",
                ]
                if c in forecast_panel.columns
            ]
        ].drop_duplicates("codigo_municipio", keep="last")
        forecast = forecast.merge(static, on="codigo_municipio", how="left")
        forecast["model_variant"] = registry.get("model_variant")
        forecast["model_uri"] = cfg.model_uri
        forecast["source_panel_max_date"] = source_cutoff
        forecast["generated_at"] = generated_at
        forecast["fonte"] = "previsão"
        forecast = forecast[
            [
                "codigo_municipio",
                "nome_municipio",
                "sg_uf",
                "cd_rgint",
                "nm_rgint",
                "accident_date",
                "predicted_accident_count",
                "prediction_raw",
                "prediction_scaled",
                "forecast_horizon_day",
                "model_variant",
                "model_uri",
                "source_panel_max_date",
                "generated_at",
                "fonte",
            ]
        ].sort_values(["accident_date", "codigo_municipio"])

        output_dir = self._staging / "forecast_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        forecast.to_parquet(output_dir / "forecast.parquet", index=False)
        manifest = {
            "generated_at": generated_at,
            "model_uri": cfg.model_uri,
            "model_variant": registry.get("model_variant"),
            "source_panel_max_date": str(source_cutoff),
            "horizon_days": cfg.horizon_days,
            "forecast_min_date": str(min(future_dates)),
            "forecast_max_date": str(max(future_dates)),
            "feature_input_subpath": cfg.feature_input_subpath,
            "output_subpath": cfg.output_subpath,
            "n_rows": int(len(forecast)),
        }
        (output_dir / "forecast_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        dest = self.datalake.persist_directory(output_dir, cfg.output_subpath)
        self.logger.info("Forecast persisted under %s", dest)
        return dest

    def cleanup(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Municipio-day Mini-ROCKET + Ridge pipeline"
    )
    parser.add_argument(
        "phase",
        choices=["featurize", "train", "predict"],
        help="Pipeline phase to execute",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Workspace root (default: repo root derived from __file__)",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()

    if args.phase == "featurize":
        feat_config = MunicipioDayFeaturizationConfig.from_yaml(
            project_root / DEFAULT_FEATURIZATION_CONFIG_PATH,
            project_root=project_root,
        )
        MunicipioDayFeaturizer(config=feat_config).run()

    elif args.phase == "train":
        exp_config = MunicipioDayExperimentConfig.from_yaml(
            project_root / DEFAULT_EXPERIMENT_CONFIG_PATH,
            project_root=project_root,
        )
        MunicipioDayRegressionExperiment(config=exp_config).run()

    elif args.phase == "predict":
        forecast_config = MunicipioDayForecastConfig.from_yaml(
            project_root / DEFAULT_EXPERIMENT_CONFIG_PATH,
            project_root=project_root,
        )
        forecaster = MunicipioDayForecaster(config=forecast_config)
        try:
            forecaster.run()
        finally:
            forecaster.cleanup()


if __name__ == "__main__":
    main()
