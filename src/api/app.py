from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.ml.municipio_day_regression import (
    DEFAULT_CHAMPION_ALIAS,
    DEFAULT_FEATURE_INPUT_SUBPATH,
    DEFAULT_FORECAST_OUTPUT_SUBPATH,
    DEFAULT_REGISTERED_MODEL_NAME,
    PROJECT_ROOT,
    _import_mlflow,
)

app = FastAPI(title="Radar PRF BR-101 Prediction API")

_FORECAST_LOCK = threading.Lock()
_FORECAST_STATE: dict[str, Any] = {
    "status": "idle",
    "detail": None,
}


def _datalake_root() -> Path:
    return Path(os.environ.get("DATALAKE_LOCAL_ROOT", PROJECT_ROOT / "data"))


def _forecast_dir() -> Path:
    subpath = os.environ.get(
        "ML_MUNICIPIO_DAY_FORECAST_OUTPUT_SUBPATH",
        DEFAULT_FORECAST_OUTPUT_SUBPATH,
    )
    return _datalake_root() / subpath


def _feature_file() -> Path:
    subpath = os.environ.get(
        "ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH",
        DEFAULT_FEATURE_INPUT_SUBPATH,
    )
    return _datalake_root() / subpath / "panel_with_features.parquet"


def _model_uri() -> str:
    return os.environ.get(
        "ML_MUNICIPIO_DAY_MODEL_URI",
        "models:/"
        f"{os.environ.get('ML_MUNICIPIO_DAY_REGISTERED_MODEL_NAME', DEFAULT_REGISTERED_MODEL_NAME)}"
        f"@{os.environ.get('ML_MUNICIPIO_DAY_CHAMPION_ALIAS', DEFAULT_CHAMPION_ALIAS)}",
    )


def _forecast_manifest_path() -> Path:
    return _forecast_dir() / "forecast_manifest.json"


def _forecast_parquet_path() -> Path:
    return _forecast_dir() / "forecast.parquet"


def _forecast_horizon_days() -> int:
    return int(os.environ.get("ML_MUNICIPIO_DAY_FORECAST_HORIZON_DAYS", "30"))


def _read_forecast_manifest() -> dict[str, Any] | None:
    path = _forecast_manifest_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _feature_cutoff_date() -> str | None:
    path = _feature_file()
    if not path.exists():
        return None
    panel = pd.read_parquet(path, columns=["accident_date"])
    if panel.empty:
        return None
    cutoff = pd.to_datetime(panel["accident_date"]).dt.date.max()
    return str(cutoff)


def _forecast_is_current() -> bool:
    if not _forecast_parquet_path().exists():
        return False
    manifest = _read_forecast_manifest()
    if manifest is None:
        return False

    feature_cutoff = _feature_cutoff_date()
    if (
        feature_cutoff is not None
        and str(manifest.get("source_panel_max_date")) != feature_cutoff
    ):
        return False
    return int(manifest.get("horizon_days", 0)) == _forecast_horizon_days()


def _run_phase(phase: str) -> None:
    env = os.environ.copy()
    env.setdefault("DATALAKE_BACKEND", "local")
    env.setdefault("DATALAKE_LOCAL_ROOT", str(_datalake_root()))
    env.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    cmd = [
        sys.executable,
        "-m",
        "src.ml.municipio_day_regression",
        phase,
        "--project-root",
        str(PROJECT_ROOT),
    ]
    print(f"Starting pipeline phase {phase!r}: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline phase {phase!r} failed with exit code {exc.returncode}",
        ) from exc
    print(f"Finished pipeline phase {phase!r}", flush=True)


def _champion_available() -> bool:
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    try:
        mlflow.pyfunc.load_model(_model_uri())
    except Exception:
        return False
    return True


def _forecast_available_response(status: str = "available") -> dict[str, Any]:
    return {
        "status": status,
        "forecast_path": str(_forecast_parquet_path()),
        "manifest_path": str(_forecast_manifest_path()),
    }


def _ensure_forecast(force: bool = False) -> dict[str, Any]:
    if not force and _forecast_is_current():
        _FORECAST_STATE.update(_forecast_available_response())
        return _forecast_available_response()

    if not _FORECAST_LOCK.acquire(blocking=False):
        return {
            **_forecast_available_response("running"),
            "detail": "Forecast generation is already running.",
        }

    try:
        _FORECAST_STATE.update(
            {
                **_forecast_available_response("running"),
                "detail": "Forecast generation started.",
            }
        )
        if not _feature_file().exists():
            _run_phase("featurize")
        if not _champion_available():
            _run_phase("train")
        _run_phase("predict")
    except Exception as exc:
        _FORECAST_STATE.update(
            {
                **_forecast_available_response("failed"),
                "detail": str(exc),
            }
        )
        raise
    finally:
        _FORECAST_LOCK.release()

    _FORECAST_STATE.update(_forecast_available_response("generated"))
    return _forecast_state_safe()


def _forecast_state_safe() -> dict[str, Any]:
    return {
        **_FORECAST_STATE,
        "forecast_path": str(_forecast_parquet_path()),
        "manifest_path": str(_forecast_manifest_path()),
    }


def _start_forecast_background(force: bool = False) -> dict[str, Any]:
    if not force and _forecast_is_current():
        return _forecast_available_response()
    if _FORECAST_LOCK.locked():
        return {
            **_forecast_state_safe(),
            "status": "running",
            "detail": "Forecast generation is already running.",
        }

    def _target() -> None:
        try:
            _ensure_forecast(force=force)
        except Exception:
            return

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    _FORECAST_STATE.update(
        {
            **_forecast_available_response("running"),
            "detail": "Forecast generation started in background.",
        }
    )
    return _forecast_state_safe()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mlflow_tracking_uri": os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        ),
        "model_uri": _model_uri(),
        "forecast_exists": _forecast_parquet_path().exists(),
        "forecast_current": _forecast_is_current(),
        "forecast_horizon_days": _forecast_horizon_days(),
        "forecast_status": _FORECAST_STATE.get("status"),
    }


@app.post("/forecast/ensure")
def forecast_ensure(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    if not bool(payload.get("wait", True)):
        return _start_forecast_background(force=bool(payload.get("force", False)))
    return _ensure_forecast(force=bool(payload.get("force", False)))


@app.post("/predict")
def predict(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    ensure_result = _ensure_forecast(force=bool(payload.get("force", False)))
    if not _forecast_parquet_path().exists():
        raise HTTPException(status_code=404, detail="Forecast table was not generated")

    forecast = pd.read_parquet(_forecast_parquet_path())
    if "codigo_municipio" in payload:
        forecast = forecast[
            forecast["codigo_municipio"].astype(str) == str(payload["codigo_municipio"])
        ]
    if "start_date" in payload:
        start_date = pd.to_datetime(payload["start_date"]).date()
        forecast = forecast[
            pd.to_datetime(forecast["accident_date"]).dt.date >= start_date
        ]
    if "end_date" in payload:
        end_date = pd.to_datetime(payload["end_date"]).date()
        forecast = forecast[
            pd.to_datetime(forecast["accident_date"]).dt.date <= end_date
        ]
    if "limit" in payload:
        forecast = forecast.head(int(payload["limit"]))

    return {
        **ensure_result,
        "model_uri": _model_uri(),
        "predictions": forecast.to_dict(orient="records"),
    }
