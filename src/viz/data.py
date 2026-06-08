from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from urllib import request

import pandas as pd
import streamlit as st
from shapely import wkt
from shapely.geometry import mapping

Granularity = Literal["dia", "semana"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_PATH = PROJECT_ROOT / "data" / "gold" / "br101_sc_municipio_panel"

PANEL_TABLES = {
    "dia": {
        "table": "canonical_accidents_by_municipio_day",
        "date_column": "accident_date",
        "end_column": None,
    },
    "semana": {
        "table": "canonical_accidents_by_municipio_week",
        "date_column": "week_start",
        "end_column": "week_end",
    },
}

OPTIONAL_FORECAST_PATHS = (
    PROJECT_ROOT / "data" / "gold" / "ml" / "municipio_day_forecast",
    PROJECT_ROOT / "data" / "gold" / "ml" / "municipio_day_predictions",
)
FORECAST_HORIZON_DAYS = 30


def panel_root() -> Path:
    return Path(os.environ.get("VIZ_PANEL_PATH", DEFAULT_PANEL_PATH)).resolve()


def forecast_path() -> Path | None:
    explicit = os.environ.get("VIZ_FORECAST_PATH")
    if explicit:
        candidate = Path(explicit).resolve()
        return candidate if candidate.exists() else None
    for candidate in OPTIONAL_FORECAST_PATHS:
        if candidate.exists():
            return candidate
    return None


def _read_forecast_manifest() -> dict | None:
    path = forecast_path()
    if path is None or not path.is_dir():
        return None
    manifest_path = path / "forecast_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def forecast_is_current(historical_max_date, historical_period_start_date=None) -> bool:
    path = forecast_path()
    if path is None:
        return False
    manifest = _read_forecast_manifest()
    if manifest is None:
        return False

    source_cutoff = manifest.get("source_panel_max_date")
    if source_cutoff is None:
        return False
    source_cutoff_date = pd.to_datetime(source_cutoff).date()
    if historical_period_start_date is not None:
        if (
            not historical_period_start_date
            <= source_cutoff_date
            <= historical_max_date
        ):
            return False
    elif source_cutoff_date != historical_max_date:
        return False
    return int(manifest.get("horizon_days", 0)) == FORECAST_HORIZON_DAYS


def ensure_forecast_available(historical_max_date) -> None:
    api_url = os.environ.get("VIZ_API_URL")
    if not api_url or forecast_is_current(historical_max_date):
        return
    state_key = f"municipio_day_forecast_ensure_requested_{historical_max_date}"
    if st.session_state.get(state_key):
        return
    st.session_state[state_key] = True

    endpoint = f"{api_url.rstrip('/')}/forecast/ensure"
    payload = json.dumps({"force": False, "wait": False}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        request.urlopen(req, timeout=5).read()
    except Exception:
        return


@st.cache_data(show_spinner=False)
def load_forecast_manifest() -> dict | None:
    return _read_forecast_manifest()


@st.cache_data(show_spinner="Carregando municípios em escopo...")
def load_municipios(panel_path: str) -> pd.DataFrame:
    municipios = pd.read_parquet(Path(panel_path) / "municipios_in_scope")
    required = {"codigo_municipio", "nome_municipio", "geometry"}
    missing = required.difference(municipios.columns)
    if missing:
        raise ValueError(f"municipios_in_scope sem colunas obrigatórias: {missing}")

    municipios = municipios.copy()
    municipios["codigo_municipio"] = municipios["codigo_municipio"].astype(str)
    return municipios


@st.cache_data(show_spinner="Preparando geometrias...")
def build_municipio_features(panel_path: str) -> list[dict]:
    municipios = load_municipios(panel_path)
    features: list[dict] = []

    for row in municipios.itertuples(index=False):
        geom = wkt.loads(row.geometry)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "codigo_municipio": row.codigo_municipio,
                    "nome_municipio": row.nome_municipio,
                    "nm_rgint": getattr(row, "nm_rgint", None),
                    "polygon_area_km2": getattr(row, "polygon_area_km2", None),
                },
            }
        )
    return features


@st.cache_data(show_spinner="Calculando enquadramento do mapa...")
def load_bounds(panel_path: str) -> tuple[list[float], list[list[float]]]:
    features = build_municipio_features(panel_path)
    lon_values: list[float] = []
    lat_values: list[float] = []

    def collect(coords: object) -> None:
        if not isinstance(coords, (list, tuple)):
            return
        if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
            lon_values.append(float(coords[0]))
            lat_values.append(float(coords[1]))
            return
        for item in coords:
            collect(item)

    for feature in features:
        collect(feature["geometry"]["coordinates"])

    if not lon_values or not lat_values:
        return [-27.3, -48.8], [[-28.7, -49.8], [-25.8, -48.0]]

    min_lon, max_lon = min(lon_values), max(lon_values)
    min_lat, max_lat = min(lat_values), max(lat_values)
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    return center, bounds


@st.cache_data(show_spinner="Carregando painel histórico de acidentes...")
def load_historical_panel(panel_path: str, granularity: Granularity) -> pd.DataFrame:
    config = PANEL_TABLES[granularity]
    table_path = Path(panel_path) / config["table"]
    if not table_path.exists():
        raise FileNotFoundError(f"Tabela gold não encontrada: {table_path}")

    panel = pd.read_parquet(table_path).copy()
    panel["codigo_municipio"] = panel["codigo_municipio"].astype(str)
    panel[config["date_column"]] = pd.to_datetime(panel[config["date_column"]]).dt.date
    end_column = config["end_column"]
    if end_column and end_column in panel.columns:
        panel[end_column] = pd.to_datetime(panel[end_column]).dt.date

    panel["fonte"] = "histórico"
    return panel


def load_panel(panel_path: str, granularity: Granularity) -> pd.DataFrame:
    config = PANEL_TABLES[granularity]
    panel = load_historical_panel(panel_path, granularity).copy()
    cutoff_column = config["end_column"] or config["date_column"]
    historical_max_date = panel[cutoff_column].max()
    historical_period_start_date = (
        panel[config["date_column"]].max() if config["end_column"] else None
    )
    forecast_current = forecast_is_current(
        historical_max_date,
        historical_period_start_date,
    )
    ensure_forecast_available(historical_max_date)
    forecast = load_forecast(granularity) if forecast_current else None
    if forecast is not None and not forecast.empty:
        panel = merge_forecast(panel, forecast, granularity)

    return panel


def load_forecast(granularity: Granularity) -> pd.DataFrame | None:
    path = forecast_path()
    if path is None:
        return None

    if path.is_dir():
        forecast_file = path / "forecast.parquet"
        forecast = (
            pd.read_parquet(forecast_file)
            if forecast_file.exists()
            else pd.read_parquet(path)
        )
    elif path.suffix.lower() in {".parquet", ".pq"}:
        forecast = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        forecast = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        forecast = pd.read_json(path)
    else:
        return None

    return normalize_forecast(forecast, granularity)


def normalize_forecast(
    forecast: pd.DataFrame, granularity: Granularity
) -> pd.DataFrame:
    if forecast.empty:
        return forecast

    prepared = forecast.copy()
    date_column = PANEL_TABLES[granularity]["date_column"]
    candidate_date_columns = [
        date_column,
        "accident_date",
        "date",
        "data",
        "prediction_date",
        "ds",
    ]
    candidate_value_columns = [
        "accident_count",
        "predicted_accident_count",
        "prediction",
        "yhat",
        "forecast",
    ]

    source_date = next(
        (c for c in candidate_date_columns if c in prepared.columns), None
    )
    source_value = next(
        (c for c in candidate_value_columns if c in prepared.columns), None
    )
    if (
        source_date is None
        or source_value is None
        or "codigo_municipio" not in prepared
    ):
        return pd.DataFrame()

    prepared["codigo_municipio"] = prepared["codigo_municipio"].astype(str)
    prepared["accident_count"] = pd.to_numeric(prepared[source_value], errors="coerce")
    prepared["fonte"] = "previsão"

    source_dates = pd.to_datetime(prepared[source_date], errors="coerce")
    if granularity == "semana":
        prepared[date_column] = (
            source_dates - pd.to_timedelta(source_dates.dt.weekday, unit="D")
        ).dt.date
        prepared["week_end"] = (
            pd.to_datetime(prepared[date_column]) + pd.Timedelta(days=6)
        ).dt.date

        group_columns = ["codigo_municipio", date_column, "week_end", "fonte"]
        for optional in ("nome_municipio",):
            if optional in prepared.columns:
                group_columns.append(optional)
        return (
            prepared.dropna(subset=[date_column, "accident_count"])
            .groupby(group_columns, as_index=False, dropna=False)["accident_count"]
            .sum()
        )

    prepared[date_column] = source_dates.dt.date

    keep_columns = ["codigo_municipio", date_column, "accident_count", "fonte"]
    for optional in ("nome_municipio", "week_end"):
        if optional in prepared.columns:
            keep_columns.append(optional)
    return prepared[keep_columns].dropna(subset=[date_column, "accident_count"])


def merge_forecast(
    panel: pd.DataFrame,
    forecast: pd.DataFrame,
    granularity: Granularity,
) -> pd.DataFrame:
    date_column = PANEL_TABLES[granularity]["date_column"]
    last_historical_date = panel[date_column].max()
    future_forecast = forecast[forecast[date_column] > last_historical_date].copy()
    if future_forecast.empty:
        return panel

    municipios = panel[
        ["codigo_municipio", "nome_municipio", "sg_uf", "cd_rgint", "nm_rgint"]
    ].drop_duplicates("codigo_municipio")
    future_forecast = future_forecast.merge(
        municipios,
        on="codigo_municipio",
        how="left",
        suffixes=("", "_hist"),
    )
    if "nome_municipio_hist" in future_forecast:
        future_forecast["nome_municipio"] = future_forecast["nome_municipio"].fillna(
            future_forecast["nome_municipio_hist"]
        )

    for column in panel.columns:
        if column not in future_forecast.columns:
            future_forecast[column] = None
    future_forecast = future_forecast[panel.columns]
    return pd.concat([panel, future_forecast], ignore_index=True)


def period_options(panel: pd.DataFrame, granularity: Granularity) -> list:
    date_column = PANEL_TABLES[granularity]["date_column"]
    return sorted(panel[date_column].dropna().unique().tolist())


def period_source_label(rows: pd.DataFrame) -> str:
    sources = set(rows.get("fonte", pd.Series(dtype=str)).dropna().astype(str))
    if not sources:
        return "-"
    if sources == {"histórico"}:
        return "Histórico"
    if sources == {"previsão"}:
        return "Previsão"
    return "Misto"


def period_source_options(
    panel: pd.DataFrame,
    granularity: Granularity,
    periods: list,
) -> dict:
    return {
        period: period_source_label(rows_for_period(panel, granularity, period))
        for period in periods
    }


def rows_for_period(
    panel: pd.DataFrame,
    granularity: Granularity,
    period,
) -> pd.DataFrame:
    date_column = PANEL_TABLES[granularity]["date_column"]
    return panel[panel[date_column] == period].copy()


def as_geojson(features: list[dict], values: pd.DataFrame, metric: str) -> dict:
    indexed_values = values.set_index("codigo_municipio").to_dict(orient="index")
    rendered_features: list[dict] = []
    for feature in features:
        codigo = feature["properties"]["codigo_municipio"]
        row = indexed_values.get(codigo, {})
        properties = {**feature["properties"], **row}
        properties[metric] = float(properties.get(metric) or 0.0)
        rendered_features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": json.loads(json.dumps(properties, default=str)),
            }
        )
    return {"type": "FeatureCollection", "features": rendered_features}
