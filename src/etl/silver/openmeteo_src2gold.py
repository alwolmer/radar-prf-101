from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from pyproj import Transformer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter, S3DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DEFAULT_GEODETIC_CRS = "epsg:4674"
DEFAULT_PROJECTED_CRS = "epsg:5880"
DEFAULT_DAILY_FEATURES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "sunrise",
    "sunset",
    "precipitation_sum",
    "precipitation_hours",
)
DEFAULT_MUNICIPIOS_SUBPATH = "gold/br101_sc_municipio_panel/municipios_in_scope"
DEFAULT_GOLD_SUBPATH = "gold/br101_sc_municipio_panel/daily_weather"
DEFAULT_FORECAST_HORIZON_DAYS = 15
DEFAULT_SOURCE_JSON_DIR = "data/bronze/openmeteo-weather"


def _configure_spark_s3(spark: SparkSession, datalake: DatalakeAdapter) -> None:
    if not isinstance(datalake, S3DatalakeAdapter):
        return

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

    import boto3

    creds = boto3.session.Session().get_credentials().get_frozen_credentials()
    hadoop_conf.set("fs.s3a.access.key", creds.access_key)
    hadoop_conf.set("fs.s3a.secret.key", creds.secret_key)
    if creds.token:
        hadoop_conf.set("fs.s3a.session.token", creds.token)
    if datalake.endpoint_url:
        hadoop_conf.set("fs.s3a.endpoint", datalake.endpoint_url)
    if datalake.region_name:
        hadoop_conf.set("fs.s3a.endpoint.region", datalake.region_name)
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.path.style.access", "true")


def _normalize_string(value: Any, *, upper: bool = False) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized.upper() if upper else normalized


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _normalize_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _normalize_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)):
        timestamp = datetime.fromtimestamp(value, tz=UTC)
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        timestamp = datetime.fromisoformat(text)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
    return timestamp


def _build_day_sequence(start_day: date, end_day: date) -> list[date]:
    days: list[date] = []
    current = start_day
    while current <= end_day:
        days.append(current)
        current += timedelta(days=1)
    return days


def _as_response_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        responses = payload.get("responses")
        if isinstance(responses, list):
            return [item for item in responses if isinstance(item, dict)]
        if "daily" in payload:
            return [payload]
    return []


def _round_coord(value: float) -> float:
    return round(float(value), 6)


def _date_range_from_responses(
    responses: Iterable[dict[str, Any]],
) -> tuple[date, date]:
    observed: list[date] = []
    for response in responses:
        daily = response.get("daily") or {}
        for raw_day in daily.get("time") or []:
            day = _normalize_date(raw_day)
            if day is not None:
                observed.append(day)

    if not observed:
        raise ValueError("Local Open-Meteo JSON does not contain any daily time values")
    return min(observed), max(observed)


def _summarize_dates(days: list[date], *, limit: int = 5) -> str:
    if not days:
        return ""
    head = ", ".join(day.isoformat() for day in days[:limit])
    suffix = f", ... {len(days) - limit} more" if len(days) > limit else ""
    return f"{head}{suffix}"


def _daily_value(daily: dict[str, Any], feature_name: str, index: int) -> Any:
    values = daily.get(feature_name)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _point_within_candidates(
    *,
    point_proj: Point,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for candidate in candidates:
        minx, miny, maxx, maxy = candidate["_bounds"]
        if not (minx <= point_proj.x <= maxx and miny <= point_proj.y <= maxy):
            continue
        geometry_proj: BaseGeometry = candidate["_geometry_proj"]
        if point_proj.within(geometry_proj) or point_proj.intersects(geometry_proj):
            return candidate
    return None


class OpenMeteoSrc2Gold(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="openmeteo_src2gold")
        self.datalake: DatalakeAdapter = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.municipios_subpath = str(
            self.config.get("municipios_subpath", DEFAULT_MUNICIPIOS_SUBPATH)
        )
        self.gold_subpath = str(self.config.get("gold_subpath", DEFAULT_GOLD_SUBPATH))
        self.api_url = str(self.config.get("api_url", DEFAULT_API_URL))
        raw_local_json_paths = self.config.get("local_json_paths")
        if raw_local_json_paths is None:
            raw_local_json_path = self.config.get("local_json_path")
            raw_local_json_paths = [raw_local_json_path] if raw_local_json_path else []
        if isinstance(raw_local_json_paths, (str, Path)):
            raw_local_json_paths = [raw_local_json_paths]
        self.local_json_paths = [str(path) for path in raw_local_json_paths if path]
        raw_local_json_dirs = self.config.get("local_json_dirs", [])
        if isinstance(raw_local_json_dirs, (str, Path)):
            raw_local_json_dirs = [raw_local_json_dirs]
        self.local_json_dirs = [str(path) for path in raw_local_json_dirs if path]
        self.start_date_from_existing_max = bool(
            self.config.get("start_date_from_existing_max", False)
        )
        self.end_date_today = bool(self.config.get("end_date_today", False))
        self.request_timeout = int(self.config.get("request_timeout", 120))
        self.coordinate_match_tolerance_degrees = float(
            self.config.get("coordinate_match_tolerance_degrees", 0.2)
        )
        self.api_coordinate_chunk_size = int(
            self.config.get("api_coordinate_chunk_size", 50)
        )
        self.geodetic_crs = str(
            self.config.get("geodetic_crs", DEFAULT_GEODETIC_CRS)
        ).lower()
        self.projected_crs = str(
            self.config.get("projected_crs", DEFAULT_PROJECTED_CRS)
        ).lower()
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError("OpenMeteoSrc2Gold requires a Spark session")

    def _extract_dataset(self, subpath: str, staging_name: str) -> DataFrame:
        uri = self.datalake.uri_for(subpath).replace("s3://", "s3a://")
        if uri.startswith("s3a://"):
            _configure_spark_s3(self.spark, self.datalake)
        else:
            uri = str(
                self.datalake.stage_directory(subpath, self._staging_dir / staging_name)
            )
        return self.spark.read.option("basePath", uri).parquet(uri)

    def _load_local_json(self) -> list[dict[str, Any]] | None:
        local_json_paths = list(self.local_json_paths)
        for raw_dir in self.local_json_dirs:
            local_dir = Path(raw_dir).expanduser()
            if not local_dir.is_absolute():
                local_dir = PROJECT_ROOT / local_dir
            if not local_dir.is_dir():
                raise FileNotFoundError(
                    f"Local Open-Meteo JSON directory not found: {local_dir}"
                )
            local_json_paths.extend(
                str(path) for path in sorted(local_dir.glob("*.json"))
            )

        if not local_json_paths:
            return None

        responses: list[dict[str, Any]] = []
        for raw_path in local_json_paths:
            local_path = Path(raw_path).expanduser()
            if not local_path.is_absolute():
                local_path = PROJECT_ROOT / local_path
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"Local Open-Meteo JSON not found: {local_path}"
                )

            with local_path.open("r", encoding="utf-8") as source_file:
                payload = json.load(source_file)

            file_responses = _as_response_list(payload)
            if not file_responses:
                if isinstance(payload, dict) and payload.get("error"):
                    raise ValueError(
                        "Local Open-Meteo JSON contains an error response: "
                        f"{local_path} ({payload.get('reason')})"
                    )
                raise ValueError(
                    f"No Open-Meteo response objects found in {local_path}"
                )
            for response in file_responses:
                response["_source_json_path"] = str(local_path)
            responses.extend(file_responses)
            self.logger.info(
                "Loaded %s Open-Meteo response object(s) from %s",
                len(file_responses),
                local_path,
            )

        if not responses:
            raise ValueError("No Open-Meteo response objects found in local JSON files")
        self.logger.info(
            "Loaded %s total Open-Meteo response object(s) from %s local file(s)",
            len(responses),
            len(local_json_paths),
        )
        return responses

    def extract(self) -> dict[str, Any]:
        return {
            "municipios": self._extract_dataset(
                self.municipios_subpath,
                "extract_municipios_in_scope",
            ),
            "local_responses": self._load_local_json(),
        }

    def _resolve_date_range(
        self, local_responses: list[dict[str, Any]] | None
    ) -> tuple[date, date]:
        start_day = _normalize_date(self.config.get("start_date"))
        end_day = _normalize_date(self.config.get("end_date"))

        if self.start_date_from_existing_max:
            if start_day is not None:
                raise ValueError(
                    "Use either --start-date or --start-date-from-existing-max, not both"
                )
            start_day = self._existing_max_weather_date()
        if self.end_date_today:
            if end_day is not None:
                raise ValueError("Use either --end-date or --end-date-today, not both")
            end_day = date.today()

        if start_day is None and end_day is None and local_responses is not None:
            return _date_range_from_responses(local_responses)

        today = date.today()
        if start_day is None:
            start_day = today
        if end_day is None:
            end_day = start_day + timedelta(days=DEFAULT_FORECAST_HORIZON_DAYS)
        if end_day < start_day:
            raise ValueError("end_date must be greater than or equal to start_date")
        return start_day, end_day

    def _existing_max_weather_date(self) -> date:
        existing = self._read_existing_gold()
        if existing is None:
            raise FileNotFoundError(
                "Cannot derive API start date because the existing weather gold table "
                f"does not exist: {self.gold_subpath}"
            )
        max_rows = existing.select(
            F.max("weather_date").alias("max_weather_date")
        ).collect()
        if not max_rows or max_rows[0].max_weather_date is None:
            raise ValueError(
                f"Cannot derive API start date because {self.gold_subpath} has no "
                "weather_date values"
            )
        max_day = _normalize_date(max_rows[0].max_weather_date)
        if max_day is None:
            raise ValueError("Existing weather gold max(weather_date) is invalid")
        self.logger.info("Using existing max weather_date as start date: %s", max_day)
        return max_day

    def _build_scope_points(self, municipios: DataFrame) -> list[dict[str, Any]]:
        geodetic_to_projected = Transformer.from_crs(
            self.geodetic_crs,
            self.projected_crs,
            always_xy=True,
        )
        projected_to_geodetic = Transformer.from_crs(
            self.projected_crs,
            self.geodetic_crs,
            always_xy=True,
        )

        rows = (
            municipios.select(
                "codigo_municipio",
                "nome_municipio",
                "sg_uf",
                "cd_rgint",
                "nm_rgint",
                "geometry",
            )
            .orderBy("codigo_municipio")
            .collect()
        )

        candidates: list[dict[str, Any]] = []
        for row in rows:
            geometry_wkt = _normalize_string(row.geometry)
            if geometry_wkt is None:
                continue
            geometry_geodetic = shapely_wkt.loads(geometry_wkt)
            if geometry_geodetic.is_empty:
                continue
            geometry_proj = shapely_transform(
                geodetic_to_projected.transform,
                geometry_geodetic,
            )
            if geometry_proj.is_empty:
                continue
            candidates.append(
                {
                    "codigo_municipio": _normalize_string(row.codigo_municipio),
                    "nome_municipio": _normalize_string(row.nome_municipio),
                    "sg_uf": _normalize_string(row.sg_uf, upper=True),
                    "cd_rgint": _normalize_string(row.cd_rgint),
                    "nm_rgint": _normalize_string(row.nm_rgint),
                    "_geometry_proj": geometry_proj,
                    "_bounds": geometry_proj.bounds,
                }
            )

        if not candidates:
            raise ValueError(
                "No valid municipio geometries found in municipios_in_scope"
            )

        points: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()
        for candidate in candidates:
            centroid_proj = candidate["_geometry_proj"].centroid
            centroid_geodetic = shapely_transform(
                projected_to_geodetic.transform,
                centroid_proj,
            )
            longitude = _round_coord(centroid_geodetic.x)
            latitude = _round_coord(centroid_geodetic.y)
            coordinate_key = (latitude, longitude)

            point_match = _point_within_candidates(
                point_proj=centroid_proj,
                candidates=candidates,
            )
            match_status = "matched_centroid_superposition"
            if point_match is None:
                point_match = candidate
                match_status = "fallback_source_polygon_centroid"
                self.logger.warning(
                    "Centroid %.6f, %.6f for municipio %s did not fall within any "
                    "municipio polygon; using source municipio code",
                    latitude,
                    longitude,
                    candidate["codigo_municipio"],
                )

            if coordinate_key in seen:
                self.logger.warning(
                    "Duplicate requested centroid coordinate %.6f, %.6f encountered",
                    latitude,
                    longitude,
                )
            seen.add(coordinate_key)

            points.append(
                {
                    "codigo_municipio": point_match["codigo_municipio"],
                    "nome_municipio": point_match["nome_municipio"],
                    "sg_uf": point_match["sg_uf"],
                    "cd_rgint": point_match["cd_rgint"],
                    "nm_rgint": point_match["nm_rgint"],
                    "latitude": latitude,
                    "longitude": longitude,
                    "coordinate_key": coordinate_key,
                    "municipio_match_status": match_status,
                }
            )

        self.logger.info("Built %s requested municipio centroid point(s)", len(points))
        return points

    def _fetch_api_responses(
        self,
        *,
        points: list[dict[str, Any]],
        start_day: date,
        end_day: date,
    ) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for index in range(0, len(points), self.api_coordinate_chunk_size):
            chunk = points[index : index + self.api_coordinate_chunk_size]
            params = {
                "latitude": ",".join(str(point["latitude"]) for point in chunk),
                "longitude": ",".join(str(point["longitude"]) for point in chunk),
                "start_date": start_day.isoformat(),
                "end_date": end_day.isoformat(),
                "daily": ",".join(DEFAULT_DAILY_FEATURES),
                "timezone": "GMT",
            }
            self.logger.info(
                "Requesting Open-Meteo daily forecast for %s point(s), %s to %s",
                len(chunk),
                start_day,
                end_day,
            )
            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            batch = _as_response_list(response.json())
            if not batch:
                raise ValueError("Open-Meteo API response did not contain daily data")
            responses.extend(batch)

        self.logger.info("Fetched %s Open-Meteo response object(s)", len(responses))
        return responses

    def _nearest_response(
        self,
        point: dict[str, Any],
        responses: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float | None]:
        best_response: dict[str, Any] | None = None
        best_distance: float | None = None
        for response in responses:
            response_lat = _normalize_float(response.get("latitude"))
            response_lon = _normalize_float(response.get("longitude"))
            if response_lat is None or response_lon is None:
                continue
            distance = math.hypot(
                float(point["latitude"]) - response_lat,
                float(point["longitude"]) - response_lon,
            )
            if best_distance is None or distance < best_distance:
                best_response = response
                best_distance = distance

        if (
            best_distance is not None
            and best_distance <= self.coordinate_match_tolerance_degrees
        ):
            return best_response, best_distance
        return None, best_distance

    def _matched_responses(
        self,
        point: dict[str, Any],
        responses: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], float]], float | None]:
        matched: list[tuple[dict[str, Any], float]] = []
        nearest_distance: float | None = None
        for response in responses:
            response_lat = _normalize_float(response.get("latitude"))
            response_lon = _normalize_float(response.get("longitude"))
            if response_lat is None or response_lon is None:
                continue
            distance = math.hypot(
                float(point["latitude"]) - response_lat,
                float(point["longitude"]) - response_lon,
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
            if distance <= self.coordinate_match_tolerance_degrees:
                matched.append((response, distance))

        matched.sort(key=lambda item: item[1])
        return matched, nearest_distance

    def _build_records(
        self,
        *,
        points: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        start_day: date,
        end_day: date,
        source_kind: str,
    ) -> list[dict[str, Any]]:
        requested_days = _build_day_sequence(start_day, end_day)
        requested_day_set = set(requested_days)
        fetched_at = datetime.now(UTC).replace(tzinfo=None)
        records: list[dict[str, Any]] = []

        for point in points:
            matched_responses, nearest_distance = self._matched_responses(
                point, responses
            )
            if not matched_responses:
                self.logger.warning(
                    "No %s Open-Meteo response matched municipio %s centroid "
                    "%.6f, %.6f within %.4f degree(s); nearest distance=%s",
                    source_kind,
                    point["codigo_municipio"],
                    point["latitude"],
                    point["longitude"],
                    self.coordinate_match_tolerance_degrees,
                    (
                        f"{nearest_distance:.6f}"
                        if nearest_distance is not None
                        else "n/a"
                    ),
                )
                continue

            day_lookup: dict[date, tuple[dict[str, Any], int, float]] = {}
            for response, distance in matched_responses:
                daily = response.get("daily") or {}
                raw_days = daily.get("time") or []
                for index, raw_day in enumerate(raw_days):
                    parsed_day = _normalize_date(raw_day)
                    if parsed_day is None or parsed_day not in requested_day_set:
                        continue
                    existing = day_lookup.get(parsed_day)
                    if existing is None or distance < existing[2]:
                        day_lookup[parsed_day] = (response, index, distance)

            missing_days = sorted(requested_day_set.difference(day_lookup))
            if source_kind == "local_json" and missing_days:
                self.logger.warning(
                    "Local Open-Meteo JSON is missing %s requested day(s) for "
                    "municipio %s: %s",
                    len(missing_days),
                    point["codigo_municipio"],
                    _summarize_dates(missing_days),
                )

            for weather_date in requested_days:
                lookup = day_lookup.get(weather_date)
                if lookup is None:
                    continue
                response, daily_index, distance = lookup
                daily = response.get("daily") or {}
                response_lat = _normalize_float(response.get("latitude"))
                response_lon = _normalize_float(response.get("longitude"))

                records.append(
                    {
                        "codigo_municipio": point["codigo_municipio"],
                        "nome_municipio": point["nome_municipio"],
                        "sg_uf": point["sg_uf"],
                        "cd_rgint": point["cd_rgint"],
                        "nm_rgint": point["nm_rgint"],
                        "weather_date": weather_date,
                        "year": weather_date.year,
                        "month": weather_date.month,
                        "day": weather_date.day,
                        "latitude": float(point["latitude"]),
                        "longitude": float(point["longitude"]),
                        "openmeteo_latitude": response_lat,
                        "openmeteo_longitude": response_lon,
                        "openmeteo_elevation_m": _normalize_float(
                            response.get("elevation")
                        ),
                        "temperature_2m_max": _normalize_float(
                            _daily_value(daily, "temperature_2m_max", daily_index)
                        ),
                        "temperature_2m_min": _normalize_float(
                            _daily_value(daily, "temperature_2m_min", daily_index)
                        ),
                        "wind_speed_10m_max": _normalize_float(
                            _daily_value(daily, "wind_speed_10m_max", daily_index)
                        ),
                        "sunrise": _normalize_timestamp(
                            _daily_value(daily, "sunrise", daily_index)
                        ),
                        "sunset": _normalize_timestamp(
                            _daily_value(daily, "sunset", daily_index)
                        ),
                        "precipitation_sum": _normalize_float(
                            _daily_value(daily, "precipitation_sum", daily_index)
                        ),
                        "precipitation_hours": _normalize_float(
                            _daily_value(daily, "precipitation_hours", daily_index)
                        ),
                        "source_kind": source_kind,
                        "source_url": self.api_url if source_kind == "api" else None,
                        "source_json_path": response.get("_source_json_path"),
                        "source_fetched_at": fetched_at,
                        "coordinate_match_distance_degrees": distance,
                        "municipio_match_status": point["municipio_match_status"],
                    }
                )

        self.logger.info(
            "Built %s Open-Meteo municipio-day record(s) for %s to %s",
            len(records),
            start_day,
            end_day,
        )
        return records

    def _schema(self) -> T.StructType:
        return T.StructType(
            [
                T.StructField("codigo_municipio", T.StringType(), False),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("weather_date", T.DateType(), False),
                T.StructField("year", T.IntegerType(), False),
                T.StructField("month", T.IntegerType(), False),
                T.StructField("day", T.IntegerType(), False),
                T.StructField("latitude", T.DoubleType(), True),
                T.StructField("longitude", T.DoubleType(), True),
                T.StructField("openmeteo_latitude", T.DoubleType(), True),
                T.StructField("openmeteo_longitude", T.DoubleType(), True),
                T.StructField("openmeteo_elevation_m", T.DoubleType(), True),
                T.StructField("temperature_2m_max", T.DoubleType(), True),
                T.StructField("temperature_2m_min", T.DoubleType(), True),
                T.StructField("wind_speed_10m_max", T.DoubleType(), True),
                T.StructField("sunrise", T.TimestampType(), True),
                T.StructField("sunset", T.TimestampType(), True),
                T.StructField("precipitation_sum", T.DoubleType(), True),
                T.StructField("precipitation_hours", T.DoubleType(), True),
                T.StructField("source_kind", T.StringType(), True),
                T.StructField("source_url", T.StringType(), True),
                T.StructField("source_json_path", T.StringType(), True),
                T.StructField("source_fetched_at", T.TimestampType(), False),
                T.StructField(
                    "coordinate_match_distance_degrees", T.DoubleType(), True
                ),
                T.StructField("municipio_match_status", T.StringType(), True),
            ]
        )

    def transform(self, data: dict[str, Any]) -> DataFrame:
        local_responses = data["local_responses"]
        start_day, end_day = self._resolve_date_range(local_responses)
        points = self._build_scope_points(data["municipios"])
        if local_responses is None:
            responses = self._fetch_api_responses(
                points=points,
                start_day=start_day,
                end_day=end_day,
            )
            source_kind = "api"
        else:
            responses = local_responses
            source_kind = "local_json"

        records = self._build_records(
            points=points,
            responses=responses,
            start_day=start_day,
            end_day=end_day,
            source_kind=source_kind,
        )
        return self.spark.createDataFrame(records, schema=self._schema())

    def _read_existing_gold(self) -> DataFrame | None:
        uri = self.datalake.uri_for(self.gold_subpath).replace("s3://", "s3a://")
        try:
            if uri.startswith("s3a://"):
                _configure_spark_s3(self.spark, self.datalake)
                return self.spark.read.option("basePath", uri).parquet(uri)

            local_path = self.datalake.stage_directory(
                self.gold_subpath,
                self._staging_dir / "existing_gold",
            )
            return self.spark.read.option("basePath", str(local_path)).parquet(
                str(local_path)
            )
        except FileNotFoundError:
            return None

    def _upsert_existing(self, new_data: DataFrame) -> DataFrame:
        existing = self._read_existing_gold()
        if existing is None:
            return new_data

        aligned_existing = existing.select(
            *[
                F.col(field.name).cast(field.dataType).alias(field.name)
                for field in self._schema().fields
                if field.name in existing.columns
            ]
        )
        combined = aligned_existing.unionByName(new_data, allowMissingColumns=True)
        window = Window.partitionBy("codigo_municipio", "weather_date").orderBy(
            F.col("source_fetched_at").desc_nulls_last()
        )
        upserted = (
            combined.withColumn("_row_number", F.row_number().over(window))
            .filter(F.col("_row_number") == F.lit(1))
            .drop("_row_number")
        )
        self.logger.info(
            "Upserted Open-Meteo gold table: existing=%s, new=%s, final=%s",
            existing.count(),
            new_data.count(),
            upserted.count(),
        )
        return upserted

    def load(self, data: DataFrame) -> str:
        staging_output_dir = self._staging_dir / "gold_openmeteo_weather_municipio_day"
        upserted = self._upsert_existing(data)
        upserted.write.mode("overwrite").partitionBy("year").parquet(
            str(staging_output_dir)
        )
        destination = self.datalake.persist_directory(
            staging_output_dir,
            self.gold_subpath,
        )
        self.logger.info(
            "Saved Open-Meteo municipio-day gold feature store to %s",
            destination,
        )
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/upsert Open-Meteo daily weather features by municipio."
    )
    parser.add_argument("--start-date", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--source-json",
        action="append",
        help="Local Open-Meteo JSON response to process instead of calling the API.",
    )
    parser.add_argument(
        "--source-json-dir",
        action="append",
        help=(
            "Directory containing local Open-Meteo JSON responses. All *.json files "
            "are processed in sorted order. Can be repeated."
        ),
    )
    parser.add_argument(
        "--start-date-from-existing-max",
        action="store_true",
        help="Use max(weather_date) from the destination gold table as start date.",
    )
    parser.add_argument(
        "--end-date-today",
        action="store_true",
        help="Use the current local date as the inclusive end date.",
    )
    parser.add_argument(
        "--municipios-subpath",
        default=DEFAULT_MUNICIPIOS_SUBPATH,
        help="Datalake table containing municipio polygons in scope.",
    )
    parser.add_argument(
        "--gold-subpath",
        default=DEFAULT_GOLD_SUBPATH,
        help="Destination datalake table for weather features.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Open-Meteo API URL used when --source-json is not provided.",
    )
    parser.add_argument(
        "--coordinate-match-tolerance-degrees",
        type=float,
        default=0.2,
        help="Maximum nearest-response distance for matching local/API grid points.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Open-Meteo request timeout in seconds.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "municipios_subpath": args.municipios_subpath,
        "gold_subpath": args.gold_subpath,
        "api_url": args.api_url,
        "coordinate_match_tolerance_degrees": args.coordinate_match_tolerance_degrees,
        "request_timeout": args.request_timeout,
    }
    if args.start_date:
        config["start_date"] = args.start_date
    if args.end_date:
        config["end_date"] = args.end_date
    if args.source_json:
        config["local_json_paths"] = args.source_json
    if args.source_json_dir:
        config["local_json_dirs"] = args.source_json_dir
    if args.start_date_from_existing_max:
        config["start_date_from_existing_max"] = True
    if args.end_date_today:
        config["end_date_today"] = True
    return config


if __name__ == "__main__":
    parsed_args = _parse_args()
    spark = build_spark_session("Open-Meteo Src to Gold")
    job = OpenMeteoSrc2Gold(spark=spark, config=_config_from_args(parsed_args))
    job.run()
    spark.stop()
