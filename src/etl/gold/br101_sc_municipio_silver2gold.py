from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from pyproj import Transformer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter, S3DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_GEODETIC_CRS = "epsg:4674"
DEFAULT_PROJECTED_CRS = "epsg:5880"
DEFAULT_CANONICAL_POLICY = "strict_declared_and_inside"
SC_UF = "SC"


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
    return float(value)


def _normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def _normalize_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _quarter_for_month(month: int) -> int:
    return ((month - 1) // 3) + 1


def _iter_line_geometries(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.is_empty:
        return
    geometry_type = geometry.geom_type
    if geometry_type == "LineString":
        yield geometry
        return
    if geometry_type == "MultiLineString":
        for part in geometry.geoms:
            if not part.is_empty and part.length > 0:
                yield part
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_line_geometries(part)


def _assign_accident_class(
    *,
    has_valid_coords: bool,
    is_br101_declared: bool,
    inside_canonical_corridor: bool,
) -> str:
    if not has_valid_coords:
        return "missing_or_invalid_coords"
    if is_br101_declared and inside_canonical_corridor:
        return "declared_and_inside"
    if is_br101_declared and not inside_canonical_corridor:
        return "declared_outside"
    if (not is_br101_declared) and inside_canonical_corridor:
        return "undeclared_inside"
    return "outside_all"


def _build_day_sequence(start_day: date, end_day: date) -> list[date]:
    days: list[date] = []
    current = start_day
    while current <= end_day:
        days.append(current)
        current += timedelta(days=1)
    return days


def _build_week_sequence(start_week: date, end_week: date) -> list[date]:
    weeks: list[date] = []
    current = start_week
    while current <= end_week:
        weeks.append(current)
        current += timedelta(days=7)
    return weeks


class Br101ScMunicipioSilver2Gold(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            spark=spark,
            config=config,
            job_name="br101_sc_municipio_silver2gold",
        )
        self.datalake: DatalakeAdapter = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.prf_silver_subpath = str(
            self.config.get(
                "prf_silver_subpath",
                "silver/prf_accidents_standardized",
            )
        )
        self.dnit_centerline_union_subpath = str(
            self.config.get(
                "dnit_centerline_union_subpath",
                "silver/dnit_br101_corridor/br101_centerline_union",
            )
        )
        self.dnit_corridor_union_subpath = str(
            self.config.get(
                "dnit_corridor_union_subpath",
                "silver/dnit_br101_corridor/br101_corridor_union",
            )
        )
        self.municipalities_silver_subpath = str(
            self.config.get(
                "municipalities_silver_subpath",
                "silver/ibge_territorial_preprocessed/municipalities_preprocessed",
            )
        )
        self.gold_subpath = str(
            self.config.get(
                "gold_subpath",
                "gold/br101_sc_municipio_panel",
            )
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.geodetic_crs = str(
            self.config.get("geodetic_crs", DEFAULT_GEODETIC_CRS)
        ).lower()
        self.projected_crs = str(
            self.config.get("projected_crs", DEFAULT_PROJECTED_CRS)
        ).lower()
        self.canonical_policy = str(
            self.config.get("canonical_policy", DEFAULT_CANONICAL_POLICY)
        )
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError("Br101ScMunicipioSilver2Gold requires a Spark session")

    def _extract_dataset(self, subpath: str, staging_name: str) -> DataFrame:
        uri = self.datalake.uri_for(subpath).replace("s3://", "s3a://")
        if uri.startswith("s3a://"):
            _configure_spark_s3(self.spark, self.datalake)
        else:
            uri = str(
                self.datalake.stage_directory(subpath, self._staging_dir / staging_name)
            )
        return self.spark.read.option("basePath", uri).parquet(uri)

    def extract(self) -> dict[str, DataFrame]:
        return {
            "prf_accidents": self._extract_dataset(
                self.prf_silver_subpath, "extract_prf_accidents"
            ),
            "dnit_centerline_union": self._extract_dataset(
                self.dnit_centerline_union_subpath,
                "extract_dnit_centerline_union",
            ),
            "dnit_corridor_union": self._extract_dataset(
                self.dnit_corridor_union_subpath,
                "extract_dnit_corridor_union",
            ),
            "municipalities": self._extract_dataset(
                self.municipalities_silver_subpath,
                "extract_municipalities",
            ),
        }

    def _parse_single_union_geometry(
        self,
        dataframe: DataFrame,
        *,
        dataset_name: str,
        transformer: Transformer,
    ) -> BaseGeometry:
        union_row = dataframe.select("geometry").limit(1).collect()
        if not union_row:
            raise ValueError(f"{dataset_name} is empty")

        geometry_wkt = _normalize_string(union_row[0].geometry)
        if geometry_wkt is None:
            raise ValueError(f"{dataset_name} does not contain a valid geometry")

        geometry_geodetic = shapely_wkt.loads(geometry_wkt)
        if geometry_geodetic.is_empty:
            raise ValueError(f"{dataset_name} geometry is empty")

        geometry_proj = shapely_transform(transformer.transform, geometry_geodetic)
        if geometry_proj.is_empty:
            raise ValueError(f"{dataset_name} projected geometry is empty")
        return geometry_proj

    def _point_within_candidates(
        self,
        *,
        point_proj: Point,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for candidate in candidates:
            minx, miny, maxx, maxy = candidate["_bounds"]
            if not (minx <= point_proj.x <= maxx and miny <= point_proj.y <= maxy):
                continue
            if point_proj.within(candidate["_geometry_proj"]):
                return candidate
        return None

    def _build_municipio_scope_artifacts(
        self,
        *,
        municipalities: DataFrame,
        road_union_proj: BaseGeometry,
        corridor_union_proj: BaseGeometry,
        projected_to_geodetic: Transformer,
        geodetic_to_projected: Transformer,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]
    ]:
        retained_municipio_records: list[dict[str, Any]] = []
        section_records: list[dict[str, Any]] = []
        sections_by_municipio: dict[str, list[dict[str, Any]]] = defaultdict(list)

        municipio_rows = (
            municipalities.filter(F.col("sg_uf") == F.lit(SC_UF))
            .select(
                "codigo_municipio",
                "nome_municipio",
                "sg_uf",
                "cd_uf",
                "nm_uf",
                "cd_rgi",
                "nm_rgi",
                "cd_rgint",
                "nm_rgint",
                "cd_regia",
                "nm_regia",
                "sigla_rg",
                "cd_concu",
                "nm_concu",
                "area_km2_ibge",
                "tipo_territorial",
                "data_ingestion",
                "geometry",
            )
            .orderBy("codigo_municipio")
            .collect()
        )

        for row in municipio_rows:
            geometry_wkt = _normalize_string(row.geometry)
            if geometry_wkt is None:
                continue

            geometry_geodetic = shapely_wkt.loads(geometry_wkt)
            if geometry_geodetic.is_empty:
                continue

            geometry_proj = shapely_transform(
                geodetic_to_projected.transform, geometry_geodetic
            )
            if geometry_proj.is_empty or not geometry_proj.intersects(
                corridor_union_proj
            ):
                continue

            retained_record = {
                "codigo_municipio": _normalize_string(row.codigo_municipio),
                "nome_municipio": _normalize_string(row.nome_municipio),
                "sg_uf": _normalize_string(row.sg_uf, upper=True),
                "cd_uf": _normalize_string(row.cd_uf),
                "nm_uf": _normalize_string(row.nm_uf),
                "cd_rgi": _normalize_string(row.cd_rgi),
                "nm_rgi": _normalize_string(row.nm_rgi),
                "cd_rgint": _normalize_string(row.cd_rgint),
                "nm_rgint": _normalize_string(row.nm_rgint),
                "cd_regia": _normalize_string(row.cd_regia),
                "nm_regia": _normalize_string(row.nm_regia),
                "sigla_rg": _normalize_string(row.sigla_rg, upper=True),
                "cd_concu": _normalize_string(row.cd_concu),
                "nm_concu": _normalize_string(row.nm_concu),
                "area_km2_ibge": _normalize_float(row.area_km2_ibge),
                "polygon_area_km2": float(geometry_proj.area / 1_000_000.0),
                "tipo_territorial": _normalize_string(row.tipo_territorial),
                "geometry_type": geometry_proj.geom_type,
                "geometry_crs": self.geodetic_crs.upper(),
                "geometry": shapely_transform(
                    projected_to_geodetic.transform, geometry_proj
                ).wkt,
                "data_ingestion": _normalize_datetime(row.data_ingestion),
                "_geometry_proj": geometry_proj,
                "_bounds": geometry_proj.bounds,
            }
            retained_municipio_records.append(retained_record)

            section_geometry = geometry_proj.intersection(road_union_proj)
            section_linework = list(_iter_line_geometries(section_geometry))
            if not section_linework:
                continue

            line_geodetic = shapely_transform(
                projected_to_geodetic.transform, section_geometry
            )
            section_record = {
                "section_id": f"municipio_{retained_record['codigo_municipio']}",
                "codigo_municipio": retained_record["codigo_municipio"],
                "nome_municipio": retained_record["nome_municipio"],
                "sg_uf": retained_record["sg_uf"],
                "cd_uf": retained_record["cd_uf"],
                "nm_uf": retained_record["nm_uf"],
                "cd_rgi": retained_record["cd_rgi"],
                "nm_rgi": retained_record["nm_rgi"],
                "cd_rgint": retained_record["cd_rgint"],
                "nm_rgint": retained_record["nm_rgint"],
                "polygon_area_km2": retained_record["polygon_area_km2"],
                "road_length_m": float(sum(line.length for line in section_linework)),
                "geometry_type": section_geometry.geom_type,
                "geometry_crs": self.geodetic_crs.upper(),
                "geometry": line_geodetic.wkt,
                "_geometry_proj": section_geometry,
            }
            section_records.append(section_record)
            sections_by_municipio[retained_record["codigo_municipio"]].append(
                section_record
            )

        if not retained_municipio_records:
            raise ValueError(
                "No SC municipios intersecting the BR-101 corridor union were found"
            )
        if not section_records:
            raise ValueError("No BR-101 road sections by municipio could be built")

        for section_group in sections_by_municipio.values():
            section_group.sort(key=lambda item: item["section_id"])

        return retained_municipio_records, section_records, sections_by_municipio

    def _build_canonical_artifacts(
        self,
        *,
        prf_accidents: DataFrame,
        retained_municipio_records: list[dict[str, Any]],
        corridor_union_proj: BaseGeometry,
        geodetic_to_projected: Transformer,
    ) -> tuple[
        list[dict[str, Any]],
        dict[tuple[str | None, date | None], dict[str, Any]],
        dict[tuple[str | None, date | None], dict[str, Any]],
        date,
        date,
        date,
        date,
    ]:
        corridor_bounds = corridor_union_proj.bounds

        # Pre-filter to SC in Spark before collecting to Python
        sc_accidents = prf_accidents.filter(F.col("uf") == F.lit(SC_UF))

        accident_columns = [
            "id",
            "data_inversa",
            "dia_semana",
            "horario",
            "uf",
            "br",
            "km",
            "municipio",
            "causa_acidente",
            "tipo_acidente",
            "classificacao_acidente",
            "fase_dia",
            "sentido_via",
            "condicao_metereologica",
            "tipo_pista",
            "tracado_via",
            "uso_solo",
            "pessoas",
            "mortos",
            "feridos_leves",
            "feridos_graves",
            "ilesos",
            "ignorados",
            "feridos",
            "veiculos",
            "latitude",
            "longitude",
            "br_canonical",
            "is_br101_declared",
            "timestamp",
            "year",
            "month",
            "quarter",
            "week_start",
            "fatal_victims_occ",
            "has_valid_timestamp",
            "has_valid_coords",
        ]
        available_columns = [c for c in accident_columns if c in sc_accidents.columns]

        # Determine temporal bounds from the full SC silver dataset
        time_bounds = (
            sc_accidents.filter(
                F.col("timestamp").isNotNull() & F.col("week_start").isNotNull()
            )
            .select(
                F.min(F.to_date(F.col("timestamp"))).alias("min_date"),
                F.max(F.to_date(F.col("timestamp"))).alias("max_date"),
                F.min("week_start").alias("min_week"),
                F.max("week_start").alias("max_week"),
            )
            .collect()
        )
        if not time_bounds or time_bounds[0].min_date is None:
            raise ValueError(
                "PRF silver SC dataset does not contain any valid timestamps"
            )
        start_day = _normalize_date(time_bounds[0].min_date)
        end_day = _normalize_date(time_bounds[0].max_date)
        start_week = _normalize_date(time_bounds[0].min_week)
        end_week = _normalize_date(time_bounds[0].max_week)
        if any(v is None for v in (start_day, end_day, start_week, end_week)):
            raise ValueError(
                "Unable to determine panel temporal coverage from PRF silver SC data"
            )

        accident_rows = sc_accidents.select(*available_columns).collect()

        canonical_records: list[dict[str, Any]] = []
        day_metrics: dict[tuple[str | None, date | None], dict[str, Any]] = defaultdict(
            lambda: {
                "accident_count": 0,
                "fatal_victims": 0.0,
                "people_involved": 0.0,
                "fatal_accident_count": 0,
            }
        )
        week_metrics: dict[tuple[str | None, date | None], dict[str, Any]] = (
            defaultdict(
                lambda: {
                    "accident_count": 0,
                    "fatal_victims": 0.0,
                    "people_involved": 0.0,
                    "fatal_accident_count": 0,
                }
            )
        )

        minx, miny, maxx, maxy = corridor_bounds
        for row in accident_rows:
            latitude = _normalize_float(getattr(row, "latitude", None))
            longitude = _normalize_float(getattr(row, "longitude", None))
            has_valid_coords = bool(getattr(row, "has_valid_coords", False))
            has_valid_coords = (
                has_valid_coords and latitude is not None and longitude is not None
            )
            is_br101_declared = bool(getattr(row, "is_br101_declared", False))

            point_proj: Point | None = None
            inside_canonical_corridor = False
            if has_valid_coords:
                point_proj = shapely_transform(
                    geodetic_to_projected.transform,
                    Point(longitude, latitude),
                )
                inside_bbox = (
                    minx <= point_proj.x <= maxx and miny <= point_proj.y <= maxy
                )
                if inside_bbox:
                    inside_canonical_corridor = corridor_union_proj.intersects(
                        point_proj
                    )

            accident_class = _assign_accident_class(
                has_valid_coords=has_valid_coords,
                is_br101_declared=is_br101_declared,
                inside_canonical_corridor=inside_canonical_corridor,
            )
            if accident_class != "declared_and_inside":
                continue

            municipio_match = self._point_within_candidates(
                point_proj=point_proj,
                candidates=retained_municipio_records,
            )
            mortos = _normalize_float(getattr(row, "mortos", None)) or 0.0
            pessoas = _normalize_float(getattr(row, "pessoas", None)) or 0.0
            week_start_val = _normalize_date(getattr(row, "week_start", None))
            timestamp_val = _normalize_datetime(getattr(row, "timestamp", None))
            accident_date = timestamp_val.date() if timestamp_val is not None else None
            has_fatality_occ = mortos > 0.0
            codigo_municipio = (
                municipio_match["codigo_municipio"] if municipio_match else None
            )

            canonical_record = {
                "id": _normalize_int(getattr(row, "id", None)),
                "timestamp": timestamp_val,
                "data_inversa": _normalize_string(getattr(row, "data_inversa", None)),
                "horario": _normalize_string(getattr(row, "horario", None)),
                "year": _normalize_int(getattr(row, "year", None)),
                "month": _normalize_int(getattr(row, "month", None)),
                "quarter": _normalize_int(getattr(row, "quarter", None)),
                "week_start": week_start_val,
                "uf": _normalize_string(getattr(row, "uf", None), upper=True),
                "municipio": _normalize_string(getattr(row, "municipio", None)),
                "br": _normalize_string(getattr(row, "br", None)),
                "dia_semana": _normalize_string(getattr(row, "dia_semana", None)),
                "fase_dia": _normalize_string(getattr(row, "fase_dia", None)),
                "condicao_metereologica": _normalize_string(
                    getattr(row, "condicao_metereologica", None)
                ),
                "tipo_acidente": _normalize_string(getattr(row, "tipo_acidente", None)),
                "classificacao_acidente": _normalize_string(
                    getattr(row, "classificacao_acidente", None)
                ),
                "pessoas": pessoas,
                "mortos": mortos,
                "fatal_victims_occ": _normalize_float(
                    getattr(row, "fatal_victims_occ", None)
                ),
                "has_fatality_occ": has_fatality_occ,
                "codigo_municipio": codigo_municipio,
                "nome_municipio": (
                    municipio_match["nome_municipio"] if municipio_match else None
                ),
                "depends_on_tolerant_geometry": False,
            }
            canonical_records.append(canonical_record)

            day_key = (codigo_municipio, accident_date)
            day_entry = day_metrics[day_key]
            day_entry["accident_count"] += 1
            day_entry["fatal_victims"] += mortos
            day_entry["people_involved"] += pessoas
            day_entry["fatal_accident_count"] += int(has_fatality_occ)

            week_key = (codigo_municipio, week_start_val)
            week_entry = week_metrics[week_key]
            week_entry["accident_count"] += 1
            week_entry["fatal_victims"] += mortos
            week_entry["people_involved"] += pessoas
            week_entry["fatal_accident_count"] += int(has_fatality_occ)

        return (
            canonical_records,
            day_metrics,
            week_metrics,
            start_day,
            end_day,
            start_week,
            end_week,
        )

    def _create_dataframe(
        self, records: list[dict[str, Any]], schema: T.StructType
    ) -> DataFrame:
        return self.spark.createDataFrame(records, schema=schema)

    def transform(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
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

        road_union_proj = self._parse_single_union_geometry(
            data["dnit_centerline_union"],
            dataset_name="DNIT BR-101 centerline union",
            transformer=geodetic_to_projected,
        )
        corridor_union_proj = self._parse_single_union_geometry(
            data["dnit_corridor_union"],
            dataset_name="DNIT BR-101 corridor union",
            transformer=geodetic_to_projected,
        )

        retained_municipio_records, section_records, sections_by_municipio = (
            self._build_municipio_scope_artifacts(
                municipalities=data["municipalities"],
                road_union_proj=road_union_proj,
                corridor_union_proj=corridor_union_proj,
                projected_to_geodetic=projected_to_geodetic,
                geodetic_to_projected=geodetic_to_projected,
            )
        )
        self.logger.info(
            "Built %s retained SC municipios and %s BR-101 road section(s) by municipio",
            len(retained_municipio_records),
            len(section_records),
        )

        (
            canonical_records,
            day_metrics,
            week_metrics,
            panel_start_day,
            panel_end_day,
            panel_start_week,
            panel_end_week,
        ) = self._build_canonical_artifacts(
            prf_accidents=data["prf_accidents"],
            retained_municipio_records=retained_municipio_records,
            corridor_union_proj=corridor_union_proj,
            geodetic_to_projected=geodetic_to_projected,
        )
        self.logger.info(
            "Built %s canonical SC accident row(s)",
            len(canonical_records),
        )

        panel_days = _build_day_sequence(panel_start_day, panel_end_day)
        panel_weeks = _build_week_sequence(panel_start_week, panel_end_week)

        # Build municipio metadata lookup from retained records
        municipio_meta: dict[str, dict[str, Any]] = {
            record["codigo_municipio"]: record
            for record in retained_municipio_records
            if record["codigo_municipio"] is not None
        }

        # Build zero-padded day panel
        day_panel_records: list[dict[str, Any]] = []
        for codigo_municipio in sorted(municipio_meta):
            meta = municipio_meta[codigo_municipio]
            for accident_date in panel_days:
                metrics = day_metrics.get(
                    (codigo_municipio, accident_date),
                    {
                        "accident_count": 0,
                        "fatal_victims": 0.0,
                        "people_involved": 0.0,
                        "fatal_accident_count": 0,
                    },
                )
                day_panel_records.append(
                    {
                        "codigo_municipio": codigo_municipio,
                        "nome_municipio": meta["nome_municipio"],
                        "sg_uf": meta["sg_uf"],
                        "cd_rgint": meta["cd_rgint"],
                        "nm_rgint": meta["nm_rgint"],
                        "accident_date": accident_date,
                        "year": accident_date.year,
                        "month": accident_date.month,
                        "day": accident_date.day,
                        "accident_count": int(metrics["accident_count"]),
                        "fatal_victims": float(metrics["fatal_victims"]),
                        "people_involved": float(metrics["people_involved"]),
                        "fatal_accident_count": int(metrics["fatal_accident_count"]),
                        "level_observed": metrics["accident_count"] > 0,
                        "canonical_policy": self.canonical_policy,
                    }
                )

        # Build zero-padded week panel
        week_panel_records: list[dict[str, Any]] = []
        for codigo_municipio in sorted(municipio_meta):
            meta = municipio_meta[codigo_municipio]
            for week_start in panel_weeks:
                metrics = week_metrics.get(
                    (codigo_municipio, week_start),
                    {
                        "accident_count": 0,
                        "fatal_victims": 0.0,
                        "people_involved": 0.0,
                        "fatal_accident_count": 0,
                    },
                )
                accident_count = int(metrics["accident_count"])
                fatal_victims = float(metrics["fatal_victims"])
                people_involved = float(metrics["people_involved"])
                fatal_accident_count = int(metrics["fatal_accident_count"])
                month = week_start.month
                week_panel_records.append(
                    {
                        "codigo_municipio": codigo_municipio,
                        "nome_municipio": meta["nome_municipio"],
                        "sg_uf": meta["sg_uf"],
                        "cd_rgint": meta["cd_rgint"],
                        "nm_rgint": meta["nm_rgint"],
                        "week_start": week_start,
                        "week_end": week_start + timedelta(days=6),
                        "year": week_start.year,
                        "month": month,
                        "quarter": _quarter_for_month(month),
                        "accident_count": accident_count,
                        "fatal_victims": fatal_victims,
                        "people_involved": people_involved,
                        "fatal_accident_count": fatal_accident_count,
                        "fatal_victim_share": (
                            fatal_victims / people_involved
                            if people_involved > 0
                            else None
                        ),
                        "fatal_accident_share": (
                            fatal_accident_count / accident_count
                            if accident_count > 0
                            else None
                        ),
                        "level_observed": accident_count > 0,
                        "canonical_policy": self.canonical_policy,
                    }
                )

        canonical_public = sorted(
            canonical_records,
            key=lambda item: (
                item["timestamp"] or datetime.min,
                item["id"] or -1,
            ),
        )
        section_public = sorted(
            [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in section_records
            ],
            key=lambda item: item["section_id"],
        )
        retained_municipio_public = sorted(
            [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in retained_municipio_records
            ],
            key=lambda item: item["codigo_municipio"] or "",
        )

        # Schemas
        municipio_scope_schema = T.StructType(
            [
                T.StructField("codigo_municipio", T.StringType(), True),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_uf", T.StringType(), True),
                T.StructField("nm_uf", T.StringType(), True),
                T.StructField("cd_rgi", T.StringType(), True),
                T.StructField("nm_rgi", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("cd_regia", T.StringType(), True),
                T.StructField("nm_regia", T.StringType(), True),
                T.StructField("sigla_rg", T.StringType(), True),
                T.StructField("cd_concu", T.StringType(), True),
                T.StructField("nm_concu", T.StringType(), True),
                T.StructField("area_km2_ibge", T.DoubleType(), True),
                T.StructField("polygon_area_km2", T.DoubleType(), True),
                T.StructField("tipo_territorial", T.StringType(), True),
                T.StructField("geometry_type", T.StringType(), True),
                T.StructField("geometry_crs", T.StringType(), True),
                T.StructField("geometry", T.StringType(), True),
                T.StructField("data_ingestion", T.TimestampType(), True),
            ]
        )

        section_schema = T.StructType(
            [
                T.StructField("section_id", T.StringType(), True),
                T.StructField("codigo_municipio", T.StringType(), True),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_uf", T.StringType(), True),
                T.StructField("nm_uf", T.StringType(), True),
                T.StructField("cd_rgi", T.StringType(), True),
                T.StructField("nm_rgi", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("polygon_area_km2", T.DoubleType(), True),
                T.StructField("road_length_m", T.DoubleType(), True),
                T.StructField("geometry_type", T.StringType(), True),
                T.StructField("geometry_crs", T.StringType(), True),
                T.StructField("geometry", T.StringType(), True),
            ]
        )

        canonical_schema = T.StructType(
            [
                T.StructField("id", T.LongType(), True),
                T.StructField("timestamp", T.TimestampType(), True),
                T.StructField("data_inversa", T.StringType(), True),
                T.StructField("horario", T.StringType(), True),
                T.StructField("year", T.IntegerType(), True),
                T.StructField("month", T.IntegerType(), True),
                T.StructField("quarter", T.IntegerType(), True),
                T.StructField("week_start", T.DateType(), True),
                T.StructField("uf", T.StringType(), True),
                T.StructField("municipio", T.StringType(), True),
                T.StructField("br", T.StringType(), True),
                T.StructField("dia_semana", T.StringType(), True),
                T.StructField("fase_dia", T.StringType(), True),
                T.StructField("condicao_metereologica", T.StringType(), True),
                T.StructField("tipo_acidente", T.StringType(), True),
                T.StructField("classificacao_acidente", T.StringType(), True),
                T.StructField("pessoas", T.DoubleType(), True),
                T.StructField("mortos", T.DoubleType(), True),
                T.StructField("fatal_victims_occ", T.DoubleType(), True),
                T.StructField("has_fatality_occ", T.BooleanType(), True),
                T.StructField("codigo_municipio", T.StringType(), True),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("depends_on_tolerant_geometry", T.BooleanType(), True),
            ]
        )

        day_panel_schema = T.StructType(
            [
                T.StructField("codigo_municipio", T.StringType(), True),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("accident_date", T.DateType(), True),
                T.StructField("year", T.IntegerType(), True),
                T.StructField("month", T.IntegerType(), True),
                T.StructField("day", T.IntegerType(), True),
                T.StructField("accident_count", T.IntegerType(), True),
                T.StructField("fatal_victims", T.DoubleType(), True),
                T.StructField("people_involved", T.DoubleType(), True),
                T.StructField("fatal_accident_count", T.IntegerType(), True),
                T.StructField("level_observed", T.BooleanType(), True),
                T.StructField("canonical_policy", T.StringType(), True),
            ]
        )

        week_panel_schema = T.StructType(
            [
                T.StructField("codigo_municipio", T.StringType(), True),
                T.StructField("nome_municipio", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("week_start", T.DateType(), True),
                T.StructField("week_end", T.DateType(), True),
                T.StructField("year", T.IntegerType(), True),
                T.StructField("month", T.IntegerType(), True),
                T.StructField("quarter", T.IntegerType(), True),
                T.StructField("accident_count", T.IntegerType(), True),
                T.StructField("fatal_victims", T.DoubleType(), True),
                T.StructField("people_involved", T.DoubleType(), True),
                T.StructField("fatal_accident_count", T.IntegerType(), True),
                T.StructField("fatal_victim_share", T.DoubleType(), True),
                T.StructField("fatal_accident_share", T.DoubleType(), True),
                T.StructField("level_observed", T.BooleanType(), True),
                T.StructField("canonical_policy", T.StringType(), True),
            ]
        )

        return {
            "municipios_in_scope": self._create_dataframe(
                retained_municipio_public,
                municipio_scope_schema,
            ),
            "road_sections_by_municipio": self._create_dataframe(
                section_public,
                section_schema,
            ),
            "canonical_accidents_by_municipio": self._create_dataframe(
                canonical_public,
                canonical_schema,
            ),
            "canonical_accidents_by_municipio_day": self._create_dataframe(
                day_panel_records,
                day_panel_schema,
            ),
            "canonical_accidents_by_municipio_week": self._create_dataframe(
                week_panel_records,
                week_panel_schema,
            ),
        }

    def load(self, data: dict[str, DataFrame]) -> str:
        staging_output_dir = self._staging_dir / "gold_br101_sc_municipio_panel"
        partition_map = {
            "municipios_in_scope": [],
            "road_sections_by_municipio": [],
            "canonical_accidents_by_municipio": ["year"],
            "canonical_accidents_by_municipio_day": ["year"],
            "canonical_accidents_by_municipio_week": ["year"],
        }

        for artifact_name, dataframe in data.items():
            artifact_staging_dir = staging_output_dir / artifact_name
            writer = dataframe.write.mode(self.write_mode)
            partition_columns = partition_map.get(artifact_name, [])
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            writer.parquet(str(artifact_staging_dir))

        destination = self.datalake.persist_directory(
            staging_output_dir,
            self.gold_subpath,
        )
        self.logger.info("Saved BR-101 SC municipio gold artifacts to %s", destination)
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    spark = build_spark_session("BR-101 SC Municipio Panel Silver to Gold")
    job = Br101ScMunicipioSilver2Gold(spark=spark)
    job.run()
    spark.stop()
