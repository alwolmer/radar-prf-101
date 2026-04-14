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


def _build_week_sequence(start_week: date, end_week: date) -> list[date]:
    weeks: list[date] = []
    current_week = start_week
    while current_week <= end_week:
        weeks.append(current_week)
        current_week += timedelta(days=7)
    return weeks


class Br101RgiWeeklyPanelSilver2Gold(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            spark=spark,
            config=config,
            job_name="br101_rgi_weekly_panel_silver2gold",
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
        self.rgis_silver_subpath = str(
            self.config.get(
                "rgis_silver_subpath",
                "silver/ibge_territorial_preprocessed/rgis_preprocessed",
            )
        )
        self.gold_subpath = str(
            self.config.get(
                "gold_subpath",
                "gold/br101_rgi_weekly_panel",
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
            raise ValueError("Br101RgiWeeklyPanelSilver2Gold requires a Spark session")

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
            "rgis": self._extract_dataset(
                self.rgis_silver_subpath,
                "extract_rgis",
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

    def _build_rgi_scope_artifacts(
        self,
        *,
        rgis: DataFrame,
        road_union_proj: BaseGeometry,
        corridor_union_proj: BaseGeometry,
        projected_to_geodetic: Transformer,
        geodetic_to_projected: Transformer,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]
    ]:
        retained_rgi_records: list[dict[str, Any]] = []
        section_records: list[dict[str, Any]] = []
        sections_by_rgi: dict[str, list[dict[str, Any]]] = defaultdict(list)

        rgi_rows = (
            rgis.select(
                "codigo_rgi",
                "nome_rgi",
                "sg_uf",
                "cd_uf",
                "nm_uf",
                "cd_rgint",
                "nm_rgint",
                "cd_regia",
                "nm_regia",
                "sigla_rg",
                "area_km2_ibge",
                "tipo_territorial",
                "data_ingestion",
                "geometry",
            )
            .orderBy("sg_uf", "codigo_rgi")
            .collect()
        )

        for row in rgi_rows:
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
                "codigo_rgi": _normalize_string(row.codigo_rgi),
                "nome_rgi": _normalize_string(row.nome_rgi),
                "sg_uf": _normalize_string(row.sg_uf, upper=True),
                "cd_uf": _normalize_string(row.cd_uf),
                "nm_uf": _normalize_string(row.nm_uf),
                "cd_rgint": _normalize_string(row.cd_rgint),
                "nm_rgint": _normalize_string(row.nm_rgint),
                "cd_regia": _normalize_string(row.cd_regia),
                "nm_regia": _normalize_string(row.nm_regia),
                "sigla_rg": _normalize_string(row.sigla_rg, upper=True),
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
            retained_rgi_records.append(retained_record)

            section_geometry = geometry_proj.intersection(road_union_proj)
            section_linework = list(_iter_line_geometries(section_geometry))
            if not section_linework:
                continue

            canonical_section_geometry = section_geometry
            line_geodetic = shapely_transform(
                projected_to_geodetic.transform, canonical_section_geometry
            )
            section_record = {
                "section_id": f"rgi_{retained_record['codigo_rgi']}",
                "codigo_rgi": retained_record["codigo_rgi"],
                "nome_rgi": retained_record["nome_rgi"],
                "sg_uf": retained_record["sg_uf"],
                "cd_uf": retained_record["cd_uf"],
                "nm_uf": retained_record["nm_uf"],
                "cd_rgint": retained_record["cd_rgint"],
                "nm_rgint": retained_record["nm_rgint"],
                "polygon_area_km2": retained_record["polygon_area_km2"],
                "road_length_m": float(sum(line.length for line in section_linework)),
                "geometry_type": canonical_section_geometry.geom_type,
                "geometry_crs": self.geodetic_crs.upper(),
                "geometry": line_geodetic.wkt,
                "_geometry_proj": canonical_section_geometry,
            }
            section_records.append(section_record)
            sections_by_rgi[retained_record["codigo_rgi"]].append(section_record)

        if not retained_rgi_records:
            raise ValueError(
                "No RGIs intersecting the BR-101 corridor union were found"
            )
        if not section_records:
            raise ValueError("No BR-101 road sections by RGI could be built")

        for section_group in sections_by_rgi.values():
            section_group.sort(key=lambda item: item["section_id"])

        return retained_rgi_records, section_records, sections_by_rgi

    def _build_canonical_artifacts(
        self,
        *,
        prf_accidents: DataFrame,
        retained_rgi_records: list[dict[str, Any]],
        sections_by_rgi: dict[str, list[dict[str, Any]]],
        corridor_union_proj: BaseGeometry,
        geodetic_to_projected: Transformer,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[tuple[str, date], dict[str, float]],
        dict[tuple[str, date], dict[str, float]],
        date,
        date,
    ]:
        corridor_bounds = corridor_union_proj.bounds
        rgis_by_uf: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in retained_rgi_records:
            rgis_by_uf[record["sg_uf"]].append(record)

        ordered_rgis = sorted(
            retained_rgi_records,
            key=lambda item: (
                item["sg_uf"] or "",
                item["codigo_rgi"] or "",
            ),
        )

        week_bounds = (
            prf_accidents.filter(F.col("week_start").isNotNull())
            .select(
                F.min("week_start").alias("min_week_start"),
                F.max("week_start").alias("max_week_start"),
            )
            .collect()
        )
        if not week_bounds or week_bounds[0].min_week_start is None:
            raise ValueError(
                "PRF silver dataset does not contain any valid week_start value"
            )

        start_week = _normalize_date(week_bounds[0].min_week_start)
        end_week = _normalize_date(week_bounds[0].max_week_start)
        if start_week is None or end_week is None:
            raise ValueError(
                "Unable to determine panel temporal coverage from PRF silver"
            )

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
        available_columns = [
            column_name
            for column_name in accident_columns
            if column_name in prf_accidents.columns
        ]
        accident_rows = prf_accidents.select(*available_columns).collect()

        canonical_records: list[dict[str, Any]] = []
        unmatched_records: list[dict[str, Any]] = []
        section_week_metrics: dict[tuple[str, date], dict[str, float]] = defaultdict(
            lambda: {
                "accident_count": 0.0,
                "fatal_victims": 0.0,
                "people_involved": 0.0,
                "fatal_accident_count": 0.0,
            }
        )
        rgi_week_metrics: dict[tuple[str, date], dict[str, float]] = defaultdict(
            lambda: {
                "accident_count": 0.0,
                "fatal_victims": 0.0,
                "people_involved": 0.0,
                "fatal_accident_count": 0.0,
            }
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

            accident_uf = _normalize_string(getattr(row, "uf", None), upper=True)
            rgi_match = self._point_within_candidates(
                point_proj=point_proj,
                candidates=rgis_by_uf.get(accident_uf, []),
            )
            if rgi_match is None and point_proj is not None:
                rgi_match = self._point_within_candidates(
                    point_proj=point_proj,
                    candidates=ordered_rgis,
                )

            rgi_match_status = (
                "matched_exact_polygon"
                if rgi_match is not None
                else "unmatched_exact_polygon"
            )

            section_match: dict[str, Any] | None = None
            section_match_status = "unmatched_exact_polygon"
            if rgi_match is not None and point_proj is not None:
                candidate_sections = sections_by_rgi.get(rgi_match["codigo_rgi"], [])
                if candidate_sections:
                    section_match = min(
                        candidate_sections,
                        key=lambda item: (
                            point_proj.distance(item["_geometry_proj"]),
                            -(item["road_length_m"] or 0.0),
                            item["section_id"],
                        ),
                    )
                    section_match_status = "matched_nearest_section"
                else:
                    section_match_status = "unmatched_no_section_in_rgi"

            mortos = _normalize_float(getattr(row, "mortos", None)) or 0.0
            pessoas = _normalize_float(getattr(row, "pessoas", None)) or 0.0
            week_start = _normalize_date(getattr(row, "week_start", None))
            has_fatality_occ = mortos > 0.0

            canonical_record = {
                "id": _normalize_int(getattr(row, "id", None)),
                "timestamp": _normalize_datetime(getattr(row, "timestamp", None)),
                "data_inversa": _normalize_string(getattr(row, "data_inversa", None)),
                "horario": _normalize_string(getattr(row, "horario", None)),
                "year": _normalize_int(getattr(row, "year", None)),
                "month": _normalize_int(getattr(row, "month", None)),
                "quarter": _normalize_int(getattr(row, "quarter", None)),
                "week_start": week_start,
                "uf": accident_uf,
                "municipio": _normalize_string(getattr(row, "municipio", None)),
                "br": _normalize_string(getattr(row, "br", None)),
                "br_canonical": _normalize_string(getattr(row, "br_canonical", None)),
                "km": _normalize_float(getattr(row, "km", None)),
                "latitude": latitude,
                "longitude": longitude,
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
                "inside_canonical_corridor": inside_canonical_corridor,
                "accident_class": accident_class,
                "canonical_policy": self.canonical_policy,
                "codigo_rgi": rgi_match["codigo_rgi"] if rgi_match else None,
                "nome_rgi": rgi_match["nome_rgi"] if rgi_match else None,
                "cd_rgint": rgi_match["cd_rgint"] if rgi_match else None,
                "nm_rgint": rgi_match["nm_rgint"] if rgi_match else None,
                "rgi_match_status": rgi_match_status,
                "section_id": section_match["section_id"] if section_match else None,
                "section_match_status": section_match_status,
                "assignment_rule": "official_polygon_exact_then_nearest_section_within_rgi",
                "depends_on_tolerant_geometry": False,
            }
            canonical_records.append(canonical_record)

            if (
                canonical_record["rgi_match_status"] != "matched_exact_polygon"
                or canonical_record["section_match_status"] != "matched_nearest_section"
            ):
                unmatched_records.append(canonical_record.copy())

            if (
                canonical_record["section_id"] is not None
                and canonical_record["codigo_rgi"] is not None
                and canonical_record["week_start"] is not None
            ):
                section_metrics = section_week_metrics[
                    (canonical_record["section_id"], canonical_record["week_start"])
                ]
                section_metrics["accident_count"] += 1.0
                section_metrics["fatal_victims"] += mortos
                section_metrics["people_involved"] += pessoas
                section_metrics["fatal_accident_count"] += float(has_fatality_occ)

                rgi_metrics = rgi_week_metrics[
                    (canonical_record["codigo_rgi"], canonical_record["week_start"])
                ]
                rgi_metrics["accident_count"] += 1.0
                rgi_metrics["fatal_victims"] += mortos
                rgi_metrics["people_involved"] += pessoas
                rgi_metrics["fatal_accident_count"] += float(has_fatality_occ)

        return (
            canonical_records,
            unmatched_records,
            section_week_metrics,
            rgi_week_metrics,
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

        retained_rgi_records, section_records, sections_by_rgi = (
            self._build_rgi_scope_artifacts(
                rgis=data["rgis"],
                road_union_proj=road_union_proj,
                corridor_union_proj=corridor_union_proj,
                projected_to_geodetic=projected_to_geodetic,
                geodetic_to_projected=geodetic_to_projected,
            )
        )
        self.logger.info(
            "Built %s retained RGIs and %s BR-101 road section(s) by RGI",
            len(retained_rgi_records),
            len(section_records),
        )

        (
            canonical_records,
            unmatched_records,
            section_week_metrics,
            rgi_week_metrics,
            panel_start_week,
            panel_end_week,
        ) = self._build_canonical_artifacts(
            prf_accidents=data["prf_accidents"],
            retained_rgi_records=retained_rgi_records,
            sections_by_rgi=sections_by_rgi,
            corridor_union_proj=corridor_union_proj,
            geodetic_to_projected=geodetic_to_projected,
        )
        self.logger.info(
            "Built %s canonical accident row(s), with %s unmatched assignment row(s)",
            len(canonical_records),
            len(unmatched_records),
        )

        panel_weeks = _build_week_sequence(panel_start_week, panel_end_week)

        retained_rgi_public = sorted(
            [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in retained_rgi_records
            ],
            key=lambda item: (
                item["sg_uf"] or "",
                item["codigo_rgi"] or "",
            ),
        )
        section_public = sorted(
            [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in section_records
            ],
            key=lambda item: item["section_id"],
        )
        canonical_public = sorted(
            canonical_records,
            key=lambda item: (
                item["timestamp"] or datetime.min,
                item["id"] or -1,
            ),
        )
        unmatched_public = sorted(
            unmatched_records,
            key=lambda item: (
                item["timestamp"] or datetime.min,
                item["id"] or -1,
            ),
        )

        section_panel_records: list[dict[str, Any]] = []
        for section in section_public:
            for week_start in panel_weeks:
                metrics = section_week_metrics.get(
                    (section["section_id"], week_start),
                    {
                        "accident_count": 0.0,
                        "fatal_victims": 0.0,
                        "people_involved": 0.0,
                        "fatal_accident_count": 0.0,
                    },
                )
                accident_count = int(metrics["accident_count"])
                fatal_victims = float(metrics["fatal_victims"])
                people_involved = float(metrics["people_involved"])
                fatal_accident_count = int(metrics["fatal_accident_count"])
                month = week_start.month

                section_panel_records.append(
                    {
                        "section_id": section["section_id"],
                        "codigo_rgi": section["codigo_rgi"],
                        "nome_rgi": section["nome_rgi"],
                        "sg_uf": section["sg_uf"],
                        "cd_rgint": section["cd_rgint"],
                        "nm_rgint": section["nm_rgint"],
                        "road_length_m": section["road_length_m"],
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

        rgi_metadata: dict[str, dict[str, Any]] = {}
        for section in section_public:
            metadata = rgi_metadata.setdefault(
                section["codigo_rgi"],
                {
                    "codigo_rgi": section["codigo_rgi"],
                    "nome_rgi": section["nome_rgi"],
                    "sg_uf": section["sg_uf"],
                    "cd_rgint": section["cd_rgint"],
                    "nm_rgint": section["nm_rgint"],
                    "road_length_m": 0.0,
                    "section_count": 0,
                },
            )
            metadata["road_length_m"] += float(section["road_length_m"] or 0.0)
            metadata["section_count"] += 1

        rgi_panel_records: list[dict[str, Any]] = []
        for codigo_rgi in sorted(rgi_metadata):
            metadata = rgi_metadata[codigo_rgi]
            for week_start in panel_weeks:
                metrics = rgi_week_metrics.get(
                    (codigo_rgi, week_start),
                    {
                        "accident_count": 0.0,
                        "fatal_victims": 0.0,
                        "people_involved": 0.0,
                        "fatal_accident_count": 0.0,
                    },
                )
                accident_count = int(metrics["accident_count"])
                fatal_victims = float(metrics["fatal_victims"])
                people_involved = float(metrics["people_involved"])
                fatal_accident_count = int(metrics["fatal_accident_count"])
                month = week_start.month

                rgi_panel_records.append(
                    {
                        "codigo_rgi": metadata["codigo_rgi"],
                        "nome_rgi": metadata["nome_rgi"],
                        "sg_uf": metadata["sg_uf"],
                        "cd_rgint": metadata["cd_rgint"],
                        "nm_rgint": metadata["nm_rgint"],
                        "road_length_m": metadata["road_length_m"],
                        "section_count": metadata["section_count"],
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

        rgi_scope_schema = T.StructType(
            [
                T.StructField("codigo_rgi", T.StringType(), True),
                T.StructField("nome_rgi", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_uf", T.StringType(), True),
                T.StructField("nm_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("cd_regia", T.StringType(), True),
                T.StructField("nm_regia", T.StringType(), True),
                T.StructField("sigla_rg", T.StringType(), True),
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
                T.StructField("codigo_rgi", T.StringType(), True),
                T.StructField("nome_rgi", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_uf", T.StringType(), True),
                T.StructField("nm_uf", T.StringType(), True),
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
                T.StructField("br_canonical", T.StringType(), True),
                T.StructField("km", T.DoubleType(), True),
                T.StructField("latitude", T.DoubleType(), True),
                T.StructField("longitude", T.DoubleType(), True),
                T.StructField("dia_semana", T.StringType(), True),
                T.StructField("fase_dia", T.StringType(), True),
                T.StructField("condicao_metereologica", T.StringType(), True),
                T.StructField("tipo_acidente", T.StringType(), True),
                T.StructField("classificacao_acidente", T.StringType(), True),
                T.StructField("pessoas", T.DoubleType(), True),
                T.StructField("mortos", T.DoubleType(), True),
                T.StructField("fatal_victims_occ", T.DoubleType(), True),
                T.StructField("has_fatality_occ", T.BooleanType(), True),
                T.StructField("inside_canonical_corridor", T.BooleanType(), True),
                T.StructField("accident_class", T.StringType(), True),
                T.StructField("canonical_policy", T.StringType(), True),
                T.StructField("codigo_rgi", T.StringType(), True),
                T.StructField("nome_rgi", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("rgi_match_status", T.StringType(), True),
                T.StructField("section_id", T.StringType(), True),
                T.StructField("section_match_status", T.StringType(), True),
                T.StructField("assignment_rule", T.StringType(), True),
                T.StructField("depends_on_tolerant_geometry", T.BooleanType(), True),
            ]
        )
        section_panel_schema = T.StructType(
            [
                T.StructField("section_id", T.StringType(), True),
                T.StructField("codigo_rgi", T.StringType(), True),
                T.StructField("nome_rgi", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("road_length_m", T.DoubleType(), True),
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
        rgi_panel_schema = T.StructType(
            [
                T.StructField("codigo_rgi", T.StringType(), True),
                T.StructField("nome_rgi", T.StringType(), True),
                T.StructField("sg_uf", T.StringType(), True),
                T.StructField("cd_rgint", T.StringType(), True),
                T.StructField("nm_rgint", T.StringType(), True),
                T.StructField("road_length_m", T.DoubleType(), True),
                T.StructField("section_count", T.IntegerType(), True),
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
            "rgis_in_scope": self._create_dataframe(
                retained_rgi_public,
                rgi_scope_schema,
            ),
            "road_sections_by_rgi": self._create_dataframe(
                section_public,
                section_schema,
            ),
            "canonical_accidents_by_rgi_section": self._create_dataframe(
                canonical_public,
                canonical_schema,
            ),
            "canonical_accidents_unmatched_assignments": self._create_dataframe(
                unmatched_public,
                canonical_schema,
            ),
            "historical_accidents_by_rgi_section_week": self._create_dataframe(
                section_panel_records,
                section_panel_schema,
            ),
            "historical_accidents_by_rgi_week": self._create_dataframe(
                rgi_panel_records,
                rgi_panel_schema,
            ),
        }

    def load(self, data: dict[str, DataFrame]) -> str:
        staging_output_dir = self._staging_dir / "gold_br101_rgi_weekly_panel"
        partition_map = {
            "rgis_in_scope": ["sg_uf"],
            "road_sections_by_rgi": ["sg_uf"],
            "canonical_accidents_by_rgi_section": ["year"],
            "canonical_accidents_unmatched_assignments": ["year"],
            "historical_accidents_by_rgi_section_week": ["year"],
            "historical_accidents_by_rgi_week": ["year"],
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
        self.logger.info("Saved BR-101 RGI gold artifacts to %s", destination)
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    spark = build_spark_session("BR-101 RGI Weekly Panel Silver to Gold")
    job = Br101RgiWeeklyPanelSilver2Gold(spark=spark)
    job.run()
    spark.stop()
