from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sedona.spark import SedonaContext

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter, S3DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_GEODETIC_CRS = "epsg:4674"
DEFAULT_PROJECTED_CRS = "epsg:5880"
DEFAULT_BUFFER_METERS = 500.0


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


class DnitBronze2Silver(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="dnit_bronze2silver")
        self.datalake: DatalakeAdapter = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(
            self.config.get("bronze_subpath", "bronze/dnit_road_network")
        )
        self.silver_subpath = str(
            self.config.get("silver_subpath", "silver/dnit_br101_corridor")
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.geodetic_crs = str(
            self.config.get("geodetic_crs", DEFAULT_GEODETIC_CRS)
        ).lower()
        self.projected_crs = str(
            self.config.get("projected_crs", DEFAULT_PROJECTED_CRS)
        ).lower()
        self.corridor_buffer_meters = float(
            self.config.get("corridor_buffer_meters", DEFAULT_BUFFER_METERS)
        )
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)
        self.spark = SedonaContext.create(self.spark)

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError("DnitBronze2Silver requires a Spark session")
        buffer_meters = float(
            self.config.get("corridor_buffer_meters", DEFAULT_BUFFER_METERS)
        )
        if buffer_meters <= 0:
            raise ValueError("corridor_buffer_meters must be positive")

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def extract(self) -> DataFrame:
        uri = self.datalake.uri_for(self.bronze_subpath).replace("s3://", "s3a://")
        if uri.startswith("s3a://"):
            _configure_spark_s3(self.spark, self.datalake)
        else:
            uri = str(
                self.datalake.stage_directory(
                    self.bronze_subpath,
                    self._staging_dir / "extract",
                )
            )

        return self.spark.read.option("basePath", uri).parquet(uri)

    def transform(self, data: DataFrame) -> dict[str, DataFrame]:
        raw_segments = (
            data.filter(F.col("br_canonical") == F.lit("101"))
            .select(
                F.col("snapshot_year").cast("int").alias("snapshot_year"),
                F.col("versao_snv"),
                F.col("geometry"),
            )
            .withColumn("geometry_geodetic", F.expr("ST_GeomFromWKT(geometry)"))
            .drop("geometry")
            .filter(F.col("geometry_geodetic").isNotNull())
            .withColumn(
                "geometry_proj",
                F.expr(
                    "ST_Transform("
                    f"geometry_geodetic, '{self.geodetic_crs}', '{self.projected_crs}'"
                    ")"
                ),
            )
            .drop("geometry_geodetic")
            .filter(F.expr("NOT ST_IsEmpty(geometry_proj)"))
        )

        if raw_segments.limit(1).count() == 0:
            raise ValueError("No canonical BR-101 DNIT geometries were found in bronze")

        centerlines_proj = raw_segments.groupBy("snapshot_year").agg(
            F.first("versao_snv", ignorenulls=True).alias("versao_snv"),
            F.expr("ST_Union_Aggr(geometry_proj)").alias("geometry_proj"),
        )

        centerlines_proj = centerlines_proj.filter(
            F.col("geometry_proj").isNotNull()
        ).filter(F.expr("NOT ST_IsEmpty(geometry_proj)"))

        yearly_centerlines = centerlines_proj.select(
            "snapshot_year",
            "versao_snv",
            F.regexp_replace(
                F.expr("ST_GeometryType(geometry_proj)"), r"^ST_", ""
            ).alias("geometry_type"),
            F.lit(self.geodetic_crs.upper()).alias("geometry_crs"),
            F.expr(
                "ST_AsText("
                "ST_Transform("
                f"geometry_proj, '{self.projected_crs}', '{self.geodetic_crs}'"
                ")"
                ")"
            ).alias("geometry"),
        )

        corridors_proj = centerlines_proj.select(
            "snapshot_year",
            "versao_snv",
            F.expr(f"ST_Buffer(geometry_proj, {self.corridor_buffer_meters})").alias(
                "geometry_proj"
            ),
        ).filter(F.expr("NOT ST_IsEmpty(geometry_proj)"))

        yearly_corridors = corridors_proj.select(
            "snapshot_year",
            "versao_snv",
            F.lit(int(self.corridor_buffer_meters)).alias("buffer_meters"),
            (F.expr("ST_Area(geometry_proj)") / F.lit(1_000_000.0)).alias(
                "corridor_area_km2"
            ),
            F.lit(self.geodetic_crs.upper()).alias("geometry_crs"),
            F.expr(
                "ST_AsText("
                "ST_Transform("
                f"geometry_proj, '{self.projected_crs}', '{self.geodetic_crs}'"
                ")"
                ")"
            ).alias("geometry"),
        )

        road_union_proj = centerlines_proj.agg(
            F.expr("ST_Union_Aggr(geometry_proj)").alias("geometry_proj")
        ).filter(F.expr("geometry_proj IS NOT NULL AND NOT ST_IsEmpty(geometry_proj)"))

        centerline_union = road_union_proj.select(
            F.lit("union_all_years").alias("corridor_policy"),
            F.regexp_replace(
                F.expr("ST_GeometryType(geometry_proj)"), r"^ST_", ""
            ).alias("geometry_type"),
            F.lit(self.geodetic_crs.upper()).alias("geometry_crs"),
            F.expr(
                "ST_AsText("
                "ST_Transform("
                f"geometry_proj, '{self.projected_crs}', '{self.geodetic_crs}'"
                ")"
                ")"
            ).alias("geometry"),
        )

        corridor_union_proj = corridors_proj.agg(
            F.expr("ST_Union_Aggr(geometry_proj)").alias("geometry_proj")
        ).filter(F.expr("geometry_proj IS NOT NULL AND NOT ST_IsEmpty(geometry_proj)"))

        corridor_union = corridor_union_proj.select(
            F.lit("union_all_years").alias("corridor_policy"),
            F.lit(int(self.corridor_buffer_meters)).alias("buffer_meters"),
            (F.expr("ST_Area(geometry_proj)") / F.lit(1_000_000.0)).alias(
                "corridor_area_km2"
            ),
            F.lit(self.geodetic_crs.upper()).alias("geometry_crs"),
            F.expr(
                "ST_AsText("
                "ST_Transform("
                f"geometry_proj, '{self.projected_crs}', '{self.geodetic_crs}'"
                ")"
                ")"
            ).alias("geometry"),
        )

        return {
            "br101_centerlines_by_snapshot": yearly_centerlines,
            "br101_corridors_by_snapshot": yearly_corridors,
            "br101_centerline_union": centerline_union,
            "br101_corridor_union": corridor_union,
        }

    def load(self, data: dict[str, DataFrame]) -> str:
        staging_output_dir = self._staging_dir / "silver_dnit_br101_corridor"

        yearly_datasets = {
            "br101_centerlines_by_snapshot",
            "br101_corridors_by_snapshot",
        }

        for artifact_name, dataframe in data.items():
            artifact_staging_dir = staging_output_dir / artifact_name
            writer = dataframe.write.mode(self.write_mode)
            if artifact_name in yearly_datasets:
                writer = writer.partitionBy("snapshot_year")
            writer.parquet(str(artifact_staging_dir))

        destination = self.datalake.persist_directory(
            staging_output_dir, self.silver_subpath
        )
        self.logger.info("Saved DNIT silver corridor datasets to %s", destination)
        return destination


if __name__ == "__main__":
    spark = build_spark_session("DNIT Bronze to Silver", include_sedona=True)
    job = DnitBronze2Silver(spark=spark)
    job.run()
    spark.stop()
