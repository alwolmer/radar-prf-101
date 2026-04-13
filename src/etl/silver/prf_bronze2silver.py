from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter, S3DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
BRAZIL_BOUNDS: dict[str, int | float] = {
    "lat_min": -35.0,
    "lat_max": 6.0,
    "lon_min": -75.0,
    "lon_max": -28.0,
}
INTEGER_LIKE_COLUMNS: list[str] = [
    "id",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
]


def normalize_numeric(column_name: str) -> F.Column:
    raw_text: Column = F.trim(F.col(column_name).cast("string"))
    cleaned: Column = F.regexp_replace(raw_text, ",", ".")

    return (
        F.when(F.col(column_name).isNull(), F.lit(None).cast("double"))
        .when(cleaned == "", F.lit(None).cast("double"))
        .when(F.lower(cleaned).isin("nan", "none", "null"), F.lit(None).cast("double"))
        .otherwise(cleaned.cast("double"))
    )


def canonicalize_road_code(column_name: str) -> F.Column:
    extracted: Column = F.regexp_extract(F.col(column_name).cast("string"), r"(\d+)", 1)
    stripped: Column = F.regexp_replace(extracted, r"^0+", "")
    return F.when(stripped == "", F.lit(None).cast("string")).otherwise(stripped)


def prepare_week_start(column_name: str) -> F.Column:
    return F.to_date(F.date_trunc("week", F.col(column_name)))


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


def build_partition_uri(base_uri: str, partition_name: str) -> str:
    if "://" in base_uri:
        stripped = base_uri.rstrip("/")
        return f"{stripped}/{partition_name}"
    return str(Path(base_uri) / partition_name)


class PrfBronze2Silver(BaseETLJob):
    def __init__(self, spark: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(spark=spark, config=config, job_name="prf_bronze2silver")
        self.datalake: DatalakeAdapter = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(
            self.config.get("bronze_subpath", "bronze/prf_accidents")
        )
        self.silver_subpath = str(
            self.config.get("silver_subpath", "silver/prf_accidents_standardized")
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError("PrfBronze2Silver requires a Spark session")

    def extract(self) -> DataFrame:
        uri = self.datalake.uri_for(self.bronze_subpath).replace("s3://", "s3a://")
        if uri.startswith("s3a://"):
            if self.spark is None:
                raise ValueError("Spark session is required to configure S3 access")
            _configure_spark_s3(self.spark, self.datalake)
        else:
            uri = str(
                self.datalake.stage_directory(
                    self.bronze_subpath,
                    self._staging_dir / "extract",
                )
            )
        if self.spark is None:
            raise ValueError("Spark session is required to read data")
        return self.spark.read.option("basePath", uri).parquet(
            build_partition_uri(uri, "br=101")
        )

    def transform(self, data: DataFrame) -> DataFrame:
        base_projection: list[F.Column] = []
        for column_name in data.columns:
            if column_name == "uf":
                base_projection.append(
                    F.upper(F.trim(F.col("uf").cast("string"))).alias("uf")
                )
            elif column_name == "municipio":
                base_projection.append(
                    F.trim(F.col("municipio").cast("string")).alias("municipio")
                )
            elif (
                column_name in {"km", "latitude", "longitude"}
                or column_name in INTEGER_LIKE_COLUMNS
            ):
                base_projection.append(
                    normalize_numeric(column_name).alias(column_name)
                )
            else:
                base_projection.append(F.col(column_name))

        normalized: DataFrame = data.select(*base_projection)

        date_text: Column = F.trim(F.col("data_inversa").cast("string"))
        time_text: Column = F.trim(F.col("horario").cast("string"))
        timestamp_input: Column = F.when(
            date_text.isNull()
            | time_text.isNull()
            | (date_text == "")
            | (time_text == ""),
            F.lit(None).cast("string"),
        ).otherwise(F.concat(date_text, F.lit(" "), time_text))
        timestamp: Column = F.expr(
            "try_to_timestamp(_timestamp_input, 'yyyy-MM-dd HH:mm:ss')"
        )
        br_canonical: Column = canonicalize_road_code("br")
        has_valid_coords: Column = F.col("latitude").between(
            BRAZIL_BOUNDS["lat_min"], BRAZIL_BOUNDS["lat_max"]
        ) & F.col("longitude").between(
            BRAZIL_BOUNDS["lon_min"], BRAZIL_BOUNDS["lon_max"]
        )

        passthrough_columns: list[Column] = [
            F.col(column_name)
            for column_name in normalized.columns
            if column_name not in {"latitude", "longitude"}
        ]

        standardized: DataFrame = normalized.withColumn(
            "_timestamp_input", timestamp_input
        ).select(
            *passthrough_columns,
            br_canonical.alias("br_canonical"),
            (br_canonical == F.lit("101")).alias("is_br101_declared"),
            timestamp.alias("timestamp"),
            F.year(timestamp).alias("year"),
            F.month(timestamp).alias("month"),
            F.quarter(timestamp).alias("quarter"),
            F.to_date(F.date_trunc("week", timestamp)).alias("week_start"),
            F.coalesce(F.col("mortos"), F.lit(0.0))
            .cast("int")
            .alias("fatal_victims_occ"),
            timestamp.isNotNull().alias("has_valid_timestamp"),
            has_valid_coords.alias("has_valid_coords"),
            F.when(has_valid_coords, F.col("latitude"))
            .otherwise(F.lit(None).cast("double"))
            .alias("latitude"),
            F.when(has_valid_coords, F.col("longitude"))
            .otherwise(F.lit(None).cast("double"))
            .alias("longitude"),
        )

        standardized: DataFrame = standardized.drop("_timestamp_input")
        # Silver PRF accidents are intentionally restricted to canonical BR-101 records only.
        standardized: DataFrame = standardized.filter(
            F.col("br_canonical") == F.lit("101")
        )

        ordered_columns = (
            [
                column_name
                for column_name in normalized.columns
                if column_name not in {"latitude", "longitude"}
            ]
            + [
                "latitude",
                "longitude",
            ]
            + [
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
        )
        return standardized.select(*ordered_columns)

    def load(self, data: DataFrame) -> str:
        staging_output_dir = self._staging_dir / "silver_prf_accidents_standardized"
        data.write.mode(self.write_mode).partitionBy("year").parquet(
            str(staging_output_dir)
        )
        destination = self.datalake.persist_directory(
            staging_output_dir, self.silver_subpath
        )
        self.logger.info("Saved partitioned PRF silver dataset to %s", destination)
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    spark = build_spark_session("PRF Bronze to Silver")
    job = PrfBronze2Silver(spark=spark)
    job.run()
    spark.stop()
