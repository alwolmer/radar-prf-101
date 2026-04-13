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
DEFAULT_GEODETIC_CRS = "epsg:4674"
DEFAULT_PROJECTED_CRS = "epsg:5880"
BR101_UFS: tuple[str, ...] = (
    "AL",
    "BA",
    "ES",
    "PB",
    "PE",
    "PR",
    "RJ",
    "RN",
    "RS",
    "SC",
    "SE",
    "SP",
)


def normalize_string(column_name: str) -> Column:
    cleaned: Column = F.trim(F.col(column_name).cast("string"))
    return F.when(F.col(column_name).isNull() | (cleaned == ""), F.lit(None)).otherwise(
        cleaned
    )


def normalize_numeric(column_name: str) -> Column:
    cleaned: Column = F.regexp_replace(
        F.trim(F.col(column_name).cast("string")), ",", "."
    )
    return (
        F.when(F.col(column_name).isNull(), F.lit(None).cast("double"))
        .when(cleaned == "", F.lit(None).cast("double"))
        .when(F.lower(cleaned).isin("nan", "none", "null"), F.lit(None).cast("double"))
        .otherwise(cleaned.cast("double"))
    )


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


class IbgeBronze2Silver(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="ibge_bronze2silver")
        self.datalake: DatalakeAdapter = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.municipios_bronze_subpath = str(
            self.config.get("municipios_bronze_subpath", "bronze/ibge/municipios")
        )
        self.rgi_bronze_subpath = str(
            self.config.get("rgi_bronze_subpath", "bronze/ibge/rgi")
        )
        self.silver_subpath = str(
            self.config.get("silver_subpath", "silver/ibge_territorial_preprocessed")
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.geodetic_crs = str(
            self.config.get("geodetic_crs", DEFAULT_GEODETIC_CRS)
        ).lower()
        self.projected_crs = str(
            self.config.get("projected_crs", DEFAULT_PROJECTED_CRS)
        ).lower()
        self.limit_to_br101_ufs = bool(self.config.get("limit_to_br101_ufs", False))
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError("IbgeBronze2Silver requires a Spark session")

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
            "municipalities": self._extract_dataset(
                self.municipios_bronze_subpath, "extract_municipios"
            ),
            "rgis": self._extract_dataset(self.rgi_bronze_subpath, "extract_rgis"),
        }

    def _apply_common_geometry_standardization(self, dataframe: DataFrame) -> DataFrame:
        standardized = dataframe.withColumn(
            "geometry_geodetic", F.col("geometry")
        ).drop("geometry")
        standardized = standardized.filter(
            F.col("geometry_geodetic").isNotNull()
        ).filter(F.expr("NOT ST_IsEmpty(geometry_geodetic)"))
        standardized = standardized.withColumn(
            "geometry_proj",
            F.expr(
                "ST_Transform("
                f"geometry_geodetic, '{self.geodetic_crs}', '{self.projected_crs}'"
                ")"
            ),
        )
        standardized = standardized.filter(F.col("geometry_proj").isNotNull()).filter(
            F.expr("NOT ST_IsEmpty(geometry_proj)")
        )

        if self.limit_to_br101_ufs:
            standardized = standardized.filter(F.col("is_br101_uf_candidate"))

        return standardized

    def _transform_municipalities(self, dataframe: DataFrame) -> DataFrame:
        standardized = (
            dataframe.select(
                normalize_string("codigo_municipio").alias("codigo_municipio"),
                normalize_string("nome_municipio").alias("nome_municipio"),
                F.upper(normalize_string("sg_uf")).alias("sg_uf"),
                normalize_string("cd_uf").alias("cd_uf"),
                normalize_string("nm_uf").alias("nm_uf"),
                normalize_string("cd_rgi").alias("cd_rgi"),
                normalize_string("nm_rgi").alias("nm_rgi"),
                normalize_string("cd_rgint").alias("cd_rgint"),
                normalize_string("nm_rgint").alias("nm_rgint"),
                normalize_string("cd_regia").alias("cd_regia"),
                normalize_string("nm_regia").alias("nm_regia"),
                F.upper(normalize_string("sigla_rg")).alias("sigla_rg"),
                normalize_string("cd_concu").alias("cd_concu"),
                normalize_string("nm_concu").alias("nm_concu"),
                normalize_numeric("area_km2").alias("area_km2_ibge"),
                normalize_string("tipo_territorial").alias("tipo_territorial"),
                F.col("data_ingestion"),
                F.col("geometry"),
            )
            .withColumn("is_br101_uf_candidate", F.col("sg_uf").isin(*BR101_UFS))
            .withColumn("territorial_level", F.lit("municipio"))
        )

        standardized = self._apply_common_geometry_standardization(standardized)

        ordered_columns = [
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
            "polygon_area_km2",
            "territorial_level",
            "tipo_territorial",
            "is_br101_uf_candidate",
            "geometry_type",
            "geometry_crs",
            "geometry",
            "data_ingestion",
        ]

        return (
            standardized.select(
                "*",
                (F.expr("ST_Area(geometry_proj)") / F.lit(1_000_000.0)).alias(
                    "polygon_area_km2"
                ),
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
            .drop("geometry_geodetic", "geometry_proj")
            .select(*ordered_columns)
        )

    def _transform_rgis(self, dataframe: DataFrame) -> DataFrame:
        standardized = (
            dataframe.select(
                normalize_string("codigo_rgi").alias("codigo_rgi"),
                normalize_string("nome_rgi").alias("nome_rgi"),
                F.upper(normalize_string("sg_uf")).alias("sg_uf"),
                normalize_string("cd_uf").alias("cd_uf"),
                normalize_string("nm_uf").alias("nm_uf"),
                normalize_string("cd_rgint").alias("cd_rgint"),
                normalize_string("nm_rgint").alias("nm_rgint"),
                normalize_string("cd_regia").alias("cd_regia"),
                normalize_string("nm_regia").alias("nm_regia"),
                F.upper(normalize_string("sigla_rg")).alias("sigla_rg"),
                normalize_numeric("area_km2").alias("area_km2_ibge"),
                normalize_string("tipo_territorial").alias("tipo_territorial"),
                F.col("data_ingestion"),
                F.col("geometry"),
            )
            .withColumn("is_br101_uf_candidate", F.col("sg_uf").isin(*BR101_UFS))
            .withColumn("territorial_level", F.lit("rgi"))
        )

        standardized = self._apply_common_geometry_standardization(standardized)

        ordered_columns = [
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
            "polygon_area_km2",
            "territorial_level",
            "tipo_territorial",
            "is_br101_uf_candidate",
            "geometry_type",
            "geometry_crs",
            "geometry",
            "data_ingestion",
        ]

        return (
            standardized.select(
                "*",
                (F.expr("ST_Area(geometry_proj)") / F.lit(1_000_000.0)).alias(
                    "polygon_area_km2"
                ),
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
            .drop("geometry_geodetic", "geometry_proj")
            .select(*ordered_columns)
        )

    def transform(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
        return {
            "municipalities_preprocessed": self._transform_municipalities(
                data["municipalities"]
            ),
            "rgis_preprocessed": self._transform_rgis(data["rgis"]),
        }

    def load(self, data: dict[str, DataFrame]) -> str:
        staging_output_dir = self._staging_dir / "silver_ibge_territorial_preprocessed"

        for artifact_name, dataframe in data.items():
            artifact_staging_dir = staging_output_dir / artifact_name
            dataframe.write.mode(self.write_mode).partitionBy("sg_uf").parquet(
                str(artifact_staging_dir)
            )

        destination = self.datalake.persist_directory(
            staging_output_dir, self.silver_subpath
        )
        self.logger.info("Saved IBGE silver preprocessing datasets to %s", destination)
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    spark = build_spark_session("IBGE Bronze to Silver", include_sedona=True)
    job = IbgeBronze2Silver(spark=spark)
    job.run()
    spark.stop()
