from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.bronze.ibge_utils import (
    download_and_extract_ibge_zip,
    fix_shapefile_string_encoding,
    load_shapefile_as_dataframe,
    resolve_column_name,
)
from src.etl.datalake import DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

IBGE_MUNICIPIOS_URL: str = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/municipio_2024/"
    "Brasil/BR_Municipios_2024.zip"
)


class IbgeMunicipiosSrc2Bronze(BaseETLJob):
    """ETL job to ingest IBGE municipality boundaries into bronze layer.

    This job handles municipality polygons from IBGE 2024 territorial boundaries.
    Preserves all IBGE columns for complete context and downstream flexibility.

    Output schema (16 columns):
        Core identifiers:
        - codigo_municipio: str (CD_MUN)
        - nome_municipio: str (NM_MUN)
        - sg_uf: str (SIGLA_UF) - state code

        Geographic relationships:
        - cd_rgi, nm_rgi: Immediate Geographic Region
        - cd_rgint, nm_rgint: RGI Interior
        - cd_uf, nm_uf: State full
        - cd_regia, nm_regia, sigla_rg: Geographic Region
        - cd_concu, nm_concu: Concurrency

        Geospatial & metadata:
        - area_km2: Area in square kilometers
        - geometry: Sedona geometry
        - tipo_territorial: str (always "municipios")
        - data_ingestion: timestamp

    Partitioning: by sg_uf (state code)
    """

    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            spark=spark, config=config, job_name="ibge_municipios_src2bronze"
        )
        self.datalake = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(
            self.config.get("bronze_subpath", "bronze/ibge/municipios")
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.request_timeout = int(self.config.get("request_timeout", 120))
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def cleanup(self) -> None:
        """Clean up temporary directory."""
        self._temp_dir.cleanup()

    def extract(self) -> Path:
        """Download and extract municipality shapefile.

        Returns:
            Path to extracted .shp file
        """
        self.logger.info("Downloading IBGE municipios archive")
        shp_path = download_and_extract_ibge_zip(
            data_type="municipios",
            source_url=IBGE_MUNICIPIOS_URL,
            staging_dir=self._staging_dir,
            request_timeout=self.request_timeout,
        )
        self.logger.info("Successfully extracted municipios shapefile: %s", shp_path)
        return shp_path

    def transform(self, shp_path: Path) -> DataFrame:
        """Transform municipality shapefile into standardized format.

        Preserves all IBGE columns while renaming key identifiers to standard names.

        Args:
            shp_path: Path to municipality .shp file

        Returns:
            Standardized DataFrame with complete municipality data
        """

        municipios_df = load_shapefile_as_dataframe(shp_path, self.spark)

        # Resolve column names (case-insensitive)
        nm_mun_column = resolve_column_name(municipios_df, "nm_mun")
        cd_mun_column = resolve_column_name(municipios_df, "cd_mun")
        sigla_uf_column = resolve_column_name(municipios_df, "sigla_uf")
        cd_uf_column = resolve_column_name(municipios_df, "cd_uf")
        nm_uf_column = resolve_column_name(municipios_df, "nm_uf")
        cd_rgi_column = resolve_column_name(municipios_df, "cd_rgi")
        nm_rgi_column = resolve_column_name(municipios_df, "nm_rgi")
        cd_rgint_column = resolve_column_name(municipios_df, "cd_rgint")
        nm_rgint_column = resolve_column_name(municipios_df, "nm_rgint")
        cd_regia_column = resolve_column_name(municipios_df, "cd_regia")
        nm_regia_column = resolve_column_name(municipios_df, "nm_regia")
        sigla_rg_column = resolve_column_name(municipios_df, "sigla_rg")
        cd_concu_column = resolve_column_name(municipios_df, "cd_concu")
        nm_concu_column = resolve_column_name(municipios_df, "nm_concu")
        area_km2_column = resolve_column_name(municipios_df, "area_km2")
        geometry_column = resolve_column_name(municipios_df, "geometry")

        standardized = (
            municipios_df.select(
                # Core municipality identifiers
                F.col(cd_mun_column).alias("codigo_municipio"),
                fix_shapefile_string_encoding(nm_mun_column).alias("nome_municipio"),
                F.col(sigla_uf_column).alias("sg_uf"),
                F.col(cd_uf_column).alias("cd_uf"),
                fix_shapefile_string_encoding(nm_uf_column).alias("nm_uf"),
                # RGI (Immediate Geographic Region)
                F.col(cd_rgi_column).alias("cd_rgi"),
                fix_shapefile_string_encoding(nm_rgi_column).alias("nm_rgi"),
                F.col(cd_rgint_column).alias("cd_rgint"),
                fix_shapefile_string_encoding(nm_rgint_column).alias("nm_rgint"),
                # Geographic Region
                F.col(cd_regia_column).alias("cd_regia"),
                fix_shapefile_string_encoding(nm_regia_column).alias("nm_regia"),
                F.col(sigla_rg_column).alias("sigla_rg"),
                # Concurrency
                F.col(cd_concu_column).alias("cd_concu"),
                fix_shapefile_string_encoding(nm_concu_column).alias("nm_concu"),
                # Geospatial
                F.col(area_km2_column).alias("area_km2"),
                F.col(geometry_column).alias("geometry"),
            )
            .withColumn("tipo_territorial", F.lit("municipios"))
            .withColumn("data_ingestion", F.current_timestamp())
        )

        self.logger.info(
            "Transformed %s municipality records with %d columns",
            standardized.count(),
            len(standardized.columns),
        )
        return standardized

    def load(self, data: DataFrame) -> str:
        """Load transformed data into bronze layer partitioned by state.

        Args:
            data: Transformed DataFrame

        Returns:
            Path to saved data in datalake
        """
        staging_output_dir = self._staging_dir / "bronze_ibge_municipios"

        data.write.mode(self.write_mode).partitionBy("sg_uf").parquet(
            str(staging_output_dir)
        )

        destination = self.datalake.persist_directory(
            staging_output_dir, self.bronze_subpath
        )

        self.logger.info(
            "Saved partitioned IBGE municipios bronze dataset to %s", destination
        )
        return destination


if __name__ == "__main__":
    spark: SparkSession = build_spark_session(
        "IBGE Municipios Src to Bronze", include_sedona=True
    )
    job = IbgeMunicipiosSrc2Bronze(spark=spark)
    job.run()
    spark.stop()
