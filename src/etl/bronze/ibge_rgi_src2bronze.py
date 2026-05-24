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

IBGE_RGI_URL: str = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
    "malhas_territoriais/malhas_municipais/municipio_2024/"
    "Brasil/BR_RG_Imediatas_2024.zip"
)


class IbgeRgiSrc2Bronze(BaseETLJob):
    """ETL job to ingest IBGE Regiões Geográficas Imediatas into bronze layer.

    This job handles RGI (Immediate Geographic Regions) polygons from IBGE 2024.
    Preserves all IBGE columns for complete context and downstream flexibility.

    Output schema (12 columns):
        Core identifiers:
        - codigo_rgi: str (CD_RGI)
        - nome_rgi: str (NM_RGI)
        - sg_uf: str (SIGLA_UF) - state code

        Geographic relationships:
        - cd_rgint, nm_rgint: RGI Interior
        - cd_uf, nm_uf: State full
        - cd_regia, nm_regia, sigla_rg: Geographic Region

        Geospatial & metadata:
        - area_km2: Area in square kilometers
        - geometry: Sedona geometry
        - tipo_territorial: str (always "rgi")
        - data_ingestion: timestamp

    Partitioning: by sg_uf (state code)
    """

    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="ibge_rgi_src2bronze")
        self.datalake = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(self.config.get("bronze_subpath", "bronze/ibge/rgi"))
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
        """Download and extract RGI shapefile.

        Returns:
            Path to extracted .shp file
        """
        self.logger.info("Downloading IBGE RGI archive")
        shp_path = download_and_extract_ibge_zip(
            data_type="rgi",
            source_url=IBGE_RGI_URL,
            staging_dir=self._staging_dir,
            request_timeout=self.request_timeout,
        )
        self.logger.info("Successfully extracted RGI shapefile: %s", shp_path)
        return shp_path

    def transform(self, shp_path: Path) -> DataFrame:
        """Transform RGI shapefile into standardized format.

        Preserves all IBGE columns while renaming key identifiers to standard names.

        Args:
            shp_path: Path to RGI .shp file

        Returns:
            Standardized DataFrame with complete RGI data
        """

        rgi_df = load_shapefile_as_dataframe(shp_path, self.spark)

        # Resolve column names (case-insensitive)
        cd_rgi_column = resolve_column_name(rgi_df, "cd_rgi")
        nm_rgi_column = resolve_column_name(rgi_df, "nm_rgi")
        cd_rgint_column = resolve_column_name(rgi_df, "cd_rgint")
        nm_rgint_column = resolve_column_name(rgi_df, "nm_rgint")
        sigla_uf_column = resolve_column_name(rgi_df, "sigla_uf")
        cd_uf_column = resolve_column_name(rgi_df, "cd_uf")
        nm_uf_column = resolve_column_name(rgi_df, "nm_uf")
        cd_regia_column = resolve_column_name(rgi_df, "cd_regia")
        nm_regia_column = resolve_column_name(rgi_df, "nm_regia")
        sigla_rg_column = resolve_column_name(rgi_df, "sigla_rg")
        area_km2_column = resolve_column_name(rgi_df, "area_km2")
        geometry_column = resolve_column_name(rgi_df, "geometry")

        standardized = (
            rgi_df.select(
                # Core RGI identifiers
                F.col(cd_rgi_column).alias("codigo_rgi"),
                fix_shapefile_string_encoding(nm_rgi_column).alias("nome_rgi"),
                F.col(sigla_uf_column).alias("sg_uf"),
                F.col(cd_uf_column).alias("cd_uf"),
                fix_shapefile_string_encoding(nm_uf_column).alias("nm_uf"),
                # RGI Interior
                F.col(cd_rgint_column).alias("cd_rgint"),
                fix_shapefile_string_encoding(nm_rgint_column).alias("nm_rgint"),
                # Geographic Region
                F.col(cd_regia_column).alias("cd_regia"),
                fix_shapefile_string_encoding(nm_regia_column).alias("nm_regia"),
                F.col(sigla_rg_column).alias("sigla_rg"),
                # Geospatial
                F.col(area_km2_column).alias("area_km2"),
                F.col(geometry_column).alias("geometry"),
            )
            .withColumn("tipo_territorial", F.lit("rgi"))
            .withColumn("data_ingestion", F.current_timestamp())
        )

        self.logger.info(
            "Transformed %s RGI records with %d columns",
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
        staging_output_dir = self._staging_dir / "bronze_ibge_rgi"

        data.write.mode(self.write_mode).partitionBy("sg_uf").parquet(
            str(staging_output_dir)
        )

        destination = self.datalake.persist_directory(
            staging_output_dir, self.bronze_subpath
        )

        self.logger.info("Saved partitioned IBGE RGI bronze dataset to %s", destination)
        return destination


if __name__ == "__main__":
    spark: SparkSession = build_spark_session(
        "IBGE RGI Src to Bronze", include_sedona=True
    )
    job = IbgeRgiSrc2Bronze(spark=spark)
    job.run()
    spark.stop()
