from __future__ import annotations

from functools import reduce
from importlib.resources import path
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from sedona.spark import SedonaContext
from sedona.core.formatMapper.shapefileParser import ShapefileReader as SedonaShapefileReader
from sedona.utils.adapter import Adapter

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

DNIT_URLS: dict[int, str] = {
    2017: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=201703A.zip",
    2021: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=202107A.zip",
    2026: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=202601A.zip",
}


def canonicalize_road_code(column_name: str) -> Column:
    extracted: Column = F.regexp_extract(F.col(column_name).cast("string"), r"(\d+)", 1)
    stripped: Column = F.regexp_replace(extracted, r"^0+", "")
    return F.when(stripped == "", F.lit(None).cast("string")).otherwise(stripped)


def resolve_column_name(dataframe: DataFrame, expected_name: str) -> str:
    normalized_map: dict[str, str] = {
        column_name.casefold(): column_name for column_name in dataframe.columns
    }
    try:
        return normalized_map[expected_name.casefold()]
    except KeyError as exc:
        available_columns: str = ", ".join(dataframe.columns)
        raise ValueError(
            f"Column '{expected_name}' was not found in DNIT shapefile. "
            f"Available columns: {available_columns}"
        ) from exc


def load_filtered_dnit_snapshot(
    snapshot_year: int,
    path: Path,
    spark: SparkSession,
    road_code: str = "101",
) -> DataFrame:
    # passar o .parent para garantir que o Sedona veja todos os arquivos (.shp, .dbf, .shx) na pasta
    raw_spatial_rdd = SedonaShapefileReader.readToGeometryRDD(spark.sparkContext, str(path.parent))
    
    #usa adaptar para converter o RDD em DataFrame
    dnit = Adapter.toDf(raw_spatial_rdd, spark)

    vl_br_column: str = resolve_column_name(dnit, "vl_br")
    sg_uf_column: str = resolve_column_name(dnit, "sg_uf")
    versao_snv_column: str = resolve_column_name(dnit, "versao_snv")
    geometry_column: str = resolve_column_name(dnit, "geometry")

    selected = dnit.select(
        F.col(vl_br_column).alias("vl_br"),
        F.col(sg_uf_column).alias("sg_uf"),
        F.col(versao_snv_column).alias("versao_snv"),
        F.col(geometry_column).alias("geometry"),
    )
    with_br_canonical: DataFrame = selected.withColumn(
        "br_canonical", canonicalize_road_code("vl_br")
    )

    return with_br_canonical.filter(
        F.col("br_canonical") == F.lit(road_code)
    ).withColumn("snapshot_year", F.lit(snapshot_year).cast("int"))


class DnitSrc2Bronze(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="dnit_src2bronze")
        self.datalake = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(
            self.config.get("bronze_subpath", "bronze/dnit_road_network")
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.request_timeout = int(self.config.get("request_timeout", 120))
        self._temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
            dir="/tmp"
        )
        self._staging_dir = Path(self._temp_dir.name)

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def _download_dnit_zip(self, *, year: int, source_url: str) -> Path:
        response = requests.get(source_url, timeout=self.request_timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zipped_payload:
            shp_members: list[str] = [
                member
                for member in zipped_payload.namelist()
                if member.lower().endswith(".shp")
            ]
            if not shp_members:
                raise FileNotFoundError(
                    f"No SHP file found inside DNIT archive for {year}"
                )

            extract_dir = self._staging_dir / f"{year}_dnit_shp"
            zipped_payload.extractall(extract_dir)
            self.logger.info("Downloaded and extracted DNIT archive for %s", year)
            return extract_dir / shp_members[0]

    def extract(self) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []

        for year, url in DNIT_URLS.items():
            try:
                shp_path = self._download_dnit_zip(year=year, source_url=url)
                datasets.append(
                    {
                        "year": year,
                        "shp_path": shp_path,
                    }
                )
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    "Failed to download DNIT data for year %s: %s", year, e
                )

        if not datasets:
            raise RuntimeError("Failed to download any DNIT data.")

        return datasets

    def transform(self, data: list[dict[str, Any]]) -> DataFrame:
        transformed: list[DataFrame] = [
            load_filtered_dnit_snapshot(
                item["year"], item["shp_path"], self.spark, road_code="101"
            )
            for item in data
        ]

        if not transformed:
            raise ValueError("No DNIT datasets were transformed")

        combined = reduce(DataFrame.unionByName, transformed)

        return combined.select(
            "vl_br",
            "sg_uf",
            "versao_snv",
            "br_canonical",
            "snapshot_year",
            F.expr("ST_AsText(geometry)").alias("geometry"),
        )

    def load(self, data: DataFrame) -> str:
        staging_output_dir = self._staging_dir / "bronze_dnit"
        (
            data.write.mode(self.write_mode)
            .partitionBy("snapshot_year")
            .parquet(str(staging_output_dir))
        )
        destination = self.datalake.persist_directory(
            staging_output_dir, self.bronze_subpath
        )
        self.logger.info("Saved partitioned DNIT bronze dataset to %s", destination)
        return destination


if __name__ == "__main__":

    base_spark: SparkSession = build_spark_session("DNIT Src to Bronze", include_sedona=True)
    
    #Registrar o Sedona
    #injeta as funções espaciais e os leitores (Shapefile) de forma segura
    spark = SedonaContext.create(base_spark)

    job = DnitSrc2Bronze(spark=spark)
    job.run()
    
    spark.stop()
