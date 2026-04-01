from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from pyspark.sql import DataFrame, SparkSession

from src.etl.base_job import BaseETLJob
from src.etl.datalake import DatalakeAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DNIT_URLS = {
    2017: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=201703A.zip",
    2021: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=202107A.zip",
    2026: "https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV%20Bases%20Geom%C3%A9tricas%20(2013-Atual)%20(SHP)&files=202601A.zip",
}


def canonicalize_road_code(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    stripped = extracted.str.lstrip("0")
    return stripped.replace("", pd.NA)


def load_filtered_dnit_snapshot(
    snapshot_year: int, path: Path, road_code: str = "101"
) -> gpd.GeoDataFrame:
    dnit = gpd.read_file(path, columns=["vl_br", "sg_uf", "versao_snv", "geometry"])
    dnit["br_canonical"] = canonicalize_road_code(dnit["vl_br"])
    dnit = dnit.loc[dnit["br_canonical"].eq(road_code)].copy()
    dnit["snapshot_year"] = snapshot_year
    return dnit


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
        self._temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging_dir = Path(self._temp_dir.name)

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def _download_dnit_zip(self, *, year: int, source_url: str) -> Path:
        response = requests.get(source_url, timeout=self.request_timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zipped_payload:
            shp_members = [
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
        gdfs = [
            load_filtered_dnit_snapshot(item["year"], item["shp_path"], road_code="101")
            for item in data
        ]

        if not gdfs:
            raise ValueError("No DNIT datasets were transformed")

        br101_centerlines_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs
        )

        # Convert geometry to WKT to be able to create a Spark DataFrame
        br101_centerlines_gdf["geometry_wkt"] = br101_centerlines_gdf.geometry.to_wkt()
        spark_df = self.spark.createDataFrame(
            br101_centerlines_gdf.drop(columns=["geometry"])
        )

        return spark_df.withColumnRenamed("geometry_wkt", "geometry")

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
    spark = SparkSession.builder.appName("DNIT Src to Bronze").getOrCreate()
    job = DnitSrc2Bronze(spark=spark)
    job.run()
    spark.stop()

