from __future__ import annotations

#import io
import re
import tempfile
#import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import geopandas as gpd
#import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.etl.base_job import BaseETLJob
from src.etl.datalake import DatalakeAdapter

# from here:

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GEODETIC_CRS = "EPSG:4674"
PROJECTED_CRS = "EPSG:5880"
CORRIDOR_BUFFER_METERS = 500
BR101_UFS = {"PB", "BA", "PR", "AL", "PE", "ES", "RN", "RS", "SC", "SE", "SP", "RJ"}
#excluding SP and RS

def canonicalize_road_code(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    stripped = extracted.str.lstrip("0")
    return stripped.replace("", pd.NA)


def load_filtered_dnit_snapshot(
    snapshot_year: int, path: Path, road_code: str = "101") -> gpd.GeoDataFrame:

    dnit = gpd.read_file(path, columns=["vl_br", "sg_uf", "versao_snv", "geometry"])
    dnit["br_canonical"] = canonicalize_road_code(dnit["vl_br"])
    dnit = dnit.loc[dnit["br_canonical"].eq(road_code)].copy()
    dnit["snapshot_year"] = snapshot_year
    return dnit

def keep_line_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    line_types = {"LineString", "MultiLineString"}
    filtered = gdf.loc[gdf.geometry.geom_type.isin(line_types)].copy()
    return filtered.loc[filtered.geometry.length > 0].copy()


class DnitSrc2Bronze(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None) -> None:
        super().__init__(spark=spark, config=config, job_name="dnit_src2bronze")
        self.datalake = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(self.config.get("bronze_subpath", "bronze/dnit_road_network"))
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self._temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging_dir = Path(self._temp_dir.name)
        self.dnit_files = {
            2017: self.datalake.source_dir / "201703A.zip",
            2021: self.datalake.source_dir / "202107A.zip",
            2026: self.datalake.source_dir / "202601A.zip",
        }
        self.municipalities_path = self.datalake.source_dir / "BR_Municipios_2024.zip"
        self.rgi_path = self.datalake.source_dir / "BR_RG_Imediatas_2024.zip"

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    ##### EXTRACT #####
    def extract(self) -> dict[str, Any]:
        dnit_paths = {
            year: path for year, path in self.dnit_files.items() if path.exists()
        }
        return {
            "dnit": dnit_paths,
            "municipalities": self.municipalities_path,
            "rgi": self.rgi_path,
        }
    
    ##### TRANSFORM #####
    def transform(self, data: dict[str, Any]) -> DataFrame:
        # Step 1. Building the BR-101 Corridor with GeoPandas
        gdfs = [
            load_filtered_dnit_snapshot(year, path, road_code="101")
            for year, path in data["dnit"].items()
        ]
        br101_centerlines_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

        # Step 2. Merge the centerlines and create the buffered corridor.
        centerlines_proj = br101_centerlines_gdf.to_crs(PROJECTED_CRS)
        road_union_proj = centerlines_proj.unary_union
        corridor_union_proj = road_union_proj.buffer(CORRIDOR_BUFFER_METERS)

        # Step 3. Load and filter municipalities that intersect the buffered corridor.
        gdf_mun = gpd.read_file(data["municipalities"]).to_crs(PROJECTED_CRS)
        retained_mun_proj = gdf_mun.loc[
            gdf_mun["SIGLA_UF"].isin(BR101_UFS)
            # Usa o corredor com buffer para a interseção
            & gdf_mun.geometry.intersects(corridor_union_proj)
        ].copy()

        # Step 4. Create highway sections by municipality
        mun_sections_proj = retained_mun_proj.copy()
        mun_sections_proj["geometry"] = mun_sections_proj.geometry.intersection(road_union_proj)
        mun_sections_proj = keep_line_geometries(mun_sections_proj.explode(index_parts=False))
        
        # Step 5. Converting dataframe to Spark
        # The geometry is converted to WKT (Well-Known Text) to be stored as a string.
        mun_sections_proj["geometry_wkt"] = mun_sections_proj.geometry.to_wkt()
        spark_df = self.spark.createDataFrame(
            mun_sections_proj.drop(columns=["geometry"])
        )

        return spark_df.withColumnRenamed("geometry_wkt", "geometry")

    ##### LOAD #####
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
        self.logger.info(
            f"Salvo dataset bronze do DNIT particionado em: {destination}"
        )
        return destination

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DNIT Src to Bronze").getOrCreate()
    job = DnitSrc2Bronze(spark=spark)
    job.run()
    spark.stop()

