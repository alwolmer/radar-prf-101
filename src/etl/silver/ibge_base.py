from __future__ import annotations

import tempfile
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.etl.base_job import BaseETLJob
from src.etl.datalake import DatalakeAdapter
from src.etl.transform.geospatial import BR101_UFS, GEODETIC_CRS, PROJECTED_CRS


class IbgeBaseBronze2Silver(BaseETLJob):
    bronze_subpath: str
    silver_subpath: str
    select_columns: list[str]

    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name=self._job_name())
        self.datalake = DatalakeAdapter.from_env(
            project_root=Path(__file__).resolve().parents[3],
            config=self.config.get("datalake"),
        )
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self._temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging_dir = Path(self._temp_dir.name)

    @abstractmethod
    def _job_name(self) -> str: ...

    def validate_config(self) -> None:
        super().validate_config()
        if self.spark is None:
            raise ValueError(f"{self.__class__.__name__} requires a Spark session")

    def extract(self) -> DataFrame:
        zip_path = self.datalake.stage_file(
            self.bronze_subpath,
            self._staging_dir / "extract",
        )
        unzip_dir = self._staging_dir / "shapefile"
        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unzip_dir)
        return self.spark.read.format("shapefile").load(str(unzip_dir))

    def transform(self, data: DataFrame) -> DataFrame:
        uf_expr = F.upper(F.trim(F.col("SIGLA_UF")))
        filtered = data.filter(uf_expr.isin([*BR101_UFS]))
        geom_col = _find_geometry_col(filtered)
        area_expr = (
            F.expr(
                f"ST_Area(ST_Transform({geom_col}, '{GEODETIC_CRS}', '{PROJECTED_CRS}'))"
            )
            / F.lit(1_000_000)
        ).alias("polygon_area_km2")
        kept = [F.col(c) for c in filtered.columns if c in self.select_columns]
        result = filtered.select(*kept, area_expr)
        result = result.select(
            *[F.col(c) for c in self.select_columns if c != "geometry"],
            F.col("polygon_area_km2"),
            F.col(_find_geometry_col(filtered)).alias("geometry"),
        )
        return result

    def load(self, data: DataFrame) -> str:
        staging_output_dir = self._staging_dir / "silver_output"
        data.write.format("geoparquet").mode(self.write_mode).save(
            str(staging_output_dir)
        )
        destination = self.datalake.persist_directory(
            staging_output_dir, self.silver_subpath
        )
        self.logger.info("Saved IBGE silver dataset to %s", destination)
        return destination

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


def _find_geometry_col(df: DataFrame) -> str:
    for field in df.schema.fields:
        if hasattr(field, "dataType") and field.dataType.typeName() == "geometry":
            return field.name
    candidates = {"geometry", "geom", "the_geom", "shape"}
    for c in df.columns:
        if c.lower() in candidates:
            return c
    raise ValueError(f"No geometry column found in DataFrame. Columns: {df.columns}")
