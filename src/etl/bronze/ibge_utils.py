"""Shared utilities for IBGE ETL jobs."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from sedona.core.formatMapper.shapefileParser import (
    ShapefileReader as SedonaShapefileReader,
)
from sedona.utils.adapter import Adapter


def resolve_column_name(dataframe: DataFrame, expected_name: str) -> str:
    """Resolve column name case-insensitively from dataframe.

    Args:
        dataframe: Spark DataFrame to search
        expected_name: Column name to find (case-insensitive)

    Returns:
        Actual column name from dataframe

    Raises:
        ValueError: If column not found
    """
    normalized_map: dict[str, str] = {
        column_name.casefold(): column_name for column_name in dataframe.columns
    }
    try:
        return normalized_map[expected_name.casefold()]
    except KeyError as exc:
        available_columns: str = ", ".join(dataframe.columns)
        raise ValueError(
            f"Column '{expected_name}' was not found in IBGE shapefile. "
            f"Available columns: {available_columns}"
        ) from exc


def download_and_extract_ibge_zip(
    *,
    data_type: str,
    source_url: str,
    staging_dir: Path,
    request_timeout: int = 120,
) -> Path:
    """Download IBGE ZIP archive and extract shapefile.

    Args:
        data_type: Type of data (municipios, rgi)
        source_url: URL to download from
        staging_dir: Temporary directory for extraction
        request_timeout: HTTP request timeout in seconds

    Returns:
        Path to extracted .shp file

    Raises:
        FileNotFoundError: If no .shp file found in archive
        requests.exceptions.RequestException: If download fails
    """
    response = requests.get(source_url, timeout=request_timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zipped_payload:
        shp_members: list[str] = [
            member
            for member in zipped_payload.namelist()
            if member.lower().endswith(".shp")
        ]
        if not shp_members:
            raise FileNotFoundError(
                f"No SHP file found inside IBGE archive for {data_type}"
            )

        extract_dir = staging_dir / f"ibge_{data_type}_shp"
        zipped_payload.extractall(extract_dir)
        return extract_dir / shp_members[0]


def fix_shapefile_string_encoding(column_name: str) -> Column:
    """Re-decode a string column corrupted by Sedona reading UTF-8 DBF bytes as ISO-8859-1.

    Sedona's ShapefileReader decodes .dbf string fields as ISO-8859-1 by default,
    but IBGE shapefiles encode text as UTF-8.  The round-trip
    encode(iso-8859-1) -> decode(utf-8) recovers the original characters.
    """
    return F.decode(F.encode(F.col(column_name), "iso-8859-1"), "utf-8")


def load_shapefile_as_dataframe(
    shapefile_path: Path,
    spark: SparkSession,
) -> DataFrame:
    """Load shapefile using Sedona and return as DataFrame.

    Uses parent directory so Sedona can read the full shapefile sidecar set
    (.shp, .shx, .dbf, .prj).

    Args:
        shapefile_path: Path to .shp file
        spark: SparkSession with Sedona

    Returns:
        Spark DataFrame with geometry column
    """
    raw_spatial_rdd = SedonaShapefileReader.readToGeometryRDD(
        spark.sparkContext, str(shapefile_path.parent)
    )
    return Adapter.toDf(raw_spatial_rdd, spark)
