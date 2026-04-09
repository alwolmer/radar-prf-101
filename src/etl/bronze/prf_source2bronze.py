from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.etl.base_job import BaseETLJob, build_spark_session
from src.etl.datalake import DatalakeAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PAGE_URL = (
    "https://www.gov.br/prf/pt-br/acesso-a-informacao/"
    "dados-abertos/dados-abertos-da-prf"
)
OCCURRENCE_REFERENCE_REGEX = re.compile(
    r"(20(?:1[7-9]|2[0-6])).*?(Agrupados por ocorrência)",
    re.IGNORECASE,
)
GOOGLE_DRIVE_FILE_ID_REGEX = re.compile(r"/d/([a-zA-Z0-9_-]+)")


class PrfSrc2Bronze(BaseETLJob):
    def __init__(
        self, spark: SparkSession, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(spark=spark, config=config, job_name="prf_src2bronze")
        self.source_page_url = self.config.get(
            "source_page_url", DEFAULT_SOURCE_PAGE_URL
        )
        self.datalake = DatalakeAdapter.from_env(
            project_root=PROJECT_ROOT,
            config=self.config.get("datalake"),
        )
        self.bronze_subpath = str(
            self.config.get("bronze_subpath", "bronze/prf_accidents")
        )
        self.request_timeout = int(self.config.get("request_timeout", 120))
        self.write_mode = str(self.config.get("write_mode", "overwrite"))
        self.csv_read_options = {
            "header": True,
            "sep": ";",
            "encoding": "iso-8859-1",
        }
        self._temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self._staging_dir = Path(self._temp_dir.name)

    @staticmethod
    def _extract_cell_text(value: object) -> str:
        if isinstance(value, tuple):
            return str(value[0] or "").strip()
        return str(value or "").strip()

    @staticmethod
    def _extract_cell_link(value: object) -> str | None:
        if isinstance(value, tuple):
            link = value[1]
            return str(link).strip() if link else None
        return None

    @staticmethod
    def _normalize_occurrence_reference(reference: str) -> str | None:
        match = OCCURRENCE_REFERENCE_REGEX.search(reference)
        if not match:
            return None
        return f"{match.group(1)} - Agrupados por ocorrência"

    def _download_occurrence_csv(self, *, year: str, source_url: str) -> Path:
        file_id_match = GOOGLE_DRIVE_FILE_ID_REGEX.search(source_url)
        if file_id_match is None:
            raise ValueError(
                f"Could not extract Google Drive file id from URL: {source_url}"
            )

        file_id = file_id_match.group(1)
        download_url = (
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t"
        )
        response = requests.get(download_url, timeout=self.request_timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zipped_payload:
            csv_members = [
                member
                for member in zipped_payload.namelist()
                if member.lower().endswith(".csv")
            ]
            if not csv_members:
                raise FileNotFoundError(
                    f"No CSV file found inside PRF archive for {year}"
                )

            extracted_csv_path = (
                self._staging_dir / f"{year}_Agrupados por ocorrência.csv"
            )
            with zipped_payload.open(csv_members[0]) as source_file:
                extracted_csv_path.write_bytes(source_file.read())

        self.logger.info("Downloaded PRF occurrence archive for %s", year)
        return extracted_csv_path

    def extract_urls(self) -> list[dict[str, str]]:
        raw_tables = pd.read_html(self.source_page_url, extract_links="all")
        urls_by_reference: dict[str, str] = {}

        for table in raw_tables:
            if table.empty or len(table.columns) < 2:
                continue

            ref_col, link_col = table.columns[:2]
            normalized_refs = table[ref_col].apply(self._extract_cell_text)
            raw_links = table[link_col].apply(self._extract_cell_link)

            for reference, link in zip(normalized_refs, raw_links, strict=False):
                normalized_reference = self._normalize_occurrence_reference(reference)
                if normalized_reference and link:
                    urls_by_reference[normalized_reference] = link

        extracted = [
            {
                "reference": reference,
                "year": reference.split(" - ", maxsplit=1)[0],
                "link": link,
            }
            for reference, link in urls_by_reference.items()
        ]
        extracted.sort(key=lambda item: item["year"])

        if not extracted:
            raise ValueError("No PRF occurrence URLs were found on the source page")

        self.logger.info("Extracted %s PRF occurrence URLs", len(extracted))
        return extracted

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def extract(self) -> list[dict[str, str]]:
        datasets: list[dict[str, str]] = []

        for url_info in self.extract_urls():
            extracted_csv_path = self._download_occurrence_csv(
                year=url_info["year"],
                source_url=url_info["link"],
            )
            datasets.append(
                {
                    "reference": url_info["reference"],
                    "year": url_info["year"],
                    "csv_path": str(extracted_csv_path),
                }
            )

        return datasets

    def transform(self, data: list[dict[str, str]]) -> DataFrame:
        transformed: list[DataFrame] = []

        for dataset in data:
            dataframe = self.spark.read.options(**self.csv_read_options).csv(
                dataset["csv_path"]
            )
            transformed.append(
                dataframe.withColumn(
                    "source_year_file",
                    F.lit(int(dataset["year"])).cast("int"),
                )
            )

        if not transformed:
            raise ValueError("No PRF occurrence datasets were transformed")

        combined = transformed[0]
        for dataframe in transformed[1:]:
            combined = combined.unionByName(dataframe)

        return combined

    def load(self, data: DataFrame) -> str:
        staging_output_dir = self._staging_dir / "bronze_prf"
        data.write.mode(self.write_mode).partitionBy("br", "source_year_file").parquet(
            str(staging_output_dir)
        )
        destination = self.datalake.persist_directory(
            staging_output_dir, self.bronze_subpath
        )
        self.logger.info("Saved partitioned PRF bronze dataset to %s", destination)
        return destination


if __name__ == "__main__":
    spark = build_spark_session("PRF Src to Bronze")
    job = PrfSrc2Bronze(spark=spark)
    job.run()
    spark.stop()
