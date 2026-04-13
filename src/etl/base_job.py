from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext

Builder = SparkSession.Builder


DEFAULT_SPARK_PACKAGES: tuple[str, ...] = (
    "org.apache.hadoop:hadoop-aws:3.4.2",
    "com.amazonaws:aws-java-sdk-bundle:1.12.367",
)
DEFAULT_SEDONA_PACKAGES: tuple[str, ...] = (
    "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.8.1",
    "org.datasyslab:geotools-wrapper:1.8.1-33.1",
)


def _dedupe_packages(packages: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for package in packages:
        if package in seen:
            continue
        seen.add(package)
        deduped.append(package)
    return deduped


def build_spark_session(
    app_name: str,
    *,
    include_sedona: bool = False,
    extra_packages: tuple[str, ...] = (),
) -> SparkSession:
    ivy_cache_dir = "/tmp/.ivy2"
    if not os.environ.get("SPARK_LOCAL_HOSTNAME"):
        os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    if not os.environ.get("SPARK_LOCAL_IP"):
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ.setdefault("IVY_HOME", ivy_cache_dir)

    packages: list[str] = list(DEFAULT_SPARK_PACKAGES)
    builder: Builder = SparkSession.builder.appName(app_name)

    if include_sedona:
        builder: Builder = SedonaContext.builder().appName(app_name)
        packages.extend(DEFAULT_SEDONA_PACKAGES)

    packages.extend(extra_packages)
    package_list: list[str] = _dedupe_packages(packages)

    if package_list:
        builder: Builder = builder.config("spark.jars.packages", ",".join(package_list))

    builder = builder.config("spark.jars.ivy", ivy_cache_dir)
    builder = builder.config("spark.driver.host", os.environ["SPARK_LOCAL_IP"])
    builder = builder.config("spark.driver.bindAddress", os.environ["SPARK_LOCAL_IP"])

    return builder.getOrCreate()


class BaseETLJob(ABC):
    """Reusable ETL base class with a small, explicit lifecycle contract."""

    def __init__(
        self,
        *,
        spark: SparkSession,
        config: Mapping[str, Any] | None = None,
        job_name: str | None = None,
    ) -> None:
        self.spark: SparkSession = spark
        self.config = dict(config or {})
        self.job_name = job_name or self.__class__.__name__
        self.logger = logging.getLogger(self.job_name)
        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration shared by all jobs."""
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")

    @abstractmethod
    def extract(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Release temporary job resources."""

    def run(self) -> Any:
        self.logger.info("Starting job %s", self.job_name)
        try:
            data = self.extract()
            transformed_data = self.transform(data)
            result = self.load(transformed_data)
            self.logger.info("Finished job %s", self.job_name)
            return result
        except Exception:
            self.logger.exception("Job %s failed", self.job_name)
            raise
        finally:
            self.cleanup()
