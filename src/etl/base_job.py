from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext

DEFAULT_SPARK_VERSION = "3.5"
DEFAULT_SCALA_VERSION = "2.12"
DEFAULT_SEDONA_VERSION = "1.5.1"
DEFAULT_GEOTOOLS_WRAPPER_VERSION = "1.5.1-28.2"
DEFAULT_HADOOP_AWS_VERSION = "3.4.2"
DEFAULT_AWS_JAVA_SDK_BUNDLE_VERSION = "1.12.367"
DEFAULT_SPARK_JARS_DIR = "/opt/spark-jars"
DEFAULT_SPARK_REPOSITORIES = (
    "https://artifacts.unidata.ucar.edu/repository/unidata-all/",
)
DEFAULT_SEDONA_SQL_EXTENSIONS = "org.apache.sedona.sql.SedonaSqlExtensions"
DEFAULT_SPARK_LOG_LEVEL = "WARN"
DEFAULT_ETL_LOG_LEVEL = "INFO"
DEFAULT_SPARK_DRIVER_MEMORY = "2g"
DEFAULT_SPARK_DEFAULT_PARALLELISM = "4"
DEFAULT_SPARK_SQL_SHUFFLE_PARTITIONS = "4"
DEFAULT_SPARK_SQL_DEBUG_MAX_TO_STRING_FIELDS = "200"
DEFAULT_PARQUET_BLOCK_SIZE = str(32 * 1024 * 1024)
DEFAULT_G1_YOUNG_GENERATION_COLLECTORS = "G1 Young Generation"
DEFAULT_G1_OLD_GENERATION_COLLECTORS = "G1 Old Generation,G1 Concurrent GC"
JVM_QUIET_LOGGERS: tuple[str, ...] = (
    "org.apache.spark.sql.types.UDTRegistration",
    "org.apache.spark.sql.catalyst.analysis.SimpleFunctionRegistry",
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


def _configure_logging() -> None:
    level_name = os.environ.get("ETL_LOG_LEVEL", DEFAULT_ETL_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    root_logger.setLevel(level)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _sedona_packages() -> tuple[str, ...]:
    spark_version = os.environ.get("SPARK_VERSION", DEFAULT_SPARK_VERSION)
    scala_version = os.environ.get("SCALA_VERSION", DEFAULT_SCALA_VERSION)
    sedona_version = os.environ.get("SEDONA_VERSION", DEFAULT_SEDONA_VERSION)
    geotools_wrapper_version = os.environ.get(
        "GEOTOOLS_WRAPPER_VERSION", DEFAULT_GEOTOOLS_WRAPPER_VERSION
    )
    return (
        f"org.apache.sedona:sedona-spark-shaded-{spark_version}_{scala_version}:{sedona_version}",
        f"org.datasyslab:geotools-wrapper:{geotools_wrapper_version}",
    )


def _spark_packages() -> tuple[str, ...]:
    hadoop_aws_version = os.environ.get(
        "HADOOP_AWS_VERSION", DEFAULT_HADOOP_AWS_VERSION
    )
    aws_java_sdk_bundle_version = os.environ.get(
        "AWS_JAVA_SDK_BUNDLE_VERSION", DEFAULT_AWS_JAVA_SDK_BUNDLE_VERSION
    )
    return (
        f"org.apache.hadoop:hadoop-aws:{hadoop_aws_version}",
        f"com.amazonaws:aws-java-sdk-bundle:{aws_java_sdk_bundle_version}",
    )


def _preinstalled_jars() -> tuple[str, ...]:
    jars_dir = Path(os.environ.get("SPARK_JARS_DIR", DEFAULT_SPARK_JARS_DIR))
    if not jars_dir.exists():
        return ()
    return tuple(str(path) for path in sorted(jars_dir.glob("*.jar")))


def _spark_runtime_configs() -> dict[str, str]:
    return {
        "spark.driver.memory": os.environ.get(
            "SPARK_DRIVER_MEMORY", DEFAULT_SPARK_DRIVER_MEMORY
        ),
        "spark.default.parallelism": os.environ.get(
            "SPARK_DEFAULT_PARALLELISM", DEFAULT_SPARK_DEFAULT_PARALLELISM
        ),
        "spark.sql.shuffle.partitions": os.environ.get(
            "SPARK_SQL_SHUFFLE_PARTITIONS", DEFAULT_SPARK_SQL_SHUFFLE_PARTITIONS
        ),
        "spark.sql.debug.maxToStringFields": os.environ.get(
            "SPARK_SQL_DEBUG_MAX_TO_STRING_FIELDS",
            DEFAULT_SPARK_SQL_DEBUG_MAX_TO_STRING_FIELDS,
        ),
        "spark.hadoop.parquet.block.size": os.environ.get(
            "PARQUET_BLOCK_SIZE", DEFAULT_PARQUET_BLOCK_SIZE
        ),
        "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors": os.environ.get(
            "SPARK_EVENTLOG_GC_METRICS_YOUNG_GENERATION_GARBAGE_COLLECTORS",
            DEFAULT_G1_YOUNG_GENERATION_COLLECTORS,
        ),
        "spark.eventLog.gcMetrics.oldGenerationGarbageCollectors": os.environ.get(
            "SPARK_EVENTLOG_GC_METRICS_OLD_GENERATION_GARBAGE_COLLECTORS",
            DEFAULT_G1_OLD_GENERATION_COLLECTORS,
        ),
    }


def _log_phase(logger: logging.Logger, phase_name: str) -> tuple[str, float]:
    logger.info("Starting %s", phase_name)
    return phase_name, time.perf_counter()


def _finish_phase(logger: logging.Logger, phase_name: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    logger.info("Finished %s in %.2fs", phase_name, elapsed)


def _run_phase(logger: logging.Logger, phase_name: str, operation: Any) -> Any:
    _, start_time = _log_phase(logger, phase_name)
    try:
        result = operation()
    except Exception:
        elapsed = time.perf_counter() - start_time
        logger.exception("Failed %s after %.2fs", phase_name, elapsed)
        raise
    _finish_phase(logger, phase_name, start_time)
    return result


def _configure_jvm_logging(spark: SparkSession) -> None:
    try:
        jvm = spark._jvm
        configurator = jvm.org.apache.logging.log4j.core.config.Configurator
        level = jvm.org.apache.logging.log4j.Level.ERROR
        for logger_name in JVM_QUIET_LOGGERS:
            configurator.setLevel(logger_name, level)
    except Exception:
        logging.getLogger(__name__).debug(
            "Unable to configure JVM logger levels for noisy Spark warnings",
            exc_info=True,
        )


def build_spark_session(
    app_name: str,
    *,
    include_sedona: bool = False,
    extra_packages: tuple[str, ...] = (),
) -> SparkSession:
    _configure_logging()

    if not os.environ.get("SPARK_LOCAL_HOSTNAME"):
        os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    if not os.environ.get("SPARK_LOCAL_IP"):
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    builder = SparkSession.builder.appName(app_name)
    preinstalled_jars = _preinstalled_jars()
    packages: list[str] = list(extra_packages)

    if include_sedona:
        builder = SedonaContext.builder().appName(app_name)
        builder = builder.config("spark.sql.extensions", DEFAULT_SEDONA_SQL_EXTENSIONS)

    if preinstalled_jars:
        builder = builder.config("spark.jars", ",".join(preinstalled_jars))
    else:
        packages = list(_spark_packages()) + packages
        if include_sedona:
            packages.extend(_sedona_packages())

        package_list: list[str] = _dedupe_packages(packages)
        if package_list:
            builder = builder.config("spark.jars.packages", ",".join(package_list))
        repositories = _csv_env("SPARK_MAVEN_REPOSITORIES", DEFAULT_SPARK_REPOSITORIES)
        if repositories:
            builder = builder.config("spark.jars.repositories", ",".join(repositories))
        builder = builder.config("spark.jars.ivy", "/tmp/.ivy2")

    for key, value in _spark_runtime_configs().items():
        builder = builder.config(key, value)

    builder = builder.config("spark.driver.host", os.environ["SPARK_LOCAL_IP"])
    builder = builder.config("spark.driver.bindAddress", os.environ["SPARK_LOCAL_IP"])

    spark = builder.getOrCreate()
    if include_sedona:
        # SedonaContext.create imports the JVM-side Sedona classes and SQL functions.
        spark = SedonaContext.create(spark)
    spark.sparkContext.setLogLevel(
        os.environ.get("SPARK_LOG_LEVEL", DEFAULT_SPARK_LOG_LEVEL)
    )
    _configure_jvm_logging(spark)
    return spark


class BaseETLJob(ABC):
    """Reusable ETL base class with a small, explicit lifecycle contract."""

    def __init__(
        self,
        *,
        spark: SparkSession,
        config: Mapping[str, Any] | None = None,
        job_name: str | None = None,
    ) -> None:
        _configure_logging()
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
        job_start = time.perf_counter()
        self.logger.info("Starting job %s", self.job_name)
        try:
            data = _run_phase(self.logger, "extract phase", self.extract)
            transformed_data = _run_phase(
                self.logger, "transform phase", lambda: self.transform(data)
            )
            result = _run_phase(
                self.logger, "load phase", lambda: self.load(transformed_data)
            )

            self.logger.info(
                "Finished job %s in %.2fs",
                self.job_name,
                time.perf_counter() - job_start,
            )
            return result
        except Exception:
            self.logger.exception("Job %s failed", self.job_name)
            raise
        finally:
            _run_phase(self.logger, "cleanup phase", self.cleanup)
