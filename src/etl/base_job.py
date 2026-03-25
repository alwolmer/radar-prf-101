from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pyspark.sql import SparkSession


class BaseETLJob(ABC):
    """Reusable ETL base class with a small, explicit lifecycle contract."""

    def __init__(
        self,
        *,
        spark: Any | None = None,
        config: Mapping[str, Any] | None = None,
        job_name: str | None = None,
    ) -> None:
        self.spark: SparkSession | None = spark
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
