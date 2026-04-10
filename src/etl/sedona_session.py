from __future__ import annotations

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext


def create_sedona_session(app_name: str) -> SparkSession:
    config = (
        SedonaContext.builder()
        .appName(app_name)
        .config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.8.1",
        )
        .config(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer",
        )
        .config(
            "spark.kryo.registrator",
            "org.apache.sedona.core.serde.SedonaKryoRegistrator",
        )
        .getOrCreate()
    )
    return SedonaContext.create(config)
