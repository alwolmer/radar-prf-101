from __future__ import annotations

from src.etl.sedona_session import create_sedona_session
from src.etl.silver.ibge_base import IbgeBaseBronze2Silver


class IbgeMunBronze2Silver(IbgeBaseBronze2Silver):
    bronze_subpath = "bronze/ibge_territorial/BR_Municipios_2024.zip"
    silver_subpath = "silver/municipalities"
    select_columns = [
        "CD_MUN",
        "NM_MUN",
        "CD_RGI",
        "NM_RGI",
        "SIGLA_UF",
        "geometry",
    ]

    def _job_name(self) -> str:
        return "ibge_mun_bronze2silver"


if __name__ == "__main__":
    spark = create_sedona_session("IBGE Municipality Bronze to Silver")
    job = IbgeMunBronze2Silver(spark=spark)
    job.run()
    spark.stop()
