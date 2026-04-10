from __future__ import annotations

from src.etl.sedona_session import create_sedona_session
from src.etl.silver.ibge_base import IbgeBaseBronze2Silver


class IbgeRgiBronze2Silver(IbgeBaseBronze2Silver):
    bronze_subpath = "bronze/ibge_territorial/BR_RG_Imediatas_2024.zip"
    silver_subpath = "silver/rgis"
    select_columns = [
        "CD_RGI",
        "NM_RGI",
        "CD_RGINT",
        "NM_RGINT",
        "SIGLA_UF",
        "geometry",
    ]

    def _job_name(self) -> str:
        return "ibge_rgi_bronze2silver"


if __name__ == "__main__":
    spark = create_sedona_session("IBGE RGI Bronze to Silver")
    job = IbgeRgiBronze2Silver(spark=spark)
    job.run()
    spark.stop()
