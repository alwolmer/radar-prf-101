"""Scrape PRF open-data page and cache cleaned download URLs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_urls.csv"

START_YEAR = 2017
END_YEAR = 2026


def _year_pattern() -> str:
    """Build a regex alternation matching all years in [START_YEAR, END_YEAR]."""
    return "|".join(str(y) for y in range(START_YEAR, END_YEAR + 1))


def _extract_year_and_grouping(ref: str) -> str:
    year_match = pd.Series(ref).str.extract(rf"({_year_pattern()})")
    grouping_match = pd.Series(ref).str.extract(
        r"(Agrupados por ocorrência|Agrupados por pessoa - Todas as causas e tipos de acidentes)"
    )
    if not year_match.empty and not grouping_match.empty:
        return f"{year_match.iloc[0, 0]} - {grouping_match.iloc[0, 0]}"
    return ref


def extract_urls() -> pd.DataFrame:
    raw_urls = pd.read_html(
        "https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf",
        extract_links="all",
    )

    urls = raw_urls[3]
    ref_col, link_col = urls.columns[:2]

    urls["Referência"] = urls[ref_col].apply(
        lambda x: x[0] if isinstance(x, tuple) else x
    )
    urls["Link"] = urls[link_col].apply(
        lambda x: x[1] if isinstance(x, tuple) else None
    )

    urls = urls[["Referência", "Link"]]

    if str(urls.iloc[0]["Referência"]).strip().lower() == "referência":
        urls = urls.iloc[1:].reset_index(drop=True)

    year_re = _year_pattern()
    urls = urls[
        urls["Referência"].str.contains(rf"(?:{year_re})", na=False)
        & urls["Referência"].str.contains(
            r"Agrupados por ocorrência|Agrupados por pessoa - Todas as causas e tipos de acidentes",
            na=False,
        )
    ].reset_index(drop=True)

    urls["Referência"] = urls["Referência"].apply(_extract_year_and_grouping)
    urls.rename(columns={"Referência": "Year and Grouping"}, inplace=True)

    return urls


def main() -> None:
    urls = extract_urls()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urls.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(urls)} URLs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
