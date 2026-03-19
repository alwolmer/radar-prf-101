"""Download raw PRF data from cached URLs and persist to data/bronze."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URLS_PATH = PROJECT_ROOT / "data" / "cleaned_urls.csv"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"


def download_and_cache(urls_df: pd.DataFrame) -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in urls_df.iterrows():
        url = row["Link"]
        year = row["Year and Grouping"].split(" - ")[0]
        grouping = row["Year and Grouping"].split(" - ")[1]

        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if not match:
            print(f"Skipping {row['Year and Grouping']}: no file ID found in URL")
            continue

        file_id = match.group(1)
        download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

        response = requests.get(download_url, timeout=120)
        if response.status_code != 200:
            print(
                f"Failed to download {row['Year and Grouping']}: HTTP {response.status_code}"
            )
            continue

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file in z.namelist():
                if file.endswith(".csv"):
                    with z.open(file) as f:
                        df_csv = pd.read_csv(f, sep=";", encoding="latin-1")
                        output_path = BRONZE_DIR / f"{year}_{grouping}.csv.gz"
                        df_csv.to_csv(output_path, index=False, compression="gzip")
                        print(f"Saved {output_path}")


def main() -> None:
    urls_df = pd.read_csv(URLS_PATH)
    download_and_cache(urls_df)


if __name__ == "__main__":
    main()
