import json
import io

path = "notebooks/eda.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "csv.gz" in line:
                source[i] = line.replace("csv.gz", "parquet")
            elif "pd.read_csv" in line and "compression=" in line:
                source[i] = line.replace("pd.read_csv(doc, compression=\"gzip\")", "pd.read_parquet(doc)")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
