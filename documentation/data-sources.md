# Data Sources

This document lists the external datasets used in the project, what each one
contributes to the BR-101 workflow, and how the data is brought into the local
`data/bronze` cache.

## 1. PRF Accident Data

### Official source

- PRF open-data portal:
  `https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf`

### What it provides

- PRF accident occurrence tables by year
- PRF person-level accident tables by year

The project currently uses the occurrence tables as the primary input for the
BR-101 EDA workflow. The person table is available in the bronze layer but is
not an input for the current notebook flow (may be considered later)

### Local ingestion workflow

#### URL discovery

- Script: [`src/etl/extract/extract_urls.py`](../src/etl/extract/extract_urls.py)
- Output: `data/cleaned_urls.csv`

That script scrapes the PRF open-data page, filters the available records to
the years `2017` through `2026`, and keeps the two groupings used by the
project:

- `Agrupados por ocorrência`
- `Agrupados por pessoa - Todas as causas e tipos de acidentes`

#### Download and bronze caching

- Script: [`src/etl/extract/extract_data.py`](../src/etl/extract/extract_data.py)
- Input: `data/cleaned_urls.csv`
- Output directory: `data/bronze`

That script reads the cached URL list, downloads the raw ZIP payloads, extracts
the CSV files, and stores them in compressed form under `data/bronze` with the
pattern:

- `{year}_Agrupados por ocorrência.csv.gz`
- `{year}_Agrupados por pessoa.csv.gz`

### Local refresh commands

- `make extract-urls`
- `make extract-data`

## 2. DNIT BR-101 Road Geometry

### Official source

- DNIT cloud archive for SNV geometric bases:
  `https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr`

### What it provides

- MultiLineString road geometry used to represent the BR-101 trace
- Historical snapshots used to build the canonical BR-101 corridor

### Snapshots currently used

- `201703A.zip`
- `202107A.zip`
- `202601A.zip`

Direct download examples used by the project:

- 2017:
  `https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV Bases Geométricas (2013-Atual) (SHP)&files=201703A.zip`
- 2021:
  `https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV Bases Geométricas (2013-Atual) (SHP)&files=202107A.zip`
- 2026:
  `https://servicos.dnit.gov.br/dnitcloud/index.php/s/oTpPRmYs5AAdiNr/download?path=/SNV Bases Geométricas (2013-Atual) (SHP)&files=202601A.zip`

### Project role

These files are used to:

- isolate BR-101 features from DNIT's national road base
- dissolve each snapshot into a single road geometry
- buffer each yearly trace
- union the yearly buffers into the canonical BR-101 corridor used for spatial
  accident inclusion

## 3. IBGE Territorial Boundaries

### Official sources

- Municipal boundaries:
  `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_Municipios_2024.zip`
- Regiões Geográficas Imediatas:
  `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_RG_Imediatas_2024.zip`

### What they provide

- Official municipality polygons
- Official RGI polygons
- The municipality-to-RGI hierarchy used for attribution

### Project role

These layers are used to:

- retain only the territorial units crossed by BR-101
- attribute canonical accidents to municipality and RGI
- build road-section outputs by municipality and by RGI

## 4. Local Bronze-Layer Summary

The main raw inputs expected in `data/bronze` are:

- PRF occurrence tables for `2017` through `2026`
- PRF person tables for `2017` through `2026`
- DNIT road geometry snapshots `201703A.zip`, `202107A.zip`, and `202601A.zip`
- IBGE municipality polygons `BR_Municipios_2024.zip`
- IBGE RGI polygons `BR_RG_Imediatas_2024.zip`

## 5. Notes

- The bronze cache is versioned with DVC. See the DVC workflow notes in
  [`README.md`](/home/arthur/Documents/projects/radar-prf-101/README.md).
- The current EDA implementation under `notebooks/eda_runbook_steps_1_6.py`
  and `notebooks/eda_runbook_step_7.py` is occurrence-table first.
- Weather and climate data are mentioned in the runbook as future enrichment
  sources, but they are not yet defined as concrete project inputs.
