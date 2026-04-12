# radar-prf-101

**Nome do projeto:** `radar-prf-101`

**Descrição:** Ingestão, versionamento e análise de dados abertos da Polícia Rodoviária Federal (PRF) sobre acidentes em rodovias federais, com adição de dados geométricos do DNIT (SNV) para contexto espacial — em especial a BR-101. O repositório implementa uma pipeline ETL em camadas bronze e silver, orquestrado pelo DVC, com leitura/escrita em datalake local ou S3 (opcional).

Documentação complementar do pipeline e decisões de arquitetura: [`ARQUITETURA.md`](ARQUITETURA.md).

---

## Fonte dos dados

| Fonte | Descrição |
| --- | --- |
| **PRF — Dados abertos** | Página oficial de dados abertos da PRF, que lista conjuntos históricos de acidentes “agrupados por ocorrência” por ano. O camada bronze interpreta as tabelas HTML da página e obtém os links de download. URL padrão usada no código: `https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf`. |
| **Google Drive (espelho PRF)** | Os arquivos publicados pela PRF costumam ser disponibilizados via links do Google Drive. O extrator baixa arquivos ZIP, extrai o CSV e processa com separador `;` e encoding `ISO-8859-1`. |
| **DNIT — SNV (shapefile)** | Bases geométricas do Sistema Nacional de Viação (SNV), obtidas por download direto dos endpoints configurados em `src/etl/bronze/dnit_source2bronze.py` (snapshots por ano). |

Os dados são públicos; a disponibilidade e o formato podem mudar quando o órgão atualizar a página ou os arquivos.

---

## Ingestão e extração

**O que é:** coleta bruta a partir das fontes acima, sem padronização analítica final.

- **PRF:** requisição HTTP à página de dados abertos; `pandas.read_html` para localizar referências e links; download ZIP via URL do Google Drive; descompactação e leitura do CSV no Spark.
- **DNIT:** download dos ZIPs do SNV; leitura de shapefile no Spark (com Apache Sedona), filtragem por rodovia (ex.: BR-101) e metadados de snapshot.

Stages DVC correspondentes: `prf_source2bronze`, `dnit_source2bronze` (ver [`dvc.yaml`](dvc.yaml)).

---

## Transformação

**O que é:** limpeza, tipagem, regras de negócio e preparação para análise.

- **Bronze → Silver (PRF):** leitura do parquet bronze; normalização numérica (vírgula decimal, nulos); padronização de códigos de rodovia; validações geográficas (limites do Brasil); agregações temporais (ex.: início de semana); escrita silver particionada. Implementação: [`src/etl/silver/prf_bronze2silver.py`](src/etl/silver/prf_bronze2silver.py).
- **Bronze → Silver (DNIT):** reprojeção CRS, buffers em metros ao redor da geometria da BR-101 e materialização de camada de corredor para uso espacial. Implementação: [`src/etl/silver/dnit_bronze2silver.py`](src/etl/silver/dnit_bronze2silver.py).

---

## Carregamento

**O que é:** persistência das saídas do pipeline no “datalake” configurado.

- **Backend local (padrão):** gravação sob o diretório `data/`, com subpastas `bronze/` e `silver/`, em **Parquet** (particionado conforme cada job).
- **Backend S3 (opcional):** quando `DATALAKE_BACKEND=s3` e variáveis de bucket/região estão definidas, o mesmo adaptador grava no objeto storage; o Spark usa o filesystem S3A com credenciais via `boto3`. Ver [`src/etl/datalake.py`](src/etl/datalake.py).

O DVC rastreia os artefatos versionados e o grafo de stages que os produz.

---

## Destino (visualização e consumo)

**O que é:** onde os dados ficam disponíveis para exploração hoje e o que falta para um produto de BI.

- **Hoje:** datasets silver (e bronze) em disco local em `data/silver/` (ex.: `prf_accidents_standardized`, `dnit_br101_corridor`), consumíveis por **notebooks** em [`notebooks/`](notebooks/) (EDA com pandas/Polars/GeoPandas etc.) ou por qualquer ferramenta que leia Parquet.
- **Ainda não implementado neste repositório:** camada gold, API, dashboard web ou catálogo de dados corporativo. Esses seriam os destinos naturais para visualização ampla (ver sugestões em [`ARQUITETURA.md`](ARQUITETURA.md)).

---

## Ferramentas já aplicadas

| Categoria | Ferramentas |
| --- | --- |
| **Linguagem e pacotes** | Python 3.13; PySpark; Apache Sedona; pandas; Polars; GeoPandas; requests; boto3; PyArrow; scikit-learn; statsmodels; holidays; matplotlib; seaborn; mlflow (dependência declarada) |
| **Orquestração / dados** | DVC (pipelines e versionamento de artefatos em `dvc.yaml`) |
| **Qualidade e testes** | Ruff; pytest; pre-commit; GitHub Actions (`.github/workflows/pre-commit.yml`) |
| **Ambiente** | `uv` (`pyproject.toml`, `uv.lock`) |

---

## Requirements

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js com `npx` disponível se quiser sincronizar skills locais a partir do lock do repositório

## Manage Python Dependencies With uv

Instalar ou atualizar o ambiente local:

```bash
uv sync --all-groups
```

Adicionar dependência de runtime:

```bash
uv add <package>
```

Adicionar dependência de desenvolvimento:

```bash
uv add --dev <package>
```

Remover dependência:

```bash
uv remove <package>
```

Após editar dependências manualmente, regenere o lockfile:

```bash
uv lock
uv sync --all-groups
```

O `Makefile` do repositório usa o mesmo ambiente `uv`:

```bash
make install
make lint
make test
```

## Sync Agents From `skills-lock.json`

Este repositório inclui `skills-lock.json` com skills fixadas. Para restaurar o conjunto localmente:

```bash
npx skills install
```

## Set Up Local pre-commit

Instalar o hook em `.git/hooks`:

```bash
uv run pre-commit install
```

Executar as mesmas verificações manualmente:

```bash
uv run pre-commit run --all-files
```

O fluxo de pull request em [`.github/workflows/pre-commit.yml`](.github/workflows/pre-commit.yml) roda o mesmo conjunto `pre-commit` para PRs direcionados a `develop` ou `main`.

## PRF/DNIT: Bronze e Silver com DVC

O pipeline ETL está descrito em [`dvc.yaml`](dvc.yaml):

| Stage | Saída versionada |
| --- | --- |
| `prf_source2bronze` | `data/bronze/prf_accidents` |
| `dnit_source2bronze` | `data/bronze/dnit_road_network` |
| `prf_bronze2silver` | `data/silver/prf_accidents_standardized` |
| `dnit_bronze2silver` | `data/silver/dnit_br101_corridor` |

Reproduzir tudo ou por etapa:

```bash
make dvc-repro
make dvc-repro-bronze
make dvc-repro-silver
```

Inspecionar estado ou restaurar artefatos rastreados:

```bash
make dvc-status
make dvc-checkout
```

Os stages de bronze ainda leem as fontes na internet na hora da execução. O DVC versiona as saídas materializadas em `data/` e o grafo que liga cada saída aos scripts ETL concretos.
