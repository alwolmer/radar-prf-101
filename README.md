# radar-prf-101

## Nome e descrição do projeto

**radar-prf-101** é um conjunto de utilitários para **extrair, versionar e analisar** dados abertos de acidentes da Polícia Rodoviária Federal (PRF), com foco no contexto da **BR-101**. O objetivo é sustentar um fluxo tipo *datalake* (camadas bronze e silver), exploração em notebooks e, no horizonte do projeto, produtos analíticos e espaciais mais completos.

O repositório está **em andamento**: parte da arquitetura-alvo e fontes auxiliares já estão descritas em [`documentation/`](documentation/), enquanto o pipeline ETL versionado no DVC cobre hoje a linha PRF bronze → silver.

---

## Fonte dos dados

| Fonte | Papel no projeto |
|--------|-------------------|
| **PRF — dados abertos** | Tabela principal de **acidentes por ocorrência** (anos em foco no ETL). A ingestão bronze lê o [portal de dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) e resolve downloads (incluindo referências a arquivos no Google Drive quando aplicável). |
| **DNIT — geometria rodoviária (SNV)** | Bases geométricas usadas no desenho do corredor da BR-101 (etapas de EDA / especificação; evolução para pipelines dedicados). |
| **IBGE — malhas territoriais** | Limites municipais e regiões geográficas imediatas para atribuição territorial ao longo da BR-101 (idem). |

Detalhes, URLs, nomes de arquivos esperados no bronze e notas de uso estão em **[`documentation/data-sources.md`](documentation/data-sources.md)**. A visão de arquitetura-alvo está em **[`documentation/br101-architecture.md`](documentation/br101-architecture.md)**.

---

## Ferramentas já aplicadas

- **Python 3.13** e gestão de dependências com **[uv](https://docs.astral.sh/uv/)** (`pyproject.toml`, `uv.lock`).
- **Apache Spark (PySpark)** nos jobs ETL modulares em `src/etl/` (`BaseETLJob`, bronze e silver da PRF).
- **DVC** para versionar saídas materializadas e o grafo de estágios (`dvc.yaml` / `dvc.lock`): bronze `data/bronze/prf_accidents` e silver `data/silver/prf_accidents_standardized`.
- **Adaptador de datalake** (`DatalakeAdapter`) com backend **local** ou **S3** (AWS via `boto3`), configurável por ambiente.
- **Qualidade de código**: **Ruff** (lint e formatação), **pre-commit** (hooks locais), **pytest** para testes; **GitHub Actions** reproduz o fluxo do pre-commit em PRs para `develop` e `main` (ver [`.github/workflows/pre-commit.yml`](.github/workflows/pre-commit.yml)).
- **Análise e ciência de dados**: stack declarada no projeto inclui **pandas**, **Polars**, **GeoPandas**, **scikit-learn**, **statsmodels**, **Matplotlib/Seaborn**, **MLflow**, **holidays**, entre outras, para notebooks e evoluções futuras do pipeline.
- **Notebooks** em `notebooks/` para EDA.
- **Agent skills**: `skills-lock.json` na pasta `config/`; para alinhar skills locais ao lockfile, use `npx skills install` (requer Node.js com `npx`).

---

## Requisitos

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js com `npx` disponível, se quiser sincronizar as agent skills locais

---

## Dependências Python com uv

Instalar ou atualizar o ambiente local:

```bash
uv sync --all-groups
```

Adicionar dependência de runtime:

```bash
uv add <pacote>
```

Adicionar dependência de desenvolvimento:

```bash
uv add --dev <pacote>
```

Remover dependência:

```bash
uv remove <pacote>
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

---

## Sincronizar agent skills (`config/skills-lock.json`)

Para restaurar o conjunto de skills fixado no lockfile:

```bash
npx skills install
```

O comando lê `config/skills-lock.json` e alinha as skills instaladas ao conjunto fixado.

---

## pre-commit local

Instalar o hook em `.git/hooks`:

```bash
uv run pre-commit install
```

Executar as mesmas verificações em todo o repositório:

```bash
uv run pre-commit run --all-files
```

---

## Versionamento bronze/silver da PRF com DVC

O pipeline ETL da PRF está descrito em [`dvc.yaml`](dvc.yaml), com dois estágios:

- `prf_source2bronze` — materializa `data/bronze/prf_accidents`
- `prf_bronze2silver` — materializa `data/silver/prf_accidents_standardized`

Reproduzir os dois estágios ou apenas um:

```bash
make dvc-repro
make dvc-repro-bronze
make dvc-repro-silver
```

Inspecionar o estado dos dados rastreados ou restaurar os arquivos versionados:

```bash
make dvc-status
make dvc-checkout
```

O estágio bronze ainda consulta a página viva dos dados abertos da PRF e arquivos em Google Drive **no momento da execução**. O DVC versiona as saídas locais sob `data/` e o grafo que liga essas saídas aos scripts de ETL concretos.

Alternativa aos alvos `make dvc-repro-*`:

```bash
make prf-source2bronze
make prf-bronze2silver
```

---

## Documentação adicional

- [`documentation/eda-runbook.md`](documentation/eda-runbook.md) — roteiro de EDA
- [`documentation/big_data_mlops_rqmts.md`](documentation/big_data_mlops_rqmts.md) — requisitos Big Data / MLOps
