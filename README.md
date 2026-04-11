# radar-prf-101

Utilities for extracting, caching, and analyzing PRF open-data datasets.

## Requirements

- `make` is required to run the project targets documented in this repository
- Docker with the `docker compose` plugin for the `Makefile` targets
- Python 3.11 or 3.12 if you want to use `uv` directly on the host for dependency maintenance
- [`uv`](https://docs.astral.sh/uv/) if you want to manage Python dependencies outside the container
- Node.js with `npx` available if you want to sync local agent skills

## Run The Project With Docker

The supported local runtime is the `spark-env` Docker Compose service. Most
project commands are executed through the repository `Makefile`, which shells
into that container with `docker compose exec`.

Build the image, start the service, and install the locked dependencies inside
the container:

```bash
make docker-build
make install
```

Common development commands:

```bash
make lint
make test
make bronze
make silver
make dvc-repro
```

`make install`, `make lint`, `make test`, `make bronze`, `make silver`, and the
DVC targets all run inside the container. Runtime targets only ensure the
service is running before executing `docker compose exec`; rebuild the image
with `make docker-build` when the Docker context changes.

The Spark runtime coordinates used by the ETL jobs are configured in
`.env.docker` and loaded by `docker-compose.yml`. During the image build,
`scripts/fetch_spark_jars.sh` pre-downloads the Spark, Sedona, Geotools, and
Hadoop AWS JARs into the image, and `src/etl/base_job.py` reads the same
runtime settings to configure Spark defaults such as driver memory, shuffle
parallelism, Parquet block size, and ETL log level.

After changing `.env.docker`, rebuild and restart the service before running the
ETL targets again:

```bash
make docker-build
make docker-up
```

## Manage Python Dependencies With uv

Install or refresh the local environment:

```bash
uv sync --all-groups
```

Add a runtime dependency:

```bash
uv add <package>
```

Add a development dependency:

```bash
uv add --dev <package>
```

Remove a dependency:

```bash
uv remove <package>
```

After editing dependencies manually, regenerate the lockfile:

```bash
uv lock
uv sync --all-groups
```

## Sync Agents From `skills-lock.json`

This repository includes a `skills-lock.json` file for pinned agent skills. To restore the locked skill set locally, run:

```bash
npx skills install
```

That command reads `skills-lock.json` and syncs your installed skills to the locked set.

## Set Up Local pre-commit

Install the hook into `.git/hooks`:

```bash
uv run pre-commit install
```

Run the same checks manually across the repository:

```bash
uv run pre-commit run --all-files
```

The pull request workflow in [`.github/workflows/pre-commit.yml`](.github/workflows/pre-commit.yml) runs the same `pre-commit` suite for pull requests targeting `develop` or `main`.

## PRF Bronze/Silver Versioning With DVC

The implemented PRF ETL pipeline is tracked in [`dvc.yaml`](dvc.yaml) with two stages:

- `prf_source2bronze` versions `data/bronze/prf_accidents`
- `prf_bronze2silver` versions `data/silver/prf_accidents_standardized`

Reproduce both stages, or run them individually:

```bash
make dvc-repro
make dvc-repro-bronze
make dvc-repro-silver
```

Inspect the tracked data state or restore the tracked files:

```bash
make dvc-status
make dvc-checkout
```

The bronze stage still reads the live PRF open-data page and Google Drive archives at execution time. DVC versions the materialized local outputs under `data/` plus the stage graph that binds those outputs to the concrete ETL scripts.
