# radar-prf-101

Utilities for extracting, caching, and analyzing PRF open-data datasets.

## Requirements

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js with `npx` available if you want to sync local agent skills

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

The repository `Makefile` uses the same `uv` environment:

```bash
make install
make lint
make test
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
