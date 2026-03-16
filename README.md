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

## Bronze Cache Versioning With DVC

The `data/bronze` directory is tracked by DVC through [`data/bronze.dvc`](data/bronze.dvc).

Refresh the source URL cache:

```bash
make extract-urls
```

Refresh the bronze layer and update the DVC pointer in the same step:

```bash
make extract-data
```

Inspect the tracked data state or restore the tracked files:

```bash
make dvc-status
make dvc-checkout
```

When `data/bronze` changes, commit the updated `data/bronze.dvc` file together with any code changes that produced the new cache contents.
