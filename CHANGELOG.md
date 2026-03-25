# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- A reusable ETL foundation under `src/etl/` with `BaseETLJob` and a datalake adapter that supports both local storage and S3-backed persistence.
- Spark-based PRF pipeline jobs for `source2bronze` and `bronze2silver`, including partitioned Parquet outputs for `data/bronze/prf_accidents` and `data/silver/prf_accidents_standardized`.
- DVC stage definitions and lock metadata for reproducing the PRF bronze and silver datasets from the implemented ETL jobs.
- A `.env.example` file documenting datalake backend configuration for local and S3 execution.
- A target-state BR-101 architecture document covering the bronze, silver, gold, and gold-to-gold flow, the DNIT/IBGE auxiliary dataset role, partitioned persistence, DVC, and the Spark-to-EMR Serverless scaling path.

### Changed

- The `Makefile` now exposes PRF ETL and DVC reproduction commands for the staged bronze and silver pipeline.
- The `README.md` now documents PRF bronze/silver versioning through `dvc.yaml` instead of the older bronze-directory-only workflow.
- Project dependencies now include `boto3`, `python-dotenv`, and `pyspark` to support the new datalake and Spark ETL runtime.
- `.dvcignore` now ignores generated `*.crc` files produced by DVC operations.

## [0.2.0] - 2026-03-16

### Added

- DVC tracking for the bronze-layer cache in `data/bronze`.
- A GitHub Actions workflow that runs the repository `pre-commit` checks on pull requests to `develop` and `main`.
- A contributor `README.md` with `uv`, agent-skill sync, local `pre-commit`, and DVC usage guidance.

### Changed

- `make extract-data` now refreshes the bronze cache and updates the DVC pointer in one command.
- Project metadata versioning has been bumped to `0.2.0`.
