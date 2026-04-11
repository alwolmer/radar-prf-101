# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added

- A Docker-first local Spark runtime with a repository `Dockerfile`, a local `docker-compose.yml` service, and `.env.docker` for shared Spark, Sedona, Hadoop AWS, and ETL logging configuration.
- `scripts/fetch_spark_jars.sh`, which resolves the runtime Spark/Sedona dependency set during image build and bakes the required JARs into the container image instead of relying on per-run package downloads.

### Changed

- The project `Makefile` now assumes a Docker-based workflow: `make` targets execute inside the `spark-env` container, auto-start the service when needed, and use container-local cache directories for `uv` and DVC operations.
- `src/etl/base_job.py` now builds Spark sessions from environment-driven runtime coordinates, prefers preinstalled JARs when present, applies shared local Spark defaults, configures quieter logging, and records per-phase ETL timings.
- The DNIT bronze and silver ETL jobs were updated to align with the Docker/Sedona runtime: shapefiles are loaded through Sedona's shapefile reader, PRF and DNIT source jobs emit clearer download/source logging, and the DNIT silver union artifacts are materialized on the driver with Shapely/PyProj plus explicit caching and write-time logging.
- Python runtime support is now pinned to `>=3.11,<3.13`, with dependency updates that align the local environment to the containerized Spark stack, including `pyspark==3.5.1`, `apache-sedona==1.5.1`, and `pandas>=2.0.0`.

## [0.3.1]

### Added

- A Spark/Sedona DNIT `source2bronze` job that materializes partitioned BR-101 road-network snapshots at `data/bronze/dnit_road_network`.
- A Spark/Sedona DNIT `bronze2silver` job that builds the canonical BR-101 centerline and 500-meter corridor artifacts at `data/silver/dnit_br101_corridor`, including yearly snapshots and union-all-years outputs.

### Changed

- Spark session bootstrap is now centralized in `src/etl/base_job.py`, including optional Sedona dependencies for geospatial jobs.
- `PrfBronze2Silver` now reads only the `br=101` bronze partition during extract, and the bronze/silver orchestration targets now run all jobs per layer through the `Makefile` and `dvc.yaml`.
- `dvc.yaml`, `dvc.lock`, and the `Makefile` now include the DNIT silver stage and command targets for reproducing the BR-101 corridor artifacts.

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
