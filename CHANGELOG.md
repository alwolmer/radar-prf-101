# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-16

### Added

- DVC tracking for the bronze-layer cache in `data/bronze`.
- A GitHub Actions workflow that runs the repository `pre-commit` checks on pull requests to `develop` and `main`.
- A contributor `README.md` with `uv`, agent-skill sync, local `pre-commit`, and DVC usage guidance.

### Changed

- `make extract-data` now refreshes the bronze cache and updates the DVC pointer in one command.
- Project metadata versioning has been bumped to `0.2.0`.
