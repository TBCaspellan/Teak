# Outlier Formula OPEN Runner

This repository executes the public-data `v1.0a-OPEN` research pipeline in GitHub Actions.

## Canonical pre-outcome production gate

`.github/workflows/feature-coverage-local-sec.yml` is the canonical integrated production gate before any forward outcomes are accessed. It restores the SEC EDGAR DuckDB from GitHub Actions cache, downloads it from Hugging Face only on a cache miss, verifies the local database, then runs the point-in-time SEC + market-data feature audit. The audit artifact must continue to report `NO_FORWARD_OUTCOMES_ACCESSED: true`.

The cached local database is intentional: it removes repeated remote DuckDB/Hugging Face reads from normal production validation and avoids rate-limit failures such as HTTP 429.

## Remote SEC workflow

`.github/workflows/feature-coverage-audit.yml` is diagnostic-only and can be started manually with `workflow_dispatch`. It exists to test the remote SEC transport path; its success or failure does not determine production readiness.

Research artifacts, feature coverage diagnostics, and validation reports are uploaded by GitHub Actions.
