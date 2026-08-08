# Outlier Formula OPEN Runner

This repository is being used to execute the public-data `v1.0a-OPEN` research pipeline in GitHub Actions because the ChatGPT sandbox does not permit outbound Python network access to SEC EDGAR / market-data hosts.

The workflow decodes the frozen research package, runs its test suite, then performs a real SEC + market-data network smoke test. Results are uploaded as GitHub Actions artifacts.
