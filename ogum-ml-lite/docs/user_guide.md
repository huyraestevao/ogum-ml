# User guide

This guide describes how to work with the Ogum-ML command line, Streamlit
frontend and REST API for day-to-day sintering analysis.

## Command line interface

The CLI entrypoint is available as `ogum-ml`. List the available commands:

```bash
ogum-ml --help
```

Key commands:

- `ogum-ml prep`: ingest spreadsheets or CSV files and produce normalized
  tables for densification analysis.
- `ogum-ml msc`: compute master sintering curves, θ(Ea) tables and the core
  kinetic metrics of the Ogum pipeline.
- `ogum-ml segmentation`: run threshold or data-driven segmentation to
  identify densification stages.
- `ogum-ml ml`: train and compare machine learning models using the built-in
  benchmarking harness.
- `ogum-ml sim`: execute the finite element based simulation workflow
  (requires the `sim` extra).

Every sub-command includes contextual help (`ogum-ml msc --help`). Detailed
examples live alongside the tutorials in the documentation site.

## Streamlit frontend

Start the full-featured frontend from a local checkout:

```bash
streamlit run app/streamlit_app.py
```

The sidebar exposes:

1. Workspace management and data uploads.
2. Mode selection (guided wizard, advanced dashboard, educational mode).
3. Telemetry, language settings and theme overrides.

Each page mirrors the CLI commands and orchestrates them through background
workers. Generated artifacts are saved inside `artifacts/` and surfaced in the
UI for download.

## REST API

The FastAPI server exposes programmatic access to the same core features. Run
it locally:

```bash
uvicorn server.api_main:app --reload
```

Published endpoints include:

- `POST /runs/import`: ingest a dataset and return normalized tables.
- `POST /runs/msc`: build master sintering curves and activation energy tables.
- `POST /runs/segment`: derive stage-aware features and segmentation reports.
- `POST /runs/ml`: execute training jobs using the scheduler abstraction.
- `POST /runs/publish`: push artifacts to integrated repositories (Zenodo and
  Figshare).

The interactive API reference is available at `/docs` once the server is
running. Stable schemas for request and response payloads are tracked in the
`server/schemas/` module.
