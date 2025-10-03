# API reference overview

The Ogum-ML 1.0 public API is composed of the following Python modules:

- `ogum_lite.__init__`: curated exports for data ingestion, feature engineering,
  master sintering curves and machine learning helpers.
- `ogum_lite.cli`: entrypoints for the command line interface.
- `ogum_lite.api`: FastAPI routers and dependency wiring.
- `ogum_lite.sim`: simulation primitives that power the `sim` CLI and REST
  endpoints.
- `ogum_lite.ui`: orchestration helpers shared by the Streamlit and Gradio
  applications.

Auto-generated documentation is built with `mkdocstrings` during the release
pipeline. Run locally:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

The configuration used by the CI lives in `docs/mkdocs.yml`. Extend the modules
listed in the `mkdocstrings.handlers.python` section to publish additional API
surfaces.
