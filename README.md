# Ogum-ML 1.0

[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://huyraestevao.github.io/ogum-ml/)
[![PyPI](https://img.shields.io/pypi/v/ogum-ml.svg)](https://pypi.org/project/ogum-ml/)
[![Docker](https://img.shields.io/docker/pulls/huyraestevao/ogum-ml.svg)](https://hub.docker.com/r/huyraestevao/ogum-ml)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

Ogum-ML is the official sintering analytics toolkit powering the Ogum ecosystem.
Release **1.0.0** freezes the public Python modules, CLI commands, FastAPI
endpoints and Streamlit user experience while delivering automated distribution
to PyPI, Docker Hub and GitHub Pages.

## Installation

```bash
pip install ogum-ml
```

Optional extras:

```bash
pip install "ogum-ml[ml]"   # Gradient boosting benchmarks
pip install "ogum-ml[sim]"  # Simulation tooling
pip install "ogum-ml[pdf]"  # PDF export support
```

## Docker image

```bash
docker run --rm -p 8501:8501 -p 8000:8000 huyraestevao/ogum-ml:1.0.0
```

The container exposes the Streamlit frontend at `http://localhost:8501` and the
FastAPI service at `http://localhost:8000`. Attach to the container to access
the `ogum-ml` CLI.

## Documentation

The official guides and tutorials live at
[https://huyraestevao.github.io/ogum-ml/](https://huyraestevao.github.io/ogum-ml/)
and cover:

- Jupyter-based tutorials for master sintering curves, segmentation, ML and
  simulation.
- User guide for the CLI, Streamlit frontend and REST API.
- Contributor handbook and API reference generated with MkDocs.

## Contributing

Read [`docs/contributor_guide.md`](ogum-ml-lite/docs/contributor_guide.md) for
instructions on environment setup, quality gates and the release pipeline.
Future plans are documented in [`ROADMAP.md`](ogum-ml-lite/ROADMAP.md).

## Citation

Please cite Ogum-ML using the metadata provided in
[`CITATION.cff`](ogum-ml-lite/CITATION.cff).
