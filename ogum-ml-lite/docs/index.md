# Ogum-ML 1.0

Welcome to the official documentation of **Ogum-ML**. This site consolidates the
user guides, tutorials and API reference for the 1.0 long-term support release.

## Installation

### Python package

```bash
pip install ogum-ml
```

The default installation ships the CLI, feature engineering utilities, master
sintering curve helpers and the FastAPI server. Optional extras are available
for machine learning benchmarks, physics-informed simulation and PDF exports:

```bash
pip install "ogum-ml[ml]"   # Gradient boosting benchmarks
pip install "ogum-ml[sim]"  # Meshing and sintering simulations
pip install "ogum-ml[pdf]"  # PDF report generation
```

### Docker image

A batteries-included Docker image is published for each release tag:

```bash
docker run --rm -p 8501:8501 -p 8000:8000 huyraestevao/ogum-ml:1.0.0
```

The container launches the Streamlit frontend on port 8501 and the FastAPI
backend on port 8000. The `ogum-ml` CLI is available through `docker exec`.

## Quick start

1. Open the [Jupyter tutorials](tutorials/) to learn how to import datasets,
   calculate master sintering curves and benchmark models.
2. Follow the [user guide](./user_guide.md) for a tour of the CLI and
   Streamlit-based frontend.
3. Explore the [API reference](api_reference/) for module-by-module details of
   the public Ogum-ML surface.

## Support matrix

| Component  | Status             |
| ---------- | ------------------ |
| CLI        | Stable / frozen    |
| FastAPI    | Stable / frozen    |
| Frontend   | Stable (Streamlit) |
| Python API | Stable             |

## Release notes

Ogum-ML 1.0.0 focuses on API stability, production documentation and release
automation. Consult [`ROADMAP.md`](../ROADMAP.md) for the community roadmap and
[`CITATION.cff`](../CITATION.cff) for academic references.
