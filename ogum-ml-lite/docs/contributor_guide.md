# Contributor guide

## Local environment

1. Create a virtual environment targeting Python 3.11.
2. Install development dependencies:

```bash
pip install -e .[dev,ml,sim,pdf]
```

3. Run quality gates before opening a pull request:

```bash
ruff check .
black --check .
pytest -q
```

## Documentation

Documentation lives in `docs/` and is built with MkDocs. Preview changes with:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

Pull requests must include updates to tutorials or guides when user-facing
behavior changes.

## Release workflow

- Tag releases with the pattern `vX.Y.Z`.
- GitHub Actions builds the docs, packages the wheel/sdist and publishes to PyPI
  and Docker Hub using repository secrets.
- Verify the published artifacts:
  - `pip install ogum-ml`
  - `docker run huyraestevao/ogum-ml:X.Y.Z --help`

## Community standards

- Use English in commit messages and pull requests.
- Reference issues or discussions that motivate the change.
- Keep the public API backward compatible within the 1.x series.
