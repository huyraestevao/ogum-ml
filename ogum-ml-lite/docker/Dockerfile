FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/ogum

COPY pyproject.toml README.md LICENSE ./
COPY ogum_lite ./ogum_lite
COPY artifacts ./artifacts

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

RUN useradd --create-home --shell /bin/bash ogum
USER ogum
WORKDIR /work

ENTRYPOINT ["python", "-m", "ogum_lite.cli"]
CMD ["--help"]
