FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System libraries needed for lxml, Pillow, and PDF rendering (pypdfium2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml GENERAL_PLAN.md ./
COPY api ./api
COPY discovery ./discovery
COPY intelligence ./intelligence
COPY orchestrator ./orchestrator
COPY registry ./registry
COPY scraper ./scraper
COPY scripts ./scripts
COPY validation ./validation
COPY verification ./verification

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
