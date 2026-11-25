# GENIE AI — ML & Verification Infrastructure

> **The Machine Learning, Computer Vision, and Product Intelligence backbone powering Genium.**

GENIE AI verifies consumable products (starting with pharmaceuticals in Nigeria) through multi-signal defense-in-depth: fusing GS1/2D barcode payloads, NAFDAC regulatory registries, batch/expiry consistency, DINOv2 global retrieval, LightGlue geometric alignment, and PatchCore patch-level anomaly detection into calibrated three-state verdicts within a **< 4-second latency budget**.

---

## Architecture Overview

```
                        +------------------------------------------+
                        |       PRODUCT INTELLIGENCE DATABASE      |
                        |  - PostgreSQL 16 (Relational & Audit)    |
                        |  - pgvector / Qdrant (Visual Vectors)    |
                        |  - S3 / R2 Object Storage (Images)       |
                        +--------------------+---------------------+
                                             ^
                  OFFLINE PLANE              |              ONLINE PLANE
         +-------------------------------+   |   +--------------------------------+
         | 1. Discovery & Web Ingestion  |   |   | 1. Mobile Scan Ingestion       |
         | 2. Structured Extraction      +---+   |    (FastAPI Gateway)           |
         |    (Greenbook SQLite + LLM)   |       +---------------+----------------+
         | 3. Source & Evidence Gate     |                       |
         | 4. Pack Image Curation (dHash)|       +---------------v----------------+
         | 5. SCD Type 2 Versioning      |       | 2. Quality Gate (<150ms)       |
         | 6. Visual Vector Indexing     |       +---------------+----------------+
         | 7. PatchCore Memory Banks     |                       |
         +-------------------------------+       +---------------v----------------+
                                                 | 3. Pack Detect & Rectification |
                                                 +---------------+----------------+
                                                                 |
                                     +---------------------------+---------------------------+
                                     |                           |                           |
                         +-----------v-----------+   +-----------v-----------+   +-----------v-----------+
                         | 4. Barcode Subsystem  |   | 5. OCR & Key-Field    |   | 6. Computer Vision    |
                         |    - YOLO Detect      |   |    - Text Detection   |   |    - DINOv2 Embed     |
                         |    - Multi-Decoder    |   |    - PP-OCRv5 / docTR |   |    - LightGlue Align  |
                         |    - GS1 AI Parser    |   |    - NAFDAC & Batch   |   |    - PatchCore Anomaly|
                         +-----------+-----------+   +-----------+-----------+   +-----------+-----------+
                                     |                           |                           |
                                     +---------------------------+---------------------------+
                                                                 |
                                                 +---------------v----------------+
                                                 | 7. Candidate Product Resolver  |
                                                 +---------------+----------------+
                                                                 |
                                                 +---------------v----------------+
                                                 | 8. Calibrated Signal Fusion    |
                                                 |    (Hard Rules + Platt Scaling)|
                                                 +---------------+----------------+
                                                                 |
                                                 +---------------v----------------+
                                                 | 9. Explainable 3-State Verdict |
                                                 |    + Decoupled Pharmacy Router |
                                                 +--------------------------------+
```

---

## Directory Structure

```text
ml_infra/
├── pyproject.toml               # Python project configuration & dependencies
├── Makefile                     # Developer tasks (lint, test, run)
├── ML_BUILD_PLAN.md             # Complete technical architecture specification
├── docker/                      # Dockerfiles & docker-compose configurations
├── infra/                       # Cloud infrastructure (Terraform, Kubernetes)
├── configs/                     # Source registries, thresholds, model paths, fusion weights
├── data/                        # DVC pointers, Greenbook mirror, reference images
├── src/genie/
│   ├── common/                  # Configuration, logging, telemetry, errors
│   ├── schemas/                 # Pydantic v2 domain schemas (Product, Scan, Verdict)
│   ├── db/                      # SQLAlchemy 2.0 models, session management, migrations
│   ├── storage/                 # S3/R2 object storage & Qdrant/pgvector adapters
│   ├── discovery/               # Multi-angle search, trust tiering, crawling
│   ├── extraction/              # Schema-constrained LLM extraction with span grounding
│   ├── validation/              # Evidence scoring & promotion gates
│   ├── versioning/              # SCD Type 2 product packaging version lifecycle
│   ├── barcode/                 # Barcode detection, enhancement, multi-decode, GS1 AI
│   ├── vision/                  # Quality gate, pack rect, DINOv2, LightGlue, PatchCore
│   ├── verification/            # Pipeline orchestrator, signals, calibrated fusion
│   ├── pharmacy/                # Decoupled verified pharmacy discovery & inventory
│   └── api/                     # FastAPI endpoints (POST /v1/verify, /v1/products, etc.)
├── training/                    # Model training & fine-tuning pipelines
├── evaluation/                  # Benchmark test suites (barcode, vision, OCR, e2e)
├── scripts/                     # Sync Greenbook mirror, build vector index
└── tests/                       # Unit & integration test suites
```

---

## Quickstart

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key configuration values:
- `DATABASE_URL`: `postgresql+asyncpg://postgres:postgres@localhost:5432/genie`
- `QDRANT_HOST`: `localhost:6333`
- `GEMINI_API_KEY`: Google GenAI API key for LLM dossier extraction
- `STORAGE_BACKEND`: `local` (or `s3`)

### 3. Running the Verification API
```bash
uvicorn genie.api.main:app --reload --port 8000
```
API Documentation will be live at `http://localhost:8000/docs`.

### 4. Running the Tests
```bash
pytest -v
```

---

## Core Operational Concepts

- **3-State Verdicts**: `Consistent with genuine`, `Suspicious`, and `Cannot verify`.
- **SCD Type 2 Packaging Versions**: Legitimate packaging changes are version-bounded with overlapping transition windows, preventing false alarms on genuine legacy stock.
- **Decoupled Pharmacy Discovery**: Pharmacy licensing is verified separately from product authenticity, and pharmacies are only shown as stocking an item when confirmed fresh inventory exists.
