# Genium

**Snap a pack, spot a fake — multi-signal drug & consumable verification in seconds.**

Genium is a hackathon-built counterfeit detection and product intelligence platform (starting with pharmaceuticals in Nigeria). It combines a Next.js web/PWA frontend (`app/`), a product intelligence & orchestration backend (`backend/`), and a stateless computer-vision & signal-fusion ML service (`ml_infra/`) to turn a single smartphone photo into an explainable **3-state authenticity verdict** (`Consistent with genuine`, `Suspicious`, or `Cannot verify`).


## Why Genium?

Counterfeit pharmaceuticals often pass naive single-check scanners (for example, a fake box with a copied barcode or a lookalike logo). Genium uses **defense-in-depth multi-signal fusion**:

1. **Barcode & GS1 Intelligence** — Localizes and decodes 1D/2D barcodes (`zxing-cpp`, `pyzbar`, `pylibdmtx`), validates GTIN check digits, and parses GS1 Application Identifiers (GTIN, batch/lot, expiry).
2. **NAFDAC & Regulatory Registry Checks** — Extracts NAFDAC registration numbers and batch/expiry fields via OCR + regex and cross-checks them against Greenbook/registry records.
3. **Computer Vision & Print Forensics** —
   - **Quality Gate** (blur/glare checks before wasting compute)
   - **Pack Rectification & Alignment** using **OpenCV ORB + RANSAC homography** (lightweight, zero PyTorch required)
   - **Color Fidelity** (lighting-normalized CIELAB $\Delta E_{ab}$ & HSV histogram correlation)
   - **Layout & Critical Element Positioning** (4×4 grid SSIM, dilated edge IoU, and landmark shift/missing checks)
   - **Surface Texture Forensics** (8-neighbor Local Binary Pattern histogram intersection + high-frequency Difference-of-Gaussians energy comparison — **if the package surface texture differs from the canonical reference image, it is immediately flagged as `Suspicious` / fake**)
   - **Lightweight Reference-Difference Anomaly Heatmap** (with full PatchCore memory-bank scoring marked as a post-hackathon roadmap item)
4. **Calibrated Multi-Signal Fusion** — Combines deterministic **hard invalidation rules** (invalid GTIN check digit, revoked NAFDAC license, batch/expiry mismatch, surface texture mismatch) with Platt-scaled weighted signal consensus.
5. **Hackathon-Lean Storage** — Uses **SQLite (`aiosqlite`) + NumPy cosine similarity** (`SQLiteVectorStore`) and local image storage instead of heavy Postgres/pgvector/Qdrant containers.

---

## Repository Architecture

```text
Genium/
├── app/                     # Next.js 16 + React 19 + Tailwind CSS v4 frontend (with offline route)
├── backend/                 # Product intelligence, web discovery, Greenbook registry & orchestrator API
└── ml_infra/                # Stateless FastAPI ML & CV verification service
    ├── configs/             # YAML weights, thresholds, model & source configs
    ├── docker/              # Lightweight Dockerfile.api, Dockerfile.worker, docker-compose.yml (SQLite)
    ├── evaluation/          # Barcode, OCR, and end-to-end verification benchmarks
    ├── images/              # Sample test packaging images & pipeline verdict outputs
    ├── infra/               # Kubernetes & Terraform deployment manifests
    ├── scripts/             # Greenbook sync, vector index builder, batch pipeline runner
    ├── src/genie/
    │   ├── api/             # FastAPI gateway (/health, /v1/verify, /v1/discovery)
    │   ├── barcode/         # Barcode detection, enhancement, multi-decoder, GS1 & GTIN parsers
    │   ├── common/          # Pydantic settings, logging, exceptions
    │   ├── discovery/       # Source planning, runner, trust scoring
    │   ├── extraction/      # HTML cleaning, image harvesting, LLM & NAFDAC parsers
    │   ├── schemas/         # Pydantic domain models (ScanRequest, VerificationResponse, VerdictState)
    │   ├── storage/         # LocalObjectStore & SQLiteVectorStore / InMemoryVectorStore
    │   ├── validation/      # Source scoring & staging
    │   ├── verification/    # VerificationPipeline, signal evaluators, CalibratedFusionEngine, ExplainabilityEngine
    │   ├── versioning/      # Packaging version diffing & lifecycle
    │   └── vision/          # QualityGate, PackDetectorRectifier, ORB+RANSAC Matcher, OCR, PackComparator, VLM review
    ├── tests/               # Pytest unit & integration test suite
    └── training/            # Offline training & synthetic augmentation scripts (roadmap/optional)
```

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend (`app/`)** | Next.js 16, React 19, TypeScript, Tailwind CSS v4, Framer Motion, `@ui-sdk/react`, `pnpm` |
| **Backend (`backend/`)** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, `httpx`, `trafilatura`, `ddgs`, `google-genai` |
| **ML Infra (`ml_infra/`)** | Python 3.11+, FastAPI, OpenCV (`opencv-python-headless`), NumPy, `onnxruntime`, `rapidocr-onnxruntime`, `zxing-cpp`, `pyzbar`, `pylibdmtx`, `biip`, `aiosqlite`, `pytest` |

---

## Quickstart (Hackathon Local Setup)

### 1. Frontend (`app/`)
```bash
cd app
pnpm install
pnpm dev
```
Runs the Next.js app at `http://localhost:3000`.

### 2. ML Verification Service (`ml_infra/`)
```bash
cd ml_infra
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run the FastAPI ML service:
```bash
uvicorn genie.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Health check: `GET http://localhost:8000/health`
- Interactive Swagger docs: `http://localhost:8000/docs`

Or run via Docker Compose (SQLite-backed, no Postgres or Qdrant containers needed):
```bash
cd ml_infra/docker
docker compose up --build
```

### 3. Product Intelligence Backend (`backend/`)
```bash
cd backend
pip install -e ".[dev]"
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Running Tests (`ml_infra/`)

```bash
cd ml_infra
# Windows PowerShell:
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/ -v
# macOS/Linux:
PYTHONPATH=src pytest tests/ -v
```

---

## Verification Verdicts & Scoring Rules

Every scan produces one of three calibrated verdicts:

- **`Consistent with genuine`** — Multiple independent signals agree above the category threshold.
- **`Suspicious`** — Triggered by low calibrated consensus OR any deterministic **hard invalidation rule**:
  - Invalid GTIN check digit
  - Revoked NAFDAC registration number
  - Batch / expiry contradiction
  - **Surface texture mismatch** (the scanned drug package surface texture differs from the canonical reference image)
- **`Cannot verify`** — Insufficient verifiable signals or intermediate confidence.

---

## Roadmap (Post-Hackathon)

- **PatchCore Memory-Bank Anomaly Detection**: Per-product coreset memory banks for fine-grained defect localization (currently stubbed/optional behind lazy imports while the demo uses lightweight OpenCV reference-difference heatmaps).
- **Production Vector & Relational Scaling**: Swap `SQLiteVectorStore` for managed pgvector/Qdrant when catalog size outgrows in-process NumPy cosine similarity.
- **SCD Type 2 Packaging Transitions**: Automated multi-version packaging rollout windows to prevent false positives on legitimate legacy stock.
