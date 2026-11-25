# Genium: Medicine and Consumable Product Verification Platform

## Overview

Genium is a multi-tiered product intelligence and counterfeit verification platform designed to protect consumers, pharmacists, and healthcare providers from counterfeit, adulterated, and cloned pharmaceuticals and packaged consumables. By combining camera-based computer vision, barcode and GS1 standards parsing, regulatory cross-checks against National Agency for Food and Drug Administration and Control (NAFDAC) Greenbook records, and calibrated multi-signal fusion, Genium delivers transparent, explainable authenticity verdicts from a single packaging scan.

---

## Problem Statement

Counterfeit pharmaceuticals and substandard packaged goods pose a severe public health risk across West Africa and emerging markets:

1. **Cloned Registration Numbers**: Counterfeiters routinely copy valid NAFDAC registration numbers from legitimate, low-cost medicines and print them onto counterfeit antibiotics or antimalarials. A basic manual lookup only confirms that a number exists — not whether it belongs to the product, manufacturer, or packaging in the consumer's hand.
2. **Incomplete Scratch-Code Coverage**: Mobile Authentication Service (MAS) scratch-off codes cover only a subset of drug categories, leaving many essential medicines and packaged foods without point-of-sale verification.
3. **Subtle Physical & Surface Tampering**: High-grade counterfeits often replicate logos and layouts while using inferior cardstock, different surface coatings, shifted regulatory badges, or altered ink chemistry that naked-eye inspection misses.
4. **Slow Product & Version Discovery**: Legitimate manufacturers periodically update packaging artwork. Without version-aware reference indexing and automated dossier discovery, genuine legacy stock can be misclassified or new product lines left unverified.

Genium addresses these challenges through defense-in-depth verification: fusing deterministic regulatory and barcode rules with geometric alignment, color fidelity, layout positioning, and surface texture forensics.

---

## System Architecture

Genium is organized as a modular monorepo comprising three specialized subsystems:

```text
Genium/
├── app/                          # Next.js 16 Web & Mobile-First Scanner Frontend
│   ├── app/                      # App Router pages, global styles, and /offline support
│   ├── public/                   # Static assets and icons
│   └── package.json              # Next.js 16, React 19, Tailwind CSS v4, Framer Motion
├── backend/                      # Product Intelligence & Verification Orchestrator API
│   ├── api/                      # FastAPI entrypoint (/v1/verify, /v1/products, /v1/scans, /v1/jobs)
│   ├── discovery/                # Multi-angle product discovery planner, runner, and trust scoring
│   ├── intelligence/             # LLM synthesis, image harvesting, and dossier storage
│   ├── orchestrator/             # Scan orchestration, async discovery jobs, and persistence
│   ├── registry/                 # NAFDAC Greenbook and trusted source registries
│   ├── scraper/                  # Web search and structured content extraction
│   ├── validation/               # Evidence scoring and reference promotion
│   └── verification/             # Verification engine, image scan handler, and ml_infra adapters
└── ml_infra/                     # Stateless Computer Vision & Signal-Fusion ML Service
    ├── configs/                  # YAML fusion weights, category thresholds, model & source configs
    ├── docker/                   # Dockerfiles (API, worker) and SQLite-backed docker-compose.yml
    ├── evaluation/               # Barcode, OCR, and end-to-end verification benchmarks
    ├── images/                   # Sample packaging images and pipeline verdict fixtures
    ├── infra/                    # Kubernetes and Terraform deployment manifests
    ├── scripts/                  # Greenbook sync, vector index builder, and batch runner scripts
    ├── src/genie/
    │   ├── api/                  # FastAPI service (/health, /v1/verify, /v1/reference/index, /v1/discovery/jobs)
    │   ├── barcode/              # Barcode localization, enhancement, multi-decoder, GS1 AI & GTIN
    │   ├── common/               # Pydantic settings, structured logging, domain exceptions
    │   ├── discovery/            # Discovery planning and trust tiering
    │   ├── extraction/           # HTML cleaning, image harvesting, LLM & NAFDAC parsers
    │   ├── schemas/              # Pydantic v2 contracts (ScanRequest, VerificationResponse, VerdictState)
    │   ├── storage/              # LocalObjectStore & SQLiteVectorStore (aiosqlite + NumPy cosine similarity)
    │   ├── validation/           # Source scoring and staging
    │   ├── verification/         # VerificationPipeline, 5 signal evaluators, CalibratedFusionEngine, ExplainabilityEngine
    │   ├── versioning/           # Packaging version diffing and lifecycle management
    │   └── vision/               # Quality gate, pack rectification, ORB+RANSAC matcher, OCR, PackComparator, VLM triage
    ├── tests/                    # Pytest unit and integration test suites
    └── training/                 # Offline model training and synthetic counterfeit augmentation scripts
```

### High-Level Data Flow

1. **Image Acquisition (`app/`)**: A consumer or pharmacist captures a photo of product packaging using the Next.js web interface (with offline fallback support at `/offline`).
2. **Quality Gate & Rectification (`ml_infra/`)**: The ML service evaluates blur and glare thresholds (`ImageQualityGate`). Acceptable images are perspective-rectified (`PackDetectorRectifier`).
3. **Parallel Evidence Extraction (`ml_infra/`)**:
   - **Barcode Branch**: Localizes 1D/2D barcodes (`BarcodeDetector`), decodes payloads via `zxing-cpp`, `pyzbar`, and `pylibdmtx` (`MultiDecoder`), validates GTIN check digits, and parses GS1 Application Identifiers (batch/lot, expiry).
   - **OCR & Regulatory Branch**: Extracts NAFDAC registration numbers, batch numbers, and expiry dates (`PackagingOCR` with `rapidocr-onnxruntime`).
   - **Visual Retrieval Branch**: Generates a normalized visual embedding (`VisualEmbedder`) and queries `SQLiteVectorStore` (using `aiosqlite` and NumPy cosine similarity) or backend-supplied reference metadata to resolve the candidate product.
4. **Reference Packaging Forensics (`PackComparator`)**: When a canonical reference image is available, the query image is aligned via **OpenCV ORB keypoints + RANSAC homography** (`GeometricMatcher`) and compared across four forensic dimensions:
   - **Color Fidelity**: Lighting-normalized CIELAB $\Delta E_{ab}$ and HSV histogram correlation.
   - **Structural Layout & Element Positioning**: 4×4 grid SSIM, dilated Canny edge IoU, and positional/NCC checks on critical landmarks (brand logo, NAFDAC badge, barcode, security seal).
   - **Surface Texture Forensics**: 8-neighbor Local Binary Pattern (LBP) histogram intersection and Difference-of-Gaussians high-frequency energy comparison. If the scanned package surface texture differs from the canonical reference image, it triggers an immediate hard invalidation.
   - **Localized Anomaly Heatmap**: Lightweight OpenCV reference-difference scoring and false-color Jet overlay generation.
5. **Calibrated Signal Fusion (`CalibratedFusionEngine`)**: Combines deterministic hard invalidation rules with Platt-scaled weighted consensus across five independent signals (`barcode`, `nafdac`, `batch`, `visual`, `history`).
6. **Orchestration & Continuous Discovery (`backend/`)**: The orchestrator persists scan runs, cross-checks NAFDAC Greenbook records, and automatically enqueues asynchronous web discovery jobs when a product dossier is missing or incomplete.

---

## The 3-State Verdict Engine

Because a camera scan cannot perform laboratory chemical assays, Genium communicates authenticity transparently using three calibrated verdict states (`VerdictState`):

| Verdict | Meaning | Criteria | User Guidance |
| :--- | :--- | :--- | :--- |
| **`Consistent with genuine`** | Strong multi-signal match | Calibrated consensus meets or exceeds the category threshold (`genuine_min_calibrated_prob`), regulatory and barcode checks pass, and visual/surface forensics align with the canonical reference. | Packaging records, identifiers, and physical print/surface characteristics align with the verified product reference. |
| **`Suspicious`** | Counterfeit or tampering risk | Calibrated consensus falls below `suspicious_max_calibrated_prob`, **or** any deterministic hard invalidation rule triggers (invalid GTIN check digit, revoked NAFDAC license, batch/expiry contradiction, or **surface texture mismatch**). | Do not consume or dispense without professional verification. Inspect the explanation details and report suspected counterfeits to NAFDAC. |
| **`Cannot verify`** | Insufficient or ambiguous evidence | Available verifiable signals fall below the minimum information threshold, or confidence lies between the suspicious and genuine cutoffs. | Retake a clearer, glare-free photo of the front panel, barcode, and NAFDAC number, or consult a licensed pharmacist. |

---

## Technical Stack

### Frontend (`app/`)
- **Framework**: Next.js 16 (App Router) with React 19 and TypeScript 5
- **Styling & Motion**: Tailwind CSS v4, Framer Motion
- **UI & SDKs**: `@ui-sdk/core`, `@ui-sdk/react`
- **Package Manager**: `pnpm`

### Product Intelligence Backend (`backend/`)
- **Framework**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2
- **Discovery & Extraction**: `httpx`, `curl_cffi`, `ddgs`, `trafilatura`, `lxml`, `resiliparse`, `pypdfium2`
- **AI Synthesis**: `google-genai` (Gemini) for dossier synthesis and vision-assisted fallback verification

### ML & Computer Vision Service (`ml_infra/`)
- **Framework**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, `pydantic-settings`
- **Computer Vision & Forensics**: OpenCV (`opencv-python-headless`), NumPy, Pillow, `onnxruntime`, `rapidocr-onnxruntime`
- **Barcode & GS1**: `zxing-cpp`, `pyzbar`, `pylibdmtx`, `biip`
- **Storage**: `aiosqlite` + NumPy cosine similarity (`SQLiteVectorStore`) and local object storage (`LocalImageStorage`) / optional S3 (`boto3`)
- **Testing & Quality**: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`

---

## API Specifications

### ML Verification Service (`ml_infra/` — Default Port `8000`)

#### 1. Health Check (`GET /health`)
Returns service readiness and runtime mode (`stateless`).

#### 2. Stateless Scan Verification (`POST /v1/verify`)
Accepts a `ScanRequest` (base64-encoded packaging images, optional manual GTIN/NAFDAC overrides, and optional backend-supplied `reference_candidate`) and returns a `VerificationResponse`:
- `verdict`: `"Consistent with genuine"` | `"Suspicious"` | `"Cannot verify"`
- `confidence`: Calibrated probability score in `[0.0, 1.0]`
- `matched_product_name`, `matched_version_label`, `is_legacy_version`
- `explanation`: Human-readable summary of all contributing signals
- `signals`: `SignalBreakdown` (`barcode_score`, `nafdac_score`, `batch_consistency_score`, `visual_similarity_score`, `history_match_score`, `reasons`)
- `execution_time_ms`

#### 3. Reference Pack Indexing (`POST /v1/reference/index`)
Indexes a canonical product packaging image into local/object storage and `SQLiteVectorStore`, computing SHA-256, 64-bit dHash, visual embedding, and semantic region metadata.

#### 4. Discovery Job Trigger (`POST /v1/discovery/jobs`)
Queues an asynchronous product discovery job for unknown products.

---

### Product Intelligence & Orchestrator API (`backend/` — Default Port `8001`)

- `GET /v1/health` — Orchestrator health and registry status
- `POST /v1/verify` — Structured `ScanInput` verification with optional `auto_discover`
- `POST /v1/verify/ml-infra` — Bridge endpoint accepting `MLInfraVerifyPayload` from `ml_infra`
- `POST /v1/verify/image` — Direct `multipart/form-data` image upload verification (`file`, `compare_visual`, `auto_discover`)
- `GET /v1/products` & `GET /v1/products/{slug}` — List and inspect verified product dossiers
- `POST /v1/discovery/jobs`, `GET /v1/jobs`, `GET /v1/jobs/{job_id}` — Trigger and monitor product discovery jobs
- `GET /v1/scans`, `GET /v1/scans/{scan_id}` — Query persisted scan runs and verdicts

---

## Local Development Setup

### Prerequisites
- **Python 3.11+**
- **Node.js 20+** and **`pnpm`** (`npm install -g pnpm`)
- Optional: `GEMINI_API_KEY` / `NVIDIA_API_KEY` for LLM dossier synthesis and VLM second-opinion triage

### 1. ML Verification Service (`ml_infra/`)
```bash
cd ml_infra
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env

uvicorn genie.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is available at `http://localhost:8000/docs`.

### 2. Product Intelligence Backend (`backend/`)
```bash
cd backend
python -m venv .venv
# Activate venv, then:
pip install -e ".[dev]"

uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Frontend Web Application (`app/`)
```bash
cd app
pnpm install
pnpm dev
```
Open `http://localhost:3000` in your browser.

---

## Running Tests (`ml_infra/`)

```bash
cd ml_infra

# Windows PowerShell:
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/ -v

# macOS / Linux:
PYTHONPATH=src pytest tests/ -v
```

---

## Container Deployment

The ML service includes a lightweight Docker Compose setup (`ml_infra/docker/docker-compose.yml`) backed by SQLite and local volume storage (no external Postgres or Qdrant containers required):

```bash
cd ml_infra/docker
docker compose up --build
```

Kubernetes (`ml_infra/infra/k8s/deployment.yaml`) and Terraform (`ml_infra/infra/terraform/main.tf`) manifests are also provided for cloud scaling.

---

## Scope, Roadmap, and Responsible AI Boundaries

1. **Physical Packaging Verification Only**: Genium verifies barcode integrity, regulatory registration consistency, print/layout alignment, color fidelity, and surface texture. It does not perform chemical spectroscopy or laboratory purity testing.
2. **No Medical Advice**: Genium provides packaging authenticity context and regulatory guidance; it does not prescribe medications or replace licensed pharmacists and physicians.
3. **Post-Verification Roadmap**:
   - **PatchCore Memory-Bank Anomaly Detection**: Per-product coreset memory banks (`training/anomaly/build_patchcore.py`) are implemented as an offline/roadmap capability; live verification currently uses lightweight OpenCV reference-difference scoring so no per-product training data is required at onboarding.
   - **Managed Vector Scaling**: `SQLiteVectorStore` can be swapped behind the `VectorStore` protocol for managed vector databases as catalog size grows beyond thousands of SKUs.

---

## License

This project is licensed under the MIT License. See the repository license configuration for details.
