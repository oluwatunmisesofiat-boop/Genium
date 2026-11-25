"""Thin FastAPI entrypoint exposing GENIE AI verification, products, and async discovery jobs.

Run locally:
  cd /Users/Cridiv/Documents/Genie/genie-ai
  PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from discovery.models import DiscoverySeed
from orchestrator import GenieOrchestrator, OrchestratedVerificationResponse
from verification.adapters import (
    MLInfraVerifyPayload,
    scan_input_from_ml_infra,
    to_ml_infra_response,
)
from verification.image_scan import (
    ImageVerificationResponse,
    infer_mime_type,
    verify_image_bytes,
)
from verification.schemas import ScanInput


class DiscoveryJobRequest(BaseModel):
    product_name: Optional[str] = None
    nafdac_number: Optional[str] = None
    manufacturer: Optional[str] = None
    gtin: Optional[str] = None


def create_app(orchestrator: Optional[GenieOrchestrator] = None) -> FastAPI:
    """Create the GENIE AI FastAPI application wired to a `GenieOrchestrator`."""
    owns_orchestrator = orchestrator is None
    orch = orchestrator or GenieOrchestrator()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_orchestrator:
                orch.shutdown(wait=False)

    app = FastAPI(
        title="GENIE AI — Product Intelligence & Verification API",
        version="0.1.0",
        description="Multi-signal pharmaceutical counterfeit verification with coverage-based graceful degradation and async product discovery.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.orchestrator = orch

    @app.get("/v1/health")
    def get_health() -> Dict[str, Any]:
        return orch.health()

    @app.post("/v1/verify", response_model=OrchestratedVerificationResponse)
    def post_verify(
        scan: ScanInput,
        auto_discover: bool = Query(default=True, description="Enqueue async discovery if product needs enrichment"),
    ) -> OrchestratedVerificationResponse:
        return orch.verify_scan(scan, auto_discover=auto_discover)

    @app.post("/v1/verify/ml-infra")
    def post_verify_ml_infra(
        payload: MLInfraVerifyPayload,
        auto_discover: bool = Query(
            default=True,
            description="Enqueue async discovery if product needs enrichment",
        ),
    ) -> Dict[str, Any]:
        """Bridge endpoint for `ml_infra`: accepts OCR/Barcode/PackComparator payload and returns `ml_infra` VerificationResponse."""
        scan = scan_input_from_ml_infra(payload)
        orch_resp = orch.verify_scan(scan, auto_discover=auto_discover)
        return to_ml_infra_response(orch_resp, payload=payload)

    @app.post("/v1/verify/image", response_model=ImageVerificationResponse)
    async def post_verify_image(
        file: UploadFile = File(..., description="JPEG, PNG, or WebP photo of a medicine pack"),
        compare_visual: bool = Query(
            default=True,
            description="Compare photo against approved reference packaging using the Vision LLM",
        ),
        auto_discover: bool = Query(
            default=True,
            description="Enqueue async discovery if product needs enrichment",
        ),
    ) -> ImageVerificationResponse:
        try:
            mime_type = infer_mime_type(file.filename, file.content_type)
            image_bytes = await file.read()
            return verify_image_bytes(
                image_bytes,
                mime_type,
                orchestrator=orch,
                compare_visual=compare_visual,
                auto_discover=auto_discover,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/products")
    def get_products() -> List[Dict[str, Any]]:
        return orch.list_products()

    @app.get("/v1/products/{slug}")
    def get_product_detail(slug: str) -> Dict[str, Any]:
        data = orch.get_product(slug)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Product dossier {slug!r} not found")
        return data

    @app.post("/v1/discovery/jobs")
    def post_discovery_job(req: DiscoveryJobRequest) -> Dict[str, Any]:
        try:
            seed = DiscoverySeed(
                product_name=req.product_name,
                nafdac_number=req.nafdac_number,
                manufacturer=req.manufacturer,
                gtin=req.gtin,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return orch.trigger_discovery(seed)

    @app.get("/v1/jobs")
    def get_jobs(
        limit: int = Query(default=20, ge=1, le=100),
        status: Optional[str] = Query(default=None),
    ) -> List[Dict[str, Any]]:
        return orch.list_jobs(limit=limit, status=status)

    @app.get("/v1/jobs/{job_id}")
    def get_job_detail(job_id: str) -> Dict[str, Any]:
        job = orch.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Discovery job {job_id!r} not found")
        return job

    @app.get("/v1/scans")
    def get_scans(
        limit: int = Query(default=20, ge=1, le=100),
        verdict: Optional[str] = Query(default=None),
        nafdac_number: Optional[str] = Query(default=None),
    ) -> List[Dict[str, Any]]:
        return orch.list_scans(limit=limit, verdict=verdict, nafdac_number=nafdac_number)

    @app.get("/v1/scans/{scan_id}")
    def get_scan_detail(scan_id: str) -> Dict[str, Any]:
        scan = orch.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail=f"Scan run {scan_id!r} not found")
        return scan

    return app


app = create_app()
