"""Async discovery job management API router."""
import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(tags=["Discovery"])


class DiscoveryJobRequest(BaseModel):
    product_name: Optional[str] = None
    nafdac_number: Optional[str] = None
    gtin: Optional[str] = None


class DiscoveryJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/discovery/jobs", response_model=DiscoveryJobResponse)
async def trigger_discovery_job(
    request: DiscoveryJobRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid.uuid4())
    # In production, triggers Temporal workflow or async worker
    return DiscoveryJobResponse(
        job_id=job_id,
        status="pending",
        message="Asynchronous product discovery and verification job queued.",
    )
