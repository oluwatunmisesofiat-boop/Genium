"""Stateless verification and reference indexing API router."""
import base64
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
from pydantic import BaseModel, Field

from genie.api.deps import get_vector_store
from genie.schemas.scan import ScanRequest
from genie.schemas.verdict import VerificationResponse
from genie.storage.vector_store import VectorStore
from genie.verification.pipeline import VerificationPipeline
from genie.vision.reference_index import ReferencePackIndexer

router = APIRouter(tags=["Verification"])


class ReferenceIndexRequest(BaseModel):
    product_id: str
    product_version_id: str
    brand_name: str
    manufacturer: str
    image_bytes_base64: str
    nafdac_number: Optional[str] = None
    gtin_list: List[str] = Field(default_factory=list)
    version_label: str = "v1.0"
    status: str = "active"
    valid_to: Optional[str] = None
    shelf_life_months: int = 24
    face_type: str = "front"
    semantic_regions: Optional[Dict[str, Any]] = None


class ReferenceIndexResponse(BaseModel):
    reference_image_id: str
    storage_path: str
    sha256_hash: str
    dhash_64bit: str
    semantic_regions: Dict[str, Any]


@router.post("/verify", response_model=VerificationResponse)
async def verify_scan(
    request: ScanRequest,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Stateless Genium verification endpoint.
    Accepts consumer scan images plus optional Backend-supplied candidate metadata/reference image,
    executes full ML verification (<4s), and returns verdict, scores, and explanation for Backend persistence.
    """
    pipeline = VerificationPipeline(vector_store=vector_store)
    return await pipeline.verify_scan(request)


@router.post("/reference/index", response_model=ReferenceIndexResponse)
async def index_reference_pack(
    request: ReferenceIndexRequest,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Indexes a canonical reference pack image into Object Storage and VectorStore."""
    img_bytes = base64.b64decode(request.image_bytes_base64)
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reference image payload",
        )

    indexer = ReferencePackIndexer(vector_store=vector_store)
    indexed = await indexer.index_reference_pack(
        product_id=request.product_id,
        product_version_id=request.product_version_id,
        image_bgr=img_np,
        brand_name=request.brand_name,
        manufacturer=request.manufacturer,
        nafdac_number=request.nafdac_number,
        gtin_list=request.gtin_list,
        version_label=request.version_label,
        status=request.status,
        valid_to=request.valid_to,
        shelf_life_months=request.shelf_life_months,
        face_type=request.face_type,
        semantic_regions=request.semantic_regions,
    )
    return ReferenceIndexResponse(
        reference_image_id=indexed.reference_image_id,
        storage_path=indexed.storage_path,
        sha256_hash=indexed.sha256_hash,
        dhash_64bit=indexed.dhash_64bit,
        semantic_regions=indexed.semantic_regions,
    )
