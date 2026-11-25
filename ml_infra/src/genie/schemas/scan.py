"""Scan input schemas."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScanImage(BaseModel):
    image_bytes_base64: Optional[str] = None
    image_url: Optional[str] = None
    face_hint: Optional[str] = Field(None, json_schema_extra={"example": "front"})


class ReferenceCandidateInput(BaseModel):
    """Canonical product metadata and reference pack image supplied by the Backend DB."""
    product_id: Optional[str] = None
    brand_name: Optional[str] = None
    manufacturer: Optional[str] = None
    primary_nafdac_number: Optional[str] = None
    gtin_list: List[str] = Field(default_factory=list)
    version_label: Optional[str] = None
    is_legacy_version: bool = False
    in_transition_window: bool = True
    batch_number_regex: Optional[str] = None
    nafdac_status: str = "active"
    reference_image_base64: Optional[str] = None
    reference_image_path: Optional[str] = None
    semantic_regions: Optional[Dict[str, Any]] = None


class DecodedBarcode(BaseModel):
    gtin: Optional[str] = None
    batch: Optional[str] = None
    expiry: Optional[str] = None
    serial: Optional[str] = None
    symbology: str
    raw_payload: str
    check_digit_valid: bool = True


class OCRExtractedFields(BaseModel):
    nafdac_number: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    brand_name: Optional[str] = None
    raw_text_lines: List[str] = Field(default_factory=list)


class ScanRequest(BaseModel):
    images: List[ScanImage] = Field(..., min_length=1)
    manual_gtin: Optional[str] = None
    manual_nafdac: Optional[str] = None
    user_latitude: Optional[float] = None
    user_longitude: Optional[float] = None
    user_session_id: Optional[str] = None
    reference_candidate: Optional[ReferenceCandidateInput] = None
