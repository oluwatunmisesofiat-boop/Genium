"""Verification verdict and explanation schemas."""
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class VerdictState(str, Enum):
    CONSISTENT_WITH_GENUINE = "Consistent with genuine"
    SUSPICIOUS = "Suspicious"
    CANNOT_VERIFY = "Cannot verify"


class SignalBreakdown(BaseModel):
    barcode_score: Optional[float] = None
    nafdac_score: Optional[float] = None
    batch_consistency_score: Optional[float] = None
    visual_similarity_score: Optional[float] = None
    patch_anomaly_score: Optional[float] = None
    history_match_score: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)


class SaferAlternative(BaseModel):
    pharmacy_id: UUID
    pharmacy_name: str
    address: str
    distance_km: float
    pharmacy_verified: bool
    stock_confirmed: bool


class VerificationResponse(BaseModel):
    verdict: VerdictState
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_product_name: Optional[str] = None
    matched_version_label: Optional[str] = None
    is_legacy_version: bool = False
    explanation: str
    signals: SignalBreakdown
    visual_anomaly_heatmap_url: Optional[str] = None
    safer_alternatives: List[SaferAlternative] = Field(default_factory=list)
    execution_time_ms: float
