"""Pydantic v2 schemas for GENIE AI's Multi-Signal Verification Engine (Phase D)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

VerificationVerdict = Literal[
    "authentic",
    "likely_authentic",
    "registered_unverified",
    "suspicious",
    "likely_counterfeit",
    "unknown",
]

MLInfraVerdict = Literal[
    "Consistent with genuine",
    "Suspicious",
    "Cannot verify",
]

SignalSourceKind = Literal["nafdac_derived", "independent"]
SignalStatus = Literal["available", "unavailable", "not_applicable"]
SignalDirection = Literal["positive", "negative", "neutral"]
HardNegativeSeverity = Literal["suspicious", "likely_counterfeit"]


def to_ml_infra_verdict(verdict: VerificationVerdict) -> MLInfraVerdict:
    """Map GENIE's 6-state verification verdict to `ml_infra`'s 3-state `VerdictState`."""
    if verdict in ("authentic", "likely_authentic"):
        return "Consistent with genuine"
    if verdict in ("suspicious", "likely_counterfeit"):
        return "Suspicious"
    return "Cannot verify"


class VisualAnomaly(BaseModel):
    region: str = "general"
    description: str
    severity: Literal["low", "medium", "high"] = "medium"


class VisualMatchInput(BaseModel):
    """Output from the Computer Vision adapter (or MockCVAdapter) for a scanned pack."""

    matched_version_id: Optional[str] = None
    matched_edition_id: Optional[str] = None
    is_historical_version: bool = False
    similarity_score: float = Field(ge=0.0, le=1.0)
    angle: str = "front_box"
    anomalies: List[VisualAnomaly] = Field(default_factory=list)
    model_version: str = "mock-cv-v1"


class ScanInput(BaseModel):
    """Consumer scan payload submitted by Genium to GENIE AI's verification engine."""

    product_name_hint: Optional[str] = None
    nafdac_number: Optional[str] = None
    gtin: Optional[str] = None
    batch_number: Optional[str] = None
    mfg_date: Optional[str] = None
    expiry_date: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    observed_security_features: List[str] = Field(default_factory=list)
    features_checked: List[str] = Field(default_factory=list)
    verified_features: List[str] = Field(default_factory=list)
    visual_match: Optional[VisualMatchInput] = None
    scan_date: Optional[str] = None  # ISO YYYY-MM-DD override for deterministic testing


class SignalResult(BaseModel):
    """Uniform output returned by every verification signal evaluator."""

    name: str
    source: SignalSourceKind
    status: SignalStatus
    direction: SignalDirection = "neutral"
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    hard_negative: Optional[HardNegativeSeverity] = None
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    unlocks_likely_authentic: bool = False
    summary: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
    limitation: Optional[str] = None


class CounterfeitTellHit(BaseModel):
    tell_type: str
    summary: str
    matched_on: str
    alert_ref: Optional[str] = None
    source_url: Optional[str] = None
    source_kind: SignalSourceKind = "independent"


class VerificationResult(BaseModel):
    """Full transparent verification output returned to Genium."""

    verdict: VerificationVerdict
    ml_infra_verdict: MLInfraVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    reference_level: Literal["A", "B", "C"] = "C"
    matched_product: Optional[Dict[str, Any]] = None
    matched_version: Optional[Dict[str, Any]] = None
    unregistered_product: bool = False
    recommend_discovery: bool = False
    signals: List[SignalResult] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    counterfeit_tell_hits: List[CounterfeitTellHit] = Field(default_factory=list)
    summary: str = ""
