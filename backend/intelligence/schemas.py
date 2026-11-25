"""Pydantic v2 schemas for Step 3 Dossier Synthesis & Visual Verification."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceClaim(BaseModel):
    field_name: str
    value: str
    source_url: str
    verbatim_quote: str

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(item) for item in v if item is not None)
        if isinstance(v, dict):
            return str(v)
        return str(v)


class SecurityFeatureSpec(BaseModel):
    feature_type: Literal[
        "scratch_pin",
        "hologram",
        "tamper_seal",
        "tablet_debossing",
        "qr_code",
        "braille",
        "other",
    ] = "other"
    description: str
    required: bool = False
    verifiable: bool = False
    source_url: Optional[str] = None
    verbatim_quote: Optional[str] = None

    @field_validator("feature_type", mode="before")
    @classmethod
    def _normalize_feature_type(cls, v: Any) -> str:
        allowed = {
            "scratch_pin",
            "hologram",
            "tamper_seal",
            "tablet_debossing",
            "qr_code",
            "braille",
            "other",
        }
        val = str(v or "other").strip().lower()
        return val if val in allowed else "other"


class CounterfeitTell(BaseModel):
    tell_type: Literal[
        "compromised_batch",
        "packaging_defect",
        "spelling_error",
        "wrong_color_or_foil",
        "missing_security_feature",
        "fake_manufacturer_info",
        "other",
    ] = "other"
    summary: str
    affected_batches: List[str] = Field(default_factory=list)
    fake_indicator: Optional[str] = None
    genuine_indicator: Optional[str] = None
    alert_ref: Optional[str] = None
    source_url: Optional[str] = None
    verbatim_quote: Optional[str] = None
    source_kind: Literal["nafdac_derived", "independent"] = "independent"

    @field_validator("tell_type", mode="before")
    @classmethod
    def _normalize_tell_type(cls, v: Any) -> str:
        allowed = {
            "compromised_batch",
            "packaging_defect",
            "spelling_error",
            "wrong_color_or_foil",
            "missing_security_feature",
            "fake_manufacturer_info",
            "other",
        }
        val = str(v or "other").strip().lower()
        return val if val in allowed else "other"

    @field_validator("affected_batches", mode="before")
    @classmethod
    def _coerce_batches(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return [str(x).strip() for x in v if x is not None and str(x).strip()]


class RawImageCheck(BaseModel):
    candidate_index: int
    is_product_packaging: bool
    matches_target_product: bool
    is_real_photo: bool = True
    has_watermark: bool = False
    has_local_regulatory_markers: bool = False
    detected_formulation: Optional[str] = None
    detected_pack_count: Optional[int] = None
    pack_size_variant: Optional[str] = None
    packaging_edition_id: str = "edition_1"
    edition_summary: Optional[str] = None
    view_type: Literal[
        "front_box",
        "back_box",
        "blister_pack",
        "bottle",
        "leaflet",
        "other",
    ]
    visible_cues: List[str] = Field(default_factory=list)
    visible_text_snippets: List[str] = Field(default_factory=list)
    quality_ok: bool = True
    rejection_reason: Optional[str] = None

    @field_validator("visible_cues", "visible_text_snippets", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v if x is not None]


class BatchImageVerificationResponse(BaseModel):
    results: List[RawImageCheck] = Field(default_factory=list)


class ExtractedDossierResponse(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    active_ingredients: List[str] = Field(default_factory=list)
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    pack_size: Optional[str] = None
    manufacturer: Optional[str] = None
    nafdac_number: Optional[str] = None
    registration_status: Optional[str] = None
    gtins: List[str] = Field(default_factory=list)
    shelf_life_months: Optional[int] = None
    storage_statement: Optional[str] = None
    packaging_description: Optional[str] = None
    color_scheme: List[str] = Field(default_factory=list)
    distinctive_features: List[str] = Field(default_factory=list)
    expected_security_features: List[SecurityFeatureSpec] = Field(default_factory=list)
    known_counterfeit_tells: List[CounterfeitTell] = Field(default_factory=list)
    recall_alerts: List[str] = Field(default_factory=list)
    version_notes: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)

    @field_validator(
        "active_ingredients",
        "gtins",
        "color_scheme",
        "distinctive_features",
        "recall_alerts",
        "version_notes",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v if x is not None]

    @field_validator("expected_security_features", "known_counterfeit_tells", mode="before")
    @classmethod
    def _coerce_obj_lists(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        return list(v) if isinstance(v, list) else []


class VerifiedReferenceImage(BaseModel):
    image_id: str
    storage_path: str
    source_url: str
    source_page_url: str
    source_trust_tier: int
    sha256: str
    dhash: str
    width: int
    height: int
    mime_type: str
    view_type: str
    packaging_edition_id: str = "edition_1"
    pack_size_variant: Optional[str] = None
    is_real_photo: bool = True
    has_local_regulatory_markers: bool = False
    visible_cues: List[str] = Field(default_factory=list)
    visible_text_snippets: List[str] = Field(default_factory=list)
    captured_from_domain: Optional[str] = None
    captured_from_type: Optional[str] = None
    captured_at: Optional[str] = None
    market_verified: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    status: str = "staging"


class PackagingVersion(BaseModel):
    edition_id: str
    pack_size_variant: Optional[str] = None
    summary: Optional[str] = None
    is_primary: bool = False
    has_local_regulatory_markers: bool = False
    status: str = "staging"
    images: List[VerifiedReferenceImage] = Field(default_factory=list)


class CandidateProductDossier(BaseModel):
    seed: Dict
    enriched_seed: Dict
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    active_ingredients: List[str] = Field(default_factory=list)
    strength: Optional[str] = None
    dosage_form: Optional[str] = None
    pack_size: Optional[str] = None
    manufacturer: Optional[str] = None
    nafdac_number: Optional[str] = None
    registration_status: Optional[str] = None
    gtins: List[str] = Field(default_factory=list)
    shelf_life_months: Optional[int] = None
    storage_statement: Optional[str] = None
    packaging_description: Optional[str] = None
    color_scheme: List[str] = Field(default_factory=list)
    distinctive_features: List[str] = Field(default_factory=list)
    expected_security_features: List[SecurityFeatureSpec] = Field(default_factory=list)
    known_counterfeit_tells: List[CounterfeitTell] = Field(default_factory=list)
    recall_alerts: List[str] = Field(default_factory=list)
    version_notes: List[str] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)
    dossier_status: Literal["staging", "approved"] = "staging"
    evidence_summary: Optional[Dict[str, Any]] = None
    reference_level: Literal["A", "B", "C"] = "C"
    packaging_versions: List[PackagingVersion] = Field(default_factory=list)
    reference_images: List[VerifiedReferenceImage] = Field(default_factory=list)
    rejected_images: List[Dict] = Field(default_factory=list)
    model_used: str
    stats: Dict = Field(default_factory=dict)

    @field_validator(
        "active_ingredients",
        "gtins",
        "color_scheme",
        "distinctive_features",
        "recall_alerts",
        "version_notes",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v if x is not None]
