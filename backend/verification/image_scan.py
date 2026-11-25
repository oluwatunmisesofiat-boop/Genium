"""Vision-LLM image scan extractor and reference comparator for live photo testing.

Provides a self-contained path to test GENIE AI's verification engine on real
consumer drug photos before `ml_infra`'s dedicated CV/OCR service is wired in.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from intelligence.llm import LLMProvider, get_default_provider
from orchestrator.service import GenieOrchestrator, OrchestratedVerificationResponse
from verification.engine import _resolve_product
from verification.schemas import ScanInput, VisualAnomaly, VisualMatchInput
from verification.signals.visual import _approved_versions

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def load_local_dotenv(env_path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from repo `.env` without overriding existing env vars."""
    path = env_path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def infer_mime_type(filename: Optional[str], content_type: Optional[str] = None) -> str:
    """Resolve and validate an image MIME type from HTTP Content-Type or filename."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "image/jpg":
        ct = "image/jpeg"
    if ct in ALLOWED_MIME_TYPES:
        return ct
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed == "image/jpg":
            guessed = "image/jpeg"
        if guessed in ALLOWED_MIME_TYPES:
            return guessed
    raise ValueError(
        f"Unsupported image type ({content_type or filename or 'unknown'}). "
        "Use JPEG (.jpg/.jpeg), PNG (.png), or WebP (.webp)."
    )


class ExtractedImageScan(BaseModel):
    """Fields extracted directly from a consumer drug photo by a Vision LLM."""

    product_name_hint: Optional[str] = None
    nafdac_number: Optional[str] = None
    gtin: Optional[str] = None
    batch_number: Optional[str] = None
    mfg_date: Optional[str] = None
    expiry_date: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    observed_security_features: List[str] = Field(default_factory=list)
    angle: str = "front_box"
    notes: str = ""

    @field_validator("observed_security_features", mode="before")
    @classmethod
    def _coerce_features(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    @field_validator("ocr_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0.85
        return max(0.0, min(1.0, num))


class VisualComparisonOutput(BaseModel):
    """Structured Vision-LLM comparison between a scanned photo and approved reference images."""

    similarity_score: float = Field(ge=0.0, le=1.0)
    is_same_product_packaging: bool = True
    anomalies: List[VisualAnomaly] = Field(default_factory=list)
    summary: str = ""

    @field_validator("similarity_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, num))


class ImageVerificationResponse(BaseModel):
    """Full response returned by `POST /v1/verify/image` and `scripts/verify_image.py`."""

    scan_id: Optional[str] = None
    scan_json_path: Optional[str] = None
    extracted_scan: ExtractedImageScan
    verification: OrchestratedVerificationResponse
    visual_comparison_used: bool = False
    reference_images_compared: List[str] = Field(default_factory=list)
    vision_model: Optional[str] = None


_EXTRACTION_SYSTEM = (
    "You are a pharmaceutical packaging OCR & inspection assistant for Nigeria's NAFDAC market. "
    "Read ONLY what is genuinely visible in the provided photograph. "
    "NEVER invent or guess a NAFDAC registration number, GTIN/barcode, batch/lot number, "
    "manufacturing date, or expiry date if it is not clearly readable in the photo (use null instead)."
)

_EXTRACTION_PROMPT = (
    "Inspect this photograph of a pharmaceutical product pack/blister/bottle and extract:\n"
    "- product_name_hint: brand name + strength + dosage form as printed (e.g. 'Augmentin 625 mg Tablets').\n"
    "- nafdac_number: NAFDAC Reg. No. if visibly printed (e.g. '04-1927', 'A4-4238', 'C4-0143'), else null.\n"
    "- gtin: 8-14 digit EAN/GTIN barcode number if digits are clearly readable, else null.\n"
    "- batch_number: batch/lot code (BN / Batch / Lot) if visibly printed, else null.\n"
    "- mfg_date: manufacturing date string exactly as printed (e.g. '01/2025', '01/25', 'JAN 2025'), else null.\n"
    "- expiry_date: expiry date string exactly as printed (e.g. '01/2027', '01/27', 'JAN 2027'), else null.\n"
    "- ocr_text: concise transcription of all readable text on the visible face(s).\n"
    "- ocr_confidence: your honest 0.0-1.0 confidence in the text legibility (blur/glare lowers it).\n"
    "- observed_security_features: list of clearly visible physical security features (e.g. 'MAS scratch-off panel', 'hologram', 'tamper-evident seal', 'embossed braille', '2D DataMatrix QR'). Empty list if none visible.\n"
    "- angle: one of 'front_box', 'back_box', 'side_box', 'blister_front', 'blister_back', 'bottle_label', 'other'.\n"
    "- notes: short note on image quality, angle, or anything unusual."
)

_COMPARE_SYSTEM = (
    "You are a forensic pharmaceutical packaging comparator. Image 1 is a consumer's scanned photo. "
    "Images 2+ are approved authentic reference photos of the same product from licensed Nigerian pharmacies "
    "or the manufacturer. Compare layout, brand typography, logo geometry, color blocking, strength badge, "
    "and print quality — ignore normal camera perspective, lighting, shadows, or minor wear."
)


def extract_scan_from_image(
    image_bytes: bytes,
    mime_type: str,
    *,
    llm: Optional[LLMProvider] = None,
    scan_date: Optional[str] = None,
) -> Tuple[ScanInput, ExtractedImageScan, str]:
    """Extract a structured `ScanInput` from raw image bytes using the configured Vision LLM."""
    if not image_bytes:
        raise ValueError("Image payload is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")

    load_local_dotenv()
    provider = llm or get_default_provider()
    extracted = provider.generate_structured_with_images(
        prompt=_EXTRACTION_PROMPT,
        images=[(image_bytes, mime_type)],
        schema=ExtractedImageScan,
        system_instruction=_EXTRACTION_SYSTEM,
    )
    scan = ScanInput(
        product_name_hint=extracted.product_name_hint,
        nafdac_number=extracted.nafdac_number,
        gtin=extracted.gtin,
        batch_number=extracted.batch_number,
        mfg_date=extracted.mfg_date,
        expiry_date=extracted.expiry_date,
        ocr_text=extracted.ocr_text,
        ocr_confidence=extracted.ocr_confidence,
        observed_security_features=list(extracted.observed_security_features),
        scan_date=scan_date,
    )
    return scan, extracted, provider.model_name


def _select_reference_images(
    resolved_dossier: Dict[str, Any],
    products_dir: Path,
    *,
    angle: str = "front_box",
    max_refs: int = 2,
) -> List[Tuple[str, str, bytes, str]]:
    """Return up to `max_refs` approved reference images as `(version_id, rel_path, bytes, mime)`."""
    slug = str(resolved_dossier.get("_slug") or "").strip()
    if not slug:
        return []
    product_dir = products_dir / slug
    approved = _approved_versions(resolved_dossier)
    if not approved:
        return []

    candidates: List[Tuple[int, str, str, Path, str]] = []
    for ver in approved:
        ver_id = str(ver.get("edition_id") or ver.get("version_id") or "")
        ver_primary = bool(ver.get("is_primary"))
        for img in ver.get("images") or ver.get("reference_images") or []:
            if not isinstance(img, dict):
                continue
            rel = str(img.get("storage_path") or img.get("local_path") or "").strip()
            if not rel:
                continue
            full_path = product_dir / rel
            if not full_path.is_file():
                continue
            try:
                mime = infer_mime_type(full_path.name, img.get("mime_type"))
            except ValueError:
                continue
            priority = 0
            if ver_primary or img.get("is_primary"):
                priority += 10
            if img.get("has_local_regulatory_markers"):
                priority += 4
            if str(img.get("view_type") or "").lower() == (angle or "").lower():
                priority += 5
            if str(img.get("view_type") or "").lower() == "front_box":
                priority += 2
            candidates.append((priority, ver_id, rel, full_path, mime))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: List[Tuple[str, str, bytes, str]] = []
    seen_paths = set()
    for _prio, ver_id, rel, full_path, mime in candidates:
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        selected.append((ver_id, rel, full_path.read_bytes(), mime))
        if len(selected) >= max_refs:
            break
    return selected


def compare_image_with_references(
    image_bytes: bytes,
    mime_type: str,
    resolved_dossier: Dict[str, Any],
    products_dir: Path,
    *,
    llm: LLMProvider,
    angle: str = "front_box",
) -> Tuple[Optional[VisualMatchInput], List[str]]:
    """Visually compare the scanned photo against approved reference images on disk."""
    refs = _select_reference_images(resolved_dossier, products_dir, angle=angle, max_refs=2)
    if not refs:
        return None, []

    brand = resolved_dossier.get("brand_name") or "this product"
    prompt = (
        f"Image 1 is the consumer's scan of '{brand}'. "
        f"Images 2..{len(refs) + 1} are approved authentic reference photos of '{brand}'.\n"
        "Compare Image 1 against the reference images and return:\n"
        "- similarity_score (0.0-1.0): >= 0.85 if clearly the same authentic packaging design "
        "(allowing camera angle/lighting differences); 0.55-0.84 if uncertain or different pack size/edition; "
        "< 0.55 if packaging design, branding, or spelling clearly conflicts.\n"
        "- is_same_product_packaging: true if it depicts the same product's packaging.\n"
        "- anomalies: list of concrete visual discrepancies (region, description, severity='low'|'medium'|'high'). "
        "Do NOT report camera blur, angle, lighting, or background as counterfeit anomalies.\n"
        "- summary: 1-2 sentence explanation."
    )
    images: List[Tuple[bytes, str]] = [(image_bytes, mime_type)] + [
        (ref_bytes, ref_mime) for _vid, _rel, ref_bytes, ref_mime in refs
    ]
    comp = llm.generate_structured_with_images(
        prompt=prompt,
        images=images,
        schema=VisualComparisonOutput,
        system_instruction=_COMPARE_SYSTEM,
    )
    matched_version_id = refs[0][0] or None
    visual_match = VisualMatchInput(
        matched_version_id=matched_version_id,
        similarity_score=comp.similarity_score,
        angle=angle or "front_box",
        anomalies=list(comp.anomalies),
        model_version=f"vision-llm:{llm.model_name}",
    )
    return visual_match, [rel for _vid, rel, _b, _m in refs]


def verify_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    *,
    orchestrator: GenieOrchestrator,
    llm: Optional[LLMProvider] = None,
    compare_visual: bool = True,
    auto_discover: bool = True,
    scan_date: Optional[str] = None,
) -> ImageVerificationResponse:
    """End-to-end live photo verification: Vision OCR -> optional reference comparison -> Orchestrator."""
    load_local_dotenv()
    provider = llm or get_default_provider()
    scan, extracted, model_name = extract_scan_from_image(
        image_bytes,
        mime_type,
        llm=provider,
        scan_date=scan_date,
    )

    visual_used = False
    ref_paths: List[str] = []
    if compare_visual:
        all_dossiers = orchestrator._current_dossiers()
        resolved, _rec = _resolve_product(scan, all_dossiers, orchestrator.greenbook)
        if resolved is not None:
            try:
                visual_match, ref_paths = compare_image_with_references(
                    image_bytes,
                    mime_type,
                    resolved,
                    orchestrator.products_dir,
                    llm=provider,
                    angle=extracted.angle,
                )
                if visual_match is not None:
                    scan = scan.model_copy(update={"visual_match": visual_match})
                    visual_used = True
            except Exception:  # noqa: BLE001
                visual_used = False
                ref_paths = []

    verification = orchestrator.verify_scan(scan, auto_discover=auto_discover)
    resp = ImageVerificationResponse(
        scan_id=verification.scan_id,
        scan_json_path=verification.scan_json_path,
        extracted_scan=extracted,
        verification=verification,
        visual_comparison_used=visual_used,
        reference_images_compared=ref_paths,
        vision_model=model_name,
    )
    if verification.scan_id:
        json_path = orchestrator.scan_store.update_scan_payload(
            verification.scan_id,
            resp.model_dump(mode="json"),
            source_endpoint="verify_image",
        )
        resp = resp.model_copy(update={"scan_json_path": json_path})
    return resp
