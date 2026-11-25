"""Bidirectional integration bridge between `ml_infra` and GENIE AI.

Maps:
  1. `ml_infra` OCR (`OCRExtractedFields`), Barcode (`DecodedBarcode`),
     `PackComparator` (`PackComparisonResult`), and VLM extraction payloads
     -> GENIE AI `ScanInput` (with structured `VisualMatchInput` + anomalies).
  2. GENIE AI `OrchestratedVerificationResponse`
     -> `ml_infra` `VerificationResponse` contract (`VerdictState`,
     `SignalBreakdown`, `safer_alternatives`, `execution_time_ms`) plus
     `genie_ai` enrichment metadata (`discovery_job_id`, `reference_level`,
     `coverage`, `counterfeit_tell_hits`, `limitations`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.service import OrchestratedVerificationResponse
from verification.schemas import (
    ScanInput,
    SignalResult,
    VisualAnomaly,
    VisualMatchInput,
)


class MLInfraBarcodeInput(BaseModel):
    gtin: Optional[str] = None
    batch: Optional[str] = None
    expiry: Optional[str] = None
    serial: Optional[str] = None
    symbology: Optional[str] = None
    raw_payload: Optional[str] = None
    check_digit_valid: bool = True


class MLInfraOCRInput(BaseModel):
    nafdac_number: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    mfg_date: Optional[str] = None
    brand_name: Optional[str] = None
    raw_text_lines: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class MLInfraElementCheckInput(BaseModel):
    name: str
    shift_px: float = 0.0
    local_similarity: float = 1.0
    passed: bool = True
    status: str = "ok"


class MLInfraPackComparisonInput(BaseModel):
    overall_similarity_score: float = Field(ge=0.0, le=1.0)
    is_counterfeit_suspect: bool = False
    alignment_matched: bool = True
    inlier_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    reprojection_error: float = 0.0
    color_score: Optional[float] = None
    delta_e_ab: Optional[float] = None
    layout_score: Optional[float] = None
    structural_ssim: Optional[float] = None
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sharpness_ratio: Optional[float] = None
    element_checks: List[MLInfraElementCheckInput] = Field(default_factory=list)
    flagged_grid_zones: List[str] = Field(default_factory=list)
    inconsistencies: List[str] = Field(default_factory=list)
    model_version: str = "ml_infra:pack_comparator"


class MLInfraVerifyPayload(BaseModel):
    """HTTP / in-process request payload sent from `ml_infra` to GENIE AI."""

    ocr: Optional[MLInfraOCRInput] = None
    barcode: Optional[MLInfraBarcodeInput] = None
    pack_comparison: Optional[MLInfraPackComparisonInput] = None
    vlm_extraction: Optional[Dict[str, Any]] = None
    manual_gtin: Optional[str] = None
    manual_nafdac: Optional[str] = None
    product_name_hint: Optional[str] = None
    matched_version_label: Optional[str] = None
    is_legacy_version: bool = False
    retrieval_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    observed_security_features: List[str] = Field(default_factory=list)
    visual_anomaly_heatmap_url: Optional[str] = None
    safer_alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    scan_date: Optional[str] = None


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _build_visual_match(
    pack_comparison: Optional[Dict[str, Any]],
    *,
    matched_version_label: Optional[str] = None,
    is_legacy_version: bool = False,
    retrieval_similarity: Optional[float] = None,
) -> Optional[VisualMatchInput]:
    if not pack_comparison and retrieval_similarity is None:
        return None

    anomalies: List[VisualAnomaly] = []
    if pack_comparison:
        score = float(pack_comparison.get("overall_similarity_score", 0.5))
        is_suspect = bool(pack_comparison.get("is_counterfeit_suspect", False))
        anomaly_score = float(pack_comparison.get("anomaly_score", 0.0))

        for chk in pack_comparison.get("element_checks") or []:
            c = _as_dict(chk)
            if not c.get("passed", True):
                status = str(c.get("status") or "altered")
                sev = "high" if status == "missing_or_altered" else "medium"
                anomalies.append(
                    VisualAnomaly(
                        region=str(c.get("name") or "element"),
                        description=(
                            f"{c.get('name', 'element')} {status} "
                            f"(shift={float(c.get('shift_px', 0.0)):.1f}px, "
                            f"sim={float(c.get('local_similarity', 0.0)):.2f})"
                        ),
                        severity=sev,
                    )
                )

        for msg in pack_comparison.get("inconsistencies") or []:
            text = str(msg).strip()
            if not text:
                continue
            sev = "high" if (is_suspect or anomaly_score >= 0.70) else "medium"
            anomalies.append(VisualAnomaly(region="packaging", description=text, severity=sev))

        for zone in pack_comparison.get("flagged_grid_zones") or []:
            z = str(zone).strip()
            if z and not any(a.region == z for a in anomalies):
                anomalies.append(
                    VisualAnomaly(
                        region=z,
                        description=f"Visual deviation detected in {z} grid region",
                        severity="medium",
                    )
                )

        return VisualMatchInput(
            matched_version_id=matched_version_label,
            matched_edition_id=matched_version_label,
            is_historical_version=bool(is_legacy_version),
            similarity_score=max(0.0, min(1.0, score)),
            angle="front_box",
            anomalies=anomalies,
            model_version=str(pack_comparison.get("model_version") or "ml_infra:pack_comparator"),
        )

    return VisualMatchInput(
        matched_version_id=matched_version_label,
        matched_edition_id=matched_version_label,
        is_historical_version=bool(is_legacy_version),
        similarity_score=max(0.0, min(1.0, float(retrieval_similarity))),
        angle="front_box",
        anomalies=[],
        model_version="ml_infra:retrieval",
    )


def scan_input_from_ml_infra(payload: MLInfraVerifyPayload | Dict[str, Any]) -> ScanInput:
    """Convert an `ml_infra` pipeline payload into GENIE AI's `ScanInput`."""
    data = payload if isinstance(payload, MLInfraVerifyPayload) else MLInfraVerifyPayload.model_validate(payload)
    ocr = data.ocr.model_dump() if data.ocr else {}
    barcode = data.barcode.model_dump() if data.barcode else {}
    vlm = data.vlm_extraction or {}
    comp = data.pack_comparison.model_dump() if data.pack_comparison else None

    product_name = (
        data.product_name_hint
        or ocr.get("brand_name")
        or vlm.get("product_name")
    )
    nafdac_number = (
        ocr.get("nafdac_number")
        or data.manual_nafdac
        or vlm.get("nafdac_number")
    )
    gtin = barcode.get("gtin") or data.manual_gtin
    batch_number = (
        barcode.get("batch")
        or ocr.get("batch_number")
        or vlm.get("batch_number")
    )
    mfg_date = ocr.get("mfg_date") or vlm.get("manufacturing_date")
    expiry_date = (
        barcode.get("expiry")
        or ocr.get("expiry_date")
        or vlm.get("expiry_date")
    )

    text_lines: List[str] = []
    for line in ocr.get("raw_text_lines") or []:
        s = str(line).strip()
        if s:
            text_lines.append(s)
    for line in vlm.get("raw_text_detected") or []:
        s = str(line).strip()
        if s and s not in text_lines:
            text_lines.append(s)
    ocr_text = " ".join(text_lines) if text_lines else None

    ocr_conf = float(ocr.get("confidence", 0.9))
    pqa = vlm.get("packaging_quality_assessment")
    if not ocr and isinstance(pqa, dict) and pqa.get("overall_confidence") is not None:
        try:
            ocr_conf = max(0.0, min(1.0, float(pqa["overall_confidence"])))
        except (TypeError, ValueError):
            pass

    observed_features = list(data.observed_security_features)
    if isinstance(pqa, dict):
        auth_notes = str(pqa.get("authenticity_notes") or "").lower()
        if "scratch" in auth_notes and not any("scratch" in f.lower() for f in observed_features):
            observed_features.append("MAS scratch-off sticker")
        if "hologram" in auth_notes and not any("hologram" in f.lower() for f in observed_features):
            observed_features.append("Hologram")

    visual_match = _build_visual_match(
        comp,
        matched_version_label=data.matched_version_label,
        is_legacy_version=data.is_legacy_version,
        retrieval_similarity=data.retrieval_similarity,
    )

    return ScanInput(
        product_name_hint=product_name,
        nafdac_number=nafdac_number,
        gtin=gtin,
        batch_number=batch_number,
        mfg_date=mfg_date,
        expiry_date=expiry_date,
        ocr_text=ocr_text,
        ocr_confidence=ocr_conf,
        observed_security_features=observed_features,
        visual_match=visual_match,
        scan_date=data.scan_date,
    )


def _signal_score_for_ml_infra(sig: Optional[SignalResult]) -> Optional[float]:
    if sig is None or sig.status != "available":
        return None
    if sig.direction == "positive":
        return round(0.5 + 0.5 * sig.strength, 4)
    if sig.direction == "negative":
        return round(max(0.0, 0.5 - 0.5 * sig.strength), 4)
    return 0.5


def to_ml_infra_response(
    orch_resp: OrchestratedVerificationResponse,
    *,
    payload: Optional[MLInfraVerifyPayload] = None,
    safer_alternatives: Optional[List[Dict[str, Any]]] = None,
    visual_anomaly_heatmap_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Map GENIE AI's `OrchestratedVerificationResponse` to `ml_infra`'s `VerificationResponse` contract."""
    res = orch_resp.result
    by_name = {s.name: s for s in res.signals}

    reasons: List[str] = []
    for sig in res.signals:
        if sig.summary and (sig.status == "available" or sig.hard_negative):
            reasons.append(f"[{sig.name}] {sig.summary}")
    for lim in res.limitations:
        reasons.append(f"[limitation] {lim}")

    pack_comp = payload.pack_comparison if payload is not None else None
    matched_product = res.matched_product or {}
    matched_version = res.matched_version or {}

    version_label = (
        matched_version.get("edition_id")
        or (payload.matched_version_label if payload else None)
    )
    is_legacy = bool(
        matched_version.get("is_historical")
        or (payload.is_legacy_version if payload else False)
    )

    visual_score = _signal_score_for_ml_infra(by_name.get("visual"))
    if visual_score is None and pack_comp is not None:
        visual_score = round(float(pack_comp.overall_similarity_score), 4)

    patch_anomaly = round(float(pack_comp.anomaly_score), 4) if pack_comp is not None else None

    alts = safer_alternatives if safer_alternatives is not None else (
        list(payload.safer_alternatives) if payload is not None else []
    )
    heatmap = visual_anomaly_heatmap_url or (
        payload.visual_anomaly_heatmap_url if payload is not None else None
    )

    return {
        "verdict": res.ml_infra_verdict,
        "confidence": round(float(res.confidence), 4),
        "matched_product_name": matched_product.get("brand_name"),
        "matched_version_label": version_label,
        "is_legacy_version": is_legacy,
        "explanation": res.summary,
        "signals": {
            "barcode_score": _signal_score_for_ml_infra(by_name.get("gtin")),
            "nafdac_score": _signal_score_for_ml_infra(by_name.get("nafdac")),
            "batch_consistency_score": _signal_score_for_ml_infra(by_name.get("batch_expiry")),
            "visual_similarity_score": visual_score,
            "patch_anomaly_score": patch_anomaly,
            "history_match_score": 1.0 if matched_version else None,
            "reasons": reasons,
        },
        "visual_anomaly_heatmap_url": heatmap,
        "safer_alternatives": alts,
        "execution_time_ms": orch_resp.execution_time_ms,
        "genie_ai": {
            "scan_id": orch_resp.scan_id,
            "scan_json_path": orch_resp.scan_json_path,
            "verdict": res.verdict,
            "coverage": res.coverage,
            "reference_level": res.reference_level,
            "unregistered_product": res.unregistered_product,
            "recommend_discovery": res.recommend_discovery,
            "discovery_job_id": orch_resp.discovery_job_id,
            "discovery_job_status": orch_resp.discovery_job_status,
            "discovery_deduplicated": orch_resp.discovery_deduplicated,
            "counterfeit_tell_hits": [h.model_dump() for h in res.counterfeit_tell_hits],
            "limitations": list(res.limitations),
        },
    }


class MLInfraPackComparatorAdapter:
    """Optional in-process CV adapter running `ml_infra.vision.pack_compare.PackComparator`.

    Allows `GenieOrchestrator(cv_adapter=MLInfraPackComparatorAdapter(...))` to compare a
    query image against GENIE AI's approved reference images on disk when `cv2` and
    `ml_infra` are installed in the same Python environment.
    """

    def __init__(self, products_dir: Path | str, query_image_path: Optional[Path | str] = None) -> None:
        self.products_dir = Path(products_dir)
        self.query_image_path = Path(query_image_path) if query_image_path else None

    def compare_paths(
        self,
        query_image_path: Path | str,
        reference_image_path: Path | str,
        *,
        version_id: Optional[str] = None,
        is_historical: bool = False,
    ) -> VisualMatchInput:
        import cv2  # noqa: PLC0415
        from genie.vision.pack_compare import PackComparator  # noqa: PLC0415

        q_img = cv2.imread(str(query_image_path), cv2.IMREAD_COLOR)
        r_img = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
        if q_img is None or r_img is None:
            raise ValueError("Unable to decode query or reference image with OpenCV")

        comp = PackComparator().compare(query_img=q_img, reference_img=r_img)
        comp_dict = {
            "overall_similarity_score": comp.overall_similarity_score,
            "is_counterfeit_suspect": comp.is_counterfeit_suspect,
            "alignment_matched": comp.alignment_matched,
            "inlier_ratio": comp.inlier_ratio,
            "anomaly_score": comp.anomaly_score,
            "element_checks": [
                {
                    "name": e.name,
                    "shift_px": e.shift_px,
                    "local_similarity": e.local_similarity,
                    "passed": e.passed,
                    "status": e.status,
                }
                for e in comp.element_checks
            ],
            "flagged_grid_zones": list(comp.flagged_grid_zones),
            "inconsistencies": list(comp.inconsistencies),
            "model_version": "ml_infra:pack_comparator",
        }
        vm = _build_visual_match(
            comp_dict,
            matched_version_label=version_id,
            is_legacy_version=is_historical,
        )
        assert vm is not None
        return vm
