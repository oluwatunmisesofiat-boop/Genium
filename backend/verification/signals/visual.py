"""Visual packaging verification signal (`independent`, max weight=0.40) + MockCVAdapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from intelligence.synthesizer import compute_reference_level

from ..schemas import ScanInput, SignalResult, VisualMatchInput

VISUAL_MAX_WEIGHT = 0.40
VISUAL_LEVEL_B_WEIGHT = 0.30


class MockCVAdapter:
    """Deterministic mock Computer Vision adapter for offline testing and demos."""

    def __init__(self, preset_match: Optional[VisualMatchInput] = None) -> None:
        self._preset = preset_match

    def analyze(
        self,
        scan: ScanInput,
        approved_versions: List[Dict[str, Any]],
    ) -> Optional[VisualMatchInput]:
        if scan.visual_match is not None:
            return scan.visual_match
        if self._preset is not None:
            return self._preset
        if not approved_versions:
            return None
        primary = next((v for v in approved_versions if v.get("is_primary")), approved_versions[0])
        return VisualMatchInput(
            matched_version_id=str(primary.get("edition_id") or "edition_1"),
            matched_edition_id=str(primary.get("edition_id") or "edition_1"),
            is_historical_version=not bool(primary.get("is_primary", True)),
            similarity_score=0.92,
            angle="front_box",
            anomalies=[],
            model_version="mock-cv-v1",
        )


def _approved_versions(dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
    approved: List[Dict[str, Any]] = []
    for ver in dossier.get("packaging_versions") or []:
        imgs = [i for i in (ver.get("images") or []) if i.get("status") == "approved"]
        if imgs:
            copy = dict(ver)
            copy["images"] = imgs
            approved.append(copy)
    return approved


def evaluate_visual_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
    *,
    cv_adapter: Optional[Any] = None,
) -> Tuple[SignalResult, str, Optional[Dict[str, Any]]]:
    """Evaluate visual match against approved current & historical editions.

    Returns `(SignalResult, reference_level, matched_version_dict)`.
    Visual is ALWAYS applicable (`max_weight=0.40`), so Level C honestly reduces coverage.
    """
    ref_level = compute_reference_level(dossier.get("packaging_versions") or [])
    approved = _approved_versions(dossier)

    if ref_level == "C" or not approved:
        return (
            SignalResult(
                name="visual",
                source="independent",
                status="unavailable",
                direction="neutral",
                strength=0.0,
                weight=0.0,
                max_weight=VISUAL_MAX_WEIGHT,
                summary="No approved reference packaging images on file (Level C).",
                limitation="No verified reference image for this product — packaging was not visually compared.",
            ),
            "C",
            None,
        )

    eff_weight = VISUAL_MAX_WEIGHT if ref_level == "A" else VISUAL_LEVEL_B_WEIGHT
    match: Optional[VisualMatchInput] = scan.visual_match
    if match is None and cv_adapter is not None:
        match = cv_adapter.analyze(scan, approved)

    if match is None:
        return (
            SignalResult(
                name="visual",
                source="independent",
                status="unavailable",
                direction="neutral",
                strength=0.0,
                weight=0.0,
                max_weight=VISUAL_MAX_WEIGHT,
                summary=f"Approved references exist (Level {ref_level}), but no visual scan match was supplied.",
                limitation=f"Approved Level {ref_level} references exist, but no packaging photo was compared.",
            ),
            ref_level,
            None,
        )

    matched_ver = next(
        (
            v
            for v in approved
            if str(v.get("edition_id") or "") in (match.matched_edition_id or "", match.matched_version_id or "")
        ),
        approved[0],
    )
    is_historical = bool(match.is_historical_version or not matched_ver.get("is_primary", True))
    high_anomalies = [a for a in (match.anomalies or []) if a.severity == "high"]

    evidence = {
        "reference_level": ref_level,
        "similarity_score": match.similarity_score,
        "matched_edition_id": matched_ver.get("edition_id"),
        "is_historical_version": is_historical,
        "angle": match.angle,
        "anomalies": [a.model_dump() for a in (match.anomalies or [])],
        "model_version": match.model_version,
    }
    matched_version_meta = {
        "edition_id": matched_ver.get("edition_id"),
        "pack_size_variant": matched_ver.get("pack_size_variant"),
        "summary": matched_ver.get("summary"),
        "is_primary": bool(matched_ver.get("is_primary", False)),
        "is_historical_version": is_historical,
    }

    if high_anomalies or match.similarity_score < 0.45:
        detail = (
            high_anomalies[0].description
            if high_anomalies
            else f"Low visual similarity ({match.similarity_score:.2f}) against approved Level {ref_level} reference."
        )
        return (
            SignalResult(
                name="visual",
                source="independent",
                status="available",
                direction="negative",
                strength=round(max(0.75, 1.0 - match.similarity_score), 2),
                hard_negative="suspicious",
                weight=eff_weight,
                max_weight=VISUAL_MAX_WEIGHT,
                summary=f"Visual packaging anomaly detected: {detail}",
                evidence=evidence,
            ),
            ref_level,
            matched_version_meta,
        )

    if match.similarity_score >= 0.75:
        ver_note = (
            f"historical valid edition {matched_ver.get('edition_id')!r} (no false alarm)"
            if is_historical
            else f"primary edition {matched_ver.get('edition_id')!r}"
        )
        lim = (
            "Single-angle approved reference (Level B) — only one packaging view was visually compared."
            if ref_level == "B"
            else None
        )
        return (
            SignalResult(
                name="visual",
                source="independent",
                status="available",
                direction="positive",
                strength=round(match.similarity_score, 2),
                weight=eff_weight,
                max_weight=VISUAL_MAX_WEIGHT,
                unlocks_likely_authentic=True,
                summary=f"Visual match ({match.similarity_score:.2f}) against approved {ver_note}.",
                evidence=evidence,
                limitation=lim,
            ),
            ref_level,
            matched_version_meta,
        )

    return (
        SignalResult(
            name="visual",
            source="independent",
            status="available",
            direction="neutral",
            strength=round(match.similarity_score, 2),
            weight=eff_weight,
            max_weight=VISUAL_MAX_WEIGHT,
            summary=f"Inconclusive visual similarity ({match.similarity_score:.2f}) against Level {ref_level} reference.",
            evidence=evidence,
            limitation="Visual comparison was inconclusive — retake a clearer, well-lit photo of the carton.",
        ),
        ref_level,
        matched_version_meta,
    )
