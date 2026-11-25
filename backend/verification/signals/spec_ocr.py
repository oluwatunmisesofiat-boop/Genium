"""Printed-text OCR specification consistency signal (`nafdac_derived`, weight=0.15)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..schemas import ScanInput, SignalResult

SPEC_OCR_WEIGHT = 0.15
OCR_HARD_NEGATIVE_CONFIDENCE = 0.85

_DISQUALIFYING_FORMS = (
    "dispersible",
    "syrup",
    "suspension",
    "injection",
    "cream",
    "ointment",
    "drops",
)
_STRENGTH_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml)\b", re.IGNORECASE)
_SLASH_STRENGTH_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def _extract_strength_numbers(text: str) -> set[str]:
    nums: set[str] = set()
    for m in _STRENGTH_NUM_RE.finditer(text or ""):
        nums.add(m.group(1))
    for m in _SLASH_STRENGTH_RE.finditer(text or ""):
        nums.add(m.group(1))
        nums.add(m.group(2))
    return nums


def evaluate_spec_ocr_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
) -> SignalResult:
    """Compare scanned packaging OCR text against dossier brand, strength, dosage form, and manufacturer."""
    ocr = (scan.ocr_text or "").strip()
    if not ocr:
        return SignalResult(
            name="spec_ocr",
            source="nafdac_derived",
            status="unavailable",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=SPEC_OCR_WEIGHT,
            summary="No packaging OCR text was supplied.",
            limitation="Packaging printed text (OCR) was not available for spec comparison.",
        )

    ocr_low = ocr.lower()
    brand = str(dossier.get("brand_name") or "").strip()
    strength = str(dossier.get("strength") or "").strip()
    dosage_form = str(dossier.get("dosage_form") or "").strip()
    manufacturer = str(dossier.get("manufacturer") or "").strip()

    target_context = f"{brand} {dosage_form}".lower()
    mismatches: List[str] = []
    matched_fields: List[str] = []

    # 1. Brand token check
    brand_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", brand) if len(t) >= 3]
    if brand_tokens and any(re.search(rf"\b{re.escape(tok)}\b", ocr_low) for tok in brand_tokens[:2]):
        matched_fields.append("brand_name")

    # 2. Formulation mismatch check (e.g. OCR says "dispersible" or "syrup" for regular tablet)
    for form_word in _DISQUALIFYING_FORMS:
        if form_word in target_context:
            continue
        if re.search(rf"\b{re.escape(form_word)}\b", ocr_low):
            mismatches.append(f"OCR states {form_word!r} formulation, expected {dosage_form or brand!r}")

    # 3. Strength numbers check (e.g. 20/120 vs 80/480, or 625 vs 375)
    expected_nums = _extract_strength_numbers(f"{brand} {strength}")
    ocr_nums = _extract_strength_numbers(ocr)
    if expected_nums and ocr_nums:
        if expected_nums & ocr_nums:
            matched_fields.append("strength")
        else:
            mismatches.append(
                f"OCR strength numbers {sorted(ocr_nums)} contradict expected {sorted(expected_nums)}"
            )

    # 4. Manufacturer distinctive token check (soft positive if present)
    mfr_tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-z]+", manufacturer)
        if len(t) >= 3 and t.lower() not in {"limited", "ltd", "nigeria", "plc", "pharma", "pharmaceutical", "consumer", "healthcare"}
    ]
    if mfr_tokens and any(re.search(rf"\b{re.escape(tok)}\b", ocr_low) for tok in mfr_tokens):
        matched_fields.append("manufacturer")

    if mismatches:
        confident = scan.ocr_confidence >= OCR_HARD_NEGATIVE_CONFIDENCE
        return SignalResult(
            name="spec_ocr",
            source="nafdac_derived",
            status="available",
            direction="negative",
            strength=0.9 if confident else 0.5,
            hard_negative="suspicious" if confident else None,
            weight=SPEC_OCR_WEIGHT,
            max_weight=SPEC_OCR_WEIGHT,
            summary="; ".join(mismatches),
            evidence={
                "mismatches": mismatches,
                "ocr_confidence": scan.ocr_confidence,
                "matched_fields": matched_fields,
            },
            limitation=(
                None
                if confident
                else f"Possible OCR spec mismatch at low OCR confidence ({scan.ocr_confidence:.2f}) — retake clearer photo."
            ),
        )

    if matched_fields:
        score = min(1.0, 0.55 + 0.15 * len(matched_fields))
        return SignalResult(
            name="spec_ocr",
            source="nafdac_derived",
            status="available",
            direction="positive",
            strength=round(score, 2),
            weight=SPEC_OCR_WEIGHT,
            max_weight=SPEC_OCR_WEIGHT,
            summary=f"Printed text matches dossier specs ({', '.join(matched_fields)}).",
            evidence={"matched_fields": matched_fields, "ocr_confidence": scan.ocr_confidence},
        )

    return SignalResult(
        name="spec_ocr",
        source="nafdac_derived",
        status="available",
        direction="neutral",
        strength=0.0,
        weight=SPEC_OCR_WEIGHT,
        max_weight=SPEC_OCR_WEIGHT,
        summary="OCR text was readable but did not clearly confirm strength or brand details.",
        evidence={"ocr_confidence": scan.ocr_confidence},
        limitation="Printed specs could not be clearly confirmed from the OCR text.",
    )
