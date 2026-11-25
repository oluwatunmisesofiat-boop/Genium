"""GS1 GTIN barcode check-digit and cross-product link signal (`independent`, weight=0.08)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scraper.extract.structured import is_valid_gtin

from ..schemas import ScanInput, SignalResult

GTIN_WEIGHT = 0.08


def _clean_gtin(raw: Optional[str]) -> str:
    return re.sub(r"\D", "", str(raw or ""))


def evaluate_gtin_signal(
    scan: ScanInput,
    dossier: Dict[str, Any],
    *,
    all_dossiers: Optional[List[Dict[str, Any]]] = None,
) -> SignalResult:
    """Validate GS1 Mod-10 check digit and check GTIN linkage to this or another product."""
    dossier_gtins = [_clean_gtin(g) for g in (dossier.get("gtins") or []) if _clean_gtin(g)]
    submitted = _clean_gtin(scan.gtin)

    if not submitted:
        if not dossier_gtins:
            return SignalResult(
                name="gtin",
                source="independent",
                status="not_applicable",
                direction="neutral",
                strength=0.0,
                weight=0.0,
                max_weight=0.0,
                summary="No GTIN is tracked for this product and none was scanned.",
            )
        return SignalResult(
            name="gtin",
            source="independent",
            status="unavailable",
            direction="neutral",
            strength=0.0,
            weight=0.0,
            max_weight=GTIN_WEIGHT,
            summary="Barcode/GTIN was not scanned.",
            limitation="Barcode/GTIN was not provided on this scan.",
        )

    if not is_valid_gtin(submitted):
        return SignalResult(
            name="gtin",
            source="independent",
            status="available",
            direction="negative",
            strength=1.0,
            hard_negative="likely_counterfeit",
            weight=GTIN_WEIGHT,
            max_weight=GTIN_WEIGHT,
            summary=f"Scanned GTIN {submitted} fails GS1 Modulo-10 check-digit validation.",
            evidence={"submitted_gtin": submitted},
        )

    # Check if this valid GTIN belongs to a DIFFERENT known product
    target_nafdac = str(dossier.get("nafdac_number") or "").strip().upper()
    target_brand = str(dossier.get("brand_name") or "").strip().lower()
    for other in all_dossiers or []:
        other_gtins = {_clean_gtin(g) for g in (other.get("gtins") or [])}
        if submitted not in other_gtins:
            continue
        other_nafdac = str(other.get("nafdac_number") or "").strip().upper()
        other_brand = str(other.get("brand_name") or "").strip().lower()
        same_product = (
            (target_nafdac and other_nafdac and target_nafdac == other_nafdac)
            or (target_brand and other_brand and target_brand == other_brand)
        )
        if not same_product:
            return SignalResult(
                name="gtin",
                source="independent",
                status="available",
                direction="negative",
                strength=1.0,
                hard_negative="likely_counterfeit",
                weight=GTIN_WEIGHT,
                max_weight=GTIN_WEIGHT,
                summary=(
                    f"Scanned GTIN {submitted} belongs to another product ({other.get('brand_name')}), "
                    f"not {dossier.get('brand_name')}."
                ),
                evidence={
                    "submitted_gtin": submitted,
                    "belongs_to_product": other.get("brand_name"),
                },
            )

    if submitted in dossier_gtins:
        return SignalResult(
            name="gtin",
            source="independent",
            status="available",
            direction="positive",
            strength=0.4,  # Weak positive: codes can be cloned, never confirms authenticity alone
            weight=GTIN_WEIGHT,
            max_weight=GTIN_WEIGHT,
            unlocks_likely_authentic=False,
            summary=f"GTIN {submitted} has a valid check digit and matches this product (weak signal — barcodes can be cloned).",
            evidence={"submitted_gtin": submitted, "matched_dossier_gtin": True},
        )

    return SignalResult(
        name="gtin",
        source="independent",
        status="available",
        direction="neutral",
        strength=0.0,
        weight=GTIN_WEIGHT,
        max_weight=GTIN_WEIGHT,
        summary=f"GTIN {submitted} has a valid GS1 check digit, but no verified GTIN is on file for this product yet.",
        evidence={"submitted_gtin": submitted, "matched_dossier_gtin": False},
        limitation="GTIN check digit is valid, but could not be matched against a verified GTIN on file.",
    )
