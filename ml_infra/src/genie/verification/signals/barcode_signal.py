"""Barcode authenticity and mismatch signal computation."""
from typing import Any, Dict, Optional

from genie.verification.signals import SignalResult


class BarcodeSignalEvaluator:
    def evaluate(
        self,
        decoded_barcode: Optional[Dict[str, Any]],
        resolved_product_gtins: list[str] | None = None,
    ) -> SignalResult:
        if not decoded_barcode:
            return SignalResult(
                score=0.5,
                confidence=0.2,
                reasons=["Barcode not detected or unreadable."],
            )

        gtin = decoded_barcode.get("gtin")
        check_valid = decoded_barcode.get("check_digit_valid", True)

        # Hard rule 1: Checksum failure
        if not check_valid:
            return SignalResult(
                score=0.0,
                confidence=1.0,
                hard_invalidation=True,
                invalidation_reason="Invalid GTIN check digit (likely fabricated barcode).",
                reasons=["GTIN check digit checksum failed."],
            )

        # Hard rule 2: Barcode maps to completely different product
        if gtin and resolved_product_gtins and gtin not in resolved_product_gtins:
            return SignalResult(
                score=0.1,
                confidence=0.9,
                hard_invalidation=True,
                invalidation_reason="Barcode GTIN does not match packaging product identity.",
                reasons=[f"Scanned GTIN {gtin} does not match expected product GTINs."],
            )

        return SignalResult(
            score=0.95,
            confidence=0.85,
            reasons=["Valid GTIN matching registered product."],
            evidence={"gtin": gtin, "symbology": decoded_barcode.get("symbology")},
        )
