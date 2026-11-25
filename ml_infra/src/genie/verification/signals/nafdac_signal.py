"""NAFDAC regulatory registration validation signal."""
from typing import Any, Dict, Optional

from genie.verification.signals import SignalResult


class NAFDACSignalEvaluator:
    def evaluate(
        self,
        scanned_nafdac: Optional[str],
        registered_product: Optional[Dict[str, Any]],
        registry_checked: bool = True,
    ) -> SignalResult:
        if not scanned_nafdac:
            return SignalResult(
                score=0.5,
                confidence=0.2,
                reasons=["NAFDAC registration number not located by OCR."],
            )

        if not registered_product:
            if not registry_checked:
                return SignalResult(
                    score=0.5,
                    confidence=0.25,
                    reasons=[
                        f"NAFDAC number {scanned_nafdac} extracted, but no Greenbook registry reference was available to verify it."
                    ],
                )
            return SignalResult(
                score=0.2,
                confidence=0.7,
                reasons=[f"NAFDAC number {scanned_nafdac} not found in official Greenbook registry."],
            )

        # Check registration status
        status = registered_product.get("status", "active").lower()
        if status in ("expired", "revoked", "suspended"):
            return SignalResult(
                score=0.0,
                confidence=1.0,
                hard_invalidation=True,
                invalidation_reason=f"NAFDAC registration {scanned_nafdac} is {status}.",
                reasons=[f"NAFDAC registry reports registration status as {status}."],
            )

        return SignalResult(
            score=0.95,
            confidence=0.95,
            reasons=[f"NAFDAC registration {scanned_nafdac} verified active in Greenbook."],
            evidence={
                "nafdac_number": scanned_nafdac,
                "registered_brand": registered_product.get("brand_name"),
                "manufacturer": registered_product.get("manufacturer"),
            },
        )
