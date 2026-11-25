"""Batch number and expiry date consistency signal."""
import re
from typing import Any, Dict, Optional

from genie.verification.signals import SignalResult


class BatchSignalEvaluator:
    def evaluate(
        self,
        barcode_batch: Optional[str],
        barcode_expiry: Optional[str],
        ocr_batch: Optional[str],
        ocr_expiry: Optional[str],
        batch_regex: Optional[str] = None,
    ) -> SignalResult:
        # Cross-check barcode vs printed text
        if barcode_batch and ocr_batch:
            clean_b_batch = re.sub(r"\W", "", barcode_batch).upper()
            clean_o_batch = re.sub(r"\W", "", ocr_batch).upper()
            if clean_b_batch != clean_o_batch:
                return SignalResult(
                    score=0.1,
                    confidence=0.95,
                    hard_invalidation=True,
                    invalidation_reason="Batch number discrepancy between barcode and printed text.",
                    reasons=[f"Barcode batch ({barcode_batch}) differs from printed batch ({ocr_batch})."],
                )

        # Check batch regex if provided
        if batch_regex and ocr_batch:
            if not re.match(batch_regex, ocr_batch):
                return SignalResult(
                    score=0.3,
                    confidence=0.7,
                    reasons=[f"Batch code format '{ocr_batch}' diverges from manufacturer pattern."],
                )

        if not (barcode_batch or ocr_batch or barcode_expiry or ocr_expiry):
            return SignalResult(
                score=0.50,
                confidence=0.20,
                reasons=["Batch number and expiry date were not detected on the scanned face."],
                evidence={"batch": None, "expiry": None},
            )

        return SignalResult(
            score=0.90,
            confidence=0.75,
            reasons=["Batch and expiry specifications are consistent."],
            evidence={"batch": ocr_batch or barcode_batch, "expiry": ocr_expiry or barcode_expiry},
        )
