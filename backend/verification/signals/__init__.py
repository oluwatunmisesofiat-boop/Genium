"""Modular signal evaluators for GENIE AI's Verification Engine."""
from __future__ import annotations

from .batch_expiry import evaluate_batch_expiry_signal
from .gtin import evaluate_gtin_signal
from .nafdac import evaluate_nafdac_signal
from .recalls_and_tells import evaluate_recalls_and_tells_signal
from .security_features import evaluate_security_features_signal
from .spec_ocr import evaluate_spec_ocr_signal
from .visual import MockCVAdapter, evaluate_visual_signal

__all__ = [
    "MockCVAdapter",
    "evaluate_batch_expiry_signal",
    "evaluate_gtin_signal",
    "evaluate_nafdac_signal",
    "evaluate_recalls_and_tells_signal",
    "evaluate_security_features_signal",
    "evaluate_spec_ocr_signal",
    "evaluate_visual_signal",
]
