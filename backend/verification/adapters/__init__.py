"""Adapters bridging external CV/OCR pipelines (such as `ml_infra`) with GENIE AI."""
from .ml_infra_adapter import (
    MLInfraBarcodeInput,
    MLInfraOCRInput,
    MLInfraPackComparisonInput,
    MLInfraPackComparatorAdapter,
    MLInfraVerifyPayload,
    scan_input_from_ml_infra,
    to_ml_infra_response,
)

__all__ = [
    "MLInfraBarcodeInput",
    "MLInfraOCRInput",
    "MLInfraPackComparisonInput",
    "MLInfraPackComparatorAdapter",
    "MLInfraVerifyPayload",
    "scan_input_from_ml_infra",
    "to_ml_infra_response",
]
