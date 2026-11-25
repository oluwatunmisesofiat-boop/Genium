"""GENIE AI Multi-Signal Verification Engine & Graceful Degradation (Phase D)."""
from __future__ import annotations

from .engine import VerificationEngine, load_saved_dossiers
from .schemas import (
    CounterfeitTellHit,
    MLInfraVerdict,
    ScanInput,
    SignalResult,
    VerificationResult,
    VerificationVerdict,
    VisualAnomaly,
    VisualMatchInput,
    to_ml_infra_verdict,
)
from .signals import MockCVAdapter

__all__ = [
    "CounterfeitTellHit",
    "MLInfraVerdict",
    "MockCVAdapter",
    "ScanInput",
    "SignalResult",
    "VerificationEngine",
    "VerificationResult",
    "VerificationVerdict",
    "VisualAnomaly",
    "VisualMatchInput",
    "load_saved_dossiers",
    "to_ml_infra_verdict",
]
