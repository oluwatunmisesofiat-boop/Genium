"""Deterministic Evidence Gate, numeric grounding, and cross-source conflict validation."""
from __future__ import annotations

from .evidence import (
    EvidenceGateReport,
    build_source_corpus,
    normalize_text_for_match,
    validate_dossier_evidence,
)

__all__ = [
    "EvidenceGateReport",
    "build_source_corpus",
    "normalize_text_for_match",
    "validate_dossier_evidence",
]
