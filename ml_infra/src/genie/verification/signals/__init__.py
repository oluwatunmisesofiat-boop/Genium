"""Modular verification signal calculators."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SignalResult:
    score: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    hard_invalidation: bool = False
    invalidation_reason: Optional[str] = None
