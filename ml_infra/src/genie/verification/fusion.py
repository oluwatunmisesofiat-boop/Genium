"""Calibrated multi-signal fusion combining deterministic hard invalidation rules and Platt-scaled weights."""
import math
from typing import Dict, List, Optional, Tuple

from genie.schemas.verdict import SignalBreakdown, VerdictState
from genie.verification.signals import SignalResult


class CalibratedFusionEngine:
    """Combines modular signals into a calibrated 3-state verdict with compliance guards."""

    # Default thresholds by product category
    CATEGORY_THRESHOLDS = {
        "pharmaceuticals_antimalarials": {"genuine": 0.90, "suspicious": 0.40},
        "pharmaceuticals_antibiotics": {"genuine": 0.90, "suspicious": 0.40},
        "pharmaceuticals_analgesics": {"genuine": 0.88, "suspicious": 0.38},
        "fmcg_foods": {"genuine": 0.85, "suspicious": 0.35},
        "default": {"genuine": 0.88, "suspicious": 0.38},
    }

    def __init__(
        self,
        genuine_threshold: Optional[float] = None,
        suspicious_threshold: Optional[float] = None,
        platt_scale: float = 6.0,
        platt_bias: float = -2.0,
    ):
        self.override_genuine = genuine_threshold
        self.override_suspicious = suspicious_threshold
        self.platt_scale = platt_scale
        self.platt_bias = platt_bias

        # Default weights
        self.weights = {
            "barcode": 0.25,
            "nafdac": 0.30,
            "batch": 0.15,
            "visual": 0.20,
            "history": 0.10,
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Standard logistic function bounded to [0.0, 1.0]."""
        return 1.0 / (1.0 + math.exp(-max(min(x, 15.0), -15.0)))

    def fuse(
        self,
        signals: Dict[str, SignalResult],
        category: str = "pharmaceuticals_antimalarials",
    ) -> Tuple[VerdictState, float, SignalBreakdown, List[str]]:
        """Fuses independent signals, enforces 5 hard rules, and computes Platt calibrated probability."""
        reasons: List[str] = []
        breakdown = SignalBreakdown()

        # Step 1: Enforce Deterministic Hard Invalidation Rules
        for name, sig in signals.items():
            if sig.hard_invalidation:
                inv_msg = sig.invalidation_reason or f"Hard invalidation triggered by {name} signal."
                reasons.append(inv_msg)
                # Hard invalidation triggers immediate Suspicious verdict with high confidence
                return (
                    VerdictState.SUSPICIOUS,
                    0.99,
                    breakdown,
                    reasons,
                )

        # Step 2: Weighted Calibrated Combination
        total_info = 0.0
        weighted_score = 0.0

        for name, sig in signals.items():
            w = self.weights.get(name, 0.1)
            # Effective signal influence scaled by confidence
            eff_w = w * sig.confidence
            weighted_score += sig.score * eff_w
            total_info += eff_w
            reasons.extend(sig.reasons)

            # Populate breakdown fields
            if name == "barcode":
                breakdown.barcode_score = sig.score
            elif name == "nafdac":
                breakdown.nafdac_score = sig.score
            elif name == "batch":
                breakdown.batch_consistency_score = sig.score
            elif name == "visual":
                breakdown.visual_similarity_score = sig.score
            elif name == "history":
                breakdown.history_match_score = sig.score

        breakdown.reasons = reasons

        # Missing Signal Policy: If total available verifiable information < 0.25
        if total_info < 0.25:
            return (
                VerdictState.CANNOT_VERIFY,
                0.25,
                breakdown,
                ["Insufficient verifiable signals detected from uploaded scan."],
            )

        # Raw normalized consensus score in [0.0, 1.0]
        raw_score = weighted_score / total_info

        # Step 3: Platt Scaling (calibrated logistic transformation)
        # Centered around raw_score = 0.5 -> z = 0 -> calibrated_p = 0.5
        z = (raw_score - 0.5) * self.platt_scale
        calibrated_p = self._sigmoid(z)

        # Determine threshold cutoffs for category
        thresh = self.CATEGORY_THRESHOLDS.get(category, self.CATEGORY_THRESHOLDS["default"])
        gen_th = self.override_genuine or thresh["genuine"]
        sus_th = self.override_suspicious or thresh["suspicious"]

        # Step 4: Map Calibrated Probability to 3-State Verdict
        if calibrated_p >= gen_th:
            verdict = VerdictState.CONSISTENT_WITH_GENUINE
            confidence = calibrated_p
        elif calibrated_p <= sus_th:
            verdict = VerdictState.SUSPICIOUS
            confidence = 1.0 - calibrated_p
        else:
            verdict = VerdictState.CANNOT_VERIFY
            confidence = calibrated_p

        return verdict, float(confidence), breakdown, reasons
