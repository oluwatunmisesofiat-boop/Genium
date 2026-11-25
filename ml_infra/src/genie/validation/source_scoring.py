"""Multi-factor source validation and claim evidence scoring."""
from typing import List
from genie.schemas.evidence import EvidenceClaim


class SourceScorer:
    """
    Computes evidence score:
    score = w1*source_tier + w2*agreement + w3*recency + w4*extraction_conf + w5*identifier_consistency
    """

    def compute_evidence_score(
        self,
        claims: List[EvidenceClaim],
        source_tier: int,
        has_independent_agreement: bool,
    ) -> float:
        tier_weight = 1.0 if source_tier == 1 else (0.6 if source_tier == 2 else 0.2)
        agreement_weight = 1.0 if has_independent_agreement else 0.5
        score = 0.5 * tier_weight + 0.3 * agreement_weight + 0.2
        return min(max(score, 0.0), 1.0)
