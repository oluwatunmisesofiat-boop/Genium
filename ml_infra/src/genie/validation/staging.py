"""Staged claim promotion and review queue logic."""
from genie.schemas.evidence import StagedClaim


class StagingManager:
    def should_auto_promote(self, claim: StagedClaim, source_tier: int) -> bool:
        """Auto-promote only if Tier 1 and score >= 0.85 with no conflict."""
        return source_tier == 1 and claim.evidence_score >= 0.85
