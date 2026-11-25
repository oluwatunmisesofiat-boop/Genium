"""Historical packaging version match signal."""
from typing import Optional

from genie.verification.signals import SignalResult


class HistorySignalEvaluator:
    """Evaluates whether packaging matches current active specs or legitimate historical revisions."""

    def evaluate(
        self,
        is_legacy_match: bool,
        version_label: Optional[str] = None,
        transition_active: bool = True,
        resolved: bool = True,
    ) -> SignalResult:
        if not resolved:
            return SignalResult(
                score=0.50,
                confidence=0.10,
                reasons=["Product identity unresolved; packaging version history cannot be verified."],
                evidence={"legacy_version": None, "transition_valid": None},
            )

        if not is_legacy_match:
            return SignalResult(
                score=1.0,
                confidence=0.95,
                reasons=["Matches current active packaging design."],
                evidence={"legacy_version": False, "transition_valid": True},
            )

        if transition_active:
            return SignalResult(
                score=0.95,
                confidence=0.90,
                reasons=[
                    f"Consistent with legitimate legacy packaging design: '{version_label}' "
                    f"(verified within legal shelf-life transition window)."
                ],
                evidence={"legacy_version": version_label, "transition_valid": True},
            )

        # Beyond allowed transition window
        return SignalResult(
            score=0.35,
            confidence=0.85,
            reasons=[
                f"Matches superseded packaging ('{version_label}') that has exceeded its permitted "
                f"shelf-life transition window. This batch may be counterfeit or expired stock."
            ],
            evidence={"legacy_version": version_label, "transition_valid": False},
        )
