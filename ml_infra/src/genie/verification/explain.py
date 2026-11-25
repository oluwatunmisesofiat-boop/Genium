"""Explainability engine compiling structured audit reasons into consumer feedback."""
from typing import List, Optional

from genie.schemas.verdict import VerdictState


class ExplainabilityEngine:
    """Compiles verified evidence into clear, legally compliant consumer explanations.

    Compliance Mandate:
    - Never issue absolute authenticity guarantees ("100% authentic", "guaranteed genuine").
    - Always use calibrated phrasing ("Consistent with genuine", "Suspicious product detected", "Cannot verify").
    - Cite exact regulatory and batch evidence in the explanation.
    """

    FORBIDDEN_PHRASES = [
        "100% authentic",
        "guaranteed authentic",
        "100% genuine",
        "guaranteed genuine",
        "totally real",
    ]

    @classmethod
    def assemble_explanation(
        cls,
        verdict: VerdictState,
        product_name: Optional[str],
        version_label: Optional[str],
        is_legacy: bool,
        reasons: List[str],
        heatmap_url: Optional[str] = None,
    ) -> str:
        prod_title = product_name or "scanned product"
        evidence_summary = "; ".join(reasons[:2]) if reasons else "No anomalous discrepancies detected"

        if verdict == VerdictState.CONSISTENT_WITH_GENUINE:
            if is_legacy:
                text = (
                    f"Consistent with legitimate genuine packaging for {prod_title} "
                    f"({version_label}). This is an authentic previous packaging design that remains valid "
                    f"within the permitted shelf-life transition window. Evidence: {evidence_summary}."
                )
            else:
                text = (
                    f"Consistent with genuine {prod_title}. Packaging, regulatory credentials, "
                    f"and barcode specifications match official verified references. Evidence: {evidence_summary}."
                )

        elif verdict == VerdictState.SUSPICIOUS:
            primary_reason = reasons[0] if reasons else "Multiple structural packaging discrepancies detected"
            text = (
                f"Suspicious product detected for {prod_title}. We strongly recommend not consuming this item. "
                f"Reason: {primary_reason}. Please report this scan to NAFDAC enforcement."
            )

        else:
            # Cannot verify
            guidance = reasons[0] if reasons else "Insufficient image clarity or unindexed packaging"
            text = (
                f"Cannot verify {prod_title} at this time. {guidance}. "
                f"Please ensure packaging is well-lit, glare-free, and re-scan the primary face."
            )

        if heatmap_url and verdict == VerdictState.SUSPICIOUS:
            text += f" Tampering heatmap available at: {heatmap_url}."

        # Safety sanity check
        for phrase in cls.FORBIDDEN_PHRASES:
            if phrase in text.lower():
                raise ValueError(f"Compliance violation: Forbidden phrase '{phrase}' detected in explanation")

        return text
