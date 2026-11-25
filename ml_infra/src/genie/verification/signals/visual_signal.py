"""Visual similarity and anomaly signal computation."""
from typing import List, Optional

from genie.verification.signals import SignalResult


class VisualSignalEvaluator:
    def evaluate(
        self,
        retrieval_cosine_sim: float,
        alignment_inlier_ratio: float,
        patch_anomaly_score: float,
        color_score: Optional[float] = None,
        layout_score: Optional[float] = None,
        inconsistencies: Optional[List[str]] = None,
        reference_available: bool = True,
    ) -> SignalResult:
        if not reference_available:
            reasons = (
                list(inconsistencies)
                if inconsistencies
                else ["No canonical reference pack image available in database for visual comparison."]
            )
            return SignalResult(
                score=0.50,
                confidence=0.10,
                reasons=reasons,
                evidence={
                    "cosine_similarity": retrieval_cosine_sim,
                    "reference_available": False,
                },
            )

        if color_score is not None and layout_score is not None:
            visual_score = (
                0.20 * retrieval_cosine_sim
                + 0.20 * alignment_inlier_ratio
                + 0.25 * color_score
                + 0.25 * layout_score
                + 0.10 * (1.0 - patch_anomaly_score)
            )
            confidence = 0.92
        else:
            visual_score = (
                0.4 * retrieval_cosine_sim
                + 0.3 * alignment_inlier_ratio
                + 0.3 * (1.0 - patch_anomaly_score)
            )
            confidence = 0.85

        visual_score = min(max(visual_score, 0.0), 1.0)

        reasons = list(inconsistencies) if inconsistencies else []
        if patch_anomaly_score > 0.70:
            reasons.append(f"Significant visual packaging anomaly detected (anomaly score: {patch_anomaly_score:.2f}).")
        elif not reasons:
            reasons.append(f"Visual packaging features align with reference packaging (score: {visual_score:.2f}).")

        evidence = {
            "cosine_similarity": retrieval_cosine_sim,
            "inlier_ratio": alignment_inlier_ratio,
            "anomaly_score": patch_anomaly_score,
        }
        if color_score is not None:
            evidence["color_score"] = color_score
        if layout_score is not None:
            evidence["layout_score"] = layout_score

        return SignalResult(
            score=visual_score,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence,
        )
