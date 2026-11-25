"""Comprehensive unit tests for Calibrated Signal Fusion and Explainability Engine."""
import pytest
from genie.schemas.verdict import VerdictState
from genie.verification.explain import ExplainabilityEngine
from genie.verification.fusion import CalibratedFusionEngine
from genie.verification.signals import SignalResult


def test_fusion_hard_rule_1_invalid_gtin_checksum():
    engine = CalibratedFusionEngine()
    signals = {
        "barcode": SignalResult(
            score=0.0,
            confidence=1.0,
            hard_invalidation=True,
            invalidation_reason="Invalid GTIN check digit checksum failed",
        ),
        "visual": SignalResult(score=0.98, confidence=0.95),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.SUSPICIOUS
    assert conf > 0.95
    assert "Invalid GTIN check digit" in reasons[0]


def test_fusion_hard_rule_2_gtin_product_mismatch():
    engine = CalibratedFusionEngine()
    signals = {
        "barcode": SignalResult(
            score=0.0,
            confidence=1.0,
            hard_invalidation=True,
            invalidation_reason="Barcode GTIN maps to a completely different product entity than visual identification",
        ),
        "visual": SignalResult(score=0.90, confidence=0.85),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.SUSPICIOUS
    assert "completely different product" in reasons[0]


def test_fusion_hard_rule_3_revoked_nafdac_license():
    engine = CalibratedFusionEngine()
    signals = {
        "nafdac": SignalResult(
            score=0.0,
            confidence=1.0,
            hard_invalidation=True,
            invalidation_reason="NAFDAC registration status is Revoked/Suspended",
        ),
        "barcode": SignalResult(score=0.95, confidence=0.9),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.SUSPICIOUS
    assert "Revoked/Suspended" in reasons[0]


def test_fusion_hard_rule_4_batch_number_contradiction():
    engine = CalibratedFusionEngine()
    signals = {
        "batch": SignalResult(
            score=0.0,
            confidence=1.0,
            hard_invalidation=True,
            invalidation_reason="Printed batch number directly contradicts barcode 2D matrix payload batch",
        ),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.SUSPICIOUS
    assert "contradicts barcode" in reasons[0]


def test_fusion_hard_rule_5_serial_duplicate_replay():
    engine = CalibratedFusionEngine()
    signals = {
        "barcode": SignalResult(
            score=0.0,
            confidence=1.0,
            hard_invalidation=True,
            invalidation_reason="Serial number (21) scan frequency exceeds duplicate replay threshold (suspected clone)",
        ),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.SUSPICIOUS
    assert "duplicate replay threshold" in reasons[0]


def test_fusion_genuine_verdict_platt_scaling():
    engine = CalibratedFusionEngine(genuine_threshold=0.85)
    signals = {
        "barcode": SignalResult(score=0.98, confidence=0.95, reasons=["Valid GTIN-14 checksum"]),
        "nafdac": SignalResult(score=0.95, confidence=0.95, reasons=["Active Greenbook registration"]),
        "batch": SignalResult(score=0.92, confidence=0.90, reasons=["Batch and expiry cross-checked"]),
        "visual": SignalResult(score=0.94, confidence=0.85, reasons=["Feature alignment confirmed"]),
        "history": SignalResult(score=1.0, confidence=0.90, reasons=["Matches active packaging design"]),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals, category="pharmaceuticals_antimalarials")
    assert verdict == VerdictState.CONSISTENT_WITH_GENUINE
    assert conf >= 0.85
    assert breakdown.barcode_score == 0.98
    assert breakdown.nafdac_score == 0.95


def test_fusion_insufficient_signals():
    engine = CalibratedFusionEngine()
    # Missing all major signals, very low total confidence (< 0.25)
    signals = {
        "barcode": SignalResult(score=0.5, confidence=0.1),
    }

    verdict, conf, breakdown, reasons = engine.fuse(signals)
    assert verdict == VerdictState.CANNOT_VERIFY
    assert "Insufficient verifiable signals" in reasons[0]


def test_explainability_compliance_phrasing():
    # 1. Genuine
    text_gen = ExplainabilityEngine.assemble_explanation(
        verdict=VerdictState.CONSISTENT_WITH_GENUINE,
        product_name="Coartem 80/480",
        version_label="2025_v1",
        is_legacy=False,
        reasons=["Valid GTIN", "Active NAFDAC"],
    )
    assert "Consistent with genuine" in text_gen
    assert "100% authentic" not in text_gen.lower()

    # 2. Suspicious
    text_sus = ExplainabilityEngine.assemble_explanation(
        verdict=VerdictState.SUSPICIOUS,
        product_name="Coartem 80/480",
        version_label="2025_v1",
        is_legacy=False,
        reasons=["Batch number contradiction detected"],
        heatmap_url="https://s3.aws.com/heatmap.jpg",
    )
    assert "Suspicious product detected" in text_sus
    assert "heatmap.jpg" in text_sus

    # 3. Forbidden phrases trigger compliance error
    with pytest.raises(ValueError, match="Compliance violation"):
        # Artificially inject forbidden phrase through mock
        ExplainabilityEngine.FORBIDDEN_PHRASES.append("illegal_phrase")
        try:
            ExplainabilityEngine.assemble_explanation(
                verdict=VerdictState.CONSISTENT_WITH_GENUINE,
                product_name="illegal_phrase Product",
                version_label=None,
                is_legacy=False,
                reasons=[],
            )
        finally:
            ExplainabilityEngine.FORBIDDEN_PHRASES.remove("illegal_phrase")
