"""End-to-End integration test scenarios verifying the stateless scan verification pipeline."""
import base64
import cv2
import numpy as np
import pytest

from genie.schemas.scan import ReferenceCandidateInput, ScanImage, ScanRequest
from genie.schemas.verdict import VerdictState
from genie.storage.vector_store import InMemoryVectorStore
from genie.verification.pipeline import VerificationPipeline


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


def _generate_synthetic_pack_b64(
    text="COARTEM 80/480",
    nafdac="04-1234",
    batch="B1001",
    expiry="12/2027",
) -> str:
    """Creates a synthetic clear packaging image and returns base64 JPEG bytes."""
    img = np.full((512, 512, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (492, 120), (40, 90, 180), -1)  # Blue banner
    cv2.putText(img, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.putText(img, f"NAFDAC REG NO: {nafdac}", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, f"BATCH: {batch} EXP: {expiry}", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    cv2.circle(img, (400, 240), 40, (100, 30, 180), 3)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode("utf-8")


@pytest.mark.asyncio
async def test_e2e_scenario_1_genuine_product(
    vector_store: InMemoryVectorStore,
    monkeypatch,
):
    """Scenario 1: Verified genuine scan with Backend-supplied candidate metadata."""
    pack_b64 = _generate_synthetic_pack_b64(text="COARTEM 80/480", nafdac="04-1234", batch="B1001")

    pipeline = VerificationPipeline(vector_store)
    from genie.barcode.decode import MultiDecoder
    monkeypatch.setattr(
        MultiDecoder,
        "decode",
        lambda img: {"gtin": "06151234567891", "check_digit_valid": True, "batch": "B1001", "expiry": "2027-12-31"},
    )

    request = ScanRequest(
        images=[ScanImage(image_bytes_base64=pack_b64, face_hint="front")],
        manual_gtin="06151234567891",
        manual_nafdac="04-1234",
        reference_candidate=ReferenceCandidateInput(
            brand_name="Coartem 80/480",
            manufacturer="Novartis Pharma",
            primary_nafdac_number="04-1234",
            gtin_list=["06151234567891"],
            version_label="2025_Design_v1",
            is_legacy_version=False,
        ),
    )

    response = await pipeline.verify_scan(request)

    assert response.verdict == VerdictState.CONSISTENT_WITH_GENUINE
    assert response.confidence >= 0.85
    assert response.matched_product_name == "Coartem 80/480"
    assert response.matched_version_label == "2025_Design_v1"
    assert response.is_legacy_version is False
    assert "Consistent with genuine" in response.explanation
    assert response.execution_time_ms < 4000.0


@pytest.mark.asyncio
async def test_e2e_scenario_2_counterfeit_batch_discrepancy(
    vector_store: InMemoryVectorStore,
    monkeypatch,
):
    """Scenario 2: Counterfeit product with batch number contradiction between printed text and barcode."""
    pack_b64 = _generate_synthetic_pack_b64(text="COARTEM 80/480", batch="B9999")

    pipeline = VerificationPipeline(vector_store)
    from genie.barcode.decode import MultiDecoder
    from genie.vision.ocr import OCRExtractionResult
    monkeypatch.setattr(
        MultiDecoder,
        "decode",
        lambda img: {"gtin": "06151234567891", "batch": "LOT_ORIGINAL_123", "check_digit_valid": True},
    )
    monkeypatch.setattr(
        pipeline.ocr,
        "extract_text",
        lambda img: OCRExtractionResult(
            full_text="BN: B9999 NAFDAC: 04-1234",
            raw_text_lines=["BN: B9999", "NAFDAC: 04-1234"],
            batch_number="B9999",
            nafdac_number="04-1234",
        ),
    )

    request = ScanRequest(
        images=[ScanImage(image_bytes_base64=pack_b64, face_hint="front")],
        manual_gtin="06151234567891",
        manual_nafdac="04-1234",
        reference_candidate=ReferenceCandidateInput(
            brand_name="Coartem 80/480",
            manufacturer="Novartis Pharma",
            primary_nafdac_number="04-1234",
            gtin_list=["06151234567891"],
            version_label="2025_Design_v1",
        ),
    )

    response = await pipeline.verify_scan(request)

    assert response.verdict == VerdictState.SUSPICIOUS
    assert response.confidence >= 0.90
    assert "Suspicious product detected" in response.explanation
