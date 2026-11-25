"""Unit tests for NVIDIA Nemotron VLM review and triage."""
import json
import numpy as np
import pytest

from genie.vision.ocr import PackagingOCR
from genie.vision.vlm_review import VLMReviewResult, VLMReviewer


def _create_mock_image():
    return np.full((128, 128, 3), 200, dtype=np.uint8)


def test_vlm_reviewer_initialization():
    reviewer = VLMReviewer()
    assert reviewer.model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert "nvidia.com" in reviewer.base_url


def test_vlm_payload_construction():
    reviewer = VLMReviewer(api_key="mock_key")
    img1 = _create_mock_image()
    img2 = _create_mock_image()

    b64_1 = reviewer._encode_image_b64(img1)
    b64_2 = reviewer._encode_image_b64(img2)

    payload = reviewer._build_payload(query_b64=b64_1, ref_b64=b64_2)
    assert payload["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert len(payload["messages"]) == 2
    # Verify both image contents are present
    content = payload["messages"][1]["content"]
    image_items = [item for item in content if item.get("type") == "image_url"]
    assert len(image_items) == 2


def test_vlm_parse_clean_json():
    reviewer = VLMReviewer()
    mock_json = json.dumps({
        "score": 0.94,
        "confidence": 0.88,
        "is_consistent": True,
        "triage_verdict": "Consistent with genuine",
        "detected_nafdac": "04-9821",
        "detected_batch": "B8941",
        "detected_expiry": "12/2027",
        "reasons": ["Offset lithography rosette dots verified", "NAFDAC number crisp"],
        "discrepancies": [],
    })

    result = reviewer._parse_response(mock_json)
    assert result.score == 0.94
    assert result.confidence == 0.88
    assert result.is_consistent is True
    assert result.detected_nafdac == "04-9821"
    assert len(result.reasons) == 2
    assert len(result.discrepancies) == 0


def test_vlm_parse_markdown_fenced_json():
    reviewer = VLMReviewer()
    fenced = (
        "```json\n"
        "{\n"
        '  "score": 0.25,\n'
        '  "confidence": 0.92,\n'
        '  "is_consistent": false,\n'
        '  "triage_verdict": "Suspicious",\n'
        '  "detected_nafdac": "04-9999",\n'
        '  "reasons": ["Font substitution detected in brand logo"],\n'
        '  "discrepancies": ["Brand logo color shift Delta-E > 6.0", "Missing holographic seal"]\n'
        "}\n"
        "```"
    )

    result = reviewer._parse_response(fenced)
    assert result.score == 0.25
    assert result.is_consistent is False
    assert result.triage_verdict == "Suspicious"
    assert len(result.discrepancies) == 2


@pytest.mark.asyncio
async def test_vlm_async_review_mocked(monkeypatch):
    reviewer = VLMReviewer(api_key="mock_key")
    img = _create_mock_image()

    # Mock httpx response
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "score": 0.95,
                        "confidence": 0.90,
                        "is_consistent": True,
                        "triage_verdict": "Consistent with genuine",
                        "detected_nafdac": "A4-1234",
                        "reasons": ["Packaging verified against canonical reference"],
                        "discrepancies": [],
                    })
                }
            }
        ]
    }

    class MockAsyncResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return mock_response

    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, *args, **kwargs):
            return MockAsyncResponse()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: MockAsyncClient())

    result = await reviewer.review_packaging_async(img)
    assert result.score == 0.95
    assert result.detected_nafdac == "A4-1234"
    assert result.is_consistent is True


def test_vlm_fallback_on_unconfigured_key():
    reviewer = VLMReviewer(api_key="")
    img = _create_mock_image()
    res = reviewer.review_packaging(img)
    assert res.triage_verdict == "Cannot verify"
    assert "skipped" in res.reasons[0].lower()


def test_ocr_extract_with_vlm_fallback(monkeypatch):
    ocr = PackagingOCR()
    img = _create_mock_image()

    from genie.vision.vlm_review import VLMReviewer
    monkeypatch.setattr(
        VLMReviewer,
        "review_packaging",
        lambda self, image, reference=None, context=None: VLMReviewResult(
            score=0.90,
            confidence=0.85,
            is_consistent=True,
            detected_nafdac="04-1234",
            detected_batch="BATCH99",
            detected_expiry="11/2026",
            reasons=["VLM read packaging typography"],
        ),
    )

    data = ocr.extract_with_vlm(img)
    assert data.get("nafdac_number") == "04-1234"
    assert data.get("batch_number") == "BATCH99"
    assert data.get("expiry_date") == "11/2026"
