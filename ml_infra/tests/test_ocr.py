"""Unit tests for Packaging OCR, text detection, and regulatory field extraction."""
import cv2
import numpy as np
import pytest

from genie.vision.ocr import PackagingOCR


def test_detect_text_line_candidates():
    ocr = PackagingOCR()

    # Create 400x500 synthetic image with two text line bands
    canvas = np.ones((400, 500, 3), dtype=np.uint8) * 230
    cv2.putText(canvas, "COARTEM 80/480MG", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(canvas, "BATCH: AB1234 EXP: 12/2027", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    candidates = ocr.detect_text_line_candidates(canvas)
    assert len(candidates) >= 2
    for c in candidates:
        bx, by, bw, bh = c.bounding_box
        assert bw > 25
        assert bh > 8


def test_enhance_dot_matrix_inkjet():
    ocr = PackagingOCR()
    gray = np.ones((100, 200), dtype=np.uint8) * 150
    enhanced = ocr.enhance_dot_matrix_inkjet(gray)
    assert enhanced.shape == (100, 200)
    assert enhanced.dtype == np.uint8


def test_nafdac_regex_parsing():
    ocr = PackagingOCR()

    # Drug format A4-6238 with various prefixes
    t1 = "MEDICINE BOX NAFDAC REG NO: A4-6238 KEEP OUT OF REACH"
    assert ocr._parse_nafdac_number(t1) == "A4-6238"

    # En-dash variation: A4–6238
    t2 = "NAFDAC Reg. No. A4–6238"
    assert ocr._parse_nafdac_number(t2) == "A4-6238"

    # Numeric format: 04-1234
    t3 = "Manufactured in Nigeria NAFDAC: 04-1234"
    assert ocr._parse_nafdac_number(t3) == "04-1234"

    # B-prefix format: B4-5555
    t4 = "NRN: B4-5555"
    assert ocr._parse_nafdac_number(t4) == "B4-5555"


def test_batch_and_expiry_parsing():
    ocr = PackagingOCR()

    text = "Coartem Tablets BATCH: AB1234 EXP: 12/2027 MFD: 01/2025"
    batch = ocr._parse_batch_code(text)
    expiry = ocr._parse_expiry_date(text)
    mfg = ocr._parse_mfg_date(text)

    assert batch == "AB1234"
    assert expiry == "12/2027"
    assert mfg == "01/2025"


def test_batch_cross_check_consistency():
    ocr = PackagingOCR()

    # Case 1: Matching batch
    passed, reason = ocr.cross_check_batch_and_expiry(
        ocr_batch="AB1234",
        ocr_expiry="12/2027",
        barcode_batch="AB1234",
        barcode_expiry="2027-12-31",
    )
    assert passed is True
    assert reason is None

    # Case 2: Batch mismatch (critical counterfeit indicator)
    passed, reason = ocr.cross_check_batch_and_expiry(
        ocr_batch="AB1234",
        ocr_expiry="12/2027",
        barcode_batch="ZZ9999",
        barcode_expiry="2027-12-31",
    )
    assert passed is False
    assert "Batch mismatch" in reason


def test_extract_text_mock_pipeline():
    ocr = PackagingOCR()
    canvas = np.ones((200, 400, 3), dtype=np.uint8) * 200

    # Test execution without crashing and verify latency reporting
    res = ocr.extract_text(canvas, barcode_batch="AB1234")
    assert isinstance(res.raw_text_lines, list)
    assert res.latency_ms >= 0.0
