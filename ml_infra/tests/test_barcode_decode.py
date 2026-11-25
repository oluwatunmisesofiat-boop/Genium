"""Unit tests for multi-decoder ensemble and enhancement pipeline."""
import cv2
import numpy as np
import pytest

from genie.barcode.decode import MultiDecoder
from genie.barcode.enhance import enhance_barcode_crop


def test_enhance_barcode_crop_generates_variants():
    crop = np.ones((100, 100, 3), dtype=np.uint8) * 150
    variants = enhance_barcode_crop(crop)
    assert len(variants) >= 7
    for v in variants:
        assert isinstance(v, np.ndarray)


def test_multi_decoder_fallback_ocr():
    img = np.ones((150, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "06151234567891", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    res = MultiDecoder._fallback_ocr_digits(img)
    if res:
        assert res["gtin"] == "06151234567891"
        assert res["check_digit_valid"] is True


def test_format_result_linear_gtin():
    res = MultiDecoder._format_result("6151234567891", "EAN_13", engine="test_engine")
    assert res["gtin"] == "06151234567891"
    assert res["check_digit_valid"] is True
    assert res["decoder_engine"] == "test_engine"
