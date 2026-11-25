"""Barcode and GS1 evaluation benchmark."""
import cv2
import numpy as np
import pytest

from genie.barcode.detect import BarcodeDetector
from genie.barcode.enhance import BarcodeEnhancer
from genie.barcode.gs1_parser import GS1PayloadParser
from genie.barcode.gtin import (
    calculate_gs1_check_digit,
    inspect_gs1_prefix,
    normalize_to_gtin14,
    validate_gtin_check_digit,
)


def test_gtin_check_digit_benchmark():
    # Valid check digit for '0615123456789' is 1
    assert validate_gtin_check_digit("06151234567891") is True
    assert validate_gtin_check_digit("06151234567894") is False
    assert validate_gtin_check_digit("5012345678900") is True


def test_gs1_prefix_nigeria_hint_benchmark():
    prefix, country = inspect_gs1_prefix("06151234567891")
    assert country == "GS1 Nigeria"
    assert prefix == "615"

    prefix_raw, country_raw = inspect_gs1_prefix("6151234567891")
    assert country_raw == "GS1 Nigeria"

    prefix_other, country_other = inspect_gs1_prefix("5012345678900")
    assert country_other != "GS1 Nigeria"


def test_barcode_enhancement_and_deskew_benchmark():
    # Create synthetic barcode bars
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    for x in range(20, 180, 4):
        cv2.line(crop, (x, 15), (x, 85), (0, 0, 0), 2)

    variants = BarcodeEnhancer.generate_enhanced_variants(crop)
    # Must generate 4 rotations x multiple filters >= 8 variants
    assert len(variants) >= 8
    for var in variants:
        assert var.size > 0
        assert len(var.shape) in (2, 3)


def test_gs1_application_identifier_parsing_benchmark():
    parser = GS1PayloadParser()

    # Bracketed standard GS1 DataMatrix test payload
    raw_payload = "(01)06151234567891(17)271231(10)LOT45678(21)SER998877"
    parsed = parser.parse(raw_payload)

    assert parsed["gtin"] == "06151234567891"
    assert parsed["expiry"] == "2027-12-31"
    assert parsed["batch"] == "LOT45678"
    assert parsed["serial"] == "SER998877"
    assert parsed["check_digit_valid"] is True
    assert parsed["gs1_prefix_country"] == "GS1 Nigeria"
