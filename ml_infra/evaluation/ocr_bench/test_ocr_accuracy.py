"""OCR accuracy, regulatory NAFDAC regex parsing, and batch extraction benchmark."""
import cv2
import numpy as np
import pytest

from genie.vision.ocr import PackagingOCR


def test_nafdac_registration_formats_benchmark():
    ocr = PackagingOCR()

    test_cases = [
        # Human medicines
        ("NAFDAC REG NO: A4-1234", "A4-1234"),
        ("NAFDAC NO: B4-98765", "B4-98765"),
        ("NAFDAC REG: 04-2345", "04-2345"),
        ("Reg No: 04-567891", "04-567891"),
        # Food and cosmetics
        ("NAFDAC: 02-1234", "02-1234"),
        # Herbal formulations
        ("NAFDAC REG NO: 04-1234L", "04-1234L"),
        ("NAFDAC: B4-5678L", "B4-5678L"),
    ]

    for text_sample, expected in test_cases:
        extracted = ocr.parse_nafdac_number(text_sample)
        assert extracted == expected, f"Failed on '{text_sample}', expected {expected}, got {extracted}"


def test_batch_and_expiry_extraction_benchmark():
    ocr = PackagingOCR()

    sample_text = """
    NOVARTIS PHARMA
    COARTEM 80/480mg
    NAFDAC REG NO: 04-1234
    BN: B89412A
    MFD: 02/2025
    EXP: 01/2028
    """

    fields = ocr.parse_batch_and_expiry(sample_text)
    assert fields["batch_number"] == "B89412A"
    assert fields["expiry_date"] == "01/2028"
    assert fields["mfg_date"] == "02/2025"


def test_dot_matrix_morphological_enhancement_benchmark():
    ocr = PackagingOCR()
    # Create dotted text simulation
    dotted_img = np.full((64, 128), 245, dtype=np.uint8)
    for x in range(10, 110, 5):
        cv2.circle(dotted_img, (x, 32), 1, 0, -1)

    enhanced = ocr.enhance_dot_matrix(dotted_img)
    assert enhanced is not None
    assert enhanced.shape == dotted_img.shape
    assert enhanced.dtype == np.uint8
