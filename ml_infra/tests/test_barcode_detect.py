"""Unit tests for barcode detection, localization, and orientation estimation."""
import cv2
import numpy as np
import pytest

from genie.barcode.detect import BarcodeDetector


def test_barcode_detector_localizes_vertical_stripes():
    detector = BarcodeDetector()

    # Create 512x512 canvas with a high-contrast 1D barcode block
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 230

    # Draw vertical barcode stripes in a localized region [x: 100..300, y: 200..320]
    for x in range(100, 300, 4):
        cv2.line(canvas, (x, 200), (x, 320), (10, 10, 10), 2)

    regions = detector.detect_regions(canvas)
    assert len(regions) >= 1

    primary = regions[0]
    bx, by, bw, bh = primary.bounding_box
    assert bw > 50
    assert bh > 30
    assert primary.confidence > 0.8
    assert primary.orientation_angle in (0, 90)


def test_barcode_detector_fallback_on_blank():
    detector = BarcodeDetector()
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 200

    regions = detector.detect_regions(blank)
    assert len(regions) == 1
    # Fallback returns full image crop
    assert regions[0].bounding_box == (0, 0, 400, 400)
    assert regions[0].confidence == 0.50
