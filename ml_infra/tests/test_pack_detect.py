"""Unit tests for packaging face localization, homography rectification, and cylindrical unrolling."""
import cv2
import numpy as np
import pytest

from genie.vision.pack_detect import PackDetectorRectifier


def test_pack_detector_planar_rectification():
    rectifier = PackDetectorRectifier(target_size=(512, 512))

    # Construct a tilted rectangular carton on a dark background
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    pts = np.array([[120, 100], [480, 140], [440, 520], [80, 470]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (220, 220, 220))

    result = rectifier.rectify(img)

    assert result.rectified_image.shape == (512, 512, 3)
    assert result.homography_matrix is not None
    assert result.homography_matrix.shape == (3, 3)
    assert result.corners is not None
    assert len(result.corners) == 4
    assert result.reprojection_error < 3.0  # Exit metric: error < 3.0px
    assert result.latency_ms < 180.0       # Target latency budget check


def test_pack_detector_cylindrical_unrolling():
    rectifier = PackDetectorRectifier(target_size=(512, 512))

    # Create mock bottle image
    img = np.ones((600, 400, 3), dtype=np.uint8) * 140
    cv2.putText(img, "AMOXICILLIN SYRUP", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)

    result = rectifier.rectify(img, is_cylindrical=True)

    assert result.rectified_image.shape == (512, 512, 3)
    assert result.is_cylindrical is True
    assert result.latency_ms < 180.0
