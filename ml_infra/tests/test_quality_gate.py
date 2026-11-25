"""Unit tests for Image Quality Gating."""
import cv2
import numpy as np
import pytest

from genie.vision.quality import ImageQualityGate


def test_quality_gate_passes_clean_image(clean_synthetic_pack_image):
    gate = ImageQualityGate(min_blur_score=100.0, max_glare_ratio=0.08)
    result = gate.evaluate(clean_synthetic_pack_image)
    assert result.passed is True
    assert result.blur_score >= gate.min_blur_score
    assert result.glare_ratio <= gate.max_glare_ratio
    assert len(result.guidance_codes) == 0
    assert result.latency_ms < 120.0  # Target latency budget check


def test_quality_gate_rejects_blurry_image(blurry_pack_image):
    gate = ImageQualityGate(min_blur_score=120.0)
    result = gate.evaluate(blurry_pack_image)
    assert result.passed is False
    assert "HOLD_STEADY_BLURRY" in result.guidance_codes
    assert result.rejection_reason is not None


def test_quality_gate_rejects_specular_glare():
    # Construct an image with specular glare (V > 245, S < 35) over 15% of pack area
    img = np.ones((512, 512, 3), dtype=np.uint8) * 120
    # Add specular white hotspot
    cv2.circle(img, (256, 256), 110, (255, 255, 255), -1)

    gate = ImageQualityGate(max_glare_ratio=0.08)
    result = gate.evaluate(img)
    assert result.passed is False
    assert "REDUCE_GLARE" in result.guidance_codes
    assert result.glare_ratio > 0.08


def test_quality_gate_flags_tiny_pack():
    # Tiny 50x50 pack on a 600x600 dark background (< 20% area ratio)
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    img[275:325, 275:325] = [200, 200, 200]

    gate = ImageQualityGate(min_pack_area_ratio=0.20)
    result = gate.evaluate(img)
    assert result.passed is False
    assert "MOVE_CLOSER" in result.guidance_codes


def test_quality_gate_panel_guidance():
    img = np.ones((512, 512, 3), dtype=np.uint8) * 180
    gate = ImageQualityGate()
    result = gate.evaluate(img, require_back_panel=True)
    assert "SHOW_BACK_PANEL" in result.guidance_codes
