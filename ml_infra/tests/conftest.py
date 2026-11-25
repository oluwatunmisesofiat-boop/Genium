"""Pytest configuration and test fixtures."""
import pytest
import numpy as np


@pytest.fixture
def clean_synthetic_pack_image() -> np.ndarray:
    """Returns a clean 512x512 mock packaging image with contrast patterns."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    # Add a mock logo and high-contrast text lines
    img[50:150, 50:150] = [20, 120, 30]  # Green logo block
    img[200:240, 50:450] = [10, 10, 10]  # Dark text line
    img[260:300, 50:350] = [10, 10, 10]  # Second text line
    return img


@pytest.fixture
def blurry_pack_image() -> np.ndarray:
    """Returns an artificially blurred image that should fail quality gating."""
    import cv2
    clean = np.ones((512, 512, 3), dtype=np.uint8) * 150
    return cv2.GaussianBlur(clean, (45, 45), 0)
