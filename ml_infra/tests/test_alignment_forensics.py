"""Tests for Phase 5: Geometric alignment and region print forensics."""
import cv2
import numpy as np
import pytest

from genie.vision.match import GeometricMatcher
from genie.vision.print_forensics import PrintForensicsAnalyzer


def _create_synthetic_packaging_pattern(size=(512, 512), text="GENIE COARTEM 80/480"):
    """Generates a synthetic front-panel packaging image with rich visual features."""
    img = np.full((size[0], size[1], 3), 245, dtype=np.uint8)

    # Add background colored blocks and banners
    cv2.rectangle(img, (20, 20), (size[1] - 20, 100), (40, 80, 180), -1)  # Deep blue banner
    cv2.rectangle(img, (20, 110), (size[1] - 20, 140), (200, 150, 30), -1)  # Gold stripe

    # Add distinct high-contrast typography
    cv2.putText(img, text, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(img, "NAFDAC REG NO: A4-9821", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "BATCH: B8941EXP: 12/2027", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # Add geometric security emblem / rosette
    for r in range(10, 45, 8):
        cv2.circle(img, (400, 220), r, (80, 30, 160), 2)

    # Add grid structure
    for x in range(40, size[1] - 40, 40):
        cv2.line(img, (x, 320), (x + 20, 420), (100, 100, 100), 1)

    return img


def test_geometric_alignment_identical_images():
    matcher = GeometricMatcher(min_inliers=15, min_inlier_ratio=0.30)
    ref = _create_synthetic_packaging_pattern()
    query = ref.copy()

    res = matcher.match(query, ref)
    assert res.matched is True
    assert res.num_inliers >= 15
    assert res.inlier_ratio >= 0.30
    assert res.reprojection_error < 2.0
    assert res.aligned_image is not None
    assert res.aligned_image.shape == (512, 512, 3)


def test_geometric_alignment_with_affine_perturbation():
    matcher = GeometricMatcher(min_inliers=15, min_inlier_ratio=0.25)
    ref = _create_synthetic_packaging_pattern()

    # Apply small affine rotation and translation
    center = (256, 256)
    M = cv2.getRotationMatrix2D(center, angle=4.0, scale=0.98)
    M[0, 2] += 10.0  # translate x
    M[1, 2] += -8.0  # translate y

    query = cv2.warpAffine(ref, M, (512, 512), borderValue=(245, 245, 245))

    res = matcher.match(query, ref)
    assert res.matched is True
    assert res.num_inliers >= 15
    assert res.reprojection_error < 4.0


def test_geometric_alignment_rejects_divergent_design():
    matcher = GeometricMatcher(min_inliers=25, min_inlier_ratio=0.35)
    ref = _create_synthetic_packaging_pattern(text="GENIE COARTEM 80/480")

    # Divergent image with completely different layout
    unrelated = np.full((512, 512, 3), 50, dtype=np.uint8)
    cv2.circle(unrelated, (256, 256), 180, (200, 200, 200), -1)

    res = matcher.match(unrelated, ref)
    assert res.matched is False
    assert res.inlier_ratio < 0.35


def test_cielab_delta_e_color_fidelity():
    analyzer = PrintForensicsAnalyzer(max_delta_e=5.0)
    ref = _create_synthetic_packaging_pattern()
    query_identical = ref.copy()

    delta_e = analyzer.calculate_cielab_delta_e(query_identical, ref)
    assert delta_e < 1.0  # Identical images have near 0 delta-E

    # Introduce prominent color shift (e.g. counterfeit off-color banner)
    query_altered = ref.copy()
    query_altered[:, :, 0] = np.clip(query_altered[:, :, 0].astype(int) + 50, 0, 255)
    delta_e_altered = analyzer.calculate_cielab_delta_e(query_altered, ref)
    assert delta_e_altered > 5.0


def test_typography_edge_sharpness():
    analyzer = PrintForensicsAnalyzer(min_sharpness=10.0)
    sharp_img = _create_synthetic_packaging_pattern()
    blurry_img = cv2.GaussianBlur(sharp_img, (19, 19), 0)

    sharp_score = analyzer.edge_sharpness(sharp_img)
    blurry_score = analyzer.edge_sharpness(blurry_img)

    assert sharp_score > blurry_score
    assert sharp_score > 10.0
    assert blurry_score < sharp_score * 0.5


def test_regulatory_symbol_verification():
    analyzer = PrintForensicsAnalyzer()
    img = _create_synthetic_packaging_pattern()

    # Extract the circular emblem as symbol template
    symbol_template = img[170:270, 350:450].copy()
    query_region = img[160:280, 340:460].copy()

    sim, passed = analyzer.verify_regulatory_symbol(query_region, symbol_template)
    assert sim >= 0.70
    assert passed is True

    # Corrupted / mismatched query region
    mismatched = np.zeros_like(query_region)
    sim_bad, passed_bad = analyzer.verify_regulatory_symbol(mismatched, symbol_template)
    assert passed_bad is False


def test_forensics_analyzer_pipeline():
    analyzer = PrintForensicsAnalyzer(max_delta_e=5.0, min_sharpness=8.0)
    ref = _create_synthetic_packaging_pattern()
    query = ref.copy()

    result = analyzer.analyze_forensics(
        query_aligned=query,
        reference_aligned=ref,
        logo_bbox=(20, 20, 100, 480),
    )

    assert result.color_pass is True
    assert result.sharpness_pass is True
    assert result.overall_pass is True
    assert len(result.discrepancies) == 0
