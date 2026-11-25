"""Tests for Phase 6: PatchCore visual anomaly detection and coreset subsampling."""
import cv2
import numpy as np
import pytest

from genie.vision.anomaly import PatchCoreAnomalyDetector
from training.anomaly.build_patchcore import (
    build_memory_bank_for_images,
    extract_patch_features,
    greedy_coreset_subsampling,
)


def _create_genuine_pack_face(size=(256, 256)):
    """Creates a synthetic front-panel packaging face."""
    img = np.full((size[0], size[1], 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (size[1] - 20, 70), (30, 90, 180), -1)  # Blue banner
    cv2.putText(img, "COARTEM 80/480", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.circle(img, (200, 150), 30, (50, 160, 50), -1)  # Green circular badge
    return img


def test_patch_feature_extraction():
    img = _create_genuine_pack_face(size=(256, 256))
    feats, coords = extract_patch_features(img, patch_size=32, stride=16)

    assert len(feats) > 0
    assert len(feats) == len(coords)
    # Unit normalization check
    norms = np.linalg.norm(feats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_greedy_coreset_subsampling():
    # 200 random vectors of dim 16
    np.random.seed(42)
    features = np.random.randn(200, 16).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    coreset = greedy_coreset_subsampling(features, subsample_ratio=0.10, min_samples=20)
    assert coreset.shape[0] == 20
    assert coreset.shape[1] == 16


def test_patchcore_genuine_match():
    ref = _create_genuine_pack_face()
    memory_bank = build_memory_bank_for_images([ref], subsample_ratio=0.30, patch_size=32, stride=16)

    detector = PatchCoreAnomalyDetector(memory_bank=memory_bank, anomaly_threshold=0.45)
    query_identical = ref.copy()

    res = detector.evaluate(query_identical)
    assert res.is_anomalous is False
    assert res.anomaly_score < 0.30
    assert res.heatmap_bytes is not None


def test_patchcore_detects_tampered_patch():
    ref = _create_genuine_pack_face()
    memory_bank = build_memory_bank_for_images([ref], subsample_ratio=0.30, patch_size=32, stride=16)

    detector = PatchCoreAnomalyDetector(memory_bank=memory_bank, anomaly_threshold=0.45)

    # Tampered query: alter green badge into bright red and add black scratch
    tampered = ref.copy()
    cv2.circle(tampered, (200, 150), 30, (0, 0, 240), -1)  # Red counterfeit badge
    cv2.line(tampered, (40, 120), (120, 120), (0, 0, 0), 5)  # Altered text line

    res = detector.evaluate(tampered)
    assert res.is_anomalous is True
    assert res.anomaly_score > 0.40
    assert len(res.spatial_anomaly_coords) > 0
    assert res.heatmap_bytes is not None
