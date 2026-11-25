"""Builds PatchCore memory banks and coreset subsampling from verified genuine references."""
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from genie.common.logging import get_logger

logger = get_logger("build_patchcore")


def extract_patch_features(
    image: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extracts dense localized patch representations from a rectified pack image ($512x512$).

    Computes multi-channel spatial-color and gradient texture descriptors
    per neighborhood patch to form patch embedding tokens.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    h, w = img_rgb.shape[:2]

    # Compute color and gradient maps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    patch_features = []
    patch_coords = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop_rgb = img_rgb[y : y + patch_size, x : x + patch_size]
            crop_mag = mag[y : y + patch_size, x : x + patch_size]

            # Mean, standard deviation, and median across RGB channels + gradient magnitude
            rgb_mean = np.mean(crop_rgb, axis=(0, 1)) / 255.0
            rgb_std = np.std(crop_rgb, axis=(0, 1)) / 255.0
            mag_mean = np.mean(crop_mag) / 255.0
            mag_std = np.std(crop_mag) / 255.0

            # 8-bin spatial orientation histogram
            hist, _ = np.histogram(crop_mag, bins=8, range=(0, 255), density=True)

            feat = np.concatenate([rgb_mean, rgb_std, [mag_mean, mag_std], hist]).astype(np.float32)
            # Normalize patch token
            norm = np.linalg.norm(feat) + 1e-7
            feat = feat / norm

            patch_features.append(feat)
            patch_coords.append((y, x))

    return np.array(patch_features, dtype=np.float32), patch_coords


def greedy_coreset_subsampling(
    features: np.ndarray,
    subsample_ratio: float = 0.10,
    min_samples: int = 50,
) -> np.ndarray:
    """Greedy Minimax Facility Location (k-center) coreset subsampling.

    Iteratively selects patch embeddings that maximize the minimum distance
    to already selected coreset points, compressing the memory bank by ~90%
    while preserving full coverage of normal feature variations.
    """
    n_points = features.shape[0]
    n_select = max(min_samples, int(n_points * subsample_ratio))
    n_select = min(n_select, n_points)

    if n_select == n_points:
        return features

    # Start with point furthest from origin
    norms = np.linalg.norm(features, axis=1)
    first_idx = int(np.argmax(norms))

    selected_indices = [first_idx]
    # Initialize min distances to first selected point
    min_distances = np.linalg.norm(features - features[first_idx], axis=1)

    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)

        # Update min distances with the newly selected point
        dist_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_distances = np.minimum(min_distances, dist_new)

    return features[selected_indices]


class PatchCoreMemoryBank:
    """Serializes and manages coreset memory banks for genuine reference packs."""

    def __init__(self, coreset: np.ndarray, patch_size: int = 32, stride: int = 16):
        self.coreset = coreset  # shape: (N_coreset, D)
        self.patch_size = patch_size
        self.stride = stride

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coreset=self.coreset,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        logger.info(f"Saved PatchCore memory bank to {path} ({self.coreset.shape[0]} points)")

    @classmethod
    def load(cls, filepath: str | Path) -> "PatchCoreMemoryBank":
        data = np.load(filepath)
        return cls(
            coreset=data["coreset"],
            patch_size=int(data["patch_size"]),
            stride=int(data["stride"]),
        )


def build_memory_bank_for_images(
    reference_images: List[np.ndarray],
    subsample_ratio: float = 0.10,
    patch_size: int = 32,
    stride: int = 16,
) -> PatchCoreMemoryBank:
    """Builds a compressed PatchCore memory bank from a set of genuine reference images."""
    all_patch_features = []
    for img in reference_images:
        feats, _ = extract_patch_features(img, patch_size=patch_size, stride=stride)
        all_patch_features.append(feats)

    merged_features = np.vstack(all_patch_features)
    logger.info(f"Extracted {merged_features.shape[0]} total patch features across references")

    coreset = greedy_coreset_subsampling(merged_features, subsample_ratio=subsample_ratio)
    logger.info(f"Subsampled coreset down to {coreset.shape[0]} points ({subsample_ratio * 100:.1f}%)")

    return PatchCoreMemoryBank(coreset=coreset, patch_size=patch_size, stride=stride)
