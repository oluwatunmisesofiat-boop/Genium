"""PatchCore patch-level anomaly detection, neighborhood distance scoring, and heatmap rendering."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from training.anomaly.build_patchcore import PatchCoreMemoryBank, extract_patch_features


@dataclass
class AnomalyResult:
    anomaly_score: float  # [0.0, 1.0] calibrated anomaly score
    is_anomalous: bool
    max_patch_distance: float
    mean_top_k_distance: float
    heatmap_bytes: Optional[bytes] = None
    spatial_anomaly_coords: List[Tuple[int, int]] = None  # [(y, x)] bounding locations of defect patches


class PatchCoreAnomalyDetector:
    """Evaluates patch deviations against a memory bank of genuine reference packs.

    Computes nearest-neighbor Euclidean distances in patch embedding space,
    aggregates image-level anomaly scores, and generates Jet false-color heatmaps.
    """

    def __init__(
        self,
        memory_bank: Optional[PatchCoreMemoryBank] = None,
        anomaly_threshold: float = 0.50,
        top_k: int = 5,
    ):
        self.memory_bank = memory_bank
        self.anomaly_threshold = anomaly_threshold
        self.top_k = top_k

    def set_memory_bank(self, memory_bank: PatchCoreMemoryBank) -> None:
        self.memory_bank = memory_bank

    def evaluate(
        self,
        aligned_query: np.ndarray,
        aligned_reference: Optional[np.ndarray] = None,
    ) -> AnomalyResult:
        """Evaluates localized patch anomalies on an aligned packaging image."""
        h, w = aligned_query.shape[:2]

        if self.memory_bank is not None:
            # Full Memory-Bank Nearest Neighbor Search
            patch_feats, patch_coords = extract_patch_features(
                aligned_query,
                patch_size=self.memory_bank.patch_size,
                stride=self.memory_bank.stride,
            )

            # Compute min Euclidean distance to coreset points for each query patch
            coreset = self.memory_bank.coreset  # (M, D)
            # Compute distance matrix: (P, M)
            # ||p - m||^2 = ||p||^2 + ||m||^2 - 2 p.m
            p_sq = np.sum(patch_feats**2, axis=1, keepdims=True)
            m_sq = np.sum(coreset**2, axis=1, keepdims=True).T
            dot = np.dot(patch_feats, coreset.T)
            dists_sq = np.maximum(0.0, p_sq + m_sq - 2 * dot)
            patch_dists = np.sqrt(np.min(dists_sq, axis=1))

            # Image-level score: mean of top-k largest patch anomalies
            sorted_dists = np.sort(patch_dists)[::-1]
            k = min(self.top_k, len(sorted_dists))
            mean_top_k = float(np.mean(sorted_dists[:k]))
            max_dist = float(sorted_dists[0])

            # Reconstruct 2D anomaly grid for heatmap
            patch_size = self.memory_bank.patch_size
            stride = self.memory_bank.stride
            grid_h = (h - patch_size) // stride + 1
            grid_w = (w - patch_size) // stride + 1

            anomaly_map = patch_dists.reshape((grid_h, grid_w))

            # Find anomalous patch locations
            anom_patches = []
            for i, d in enumerate(patch_dists):
                if d > self.anomaly_threshold:
                    anom_patches.append(patch_coords[i])

        elif aligned_reference is not None:
            # Reference Difference Fallback
            diff = cv2.absdiff(aligned_query, aligned_reference)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
            smoothed = cv2.GaussianBlur(diff_gray, (15, 15), 0)
            norm_diff = smoothed.astype(np.float32) / 255.0

            mean_top_k = float(np.mean(norm_diff) * 2.5)
            max_dist = float(np.max(norm_diff))
            anomaly_map = cv2.resize(norm_diff, (64, 64))
            anom_patches = []
        else:
            return AnomalyResult(
                anomaly_score=0.0,
                is_anomalous=False,
                max_patch_distance=0.0,
                mean_top_k_distance=0.0,
                heatmap_bytes=None,
                spatial_anomaly_coords=[],
            )

        # Calibrate score to [0.0, 1.0] range
        calibrated_score = float(np.clip(mean_top_k, 0.0, 1.0))
        is_anomalous = calibrated_score > self.anomaly_threshold

        # Upsample anomaly map to full image resolution (512, 512)
        heatmap_upsampled = cv2.resize(
            anomaly_map, (w, h), interpolation=cv2.INTER_CUBIC
        )
        # Normalize to [0, 255]
        norm_map = np.clip(heatmap_upsampled / (self.anomaly_threshold * 1.5 + 1e-5), 0.0, 1.0)
        heatmap_uint8 = (norm_map * 255).astype(np.uint8)

        # Apply Jet colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend with query image for human-inspectable explainability overlay
        overlay = cv2.addWeighted(aligned_query, 0.65, heatmap_color, 0.35, 0)
        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return AnomalyResult(
            anomaly_score=calibrated_score,
            is_anomalous=is_anomalous,
            max_patch_distance=max_dist,
            mean_top_k_distance=mean_top_k,
            heatmap_bytes=buffer.tobytes(),
            spatial_anomaly_coords=anom_patches,
        )
