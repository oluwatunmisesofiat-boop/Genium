"""Image Quality Gating evaluating blur, specular glare, illumination, and framing."""
from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class QualityGateResult:
    passed: bool
    blur_score: float
    glare_ratio: float
    pack_area_ratio: float
    guidance_codes: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    latency_ms: float = 0.0


class ImageQualityGate:
    """
    Fast CPU-based quality evaluation completed in <120ms.
    Enforces quality thresholds for blur, specular glare, framing, and panel guidance.
    """

    def __init__(
        self,
        min_blur_score: float = 120.0,
        max_glare_ratio: float = 0.08,
        min_pack_area_ratio: float = 0.20,
        max_pack_area_ratio: float = 0.95,
    ):
        self.min_blur_score = min_blur_score
        self.max_glare_ratio = max_glare_ratio  # Spec: flag if glare > 8% of pack surface
        self.min_pack_area_ratio = min_pack_area_ratio
        self.max_pack_area_ratio = max_pack_area_ratio

    def evaluate(
        self,
        image_np: np.ndarray,
        require_back_panel: bool = False,
    ) -> QualityGateResult:
        t0 = time.perf_counter()

        # Image format normalization
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        else:
            gray = image_np
            hsv = None

        guidance: List[str] = []

        # 1. Framing & Pack Region Estimation
        pack_area_ratio, pack_mask = self._estimate_pack_region(gray)
        if pack_area_ratio < self.min_pack_area_ratio:
            guidance.append("MOVE_CLOSER")

        # 2. Specular Glare Evaluation in HSV Color Space (V > 245, S < 35)
        # Exclude border-connected white studio backgrounds so clean white-background product photos pass.
        glare_ratio = 0.0
        if hsv is not None:
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            raw_glare = ((v_channel > 245) & (s_channel < 35)).astype(np.uint8)

            h_img, w_img = raw_glare.shape[:2]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_glare, connectivity=8)
            filtered_glare = np.zeros_like(raw_glare, dtype=bool)
            for lbl in range(1, num_labels):
                x, y, bw, bh, area = stats[lbl]
                touches_border = (x == 0) or (y == 0) or (x + bw >= w_img) or (y + bh >= h_img)
                if touches_border and (area / float(raw_glare.size)) >= 0.015:
                    continue
                filtered_glare |= (labels == lbl)

            if pack_mask is not None and np.count_nonzero(pack_mask) > 0:
                pack_pixel_count = np.count_nonzero(pack_mask)
                glare_in_pack = np.count_nonzero(filtered_glare & (pack_mask > 0))
                glare_ratio = float(glare_in_pack / pack_pixel_count)
            else:
                glare_ratio = float(np.count_nonzero(filtered_glare) / filtered_glare.size)

            if glare_ratio > self.max_glare_ratio:
                guidance.append("REDUCE_GLARE")

        # 3. Blur Evaluation via Modified Laplacian Variance
        blur_score = self._compute_blur_score(gray, pack_mask)
        if blur_score < self.min_blur_score:
            guidance.append("HOLD_STEADY_BLURRY")

        # 4. Panel Guidance
        if require_back_panel:
            guidance.append("SHOW_BACK_PANEL")

        passed = len(guidance) == 0
        rejection_reason = None
        if not passed:
            rejection_reason = f"Image failed quality thresholds: {', '.join(guidance)}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return QualityGateResult(
            passed=passed,
            blur_score=blur_score,
            glare_ratio=glare_ratio,
            pack_area_ratio=pack_area_ratio,
            guidance_codes=guidance,
            rejection_reason=rejection_reason,
            latency_ms=elapsed_ms,
        )

    def _estimate_pack_region(self, gray: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        h, w = gray.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return 0.0, None

        # Otsu thresholding with morphological closure
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1.0, None

        largest_contour = max(contours, key=cv2.contourArea)
        pack_area = float(cv2.contourArea(largest_contour))
        area_ratio = pack_area / total_pixels

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        return area_ratio, mask

    def _compute_blur_score(self, gray: np.ndarray, pack_mask: Optional[np.ndarray]) -> float:
        # Resize to standardized dimensions to ensure resolution-independent variance
        target_h = 512
        h, w = gray.shape[:2]
        if h > 0 and h != target_h:
            scale = target_h / float(h)
            target_w = int(w * scale)
            resized_gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            resized_gray = gray

        laplacian = cv2.Laplacian(resized_gray, cv2.CV_64F, ksize=3)
        return float(laplacian.var())
