"""Reference-to-query packaging comparison for counterfeit detection.

Compares a rectified query packaging image against a canonical reference pack image
from the database across multiple forensic dimensions:
1. Geometric homography alignment (ORB + RANSAC)
2. Lighting-normalized CIELAB color fidelity & HSV histogram correlation
3. Critical element positioning (logo, NAFDAC badge, barcode, seals) & 4x4 grid SSIM
4. Patch-level visual anomaly scoring and edge sharpness forensics
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from genie.vision.anomaly import PatchCoreAnomalyDetector
from genie.vision.match import AlignmentResult, GeometricMatcher
from genie.vision.print_forensics import PrintForensicsAnalyzer


@dataclass
class ElementPositionCheck:
    name: str
    expected_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in 512x512 space
    shift_px: float
    local_similarity: float
    passed: bool
    status: str  # "ok" | "shifted" | "missing_or_altered"


@dataclass
class PackComparisonResult:
    overall_similarity_score: float  # [0.0, 1.0]
    is_counterfeit_suspect: bool
    alignment_matched: bool
    inlier_ratio: float
    reprojection_error: float
    color_score: float
    delta_e_ab: float
    color_histogram_corr: float
    layout_score: float
    structural_ssim: float
    edge_alignment_score: float
    anomaly_score: float
    sharpness_ratio: float
    element_checks: List[ElementPositionCheck] = field(default_factory=list)
    flagged_grid_zones: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    heatmap_bytes: Optional[bytes] = None


_GRID_ZONE_NAMES = [
    ["top-left", "top-center-left", "top-center-right", "top-right"],
    ["upper-mid-left", "upper-mid-center-left", "upper-mid-center-right", "upper-mid-right"],
    ["lower-mid-left", "lower-mid-center-left", "lower-mid-center-right", "lower-mid-right"],
    ["bottom-left", "bottom-center-left", "bottom-center-right", "bottom-right"],
]


class PackComparator:
    """Compares a consumer scan against a canonical reference packaging image."""

    def __init__(
        self,
        canonical_size: Tuple[int, int] = (512, 512),
        max_delta_e_ab: float = 10.0,
        max_element_shift_px: float = 18.0,
        min_element_ncc: float = 0.45,
        suspect_threshold: float = 0.65,
    ):
        self.canonical_size = canonical_size
        self.max_delta_e_ab = max_delta_e_ab
        self.max_element_shift_px = max_element_shift_px
        self.min_element_ncc = min_element_ncc
        self.suspect_threshold = suspect_threshold

        self.matcher = GeometricMatcher(
            min_inliers=20,
            min_inlier_ratio=0.30,
            max_reprojection_error=5.0,
        )
        self.anomaly_detector = PatchCoreAnomalyDetector()

    @staticmethod
    def _compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        """Computes mean Structural Similarity Index (SSIM) between two grayscale images."""
        a = gray_a.astype(np.float32)
        b = gray_b.astype(np.float32)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

        mu_a_sq = mu_a * mu_a
        mu_b_sq = mu_b * mu_b
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

        num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
        den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
        ssim_map = num / (den + 1e-6)
        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))

    @staticmethod
    def _normalize_luminance(query_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
        """Matches query L* channel mean/std to reference L* to reduce ambient lighting bias."""
        q_lab = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        q_l, q_a, q_b = cv2.split(q_lab)
        r_l, _, _ = cv2.split(r_lab)

        q_std = float(np.std(q_l)) + 1e-5
        r_std = float(np.std(r_l)) + 1e-5
        q_l_norm = (q_l - float(np.mean(q_l))) * (r_std / q_std) + float(np.mean(r_l))
        q_l_norm = np.clip(q_l_norm, 0.0, 255.0)

        merged = cv2.merge([q_l_norm, q_a, q_b]).astype(np.uint8)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _compare_colors(
        self,
        aligned_query: np.ndarray,
        ref_img: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Evaluates color consistency using chromaticity Delta-E (a*, b*) and HSV histogram correlation."""
        q_norm = self._normalize_luminance(aligned_query, ref_img)

        q_lab = cv2.cvtColor(q_norm, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Pure chromaticity distance in (a*, b*) plane
        da = q_lab[:, :, 1] - r_lab[:, :, 1]
        db = q_lab[:, :, 2] - r_lab[:, :, 2]
        delta_e_ab = float(np.mean(np.sqrt(da * da + db * db)))

        # 2D (a*, b*) chromaticity histogram correlation (avoids HSV hue singularity on white/gray backgrounds)
        hist_q = cv2.calcHist([q_lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        hist_r = cv2.calcHist([r_lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist_q, hist_q, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        hist_corr = float(cv2.compareHist(hist_q, hist_r, cv2.HISTCMP_CORREL))
        hist_corr_clipped = float(np.clip(hist_corr, 0.0, 1.0))

        # Calibrate Delta-E_ab into [0, 1] score (<= 3.0 -> ~1.0, >= 25.0 -> ~0.0)
        chroma_score = float(np.clip(1.0 - max(0.0, delta_e_ab - 2.5) / 22.5, 0.0, 1.0))
        color_score = float(0.65 * chroma_score + 0.35 * hist_corr_clipped)
        return color_score, delta_e_ab, hist_corr

    def _to_pixel_bbox(self, bbox: List[float]) -> Tuple[int, int, int, int]:
        w, h = self.canonical_size
        x1, y1, x2, y2 = bbox[:4]
        if max(x1, y1, x2, y2) <= 1.0:
            return (
                int(np.clip(x1 * w, 0, w - 1)),
                int(np.clip(y1 * h, 0, h - 1)),
                int(np.clip(x2 * w, 1, w)),
                int(np.clip(y2 * h, 1, h)),
            )
        return (
            int(np.clip(x1, 0, w - 1)),
            int(np.clip(y1, 0, h - 1)),
            int(np.clip(x2, 1, w)),
            int(np.clip(y2, 1, h)),
        )

    def _check_element_positions(
        self,
        aligned_query: np.ndarray,
        ref_img: np.ndarray,
        semantic_regions: Optional[Dict[str, Any]],
    ) -> List[ElementPositionCheck]:
        """Verifies that critical packaging elements appear at their expected positions."""
        if not semantic_regions:
            return []

        q_gray = cv2.cvtColor(aligned_query, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        h, w = r_gray.shape[:2]
        search_margin = 48
        checks: List[ElementPositionCheck] = []

        for name, raw_bbox in semantic_regions.items():
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                continue
            x1, y1, x2, y2 = self._to_pixel_bbox(list(raw_bbox))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            template = r_gray[y1:y2, x1:x2]
            sx1 = max(0, x1 - search_margin)
            sy1 = max(0, y1 - search_margin)
            sx2 = min(w, x2 + search_margin)
            sy2 = min(h, y2 + search_margin)
            search_patch = q_gray[sy1:sy2, sx1:sx2]

            if search_patch.shape[0] < template.shape[0] or search_patch.shape[1] < template.shape[1]:
                continue

            res = cv2.matchTemplate(search_patch, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            ncc = float(np.clip(max_val, 0.0, 1.0))

            detected_x = sx1 + max_loc[0]
            detected_y = sy1 + max_loc[1]
            shift_px = float(np.hypot(detected_x - x1, detected_y - y1))

            if ncc < self.min_element_ncc:
                status = "missing_or_altered"
                passed = False
            elif shift_px > self.max_element_shift_px:
                status = "shifted"
                passed = False
            else:
                status = "ok"
                passed = True

            checks.append(
                ElementPositionCheck(
                    name=str(name),
                    expected_bbox=(x1, y1, x2, y2),
                    shift_px=shift_px,
                    local_similarity=ncc,
                    passed=passed,
                    status=status,
                )
            )

        return checks

    def _compare_grid_layout(
        self,
        aligned_query: np.ndarray,
        ref_img: np.ndarray,
    ) -> Tuple[float, float, List[str]]:
        """Computes 4x4 spatial grid SSIM and edge-map IoU to localize layout shifts."""
        q_norm = self._normalize_luminance(aligned_query, ref_img)
        q_gray = cv2.cvtColor(q_norm, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        h, w = r_gray.shape[:2]
        cell_h, cell_w = h // 4, w // 4
        zone_ssims: List[float] = []
        flagged_zones: List[str] = []

        for r_idx in range(4):
            for c_idx in range(4):
                y1, y2 = r_idx * cell_h, (r_idx + 1) * cell_h
                x1, x2 = c_idx * cell_w, (c_idx + 1) * cell_w
                z_ssim = self._compute_ssim(q_gray[y1:y2, x1:x2], r_gray[y1:y2, x1:x2])
                zone_ssims.append(z_ssim)
                if z_ssim < 0.42:
                    flagged_zones.append(_GRID_ZONE_NAMES[r_idx][c_idx])

        mean_ssim = float(np.mean(zone_ssims)) if zone_ssims else 0.0

        # Edge-map overlap (dilated Canny edges to tolerate 2-3px alignment jitter)
        q_edges = cv2.Canny(q_gray, 60, 160)
        r_edges = cv2.Canny(r_gray, 60, 160)
        kernel = np.ones((5, 5), np.uint8)
        q_dil = cv2.dilate(q_edges, kernel, iterations=1) > 0
        r_dil = cv2.dilate(r_edges, kernel, iterations=1) > 0

        intersection = float(np.sum(q_dil & r_dil))
        union = float(np.sum(q_dil | r_dil)) + 1e-5
        edge_iou = intersection / union

        return mean_ssim, float(np.clip(edge_iou, 0.0, 1.0)), flagged_zones

    def compare(
        self,
        query_img: np.ndarray,
        reference_img: np.ndarray,
        semantic_regions: Optional[Dict[str, Any]] = None,
    ) -> PackComparisonResult:
        """Runs full visual, color, and positional comparison against a reference pack."""
        w, h = self.canonical_size
        ref_resized = cv2.resize(reference_img, (w, h), interpolation=cv2.INTER_AREA)

        # 1. Geometric Homography Alignment
        alignment: AlignmentResult = self.matcher.match(
            query_img, ref_resized, output_size=self.canonical_size
        )
        if alignment.matched and alignment.aligned_image is not None:
            aligned_query = alignment.aligned_image
        else:
            aligned_query = cv2.resize(query_img, (w, h), interpolation=cv2.INTER_AREA)

        # 2. Color Fidelity Comparison
        color_score, delta_e_ab, hist_corr = self._compare_colors(aligned_query, ref_resized)

        # 3. Structural Grid Layout & Critical Element Positioning
        structural_ssim, edge_score, flagged_zones = self._compare_grid_layout(
            aligned_query, ref_resized
        )
        element_checks = self._check_element_positions(
            aligned_query, ref_resized, semantic_regions
        )

        if element_checks:
            elem_scores = [
                c.local_similarity if c.passed else c.local_similarity * 0.4
                for c in element_checks
            ]
            elem_layout_score = float(np.mean(elem_scores))
            layout_score = float(0.50 * elem_layout_score + 0.35 * structural_ssim + 0.15 * edge_score)
        else:
            layout_score = float(0.70 * structural_ssim + 0.30 * edge_score)

        # 4. Patch-Level Anomaly & Print Sharpness
        anomaly_res = self.anomaly_detector.evaluate(
            aligned_query=aligned_query,
            aligned_reference=ref_resized,
        )
        q_sharp = PrintForensicsAnalyzer.edge_sharpness(aligned_query)
        r_sharp = PrintForensicsAnalyzer.edge_sharpness(ref_resized)
        sharpness_ratio = float(q_sharp / (r_sharp + 1e-5))

        # 5. Aggregate Overall Similarity
        align_component = alignment.inlier_ratio if alignment.matched else 0.25 * alignment.inlier_ratio
        overall_score = float(
            np.clip(
                0.20 * align_component
                + 0.30 * color_score
                + 0.35 * layout_score
                + 0.15 * (1.0 - anomaly_res.anomaly_score),
                0.0,
                1.0,
            )
        )

        # 6. Collect Human-Readable Inconsistencies
        inconsistencies: List[str] = []
        if not alignment.matched:
            inconsistencies.append(
                f"Geometric layout failed alignment with reference pack (inliers: {alignment.num_inliers}, ratio: {alignment.inlier_ratio:.2f})."
            )
        if delta_e_ab > self.max_delta_e_ab:
            inconsistencies.append(
                f"Packaging color mismatch detected (CIELAB ΔE_ab = {delta_e_ab:.1f} > {self.max_delta_e_ab:.1f}, hist corr = {hist_corr:.2f})."
            )
        for check in element_checks:
            if check.status == "missing_or_altered":
                inconsistencies.append(
                    f"Critical element '{check.name}' missing or visually altered (similarity: {check.local_similarity:.2f})."
                )
            elif check.status == "shifted":
                inconsistencies.append(
                    f"Critical element '{check.name}' positioned inconsistently (shifted by {check.shift_px:.1f}px)."
                )
        if flagged_zones:
            zone_preview = ", ".join(flagged_zones[:4])
            inconsistencies.append(
                f"Structural layout deviations detected in pack zone(s): {zone_preview}."
            )
        if sharpness_ratio < 0.55:
            inconsistencies.append(
                f"Print edge sharpness significantly lower than canonical reference (ratio: {sharpness_ratio:.2f})."
            )

        has_failed_element = any(not c.passed for c in element_checks)
        is_suspect = (
            overall_score < self.suspect_threshold
            or color_score < 0.55
            or layout_score < 0.50
            or has_failed_element
        )

        return PackComparisonResult(
            overall_similarity_score=overall_score,
            is_counterfeit_suspect=is_suspect,
            alignment_matched=alignment.matched,
            inlier_ratio=alignment.inlier_ratio,
            reprojection_error=alignment.reprojection_error,
            color_score=color_score,
            delta_e_ab=delta_e_ab,
            color_histogram_corr=hist_corr,
            layout_score=layout_score,
            structural_ssim=structural_ssim,
            edge_alignment_score=edge_score,
            anomaly_score=anomaly_res.anomaly_score,
            sharpness_ratio=sharpness_ratio,
            element_checks=element_checks,
            flagged_grid_zones=flagged_zones,
            inconsistencies=inconsistencies,
            heatmap_bytes=anomaly_res.heatmap_bytes,
        )
