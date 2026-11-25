
"""Packaging face localization, perspective homography rectification, and cylindrical unwrapping."""
from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class RectificationResult:
    rectified_image: np.ndarray
    homography_matrix: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    reprojection_error: float = 0.0
    is_cylindrical: bool = False
    latency_ms: float = 0.0


class PackDetectorRectifier:
    """
    Localizes packaging face and corrects perspective to a canonical planar view.
    Supports planar quadrilaterals (cartons, boxes, blister packs) and cylindrical unwrapping (bottles, syrups).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        onnx_model_path: Optional[str] = None,
    ):
        self.target_size = target_size
        self.onnx_model_path = onnx_model_path
        self._onnx_session = None

        if self.onnx_model_path:
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(self.onnx_model_path)
            except Exception:
                pass

    def rectify(
        self,
        image_np: np.ndarray,
        is_cylindrical: bool = False,
    ) -> RectificationResult:
        t0 = time.perf_counter()

        if is_cylindrical:
            unwrapped = self.unwrap_cylindrical_surface(image_np)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RectificationResult(
                rectified_image=unwrapped,
                is_cylindrical=True,
                latency_ms=elapsed_ms,
            )

        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        corners = self._detect_quadrilateral_corners(gray)
        homography_matrix = None
        reprojection_error = 0.0

        if corners is not None:
            rect = self._order_points(corners)
            dst = np.array([
                [0, 0],
                [self.target_size[0] - 1, 0],
                [self.target_size[0] - 1, self.target_size[1] - 1],
                [0, self.target_size[1] - 1],
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            homography_matrix = M
            warped = cv2.warpPerspective(image_np, M, self.target_size, flags=cv2.INTER_LINEAR)

            # Compute reprojection error: ||dst - M * src||
            src_homogeneous = np.hstack([rect, np.ones((4, 1), dtype="float32")])
            projected = (M @ src_homogeneous.T).T
            projected_pts = projected[:, :2] / projected[:, 2:]
            reprojection_error = float(np.mean(np.linalg.norm(dst - projected_pts, axis=1)))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RectificationResult(
                rectified_image=warped,
                homography_matrix=homography_matrix,
                corners=rect,
                reprojection_error=reprojection_error,
                latency_ms=elapsed_ms,
            )

        # Fallback: center crop to canonical target size
        h, w = image_np.shape[:2]
        crop_dim = min(h, w)
        start_y = (h - crop_dim) // 2
        start_x = (w - crop_dim) // 2
        cropped = image_np[start_y : start_y + crop_dim, start_x : start_x + crop_dim]
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RectificationResult(
            rectified_image=resized,
            reprojection_error=0.0,
            latency_ms=elapsed_ms,
        )

    def unwrap_cylindrical_surface(
        self,
        image_np: np.ndarray,
        cylinder_radius: Optional[float] = None,
    ) -> np.ndarray:
        """
        Unrolls cylindrical curved packaging labels (e.g., syrup bottles and vials)
        using cylindrical inverse mapping x' = R * arcsin((x - xc) / R).
        """
        h, w = image_np.shape[:2]
        xc = w / 2.0
        R = cylinder_radius if cylinder_radius is not None else float(w / 1.6)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            map_y[y, :] = y

        x_coords = np.arange(w, dtype=np.float32)
        dx = (x_coords - xc) / max(R, 1.0)
        # Clip to avoid arcsin domain errors outside [-0.98, 0.98]
        dx_clipped = np.clip(dx, -0.98, 0.98)
        src_x = xc + R * np.arcsin(dx_clipped)

        for y in range(h):
            map_x[y, :] = src_x

        unwrapped = cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.resize(unwrapped, self.target_size, interpolation=cv2.INTER_AREA)

    def _detect_quadrilateral_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 120)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                total_area = gray.shape[0] * gray.shape[1]
                if area > 0.15 * total_area:
                    return approx.reshape(4, 2).astype("float32")

        # Fallback: Minimum area bounding rectangle
        largest = contours[0]
        if cv2.contourArea(largest) > 0.15 * (gray.shape[0] * gray.shape[1]):
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            return np.array(box, dtype="float32")

        return None

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]       # Top-left
        rect[2] = pts[np.argmax(s)]       # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]    # Top-right
        rect[3] = pts[np.argmax(diff)]    # Bottom-left
        return rect
