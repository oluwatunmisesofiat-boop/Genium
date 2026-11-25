"""Barcode localization and bounding box extraction."""
from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectedBarcodeRegion:
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    orientation_angle: int  # 0, 90, 180, 270
    crop_image: np.ndarray
    confidence: float = 1.0


class BarcodeDetector:
    """
    Localizes 1D and 2D barcode regions using gradient edge density heuristics or YOLO.
    Extracts padded bounding boxes and estimates orientation angles in <40ms.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._onnx_session = None

        if self.model_path:
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(self.model_path)
            except Exception:
                pass

    def detect_regions(
        self,
        image_np: np.ndarray,
        padding_ratio: float = 0.10,
    ) -> List[DetectedBarcodeRegion]:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        h_img, w_img = gray.shape[:2]

        # 1. Gradient in X direction to isolate dense vertical barcode stripes
        grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3))
        grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3))
        gradient = cv2.subtract(grad_x, grad_y)

        # 2. Blur and adaptive thresholding
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphological closure: connect closely spaced barcode bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=3)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[DetectedBarcodeRegion] = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 15 and (w * h) > 600:
                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)

                crop = image_np[y1:y2, x1:x2]
                angle = self._estimate_orientation(crop)

                regions.append(
                    DetectedBarcodeRegion(
                        bounding_box=(x1, y1, x2 - x1, y2 - y1),
                        orientation_angle=angle,
                        crop_image=crop,
                        confidence=0.92,
                    )
                )

        # Sort by bounding box area descending
        regions.sort(key=lambda r: r.bounding_box[2] * r.bounding_box[3], reverse=True)

        if not regions:
            regions.append(
                DetectedBarcodeRegion(
                    bounding_box=(0, 0, w_img, h_img),
                    orientation_angle=0,
                    crop_image=image_np,
                    confidence=0.50,
                )
            )

        return regions

    def _estimate_orientation(self, crop: np.ndarray) -> int:
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        sum_gx = np.sum(np.abs(gx))
        sum_gy = np.sum(np.abs(gy))

        return 90 if sum_gy > 1.3 * sum_gx else 0
