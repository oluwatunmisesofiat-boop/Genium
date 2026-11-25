"""Synthetic counterfeit degradation and adversarial packaging defect generator."""
import random
from typing import Optional, Tuple
import cv2
import numpy as np


class SyntheticCounterfeitGenerator:
    """Generates synthetic adversarial defects and camera degradations on packaging images."""

    @staticmethod
    def apply_camera_degradations(
        image: np.ndarray,
        blur_ksize: int = 7,
        glare_intensity: float = 0.35,
        jpeg_quality: int = 60,
    ) -> np.ndarray:
        """Applies realistic camera blur, non-uniform specular glare, and JPEG compression."""
        h, w = image.shape[:2]
        degraded = image.copy()

        # 1. Specular glare simulation
        if glare_intensity > 0:
            glare = np.zeros((h, w), dtype=np.uint8)
            cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
            radius = min(h, w) // 5
            cv2.circle(glare, (cx, cy), radius, 255, -1)
            glare = cv2.GaussianBlur(glare, (51, 51), 0)

            for c in range(3):
                degraded[:, :, c] = np.clip(
                    degraded[:, :, c] + glare * glare_intensity, 0, 255
                ).astype(np.uint8)

        # 2. Gaussian motion / defocus blur
        if blur_ksize > 1:
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            degraded = cv2.GaussianBlur(degraded, (blur_ksize, blur_ksize), 0)

        # 3. JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, enc = cv2.imencode(".jpg", degraded, encode_param)
        degraded = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return degraded

    @staticmethod
    def apply_packaging_defects(
        image: np.ndarray,
        substitute_color: bool = True,
        tamper_seal_bbox: Optional[Tuple[int, int, int, int]] = None,
        alter_typography_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Simulates physical printing counterfeit defects: off-color shifts and logo/text alteration."""
        tampered = image.copy()

        # 1. Color shift perturbation (e.g. off-brand counterfeit ink formulation)
        if substitute_color:
            # Shift hue and red/blue channels
            tampered[:, :, 0] = np.clip(tampered[:, :, 0].astype(int) + 35, 0, 255)

        # 2. Missing or obliterated regulatory seal
        if tamper_seal_bbox:
            ymin, xmin, ymax, xmax = tamper_seal_bbox
            # Fill seal with background color or noise
            bg_color = np.median(tampered, axis=(0, 1)).astype(np.uint8)
            tampered[ymin:ymax, xmin:xmax] = bg_color

        # 3. Altered / replaced text typography
        if alter_typography_bbox:
            ymin, xmin, ymax, xmax = alter_typography_bbox
            tampered[ymin:ymax, xmin:xmax] = 240
            cv2.putText(
                tampered,
                "COARTEN 80/480",  # Misspelled counterfeit text
                (xmin + 5, ymax - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return tampered

    @staticmethod
    def splice_barcode_onto_carton(
        carton_image: np.ndarray,
        barcode_crop: np.ndarray,
        target_pos: Tuple[int, int] = (300, 50),
    ) -> np.ndarray:
        """Digitally splices an authentic barcode crop onto an unrelated counterfeit pack face."""
        spliced = carton_image.copy()
        y, x = target_pos
        bh, bw = barcode_crop.shape[:2]
        ch, cw = carton_image.shape[:2]

        if y + bh <= ch and x + bw <= cw:
            spliced[y : y + bh, x : x + bw] = barcode_crop
        return spliced


def apply_adversarial_counterfeit_degradations(image: np.ndarray) -> np.ndarray:
    """Helper entry point for pipeline backwards compatibility."""
    return SyntheticCounterfeitGenerator.apply_camera_degradations(image)
