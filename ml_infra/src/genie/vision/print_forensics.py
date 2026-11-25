"""Print forensics: Halftone texture, CIELAB delta-E color, and edge sharpness."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class ForensicsResult:
    delta_e: float
    color_pass: bool
    edge_sharpness: float
    sharpness_pass: bool
    symbol_similarity: float
    symbol_pass: bool
    halftone_peak_ratio: float
    halftone_pass: bool
    overall_pass: bool
    discrepancies: List[str] = field(default_factory=list)


class PrintForensicsAnalyzer:
    """Inspects packaging print quality, color fidelity, and security features."""

    def __init__(
        self,
        max_delta_e: float = 5.0,
        min_sharpness: float = 12.0,
        min_symbol_similarity: float = 0.70,
        min_halftone_ratio: float = 1.35,
    ):
        self.max_delta_e = max_delta_e
        self.min_sharpness = min_sharpness
        self.min_symbol_similarity = min_symbol_similarity
        self.min_halftone_ratio = min_halftone_ratio

    @staticmethod
    def calculate_cielab_delta_e(
        query_bgr: np.ndarray,
        ref_bgr: np.ndarray,
        white_ref_patch: Optional[np.ndarray] = None,
    ) -> float:
        """Calculates Delta-E color distance in CIELAB color space.

        Optionally normalizes query colors against a white reference patch
        to eliminate ambient smartphone lighting temperature bias.
        """
        q = query_bgr.copy().astype(np.float32)
        r = ref_bgr.copy().astype(np.float32)

        # White balance compensation via Gray World / White patch normalization
        if white_ref_patch is not None and white_ref_patch.size > 0:
            white_mean = np.mean(white_ref_patch, axis=(0, 1)) + 1e-5
            scale = 255.0 / white_mean
            q = np.clip(q * scale, 0, 255)

        query_lab = cv2.cvtColor(q.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

        # Resize if dimensions differ slightly due to cropping
        if query_lab.shape[:2] != ref_lab.shape[:2]:
            ref_lab = cv2.resize(ref_lab, (query_lab.shape[1], query_lab.shape[0]))

        delta_e = np.sqrt(np.sum((query_lab - ref_lab) ** 2, axis=2))
        return float(np.mean(delta_e))

    @staticmethod
    def edge_sharpness(img: np.ndarray) -> float:
        """Measures high-frequency gradient sharpness across printed typography contours.

        Sharp offset print exhibits steep edge transitions; blurry digital prints
        or inkjet bleeding show degraded gradient magnitude.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return float(np.mean(mag))

    @staticmethod
    def verify_regulatory_symbol(
        query_region: np.ndarray,
        symbol_template: np.ndarray,
    ) -> Tuple[float, bool]:
        """Validates presence and fidelity of NAFDAC seal, holographic patch, or symbol.

        Uses normalized cross-correlation template matching.
        """
        q_gray = (
            cv2.cvtColor(query_region, cv2.COLOR_BGR2GRAY)
            if len(query_region.shape) == 3
            else query_region
        )
        t_gray = (
            cv2.cvtColor(symbol_template, cv2.COLOR_BGR2GRAY)
            if len(symbol_template.shape) == 3
            else symbol_template
        )

        # Ensure template is smaller than or equal to query region
        qh, qw = q_gray.shape[:2]
        th, tw = t_gray.shape[:2]
        if th > qh or tw > qw:
            scale = min(qh / th, qw / tw) * 0.95
            t_gray = cv2.resize(t_gray, (max(4, int(tw * scale)), max(4, int(th * scale))))

        res = cv2.matchTemplate(q_gray, t_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        similarity = float(max_val)

        return similarity, similarity >= 0.70

    @staticmethod
    def analyze_halftone_frequencies(region: np.ndarray) -> Tuple[float, bool]:
        """Performs 2D Fast Fourier Transform (FFT) on background region to detect

        periodic halftone screen frequencies characteristic of commercial offset presses.
        Inkjet and laser counterfeits typically lack periodic rosette screens.
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        h, w = gray.shape[:2]
        if h < 32 or w < 32:
            return 1.0, True

        # Compute 2D Fourier Transform and center low frequencies
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Mask out DC / low-frequency center
        cy, cx = h // 2, w // 2
        r_dc = max(5, min(h, w) // 16)
        y, x = np.ogrid[:h, :w]
        mask_center = ((x - cx) ** 2 + (y - cy) ** 2) <= r_dc**2
        magnitude_spectrum[mask_center] = 0.0

        median_energy = float(np.median(magnitude_spectrum)) + 1e-5
        peak_energy = float(np.max(magnitude_spectrum))
        peak_ratio = peak_energy / median_energy

        # Distinct periodic peaks typically yield peak-to-median ratio > 1.35
        has_periodic_halftone = peak_ratio >= 1.35
        return peak_ratio, has_periodic_halftone

    def analyze_forensics(
        self,
        query_aligned: np.ndarray,
        reference_aligned: np.ndarray,
        logo_bbox: Optional[Tuple[int, int, int, int]] = None,
        symbol_template: Optional[np.ndarray] = None,
        symbol_region_bbox: Optional[Tuple[int, int, int, int]] = None,
        white_ref_patch: Optional[np.ndarray] = None,
    ) -> ForensicsResult:
        """Runs end-to-end forensic analysis across color, sharpness, symbol, and halftone."""
        discrepancies: List[str] = []

        # 1. CIELAB Color Fidelity
        if logo_bbox is not None:
            ymin, xmin, ymax, xmax = logo_bbox
            q_logo = query_aligned[ymin:ymax, xmin:xmax]
            r_logo = reference_aligned[ymin:ymax, xmin:xmax]
            delta_e = self.calculate_cielab_delta_e(q_logo, r_logo, white_ref_patch)
        else:
            delta_e = self.calculate_cielab_delta_e(query_aligned, reference_aligned, white_ref_patch)

        color_pass = delta_e <= self.max_delta_e
        if not color_pass:
            discrepancies.append(f"Color deviation exceeds threshold (Delta-E: {delta_e:.2f} > {self.max_delta_e})")

        # 2. Typography Edge Sharpness
        sharpness = self.edge_sharpness(query_aligned)
        sharpness_pass = sharpness >= self.min_sharpness
        if not sharpness_pass:
            discrepancies.append(f"Typography edge sharpness too low ({sharpness:.1f} < {self.min_sharpness})")

        # 3. Regulatory Symbol / Seal Verification
        symbol_sim = 1.0
        symbol_pass = True
        if symbol_template is not None and symbol_region_bbox is not None:
            ymin, xmin, ymax, xmax = symbol_region_bbox
            q_sym = query_aligned[ymin:ymax, xmin:xmax]
            symbol_sim, symbol_pass = self.verify_regulatory_symbol(q_sym, symbol_template)
            if not symbol_pass:
                discrepancies.append(f"Regulatory symbol mismatch (similarity: {symbol_sim:.2f} < {self.min_symbol_similarity})")

        # 4. Halftone Screen FFT Analysis
        # Analyze central packaging background patch
        h, w = query_aligned.shape[:2]
        bg_patch = query_aligned[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        halftone_ratio, halftone_pass = self.analyze_halftone_frequencies(bg_patch)
        if not halftone_pass:
            discrepancies.append(f"Halftone frequency pattern anomaly (ratio: {halftone_ratio:.2f})")

        overall_pass = color_pass and sharpness_pass and symbol_pass

        return ForensicsResult(
            delta_e=delta_e,
            color_pass=color_pass,
            edge_sharpness=sharpness,
            sharpness_pass=sharpness_pass,
            symbol_similarity=symbol_sim,
            symbol_pass=symbol_pass,
            halftone_peak_ratio=halftone_ratio,
            halftone_pass=halftone_pass,
            overall_pass=overall_pass,
            discrepancies=discrepancies,
        )
