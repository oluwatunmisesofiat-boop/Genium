"""Packaging text extraction, regulatory NAFDAC parsing, and batch verification."""
from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from genie.extraction.nafdac_parser import GreenbookRegistry, normalize_nafdac_number


@dataclass
class TextLineCrop:
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    text: str = ""
    confidence: float = 1.0
    crop_image: Optional[np.ndarray] = None


@dataclass
class OCRExtractionResult:
    full_text: str
    raw_text_lines: List[str]
    nafdac_number: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    mfg_date: Optional[str] = None
    greenbook_record: Optional[Dict[str, Any]] = None
    batch_cross_check_passed: bool = True
    batch_cross_check_reason: Optional[str] = None
    latency_ms: float = 0.0


class PackagingOCR:
    """
    Executes text detection, text recognition, and regulatory field extraction.
    Optimized for pharmaceutical packaging, dot-matrix inkjet batch codes, and NAFDAC formats.
    """

    # Comprehensive Nigerian NAFDAC Registration Number Patterns
    NAFDAC_PATTERNS = [
        re.compile(
            r"(?:NAFDAC(?:\s*REG(?:ISTRATION)?(?:\.|\s*NO)?)?|NRN|N\.R\.N\.?)[:\s.-]*(\b[A-Z0-9]{2}\s*[-–—]?\s*[0-9]{4,6}[A-Z]?\b)",
            re.IGNORECASE,
        ),
        re.compile(r"\b([A-Z]\d\s*[-–—]\s*\d{4,6}[A-Z]?)\b", re.IGNORECASE),
        re.compile(r"\b(\d{2}\s*[-–—]\s*\d{4,6}[A-Z]?)\b"),
    ]

    # Batch / Lot code prefixes
    BATCH_PATTERNS = [
        re.compile(r"\b(?:BATCH|LOT|B\.NO|BN|L\.NO|BATCH\s*NO)[:\s.-]*([A-Z0-9-]{3,20})\b", re.IGNORECASE),
        re.compile(r"\b(?:B/N|L/N)[:\s.-]*([A-Z0-9-]{3,20})\b", re.IGNORECASE),
    ]

    # Expiry date patterns: DD/MM/YYYY, MM/YYYY, DD-MM-YY, MMM YYYY
    EXPIRY_PATTERNS = [
        re.compile(r"\b(?:EXP|EXPIRY|EXP\.DATE|USE\s*BEFORE)[:\s.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(?:EXP|EXPIRY|EXP\.DATE)[:\s.-]*([0-9]{2}[/-][0-9]{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(?:EXP|EXPIRY)[:\s.-]*([A-Z]{3,4}\.?\s*[0-9]{4})\b", re.IGNORECASE),
    ]

    # Manufacturing date patterns: MFD, MFG, MFD.DATE
    MFG_PATTERNS = [
        re.compile(r"\b(?:MFD|MFG|MFD\.DATE|PROD|MFR)[:\s.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(?:MFD|MFG|MFD\.DATE|PROD|MFR)[:\s.-]*([0-9]{2}[/-][0-9]{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(?:MFD|MFG)[:\s.-]*([A-Z]{3,4}\.?\s*[0-9]{4})\b", re.IGNORECASE),
    ]

    def __init__(self, greenbook_db_path: Optional[str] = None, enable_vlm: bool = False):
        self.greenbook = GreenbookRegistry(db_path=greenbook_db_path)
        self.enable_vlm = enable_vlm

    def extract_text(
        self,
        image_np: np.ndarray,
        barcode_batch: Optional[str] = None,
        barcode_expiry: Optional[str] = None,
        use_vlm: bool = False,
    ) -> OCRExtractionResult:
        t0 = time.perf_counter()
        lines: List[str] = []

        # 1. High-Performance Text Extraction (PaddleOCR / docTR with fallback)
        extracted_lines = self._run_ocr_engine(image_np)
        lines.extend(extracted_lines)

        # Optional fallback to NVIDIA VLM OCR if lines are empty and explicitly enabled
        vlm_data = {}
        if not lines and (use_vlm or self.enable_vlm):
            vlm_data = self.extract_with_vlm(image_np)
            if vlm_data.get("raw_text_lines"):
                lines.extend(vlm_data["raw_text_lines"])

        full_text = " ".join(lines)
        normalized_text = self._normalize_ocr_text(full_text)

        # 2. Regulatory NAFDAC Number Parsing
        nafdac_found = self._parse_nafdac_number(normalized_text) or vlm_data.get("nafdac_number")

        # 3. Batch & Expiry Parsing
        batch_found = self._parse_batch_code(normalized_text) or vlm_data.get("batch_number")
        expiry_found = self._parse_expiry_date(normalized_text) or vlm_data.get("expiry_date")
        mfg_found = self._parse_mfg_date(normalized_text) or vlm_data.get("mfg_date")

        # 4. Local Greenbook Database Lookup (<5ms)
        greenbook_record = None
        if nafdac_found:
            greenbook_record = self.greenbook.lookup_number(nafdac_found)

        # 5. Cross-Check Printed Text vs. Barcode Data
        cross_check_passed, cross_check_reason = self.cross_check_batch_and_expiry(
            ocr_batch=batch_found,
            ocr_expiry=expiry_found,
            barcode_batch=barcode_batch,
            barcode_expiry=barcode_expiry,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return OCRExtractionResult(
            full_text=full_text,
            raw_text_lines=lines,
            nafdac_number=nafdac_found,
            batch_number=batch_found,
            expiry_date=expiry_found,
            mfg_date=mfg_found,
            greenbook_record=greenbook_record,
            batch_cross_check_passed=cross_check_passed,
            batch_cross_check_reason=cross_check_reason,
            latency_ms=elapsed_ms,
        )

    def detect_text_line_candidates(self, image_np: np.ndarray) -> List[TextLineCrop]:
        """
        Fast candidate text-line detector using directional morphological gradients.
        Isolates packaging text strips and inkjet date stamps in <30ms on CPU.
        """
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        enhanced = self.enhance_dot_matrix_inkjet(gray)

        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        connected = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(connected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[TextLineCrop] = []

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 25 and 8 < bh < 120 and (bw / float(bh)) > 1.2:
                crop = image_np[by : by + bh, bx : bx + bw]
                candidates.append(
                    TextLineCrop(
                        bounding_box=(bx, by, bw, bh),
                        crop_image=crop,
                    )
                )

        candidates.sort(key=lambda c: c.bounding_box[1])
        return candidates

    @staticmethod
    def enhance_dot_matrix_inkjet(gray: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(closed)

    def enhance_dot_matrix(self, gray: np.ndarray) -> np.ndarray:
        return self.enhance_dot_matrix_inkjet(gray)

    def parse_nafdac_number(self, text: str) -> Optional[str]:
        return self._parse_nafdac_number(text)

    def parse_batch_and_expiry(self, text: str) -> dict:
        return {
            "batch_number": self._parse_batch_code(text),
            "expiry_date": self._parse_expiry_date(text),
            "mfg_date": self._parse_mfg_date(text),
        }

    def extract_with_vlm(self, image_np: np.ndarray) -> dict:
        """Extracts text, NAFDAC number, and batch/expiry using NVIDIA Nemotron VLM."""
        from genie.common.config import settings
        if not settings.NVIDIA_API_KEY:
            return {}

        try:
            from genie.vision.vlm_review import VLMReviewer
            reviewer = VLMReviewer()
            res = reviewer.review_packaging(image_np)
            return {
                "nafdac_number": res.detected_nafdac,
                "batch_number": res.detected_batch,
                "expiry_date": res.detected_expiry,
                "mfg_date": None,
                "raw_text_lines": res.reasons,
            }
        except Exception:
            return {}

    @classmethod
    def cross_check_batch_and_expiry(
        cls,
        ocr_batch: Optional[str],
        ocr_expiry: Optional[str],
        barcode_batch: Optional[str],
        barcode_expiry: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Cross-checks printed packaging text against GS1 barcode payload.
        Discrepancy indicates counterfeit overprinting.
        """
        if ocr_batch and barcode_batch:
            clean_o = re.sub(r"\W", "", ocr_batch).upper()
            clean_b = re.sub(r"\W", "", barcode_batch).upper()
            if clean_o != clean_b:
                return (
                    False,
                    f"Batch mismatch: Printed text reads '{ocr_batch}' but barcode encodes '{barcode_batch}'.",
                )

        if ocr_expiry and barcode_expiry:
            # Extract 4-digit years from both
            ocr_year_match = re.search(r"\b(20[2-3][0-9])\b", ocr_expiry)
            bc_year_match = re.search(r"\b(20[2-3][0-9])\b", barcode_expiry)
            if ocr_year_match and bc_year_match:
                if ocr_year_match.group(1) != bc_year_match.group(1):
                    return (
                        False,
                        f"Expiry mismatch: Printed expiry '{ocr_expiry}' differs from barcode expiry '{barcode_expiry}'.",
                    )

        return True, None

    def _run_ocr_engine(self, image_np: np.ndarray) -> List[str]:
        lines: List[str] = []
        # 1. PaddleOCR
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            results = ocr.ocr(image_np, cls=True)
            if results and results[0]:
                for line in results[0]:
                    lines.append(line[1][0])
            if lines:
                return lines
        except Exception:
            pass

        # 2. RapidOCR ONNX Runtime fallback (cached singleton)
        try:
            from rapidocr_onnxruntime import RapidOCR
            if not hasattr(self, "_rapid_ocr") or self._rapid_ocr is None:
                self._rapid_ocr = RapidOCR()
            res, _ = self._rapid_ocr(image_np)
            if res:
                for item in res:
                    if len(item) >= 2 and item[1]:
                        lines.append(str(item[1]))
            if lines:
                return lines
        except Exception:
            pass

        # 3. EasyOCR fallback
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            res = reader.readtext(image_np, detail=0)
            if res:
                return [str(t) for t in res if str(t).strip()]
        except Exception:
            pass

        # 4. Tesseract fallback
        try:
            import pytesseract
            txt = pytesseract.image_to_string(image_np)
            extracted = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if extracted:
                return extracted
        except Exception:
            pass

        return lines

    def _normalize_ocr_text(self, text: str) -> str:
        t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
        t = re.sub(r"\s+", " ", t)
        return t

    def _parse_nafdac_number(self, text: str) -> Optional[str]:
        for pattern in self.NAFDAC_PATTERNS:
            match = pattern.search(text)
            if match:
                raw_match = match.group(1).strip()
                return normalize_nafdac_number(raw_match)
        return None

    def _parse_batch_code(self, text: str) -> Optional[str]:
        for pattern in self.BATCH_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip().upper()
        return None

    def _parse_expiry_date(self, text: str) -> Optional[str]:
        for pattern in self.EXPIRY_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _parse_mfg_date(self, text: str) -> Optional[str]:
        for pattern in self.MFG_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None
