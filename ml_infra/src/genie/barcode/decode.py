"""Multi-decoder ensemble for 1D barcodes and 2D GS1 DataMatrix / QR codes."""
from dataclasses import dataclass
import re
import time
from typing import Any, Dict, Optional
import cv2
import numpy as np

from genie.barcode.digital_link import parse_digital_link_url
from genie.barcode.enhance import enhance_barcode_crop
from genie.barcode.gs1_parser import GS1PayloadParser
from genie.barcode.gtin import normalize_to_gtin14, validate_gtin_check_digit


@dataclass
class DecodeResult:
    raw_text: str
    symbology: str
    gtin: Optional[str] = None
    batch: Optional[str] = None
    expiry: Optional[str] = None
    serial: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    check_digit_valid: bool = True
    decoder_engine: str = "unknown"
    latency_ms: float = 0.0


class MultiDecoder:
    """
    Ensemble decoder evaluating zxing-cpp, pyzbar, OpenCV WeChat QR, and pylibdmtx.
    Executes in <150ms.
    """

    @classmethod
    def decode(cls, image_np: np.ndarray) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        variants = enhance_barcode_crop(image_np)

        # 1. Primary: zxing-cpp
        try:
            import zxingcpp
            for var in variants:
                results = zxingcpp.read_barcodes(var)
                if results:
                    r = results[0]
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return cls._format_result(
                        r.text,
                        str(r.format),
                        engine="zxing-cpp",
                        latency_ms=elapsed,
                    )
        except ImportError:
            pass

        # 2. Secondary: pyzbar
        try:
            from pyzbar import pyzbar
            for var in variants:
                barcodes = pyzbar.decode(var)
                if barcodes:
                    b = barcodes[0]
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return cls._format_result(
                        b.data.decode("utf-8", errors="ignore"),
                        b.type,
                        engine="pyzbar",
                        latency_ms=elapsed,
                    )
        except ImportError:
            pass

        # 3. Tertiary: OpenCV WeChat QR
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode()
            for var in variants:
                res, _ = detector.detectAndDecode(var)
                if res and res[0]:
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return cls._format_result(
                        res[0],
                        "QR_CODE",
                        engine="wechat_qrcode",
                        latency_ms=elapsed,
                    )
        except Exception:
            pass

        # 4. Quaternary: pylibdmtx for stubborn DataMatrix (bounded timeout to prevent stalls on large images)
        try:
            from pylibdmtx import pylibdmtx
            for var in variants[:3]:
                dms = pylibdmtx.decode(var, timeout=40, max_count=1)
                if dms:
                    d = dms[0]
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return cls._format_result(
                        d.data.decode("utf-8", errors="ignore"),
                        "DataMatrix",
                        engine="pylibdmtx",
                        latency_ms=elapsed,
                    )
        except ImportError:
            pass

        # 5. Fallback: OCR on human-readable digits below bars
        fallback_res = cls._fallback_ocr_digits(image_np)
        if fallback_res:
            elapsed = (time.perf_counter() - t0) * 1000.0
            fallback_res["latency_ms"] = elapsed
            return fallback_res

        return None

    @classmethod
    def _fallback_ocr_digits(cls, image_np: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fallback when bars are physically damaged but printed numbers are visible."""
        try:
            from genie.vision.ocr import PackagingOCR
            ocr = PackagingOCR()
            data = ocr.extract_text(image_np)
            full_text = data.get("full_text", "")
            # Look for 13 or 14-digit sequences
            matches = re.findall(r"\b(\d{13,14})\b", full_text)
            for m in matches:
                if validate_gtin_check_digit(m):
                    return {
                        "raw_text": m,
                        "symbology": "EAN_OCR_FALLBACK",
                        "gtin": normalize_to_gtin14(m),
                        "batch": None,
                        "expiry": None,
                        "serial": None,
                        "parsed": None,
                        "check_digit_valid": True,
                        "decoder_engine": "ocr_fallback",
                    }
        except Exception:
            pass
        return None

    @classmethod
    def _format_result(
        cls,
        text: str,
        symbology: str,
        engine: str = "unknown",
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        text_clean = text.strip()

        # GS1 Digital Link QR URL
        if text_clean.startswith("http://") or text_clean.startswith("https://"):
            dl = parse_digital_link_url(text_clean)
            return {
                "raw_text": text_clean,
                "symbology": "GS1_DIGITAL_LINK",
                "gtin": dl.get("gtin"),
                "batch": dl.get("batch"),
                "expiry": dl.get("expiry"),
                "serial": dl.get("serial"),
                "parsed": dl,
                "check_digit_valid": dl.get("gtin_valid", True),
                "decoder_engine": engine,
                "latency_ms": latency_ms,
            }

        # GS1 Bracketed or FNC1 Stream
        if ("(" in text_clean and ")" in text_clean) or "\x1d" in text_clean:
            parsed = GS1PayloadParser.parse(text_clean)
            return {
                "raw_text": text_clean,
                "symbology": symbology,
                "gtin": parsed.get("gtin"),
                "batch": parsed.get("batch"),
                "expiry": parsed.get("expiry"),
                "serial": parsed.get("serial"),
                "parsed": parsed,
                "check_digit_valid": parsed.get("checks", {}).get("gtin_checkdigit", True),
                "decoder_engine": engine,
                "latency_ms": latency_ms,
            }

        # Linear EAN-13, EAN-8, UPC-A
        digits = "".join(c for c in text_clean if c.isdigit())
        if len(digits) in (8, 12, 13, 14):
            valid_check = validate_gtin_check_digit(digits)
            return {
                "raw_text": text_clean,
                "symbology": symbology,
                "gtin": normalize_to_gtin14(digits) if valid_check else digits,
                "batch": None,
                "expiry": None,
                "serial": None,
                "parsed": None,
                "check_digit_valid": valid_check,
                "decoder_engine": engine,
                "latency_ms": latency_ms,
            }

        return {
            "raw_text": text_clean,
            "symbology": symbology,
            "gtin": None,
            "batch": None,
            "expiry": None,
            "serial": None,
            "parsed": None,
            "check_digit_valid": False,
            "decoder_engine": engine,
            "latency_ms": latency_ms,
        }
