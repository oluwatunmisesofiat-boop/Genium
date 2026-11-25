"""Structured GS1 Application Identifier (AI) parsing using biip and native parsers."""
import re
from typing import Any, Dict, Optional

from genie.barcode.gtin import inspect_gs1_prefix, normalize_to_gtin14, validate_gtin_check_digit


class GS1PayloadParser:
    """
    Parses GS1 DataMatrix, GS1-128, and GS1 QR element strings into structured dictionaries.
    Completed in <5ms.
    """

    FIXED_LENGTH_AIS = {
        "00": 18,  # SSCC
        "01": 14,  # GTIN
        "02": 14,  # Content GTIN
        "11": 6,   # Production date (YYMMDD)
        "12": 6,   # Due date (YYMMDD)
        "13": 6,   # Packaging date (YYMMDD)
        "15": 6,   # Best before date (YYMMDD)
        "17": 6,   # Expiration date (YYMMDD)
        "20": 2,   # Internal variant
    }

    VARIABLE_LENGTH_AIS = {
        "10": 20,  # Batch or lot number
        "21": 20,  # Serial number
        "30": 8,   # Count of items
        "37": 8,   # Number of units
        "240": 30, # Additional product ID
    }

    @classmethod
    def parse(cls, payload: str) -> Dict[str, Any]:
        """Unified parser handling bracketed and FNC1 streams via biip or native logic."""
        try:
            import biip
            parsed_biip = biip.parse(payload)
            if parsed_biip.gs1_message:
                ai_dict = {}
                for element in parsed_biip.gs1_message.as_elements():
                    ai_dict[element.ai.ai] = str(element.value)
                return cls._normalize_parsed_dict(ai_dict)
        except Exception:
            pass

        if "(" in payload and ")" in payload:
            return cls.parse_bracketed_string(payload)
        return cls.parse_fnc1_string(payload)

    @classmethod
    def parse_bracketed_string(cls, payload: str) -> Dict[str, Any]:
        pattern = re.compile(r"\((\d{2,4})\)([^()]+)")
        matches = pattern.findall(payload)
        result: Dict[str, str] = {}
        for ai, val in matches:
            result[ai] = val.strip()
        return cls._normalize_parsed_dict(result)

    @classmethod
    def parse_fnc1_string(cls, raw: str) -> Dict[str, Any]:
        result: Dict[str, str] = {}
        fnc1 = "\x1d"
        segments = raw.split(fnc1)

        for segment in segments:
            idx = 0
            while idx < len(segment):
                ai2 = segment[idx : idx + 2]
                if ai2 in cls.FIXED_LENGTH_AIS:
                    length = cls.FIXED_LENGTH_AIS[ai2]
                    result[ai2] = segment[idx + 2 : idx + 2 + length]
                    idx += 2 + length
                    continue
                elif ai2 in cls.VARIABLE_LENGTH_AIS:
                    result[ai2] = segment[idx + 2 :]
                    break

                ai3 = segment[idx : idx + 3]
                if ai3 in cls.VARIABLE_LENGTH_AIS:
                    result[ai3] = segment[idx + 3 :]
                    break

                idx += 1

        return cls._normalize_parsed_dict(result)

    @classmethod
    def _normalize_parsed_dict(cls, data: Dict[str, str]) -> Dict[str, Any]:
        gtin = data.get("01")
        expiry = data.get("17")
        batch = data.get("10")
        serial = data.get("21")
        prod_date = data.get("11")

        formatted_expiry = None
        if expiry and len(expiry) == 6:
            yy = int(expiry[0:2])
            mm = expiry[2:4]
            dd = expiry[4:6]
            year = 2000 + yy if yy < 70 else 1900 + yy
            if dd == "00":
                dd = "28"  # End of month handling per GS1 General Specifications
            formatted_expiry = f"{year}-{mm}-{dd}"

        gtin_valid = validate_gtin_check_digit(gtin) if gtin else False
        norm_gtin = normalize_to_gtin14(gtin) if gtin and gtin_valid else gtin

        prefix_code, prefix_name = inspect_gs1_prefix(norm_gtin) if norm_gtin else (None, "Unknown")

        return {
            "gtin": norm_gtin,
            "expiry": formatted_expiry,
            "raw_expiry": expiry,
            "batch": batch,
            "serial": serial,
            "production_date": prod_date,
            "prefix_code": prefix_code,
            "prefix_origin": prefix_name,
            "gs1_prefix_country": prefix_name,
            "check_digit_valid": gtin_valid,
            "raw_ais": data,
            "checks": {
                "gtin_checkdigit": gtin_valid,
                "has_batch": bool(batch),
                "has_expiry": bool(formatted_expiry),
            },
        }
