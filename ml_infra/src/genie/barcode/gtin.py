"""GTIN validation, normalization, and prefix checks."""
import re
from typing import Optional, Tuple


def calculate_gs1_check_digit(digits: str) -> int:
    """Calculates the standard GS1 modulo-10 check digit for an 7, 11, 12, or 13-digit sequence."""
    total = 0
    # Process from right to left with weights 3 and 1
    reversed_digits = digits[::-1]
    for idx, char in enumerate(reversed_digits):
        weight = 3 if idx % 2 == 0 else 1
        total += int(char) * weight
    remainder = total % 10
    return 0 if remainder == 0 else 10 - remainder


def validate_gtin_check_digit(gtin_candidate: str) -> bool:
    """Validates the check digit for 8, 12, 13, or 14-digit GTINs."""
    digits = re.sub(r"\D", "", gtin_candidate)
    if len(digits) not in (8, 12, 13, 14):
        return False
    body = digits[:-1]
    expected_check = calculate_gs1_check_digit(body)
    return int(digits[-1]) == expected_check


def normalize_to_gtin14(gtin_candidate: str) -> str:
    """Normalizes an 8, 12, 13, or 14-digit GTIN to a standard 14-digit format by left-padding with zeros."""
    digits = re.sub(r"\D", "", gtin_candidate)
    if not validate_gtin_check_digit(digits):
        raise ValueError(f"Invalid GTIN check digit: {gtin_candidate}")
    return digits.zfill(14)


def inspect_gs1_prefix(gtin: str) -> Tuple[Optional[str], str]:
    """Returns the country or GS1 member organization hint from the GTIN prefix."""
    digits = re.sub(r"\D", "", gtin).lstrip("0")
    if digits.startswith("615"):
        return "615", "GS1 Nigeria"
    if digits.startswith(("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")):
        return digits[:3], "GS1 US / Canada"
    if digits.startswith(("50", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509")):
        return digits[:3], "GS1 UK"
    if digits.startswith(("400", "401", "402", "403", "404")):
        return digits[:3], "GS1 Germany"
    return digits[:3] if len(digits) >= 3 else None, "Other GS1 Member"
