"""GS1 Digital Link URI parser with domain security sandboxing and serial cloning analytics."""
import hashlib
import hmac
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, List, Optional, Tuple

from genie.barcode.gtin import normalize_to_gtin14, validate_gtin_check_digit

# In-memory serial scan frequency tracker (backed by Redis in production)
_SERIAL_SCAN_CACHE: Dict[str, int] = {}
_HMAC_SALT = b"genie-anti-counterfeit-salt-2026"


def hash_serial_number(serial: str) -> str:
    """Computes a privacy-preserving cryptographic hash of the product serial number."""
    return hmac.new(_HMAC_SALT, serial.encode("utf-8"), hashlib.sha256).hexdigest()


def track_serial_scan(serial: str, threshold: int = 5) -> Tuple[bool, int]:
    """
    Records serial scan and checks for serial cloning replay attacks.
    Returns (duplicate_detected, current_count).
    """
    serial_hash = hash_serial_number(serial)
    count = _SERIAL_SCAN_CACHE.get(serial_hash, 0) + 1
    _SERIAL_SCAN_CACHE[serial_hash] = count
    is_duplicate = count > threshold
    return is_duplicate, count


def parse_digital_link_url(
    url: str,
    allowed_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Parses URLs like https://example.com/01/06151234567893/10/AB1234?17=271231
    and validates the domain against authorized manufacturer registries.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Domain whitelisting against official manufacturer domains
    domain_trusted = True
    if allowed_domains:
        domain_trusted = any(domain == d or domain.endswith("." + d) for d in allowed_domains)

    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    extracted: Dict[str, str] = {}

    idx = 0
    while idx < len(path_parts) - 1:
        ai = path_parts[idx]
        val = path_parts[idx + 1]
        extracted[ai] = val
        idx += 2

    query_params = parse_qs(parsed.query)
    for q_ai, vals in query_params.items():
        if vals:
            extracted[q_ai] = vals[0]

    gtin = extracted.get("01")
    gtin_valid = validate_gtin_check_digit(gtin) if gtin else False
    serial = extracted.get("21")

    duplicate_detected = False
    scan_count = 1
    if serial:
        duplicate_detected, scan_count = track_serial_scan(serial)

    return {
        "url": url,
        "domain": domain,
        "domain_trusted": domain_trusted,
        "gtin": normalize_to_gtin14(gtin) if gtin and gtin_valid else gtin,
        "batch": extracted.get("10"),
        "expiry": extracted.get("17"),
        "serial": serial,
        "serial_duplicate_detected": duplicate_detected,
        "serial_scan_count": scan_count,
        "raw_attributes": extracted,
        "gtin_valid": gtin_valid,
    }
