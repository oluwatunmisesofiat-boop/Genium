"""Unit tests for structured GS1 parsing and Digital Link sandboxing."""
import pytest

from genie.barcode.digital_link import (
    hash_serial_number,
    parse_digital_link_url,
    track_serial_scan,
)
from genie.barcode.gs1_parser import GS1PayloadParser


def test_parse_bracketed_gs1_string():
    raw = "(01)06151234567891(17)271231(10)AB1234(21)98765"
    parsed = GS1PayloadParser.parse(raw)

    assert parsed["gtin"] == "06151234567891"
    assert parsed["expiry"] == "2027-12-31"
    assert parsed["batch"] == "AB1234"
    assert parsed["serial"] == "98765"
    assert parsed["prefix_code"] == "615"
    assert parsed["prefix_origin"] == "GS1 Nigeria"
    assert parsed["checks"]["gtin_checkdigit"] is True


def test_parse_fnc1_gs1_string():
    raw = "010615123456789117271231\x1d10BATCH99"
    parsed = GS1PayloadParser.parse_fnc1_string(raw)

    assert parsed["gtin"] == "06151234567891"
    assert parsed["expiry"] == "2027-12-31"
    assert parsed["batch"] == "BATCH99"


def test_parse_expiry_end_of_month():
    # DD=00 indicates end of month
    raw = "(01)06151234567891(17)260200"
    parsed = GS1PayloadParser.parse_bracketed_string(raw)

    assert parsed["expiry"] == "2026-02-28"


def test_digital_link_parsing_and_whitelisting():
    url = "https://novartis.com/01/06151234567891/10/LOT555?17=270630"
    res = parse_digital_link_url(url, allowed_domains=["novartis.com", "gsk.com"])

    assert res["domain"] == "novartis.com"
    assert res["domain_trusted"] is True
    assert res["gtin"] == "06151234567891"
    assert res["batch"] == "LOT555"
    assert res["expiry"] == "270630"


def test_digital_link_untrusted_domain():
    url = "https://fake-pharma-scam.net/01/06151234567891/10/LOT555"
    res = parse_digital_link_url(url, allowed_domains=["novartis.com"])

    assert res["domain"] == "fake-pharma-scam.net"
    assert res["domain_trusted"] is False


def test_serial_cloning_detection():
    test_serial = "SERIAL-XYZ-9999"
    h1 = hash_serial_number(test_serial)
    assert len(h1) == 64

    # Scans up to threshold (5) should not flag duplicate
    for _ in range(5):
        dup, count = track_serial_scan(test_serial, threshold=5)

    # 6th scan triggers duplicate detection
    dup, count = track_serial_scan(test_serial, threshold=5)
    assert dup is True
    assert count >= 6
