"""Unit tests for GTIN check digit calculation and normalization."""
from genie.barcode.gtin import (
    calculate_gs1_check_digit,
    inspect_gs1_prefix,
    normalize_to_gtin14,
    validate_gtin_check_digit,
)


def test_calculate_gs1_check_digit():
    # Coartem GTIN-14 test vector: body 0615123456789 -> check digit 1
    body = "0615123456789"
    check = calculate_gs1_check_digit(body)
    assert check == 1

    # Standard EAN-13 example: 978014103614 -> check digit 4
    assert calculate_gs1_check_digit("978014103614") == 4


def test_validate_gtin_check_digit():
    # Correct check digit is 1
    assert validate_gtin_check_digit("06151234567891") is True
    # Wrong check digits fail
    assert validate_gtin_check_digit("06151234567890") is False
    assert validate_gtin_check_digit("06151234567893") is False
    assert validate_gtin_check_digit("invalid") is False


def test_normalize_to_gtin14():
    # 13-digit EAN normalized to 14 digits
    assert normalize_to_gtin14("6151234567891") == "06151234567891"

    error_raised = False
    try:
        normalize_to_gtin14("1234567890123")
    except ValueError:
        error_raised = True
    assert error_raised is True


def test_inspect_gs1_prefix():
    prefix, hint = inspect_gs1_prefix("06151234567891")
    assert prefix == "615"
    assert hint == "GS1 Nigeria"
