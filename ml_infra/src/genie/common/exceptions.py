"""Domain-specific exceptions for Genie verification."""


class GenieException(Exception):
    """Base exception for all Genie errors."""


class QualityGateError(GenieException):
    """Raised when an uploaded scan fails image quality requirements."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class BarcodeDecodeError(GenieException):
    """Raised when a barcode cannot be localized or decoded."""


class RegistryMismatchError(GenieException):
    """Raised when registered regulatory data directly contradicts the scan."""


class ProductNotFoundError(GenieException):
    """Raised when a product cannot be resolved to any indexed catalog item."""
