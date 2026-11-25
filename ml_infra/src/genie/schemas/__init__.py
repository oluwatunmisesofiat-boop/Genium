"""Domain schemas for Genie AI."""
from genie.schemas.evidence import EvidenceClaim, SourceRecord, StagedClaim
from genie.schemas.product import ProductCreate, ProductRead
from genie.schemas.scan import ScanImage, ScanRequest
from genie.schemas.verdict import (
    SaferAlternative,
    SignalBreakdown,
    VerdictState,
    VerificationResponse,
)
from genie.schemas.version import PackagingSpec, ProductVersionCreate, ProductVersionRead

__all__ = [
    "ProductCreate",
    "ProductRead",
    "ProductVersionCreate",
    "ProductVersionRead",
    "PackagingSpec",
    "EvidenceClaim",
    "SourceRecord",
    "StagedClaim",
    "ScanRequest",
    "ScanImage",
    "VerdictState",
    "SignalBreakdown",
    "VerificationResponse",
    "SaferAlternative",
]
