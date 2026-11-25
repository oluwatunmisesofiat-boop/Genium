"""Unit tests for Pydantic v2 domain schemas."""
import uuid
from datetime import date, datetime
from genie.schemas.product import ProductCreate, ProductCategory, ProductRead
from genie.schemas.version import PackagingSpec, ProductVersionCreate
from genie.schemas.scan import ScanImage, ScanRequest
from genie.schemas.verdict import (
    SaferAlternative,
    SignalBreakdown,
    VerdictState,
    VerificationResponse,
)


def test_product_schemas():
    prod = ProductCreate(
        brand_name="Coartem",
        generic_name="Artemether / Lumefantrine",
        manufacturer="Novartis Pharma AG",
        category=ProductCategory.PHARMACEUTICALS_ANTIMALARIALS,
        active_ingredients=["Artemether", "Lumefantrine"],
        strength="80 mg / 480 mg",
        primary_nafdac_number="A4-6238",
        gtin_list=["06151234567893"],
    )
    assert prod.brand_name == "Coartem"
    assert prod.status == "staging"


def test_version_schemas():
    v = ProductVersionCreate(
        product_id=uuid.uuid4(),
        version_label="2024 Seal Redesign",
        valid_from=date(2024, 1, 1),
        packaging_spec=PackagingSpec(
            has_hologram_seal=True,
            has_tamper_seal=True,
            distinctive_features=["Green NAFDAC banner", "High-gloss brand mark"],
        ),
    )
    assert v.version_label == "2024 Seal Redesign"
    assert v.packaging_spec.has_hologram_seal is True


def test_verification_response_schema():
    resp = VerificationResponse(
        verdict=VerdictState.CONSISTENT_WITH_GENUINE,
        confidence=0.94,
        matched_product_name="Coartem",
        matched_version_label="2024 Seal Redesign",
        is_legacy_version=False,
        explanation="Product packaging aligns with genuine reference records.",
        signals=SignalBreakdown(
            barcode_score=0.95,
            nafdac_score=0.98,
            visual_similarity_score=0.91,
        ),
        execution_time_ms=1840.5,
    )
    assert resp.verdict == VerdictState.CONSISTENT_WITH_GENUINE
    assert resp.confidence == 0.94
