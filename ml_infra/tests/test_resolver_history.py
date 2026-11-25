"""Tests for Phase 7: Stateless candidate product resolution and SCD Type 2 history transition windows."""
from datetime import date
import uuid
import pytest

from genie.schemas.scan import ReferenceCandidateInput
from genie.storage.vector_store import InMemoryVectorStore
from genie.verification.resolver import ProductResolver
from genie.verification.signals.history_signal import HistorySignalEvaluator


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_resolve_priority_0_backend_candidate(vector_store: InMemoryVectorStore):
    resolver = ProductResolver(vector_store)
    candidate = await resolver.resolve(
        reference_candidate=ReferenceCandidateInput(
            brand_name="Coartem 80/480",
            manufacturer="Novartis Pharma",
            primary_nafdac_number="04-1234",
            gtin_list=["06151234567891"],
            version_label="2025_Design_v2",
            is_legacy_version=False,
        )
    )

    assert candidate is not None
    assert candidate.brand_name == "Coartem 80/480"
    assert candidate.version_label == "2025_Design_v2"
    assert candidate.resolution_method == "backend_candidate"
    assert candidate.is_legacy_version is False


@pytest.mark.asyncio
async def test_resolve_priority_1_gtin_and_2_nafdac_vector_metadata(vector_store: InMemoryVectorStore):
    await vector_store.upsert(
        id=str(uuid.uuid4()),
        vector=[0.1] * 768,
        metadata={
            "brand_name": "Amartem Soft Gel",
            "manufacturer": "Geneith Pharmaceuticals",
            "nafdac_number": "A4-8921",
            "gtin_list": ["06151234567891"],
            "version_label": "2024_Blister_v1",
            "status": "active",
        },
    )

    resolver = ProductResolver(vector_store)
    by_gtin = await resolver.resolve(gtin="06151234567891", embedding=[0.1] * 768)
    assert by_gtin is not None
    assert by_gtin.brand_name == "Amartem Soft Gel"
    assert by_gtin.resolution_method == "gtin"

    by_nafdac = await resolver.resolve(nafdac_number=" a4-8921 ", embedding=[0.1] * 768)
    assert by_nafdac is not None
    assert by_nafdac.brand_name == "Amartem Soft Gel"
    assert by_nafdac.resolution_method == "nafdac"


@pytest.mark.asyncio
async def test_resolve_scd_type2_historical_version(vector_store: InMemoryVectorStore):
    today = date(2026, 11, 24)
    legacy_emb = [0.2] * 768

    await vector_store.upsert(
        id="legacy_recent",
        vector=legacy_emb,
        metadata={
            "brand_name": "Lonart Forte",
            "manufacturer": "Greenlife Pharmaceuticals",
            "nafdac_number": "B4-5555",
            "version_label": "2022_Classic_Pack",
            "status": "superseded",
            "valid_to": "2026-04-30",
            "shelf_life_months": 24,
        },
    )

    resolver = ProductResolver(vector_store)
    candidate = await resolver.resolve(embedding=legacy_emb, current_date=today)
    assert candidate is not None
    assert candidate.is_legacy_version is True
    assert candidate.in_transition_window is True

    hist_eval = HistorySignalEvaluator()
    sig_ok = hist_eval.evaluate(
        is_legacy_match=candidate.is_legacy_version,
        version_label=candidate.version_label,
        transition_active=candidate.in_transition_window,
    )
    assert sig_ok.score >= 0.90

    # Expired transition window (> 24 months past valid_to)
    sig_expired = hist_eval.evaluate(
        is_legacy_match=True,
        version_label="2018_Old_Pack",
        transition_active=False,
    )
    assert sig_expired.score < 0.50
