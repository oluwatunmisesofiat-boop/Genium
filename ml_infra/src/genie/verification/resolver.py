"""Stateless candidate product resolution via Backend-supplied candidate metadata or Qdrant vector metadata."""
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from genie.schemas.scan import ReferenceCandidateInput
from genie.storage.vector_store import VectorStore


@dataclass
class ResolvedCandidate:
    brand_name: str
    manufacturer: str
    gtin_list: List[str] = field(default_factory=list)
    primary_nafdac_number: Optional[str] = None
    version_label: Optional[str] = None
    is_legacy_version: bool = False
    in_transition_window: bool = True
    confidence: float = 0.0
    resolution_method: str = "unknown"  # backend_candidate | gtin | nafdac | vector
    nafdac_status: str = "active"
    batch_number_regex: Optional[str] = None
    reference_image_base64: Optional[str] = None
    reference_image_path: Optional[str] = None
    semantic_regions: Optional[Dict[str, Any]] = None


class ProductResolver:
    """Resolves scan signals against Backend-supplied metadata or VectorStore payloads."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @staticmethod
    def _from_metadata(
        meta: Dict[str, Any],
        confidence: float,
        method: str,
        current_date: date,
    ) -> ResolvedCandidate:
        status = str(meta.get("status", "active"))
        is_legacy = status == "superseded"
        in_window = True

        valid_to_str = meta.get("valid_to")
        if is_legacy and valid_to_str:
            valid_to = date.fromisoformat(str(valid_to_str)) if isinstance(valid_to_str, str) else valid_to_str
            shelf_days = int(meta.get("shelf_life_months", 24)) * 30
            days_since_expiry = (current_date - valid_to).days
            in_window = days_since_expiry <= shelf_days

        return ResolvedCandidate(
            brand_name=str(meta.get("brand_name") or "Unknown Product"),
            manufacturer=str(meta.get("manufacturer") or "Unknown Manufacturer"),
            gtin_list=[str(g) for g in (meta.get("gtin_list") or [])],
            primary_nafdac_number=meta.get("nafdac_number"),
            version_label=meta.get("version_label"),
            is_legacy_version=is_legacy,
            in_transition_window=in_window,
            confidence=confidence,
            resolution_method=method,
            nafdac_status=str(meta.get("nafdac_status", "active")),
            batch_number_regex=meta.get("batch_number_regex"),
            reference_image_path=meta.get("storage_path"),
            semantic_regions=meta.get("semantic_regions"),
        )

    async def resolve(
        self,
        gtin: Optional[str] = None,
        nafdac_number: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        reference_candidate: Optional[ReferenceCandidateInput] = None,
        current_date: Optional[date] = None,
    ) -> Optional[ResolvedCandidate]:
        """Executes hierarchical resolution using Backend candidate metadata or VectorStore."""
        curr_d = current_date or date.today()

        # Priority 0: Explicit candidate metadata + reference image supplied by Backend
        if reference_candidate is not None:
            return ResolvedCandidate(
                brand_name=reference_candidate.brand_name or "Unknown Product",
                manufacturer=reference_candidate.manufacturer or "Unknown Manufacturer",
                gtin_list=list(reference_candidate.gtin_list),
                primary_nafdac_number=reference_candidate.primary_nafdac_number,
                version_label=reference_candidate.version_label,
                is_legacy_version=reference_candidate.is_legacy_version,
                in_transition_window=reference_candidate.in_transition_window,
                confidence=0.98,
                resolution_method="backend_candidate",
                nafdac_status=reference_candidate.nafdac_status,
                batch_number_regex=reference_candidate.batch_number_regex,
                reference_image_base64=reference_candidate.reference_image_base64,
                reference_image_path=reference_candidate.reference_image_path,
                semantic_regions=reference_candidate.semantic_regions,
            )

        # Priority 1: VectorStore pre-filtered by GTIN
        if gtin and embedding:
            clean_gtin = gtin.strip()
            matches = await self.vector_store.search(
                embedding, top_k=1, filter_metadata={"gtin_list": clean_gtin}
            )
            if matches:
                return self._from_metadata(matches[0].metadata, 0.98, "gtin", curr_d)

        # Priority 2: VectorStore pre-filtered by NAFDAC number
        if nafdac_number and embedding:
            clean_nafdac = nafdac_number.strip().upper()
            matches = await self.vector_store.search(
                embedding, top_k=1, filter_metadata={"nafdac_number": clean_nafdac}
            )
            if matches:
                return self._from_metadata(matches[0].metadata, 0.92, "nafdac", curr_d)

        # Priority 3: Global visual vector similarity search
        if embedding:
            matches = await self.vector_store.search(embedding, top_k=1)
            if matches and matches[0].score >= 0.75:
                return self._from_metadata(
                    matches[0].metadata, float(matches[0].score), "vector", curr_d
                )

        return None
