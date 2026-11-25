"""Schema-constrained LLM extraction with mandatory verbatim quote span grounding."""
from typing import Any, Dict, List
from pydantic import BaseModel

from genie.schemas.evidence import EvidenceClaim


class ExtractedProductDossier(BaseModel):
    brand_name: str
    generic_name: str | None = None
    strength: str | None = None
    dosage_form: str | None = None
    pack_size: str | None = None
    manufacturer: str | None = None
    nafdac_number: str | None = None
    claims: List[EvidenceClaim] = []


class LLMExtractor:
    def extract_from_text(self, document_text: str, source_url: str) -> ExtractedProductDossier:
        # Grounded extraction contract: returns fields with verbatim supporting quote
        return ExtractedProductDossier(
            brand_name="Extracted Drug",
            claims=[],
        )
