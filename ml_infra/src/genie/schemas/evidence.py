"""Evidence and provenance schemas."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    url: str
    domain: str
    trust_tier: int = Field(..., ge=1, le=3)
    trust_label: str
    last_fetched_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str


class EvidenceClaim(BaseModel):
    field_name: str
    value: str
    source_url: str
    verbatim_quote: str
    verified_in_source: bool = False


class StagedClaim(BaseModel):
    id: Optional[UUID] = None
    product_name: str
    extracted_claims: List[EvidenceClaim] = Field(default_factory=list)
    evidence_score: float = 0.0
    status: str = "staging"  # staging, approved, rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
