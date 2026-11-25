"""Product Version schemas for SCD Type 2 packaging lifecycles."""
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PackagingSpec(BaseModel):
    primary_color_hex: Optional[str] = None
    secondary_color_hex: Optional[str] = None
    has_hologram_seal: bool = False
    has_tamper_seal: bool = False
    has_micro_text: bool = False
    distinctive_features: List[str] = Field(default_factory=list)


class ProductVersionBase(BaseModel):
    version_label: str = Field(..., json_schema_extra={"example": "2024 Design with Holographic Seal"})
    valid_from: date
    valid_to: Optional[date] = None
    status: str = Field(default="active", json_schema_extra={"example": "active"})
    packaging_spec: PackagingSpec = Field(default_factory=PackagingSpec)
    batch_number_regex: Optional[str] = Field(None, json_schema_extra={"example": "^[A-Z]{2}[0-9]{4}$"})
    shelf_life_months: int = Field(default=24, json_schema_extra={"example": 24})


class ProductVersionCreate(ProductVersionBase):
    product_id: UUID
    evidence_ids: List[str] = Field(default_factory=list)


class ProductVersionRead(ProductVersionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    product_id: UUID
    evidence_ids: List[str]
    created_at: datetime
