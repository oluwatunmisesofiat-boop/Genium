"""Multi-angle search query planner for unknown product discovery."""
from typing import List, Optional
from pydantic import BaseModel


class PlannedQuery(BaseModel):
    angle: str
    query_string: str


def plan_discovery_queries(
    product_name: Optional[str] = None,
    nafdac_number: Optional[str] = None,
    gtin: Optional[str] = None,
) -> List[PlannedQuery]:
    queries: List[PlannedQuery] = []

    if product_name:
        queries.append(PlannedQuery(angle="regulatory_alerts", query_string=f"site:nafdac.gov.ng {product_name} recall OR alert"))
        queries.append(PlannedQuery(angle="manufacturer", query_string=f"{product_name} official package leaflet SmPC"))
        queries.append(PlannedQuery(angle="pharma_reference", query_string=f"{product_name} dosage form pack size Nigeria pharmacy"))
        queries.append(PlannedQuery(angle="packaging_alerts", query_string=f"{product_name} new pack OR packaging redesign Nigeria"))

    if nafdac_number:
        queries.append(PlannedQuery(angle="identifier_exact", query_string=f'"{nafdac_number}" NAFDAC'))

    if gtin:
        queries.append(PlannedQuery(angle="identifier_exact", query_string=f'"{gtin}" GTIN barcode'))

    return queries
