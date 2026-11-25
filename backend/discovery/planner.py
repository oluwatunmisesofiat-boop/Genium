"""Seed enrichment from local Greenbook and multi-angle query expansion."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .models import DiscoverySeed, SearchQuery

_CORP_SUFFIX_RE = re.compile(
    r"\b(?:limited|ltd|plc|inc|corp|nigeria|company|pharmaceuticals?|pharma)\b",
    re.IGNORECASE,
)


def enrich_seed_from_greenbook(seed: DiscoverySeed, matches: List[Dict]) -> DiscoverySeed:
    """Fill missing seed fields from the top local Greenbook match without overwriting inputs."""
    if not matches:
        return seed
    top = matches[0]

    product_name = seed.product_name
    if not product_name:
        name = str(top.get("product_name") or "").strip()
        strength = str(top.get("strength") or "").strip()
        if name and strength and strength.lower() not in name.lower():
            product_name = f"{name} {strength}"
        elif name:
            product_name = name

    manufacturer = seed.manufacturer or (str(top.get("applicant_name") or "").strip() or None)
    nafdac_number = seed.nafdac_number or (str(top.get("nafdac_number") or "").strip() or None)

    return DiscoverySeed(
        product_name=product_name,
        manufacturer=manufacturer,
        nafdac_number=nafdac_number,
        gtin=seed.gtin,
        country=seed.country,
    )


def _short_manufacturer(manufacturer: Optional[str]) -> str:
    if not manufacturer:
        return ""
    stripped = _CORP_SUFFIX_RE.sub("", manufacturer)
    cleaned = " ".join(stripped.split())
    return cleaned or " ".join(manufacturer.split())


def build_queries(seed: DiscoverySeed, *, has_smpc_pdf: bool = False) -> List[SearchQuery]:
    """Build targeted multi-angle search queries from an enriched DiscoverySeed."""
    name = (seed.product_name or "").strip()
    nafdac = (seed.nafdac_number or "").strip()
    gtin = (seed.gtin or "").strip()
    subject = name or nafdac or gtin
    mfr = _short_manufacturer(seed.manufacturer)

    queries: List[SearchQuery] = []

    # 1. Regulatory alerts & recalls on NAFDAC
    if name:
        queries.append(
            SearchQuery(
                angle="regulatory_alerts",
                query=f'"{name}" NAFDAC recall OR alert OR counterfeit',
                include_domains=["nafdac.gov.ng"],
            )
        )
    elif nafdac:
        queries.append(
            SearchQuery(
                angle="regulatory_alerts",
                query=f'"{nafdac}" NAFDAC',
                include_domains=["nafdac.gov.ng"],
            )
        )

    # 2. Official manufacturer page / packaging / SmPC
    if has_smpc_pdf:
        mfr_query = (
            f'"{subject}" {mfr} packaging OR pack OR tablets'
            if mfr
            else f'"{subject}" official manufacturer packaging OR pack'
        )
    else:
        mfr_query = (
            f'"{subject}" {mfr} official leaflet OR SmPC OR packaging'
            if mfr
            else f'"{subject}" official manufacturer leaflet OR SmPC OR packaging'
        )
    queries.append(SearchQuery(angle="manufacturer", query=mfr_query))

    # 3. Nigerian pharmacies & product pages (unquoted loose tokens so e.g. "20mg + 120mg" matches "20/120")
    loose_subject = " ".join(re.sub(r"[/+]+", " ", subject).split())
    queries.append(
        SearchQuery(
            angle="pharma_reference",
            query=f"{loose_subject} pharmacy Nigeria price OR pack OR tablets",
        )
    )

    # 4. Packaging updates / counterfeit alerts specifically mentioning the product
    queries.append(
        SearchQuery(
            angle="packaging_alerts",
            query=f'"{subject}" "new pack" OR "fake" OR "counterfeit" Nigeria',
        )
    )

    # 5. Exact identifier lookup when NAFDAC number or GTIN is known
    if gtin and nafdac:
        id_query = f'"{nafdac}" OR "{gtin}" Nigeria'
    elif nafdac:
        id_query = f'"{nafdac}" NAFDAC'
    elif gtin:
        id_query = f'"{gtin}"'
    else:
        id_query = ""
    if id_query:
        queries.append(SearchQuery(angle="identifier_exact", query=id_query))

    return queries


_TRAILING_FORM_RE = re.compile(
    r"\s+\b(?:tablets?|tabs?|capsules?|caps?|caplets?)\b\s*$",
    re.IGNORECASE,
)
_STRENGTH_TOKEN_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*[/+]\s*\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|ml|iu|%)?\b",
    re.IGNORECASE,
)


def _brand_only_term(subject: str) -> str:
    """Extract the core brand term (e.g. 'Coartem 20/120 Tablets' -> 'Coartem') for high-recall site: image queries."""
    no_form = _TRAILING_FORM_RE.sub("", subject or "").strip()
    no_strength = _STRENGTH_TOKEN_RE.sub(" ", no_form)
    cleaned = " ".join(no_strength.split()).strip(" -/+")
    return cleaned or no_form or (subject or "").strip()


def _image_exclusions(seed: DiscoverySeed, greenbook_match: Optional[Dict]) -> List[str]:
    subject = (seed.product_name or seed.nafdac_number or seed.gtin or "").strip()
    target_context = " ".join(
        [
            subject,
            str((greenbook_match or {}).get("form_name") or ""),
            str((greenbook_match or {}).get("product_name") or ""),
        ]
    ).lower()
    exclusions: List[str] = []
    if "dispersible" not in target_context:
        exclusions.append("-dispersible")
    if "syrup" not in target_context and "suspension" not in target_context:
        exclusions.append("-syrup")
    return exclusions


def build_image_query(
    seed: DiscoverySeed,
    *,
    greenbook_match: Optional[Dict] = None,
) -> str:
    """Build a focused query for open-web product packaging image search."""
    name = (seed.product_name or "").strip()
    nafdac = (seed.nafdac_number or "").strip()
    gtin = (seed.gtin or "").strip()
    subject = name or nafdac or gtin
    mfr = _short_manufacturer(seed.manufacturer)
    mfr_token = mfr.split()[0] if mfr else ""
    if mfr_token and mfr_token.lower() not in subject.lower():
        base = f"{subject} {mfr_token} pack box"
    else:
        base = f"{subject} medicine pack box"

    exclusions = _image_exclusions(seed, greenbook_match)
    return f"{base} {' '.join(exclusions)}".strip()


def build_trusted_image_queries(
    seed: DiscoverySeed,
    *,
    trusted_domains: List[str],
    greenbook_match: Optional[Dict] = None,
    max_domains_per_query: int = 5,
) -> List[str]:
    """Build lean site:-scoped image queries over verified pharmacy/manufacturer domains.

    Omits open-web terms like `pack box` and extra manufacturer words so pharmacy
    product pages (e.g. `Coartem 20/120mg x24`) actually match.
    """
    cleaned: List[str] = []
    for dom in trusted_domains or []:
        d = (dom or "").strip().lower().removeprefix("www.")
        if d and d not in cleaned:
            cleaned.append(d)
    if not cleaned:
        return [build_image_query(seed, greenbook_match=greenbook_match)]

    subject = (seed.product_name or seed.nafdac_number or seed.gtin or "").strip()
    lean_subject = _brand_only_term(subject)
    exclusions = _image_exclusions(seed, greenbook_match)
    base = f"{lean_subject} {' '.join(exclusions)}".strip()

    chunk_size = max(1, int(max_domains_per_query))
    queries: List[str] = []
    for i in range(0, len(cleaned), chunk_size):
        chunk = cleaned[i : i + chunk_size]
        site_clause = " OR ".join(f"site:{d}" for d in chunk)
        queries.append(f"{base} ({site_clause})")
    return queries


def build_trusted_pharmacy_web_queries(
    seed: DiscoverySeed,
    *,
    trusted_domains: List[str],
    max_domains_per_query: int = 5,
) -> List[SearchQuery]:
    """Build site:-scoped web queries to find actual product pages on verified pharmacies."""
    cleaned: List[str] = []
    for dom in trusted_domains or []:
        d = (dom or "").strip().lower().removeprefix("www.")
        if d and d not in cleaned:
            cleaned.append(d)
    if not cleaned:
        return []

    subject = (seed.product_name or seed.nafdac_number or seed.gtin or "").strip()
    brand = _brand_only_term(subject)
    if not brand:
        return []

    chunk_size = max(1, int(max_domains_per_query))
    out: List[SearchQuery] = []
    for i in range(0, len(cleaned), chunk_size):
        chunk = cleaned[i : i + chunk_size]
        out.append(
            SearchQuery(
                angle="trusted_pharmacy",
                query=brand,
                include_domains=chunk,
            )
        )
    return out
