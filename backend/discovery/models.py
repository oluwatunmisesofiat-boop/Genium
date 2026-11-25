"""Data models for Step 2 Multi-Angle Discovery & Trust Tiering."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DiscoverySeed:
    product_name: Optional[str] = None
    manufacturer: Optional[str] = None
    nafdac_number: Optional[str] = None
    gtin: Optional[str] = None
    country: str = "NG"

    def __post_init__(self) -> None:
        if not any(
            isinstance(v, str) and v.strip()
            for v in (self.product_name, self.nafdac_number, self.gtin)
        ):
            raise ValueError(
                "DiscoverySeed requires at least one of product_name, nafdac_number, or gtin"
            )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchQuery:
    angle: str
    query: str
    include_domains: Optional[List[str]] = None
    time_range: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RankedCandidate:
    url: str
    canonical_url: str
    domain: str
    title: str
    snippet: str
    trust_tier: int
    trust_label: str
    angles: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiscoveredSource:
    url: str
    canonical_url: str
    domain: str
    trust_tier: int
    trust_label: str
    angles: List[str]
    title: str
    content: str
    stage: str
    product: Optional[Dict] = None
    images: List[Dict] = field(default_factory=list)
    identifiers: Dict[str, List[str]] = field(default_factory=dict)
    is_smpc_pdf: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiscoveryBundle:
    seed: DiscoverySeed
    enriched_seed: DiscoverySeed
    greenbook_matches: List[Dict] = field(default_factory=list)
    queries_run: List[SearchQuery] = field(default_factory=list)
    search_errors: List[Dict] = field(default_factory=list)
    sources: List[DiscoveredSource] = field(default_factory=list)
    dropped_sources: List[Dict] = field(default_factory=list)
    failed_sources: List[Dict] = field(default_factory=list)
    image_candidates: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "seed": self.seed.to_dict(),
            "enriched_seed": self.enriched_seed.to_dict(),
            "greenbook_matches": self.greenbook_matches,
            "queries_run": [q.to_dict() for q in self.queries_run],
            "search_errors": self.search_errors,
            "sources": [s.to_dict() for s in self.sources],
            "dropped_sources": self.dropped_sources,
            "failed_sources": self.failed_sources,
            "image_candidates": self.image_candidates,
            "stats": self.stats,
        }
