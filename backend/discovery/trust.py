"""URL canonicalization, counterfeit-risk blocklist, and domain trust tiering."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import RankedCandidate

TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "ref",
        "ref_src",
        "mc_cid",
        "mc_eid",
        "igshid",
        "_hsenc",
        "_hsmi",
    }
)

BLOCKED_DOMAINS = frozenset(
    {
        "jiji.ng",
        "jiji.com",
        "konga.com",
        "jumia.com.ng",
        "aliexpress.com",
        "temu.com",
        "indiamart.com",
        "alibaba.com",
        "made-in-china.com",
        "tradeindia.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
        "x.com",
        "twitter.com",
        "reddit.com",
        "quora.com",
        "youtube.com",
        "youtu.be",
        "linkedin.com",
    }
)

TIER1_REGULATORY_DOMAINS = frozenset(
    {
        "nafdac.gov.ng",
        "greenbook.nafdac.gov.ng",
        "who.int",
        "ema.europa.eu",
        "fda.gov",
        "mhra.gov.uk",
        "medicines.org.uk",
        "pcn.gov.ng",
    }
)

TIER1_MANUFACTURER_DOMAINS = frozenset(
    {
        "gsk.com",
        "novartis.com",
        "pfizer.com",
        "sanofi.com",
        "astrazeneca.com",
        "roche.com",
        "bayer.com",
        "merck.com",
        "msd.com",
        "abbott.com",
        "cipla.com",
        "sunpharma.com",
        "emzor.com",
        "emzorpharma.com",
        "fidson.com",
        "may-baker.com",
        "mayandbaker.com",
        "swipsa.com",
        "swisspharma.com.ng",
        "juhel.com",
        "neimethplc.com.ng",
        "mepha.com",
        "getzpharma.com",
        "shalina.com",
        "mankindpharma.com",
    }
)

TIER2_PHARMA_DOMAINS = frozenset(
    {
        "drugs.com",
        "dailymed.nlm.nih.gov",
        "bnf.nice.org.uk",
        "emdexonline.com",
        "rxlist.com",
        "medscape.com",
        "webmd.com",
        "medplusnigeria.com",
        "healthplusnigeria.com",
        "drugstoc.com",
        "onehealthng.com",
        "advantagehealthafrica.com",
        "mypharmacy.ng",
        "famacare.com",
        "pharmacyplus.ng",
        "netpharmacy.ng",
    }
)

_GENERIC_ORG_WORDS = frozenset(
    {
        "pharma",
        "pharmaceutical",
        "pharmaceuticals",
        "company",
        "limited",
        "ltd",
        "nigeria",
        "nig",
        "plc",
        "inc",
        "corp",
        "labs",
        "laboratory",
        "laboratories",
        "healthcare",
        "health",
        "care",
        "medical",
        "medicine",
        "medicines",
        "med",
        "drugs",
        "drug",
        "international",
        "global",
        "group",
        "industries",
        "consumer",
        "products",
        "bio",
        "life",
        "plus",
    }
)

_TIER_BASE_SCORE = {1: 100.0, 2: 60.0, 3: 25.0}

_COUNTERFEIT_OR_ALERT_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"counterfeit|fake|falsified|substandard|seized|seizure|arraigned|arrested|"
    r"prosecuted|raid|recall|recalled|confiscated|impounded|unregistered|"
    r"suspects?|police|court|ndlea|efcc"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)
_WHATSAPP_IMAGE_RE = re.compile(r"img[-_]\d{8}[-_]wa\d+", re.IGNORECASE)
_NEWS_HOST_TOKENS = (
    "news",
    "magazine",
    "tribune",
    "punchng",
    "vanguardngr",
    "guardian.ng",
    "premiumtimesng",
    "channelstv",
    "thisdaylive",
    "dailypost",
    "leadership.ng",
    "thenationonlineng",
    "businessday",
    "blueprint.ng",
    "nairametrics",
    "pulse.ng",
    "legit.ng",
    "reuters",
    "bbc.",
    "cnn.",
    "aljazeera",
    "apnews",
)


def is_disqualified_image_source(
    url: str,
    source_page_url: str = "",
    title_or_alt: str = "",
    angles: Optional[List[str]] = None,
) -> Optional[str]:
    """Return a rejection reason if an image comes from a counterfeit alert, news article, or chat forward."""
    if angles:
        clean_angles = {str(a).strip().lower() for a in angles if str(a).strip()}
        if clean_angles and clean_angles.issubset({"regulatory_alerts", "packaging_alerts"}):
            return "alert_source_image"

    combined = " ".join(part for part in (url, source_page_url, title_or_alt) if part)
    if combined and _COUNTERFEIT_OR_ALERT_RE.search(combined):
        return "counterfeit_or_alert_source"

    if url and _WHATSAPP_IMAGE_RE.search(url):
        return "messaging_evidence_photo"

    for candidate in (source_page_url, url):
        if not candidate:
            continue
        host = (urlparse(candidate.strip()).hostname or "").lower()
        if host and any(tok in host for tok in _NEWS_HOST_TOKENS):
            return "news_press_source"

    return None


def extract_domain(url: str) -> str:
    """Return normalized lowercase domain without leading www."""
    if not url or not isinstance(url, str):
        return ""
    host = (urlparse(url.strip()).hostname or "").lower()
    return host.removeprefix("www.")


def canonicalize_url(url: str) -> str:
    """Normalize URL for deduplication (strip www., fragments, tracking params, trailing slash)."""
    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return ""

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower().removeprefix("www.")
    port = f":{parsed.port}" if parsed.port and not (
        (scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)
    ) else ""
    netloc = f"{host}{port}"

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    kept_params = sorted(
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    )
    query = urlencode(kept_params, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _domain_in(domain: str, domain_set: frozenset[str]) -> bool:
    return any(domain == d or domain.endswith(f".{d}") for d in domain_set)


def _matches_manufacturer_domain(domain: str, manufacturer_hint: Optional[str]) -> bool:
    if not domain or not manufacturer_hint:
        return False
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", manufacturer_hint)
        if len(tok) >= 3 and tok.lower() not in _GENERIC_ORG_WORDS
    ]
    if not tokens:
        return False
    # Check primary registrable label(s) before TLD (e.g. "getzpharma" in "getzpharma.com")
    labels = [lbl for lbl in domain.split(".")[:-1] if lbl]
    primary = "".join(labels)
    return any(tok in primary for tok in tokens)


def classify_url(url: str, manufacturer_hint: Optional[str] = None) -> Tuple[int, str]:
    """Return (trust_tier, trust_label) for a URL. Tier 0 means blocked."""
    domain = extract_domain(url)
    if not domain:
        return 0, "blocked"
    if _domain_in(domain, BLOCKED_DOMAINS):
        return 0, "blocked"
    if _domain_in(domain, TIER1_REGULATORY_DOMAINS):
        return 1, "regulatory"
    if _domain_in(domain, TIER1_MANUFACTURER_DOMAINS) or _matches_manufacturer_domain(
        domain, manufacturer_hint
    ):
        return 1, "official_manufacturer"
    if _domain_in(domain, TIER2_PHARMA_DOMAINS):
        return 2, "vetted_pharma"
    return 3, "general_web"


def rank_and_select(
    hits: List[Dict],
    *,
    manufacturer_hint: Optional[str] = None,
    max_urls: int = 8,
    max_per_domain: int = 2,
) -> List[RankedCandidate]:
    """Deduplicate, filter blocked domains, score by trust + angle diversity, and cap per domain."""
    if max_urls <= 0:
        return []

    by_canonical: Dict[str, RankedCandidate] = {}
    angle_order: List[str] = []

    for hit in hits:
        raw_url = str(hit.get("url") or "").strip()
        canon = canonicalize_url(raw_url)
        if not canon:
            continue

        tier, label = classify_url(canon, manufacturer_hint=manufacturer_hint)
        if tier == 0:
            continue

        domain = extract_domain(canon)
        angle = str(hit.get("angle") or "general").strip()
        if angle and angle not in angle_order:
            angle_order.append(angle)

        title = " ".join(str(hit.get("title") or "").split())
        snippet = " ".join(str(hit.get("content") or hit.get("snippet") or "").split())

        if canon in by_canonical:
            cand = by_canonical[canon]
            if angle and angle not in cand.angles:
                cand.angles.append(angle)
            if not cand.title and title:
                cand.title = title
            if not cand.snippet and snippet:
                cand.snippet = snippet
            if raw_url.startswith("https://") and not cand.url.startswith("https://"):
                cand.url = raw_url
        else:
            by_canonical[canon] = RankedCandidate(
                url=raw_url,
                canonical_url=canon,
                domain=domain,
                title=title,
                snippet=snippet,
                trust_tier=tier,
                trust_label=label,
                angles=[angle] if angle else [],
            )

    candidates = list(by_canonical.values())
    for cand in candidates:
        base = _TIER_BASE_SCORE.get(cand.trust_tier, 25.0)
        corroboration = 10.0 * max(0, len(cand.angles) - 1)
        pdf_bonus = 5.0 if urlparse(cand.canonical_url).path.lower().endswith(".pdf") else 0.0
        cand.score = base + corroboration + pdf_bonus

    candidates.sort(key=lambda c: (-c.score, c.trust_tier, c.canonical_url))

    selected: List[RankedCandidate] = []
    selected_keys: set[str] = set()
    domain_counts: Dict[str, int] = {}

    # Pass 1: guarantee angle diversity by picking top candidate per angle
    for angle in angle_order:
        if len(selected) >= max_urls:
            break
        for cand in candidates:
            if cand.canonical_url in selected_keys:
                continue
            if angle not in cand.angles:
                continue
            if domain_counts.get(cand.domain, 0) >= max_per_domain:
                continue
            selected.append(cand)
            selected_keys.add(cand.canonical_url)
            domain_counts[cand.domain] = domain_counts.get(cand.domain, 0) + 1
            break

    # Pass 2: fill remaining slots strictly by descending score
    for cand in candidates:
        if len(selected) >= max_urls:
            break
        if cand.canonical_url in selected_keys:
            continue
        if domain_counts.get(cand.domain, 0) >= max_per_domain:
            continue
        selected.append(cand)
        selected_keys.add(cand.canonical_url)
        domain_counts[cand.domain] = domain_counts.get(cand.domain, 0) + 1

    selected.sort(key=lambda c: (-c.score, c.trust_tier, c.canonical_url))
    return selected
