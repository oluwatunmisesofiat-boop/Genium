"""Source trust tiering, URL canonicalization, and SSRF security blocklist."""
from urllib.parse import urlparse
import re

BLOCKED_DOMAINS = {
    "jiji.ng",
    "aliexpress.com",
    "temu.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "quora.com",
    "olx.com",
}

TIER_1_DOMAINS = {
    "nafdac.gov.ng",
    "who.int",
    "gs1.org",
    "gs1ng.org",
    "fda.gov",
    "medicines.org.uk",
    "gsk.com",
    "novartis.com",
    "emzorpharma.com",
    "fidson.com",
    "getzpharma.com",
}

TIER_2_DOMAINS = {
    "drugs.com",
    "emdexonline.com",
    "dailymed.nlm.nih.gov",
    "medplusnigeria.com",
    "healthplusnigeria.com",
    "drugstoc.com",
}


def canonicalize_url(url: str) -> str:
    """Strips query parameters, tracking fragments, and www prefix."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{parsed.scheme}://{netloc}{parsed.path}".rstrip("/")


def is_blocked_domain(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return any(domain == b or domain.endswith("." + b) for b in BLOCKED_DOMAINS)


def get_trust_tier(url: str) -> int:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if any(domain == d or domain.endswith("." + d) for d in TIER_1_DOMAINS):
        return 1
    if any(domain == d or domain.endswith("." + d) for d in TIER_2_DOMAINS):
        return 2
    return 3
