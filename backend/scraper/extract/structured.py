"""Structured pharma, product, identifier, and candidate image extraction.

Runs in the same lxml parse pass as reader_document with zero extra network calls:
- Parses schema.org/Product and schema.org/Drug JSON-LD (including @graph).
- Extracts and validates NAFDAC registration numbers and GS1 GTIN/EAN/UPC barcodes.
- Extracts a priority-ranked, junk-filtered list of candidate product images.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

GTIN_KEYS = ("gtin", "gtin8", "gtin12", "gtin13", "gtin14")
ENTITY_TYPES = ("Drug", "Product")
MAX_DESCRIPTION_CHARS = 500
MAX_CANDIDATE_IMAGES = 8
MIN_IMAGE_DIM = 150
MAX_BANNER_ASPECT = 3.5

# NAFDAC numbers near an explicit "NAFDAC" label (covers A4-1234, A4-100160, B4-1234L, 04-1234, etc.)
_NAFDAC_LABELED_RE = re.compile(
    r"\bNAFDAC\b[^\n]{0,45}?\b((?:[A-Z]\d|\d{2})\s*[-–—]\s*\d{4,6}[A-Z]?)\b",
    re.IGNORECASE,
)
# Standalone alphanumeric pharma prefix (e.g., A4-1234, A4-100160, B4-5678L) — excludes pure MM-YYYY dates
_NAFDAC_STANDALONE_RE = re.compile(
    r"\b([A-DF]\d-\d{4,6}[A-Z]?)\b",
)
# Labeled GTIN / EAN / UPC / Barcode numbers in text
_GTIN_LABELED_RE = re.compile(
    r"\b(?:GTIN(?:-\d{1,2})?|EAN(?:-\d{1,2})?|UPC(?:-[AE])?|Barcode)\b[\s:.\-#]*([\d][\d\s\-]{6,17}\d)\b",
    re.IGNORECASE,
)

_BAD_IMG_EXTS = (".svg", ".ico", ".gif")
_JUNK_IMG_RE = re.compile(
    r"(?:^|[/_\-.])("
    r"logo|icon|favicon|sprite|avatar|banner|badge|payment|visa|mastercard|verve|paypal|"
    r"placeholder|loading|spinner|loader|social|whatsapp|facebook|instagram|twitter|"
    r"linkedin|youtube|tiktok|pixel|tracking|analytics|spacer|blank|1x1|arrow|chevron|"
    r"star|rating|cart|basket|search|menu|close|hamburger"
    r")(?:[/_\-.]|$)",
    re.IGNORECASE,
)
_JUNK_ALT_RE = re.compile(
    r"\b(logo|icon|avatar|banner|badge|payment|visa|mastercard|paypal|placeholder|"
    r"loading|spinner|facebook|instagram|twitter|whatsapp|youtube)\b",
    re.IGNORECASE,
)
_EXCLUDED_IMG_ANCESTORS = (
    "ancestor::header or ancestor::nav or ancestor::footer or ancestor::aside "
    "or ancestor::script or ancestor::style or ancestor::noscript "
    "or ancestor::*[@role='navigation' or @role='banner' or @role='contentinfo' or @role='search']"
)


def _clean_text(value: object, max_len: Optional[int] = None) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if max_len and len(cleaned) > max_len:
        return cleaned[:max_len].rsplit(" ", 1)[0].rstrip() + "…"
    return cleaned


def _extract_name(value: object) -> Optional[str]:
    if isinstance(value, dict):
        return _clean_text(value.get("name"))
    return _clean_text(value)


def _extract_string_list(value: object) -> List[str]:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        out: List[str] = []
        for entry in value:
            text = _extract_name(entry)
            if text and text not in out:
                out.append(text)
        return out
    name = _extract_name(value)
    return [name] if name else []


def _extract_jsonld_images(value: object) -> List[str]:
    urls: List[str] = []

    def _add(item: object) -> None:
        if isinstance(item, str):
            u = item.strip()
            if u and u not in urls:
                urls.append(u)
        elif isinstance(item, dict):
            u = item.get("url") or item.get("contentUrl")
            if isinstance(u, str) and u.strip() and u.strip() not in urls:
                urls.append(u.strip())
        elif isinstance(item, list):
            for sub in item:
                _add(sub)

    _add(value)
    return urls


def _normalize_types(raw_types: object) -> List[str]:
    if not isinstance(raw_types, list):
        raw_types = [raw_types]
    out: List[str] = []
    for t in raw_types:
        if isinstance(t, str) and t.strip():
            # Handle both "Product" and "https://schema.org/Product"
            out.append(t.strip().rsplit("/", 1)[-1])
    return out


def extract_product(html: str) -> Optional[Dict]:
    """Parse HTML and return the first schema.org Drug or Product JSON-LD object."""
    if not html or "application/ld+json" not in html:
        return None
    try:
        from lxml import html as lxml_html  # noqa: PLC0415

        tree = lxml_html.fromstring(html)
    except Exception:  # noqa: BLE001
        return None
    return product_from_items(jsonld_items(tree))


def product_from_items(items: Iterable[object]) -> Optional[Dict]:
    """First schema.org Drug or Product from parsed JSON-LD objects (including @graph).

    Prefers ``Drug`` over ``Product`` if both appear in the same page.
    """
    best_product: Optional[Dict] = None
    for item in items:
        if not isinstance(item, dict):
            continue
        graph = item.get("@graph")
        candidates = [g for g in graph if isinstance(g, dict)] if isinstance(graph, list) else []
        for cand in candidates + [item]:
            types = _normalize_types(cand.get("@type"))
            if "Drug" in types:
                parsed = read_product(cand, entity_type="Drug")
                if parsed:
                    return parsed
            elif "Product" in types and best_product is None:
                parsed = read_product(cand, entity_type="Product")
                if parsed:
                    best_product = parsed
    return best_product


def jsonld_items(tree):
    """Yield parsed JSON-LD dict objects (flattening top-level lists and @graph)."""
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        raw = (script.text or "").strip()
        raw = raw.replace("<![CDATA[", "").replace("]]>", "")
        try:
            data = json.loads(raw, parse_constant=lambda _c: None)
        except ValueError:
            continue

        if not isinstance(data, list):
            data = [data]
        for item in data:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                for g in graph:
                    if isinstance(g, dict):
                        yield g
            yield item


def read_product(item: Dict, entity_type: str = "Product") -> Dict:
    """Normalize schema.org Product or Drug fields into a flat dict."""
    product: Dict = {}

    name = _clean_text(item.get("name"))
    if name:
        product["name"] = name

    product.update(read_offer(item.get("offers")))

    brand = _extract_name(item.get("brand"))
    if brand:
        product["brand"] = brand

    manufacturer = _extract_name(item.get("manufacturer"))
    if manufacturer:
        product["manufacturer"] = manufacturer

    non_prop = _clean_text(item.get("nonProprietaryName"))
    if non_prop:
        product["nonProprietaryName"] = non_prop

    ingredients = _extract_string_list(item.get("activeIngredient"))
    if ingredients:
        product["activeIngredient"] = ingredients

    dosage_form = _clean_text(item.get("dosageForm"))
    if dosage_form:
        product["dosageForm"] = dosage_form

    description = _clean_text(item.get("description"), max_len=MAX_DESCRIPTION_CHARS)
    if description:
        product["description"] = description

    if item.get("sku") is not None and str(item["sku"]).strip():
        product["sku"] = str(item["sku"]).strip()

    if item.get("mpn") is not None and str(item["mpn"]).strip():
        product["mpn"] = str(item["mpn"]).strip()

    gtins: List[str] = []
    for source_dict in _gtin_containers(item):
        for key in GTIN_KEYS:
            value = source_dict.get(key)
            if value is not None:
                val = str(value).strip()
                if val and val not in gtins:
                    gtins.append(val)
    if gtins:
        product["gtin"] = gtins

    images = _extract_jsonld_images(item.get("image"))
    if images:
        product["images"] = images

    if product:
        product["entityType"] = entity_type
    return product


def _gtin_containers(item: Dict) -> List[Dict]:
    containers = [item]
    offers = item.get("offers")
    if isinstance(offers, dict):
        containers.append(offers)
    elif isinstance(offers, list):
        containers.extend(o for o in offers if isinstance(o, dict))
    return containers


def read_offer(offers) -> Dict:
    """Extract price, currency, and availability from schema.org Offer or AggregateOffer."""
    if not isinstance(offers, list):
        offers = [offers]

    for offer in offers:
        if not isinstance(offer, dict):
            continue

        price = offer.get("price", offer.get("lowPrice"))
        if price is None or isinstance(price, bool):
            continue
        price = str(price).strip().replace(",", ".")
        if not price:
            continue

        result = {"price": price}
        if offer.get("priceCurrency"):
            result["priceCurrency"] = str(offer["priceCurrency"])
        if offer.get("availability"):
            result["availability"] = str(offer["availability"]).split("/")[-1]
        return result

    return {}


# ---------------------------------------------------------------- Identifiers


def is_valid_gtin(code: str) -> bool:
    """Validate a GS1 GTIN-8, GTIN-12 (UPC), GTIN-13 (EAN), or GTIN-14 check digit."""
    digits = re.sub(r"[\s\-]", "", str(code or ""))
    if not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return False
    if set(digits) == {"0"}:
        return False
    payload = digits[:-1]
    check = int(digits[-1])
    total = 0
    for idx, ch in enumerate(reversed(payload)):
        weight = 3 if idx % 2 == 0 else 1
        total += int(ch) * weight
    expected = (10 - (total % 10)) % 10
    return expected == check


def extract_nafdac_numbers(text: str) -> List[str]:
    """Extract normalized NAFDAC registration numbers from page text."""
    if not text:
        return []
    found: List[str] = []

    def _add(raw: str) -> None:
        normalized = re.sub(r"\s*[-–—]\s*", "-", raw.strip().upper())
        if normalized not in found:
            found.append(normalized)

    for match in _NAFDAC_LABELED_RE.finditer(text):
        _add(match.group(1))
    for match in _NAFDAC_STANDALONE_RE.finditer(text):
        _add(match.group(1))
    return found


def extract_gtins(text: str, product: Optional[Dict] = None) -> List[str]:
    """Extract GS1-validated GTIN/EAN/UPC barcodes from JSON-LD and page text."""
    found: List[str] = []

    def _add(raw: str) -> None:
        digits = re.sub(r"[\s\-]", "", str(raw or ""))
        if is_valid_gtin(digits) and digits not in found:
            found.append(digits)

    if product and isinstance(product.get("gtin"), list):
        for g in product["gtin"]:
            _add(g)

    if text:
        for match in _GTIN_LABELED_RE.finditer(text):
            _add(match.group(1))

    return found


def extract_identifiers(text: str, product: Optional[Dict] = None) -> Dict[str, List[str]]:
    """Pre-extract NAFDAC registration numbers and validated GTIN barcodes."""
    return {
        "nafdac_numbers": extract_nafdac_numbers(text),
        "gtins": extract_gtins(text, product=product),
    }


# ---------------------------------------------------------------- Candidate Images


def _parse_int(val: object) -> Optional[int]:
    if val is None:
        return None
    m = re.match(r"^\s*(\d+)", str(val))
    return int(m.group(1)) if m else None


def _resolve_http_url(raw_url: str, base_url: str) -> Optional[str]:
    if not raw_url or not isinstance(raw_url, str):
        return None
    raw_url = raw_url.strip()
    if not raw_url or raw_url.startswith(("data:", "javascript:", "blob:", "#")):
        return None
    resolved = urljoin(base_url, raw_url) if base_url else raw_url
    parsed = urlparse(resolved)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return resolved


def _is_junk_image(
    url: str,
    alt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in _BAD_IMG_EXTS):
        return True
    if _JUNK_IMG_RE.search(path):
        return True
    if alt and _JUNK_ALT_RE.search(alt):
        return True
    if width is not None and width < MIN_IMAGE_DIM:
        return True
    if height is not None and height < MIN_IMAGE_DIM:
        return True
    if width and height:
        ratio = max(width / height, height / width)
        if ratio > MAX_BANNER_ASPECT:
            return True
    return False


def _img_src_from_element(el) -> Optional[str]:
    for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-large_image"):
        val = (el.get(attr) or "").strip()
        if val and not val.startswith("data:"):
            return val
    return None


def extract_images(
    tree,
    url: str = "",
    product: Optional[Dict] = None,
    meta: Optional[Dict] = None,
    max_images: int = MAX_CANDIDATE_IMAGES,
) -> List[Dict]:
    """Extract a priority-ranked, junk-filtered list of candidate product images.

    Priority:
      1. JSON-LD Product / Drug ``image``
      2. OpenGraph / Twitter meta tags (``og:image``, ``twitter:image``)
      3. Main-content ``<img>`` elements (excluding nav/header/footer/aside)
    """
    candidates: List[Dict] = []
    seen: set = set()

    def _add(
        raw_url: Optional[str],
        source: str,
        alt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if len(candidates) >= max_images or not raw_url:
            return
        resolved = _resolve_http_url(raw_url, url)
        if not resolved or resolved in seen:
            return
        clean_alt = _clean_text(alt) or ""
        if _is_junk_image(resolved, alt=clean_alt, width=width, height=height):
            return
        seen.add(resolved)
        candidates.append(
            {
                "url": resolved,
                "source": source,
                "alt": clean_alt,
                "width": width,
                "height": height,
            }
        )

    # 1. JSON-LD Product / Drug images
    if product and isinstance(product.get("images"), list):
        for img_url in product["images"]:
            _add(img_url, "jsonld")

    # 2. OpenGraph / Twitter preview images
    if meta:
        for key in ("og:image:secure_url", "og:image", "twitter:image", "twitter:image:src"):
            if key in meta:
                _add(meta[key], "og")

    # 3. Main-content <img> elements
    if tree is not None and len(candidates) < max_images:
        xpath = f"//img[not({_EXCLUDED_IMG_ANCESTORS})]"
        for el in tree.xpath(xpath):
            if len(candidates) >= max_images:
                break
            raw_src = _img_src_from_element(el)
            if not raw_src:
                continue
            width = _parse_int(el.get("width"))
            height = _parse_int(el.get("height"))
            alt = el.get("alt") or el.get("title") or ""
            _add(raw_src, "img", alt=alt, width=width, height=height)

    return candidates
