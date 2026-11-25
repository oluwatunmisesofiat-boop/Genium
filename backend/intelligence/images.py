"""SSRF-safe candidate image download, dimension filter, and SHA-256 + dHash deduplication."""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image, UnidentifiedImageError

from discovery.models import DiscoveryBundle
from discovery.trust import is_disqualified_image_source
from scraper.extract import egress

_SOURCE_PRIORITY = {"jsonld": 0, "image_search": 1, "og": 2, "img": 3}
_USER_AGENT = "Mozilla/5.0 (compatible; GenieAI/0.1)"


@dataclass
class DownloadedImage:
    url: str
    source_page_url: str
    source_trust_tier: int
    extraction_source: str
    alt: str
    mime_type: str
    width: int
    height: int
    sha256: str
    dhash: str
    raw_bytes: bytes
    captured_from_domain: Optional[str] = None
    captured_from_type: Optional[str] = None
    captured_at: Optional[str] = None


def compute_dhash(img: Image.Image) -> str:
    """Compute a 64-bit perceptual difference hash (16-char hex) using Pillow only."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(
        gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata()
    )
    bits = 0
    bit_idx = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            if pixels[offset + col] > pixels[offset + col + 1]:
                bits |= 1 << bit_idx
            bit_idx += 1
    return f"{bits:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Bitwise Hamming distance between two hex dHash fingerprints."""
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def collect_candidate_image_urls(
    bundle: DiscoveryBundle,
    max_candidates: int = 16,
    *,
    trusted_store=None,
    trusted_only: bool = True,
    funnel_out: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Collect and rank candidate image URLs from scraped sources and direct image search.

    When `trusted_store` is provided and `trusted_only=True`, only images whose source
    page (or image URL) belongs to an active `verified` pharmacy or manufacturer in
    `trusted_sources` are admitted.
    """
    ranked: List[tuple[tuple[int, int, int], Dict]] = []
    seq = 0
    raw_from_pages = 0
    raw_from_search = 0
    dropped_disqualified = 0
    dropped_untrusted = 0

    for source in bundle.sources:
        for img in source.images:
            url = str(img.get("url") or "").strip()
            if not url:
                continue
            raw_from_pages += 1
            alt = str(img.get("alt") or "")
            if is_disqualified_image_source(
                url,
                source.url,
                f"{source.title} {alt}".strip(),
                angles=source.angles,
            ):
                dropped_disqualified += 1
                continue
            trusted_match = (
                trusted_store.is_verified_image_source(source.url or url)
                if trusted_store is not None
                else None
            )
            if trusted_store is not None and trusted_only and trusted_match is None:
                dropped_untrusted += 1
                continue
            ext_src = str(img.get("source") or "img")
            sort_key = (source.trust_tier, _SOURCE_PRIORITY.get(ext_src, 4), seq)
            seq += 1
            ranked.append(
                (
                    sort_key,
                    {
                        "url": url,
                        "source_page_url": source.url,
                        "source_trust_tier": source.trust_tier,
                        "extraction_source": ext_src,
                        "alt": alt,
                        "declared_width": img.get("width"),
                        "declared_height": img.get("height"),
                        "captured_from_domain": trusted_match["domain"] if trusted_match else None,
                        "captured_from_type": trusted_match["source_type"] if trusted_match else None,
                    },
                )
            )

    for img in getattr(bundle, "image_candidates", None) or []:
        url = str(img.get("url") or "").strip()
        if not url:
            continue
        raw_from_search += 1
        page_url = str(img.get("source_page_url") or "")
        alt = str(img.get("alt") or img.get("title") or "")
        if is_disqualified_image_source(url, page_url, alt):
            dropped_disqualified += 1
            continue
        trusted_match = (
            trusted_store.is_verified_image_source(page_url or url)
            if trusted_store is not None
            else None
        )
        if trusted_store is not None and trusted_only and trusted_match is None:
            dropped_untrusted += 1
            continue
        ext_src = str(img.get("source") or "image_search")
        tier = int(img.get("source_trust_tier") or 3)
        sort_key = (tier, _SOURCE_PRIORITY.get(ext_src, 1), seq)
        seq += 1
        ranked.append(
            (
                sort_key,
                {
                    "url": url,
                    "source_page_url": page_url,
                    "source_trust_tier": tier,
                    "extraction_source": ext_src,
                    "alt": alt,
                    "declared_width": img.get("width"),
                    "declared_height": img.get("height"),
                    "captured_from_domain": (
                        trusted_match["domain"]
                        if trusted_match
                        else img.get("captured_from_domain")
                    ),
                    "captured_from_type": (
                        trusted_match["source_type"]
                        if trusted_match
                        else img.get("captured_from_type")
                    ),
                },
            )
        )

    ranked.sort(key=lambda pair: pair[0])
    out: List[Dict] = []
    seen_urls: set[str] = set()
    for _key, item in ranked:
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        out.append(item)
        if len(out) >= max_candidates:
            break

    if funnel_out is not None:
        raw_total = raw_from_pages + raw_from_search
        funnel_out.update(
            {
                "raw_total": raw_total,
                "raw_from_pages": raw_from_pages,
                "raw_from_search": raw_from_search,
                "dropped_disqualified": dropped_disqualified,
                "dropped_untrusted": dropped_untrusted,
                "after_trusted": len(out),
            }
        )
    return out


class _DownloadRejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _safe_http_download(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
) -> Tuple[bytes, str]:
    current_url = url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": _USER_AGENT}) as client:
        for _hop in range(4):
            parsed = urlparse(current_url)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
                raise _DownloadRejected("invalid_url")
            if not egress._resolve(parsed.hostname):
                raise _DownloadRejected("address_blocked")

            with client.stream("GET", current_url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    current_url = urljoin(current_url, resp.headers["location"])
                    continue
                if not (200 <= resp.status_code < 300):
                    raise _DownloadRejected(f"http_{resp.status_code}")

                content_type = (resp.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                if content_type and not content_type.startswith("image/"):
                    raise _DownloadRejected("not_image")

                content_len = resp.headers.get("content-length")
                if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                    raise _DownloadRejected("too_large")

                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise _DownloadRejected("too_large")
                    chunks.append(chunk)
                return b"".join(chunks), content_type

    raise _DownloadRejected("too_many_redirects")


def prepare_image_for_llm(
    raw_bytes: bytes,
    mime_type: str,
    *,
    max_side: int = 1024,
) -> Tuple[bytes, str]:
    """Downscale oversized images (e.g. 3000px photos) before sending to Gemini while keeping full-res on disk."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size
            if max(w, h) <= max_side and len(raw_bytes) <= 400_000:
                return raw_bytes, mime_type
            rgb = img.convert("RGB")
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=82, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except (UnidentifiedImageError, OSError, ValueError):
        return raw_bytes, mime_type


def download_and_deduplicate_images(
    candidates: List[Dict],
    *,
    max_unique: int = 6,
    min_dim: int = 350,
    max_aspect: float = 3.5,
    dhash_threshold: int = 6,
    max_bytes: int = 5_000_000,
    timeout: float = 6.0,
    max_workers: int = 6,
    downloader: Optional[Callable[[str], Tuple[bytes, str]]] = None,
) -> Tuple[List[DownloadedImage], List[Dict]]:
    """Download candidate images in parallel, drop tiny/banner/corrupt files, and deduplicate by SHA-256 + dHash."""
    if not candidates:
        return [], []

    def _fetch_one(cand: Dict) -> Tuple[Dict, Optional[Tuple[bytes, str]], Optional[str]]:
        url = str(cand.get("url") or "")
        try:
            if downloader is not None:
                return cand, downloader(url), None
            return cand, _safe_http_download(url, timeout=timeout, max_bytes=max_bytes), None
        except _DownloadRejected as exc:
            return cand, None, exc.reason
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return cand, None, f"download_error: {type(exc).__name__}"

    if downloader is not None or len(candidates) == 1:
        fetched = [_fetch_one(c) for c in candidates]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as pool:
            fetched = list(pool.map(_fetch_one, candidates))

    kept: List[DownloadedImage] = []
    rejected: List[Dict] = []
    seen_sha: set[str] = set()
    seen_dhash: List[str] = []

    for cand, payload, err_reason in fetched:
        if len(kept) >= max_unique:
            break
        url = str(cand.get("url") or "")
        if err_reason is not None or payload is None:
            rejected.append({"url": url, "reason": err_reason or "download_error"})
            continue
        raw_bytes, header_ct = payload

        if not raw_bytes:
            rejected.append({"url": url, "reason": "empty_image"})
            continue

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                width, height = img.size
                pil_fmt = img.format
                dhash = compute_dhash(img)
        except (UnidentifiedImageError, OSError, ValueError):
            rejected.append({"url": url, "reason": "invalid_image"})
            continue

        if width < min_dim or height < min_dim:
            rejected.append({"url": url, "reason": "too_small", "width": width, "height": height})
            continue

        aspect = max(width / height, height / width)
        if aspect > max_aspect:
            rejected.append({"url": url, "reason": "banner_aspect", "width": width, "height": height})
            continue

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if sha256 in seen_sha:
            rejected.append({"url": url, "reason": "duplicate_sha256"})
            continue

        if any(hamming_distance(dhash, prev) <= dhash_threshold for prev in seen_dhash):
            rejected.append({"url": url, "reason": "duplicate_dhash"})
            continue

        mime = (
            Image.MIME.get(pil_fmt or "")
            or (header_ct.split(";", 1)[0].strip().lower() if header_ct else "")
            or "image/jpeg"
        )
        seen_sha.add(sha256)
        seen_dhash.append(dhash)
        kept.append(
            DownloadedImage(
                url=url,
                source_page_url=str(cand.get("source_page_url") or ""),
                source_trust_tier=int(cand.get("source_trust_tier") or 3),
                extraction_source=str(cand.get("extraction_source") or "img"),
                alt=str(cand.get("alt") or ""),
                mime_type=mime,
                width=width,
                height=height,
                sha256=sha256,
                dhash=dhash,
                raw_bytes=raw_bytes,
                captured_from_domain=cand.get("captured_from_domain"),
                captured_from_type=cand.get("captured_from_type"),
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    return kept, rejected
